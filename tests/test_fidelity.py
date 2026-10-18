"""Independent physical and temporal contracts for ECG document rendering.

These checks use analytic ramps and isolated marks rather than a visual
similarity score. They verify document fidelity, not clinical signal realism.
"""
import unittest

import torch

from ecg_gpu import ECGRenderer, LayoutSpec, RenderConfig
from ecg_gpu.gpu_layouts import CABRERA_ORDER, standard_layouts
from ecg_gpu.gpu_raster import calibration_coverage
from ecg_gpu.schema_v05 import GAINS, LEADS, SPEEDS


class LayoutValidationTests(unittest.TestCase):
    def test_nested_sequences_are_snapshotted_and_immutable(self):
        rows, rhythm = [['I', 'II'], ['V1']], ['II']
        spec = LayoutSpec('custom', rows, rhythm)
        rows[0][0], rhythm[0] = 'III', 'V5'
        self.assertEqual(spec.rows, (('I', 'II'), ('V1',)))
        self.assertEqual(spec.rhythm, ('II',))
        self.assertIsInstance(hash(spec), int)

    def test_malformed_layouts_raise_value_error_at_construction(self):
        for name, rows, rhythm in [
            (1, (('I',),), ()), (' ', (('I',),), ()),
            ('bad', 'I', ()), ('bad', ('I',), ()),
            ('bad', ((None,),), ()), ('bad', (('I',),), 'II'),
            ('bad', (('I', '-I'),), ()), ('bad', (('V7',),), ()),
            ('bad', (('I',),), ('II',) * 4),
        ]:
            with self.subTest(rows=rows, rhythm=rhythm):
                with self.assertRaises(ValueError):
                    LayoutSpec(name, rows, rhythm)


class DocumentFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.renderer = ECGRenderer(RenderConfig(
            height=384, width=640, profile='clean', antialias_samples=2))

    def parameters(self, count=1):
        p = self.renderer.sample_parameters(
            count, photo=False, generator=torch.Generator().manual_seed(177))
        p['sequential'].fill_(True)
        p['lead_present'].fill_(1)
        return p

    def test_every_layout_has_expected_lead_polarity_and_time_windows(self):
        specs = standard_layouts()
        p = self.parameters(len(specs))
        p['layout'] = torch.arange(len(specs))
        duration = torch.linspace(3., 10., len(specs))
        data = self.renderer._instances(p, duration)
        for index, spec in enumerate(specs):
            for row, tokens in enumerate(spec.rows):
                for column, token in enumerate(tokens):
                    lead = LEADS.index(token.removeprefix('-'))
                    with self.subTest(layout=spec.name, lead=token):
                        self.assertTrue(bool(data['active'][index, lead]))
                        self.assertEqual(int(data['row'][index, lead]), row)
                        self.assertEqual(int(data['column'][index, lead]), column)
                        self.assertEqual(int(data['lead_id'][index, lead]), lead)
                        self.assertEqual(int(data['amplitude_sign'][index, lead]),
                                         -1 if token.startswith('-') else 1)
                        span = duration[index] / len(tokens)
                        torch.testing.assert_close(data['start_s'][index, lead], column * span)
                        torch.testing.assert_close(data['end_s'][index, lead], (column + 1) * span)
            for rhythm_index, token in enumerate(spec.rhythm):
                slot = 12 + rhythm_index
                self.assertEqual(int(data['lead_id'][index, slot]), LEADS.index(token))
                self.assertEqual(float(data['start_s'][index, slot]), 0.)
                torch.testing.assert_close(data['end_s'][index, slot], duration[index])

    def test_cabrera_order_and_simultaneous_windows(self):
        p = self.parameters()
        p['layout'].fill_(self.renderer.layout_names.index('12x1-cabrera'))
        data = self.renderer._instances(p, torch.tensor([10.]))
        order = data['row'][0, :12].argsort().tolist()
        tokens = [('-' if data['amplitude_sign'][0, lead] < 0 else '') + LEADS[lead]
                  for lead in order]
        self.assertEqual(tuple(tokens), CABRERA_ORDER)
        p['layout'].fill_(self.renderer.layout_names.index('3x4+r3'))
        p['sequential'].fill_(False)
        p['rhythm_leads'][0] = torch.tensor([0, 5, 11])
        data = self.renderer._instances(p, torch.tensor([10.]))
        torch.testing.assert_close(data['start_s'], torch.zeros_like(data['start_s']))
        torch.testing.assert_close(data['end_s'][0, :12], torch.full((12,), 2.5))
        torch.testing.assert_close(data['end_s'][0, 12:], torch.full((3,), 10.))
        self.assertEqual(data['lead_id'][0, 12:].tolist(), [0, 5, 11])

    def test_time_voltage_and_square_grid_use_one_physical_scale(self):
        # All supported speed/gain combinations, independent of pixel density.
        p = self.parameters(9)
        p['layout'].fill_(self.renderer.layout_names.index('3x4+r1'))
        p['speed_index'] = torch.arange(3).repeat_interleave(3)
        p['gain_index'] = torch.arange(3).repeat(3)
        duration = torch.linspace(4., 10., 9)
        data = self.renderer._instances(p, duration)
        torch.testing.assert_close(data['speed'], torch.tensor(SPEEDS[1:]).repeat_interleave(3))
        torch.testing.assert_close(data['gain'], torch.tensor(GAINS[1:]).repeat(3))
        # Horizontal mm divided by speed must recover each printed duration.
        printed_s = (data['x1'] - data['x0']) / (data['ppm'] * data['speed'])[:, None]
        torch.testing.assert_close(printed_s, data['end_s'] - data['start_s'])
        # 1 mV vertical / 1 s horizontal = gain / speed, in pixel units too.
        voltage_px = data['gain'] * data['ppm']
        second_px = data['speed'] * data['ppm']
        torch.testing.assert_close(voltage_px / second_px, data['gain'] / data['speed'])

    def test_calibration_is_one_millivolt_and_two_hundred_milliseconds(self):
        # Integer pixels make geometric measurements exact at all unit settings.
        ppm, base, left = 4., 120., 80.
        xx = torch.arange(128.)[None, None, None, :]
        yy = torch.arange(144.)[None, None, :, None]
        for speed in SPEEDS[1:]:
            for gain in GAINS[1:]:
                top = base - gain * ppm
                right = left - 2 * ppm
                start = right - .2 * speed * ppm
                mask = calibration_coverage(
                    torch.tensor([[base]]), torch.tensor([[gain * ppm]]),
                    torch.tensor([left]), torch.tensor([ppm]), torch.tensor([speed]),
                    torch.ones(1, 1, 1, 1), torch.ones(1, 1),
                    torch.ones(1, 1, dtype=torch.bool), xx, yy)[0, 0]
                top_columns = (mask[int(top)] > .5).nonzero().flatten()
                self.assertEqual(int(top_columns[0]), int(start))
                self.assertEqual(int(top_columns[-1]), int(right))
                self.assertEqual(int(top_columns[-1] - top_columns[0]), int(.2 * speed * ppm))
                self.assertTrue(bool((mask[int(top):int(base) + 1, int(start)] > .5).all()))
                self.assertEqual(int(base - top), int(gain * ppm))
                absent = calibration_coverage(
                    torch.tensor([[base]]), torch.tensor([[gain * ppm]]),
                    torch.tensor([left]), torch.tensor([ppm]), torch.tensor([speed]),
                    torch.ones(1, 1, 1, 1), torch.zeros(1, 1),
                    torch.ones(1, 1, dtype=torch.bool), xx, yy)
                self.assertEqual(float(absent.sum()), 0.)

    def test_variable_length_interpolation_never_reads_padding(self):
        signals = torch.tensor([[[0., 1., 2., 999., 999.]],
                                [[0., 2., 4., 6., 8.]]])
        times = torch.tensor([[[0., .25, 1., 3.]], [[0., .25, 1., 3.]]])
        sampled = self.renderer._sample_signal(
            signals, times, torch.tensor([2., 2.]), torch.tensor([3, 5]))
        torch.testing.assert_close(sampled, torch.tensor([[[0., .5, 2., 2.]],
                                                         [[0., 1., 4., 8.]]]))

    def test_cabrera_inversion_is_rendered_in_correct_physiological_channel(self):
        p = self.parameters()
        p['layout'].fill_(self.renderer.layout_names.index('3x4-cabrera'))
        p['gain_index'].fill_(1)
        p['speed_index'].fill_(1)
        data = self.renderer._instances(p, torch.tensor([10.]))
        signals = torch.zeros(1, 12, 1001)
        signals[:, LEADS.index('aVR')] = 1.
        original = signals.clone()
        _, _, targets = self.renderer._curves(
            signals, torch.tensor([100.]), torch.tensor([1001]), data, p, 'trace')
        lead = LEADS.index('aVR')
        x = int((data['x0'][0, lead] + data['x1'][0, lead]) / 2)
        expected_y = data['baseline'][0, lead] + data['gain'][0] * data['ppm'][0]
        actual_y = targets['traces'][0, lead, :, x].argmax()
        self.assertLess(abs(float(actual_y - expected_y)), 1.)
        torch.testing.assert_close(signals, original, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
