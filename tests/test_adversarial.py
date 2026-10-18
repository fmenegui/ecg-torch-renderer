"""Independent adversarial checks for physical identity and data provenance.

These use small CPU images and deterministic artificial signals. Passing does
not establish perceptual similarity to clinical acquisitions or GPU parity.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ecg_gpu import ECGRenderer, ECGImageLayer, RenderConfig, Choice
from ecg_gpu.gpu_cli import load_signals, normalized_input_fingerprint
from ecg_gpu.gpu_supervision import mask_bounding_boxes
from ecg_gpu.scenarios import capture_scenario


class AdversarialContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = dict(height=65, width=97, profile='clean', surface_mode='off',
                          surface_resolution=32, shadow_samples=0, antialias_samples=1)
        cls.renderer = ECGRenderer(RenderConfig(**cls.config))
        cls.fixed = {'layout': '3x4+r3', 'speed_index': 1, 'gain_index': 1,
                     'lead_present': 1., 'grid_present': 1., 'header_present': 1.,
                     'artifact_present': 0., 'pen_active': 0., 'occlusion_active': 0.}

    def render(self, signals, renderer=None, **options):
        return (renderer or self.renderer)(
            signals, options.pop('fs', 100.), photo=False, stage='all',
            parameters=options.pop('parameters', self.fixed),
            generator=torch.Generator().manual_seed(12347), **options)

    def test_all_layouts_support_two_samples_and_odd_canvas(self):
        source = torch.zeros(1, 12, 2)
        for layout in self.renderer.layout_names:
            with self.subTest(layout=layout):
                batch = self.render(source, parameters={**self.fixed, 'layout': layout})
                self.assertEqual(tuple(batch.image.shape), (1, 3, 65, 97))
                self.assertTrue(torch.isfinite(batch.image).all())
                self.assertTrue(((batch.image >= 0) & (batch.image <= 1)).all())
                for name in ['traces', 'baselines', 'valid', 'grid', 'vertices']:
                    self.assertTrue(torch.isfinite(batch.targets[name]).all(), name)
                self.assertEqual(float(batch.metadata['duration_s'][0]), float(torch.tensor(.01)))

    def test_padded_values_cannot_leak_into_visible_signal(self):
        source = torch.zeros(2, 12, 240)
        source[:, :, :60] = torch.arange(60)[None, None] * .002
        altered = source.clone()
        altered[0, :, 60:] = 1e4
        altered[1, :, 80:] = -1e4
        lengths = torch.tensor([60, 80])
        first = self.render(source, lengths=lengths)
        second = self.render(altered, lengths=lengths)
        torch.testing.assert_close(first.image, second.image, rtol=0, atol=0)
        torch.testing.assert_close(first.targets['traces'], second.targets['traces'], rtol=0, atol=0)

    def test_explicit_generator_does_not_consume_global_random_state(self):
        before = torch.random.get_rng_state().clone()
        self.render(torch.zeros(1, 12, 200))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_render_preserves_noncontiguous_source_and_disables_source_gradients(self):
        storage = torch.linspace(-.1, .1, 400)[None, None].expand(1, 12, -1).clone()
        source = storage[:, :, ::2].requires_grad_()
        before = storage.detach().clone()
        self.assertFalse(source.is_contiguous())
        batch = self.render(source)
        self.assertTrue(torch.equal(storage, before))
        self.assertIs(batch.reference, source)
        self.assertFalse(batch.image.requires_grad)
        self.assertTrue(torch.isfinite(batch.image).all())

    def test_chunk_size_does_not_change_pixels_or_lead_identity(self):
        source = torch.sin(torch.arange(200)[None, None] / 9).expand(1, 12, -1) * .2
        expected = self.render(source)
        for chunk in [1, 7, 15]:
            renderer = ECGRenderer(RenderConfig(**self.config, lead_chunk_size=chunk))
            result = self.render(source, renderer=renderer)
            torch.testing.assert_close(result.image, expected.image, rtol=0, atol=0)
            torch.testing.assert_close(result.targets['traces'], expected.targets['traces'], rtol=0, atol=0)
            self.assertTrue(torch.equal(result.metadata['lead_id'], expected.metadata['lead_id']))

    def test_fractional_categorical_overrides_are_rejected_before_integer_cast(self):
        for key, value in [('layout', -.5), ('layout', 1.00000001), ('gain_index', 1.9), ('quarter', .9),
                           ('rhythm_leads', [1., 6.1, 10.])]:
            with self.subTest(parameter=key), self.assertRaises((ValueError, RuntimeError)):
                self.render(torch.zeros(1, 12, 20), parameters={**self.fixed, key: value})

    def test_finite_float64_values_that_overflow_working_precision_are_rejected(self):
        source = torch.full((1, 12, 20), 1e40, dtype=torch.float64)
        self.assertTrue(torch.isfinite(source).all())
        with self.assertRaises((ValueError, RuntimeError)):
            self.render(source)

    def test_direct_lengths_reject_fractions_before_default_float32_rounding(self):
        for length in [20.9, 20.00000001]:
            with self.subTest(length=length), self.assertRaises((ValueError, RuntimeError)):
                self.render(torch.zeros(1, 12, 40), lengths=[length])

    def test_layer_fixed_and_choice_values_reject_lossy_identity_casts(self):
        for parameters in [{'layout': 1.9}, {'layout': 1.00000001},
                           {'layout': Choice([0., 1.00000001])}, {'layout': Choice([0., 1.9])},
                           {'rhythm_leads': [1., 6.1, 10.]},
                           {'quarter': Choice([0., .9])}]:
            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, 'integer'):
                ECGImageLayer(renderer=self.renderer, parameters=parameters)

    def test_layer_boolean_controls_reject_values_outside_zero_one(self):
        for value in [-1, .5, 2, 1e-50, Choice([0, 2])]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                ECGImageLayer(renderer=self.renderer, parameters={'sequential': value})
        for value in [True, False, 0, 1., Choice([False, True])]:
            ECGImageLayer(renderer=self.renderer, parameters={'sequential': value})

    def test_direct_boolean_controls_and_opacity_require_valid_domains(self):
        for key,value in [('sequential', -1), ('sequential', 1e-50), ('label_below', .5),
                           ('lead_present', -1.), ('ink_density', 1.01),
                           ('grid_strength', -1.)]:
            with self.subTest(parameter=key), self.assertRaises((ValueError, RuntimeError)):
                self.render(torch.zeros(1, 12, 20), parameters={**self.fixed, key: value})

    def test_scan_recipes_have_uniform_scene_illumination(self):
        source = torch.zeros(1, 12, 100)
        for name in ['clean_print', 'flatbed_scan', 'aged_scan']:
            with self.subTest(scenario=name):
                config, parameters = capture_scenario(name, height=65, width=97)
                layer = ECGImageLayer(config=config, parameters=parameters, randomize=False)
                batch = layer.render_batch(source, stage='trace', return_scene=True)
                torch.testing.assert_close(batch.scene['illumination'],
                                           torch.ones_like(batch.scene['illumination']),
                                           rtol=0, atol=torch.finfo(torch.float32).eps)

    def test_mask_boxes_handle_one_pixel_on_each_canvas_boundary(self):
        mask = torch.zeros(1, 5, 65, 97)
        expected = []
        for channel, (x, y) in enumerate([(0, 0), (96, 0), (0, 64), (96, 64)]):
            mask[0, channel, y, x] = 1
            expected.append([x, y, x + 1, y + 1])
        expected.append([-1, -1, -1, -1])
        boxes, present = mask_bounding_boxes(mask)
        self.assertEqual(boxes.tolist(), [expected])
        self.assertEqual(present.tolist(), [[True, True, True, True, False]])

    def test_cli_rejects_fractional_nonfinite_or_nonnumeric_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.npz'
            for lengths in [[20.9], [float('nan')], [float('inf')], ['20'], [True], [20 + 1j]]:
                with self.subTest(lengths=lengths):
                    np.savez(path, signals=np.zeros((12, 40), dtype=np.float32), fs=100.,
                             lengths=np.asarray(lengths))
                    with self.assertRaisesRegex(ValueError, 'lengths'):
                        load_signals(path)

    def test_cli_accepts_integral_float_lengths_without_changing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.npz'
            expected = np.zeros((12, 40), dtype=np.float32)
            np.savez(path, signals=expected, fs=100., lengths=np.array([20.]))
            signals, _, lengths = load_signals(path)
            self.assertEqual(lengths.dtype, np.int64)
            self.assertEqual(lengths.tolist(), [20])
            np.testing.assert_array_equal(signals[0], expected)

    def test_decoded_input_fingerprint_tracks_waveform_time_and_valid_samples(self):
        source = np.zeros((1, 12, 40), dtype=np.float32)
        original = normalized_input_fingerprint(source, 100., None)
        source[0, 0, 3] = .25
        changed = normalized_input_fingerprint(source, 100., None)
        self.assertNotEqual(original['sha256'], changed['sha256'])
        self.assertNotEqual(original['array_sha256']['signals'], changed['array_sha256']['signals'])
        self.assertNotEqual(changed['sha256'], normalized_input_fingerprint(source, 200., None)['sha256'])
        self.assertNotEqual(changed['sha256'], normalized_input_fingerprint(source, 100., [39])['sha256'])
        self.assertEqual(changed, normalized_input_fingerprint(source.astype('>f4'), [100.], [40]))


if __name__ == '__main__':
    unittest.main()
