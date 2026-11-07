"""Generated report text: placement, consistency with the record and transcripts."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.demo import demo_signals
    from ecg_gpu.report_text import (MEASURE_TEMPLATES, RATE, decode_report_text, encode, template_bank,
                                     ventricular_rate)


@unittest.skipUnless(TORCH, 'Torch required')
class TemplateTests(unittest.TestCase):
    def test_bank_encodes_and_is_fictional_without_statements(self):
        header, measure, footer = template_bank()
        glyphs, kinds = encode(header + measure + footer)
        self.assertEqual(len(glyphs), len(header) + len(measure) + len(footer))
        self.assertEqual(len(footer), 3 ** 5)
        text = ' '.join(header + measure + footer).lower()
        for word in ('rhythm', 'ritmo', 'infarct', 'abnormal', 'normal ecg'):
            self.assertNotIn(word, text)


@unittest.skipUnless(TORCH, 'Torch required')
class RateTests(unittest.TestCase):
    def test_rate_from_lead_ii(self):
        for bpm in (55, 80, 120):
            with self.subTest(bpm=bpm):
                values, fs, _ = demo_signals(fs=500, duration=10, heart_rate=bpm)
                signals = torch.tensor(values, dtype=torch.float32)[None]
                rate = ventricular_rate(signals, torch.tensor([500.]), torch.tensor([5000]))
                self.assertAlmostEqual(float(rate[0]), bpm, delta=2)


@unittest.skipUnless(TORCH, 'Torch required')
class RendererReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        values, _, _ = demo_signals(fs=500, duration=10, heart_rate=80)
        cls.signals = torch.tensor(values, dtype=torch.float32)[None]
        header, measure, _ = template_bank()
        cls.fc_line = len(header) + MEASURE_TEMPLATES.index(f'FC: {RATE} bpm')

    def render(self, **config):
        renderer = ECGRenderer(RenderConfig(height=720, width=960, profile='clean', antialias_samples=1,
                                            report_text=True, **config))
        # The rate line goes in the first row, which always fits the header band.
        lines = torch.tensor([[self.fc_line, 30, 60, 0, 1, 2]])
        parameters = {'speed_index': 1, 'gain_index': 1, 'report_lines': lines, 'report_present': 1.,
                      'font_size': 9.,
                      'report_footer_present': 1., 'report_text_mm': 2., 'layout': '3x4'}
        return renderer, renderer(self.signals, 500, photo=False, parameters=parameters,
                                  generator=torch.Generator().manual_seed(5))

    def test_text_sits_in_margins_and_transcripts_agree_with_record(self):
        _, batch = self.render()
        ink = batch.targets['report_text'][0, 0]
        meta = batch.metadata
        top = float(meta['top'][0])
        self.assertGreater(float(ink.sum()), 0)
        rows = ink.amax(1) > .5
        plot_rows = torch.arange(ink.shape[0], dtype=torch.float32)
        inside_plot = rows & (plot_rows > top + 1) & (plot_rows < ink.shape[0] * .9)
        self.assertFalse(bool(inside_plot.any()))
        lines = decode_report_text(meta['report_text'])[0]
        self.assertTrue(lines[-1].startswith('25 mm/s  10 mm/mV'), lines[-1])
        rate = int(meta['report_text']['ventricular_rate_bpm'][0])
        self.assertAlmostEqual(rate, 80, delta=2)
        self.assertIn(f'FC: {rate:>3} bpm', [line for line in lines if line.startswith('FC')])

    def test_footer_matches_device_filters(self):
        _, batch = self.render(device_filters=True)
        p = batch.metadata['parameters']
        footer = decode_report_text(batch.metadata['report_text'])[0][-1]
        expected = f"{float(p['filter_highpass_hz'][0]):g}-{float(p['filter_lowpass_hz'][0]):g} Hz"
        self.assertIn(expected, footer)

    def test_contracts(self):
        legacy = ECGRenderer(RenderConfig(height=96, width=128))
        self.assertNotIn('report_text', legacy.specification()['config'])
        self.assertNotIn('report_text', legacy.specification())
        renderer, _ = self.render()
        self.assertFalse(renderer.report_text.specification()['interpretation_statements'])
        from ecg_gpu import DOCUMENT_TEMPLATES
        with self.assertRaisesRegex(ValueError, 'document_template'):
            ECGRenderer(RenderConfig(height=96, width=128, report_text=True),
                        document_template=DOCUMENT_TEMPLATES['clinical'])


if __name__ == '__main__':
    unittest.main()
