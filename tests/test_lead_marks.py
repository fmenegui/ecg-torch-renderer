"""Lead-switch marks at column boundaries, RenderConfig(lead_marks=True)."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig


@unittest.skipUnless(TORCH, 'Torch required')
class LeadMarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base = dict(height=480, width=640, profile='clean', antialias_samples=2)
        cls.signals = torch.zeros(1, 12, 1000)
        cls.parameters = {'layout': '3x4', 'lead_present': 1., 'sequential': True, 'line_width': 1.5}

    def render(self, lead_marks, **extra):
        renderer = ECGRenderer(RenderConfig(**self.base, lead_marks=lead_marks))
        return renderer(self.signals, 100, photo=False, stage='trace', parameters={**self.parameters, **extra},
                        generator=torch.Generator().manual_seed(6))

    def test_marks_sit_on_column_boundaries_at_the_baseline(self):
        batch = self.render(True, mark_present=1., mark_height_mm=4.)
        marks = batch.targets['lead_marks'][0, 0]
        meta = batch.metadata
        ppm = float(meta['ppm'][0])
        boundaries = meta['x0'][0][(meta['column'][0] > 0) & meta['active'][0]]
        self.assertEqual(len(boundaries), 9)
        columns = (marks > .5).any(0).nonzero().flatten().float()
        self.assertGreater(len(columns), 0)
        distance = (columns[:, None] - boundaries[None]).abs().min(1).values
        self.assertTrue(bool((distance <= 1.5).all()))
        # Each mark spans about 4 mm vertically around a baseline.
        column = int(boundaries[0].round())
        height = float(marks[:, column].sum())
        self.assertAlmostEqual(height / (3 * ppm), 4., delta=.6)

    def test_marks_do_not_change_waveform_targets(self):
        with_marks = self.render(True, mark_present=1., mark_height_mm=4.)
        without = self.render(False)
        torch.testing.assert_close(with_marks.targets['traces'], without.targets['traces'], rtol=0, atol=0)
        self.assertNotIn('lead_marks', without.targets)

    def test_absent_marks_and_legacy_contracts(self):
        batch = self.render(True, mark_present=0.)
        self.assertEqual(float(batch.targets['lead_marks'].sum()), 0.)
        legacy = ECGRenderer(RenderConfig(**self.base))
        marked = ECGRenderer(RenderConfig(**self.base, lead_marks=True))
        self.assertNotIn('lead_marks', legacy.specification()['config'])
        legacy_p = legacy.sample_parameters(2, generator=torch.Generator().manual_seed(1))
        marked_p = marked.sample_parameters(2, generator=torch.Generator().manual_seed(1))
        for name, value in legacy_p.items():
            torch.testing.assert_close(marked_p[name], value, rtol=0, atol=0, msg=name)
        with self.assertRaisesRegex(ValueError, 'lead_marks'):
            RenderConfig(lead_marks=1)


if __name__ == '__main__':
    unittest.main()
