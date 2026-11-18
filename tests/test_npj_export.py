"""Open-ECG-Digitizer training pairs from renderer targets."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.npj_export import npj_mask, write_npj_pairs


@unittest.skipUnless(TORCH, 'Torch required')
class NpjMaskTests(unittest.TestCase):
    def targets(self):
        traces = torch.zeros(1, 12, 4, 4)
        traces[0, 3, 1, 1] = 1.
        grid = torch.zeros(1, 1, 4, 4)
        grid[0, 0, 0, :] = .7
        text = torch.zeros(1, 1, 4, 4)
        text[0, 0, 3, 3] = 1.
        pulses = torch.zeros(1, 1, 4, 4)
        pulses[0, 0, 2, 0] = 1.
        return {'traces': traces, 'grid': grid, 'text': text, 'calibration_pulses': pulses}

    def test_channels_and_pulse_class(self):
        targets = self.targets()
        mask = npj_mask(targets)
        self.assertEqual(float(mask[0, 2, 1, 1]), 1.)
        self.assertAlmostEqual(float(mask[0, 0, 0, 2]), .7, places=6)
        self.assertEqual(float(mask[0, 1, 3, 3]), 1.)
        self.assertEqual(float(mask[0, 2, 2, 0]), 1.)
        self.assertEqual(float(npj_mask(targets, pulse_class='text')[0, 1, 2, 0]), 1.)
        self.assertEqual(float(npj_mask(targets, pulse_class='background')[0, :, 2, 0].sum()), 0.)
        targets['valid'] = torch.zeros(1, 1, 4, 4)
        self.assertEqual(float(npj_mask(targets).sum()), 0.)
        # npj development masks label off-sheet pixels as text/background.
        off = npj_mask(targets, off_sheet='text')
        self.assertEqual(float(off[:, 0].sum() + off[:, 2].sum()), 0.)
        self.assertTrue(torch.equal(off[:, 1], 1 - targets['valid'][:, 0].clamp(0, 1)))

    def test_missing_targets_name_the_config(self):
        with self.assertRaisesRegex(ValueError, 'text_target=True'):
            npj_mask({'traces': torch.zeros(1, 12, 2, 2), 'grid': torch.zeros(1, 1, 2, 2)})

    def test_writes_named_pairs_and_refuses_overwrite(self):
        images = torch.rand(2, 3, 8, 10)
        masks = torch.rand(2, 3, 8, 10)
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            stems = write_npj_pairs(images, masks, directory, start_index=5)
            self.assertEqual(stems, ['000005', '000006'])
            with Image.open(Path(directory) / '000006_mask.png') as written:
                self.assertEqual(written.size, (10, 8))
                self.assertEqual(written.mode, 'RGB')
                np.testing.assert_allclose(np.asarray(written) / 255, masks[1].permute(1, 2, 0).numpy(), atol=.5 / 255 + 1e-6)
            with self.assertRaises(FileExistsError):
                write_npj_pairs(images[:1], masks[:1], directory, start_index=5)

    def test_rendered_batch_exports(self):
        renderer = ECGRenderer(RenderConfig(height=96, width=128, profile='photo', antialias_samples=1,
                                            text_target=True, calibration_target=True))
        signals = torch.sin(torch.arange(800) / 9)[None, None].expand(1, 12, -1) * .3
        # Pages may be sampled without a grid (grid_present=0 or grid style 1);
        # force a ruled grid so every channel is exercised.
        batch = renderer(signals, 100, stage='all', parameters={'grid_style': 0, 'grid_present': 1.},
                         generator=torch.Generator().manual_seed(0))
        mask = npj_mask(batch.targets)
        self.assertEqual(tuple(mask.shape), (1, 3, 96, 128))
        self.assertGreater(float(mask[:, 2].sum()), 0)
        self.assertGreater(float(mask[:, 0].sum()), 0)
        self.assertNotIn('text_target', ECGRenderer(RenderConfig(height=64, width=96)).specification()['config'])


if __name__ == '__main__':
    unittest.main()
