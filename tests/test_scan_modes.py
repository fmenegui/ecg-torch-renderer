"""Scanner observation modes: color, grayscale, bilevel and dpi."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.scan_modes import scan_observation, scan_resolution


@unittest.skipUnless(TORCH, 'Torch required')
class ScanObservationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.image = torch.rand(3, 3, 48, 64, generator=generator)
        self.ppm = torch.full((3,), 4.)

    def test_modes(self):
        dpi = torch.full((3,), 1200.)
        out = scan_observation(self.image, self.ppm, torch.tensor([0, 1, 2]), dpi, torch.full((3,), .5))
        torch.testing.assert_close(out[0], self.image[0])
        self.assertTrue(bool((out[1, 0] == out[1, 1]).all() and (out[1, 1] == out[1, 2]).all()))
        luma = .2126 * self.image[1, 0] + .7152 * self.image[1, 1] + .0722 * self.image[1, 2]
        torch.testing.assert_close(out[1, 0], luma)
        self.assertTrue(bool(((out[2] == 0) | (out[2] == 1)).all()))

    def test_resolution_holds_blocks_and_preserves_mean(self):
        # 25.4 dpi at 4 px/mm: one scanner pixel spans 4 output pixels.
        scale = torch.full((3,), 25.4 / 25.4 / 4.)
        held = scan_resolution(self.image, scale, 15)
        self.assertTrue(bool((held[:, :, :, 0] == held[:, :, :, 1]).all()))
        self.assertTrue(bool((held[:, :, :, 2] == held[:, :, :, 3]).all()))
        self.assertLess(float(held.diff(dim=-1).abs().mean()), float(self.image.diff(dim=-1).abs().mean()))
        torch.testing.assert_close(held.mean(), self.image.mean(), atol=.02, rtol=0)
        torch.testing.assert_close(scan_resolution(self.image, torch.ones(3), 15), self.image)


@unittest.skipUnless(TORCH, 'Torch required')
class ScanRendererTests(unittest.TestCase):
    def test_flatbed_records_take_scanner_modes_and_targets_stay(self):
        base = dict(height=96, width=128, profile='scan', antialias_samples=1)
        signals = torch.sin(torch.arange(800) / 9)[None, None].expand(2, 12, -1) * .3
        parameters = {'scan_mode': torch.tensor([1, 2]), 'scan_dpi': 300., 'scan_threshold': .6}
        scanned = ECGRenderer(RenderConfig(**base, scan_modes=True))
        batch = scanned(signals, 100, photo=False, parameters=parameters, generator=torch.Generator().manual_seed(1))
        image = batch.image
        self.assertTrue(bool((image[0, 0] == image[0, 1]).all()))
        self.assertTrue(bool(((image[1] == 0) | (image[1] == 1)).all()))
        plain = ECGRenderer(RenderConfig(**base))
        reference = plain(signals, 100, photo=False, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(batch.targets['traces'], reference.targets['traces'])
        self.assertNotIn('scan_modes', plain.specification()['config'])
        sampled = scanned.sample_parameters(64, photo=False, generator=torch.Generator().manual_seed(3))
        self.assertEqual(set(sampled['scan_mode'].tolist()), {0, 1, 2})
        photographed = scanned.sample_parameters(16, photo=True, generator=torch.Generator().manual_seed(3))
        self.assertTrue(bool((photographed['scan_mode'][photographed['photo'] > 0] == 0).all()))


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(TORCH, 'Torch required')
class LidShadowTests(unittest.TestCase):
    def test_shadow_darkens_the_chosen_edge_only(self):
        from ecg_gpu.scan_modes import lid_shadow
        image = torch.ones(4, 3, 40, 60)
        out = lid_shadow(image, torch.tensor([.4, .4, .4, 0.]), torch.tensor([0, 1, 2, 0]), torch.full((4,), .05))
        self.assertLess(float(out[0, 0, 20, 0]), .65)
        self.assertGreater(float(out[0, 0, 20, -1]), .99)
        self.assertLess(float(out[1, 0, 0, 30]), .65)
        self.assertGreater(float(out[1, 0, -1, 30]), .99)
        self.assertLess(float(out[2, 0, 20, -1]), .65)
        torch.testing.assert_close(out[3], image[3])
        self.assertTrue(bool((out <= image).all()))

    def test_lid_controls_only_on_flatbed_records(self):
        renderer = ECGRenderer(RenderConfig(height=64, width=96, profile='mixed', scan_modes=True))
        p = renderer.sample_parameters(128, generator=torch.Generator().manual_seed(7))
        self.assertTrue(bool((p['scan_lid_strength'][p['photo'] > 0] == 0).all()))
        flatbed = p['scan_lid_strength'][p['photo'] == 0]
        self.assertGreater(int((flatbed > 0).sum()), 0)
