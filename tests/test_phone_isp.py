"""Phone processing stage: chromatic aberration, sharpening and glare."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.phone_isp import chromatic_aberration, glare, sharpen


@unittest.skipUnless(TORCH, 'Torch required')
class PhoneStageTests(unittest.TestCase):
    def setUp(self):
        # White page with vertical black lines every 8 pixels.
        self.image = torch.ones(2, 3, 64, 96)
        self.image[..., ::8] = 0

    def test_aberration_fringes_grow_toward_edges(self):
        torch.testing.assert_close(chromatic_aberration(self.image, torch.zeros(2)), self.image)
        out = chromatic_aberration(self.image, torch.tensor([0., 3.]))
        torch.testing.assert_close(out[0], self.image[0])
        fringe = (out[1, 0] - out[1, 2]).abs()
        self.assertGreater(float(fringe[:, -10:].mean()), float(fringe[:, 43:53].mean()) + .05)

    def test_sharpening_overshoots_at_edges_and_zero_is_identity(self):
        soft = torch.nn.functional.avg_pool2d(self.image, 3, 1, 1, count_include_pad=False)
        torch.testing.assert_close(sharpen(soft, torch.zeros(2), torch.ones(2)), soft)
        sharp = sharpen(soft, torch.full((2,), 1.), torch.ones(2))
        contrast = lambda x: float((x[..., 1:] - x[..., :-1]).abs().mean())
        self.assertGreater(contrast(sharp), contrast(soft))

    def test_glare_brightens_its_center(self):
        gray = torch.full((1, 3, 64, 96), .5)
        out = glare(gray, torch.tensor([.4]), torch.tensor([.25]), torch.tensor([.5]), torch.tensor([.1]))
        self.assertGreater(float(out[0, :, 32, 24].mean()), float(out[0, :, 32, 90].mean()) + .15)
        torch.testing.assert_close(glare(gray, torch.zeros(1), torch.tensor([.5]), torch.tensor([.5]),
                                         torch.tensor([.1])), gray)

    def test_renderer_applies_only_to_photographs(self):
        base = dict(height=96, width=128, antialias_samples=1)
        signals = torch.sin(torch.arange(800) / 9)[None, None].expand(1, 12, -1) * .3
        renderer = ECGRenderer(RenderConfig(**base, profile='photo', phone_isp=True))
        p = renderer.sample_parameters(64, generator=torch.Generator().manual_seed(2))
        self.assertTrue(bool((p['ca_px'][p['photo'] == 0] == 0).all()))
        scans = ECGRenderer(RenderConfig(**base, profile='scan', phone_isp=True))
        q = scans.sample_parameters(16, generator=torch.Generator().manual_seed(2))
        self.assertEqual(float(q['ca_px'].abs().sum() + q['sharpen_amount'].abs().sum() + q['glare_strength'].sum()), 0.)
        batch = renderer(signals, 100, parameters={'ca_px': 2., 'glare_strength': .3},
                         generator=torch.Generator().manual_seed(1))
        self.assertEqual(tuple(batch.image.shape), (1, 3, 96, 128))
        self.assertNotIn('phone_isp', ECGRenderer(RenderConfig(**base)).specification()['config'])


if __name__ == '__main__':
    unittest.main()
