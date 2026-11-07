"""Mold colonies and ironing scorch on the canonical page."""
import importlib.util
import math
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.archive_damage import mold, scorch


@unittest.skipUnless(TORCH, 'Torch required')
class ArchiveDamageTests(unittest.TestCase):
    def setUp(self):
        self.page = torch.ones(2, 3, 96, 128)

    def test_mold_is_local_greenish_and_respects_active(self):
        out, alpha = mold(self.page, torch.tensor([1., 0.]), torch.full((2,), .25), torch.full((2,), .5),
                          torch.full((2,), .08), torch.full((2,), .6), torch.Generator().manual_seed(0))
        torch.testing.assert_close(out[1], self.page[1])
        self.assertGreater(float(alpha[0, 0, 48, 20:44].max()), .3)
        self.assertLess(float(alpha[0, 0, :, 100:].max()), .05)
        darkened = out[0, :, alpha[0, 0] > .3]
        self.assertGreater(float(darkened[1].mean()), float(darkened[0].mean()))

    def test_scorch_is_brown_darker_at_core_and_never_brightens(self):
        out, alpha = scorch(self.page, torch.tensor([1., 0.]), torch.full((2,), .5), torch.full((2,), .5),
                            torch.full((2,), .2), torch.full((2,), .12), torch.zeros(2), torch.full((2,), .8))
        torch.testing.assert_close(out[1], self.page[1])
        self.assertTrue(bool((out <= self.page + 1e-6).all()))
        core = out[0, :, 48, 64]
        edge = out[0, :, 48, 100]
        self.assertLess(float(core.mean()), float(edge.mean()))
        self.assertGreater(float(core[0]), float(core[2]))

    def test_renderer_integration_keeps_targets(self):
        base = dict(height=96, width=128, antialias_samples=1, profile='scan')
        signals = torch.sin(torch.arange(800) / 9)[None, None].expand(1, 12, -1) * .3
        overrides = {'mold_active': 1., 'scorch_active': 1., 'scorch_angle': math.pi / 4}
        damaged = ECGRenderer(RenderConfig(**base, archive_damage=True))
        batch = damaged(signals, 100, photo=False, parameters=overrides, generator=torch.Generator().manual_seed(1))
        plain = ECGRenderer(RenderConfig(**base))(signals, 100, photo=False, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(batch.targets['traces'], plain.targets['traces'])
        self.assertNotIn('archive_damage', ECGRenderer(RenderConfig(**base)).specification()['config'])


if __name__ == '__main__':
    unittest.main()
