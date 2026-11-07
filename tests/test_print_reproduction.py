"""Laser halftone, photocopy and fax reproduction of the canonical page."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.print_reproduction import FAX, LASER, ORIGINAL, PHOTOCOPY, fax, halftone, photocopy, reproduce


@unittest.skipUnless(TORCH, 'Torch required')
class ReproductionTests(unittest.TestCase):
    def test_halftone_is_binary_and_preserves_mean_coverage(self):
        alpha = torch.full((2, 1, 64, 64), .3)
        dots = halftone(alpha, torch.tensor([1., 2.]))
        self.assertTrue(bool(((dots == 0) | (dots == 1)).all()))
        torch.testing.assert_close(dots.mean((1, 2, 3)), torch.full((2,), 5 / 16))

    def test_photocopy_is_gray_darkens_ink_and_lifts_nothing_above_white(self):
        page = torch.ones(1, 3, 48, 64)
        page[..., 20:23, :] = torch.tensor([.1, .1, .4])[:, None, None]
        copy = photocopy(page, torch.tensor([2]), torch.Generator().manual_seed(0))
        self.assertTrue(bool((copy[:, 0] == copy[:, 1]).all()))
        self.assertLess(float(copy[0, 0, 21].mean()), .2)
        self.assertGreater(float(copy[0, 0, 5].mean()), .7)
        self.assertLessEqual(float(copy.max()), 1.)

    def test_fax_is_bilevel_and_coarser_vertically(self):
        generator = torch.Generator().manual_seed(1)
        page = torch.rand(1, 3, 64, 64, generator=generator)
        out = fax(page, torch.tensor([16.]))
        self.assertTrue(bool(((out == 0) | (out == 1)).all()))
        rows_changes = float((out[0, 0, 1:] != out[0, 0, :-1]).float().mean())
        cols_changes = float((out[0, 0, :, 1:] != out[0, 0, :, :-1]).float().mean())
        self.assertLess(rows_changes, cols_changes)

    def test_reproduce_routes_per_record(self):
        page = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(2))
        tech = torch.tensor([ORIGINAL, LASER, PHOTOCOPY, FAX])
        out = reproduce(page, tech, torch.full((4,), 8.), torch.ones(4, dtype=torch.long),
                        torch.Generator().manual_seed(3))
        torch.testing.assert_close(out[:2], page[:2])
        self.assertTrue(bool(((out[3] == 0) | (out[3] == 1)).all()))

    def test_renderer_integration(self):
        base = dict(height=96, width=128, profile='clean', antialias_samples=1)
        signals = torch.sin(torch.arange(800) / 9)[None, None].expand(4, 12, -1) * .3
        renderer = ECGRenderer(RenderConfig(**base, print_technology=True))
        batch = renderer(signals, 100, photo=False, parameters={'print_tech': torch.tensor([0, 1, 2, 3])},
                         generator=torch.Generator().manual_seed(0))
        self.assertEqual(tuple(batch.image.shape), (4, 3, 96, 128))
        plain = ECGRenderer(RenderConfig(**base))(signals, 100, photo=False, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(batch.targets['traces'], plain.targets['traces'])
        sampled = renderer.sample_parameters(200, generator=torch.Generator().manual_seed(5))
        self.assertEqual(set(sampled['print_tech'].tolist()), {0, 1, 2, 3})
        self.assertNotIn('print_technology', ECGRenderer(RenderConfig(**base)).specification()['config'])


if __name__ == '__main__':
    unittest.main()
