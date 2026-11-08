"""Pinhole parallax for lifted paper, against the exact viewing-ray intersection."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.gpu_document import relief_projection


@unittest.skipUnless(TORCH, 'Torch required')
class ReliefProjectionTests(unittest.TestCase):
    def test_first_order_matches_exact_pinhole_intersection(self):
        ys, xs = torch.meshgrid(torch.linspace(-.7, .7, 9, dtype=torch.float64),
                                torch.linspace(-1, 1, 11, dtype=torch.float64), indexing='ij')
        xy = torch.stack([xs, ys])[None]
        height = (.02 * (1 + xs * ys))[None, None]
        center = torch.tensor([[.2, -.1]], dtype=torch.float64)
        distance = 3.
        got = relief_projection(xy, height, torch.zeros(1, 2, dtype=torch.float64),
                                torch.tensor([1 / distance], dtype=torch.float64), center)
        # Ray from (center, D) through the lifted point meets the page plane at:
        exact = center[:, :, None, None] + (xy - center[:, :, None, None]) * distance / (distance - height)
        torch.testing.assert_close(got, exact, atol=2e-4, rtol=0)
        # Without parallax only the oblique relief term remains.
        oblique = relief_projection(xy, height, torch.tensor([[.3, -.2]], dtype=torch.float64))
        torch.testing.assert_close(oblique - xy, height * torch.tensor([.3, -.2], dtype=torch.float64)[None, :, None, None])


@unittest.skipUnless(TORCH, 'Torch required')
class ParallaxGeometryTests(unittest.TestCase):
    def setUp(self):
        self.renderer = ECGRenderer(RenderConfig(height=96, width=128, profile='photo', antialias_samples=1,
                                                 perspective_relief=True, surface_mode='fold'))

    def flat_parameters(self, zoom):
        p = self.renderer.sample_parameters(1, generator=torch.Generator().manual_seed(0))
        for key in ('angle', 'translate_x', 'translate_y', 'radial', 'perspective_x', 'perspective_y',
                    'curl_x', 'curl_y', 'crease_warp'):
            p[key] = torch.zeros_like(p[key])
        p['zoom'] = torch.full_like(p['zoom'], zoom)
        p['quarter'] = torch.zeros_like(p['quarter'])
        return p

    def test_center_and_scale_follow_zoom(self):
        for zoom in (1., .8):
            with self.subTest(zoom=zoom):
                p = self.flat_parameters(zoom)
                center, scale = self.renderer._parallax(self.renderer._field(p), torch.tensor([.5]))
                torch.testing.assert_close(center, torch.zeros(1, 2), atol=1e-5, rtol=0)
                torch.testing.assert_close(scale, torch.tensor([.5 * zoom]), atol=1e-4, rtol=0)

    def test_parallax_moves_lifted_content_and_only_for_photos(self):
        signals = torch.zeros(1, 12, 500)
        fields = {}
        for parallax in (0., .6):
            batch = self.renderer(signals, 100, photo=True, return_field=True,
                                  parameters={'camera_parallax': parallax, 'surface_strength': 1.},
                                  generator=torch.Generator().manual_seed(4))
            fields[parallax] = batch.inverse_field
            self.assertTrue(bool(torch.isfinite(batch.inverse_field).all()))
        self.assertGreater(float((fields[.6] - fields[0.]).abs().max()), 1e-3)
        scans = ECGRenderer(RenderConfig(height=64, width=96, profile='scan', perspective_relief=True))
        q = scans.sample_parameters(8, generator=torch.Generator().manual_seed(1))
        self.assertEqual(float(q['camera_parallax'].abs().sum()), 0.)
        self.assertNotIn('perspective_relief', ECGRenderer(RenderConfig(height=64, width=96)).specification()['config'])


if __name__ == '__main__':
    unittest.main()
