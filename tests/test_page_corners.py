"""Observed page corners against an analytic affine inverse and a rendered capture."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.page_geometry import page_corners


def affine_field(h, w, matrix, offset):
    """Field mapping output pixel centers to canonical coordinates through an affine map."""
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing='ij')
    pixels = torch.stack([xs, ys], -1)
    return (pixels @ matrix.T + offset)[None]


@unittest.skipUnless(TORCH, 'Torch required')
class PageCornerTests(unittest.TestCase):
    def test_affine_field_matches_analytic_inverse(self):
        h, w = 120, 160
        matrix = torch.tensor([[.0135, .002], [-.0015, .0175]], dtype=torch.float64)
        offset = torch.tensor([-1.08, -1.02], dtype=torch.float64)
        field = affine_field(h, w, matrix, offset)
        corners, in_frame, residual = page_corners(field)
        canonical = torch.tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]], dtype=torch.float64)
        expected = (canonical - offset) @ torch.linalg.inv(matrix).T
        inside = ((expected[:, 0] >= -.5) & (expected[:, 0] <= w - .5)
                  & (expected[:, 1] >= -.5) & (expected[:, 1] <= h - .5))
        torch.testing.assert_close(corners[0][inside], expected[inside], atol=1e-6, rtol=0)
        self.assertTrue(bool((in_frame[0] == inside).all()))
        self.assertTrue(bool((residual[0][inside] < 1e-4).all()))

    def test_flat_records_report_frame_corners(self):
        renderer = ECGRenderer(RenderConfig(height=64, width=96, profile='scan', page_corners=True))
        batch = renderer(torch.zeros(1, 12, 500), 100, photo=False, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(batch.metadata['page_corners'][0],
                                   torch.tensor([[-.5, -.5], [95.5, -.5], [95.5, 63.5], [-.5, 63.5]]))
        self.assertTrue(bool(batch.metadata['page_corners_in_frame'].all()))

    def test_photographed_corners_are_self_consistent(self):
        renderer = ECGRenderer(RenderConfig(height=192, width=256, profile='photo', antialias_samples=1,
                                            page_corners=True))
        parameters = {'zoom': .8, 'angle': .1, 'perspective_x': .05, 'perspective_y': -.04}
        batch = renderer(torch.zeros(2, 12, 500), 100, photo=True, parameters=parameters,
                         generator=torch.Generator().manual_seed(3))
        meta = batch.metadata
        self.assertEqual(tuple(meta['page_corners'].shape), (2, 4, 2))
        visible = meta['page_corners_in_frame']
        self.assertTrue(bool(visible.any()))
        self.assertTrue(bool((meta['page_corner_residual'][visible] < .5).all()))
        self.assertNotIn('page_corners', ECGRenderer(RenderConfig(height=64, width=96)).specification()['config'])


if __name__ == '__main__':
    unittest.main()
