"""Subpixel centers survive mask-size changes and U-Net target decoding."""
from dataclasses import replace
import unittest
from unittest import mock

import torch

from ecg_gpu import GridPointConfig, rasterize_grid_points, grid_point_loss, decode_grid_points


class GridPointLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def points(self, config=None):
        xy = torch.tensor([[[8.35, 9.7], [23.5, 24.5], [38.02, 15.49]]])
        present = torch.ones(1, 3, dtype=torch.bool)
        valid = torch.ones(1, 1, 48, 52)
        return xy, present, valid, rasterize_grid_points(xy, present, valid, config=config)

    def test_mask_size_preserves_fractional_centers_and_offset_targets(self):
        base = GridPointConfig(return_offsets=True)
        xy, _, _, a = self.points(base)
        _, _, _, b = self.points(replace(base, sigma_pixels=2.))
        _, _, _, c = self.points(replace(base, mask_shape='disk', radius_pixels=2.5))
        self.assertLess(float(a['grid_points_mask'].sum()), float(b['grid_points_mask'].sum()))
        self.assertTrue(((c['grid_points_mask'] == 0) | (c['grid_points_mask'] == 1)).all())
        for output in (a, b, c):
            self.assertTrue(torch.equal(a['grid_points_offsets'], output['grid_points_offsets']))
            anchor = output['grid_points_anchor_xy']
            index = anchor[..., 1]*52+anchor[..., 0]
            offset = output['grid_points_offsets'].flatten(2).gather(2, index[:, None].expand(-1, 2, -1)).transpose(1, 2)
            torch.testing.assert_close(anchor.float()+offset, xy, atol=1e-6, rtol=0)

    def test_disk_radius_matches_independent_pixel_center_distance(self):
        xy = torch.tensor([[[6.35, 7.7]]])
        valid = torch.ones(1, 1, 18, 20)
        yy, xx = torch.meshgrid(torch.arange(18), torch.arange(20), indexing='ij')
        for radius in (.3, 1., 2.5, 9.):
            c = GridPointConfig(mask_shape='disk', radius_pixels=radius)
            target = rasterize_grid_points(xy, torch.ones(1, 1, dtype=torch.bool), valid, config=c)
            expected = ((xx-6.35)**2+(yy-7.7)**2 <= radius**2).float()
            self.assertTrue(torch.equal(target['grid_points_mask'][0, 0], expected))

    def test_gaussian_values_follow_fractional_center_without_rounding(self):
        xy = torch.tensor([[[8.35, 9.7]]])
        target = rasterize_grid_points(xy, torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, 20, 24),
                                      config=GridPointConfig(sigma_pixels=1.2, radius_pixels=2.5))
        import math
        expected = math.exp(-((8-8.35)**2+(10-9.7)**2)/(2*1.2**2))
        self.assertAlmostEqual(float(target['grid_points_mask'][0, 0, 10, 8]), expected, places=6)
        self.assertEqual(float(target['grid_points_mask'][0, 0, 10, 4]), 0.)

    def test_offset_collisions_and_occluded_anchors_are_not_averaged(self):
        xy = torch.tensor([[[4.1, 5.2], [4.4, 5.3], [10.2, 12.1], [18.3, 18.2]]])
        valid = torch.ones(1, 1, 24, 24)
        valid[..., 12, 10] = 0
        output = rasterize_grid_points(xy, torch.ones(1, 4, dtype=torch.bool), valid,
                                      config=GridPointConfig(return_offsets=True))
        self.assertEqual(output['grid_points_anchor_valid'].tolist(), [[False, False, False, True]])
        self.assertEqual(int(output['grid_points_offset_valid'].sum()), 1)
        self.assertFalse(output['grid_points_offsets'][..., 5, 4].any())
        self.assertEqual(float(output['grid_points_mask'][..., 12, 10].sum()), 0.)

    def test_decoder_recovers_gaussian_subpixel_centers_for_multiple_sizes(self):
        for sigma in (.6, .8, 1.2, 2.):
            xy, _, _, targets = self.points(GridPointConfig(sigma_pixels=sigma, return_offsets=True))
            for use_offsets in (False, True):
                decoded = decode_grid_points(targets['grid_points_mask'],
                    targets['grid_points_offsets'] if use_offsets else None, max_points=8, nms_radius=2)
                found = decoded['xy'][0, decoded['valid'][0]]
                self.assertEqual(len(found), 3)
                error = torch.cdist(xy[0], found).amin(-1)
                self.assertLess(float(error.max()), 1e-4)
                self.assertFalse(decoded['truncated'].any())

    def test_decoder_empty_border_capacity_and_invalid_offsets(self):
        heatmap = torch.zeros(1, 1, 12, 12)
        self.assertFalse(decode_grid_points(heatmap)['valid'].any())
        heatmap[..., 0, 0] = .9
        heatmap[..., 8, 8] = .8
        decoded = decode_grid_points(heatmap, max_points=1)
        self.assertTrue(decoded['truncated'].all())
        self.assertEqual(decoded['xy'][0, 0].tolist(), [0., 0.])
        offsets = torch.full((1, 2, 12, 12), 2.)
        self.assertFalse(decode_grid_points(heatmap, offsets)['valid'].any())

    def test_loss_trains_both_heads_and_handles_absent_anchors(self):
        _, _, _, batch = self.points(GridPointConfig(return_offsets=True))
        model = torch.nn.Conv2d(3, 3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        image = torch.randn(1, 3, 48, 52)
        before = model.weight.detach().clone()
        logits = model(image)
        losses = grid_point_loss(logits[:, :1], logits[:, 1:], batch, pos_weight=10)
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(losses['loss']))
        self.assertTrue((model.weight.grad.flatten(1).abs().sum(-1) > 0).all())
        optimizer.step()
        self.assertFalse(torch.equal(model.weight, before))
        heat = torch.zeros(1, 1, 48, 52, requires_grad=True)
        offsets = torch.ones(1, 2, 48, 52, requires_grad=True)
        batch['grid_points_offset_valid'].zero_()
        empty = grid_point_loss(heat, offsets, batch)
        self.assertEqual(float(empty['offset_loss'].detach()), 0.)
        empty['loss'].backward()
        self.assertFalse(offsets.grad.any())
        self.assertTrue(torch.isfinite(grid_point_loss(heat, None, batch)['loss']))

    def test_new_paths_have_no_host_transfers(self):
        with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU')), \
             mock.patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy')), \
             mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar sync')):
            _, _, _, targets = self.points(GridPointConfig(return_offsets=True, mask_shape='disk', radius_pixels=2.5))
            grid_point_loss(torch.zeros(1, 1, 48, 52), torch.zeros(1, 2, 48, 52), targets)
            decode_grid_points(targets['grid_points_mask'], targets['grid_points_offsets'])

    def test_configuration_and_rasterizer_validation(self):
        for options in ({'mask_shape':'square'}, {'mask_shape':'disk'}, {'radius_pixels':0},
                        {'radius_pixels':17}, {'radius_pixels':float('nan')}, {'return_offsets':1},
                        {'refinement_steps':9}, {'refinement_steps':True}, {'max_image_error_pixels':0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                GridPointConfig(**options)
        xy = torch.zeros(1, 2, 2)
        present = torch.ones(1, 2, dtype=torch.bool)
        valid = torch.ones(1, 1, 12, 12)
        for args in ((xy[..., 0], present, valid), (xy, present, valid[:, 0]), (xy.double(), present, valid)):
            with self.assertRaises(ValueError):
                rasterize_grid_points(*args)
        with self.assertRaises(TypeError):
            rasterize_grid_points(xy, present, valid, config=object())

    def test_loss_and_decoder_invalid_contracts(self):
        _, _, _, batch = self.points(GridPointConfig(return_offsets=True))
        heat, offsets = torch.zeros(1, 1, 48, 52), torch.zeros(1, 2, 48, 52)
        for options in ({'offset_weight':-1}, {'pos_weight':0}):
            with self.assertRaises(ValueError):
                grid_point_loss(heat, offsets, batch, **options)
        with self.assertRaises(ValueError):
            grid_point_loss(heat[:, :, :-1], offsets, batch)
        with self.assertRaises(ValueError):
            grid_point_loss(heat, offsets[:, :1], batch)
        with self.assertRaises(ValueError):
            grid_point_loss(heat, offsets, {'grid_points_mask':heat})
        for options in ({'threshold':0}, {'max_points':0}, {'nms_radius':0}):
            with self.assertRaises(ValueError):
                decode_grid_points(heat, **options)
        for bad in (heat[:, 0], heat.long()):
            with self.assertRaises(ValueError):
                decode_grid_points(bad)
        with self.assertRaises(ValueError):
            decode_grid_points(heat, offsets[:, :1])


if __name__ == '__main__':
    unittest.main()
