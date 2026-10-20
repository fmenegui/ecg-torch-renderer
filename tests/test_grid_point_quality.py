"""Regression contracts for mixed precision, provenance, memory and replay."""
import copy
from dataclasses import asdict
import math
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from ecg_gpu import (ECGSupervisionLayer, GridPointConfig, RenderConfig,
                     SupervisionConfig, decode_grid_points, grid_point_loss,
                     rasterize_grid_points)


class GridPointQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def layer(self, supervision=None):
        return ECGSupervisionLayer(fs=100, photo=False, seed=71,
            config=RenderConfig(height=96, width=128, profile='clean',
                                surface_resolution=32, shadow_samples=0, antialias_samples=1),
            supervision=supervision)

    def test_half_precision_without_offsets_has_finite_loss_and_gradient(self):
        logits = torch.full((1, 1, 256, 256), 2., dtype=torch.float16, requires_grad=True)
        target = torch.zeros_like(logits)
        result = grid_point_loss(logits, None, {'grid_points_mask': target})
        torch.testing.assert_close(result['loss'], F.binary_cross_entropy_with_logits(logits.float(), target.float()))
        self.assertEqual(float(result['offset_loss'].detach()), 0.)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad > 0).all())

    def test_half_precision_support_reduction_does_not_overflow(self):
        heat = torch.zeros(1, 1, 256, 256, dtype=torch.float16)
        prediction = torch.full((1, 2, 256, 256), .25, dtype=torch.float16, requires_grad=True)
        batch = {'grid_points_mask': heat, 'grid_points_offsets': torch.zeros_like(prediction),
                 'grid_points_offset_valid': torch.ones_like(heat)}
        result = grid_point_loss(heat, prediction, batch)
        self.assertAlmostEqual(float(result['offset_loss'].detach()), .5*.25**2)
        result['loss'].backward()
        self.assertTrue((prediction.grad > 0).all())

    def test_inactive_nonfinite_offsets_have_zero_loss_and_gradient(self):
        heat = torch.zeros(1, 1, 4, 5, requires_grad=True)
        offsets = torch.full((1, 2, 4, 5), float('nan'))
        offsets[..., 1, 2] = .25
        offsets.requires_grad_()
        truth = torch.full_like(offsets, float('inf'))
        truth[..., 1, 2] = 0.
        support = torch.zeros_like(heat)
        support[..., 1, 2] = 1.
        result = grid_point_loss(heat, offsets, {'grid_points_mask': torch.zeros_like(heat),
            'grid_points_offsets': truth, 'grid_points_offset_valid': support})
        self.assertAlmostEqual(float(result['offset_loss'].detach()), .5*.25**2)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(offsets.grad).all())
        self.assertEqual(int(torch.count_nonzero(offsets.grad)), 2)

    def test_invalid_training_contracts_fail_clearly(self):
        heat = torch.zeros(1, 1, 4, 5)
        for batch, error in ((None, TypeError), ({}, ValueError),
                             ({'grid_points_mask': 'mask'}, ValueError)):
            with self.subTest(batch=batch), self.assertRaises(error):
                grid_point_loss(heat, None, batch)
        for value in (True, '1', torch.tensor(1.), float('nan')):
            for option in ('offset_weight', 'pos_weight'):
                with self.subTest(value=value, option=option), self.assertRaises(ValueError):
                    grid_point_loss(heat, None, {'grid_points_mask': heat}, **{option: value})
            with self.assertRaises(ValueError):
                decode_grid_points(heat, threshold=value)
        for value in (-.1, 1.1, float('nan')):
            with self.assertRaisesRegex(RuntimeError, 'grid_points_mask'):
                grid_point_loss(heat, None, {'grid_points_mask': torch.full_like(heat, value)})
        for key, value, message in (('grid_points_offset_valid', -1., 'support'),
                                    ('grid_points_offsets', .6, 'Supervised offsets')):
            batch = {'grid_points_mask': heat, 'grid_points_offsets': torch.zeros(1, 2, 4, 5),
                     'grid_points_offset_valid': torch.ones_like(heat)}
            batch[key].fill_(value)
            with self.assertRaisesRegex(RuntimeError, message):
                grid_point_loss(heat, torch.zeros(1, 2, 4, 5), batch)
        empty = heat[:0]
        with self.assertRaises(ValueError):
            grid_point_loss(empty, None, {'grid_points_mask': empty})
        with self.assertRaises(ValueError):
            decode_grid_points(empty)

    def test_config_validation_and_string_shorthand(self):
        for name in ('sigma_pixels', 'radius_pixels', 'max_error_pixels', 'min_visibility', 'max_image_error_pixels'):
            for value in (True, '1', torch.tensor(1.)):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    GridPointConfig(**{name: value})
        self.assertEqual(SupervisionConfig(outputs='grid_points', leads='II'),
                         SupervisionConfig(outputs=('grid_points',), leads=('II',)))
        with self.assertRaisesRegex(ValueError, 'Include'):
            SupervisionConfig(outputs='grid_points_mask', grid_point_config=GridPointConfig(return_offsets=True))
        with mock.patch('ecg_gpu.gpu_supervision.ECGImageLayer.__init__', side_effect=AssertionError('constructed')):
            with self.assertRaises(TypeError):
                ECGSupervisionLayer(supervision={})
            with self.assertRaisesRegex(ValueError, 'photo'):
                ECGSupervisionLayer(photo='yes')
        for config in (False, 0, {}):
            with self.assertRaises(TypeError):
                rasterize_grid_points(torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool),
                                      torch.ones(1, 1, 4, 5), config=config)

    def test_empty_points_and_invalid_visibility_maps(self):
        empty = rasterize_grid_points(torch.empty(1, 0, 2), torch.empty(1, 0, dtype=torch.bool),
            torch.ones(1, 1, 4, 5), config=GridPointConfig(return_offsets=True))
        self.assertFalse(empty['grid_points_mask'].any())
        self.assertFalse(empty['grid_points_offset_valid'].any())
        self.assertEqual(empty['grid_points_anchor_xy'].shape, (1, 0, 2))
        for value in (-.1, 1.1, float('nan')):
            with self.assertRaisesRegex(RuntimeError, 'valid'):
                rasterize_grid_points(torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool),
                                      torch.full((1, 1, 4, 5), value))
        with self.assertRaisesRegex(ValueError, 'B>=1'):
            rasterize_grid_points(torch.zeros(0, 1, 2), torch.ones(0, 1, dtype=torch.bool), torch.ones(0, 1, 4, 5))

    def test_direct_render_inherits_config_and_supervise_checks_provenance(self):
        config = GridPointConfig(sigma_pixels=1.2, return_offsets=True)
        layer = self.layer(SupervisionConfig(outputs='grid_points', grid_point_config=config))
        signals = torch.zeros(1, 12, 501)
        rendered = layer.render_batch(signals, return_grid_points=True)
        self.assertEqual(rendered.metadata['grid_points']['configuration'], asdict(config))
        self.assertIn('grid_points_offsets', layer.supervise(rendered))
        del rendered.targets['grid_points_offsets']
        with self.assertRaisesRegex(ValueError, 'lacks grid-point targets'):
            layer.supervise(rendered)
        other = layer.render_batch(signals, return_grid_points=True, grid_point_config=GridPointConfig())
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            layer.supervise(other)
        other.metadata.pop('grid_points')
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            layer.supervise(other)

    def test_legacy_lead_only_checkpoint_ignores_unused_grid_settings(self):
        layer = self.layer()
        signals = torch.zeros(1, 12, 501)
        layer(signals)
        saved = copy.deepcopy(layer.state_dict())
        expected = layer(signals)
        for legacy in (False, True):
            state = copy.deepcopy(saved)
            if legacy:
                state['_extra_state']['supervision']['grid_point_config'] = asdict(GridPointConfig(sigma_pixels=2.))
            replacement = self.layer(SupervisionConfig(grid_point_config=GridPointConfig(return_offsets=True)))
            replacement.load_state_dict(state)
            observed = replacement(signals)
            self.assertTrue(all(torch.equal(expected[key], observed[key]) for key in expected))
        points = self.layer(SupervisionConfig(outputs='grid_points'))
        changed = self.layer(SupervisionConfig(outputs='grid_points', grid_point_config=GridPointConfig(sigma_pixels=2.)))
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            changed.load_state_dict(points.state_dict())

    def test_chunked_splats_match_independent_point_distance_reference(self):
        # More than 2^20 patch entries exercises point-chunk boundaries.
        xy = torch.tensor([[[1.35, 2.7], [5.2, 4.1], [7.7, 6.3]]]).repeat(2, 6000, 1)
        present = torch.ones(xy.shape[:2], dtype=torch.bool)
        valid = torch.ones(2, 1, 9, 10)
        valid[1, :, :2] = 0.
        for shape in ('gaussian', 'disk'):
            config = GridPointConfig(mask_shape=shape, radius_pixels=2.5)
            actual = rasterize_grid_points(xy, present, valid, config=config)['grid_points_mask']
            expected = torch.zeros_like(actual)
            for y in range(9):
                for x in range(10):
                    distances = [(x-cx)**2+(y-cy)**2 for cx, cy in xy[0, :3].tolist()]
                    distance = min(distances)
                    value = (math.exp(-distance/(2*config.sigma_pixels**2)) if shape == 'gaussian' else 1.)
                    expected[:, 0, y, x] = value if distance <= 2.5**2 else 0.
            torch.testing.assert_close(actual, expected*valid, atol=2e-7, rtol=1e-6)
        # Large batch and few points exercises the separate batch boundary.
        xy = torch.tensor([[[1.35, 2.7]]]).expand(22000, -1, -1)
        actual = rasterize_grid_points(xy, torch.ones(22000, 1, dtype=torch.bool), torch.ones(22000, 1, 4, 4))
        mask = actual['grid_points_mask']
        torch.testing.assert_close(mask, mask[:1].expand_as(mask))


if __name__ == '__main__':
    unittest.main()
