"""Independent coordinate controls for opt-in native observation."""
import unittest

import numpy as np
import torch
from torch import nn

from ecg_gpu.digitization.native_detection import detect_native_grid


class RampDetector(nn.Module):
    """Read absolute x/y from independent RGB ramps, regardless of cropping."""
    def __init__(self, points):
        super().__init__()
        self.points = points
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        _, _, height, width = image.shape
        heat = torch.full((1, 1, height, width), -20., device=image.device)
        offsets = torch.zeros((1, 2, height, width), device=image.device)
        # Red encodes x, green y. These are original-frame values in every tile.
        absolute_x, absolute_y = image[:, 0]*255, image[:, 1]*255
        for x, y in self.points:
            anchor = (absolute_x-x).abs() < .1
            anchor &= (absolute_y-y).abs() < .1
            heat[:, 0][anchor] = 4. if self.calls == 1 else 7.
            offsets[:, 0][anchor], offsets[:, 1][anchor] = .25, -.25
        return {'heat': heat, 'offsets': offsets}


class PresetDetector(nn.Module):
    def forward(self, image):
        _, _, height, width = image.shape
        heat = torch.full((1, 1, height, width), -20., device=image.device)
        offsets = torch.zeros((1, 2, height, width), device=image.device)
        heat[0, 0, 7, 13] = 6.
        offsets[0, :, 7, 13] = torch.tensor([.25, -.25], device=image.device)
        return {'heat': heat, 'offsets': offsets}


class PaddingDetector(PresetDetector):
    def forward(self, image):
        output = super().forward(image)
        output['heat'][0, 0, -2, -2] = 12.
        return output


class UpsampledDetector(nn.Module):
    def forward(self, image):
        _, _, height, width = image.shape
        heat = torch.full((1, 1, height, width), -20., device=image.device)
        offsets = torch.zeros((1, 2, height, width), device=image.device)
        heat[0, 0, 16, 16], heat[0, 0, 16, 19] = 6., 5.
        return {'heat': heat, 'offsets': offsets}


class NativeDetectionTests(unittest.TestCase):
    @staticmethod
    def ramp(height=39, width=57):
        y, x = np.mgrid[:height, :width]
        return np.stack((x, y, np.zeros_like(x)), axis=-1).astype(np.uint8)

    def test_native_odd_tiles_and_last_source_pixels_compose_exactly(self):
        rgb = self.ramp()
        original = rgb.copy()
        truth = np.array([[5, 7], [31, 13], [55, 37], [9, 32]])
        model = RampDetector(truth)
        model.train()
        xy, score, report = detect_native_grid(model, rgb, global_size=(39, 57),
            tile_size=31, overlap=9, roi_margin=5, max_tiles=8, nms_radius=1, dedup_radius=.75)
        self.assertEqual(len(xy), len(truth))
        # Native scores win deduplication. Each point must still refer to the
        # original ramp's x/y, including the rightmost and lowest tile.
        expected = truth+[.25, -.25]
        for point in expected:
            self.assertLess(np.linalg.norm(xy-point, axis=1).min(), 1e-5)
        self.assertEqual(report['returned_native_points'], len(truth))
        self.assertTrue(any(box[2] == 57 for box in report['native_tiles']))
        self.assertTrue(any(box[3] == 39 for box in report['native_tiles']))
        self.assertEqual(report['model_passes'], model.calls)
        self.assertTrue(model.training)
        self.assertTrue((score > .99).all())
        np.testing.assert_array_equal(rgb, original)

    def test_half_pixel_letterbox_uses_rounded_axis_scales(self):
        rgb = self.ramp(32, 80)
        xy, _, report = detect_native_grid(PresetDetector(), rgb,
            global_size=(31, 47), min_roi_points=4)
        expected = (np.array([13.25, 6.75])+.5)/[47/80, 19/32]-.5
        np.testing.assert_allclose(xy, expected[None], atol=2e-6)
        self.assertEqual(report['global_resized_shape'], [19, 47])
        self.assertEqual(report['fallback_reason'], 'too few global points to propose a ROI')

    def test_native_budget_falls_back_to_whole_global_result(self):
        rgb = self.ramp()
        truth = np.array([[5, 7], [31, 13], [55, 37], [9, 32]])
        common = dict(global_size=(39, 57), tile_size=16, overlap=8,
                      roi_margin=8, nms_radius=1)
        xy, score, report = detect_native_grid(RampDetector(truth), rgb, max_tiles=1, **common)
        baseline_xy, baseline_score, baseline = detect_native_grid(RampDetector(truth), rgb,
            max_tiles=0, **common)
        np.testing.assert_array_equal(xy, baseline_xy)
        np.testing.assert_array_equal(score, baseline_score)
        self.assertTrue(report['budget_exhausted'])
        self.assertEqual(report['native_tiles'], [])
        self.assertEqual(report['model_passes'], baseline['model_passes'])

    def test_padding_peaks_cannot_exhaust_global_point_capacity(self):
        rgb = self.ramp(32, 80)
        xy, _, report = detect_native_grid(PaddingDetector(), rgb,
            global_size=(31, 47), max_points=1, min_roi_points=4)
        expected = (np.array([13.25, 6.75])+.5)/[47/80, 19/32]-.5
        np.testing.assert_allclose(xy, expected[None], atol=2e-6)
        self.assertFalse(report['global_decoder_truncated'])

    def test_no_native_observation_preserves_close_global_source_points(self):
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        xy, _, report = detect_native_grid(UpsampledDetector(), rgb,
            global_size=(40, 40), max_tiles=0, dedup_radius=1.5)
        self.assertEqual(len(xy), 2)
        self.assertLess(np.linalg.norm(xy[0]-xy[1]), 1.5)
        self.assertEqual(report['unique_candidates_before_capacity'], 2)

    def test_pixel_budget_is_explicit(self):
        model = RampDetector(np.array([[5, 7], [31, 13], [55, 37], [9, 32]]))
        _, _, report = detect_native_grid(model, self.ramp(), global_size=(39, 57),
            max_roi_pixels=100, tile_size=31, overlap=9)
        self.assertTrue(report['budget_exhausted'])
        self.assertEqual(model.calls, 1)

    def test_empty_global_detections_do_not_use_gold_or_native_roi(self):
        model = RampDetector([])
        xy, scores, report = detect_native_grid(model, self.ramp(), global_size=(39, 57))
        self.assertEqual(xy.shape, (0, 2))
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(model.calls, 1)
        self.assertFalse(report['gold_used_for_inference'])

    def test_capacity_truncation_is_reported(self):
        model = RampDetector(np.array([[5, 7], [31, 13], [55, 37], [9, 32]]))
        xy, _, report = detect_native_grid(model, self.ramp(), global_size=(39, 57),
            max_points=2, min_roi_points=4, nms_radius=1)
        self.assertEqual(len(xy), 2)
        self.assertTrue(report['global_decoder_truncated'])
        self.assertTrue(report['truncated'])

    def test_invalid_inputs_and_budgets_fail_early(self):
        rgb = self.ramp()
        for change in ({'overlap': 512}, {'max_tiles': -1}, {'threshold': float('nan')},
                       {'dedup_radius': 100.}, {'global_size': (4096, 4096)}, {'max_points': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                detect_native_grid(RampDetector([]), rgb, **change)
        with self.assertRaises(ValueError):
            detect_native_grid(RampDetector([]), rgb.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
