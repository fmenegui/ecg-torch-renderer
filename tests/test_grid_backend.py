"""Grid adapter integration with independent exact point observations."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ecg_gpu.digitization.denoising_grid import DenoisingGridNet
from ecg_gpu.digitization.grid_backend import AdaptiveGridRectifier


class GridBackendTests(unittest.TestCase):
    def test_independent_exact_lattice_runs_through_the_complete_adapter(self):
        yy, xx = np.mgrid[:96, :128]
        rgb = np.stack((xx, yy, (xx + yy) % 251), axis=-1).astype(np.uint8)
        row, column = np.mgrid[1:9, 1:12]
        xy = np.column_stack((column.ravel() * 10, row.ravel() * 10)).astype(float)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            net = DenoisingGridNet(width=4)
            torch.save({"model_class": "DenoisingGridNet", "width": 4, "threshold": .85,
                        "state_dict": net.state_dict()}, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            adapter = AdaptiveGridRectifier(checkpoint, expected_sha256=digest, max_side=256, text_orientation=False)
            seen = []

            def detect(model, image, **kwargs):
                seen.append(image.copy())
                return xy, np.ones(len(xy)), {"controlled_exact_points": True}

            with patch("ecg_gpu.digitization.grid_backend.detect_native_grid", side_effect=detect), \
                 patch("ecg_gpu.digitization.grid_backend.remove_periodic_banding", return_value=(np.zeros_like(rgb), {"applied": False})):
                corrected, evidence = adapter(rgb, source_id="independent-rgb")
            self.assertEqual(len(seen), 1)
            np.testing.assert_array_equal(seen[0], rgb)
            self.assertTrue(evidence.original_rgb_sampled)
            self.assertTrue(evidence.full_frame_preserved)
            self.assertFalse(evidence.reference_used)
            self.assertEqual(evidence.original_source_id, "independent-rgb")
            self.assertLess(max(corrected.shape[:2]), 257)
            self.assertFalse(adapter.last_report["learned_rgb_restoration_used"])
            np.testing.assert_array_equal(adapter.last_observations["detected_xy"], xy)
            np.testing.assert_array_equal(adapter.last_observations["observed_xy"], xy)
            self.assertEqual(adapter.last_report["original_source_id"], "independent-rgb")
            self.assertEqual(adapter.last_report["status"], "rectified")
            self.assertEqual(adapter.last_observations["source"].shape, adapter.last_observations["target_mm"].shape)
            json.dumps(adapter.last_report, allow_nan=False)
            # Relative ID (0,0) belongs to source (10,10), so the exact
            # physical map is (x-10)/2, (y-10)/2. Compare colors at independently
            # computed output centres, excluding its explicit white guard.
            sampling = adapter.last_report["sampling"]
            origin, density = np.asarray(sampling["output_origin_mm"]), sampling["effective_pixels_per_mm"]
            oy, ox = np.mgrid[:corrected.shape[0], :corrected.shape[1]]
            expected_x = 2 * (ox / density + origin[0]) + 10
            expected_y = 2 * (oy / density + origin[1]) + 10
            valid = (expected_x > 2) & (expected_x < 125) & (expected_y > 2) & (expected_y < 93)
            np.testing.assert_allclose(corrected[..., 0][valid], expected_x[valid], atol=1.1)
            np.testing.assert_allclose(corrected[..., 1][valid], expected_y[valid], atol=1.1)
            # A failed second image must not inherit the first geometry receipt.
            with patch("ecg_gpu.digitization.grid_backend.detect_native_grid", side_effect=RuntimeError("missing points")):
                with self.assertRaises(RuntimeError):
                    adapter(rgb, source_id="another")
            self.assertEqual(adapter.last_report["original_source_id"], "another")
            self.assertEqual(adapter.last_report["stage"], "point_detection")
            self.assertEqual(adapter.last_report["status"], "incomplete")
            self.assertNotIn("fit", adapter.last_report)
            self.assertEqual(adapter.last_observations, {})

    def test_changed_checkpoint_rejected_before_network_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"untrusted bytes")
            with self.assertRaises(ValueError):
                AdaptiveGridRectifier(checkpoint)

    def test_rejected_grid_preserves_observations_without_claiming_a_completed_mesh(self):
        image = np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)
        xy = np.array([[2., 2.], [10., 2.], [2., 10.], [10., 10.]])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            net = DenoisingGridNet(width=4)
            torch.save({"model_class": "DenoisingGridNet", "width": 4, "threshold": .85,
                        "state_dict": net.state_dict()}, checkpoint)
            adapter = AdaptiveGridRectifier(checkpoint, expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                            max_side=256, text_orientation=False)
            with patch("ecg_gpu.digitization.grid_backend.detect_native_grid", return_value=(xy, np.ones(4), {})), \
                 patch("ecg_gpu.digitization.grid_backend.remove_periodic_banding", return_value=(image, {"applied": False})):
                with self.assertRaisesRegex(ValueError, "Insufficient supported"):
                    adapter(image, source_id="rejected-source")
            np.testing.assert_array_equal(adapter.last_observations["detected_xy"], xy)
            self.assertEqual(adapter.last_report["status"], "rejected")
            self.assertEqual(adapter.last_report["original_source_id"], "rejected-source")
            self.assertNotIn("source", adapter.last_observations)


if __name__ == "__main__":
    unittest.main()
