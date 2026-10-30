"""Grid adapter integration with independent exact point observations."""

import hashlib
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
            self.assertIsNone(adapter.last_report)

    def test_changed_checkpoint_rejected_before_network_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"untrusted bytes")
            with self.assertRaises(ValueError):
                AdaptiveGridRectifier(checkpoint)


if __name__ == "__main__":
    unittest.main()
