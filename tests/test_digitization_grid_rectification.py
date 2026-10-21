"""Independent coordinate and missing-lattice checks for diagnostic warping."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.digitization_grid_rectification import grid_remap, save_rgb


@unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('scipy'),
                     'Optional digitization dependencies unavailable')
class GridRectification(unittest.TestCase):
    def points(self):
        row, col = np.mgrid[:5, :5]
        ids = np.column_stack((row.ravel(), col.ravel()))
        return ids, ids[:, ::-1]*4.+2

    def test_affine_coordinate_ramps_follow_pixel_centers(self):
        ids, canonical = self.points()
        y, x = np.mgrid[:30, :40]
        rgb = np.stack((x, y, np.zeros_like(x)), -1).astype(np.uint8)
        warped, valid = grid_remap(rgb, canonical+[3, 5], canonical, ids, (22, 22))
        yy, xx = np.nonzero(valid)
        np.testing.assert_array_equal(warped[valid, :2], np.column_stack((xx+3, yy+5)))
        self.assertFalse(valid[0].any())
        self.assertTrue((warped[~valid] == 255).all())

    def test_missing_node_never_gets_bridged_by_a_large_triangle(self):
        ids, canonical = self.points()
        keep = (ids != [2, 2]).any(axis=1)
        rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        _, valid = grid_remap(rgb, canonical[keep], canonical[keep], ids[keep], (24, 24))
        self.assertFalse(valid[10, 10])
        self.assertTrue(valid[4, 4])

    def test_duplicate_id_and_nonfinite_points_are_rejected(self):
        ids, canonical = self.points()
        rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        ids[0] = ids[1]
        with self.assertRaises(ValueError):
            grid_remap(rgb, canonical, canonical, ids, (24, 24))

    def test_only_explicitly_accepted_topology_cells_are_sampled(self):
        ids, canonical = self.points()
        rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        _, valid = grid_remap(rgb, canonical, canonical, ids, (24, 24), allowed_cells=[[0, 0]])
        self.assertTrue(valid[4, 4])
        self.assertFalse(valid[12, 12])

        ids, canonical = self.points()
        canonical[0, 0] = np.nan
        with self.assertRaises(ValueError):
            grid_remap(rgb, canonical, canonical, ids, (24, 24))

    def test_saved_png_preserves_noncontiguous_rgb_and_digest(self):
        import hashlib
        from PIL import Image
        image = np.arange(3*170*190, dtype=np.uint8).reshape(3, 170, 190).transpose(1, 2, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'control.png'
            digest = save_rgb(path, image)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            with Image.open(path) as decoded:
                np.testing.assert_array_equal(np.asarray(decoded), image)


if __name__ == '__main__':
    unittest.main()
