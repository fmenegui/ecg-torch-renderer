import importlib.util
import unittest

import numpy as np

from ecg_gpu.digitization.cell_mesh import forward_cell_map, remap_cells


class ExplicitCellMeshTests(unittest.TestCase):
    def grid(self):
        row, col = np.mgrid[:9, :12]
        ids = np.column_stack((row.ravel(), col.ravel()))
        return ids, ids[:, ::-1].astype(float)*10

    def test_shear_preserves_all_known_interior_points_and_physical_scale(self):
        ids, canonical = self.grid()
        matrix = np.array([[1., .9], [1., 1.7]])
        observed = canonical@matrix+[100, 40]
        query = np.random.default_rng(1973).uniform([0, 0], [110, 80], (5000, 2))
        cells = [[r, c] for r in range(8) for c in range(11)]
        mapped = forward_cell_map(observed, canonical, ids, query@matrix+[100, 40], cells)
        np.testing.assert_allclose(mapped, query, atol=2e-12)

    def test_forward_and_inverse_share_the_same_nonlinear_cell_complex(self):
        ids, canonical = self.grid()
        observed = canonical.copy()
        observed[:, 0] += 4*np.sin(canonical[:, 1]/30)
        observed[:, 1] += 2*np.sin(canonical[:, 0]/20)
        cells = [[r, c] for r in range(8) for c in range(11)]
        query = np.random.default_rng(13).uniform([1, 1], [109, 79], (2000, 2))
        image_points = forward_cell_map(canonical, observed, ids, query, cells)
        roundtrip = forward_cell_map(observed, canonical, ids, image_points, cells)
        np.testing.assert_allclose(roundtrip, query, atol=1e-10)

    def test_missing_corner_removes_the_whole_cell_in_both_directions(self):
        ids, canonical = self.grid()
        keep = (ids != [4, 5]).any(1)
        query = np.array([[45., 35.], [55., 45.], [15., 15.]])
        cells = [[r, c] for r in range(8) for c in range(11)]
        mapped = forward_cell_map(canonical[keep], canonical[keep], ids[keep], query, cells)
        self.assertTrue(np.isnan(mapped[:2]).all())
        np.testing.assert_allclose(mapped[2], query[2])

    def test_nonlocal_interior_overlap_returns_missing_support(self):
        ids = np.array([[0, 0], [0, 1], [1, 1], [1, 0],
                        [0, 3], [0, 4], [1, 4], [1, 3]])
        canonical = ids[:, ::-1].astype(float)*10
        observed = canonical.copy()
        observed[4:] -= [27, -1]
        result = forward_cell_map(observed, canonical, ids, np.array([[6., 3.]]), [[0, 0], [0, 3]])
        self.assertTrue(np.isnan(result).all())

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'Optional OpenCV sampler unavailable')
    def test_coordinate_ramp_rgb_is_sampled_without_moving_pixel_centers(self):
        ids, canonical = self.grid()
        y, x = np.mgrid[:110, :150]
        rgb = np.stack((x, y, np.zeros_like(x)), -1).astype(np.uint8)
        output, valid = remap_cells(rgb, canonical+[7, 13], canonical, ids, (81, 111))
        self.assertTrue(valid.all())
        yy, xx = np.mgrid[:81, :111]
        np.testing.assert_array_equal(output[..., :2], np.stack((xx+7, yy+13), -1))

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'Optional OpenCV sampler unavailable')
    def test_inverse_sampling_masks_nonlocal_source_overlap(self):
        ids = np.array([[0, 0], [0, 1], [1, 1], [1, 0],
                        [0, 3], [0, 4], [1, 4], [1, 3]])
        canonical = ids[:, ::-1].astype(float)*10
        observed = canonical.copy()
        observed[4:] -= [27, -1]
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        _, valid = remap_cells(rgb, observed, canonical, ids, (12, 42), allowed_cells=[[0, 0], [0, 3]])
        self.assertFalse(valid[3, 6])
        self.assertFalse(valid[2, 33])
        self.assertTrue(valid[1, 1])


if __name__ == '__main__':
    unittest.main()
