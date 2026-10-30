import unittest

import numpy as np

from reproductions.powerful_medical.gridder import Grid, extract_centroids, infer_missing, organize, warp_cells


class GridderTests(unittest.TestCase):
    @staticmethod
    def fixture():
        row, col = np.mgrid[:6, :8]
        return np.stack([col.ravel() * 20 + 10, row.ravel() * 20 + 10], 1), np.stack([row.ravel(), col.ravel()], 1)

    def test_weighted_centroid_is_fractional(self):
        mask = np.zeros((8, 8), float)
        mask[2, 3], mask[2, 4] = 1, 0.75
        xy = extract_centroids(mask)
        self.assertAlmostEqual(xy[0, 0], 3 + 0.75 / 1.75)
        self.assertEqual(xy[0, 1], 2)

    def test_probability_and_byte_masks_agree(self):
        mask = np.zeros((8, 8), np.uint8)
        mask[3, 3] = 255
        np.testing.assert_array_equal(extract_centroids(mask), extract_centroids(mask.astype(float) / 255))

    def test_regular_grid_preserves_relative_identity(self):
        points, ids = self.fixture()
        grid = organize(points)
        np.testing.assert_array_equal(grid.ids_rc, ids)
        self.assertEqual(grid.diagnostics["orientation"], "D4_unresolved")
        self.assertEqual(grid.diagnostics["physical_pitch"], "not_inferred")

    def test_shuffled_observations_keep_identity(self):
        points, ids = self.fixture()
        permutation = np.random.default_rng(19).permutation(len(points))
        grid = organize(points[permutation])
        np.testing.assert_array_equal(grid.ids_rc, ids[permutation])

    def test_duplicate_detections_are_not_separate_nodes(self):
        points, _ = self.fixture()
        with self.assertRaises(ValueError):
            organize(np.r_[points, points[:1]])

    def test_grid_identity_collisions_are_rejected(self):
        points, ids = self.fixture()
        ids[-1] = ids[0]
        with self.assertRaisesRegex(ValueError, "Colliding"):
            Grid(points, ids, np.ones(len(ids), bool), {})

    def test_no_closed_cell_is_rejected(self):
        with self.assertRaises(ValueError):
            organize(np.array([[0, 0], [10, 0], [20, 0], [30, 0]], float))

    def test_inferred_nodes_have_distinct_provenance(self):
        points, ids = self.fixture()
        keep = ~np.all(ids == [2, 3], axis=1)
        grid = Grid(points[keep], ids[keep], np.ones(keep.sum(), bool), {})
        complete = infer_missing(grid)
        self.assertEqual(len(complete.points_xy), len(points))
        self.assertEqual(int((~complete.observed).sum()), 1)
        np.testing.assert_allclose(complete.points_xy[-1], points[~keep][0], atol=1e-10)

    def test_identity_cell_samples_original_colour(self):
        points, ids = self.fixture()
        grid = Grid(points, ids, np.ones(len(ids), bool), {})
        yy, xx = np.mgrid[:130, :170]
        image = np.stack([xx, yy, (xx + yy) % 256], -1).astype(np.uint8)
        result = warp_cells(image, grid, pixels_per_cell=20)
        np.testing.assert_array_equal(result["rgb"], image[10:110, 10:150])
        self.assertTrue(result["support"].all())

    def test_inferred_cells_are_explicit_and_opt_in(self):
        points, ids = self.fixture()
        observed = ~np.all(ids == [2, 3], axis=1)
        grid = Grid(points, ids, observed, {})
        image = np.full((130, 170, 3), 127, np.uint8)
        native = warp_cells(image, grid, 20)
        inferred = warp_cells(image, grid, 20, include_inferred=True)
        self.assertLess(native["support"].sum(), inferred["support"].sum())
        self.assertEqual(inferred["inferred_support"].sum(), 4 * 20 * 20)


if __name__ == "__main__":
    unittest.main()
