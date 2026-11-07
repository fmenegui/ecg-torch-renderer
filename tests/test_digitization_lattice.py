"""Lattice ordering against independent analytic point sets, without oracle IDs."""
import importlib.util
import unittest

import numpy as np

from ecg_gpu.digitization.lattice import organize_lattice


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'Optional scipy unavailable')
class LatticeOrganization(unittest.TestCase):
    def make_grid(self):
        row, col = np.mgrid[:9, :12]
        ids = np.column_stack((row.ravel(), col.ravel()))
        return ids, ids[:, ::-1].astype(float)*10

    def test_permutation_translation_and_rotation_recover_relative_ids(self):
        ids, points = self.make_grid()
        angle = .23
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        permutation = np.random.default_rng(7).permutation(len(points))
        points = (points@matrix.T+[25.3, 18.2])[permutation]
        result = organize_lattice(points)
        self.assertEqual(result['status'], 'accepted')
        selected = result['selected_indices']
        predicted = np.array(result['ids'])
        truth = ids[permutation][selected]
        np.testing.assert_array_equal(predicted-predicted[0], truth-truth[0])
        self.assertEqual(len(result['cells']), 8*11)
        self.assertFalse(result['oracle_correspondence'])

    def test_missing_intersection_remains_missing(self):
        ids, points = self.make_grid()
        keep = (ids != [4, 5]).any(axis=1)
        result = organize_lattice(points[keep])
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['component_points'], len(points)-1)
        self.assertNotIn([4, 5], result['ids'])
        for forbidden in ([3, 4], [3, 5], [4, 4], [4, 5]):
            self.assertNotIn(forbidden, result['cells'])

    def test_smooth_local_deformation_preserves_neighbor_identity(self):
        ids, points = self.make_grid()
        points[:, 1] += 3*np.sin(points[:, 0]/35)
        result = organize_lattice(points)
        self.assertEqual(result['status'], 'accepted')
        predicted = np.array(result['ids'])
        truth = ids[result['selected_indices']]
        np.testing.assert_array_equal(predicted-predicted[0], truth-truth[0])

    def test_insufficient_collinear_duplicate_and_nonfinite_points_fail(self):
        self.assertEqual(organize_lattice([[0., 0.], [1., 0.]])['status'], 'rejected')
        self.assertEqual(organize_lattice([[i, 0.] for i in range(8)])['status'], 'rejected')
        self.assertEqual(organize_lattice([[0., 0.]]*6)['status'], 'rejected')
        with self.assertRaises(ValueError):
            organize_lattice([[np.nan, 0.]])


if __name__ == '__main__':
    unittest.main()
