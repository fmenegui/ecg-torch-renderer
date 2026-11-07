import unittest

import numpy as np

from ecg_gpu.digitization.cell_mesh import forward_cell_map
from ecg_gpu.digitization.cycle_lattice import organize_cycle_lattice


class CycleLatticeTests(unittest.TestCase):
    def test_two_step_links_keep_both_observed_sides_without_filling_a_missing_row(self):
        row, col = np.mgrid[:9, :12]
        ids = np.column_stack((row.ravel(), col.ravel()))
        ids = ids[ids[:, 0] != 4]
        xy = ids[:, ::-1].astype(float)*10
        result = organize_cycle_lattice(xy)
        chosen = result['selected_indices']
        inferred = np.asarray(result['ids'])
        np.testing.assert_array_equal(inferred-inferred[0], ids[chosen]-ids[chosen][0])
        self.assertEqual(len(chosen), len(xy))
        mapped = forward_cell_map(xy[chosen], inferred[:, ::-1]*10., inferred,
                                 np.array([[15., 15.], [15., 40.], [15., 65.]]), result['cells'])
        self.assertTrue(np.isfinite(mapped[[0, 2]]).all())
        self.assertTrue(np.isnan(mapped[1]).all())

    def test_permuted_curved_grid_recovers_relative_integer_identities(self):
        row, col = np.mgrid[:12, :18]
        ids = np.column_stack((row.ravel(), col.ravel()))
        xy = ids[:, ::-1].astype(float)*12
        xy[:, 1] += 3*np.sin(xy[:, 0]/35)
        angle = .2
        xy = xy@np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])+[17, 22]
        order = np.random.default_rng(7).permutation(len(xy))
        result = organize_cycle_lattice(xy[order])
        truth = ids[order][result['selected_indices']]
        inferred = np.asarray(result['ids'])
        np.testing.assert_array_equal(inferred-inferred[0], truth-truth[0])
        self.assertFalse(result['oracle_correspondence'])
        self.assertEqual(len(result['cells']), 11*17)

    def test_bad_inputs_and_collinear_points_do_not_create_a_mesh(self):
        for xy in (np.zeros((2, 2)), np.zeros((10, 2)), np.column_stack((np.arange(20), np.zeros(20)))):
            self.assertEqual(organize_cycle_lattice(xy)['status'], 'rejected')
        with self.assertRaises(ValueError):
            organize_cycle_lattice([[np.nan, 1.]])


if __name__ == '__main__':
    unittest.main()
