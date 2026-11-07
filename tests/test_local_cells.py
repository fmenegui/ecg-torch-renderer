import unittest

import numpy as np

from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.local_cells import locally_supported_cells


class LocalCellTests(unittest.TestCase):
    def test_local_filter_keeps_regular_grid_after_global_rejection(self):
        y, x = np.mgrid[:8, :10]
        xy = np.column_stack((x.ravel(), y.ravel())).astype(float)*20
        native = organize_lattice(xy)
        native['status'] = 'rejected'
        filtered = locally_supported_cells(xy, native)
        self.assertEqual(filtered['partial_status'], 'partial')
        self.assertEqual(filtered['cells'], native['cells'])
        self.assertEqual(filtered['native_status'], 'rejected')
        self.assertFalse(filtered['global_injectivity_checked'])

    def test_stretched_cells_removed_without_discarding_distant_cells(self):
        y, x = np.mgrid[:8, :10]
        xy = np.column_stack((x.ravel(), y.ravel())).astype(float)*20
        native = organize_lattice(xy)
        xy[22] += [15, 0]
        filtered = locally_supported_cells(xy, native)
        self.assertGreater(filtered['removed_cells'], 0)
        self.assertIn([5, 5], filtered['cells'])


if __name__ == '__main__':
    unittest.main()
