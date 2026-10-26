"""Independent affine/projective grids and adversarial missing-support cases."""

import importlib.util
import unittest

import numpy as np

from ecg_gpu.digitization.cell_mesh import forward_cell_map
from ecg_gpu.digitization.consensus_lattice import organize_consensus_lattice


@unittest.skipUnless(
    importlib.util.find_spec("cv2") and importlib.util.find_spec("scipy"),
    "Optional offline grid dependencies unavailable",
)
class ConsensusLatticeTests(unittest.TestCase):
    def grid(self):
        row, col = np.mgrid[:12, :18]
        return np.column_stack((row.ravel(), col.ravel()))

    def relative_ids(self, result, truth):
        selected = result["selected_indices"]
        predicted = np.asarray(result["ids"])
        np.testing.assert_array_equal(
            predicted - predicted[0], truth[selected] - truth[selected][0]
        )

    def test_anisotropic_sheared_permuted_grid_has_correct_physical_scale(self):
        ids = self.grid()
        affine = np.array([[18.0, 3.0], [2.0, 8.0]])
        observed = ids[:, ::-1] @ affine + [43.0, 19.0]
        order = np.random.default_rng(331).permutation(len(ids))
        result = organize_consensus_lattice(observed[order])
        self.relative_ids(result, ids[order])
        self.assertEqual(len(result["cells"]), 11 * 17)
        selected = result["selected_indices"]
        inferred = np.asarray(result["ids"])
        query = np.array([[2.25, 3.75], [9.2, 7.1]])
        mapped = forward_cell_map(
            observed[order][selected],
            inferred[:, ::-1] * 5.0,
            inferred,
            query @ affine + [43.0, 19.0],
            result["cells"],
        )
        np.testing.assert_allclose(mapped, query * 5, atol=1e-10)
        self.assertFalse(result["oracle_correspondence"])
        self.assertFalse(result["semantic_orientation_inferred"])

    def test_two_missing_rows_align_observed_components_without_bridging_hole(self):
        ids = self.grid()
        ids = ids[~np.isin(ids[:, 0], [5, 6])]
        observed = ids[:, ::-1] * 12.0 + [31.0, 27.0]
        result = organize_consensus_lattice(observed)
        self.relative_ids(result, ids)
        self.assertEqual(result["component_points"], len(ids))
        self.assertGreater(result["merged_components"], 0)
        selected = result["selected_indices"]
        inferred = np.asarray(result["ids"])
        query = np.array([[3.2, 2.3], [3.2, 5.6], [3.2, 8.3]]) * 12.0 + [31.0, 27.0]
        mapped = forward_cell_map(
            observed[selected],
            inferred[:, ::-1] * 5.0,
            inferred,
            query,
            result["cells"],
        )
        self.assertTrue(np.isfinite(mapped[[0, 2]]).all())
        self.assertTrue(np.isnan(mapped[1]).all())

    def test_fractional_phase_component_is_not_assigned_a_fabricated_integer_origin(
        self,
    ):
        ids = self.grid()
        ids = ids[~np.isin(ids[:, 0], [5, 6])]
        observed = ids[:, ::-1] * 12.0 + [31.0, 27.0]
        observed[ids[:, 0] >= 7, 0] += 5.4
        result = organize_consensus_lattice(observed)
        self.assertLess(result["component_points"], len(ids))
        self.assertEqual(result["merged_components"], 0)

    def test_projective_grid_keeps_relative_ids_and_input_permutation_invariance(self):
        ids = self.grid()
        canonical = ids[:, ::-1] * 12.0
        homogeneous = np.column_stack((canonical, np.ones(len(ids))))
        matrix = np.array(
            [[1.1, 0.07, 20.0], [0.04, 0.9, 30.0], [0.0005, -0.0002, 1.0]]
        )
        projected = homogeneous @ matrix.T
        observed = projected[:, :2] / projected[:, 2:]
        first = organize_consensus_lattice(observed)
        self.relative_ids(first, ids)
        order = np.random.default_rng(91).permutation(len(ids))
        second = organize_consensus_lattice(observed[order])
        self.relative_ids(second, ids[order])
        self.assertEqual(len(first["cells"]), len(second["cells"]))
        self.assertEqual(first["hypotheses"], second["hypotheses"])

    def test_collinear_duplicate_and_nonfinite_inputs_never_make_cells(self):
        for points in (
            np.empty((0, 2)),
            np.zeros((8, 2)),
            np.column_stack((np.arange(20), np.zeros(20))),
        ):
            self.assertEqual(organize_consensus_lattice(points)["status"], "rejected")
        with self.assertRaises(ValueError):
            organize_consensus_lattice([[np.nan, 0.0]])
        with self.assertRaises(ValueError):
            organize_consensus_lattice(np.zeros((3, 2)), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
