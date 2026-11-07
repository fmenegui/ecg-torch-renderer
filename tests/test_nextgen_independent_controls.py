"""Independent physical-query control for the fixed-ID refinement experiment."""
import importlib.util
import unittest

import numpy as np

from ecg_gpu.digitization.cell_mesh import forward_cell_map
from ecg_gpu.digitization.robust_refinement import refine_grid_points


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'requires optional SciPy')
class IndependentRefinementControls(unittest.TestCase):
    @staticmethod
    def fixture():
        row, col = np.mgrid[:10, :12]
        identities = np.column_stack((row.ravel(), col.ravel()))
        material = identities[:, ::-1]*5.
        # Independent analytic camera. These coefficients do not come from the
        # fitter, renderer or a fitted alignment of the evaluation queries.
        camera = np.array([[2.2, .4, 31.], [-.15, 2.7, 26.], [.0015, -.001, 1.]])

        def project(points):
            homogeneous = np.column_stack((points, np.ones(len(points))))@camera.T
            return homogeneous[:, :2]/homogeneous[:, 2:]

        truth = project(material)
        rng = np.random.default_rng(7919)
        observed = truth+rng.normal(0, .7, truth.shape)
        outliers = rng.choice(len(truth), 10, replace=False)
        observed[outliers] += rng.normal(0, 3, (10, 2))
        scores = rng.uniform(.8, 1., len(identities))
        scores[outliers] = rng.uniform(.2, .4, len(outliers))
        query_mm = rng.uniform([5., 5.], [50., 40.], (1500, 2))
        query = project(query_mm)
        cells = [[r, c] for r in range(9) for c in range(11)]
        return identities, material, observed, scores, query, query_mm, cells

    def test_refinement_improves_independent_physical_queries_with_full_support(self):
        ids, material, observed, scores, query, truth, cells = self.fixture()
        before = forward_cell_map(observed, material, ids, query, cells)
        refined, report = refine_grid_points(observed, ids, scores)
        after = forward_cell_map(refined, material, ids, query, cells)
        self.assertTrue(np.isfinite(before).all())
        self.assertTrue(np.isfinite(after).all())
        old_error = np.linalg.norm(before-truth, axis=1)
        new_error = np.linalg.norm(after-truth, axis=1)
        self.assertLess(np.sqrt(np.mean(new_error**2)), .8*np.sqrt(np.mean(old_error**2)))
        self.assertLess(np.quantile(new_error, .95), np.quantile(old_error, .95))
        self.assertGreater((new_error <= .5).mean(), (old_error <= .5).mean())
        self.assertFalse(report['gold_used'])
        self.assertFalse(report['physical_accuracy_certified'])

    def test_origin_gauge_and_point_permutation_do_not_change_refined_geometry(self):
        ids, _, observed, scores, _, _, _ = self.fixture()
        saved_xy, saved_ids, saved_scores = observed.copy(), ids.copy(), scores.copy()
        refined, _ = refine_grid_points(observed, ids, scores)
        shifted, _ = refine_grid_points(observed, ids+[10000, -20000], scores)
        order = np.random.default_rng(8191).permutation(len(ids))
        shuffled, _ = refine_grid_points(observed[order], ids[order], scores[order])
        np.testing.assert_allclose(refined, shifted, atol=1e-9, rtol=0)
        np.testing.assert_allclose(refined, shuffled[np.argsort(order)], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(observed, saved_xy)
        np.testing.assert_array_equal(ids, saved_ids)
        np.testing.assert_array_equal(scores, saved_scores)


if __name__ == '__main__':
    unittest.main()
