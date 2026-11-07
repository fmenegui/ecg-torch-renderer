import unittest

import numpy as np

try:
    import scipy  # noqa: F401
except ImportError:
    scipy = None

from ecg_gpu.digitization.robust_refinement import refine_grid_points


@unittest.skipIf(scipy is None, 'Optional SciPy dependency unavailable')
class RobustRefinementTests(unittest.TestCase):
    @staticmethod
    def fixture(rows=16, columns=24):
        row, col = np.mgrid[:rows, :columns]
        ids = np.column_stack((row.ravel(), col.ravel()))
        uv = ids[:, ::-1].astype(float)
        # Independent physical-control projection. Neither the fitter nor the
        # regularizer supplies the truth used by these assertions.
        homogeneous = np.column_stack((uv, np.ones(len(ids))))
        camera = np.array([[12., 1.2, 20.], [.7, 11., 25.], [.0008, .0012, 1.]])
        projected = homogeneous@camera.T
        return projected[:, :2]/projected[:, 2:], ids

    def test_clean_projective_grid_is_exactly_unchanged(self):
        truth, ids = self.fixture()
        original, original_ids = truth.copy(), ids.copy()
        refined, report = refine_grid_points(truth, ids)
        np.testing.assert_array_equal(refined, original)
        np.testing.assert_array_equal(truth, original)
        np.testing.assert_array_equal(ids, original_ids)
        self.assertEqual(report['reason'], 'projective observations already consistent')
        self.assertFalse(report['physical_accuracy_certified'])

    def test_noise_and_location_outliers_improve_against_independent_truth(self):
        truth, ids = self.fixture()
        rng = np.random.default_rng(20261022)
        observed = truth+rng.normal(0, .7, truth.shape)
        outliers = rng.choice(len(ids), len(ids)//20, replace=False)
        observed[outliers] += rng.normal(0, 3., (len(outliers), 2))
        scores = np.ones(len(ids))
        scores[outliers] = .2
        refined, report = refine_grid_points(observed, ids, scores)
        raw_error = np.sqrt(np.mean((observed-truth)**2))
        new_error = np.sqrt(np.mean((refined-truth)**2))
        self.assertLess(new_error, .8*raw_error)
        self.assertEqual(report['status'], 'refined')
        self.assertLessEqual(report['maximum_shift_fraction'], .2+1e-12)
        self.assertGreater(report['robust_downweighted_points'], 0)
        self.assertGreater(report['cells_guarded'], 0)

    def test_confidence_ranking_changes_a_noisy_fit(self):
        truth, ids = self.fixture()
        rng = np.random.default_rng(901)
        observed = truth+rng.normal(0, .4, truth.shape)
        corrupted = np.arange(14, len(ids), 29)
        observed[corrupted] += [2.4, -2.3]
        scores = np.ones(len(ids))
        scores[corrupted] = 0.
        uniform, _ = refine_grid_points(observed, ids)
        ranked, report = refine_grid_points(observed, ids, scores)
        self.assertGreater(np.max(np.abs(ranked-uniform)), 1e-4)
        self.assertLess(np.mean((ranked-truth)**2), np.mean((uniform-truth)**2))
        self.assertEqual(report['score_weight_min'], .25)
        self.assertEqual(report['score_weight_max'], 1.)

    def test_order_and_integer_origin_do_not_change_refinement(self):
        truth, ids = self.fixture()
        observed = truth+np.random.default_rng(34).normal(0, .4, truth.shape)
        order = np.random.default_rng(71).permutation(len(ids))
        a, _ = refine_grid_points(observed, ids)
        b, _ = refine_grid_points(observed[order], ids[order]+[700, -900])
        np.testing.assert_allclose(b, a[order], rtol=0, atol=1e-8)

    def test_missing_adjacent_ids_do_not_create_interpolated_nodes(self):
        truth, ids = self.fixture()
        refined, report = refine_grid_points(truth, ids*3)
        np.testing.assert_array_equal(refined, truth)
        self.assertEqual(report['constraints'], 0)
        self.assertEqual(report['reason'], 'insufficient adjacent-ID constraints')
        self.assertEqual(len(refined), len(ids))

    def test_disabled_budgeted_and_short_inputs_are_explicit(self):
        truth, ids = self.fixture()
        for kw in ({'strength': 0}, {'max_shift_fraction': 0}, {'max_points': 20}):
            refined, report = refine_grid_points(truth, ids, **kw)
            np.testing.assert_array_equal(refined, truth)
            self.assertEqual(report['status'], 'unchanged')
        self.assertTrue(report['budget_exhausted'])
        refined, report = refine_grid_points(truth[:8], ids[:8])
        np.testing.assert_array_equal(refined, truth[:8])
        self.assertEqual(report['reason'], 'insufficient 2D lattice support')
        empty, report = refine_grid_points(np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(report['status'], 'unchanged')

    def test_iteration_exhaustion_is_not_reported_as_convergence(self):
        truth, ids = self.fixture()
        observed = truth+np.random.default_rng(18).normal(0, .6, truth.shape)
        _, report = refine_grid_points(observed, ids, iterations=1)
        self.assertEqual(report['iterations'], 1)
        self.assertFalse(report['converged'])
        self.assertTrue(report['iteration_limit_reached'])
        self.assertTrue(report['budget_exhausted'])

    def test_optional_guard_abstains_on_coherent_crease_without_gold(self):
        row, col = np.mgrid[:10, :12]
        ids = np.column_stack((row.ravel(), col.ravel()))
        uv = ids[:, ::-1]*5.
        # Piecewise-affine camera map independent of the refinement prior.
        def camera(q):
            u, v = q.T
            return np.column_stack((2.5*u+30, 2.5*v+20+6*np.maximum(u-25, 0)/5))

        observed = camera(uv)
        unguarded, _ = refine_grid_points(observed, ids)
        guarded, report = refine_grid_points(observed, ids, crease_guard=True)
        self.assertGreater(np.max(np.linalg.norm(unguarded-observed, axis=1)), 1.)
        np.testing.assert_array_equal(guarded, observed)
        self.assertEqual(report['status'], 'unchanged')
        self.assertTrue(report['crease_guard_triggered'])
        self.assertFalse(report['curvature_guard']['true_crease_certified'])
        # A coherent detector error would also trigger. The guard certifies
        # conflict with this prior, not that a physical fold truly exists.

    def test_optional_guard_retains_noise_refinement_when_no_consensus(self):
        truth, ids = self.fixture()
        observed = truth+np.random.default_rng(20261022).normal(0, .7, truth.shape)
        unguarded, _ = refine_grid_points(observed, ids)
        guarded, report = refine_grid_points(observed, ids, crease_guard=True)
        self.assertFalse(report['crease_guard_triggered'])
        np.testing.assert_array_equal(guarded, unguarded)
        self.assertLess(np.mean((guarded-truth)**2), np.mean((observed-truth)**2))

    def test_invalid_inputs_fail_without_an_estimate(self):
        truth, ids = self.fixture()
        for points, identities, scores, kw in (
                (truth[:, 0], ids, None, {}),
                (truth, ids[:-1], None, {}),
                (truth*np.nan, ids, None, {}),
                (truth, ids+.2, None, {}),
                (truth, np.zeros_like(ids), None, {}),
                (truth, ids, np.ones(len(ids)-1), {}),
                (truth, ids, np.full(len(ids), np.inf), {}),
                (truth, ids, None, {'strength': -1}),
                (truth, ids, None, {'iterations': 0}),
                (truth, ids, None, {'iterations': 2.5}),
                (truth, ids, None, {'max_shift_fraction': .6}),
                (truth, ids, None, {'max_points': 3}),
                (truth, ids, None, {'crease_guard': 1})):
            with self.subTest(parameters=kw):
                with self.assertRaises(ValueError):
                    refine_grid_points(points, identities, scores, **kw)


if __name__ == '__main__':
    unittest.main()
