"""Independent subpixel matching and patient-paired study contracts."""
import importlib.util
import unittest

import numpy as np

from ecg_gpu.grid_evaluation import evaluate_grid_tolerances
from examples.grid_sampling_study import aggregate


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'Install the benchmark extra')
class GridToleranceTests(unittest.TestCase):
    def test_reassigns_at_each_radius_instead_of_filtering_wider_matches(self):
        # Wide optimum needs both predictions. At .45 only the first can match
        # the first reference, an edge absent from the wide assignment.
        result = evaluate_grid_tolerances([[.4, 0], [-.6, 0]], [[0, 0], [.9, 0]], radii=(.8, .45))
        self.assertEqual([r['radius_px'] for r in result], [.45, .8])
        self.assertEqual([r['tp'] for r in result], [1, 2])
        self.assertAlmostEqual(result[0]['mean_dx_px'], .4)

    def test_bias_and_euclidean_error_have_independent_analytic_reference(self):
        result = evaluate_grid_tolerances([[.3, .4], [10.3, 10.4]], [[0, 0], [10, 10]])
        self.assertEqual(result[0]['tp'], 0)
        self.assertIsNone(result[0]['mean_dx_px'])
        self.assertIsNone(result[0]['rmse_error_px'])
        score = result[-1]
        self.assertEqual(score['f1'], 1.)
        self.assertAlmostEqual(score['mean_dx_px'], .3)
        self.assertAlmostEqual(score['mean_dy_px'], .4)
        self.assertAlmostEqual(score['rmse_error_px'], .5)
        self.assertAlmostEqual(score['p95_error_px'], .5)

    def test_empty_sets_and_bad_tolerances(self):
        empty = np.empty((0, 2))
        result = evaluate_grid_tolerances(empty, [[0, 0]])
        self.assertTrue(all(r['fn'] == 1 and r['f1'] == 0 for r in result))
        self.assertIsNone(evaluate_grid_tolerances(empty, empty)[0]['f1'])
        for radii in ((), (.5, .5), (0,), (-1,), (True,), (float('nan'),), (float('inf'),)):
            with self.subTest(radii=radii), self.assertRaises(ValueError):
                evaluate_grid_tolerances(empty, empty, radii=radii)

    def test_identical_predictions_have_exactly_zero_paired_difference(self):
        cases = [{'model': 'fixed', 'key': f'{g}-{v}', 'group_id': g, 'spp': spp,
                  'predicted': np.array([[.1, 0.]]), 'reference': np.array([[0., 0.]])}
                 for g in ('a', 'b') for v in (0, 1) for spp in (16, 128)]
        result = aggregate(cases, [.25, 1.5], repeats=40)
        self.assertEqual(result['bootstrap']['groups'], 2)
        self.assertTrue(all(r['delta_f1'] == 0 and r['patient_bootstrap_95'] == [0., 0.]
                            for r in result['paired_differences']))
        self.assertTrue(all(r['images'] == 4 for r in result['by_model_spp_tolerance']))

    def test_missing_patient_is_rejected(self):
        cases = [{'model': 'fixed', 'key': g, 'group_id': g, 'spp': spp,
                  'predicted': np.array([[0., 0.]]), 'reference': np.array([[0., 0.]])}
                 for g, spp in (('a', 16), ('b', 16), ('a', 128))]
        with self.assertRaisesRegex(ValueError, 'Incomplete patient'):
            aggregate(cases, [.5], repeats=10)

    def test_missing_view_and_duplicate_view_are_rejected(self):
        cases = [{'model': 'fixed', 'key': f'{g}-{v}', 'group_id': g, 'spp': spp,
                  'predicted': np.array([[0., 0.]]), 'reference': np.array([[0., 0.]])}
                 for g in ('a', 'b') for v in (0, 1) for spp in (16, 128)]
        for malformed in (cases[:-1], cases+[cases[0]]):
            with self.assertRaisesRegex(ValueError, 'Incomplete patient'):
                aggregate(malformed, [.5], repeats=10)


if __name__ == '__main__':
    unittest.main()
