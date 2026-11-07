"""Missingness and physical timing contracts for the published-model probe."""
import unittest

import numpy as np

from examples.digitization_npj_probe import evaluate_missing


class MissingAwareEvaluation(unittest.TestCase):
    def setUp(self):
        self.fs = 100
        t = np.arange(1000)/self.fs
        self.truth = np.stack([np.sin(t*(i+2))+.2*np.cos(t*t) for i in range(12)])
        self.metadata = {'active': [[True]*12], 'lead_id': [list(range(12))],
                         'start_s': [[0.]*12], 'end_s': [[10.]*12]}

    def test_missing_output_retains_denominator_and_is_not_filled(self):
        prediction = self.truth.copy()
        prediction[6:] = np.nan
        before = prediction.copy()
        result = evaluate_missing(self.truth, prediction, self.fs, self.metadata)
        self.assertEqual(result['printed_time_coverage'], .5)
        self.assertEqual(result['leads_with_aligned_score'], 6)
        self.assertEqual(result['raw_overlap_rmse_mv'], 0.)
        self.assertIsNone(result['per_lead']['V6']['aligned'])
        np.testing.assert_array_equal(prediction, before)

    def test_entirely_missing_prediction_has_no_numeric_quality_score(self):
        result = evaluate_missing(self.truth, np.full_like(self.truth, np.nan), self.fs, self.metadata)
        self.assertEqual(result['printed_time_coverage'], 0.)
        self.assertEqual(result['leads_with_aligned_score'], 0)
        self.assertIsNone(result['mean_observed_lead_snr_db'])
        self.assertIsNone(result['raw_overlap_rmse_mv'])

    def test_dc_alignment_does_not_hide_raw_bias(self):
        result = evaluate_missing(self.truth, self.truth+.7, self.fs, self.metadata)
        self.assertAlmostEqual(result['raw_overlap_rmse_mv'], .7)
        self.assertGreater(result['mean_observed_lead_snr_db'], 100)
        self.assertAlmostEqual(result['per_lead']['I']['raw_bias_mv'], .7)

    def test_known_delay_is_found_without_erasing_raw_error(self):
        prediction = np.full_like(self.truth, np.nan)
        prediction[:, 5:] = self.truth[:, :-5]
        result = evaluate_missing(self.truth, prediction, self.fs, self.metadata)
        self.assertAlmostEqual(result['printed_time_coverage'], .995)
        self.assertGreater(result['raw_overlap_rmse_mv'], .1)
        self.assertEqual(result['per_lead']['I']['aligned']['shift_ms'], -50.)

    def test_unprinted_outputs_are_reported_separately(self):
        self.metadata['end_s'] = [[1.]*12]
        prediction = self.truth.copy()
        prediction[:, 100:] += 1000.
        result = evaluate_missing(self.truth, prediction, self.fs, self.metadata)
        self.assertEqual(result['printed_time_coverage'], 1.)
        self.assertAlmostEqual(result['unsupported_output_fraction'], .9)
        self.assertEqual(result['raw_overlap_rmse_mv'], 0.)

    def test_infinite_or_mismatched_output_is_invalid(self):
        with self.assertRaises(ValueError):
            evaluate_missing(self.truth, np.full_like(self.truth, np.inf), self.fs, self.metadata)
        with self.assertRaises(ValueError):
            evaluate_missing(self.truth, self.truth[:, :-1], self.fs, self.metadata)


if __name__ == '__main__':
    unittest.main()
