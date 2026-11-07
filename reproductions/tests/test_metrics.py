import math
import unittest

import numpy as np

from reproductions.compare import compare
from reproductions.metrics import patient_macro, score_lead, shifted


class MetricsTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(1000) / 1000
        self.y = np.sin(2 * np.pi * 7.31 * t) + 0.1 * np.cos(2 * np.pi * 39 * t)

    def test_perfect_is_perfect(self):
        result = score_lead(self.y, self.y)
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["observed"]["rmse_mv"], 0)
        self.assertTrue(math.isinf(result["observed"]["snr_db"]))
        self.assertEqual(result["observed"]["snr_status"], "perfect")

    def test_entirely_missing_retains_denominator(self):
        result = score_lead(self.y, np.full_like(self.y, np.nan))
        self.assertEqual(result["coverage"], 0)
        self.assertTrue(result["entirely_missing"])
        self.assertEqual(result["all_printed_zero_scoring_only"]["snr_db"], 0)
        self.assertIsNone(result["observed"]["rmse_mv"])
        self.assertIsNone(result["aligned"])

    def test_shift_does_not_wrap_or_shrink_denominator(self):
        p = shifted(self.y, 10)
        result = score_lead(self.y, p, fs=1000, align_ms=20)
        self.assertEqual(result["returned_samples"], 990)
        self.assertEqual(result["aligned"]["lag_samples"], -10)
        self.assertEqual(result["aligned"]["coverage_on_original_printed"], 0.99)
        self.assertEqual(result["aligned"]["rmse_mv"], 0)

    def test_holes_not_rewarded_as_finite_predictions(self):
        p = self.y.copy()
        p[10:150] = np.nan
        result = score_lead(self.y, p, align_ms=5)
        self.assertAlmostEqual(result["coverage"], 0.86)
        self.assertIsNone(result["aligned"])
        self.assertGreater(result["all_printed_zero_scoring_only"]["rmse_mv"], 0)

    def test_dc_alignment_does_not_fit_gain(self):
        result = score_lead(self.y, 2 * self.y + 13, center=True)
        self.assertGreater(result["aligned"]["rmse_mv"], 0.5)

    def test_only_printed_intervals_count(self):
        printed = np.zeros(1000, bool)
        printed[100:350] = True
        p = np.full(1000, np.nan)
        p[100:350] = self.y[100:350]
        result = score_lead(self.y, p, printed)
        self.assertEqual(result["printed_samples"], 250)
        self.assertEqual(result["coverage"], 1)

    def test_bad_reference_is_rejected(self):
        y = self.y.copy()
        y[0] = np.inf
        with self.assertRaises(ValueError):
            score_lead(y, self.y)

    def test_grouped_aggregation_uses_patient_units(self):
        result = patient_macro([{"patient_id": "a", "coverage": 0}, {"patient_id": "a", "coverage": 0}, {"patient_id": "b", "coverage": 1}], "coverage")
        self.assertEqual(result["mean"], 0.5)

    def test_missing_metric_is_visible(self):
        result = patient_macro([{"patient_id": "a", "rmse": None}, {"patient_id": "b", "rmse": 1}], "rmse")
        self.assertEqual(result["patients_scheduled"], 2)
        self.assertEqual(result["patients_with_defined_metric"], 1)

    def test_different_cohort_cannot_reproduce_a_headline(self):
        expected = {"id": "paper", "method_profile": "x", "dataset": "ahus", "cohort": "266", "metric_protocol": "native", "aggregation": "lead_mean", "unit": "mV", "metrics": {"snr_db": 19.65}}
        observed = {**expected, "dataset": "pm", "cohort": "8", "metrics": {"snr_db": 25}, "exact_cohort_verified": True, "all_scheduled_outcomes_retained": True, "execution_status": "completed"}
        result = compare(expected, observed)
        self.assertFalse(result["directly_comparable"])
        self.assertEqual(result["delta_only_if_matched"], {})
        self.assertFalse(result["reproduced"])


if __name__ == "__main__":
    unittest.main()
