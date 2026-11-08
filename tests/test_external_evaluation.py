import unittest

import numpy as np

from ecg_gpu.digitization.external_evaluation import score_external_signals


class ExternalSignalScoresTest(unittest.TestCase):
    def test_missing_samples_remain_in_denominator(self):
        a = np.tile([1.0, -1.0, 1.0, -1.0], (12, 1))
        b = a.copy()
        b[:, 2:] = np.nan
        result = score_external_signals(a, b, 100)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["raw_observed"]["rmse_mv"], 0)
        self.assertAlmostEqual(
            result["raw_all_printed_zero_missing"]["rmse_mv"], np.sqrt(0.5)
        )
        self.assertAlmostEqual(
            result["raw_all_printed_zero_missing"]["snr_db"], 10 * np.log10(2)
        )
        self.assertIsNone(result["per_lead"]["I"]["aligned_observed_dc_centered"])

    def test_total_failure_is_zero_db_and_no_observed_score(self):
        a = np.tile([1.0, -1.0, 1.0, -1.0], (12, 1))
        result = score_external_signals(a, np.full_like(a, np.nan), 100)
        self.assertEqual(result["raw_all_printed_zero_missing"]["snr_db"], 0)
        self.assertIsNone(result["raw_observed"])
        self.assertEqual(result["leads_entirely_missing"], 12)

    def test_wrong_polarity_is_not_repaired(self):
        a = np.tile([1.0, -1.0, 1.0, -1.0], (12, 1))
        result = score_external_signals(a, -a, 100, max_shift_s=0)
        raw = result["raw_all_printed_zero_missing"]
        self.assertEqual(raw["pcc"], -1)
        self.assertEqual(raw["rmse_mv"], 2)
        self.assertAlmostEqual(raw["snr_db"], -10 * np.log10(4))

    def test_printed_windows_control_denominator_only(self):
        a = np.full((12, 20), np.nan)
        a[0, 5:15] = np.arange(10)
        b = np.zeros_like(a)
        b[0, 5:15] = a[0, 5:15]
        result = score_external_signals(a, b, 100)
        self.assertEqual(result["printed_samples"], 10)
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["unsupported_output_samples"], 230)
        self.assertEqual(result["leads_expected"], 1)

    def test_time_and_dc_alignment_do_not_change_raw_score(self):
        a = np.full((12, 100), np.nan)
        a[0] = np.sin(np.arange(100) * 0.17) + np.cos(np.arange(100) * 0.031)
        b = np.full_like(a, np.nan)
        b[0, 3:] = a[0, :-3] + 2
        result = score_external_signals(a, b, 100)
        aligned = result["per_lead"]["I"]["aligned_observed_dc_centered"]
        self.assertEqual(aligned["shift_samples"], 3)
        self.assertLess(aligned["rmse_mv"], 1e-14)
        self.assertGreater(result["raw_observed"]["rmse_mv"], 1)
        self.assertEqual(result["coverage"], 0.97)

    def test_rejects_malformed_values(self):
        a = np.ones((12, 50))
        for b in (np.full_like(a, np.inf), np.ones((11, 50))):
            with self.assertRaises(ValueError):
                score_external_signals(a, b, 100)


if __name__ == "__main__":
    unittest.main()
