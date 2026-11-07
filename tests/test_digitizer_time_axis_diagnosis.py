"""Independent delayed-signal and missing-data controls for diagnosis only."""

import unittest

import numpy as np

from examples.audit_digitizer_time_axis import lag_diagnostic


class TimeAxisDiagnosisTests(unittest.TestCase):
    def signal(self):
        t = np.arange(2000) / 500
        return .2 * np.sin(2 * np.pi * (.8 * t + 2 * t * t)) + .4 * np.exp(-((t - 1.23) / .02) ** 2)

    def test_known_physical_delay_is_found_without_compressing_missing_intervals(self):
        x = self.signal()
        y = np.full_like(x, np.nan)
        y[75:] = x[:-75]
        y[500:520] = np.nan
        before = y.copy()
        result = lag_diagnostic(x, y, 500)
        self.assertEqual(result["predicted_delay_samples"], 75)
        self.assertEqual(result["predicted_delay_s"], .15)
        self.assertAlmostEqual(result["best_pcc"], 1, places=10)
        self.assertEqual(result["raw_rmse_mv_at_best_lag"], 0)
        np.testing.assert_array_equal(y, before)
        self.assertFalse(result["predictions_changed"])

    def test_a_circularly_shifted_fixture_does_not_gain_wrapped_samples(self):
        x = self.signal()
        result = lag_diagnostic(x, np.roll(x, 75), 500)
        self.assertEqual(result["predicted_delay_samples"], 75)
        self.assertEqual(result["samples_at_best_lag"], len(x) - 75)
        self.assertFalse(result["circular_shift_used"])

    def test_empty_and_constant_predictions_do_not_become_successes(self):
        x = self.signal()
        missing = lag_diagnostic(x, np.full_like(x, np.nan), 500)
        self.assertFalse(missing["eligible"])
        flat = lag_diagnostic(x, np.zeros_like(x), 500)
        self.assertFalse(flat["eligible"])
        self.assertEqual(flat["reason"], "insufficient_shared_variation")


if __name__ == "__main__":
    unittest.main()
