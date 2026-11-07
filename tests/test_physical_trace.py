import unittest

import numpy as np

from ecg_gpu.digitization.physical_trace import sample_physical_trace


class PhysicalTraceTests(unittest.TestCase):
    def kwargs(self):
        return dict(baseline_y_px=[1000.], x_origin_px=17., pixels_per_mm_x=20.,
                    pixels_per_mm_y=20., paper_speed_mm_s=25., gain_mm_mv=10.,
                    fs=500., samples=5000, max_gap_s=.0021)

    def test_occlusion_never_moves_physical_peak(self):
        t = np.arange(5000) / 500
        raw = (1000 - 100 * np.exp(-((t - 4.) / .02) ** 2))[None]
        for missing in [0, 100, 200, 500]:
            y = raw.copy()
            y[:, :missing] = np.nan
            values, receipt = sample_physical_trace(y, 17 + np.arange(5000), **self.kwargs())
            self.assertEqual(np.nanargmax(values[0]) / 500, 4.)
            self.assertEqual(int(np.isnan(values).sum()), missing)
            self.assertAlmostEqual(values[0, 2000], .5)
            self.assertFalse(receipt['time_fitted_to_visible_extent'])

    def test_axis_specific_calibration_and_explicit_baseline(self):
        kw = self.kwargs()
        kw.update(pixels_per_mm_x=10., paper_speed_mm_s=50., gain_mm_mv=5., samples=3)
        values, _ = sample_physical_trace([[950., 900., 1000.]], [17, 18, 19], **kw)
        np.testing.assert_allclose(values, [[.5, 1., 0.]])

    def test_no_bridging_missing_columns_or_unobserved_coordinate_gaps(self):
        kw = self.kwargs()
        kw.update(samples=8, fs=1000.)
        values, _ = sample_physical_trace([[980., np.nan, 940., 920.]], [17, 18, 19, 22], **kw)
        self.assertAlmostEqual(values[0, 0], .1)
        self.assertTrue(np.isnan(values[0, 1:4]).all())
        self.assertAlmostEqual(values[0, 4], .3)
        self.assertTrue(np.isnan(values[0, 5:]).all())

    def test_output_time_origin_no_edge_extrapolation(self):
        kw = self.kwargs()
        kw.update(samples=5, start_s=-.002)
        values, _ = sample_physical_trace([[980., 960.]], [17, 18], **kw)
        np.testing.assert_allclose(values, [[np.nan, .1, .2, np.nan, np.nan]], equal_nan=True)

    def test_invalid_physical_parameters(self):
        for name in ['pixels_per_mm_x', 'pixels_per_mm_y', 'paper_speed_mm_s', 'gain_mm_mv', 'fs', 'max_gap_s']:
            for value in [0, -1, np.inf, np.nan, True]:
                kw = self.kwargs()
                kw[name] = value
                with self.assertRaises(ValueError):
                    sample_physical_trace([[1., 2.]], [17, 18], **kw)
        with self.assertRaises(ValueError):
            sample_physical_trace([[1., 2.]], [18, 17], **self.kwargs())


if __name__ == '__main__':
    unittest.main()
