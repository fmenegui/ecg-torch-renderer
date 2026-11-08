"""Independent physical calibration and missingness for arbitrary paper regions."""
from dataclasses import replace
import unittest

import numpy as np

from ecg_gpu.digitization import TraceCalibration, TraceObservation, decode_trace_region, resample_trace


class RegionDecoding(unittest.TestCase):
    def setUp(self):
        self.calibration = TraceCalibration('short-aVR', 'aVR', 4., 8., 25., 10., 30., 2.5)
        y = np.arange(60)[:, None]
        self.center = 20.25+np.linspace(0, .4, 17)
        self.probability = np.clip(.9-.035*(y-self.center)**2, 0, 1)

    def test_subpixel_voltage_and_time_use_separate_physical_axes(self):
        result = decode_trace_region(self.probability, self.calibration)
        np.testing.assert_allclose(result.values_mv, (30-self.center)/80, atol=1e-12)
        self.assertEqual(result.fs, 100.)
        self.assertEqual(result.start_s, 2.5)
        self.assertEqual(result.lead, 'aVR')
        self.assertTrue(result.observed.all())

    def test_gain_speed_and_cabrera_polarity_are_independent(self):
        normal = decode_trace_region(self.probability, self.calibration)
        low_gain = decode_trace_region(self.probability, replace(self.calibration, gain_mm_per_mv=5))
        fast = decode_trace_region(self.probability, replace(self.calibration, speed_mm_per_s=50))
        inverted = decode_trace_region(self.probability, replace(self.calibration, polarity=-1))
        np.testing.assert_allclose(low_gain.values_mv, 2*normal.values_mv)
        np.testing.assert_array_equal(fast.values_mv, normal.values_mv)
        self.assertEqual(fast.fs, 2*normal.fs)
        np.testing.assert_array_equal(inverted.values_mv, -normal.values_mv)

    def test_repeated_leads_retain_distinct_instances_and_times(self):
        a = decode_trace_region(self.probability, replace(self.calibration, lead='II', instance_id='short-II', start_s=0.))
        b = decode_trace_region(self.probability, replace(self.calibration, lead='II', instance_id='rhythm-II', start_s=5.))
        self.assertNotEqual(a.instance_id, b.instance_id)
        self.assertNotEqual(a.start_s, b.start_s)
        self.assertEqual(a.lead, b.lead)

    def test_weak_boundary_and_ambiguous_columns_remain_missing(self):
        p = self.probability.copy()
        p[:, 2] = 0.
        p[:, 4] = 0.
        p[0, 4] = .9
        p[45, 6] = .99
        p[21, 6] = .99
        result = decode_trace_region(p, self.calibration)
        self.assertFalse(result.observed[[2, 4, 6]].any())
        self.assertTrue(np.isnan(result.values_mv[[2, 4, 6]]).all())
        self.assertTrue(result.observed[[0, 1, 3, 5]].all())

    def test_resampling_never_bridges_missing_intervals_or_extrapolates(self):
        source = TraceObservation('one', 'V1', np.array([0., 1., np.nan, np.nan, 4., 5.]),
            np.array([True, True, False, False, True, True]), np.ones(6), 2., 7.5)
        result = resample_trace(source, 4.)
        self.assertEqual(result.start_s, 7.5)
        np.testing.assert_allclose(result.values_mv[[0, 1, 2, 8, 9, 10]], [0., .5, 1., 4., 4.5, 5.])
        self.assertTrue(np.isnan(result.values_mv[[3, 4, 5, 6, 7, 11]]).all())
        self.assertFalse(result.values_mv.flags.writeable)

    def test_empty_region_and_single_column_do_not_invent_signal(self):
        empty = decode_trace_region(np.zeros((6, 3)), self.calibration)
        self.assertFalse(resample_trace(empty, 500).observed.any())
        isolated = TraceObservation('one', 'II', [np.nan, .5, np.nan],
            np.array([False, True, False]), [0., .9, 0.], 2., 0.)
        result = resample_trace(isolated, 4.)
        self.assertEqual(int(result.observed.sum()), 1)
        self.assertEqual(result.values_mv[2], .5)

    def test_inputs_are_immutable_and_invalid_calibration_is_rejected(self):
        for fields in ({'pixels_per_mm_x': 0}, {'gain_mm_per_mv': True}, {'speed_mm_per_s': np.nan},
                       {'lead': 'X'}, {'polarity': True}, {'start_s': -1}, {'instance_id': ''}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.calibration, **fields)
        result = decode_trace_region(self.probability, self.calibration)
        self.probability[:] = 0
        self.assertTrue(result.observed.all())
        self.assertFalse(result.values_mv.flags.writeable)
        with self.assertRaises(ValueError):
            TraceObservation('one', 'I', [np.nan], np.array([True]), [.9], 500, 0.)


if __name__ == '__main__':
    unittest.main()
