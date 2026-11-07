"""Prevent incomplete fixed-layout outputs from masquerading as generalization."""
import unittest

import numpy as np

from ecg_gpu.digitization.signals import LEADS, LeadSignal
from examples.digitization_layout_probe import evaluate_overlap, normalize_known_rows, support_mask


class LayoutProbeEvaluation(unittest.TestCase):
    def setUp(self):
        self.fs = 20
        self.t = np.arange(200)/self.fs
        self.values = np.array([(i+1)+.1*self.t for i in range(12)])

    def metadata(self, duration, *, rhythm=False):
        lead_ids = list(range(12))
        starts = [0.]*12
        ends = [duration]*12
        if rhythm:
            lead_ids.append(1)
            starts.append(0.)
            ends.append(10.)
        return {'active': [[True]*len(lead_ids)], 'lead_id': [lead_ids],
                'start_s': [starts], 'end_s': [ends]}

    def predictions(self, durations):
        return {name: LeadSignal(self.values[i, :round(duration*self.fs)], self.fs)
                for i, (name, duration) in enumerate(zip(LEADS, durations))}

    def test_full_page_short_outputs_have_low_coverage_even_when_exact(self):
        pred = self.predictions([2.5 if name != 'II' else 10 for name in LEADS])
        score = evaluate_overlap(self.values, self.fs, self.metadata(10.), pred)
        self.assertAlmostEqual(score['printed_time_coverage'], 37.5/120)
        self.assertEqual(score['unsupported_output_fraction'], 0.)
        self.assertEqual(score['raw_overlap_rmse_mv'], 0.)

    def test_unprinted_rhythm_output_is_explicit(self):
        pred = self.predictions([2.5 if name != 'II' else 10 for name in LEADS])
        score = evaluate_overlap(self.values, self.fs, self.metadata(2.5), pred)
        self.assertEqual(score['printed_time_coverage'], 1.)
        self.assertAlmostEqual(score['unsupported_output_fraction'], 7.5/37.5)
        self.assertEqual(score['per_lead']['II']['unsupported_output_samples'], 150)

    def test_repeated_ii_does_not_duplicate_coverage_denominator(self):
        pred = self.predictions([2.5 if name != 'II' else 10 for name in LEADS])
        score = evaluate_overlap(self.values, self.fs, self.metadata(2.5, rhythm=True), pred)
        self.assertEqual(score['printed_time_coverage'], 1.)
        self.assertEqual(score['unsupported_output_fraction'], 0.)
        self.assertEqual(score['per_lead']['II']['printed_samples'], 200)

    def test_time_and_dc_errors_remain_visible(self):
        pred = self.predictions([10.]*12)
        for i, name in enumerate(LEADS):
            pred[name] = LeadSignal(self.values[i, :50]+.7, self.fs, start_s=5.)
        score = evaluate_overlap(self.values, self.fs, self.metadata(10.), pred)
        # The ramp rises .5 mV between 0 and 5 s, leaving an uncorrected .2 error.
        self.assertAlmostEqual(score['raw_overlap_rmse_mv'], .2)
        self.assertAlmostEqual(score['printed_time_coverage'], .25)
        self.assertAlmostEqual(score['per_lead']['V6']['raw_bias_mv'], .2)

    def test_windows_are_half_open_and_unions_preserve_gaps(self):
        times = np.arange(9)/2
        actual = support_mask(times, [(0., 1.), (2., 3.), (.5, 1.5)])
        np.testing.assert_array_equal(actual, [True, True, True, False, True, True, False, False, False])

    def test_oracle_affine_map_preserves_independent_coordinate_ramps(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('Optional digitization dependency cv2 unavailable')
        y, x = np.mgrid[:160, :200]
        rgb = np.stack([x, y, np.zeros_like(x)], -1).astype(np.uint8)
        rows = [0]*4+[1]*4+[2]*4+[3]
        metadata = {'baseline': [[20 if r == 0 else 20+40*r for r in rows]],
            'row': [rows], 'active': [[True]*13], 'x0': [[10.]*13],
            'x1': [[190.]*13], 'canonical_millivolts_per_pixel': [1/79]}
        normalized = normalize_known_rows(rgb, metadata, cv2)
        for r, target_y in enumerate((704, 988, 1272, 1532)):
            # The desired baseline maps to source y=20+40*r at a half pixel.
            np.testing.assert_allclose(normalized[target_y, 236, :2], [10, 20.5+40*r], atol=.6)
            np.testing.assert_allclose(normalized[target_y, 4160, :2], [190, 20.5+40*r], atol=.6)
        metadata['active'][0][-1] = False
        with self.assertRaises(ValueError):
            normalize_known_rows(rgb, metadata, cv2)


if __name__ == '__main__':
    unittest.main()
