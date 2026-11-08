import unittest
from copy import deepcopy

import numpy as np

from ecg_gpu.digitization.calibration import (
    measure_step_pulse,
    parse_print_settings,
    resolve_document_calibration,
    sample_with_document_calibration,
)


def pulse_mask():
    mask = np.zeros((40, 50), bool)
    mask[30, 4:11] = True
    mask[10:31, 10] = True
    mask[10, 10:31] = True
    mask[10:31, 30] = True
    mask[30, 30:38] = True
    return mask


class DocumentCalibrationTests(unittest.TestCase):
    def evidence(self):
        identity = {'source_id': '0' * 64, 'frame_id': 'rectified-fixture'}
        return dict(identity, settings=dict(parse_print_settings('25 mm/s 10 mm/mV'), source_id=identity['source_id']),
            grid=dict(identity, pixels_per_mm_x=4., pixels_per_mm_y=2.),
            baseline=dict(identity, y_px=[100.], method='analytic_fixture'),
            time_anchor=dict(identity, x_px=217., time_s=2., method='analytic_fixture'))

    def test_observed_units_commas_inverse_gain_and_conflicts(self):
        parsed = parse_print_settings('500 Hz. 12,5 mm / sec. 1 mV/cm. 10 mm/mV')
        self.assertEqual(parsed['paper_speed_mm_s'], 12.5)
        self.assertEqual(parsed['gain_mm_mv'], 10.)
        self.assertEqual(parsed['ambiguous_fields'], [])
        conflict = parse_print_settings('25 mm/s 50 mm/s 10 mm/mV')
        self.assertIsNone(conflict['paper_speed_mm_s'])
        self.assertIn('paper_speed_mm_s', conflict['ambiguous_fields'])
        for text in ['25 10', '500 Hz', '-25 mm/s -10 mm/mV', '0 mm/s 0 mm/mV',
                     '25/50 mm/s 5/10 mm/mV', '25\nmm/s 10\nmm/mV']:
            parsed = parse_print_settings(text)
            self.assertIsNone(parsed['paper_speed_mm_s'])
            self.assertIsNone(parsed['gain_mm_mv'])
            self.assertFalse(parsed['default_values_used'])

    def test_pulse_supplies_baseline_and_scale_but_no_time_zero(self):
        result = measure_step_pulse(pulse_mask(), pixels_per_mm_x=4., pixels_per_mm_y=2.,
            pulse_amplitude_mv=1., pulse_duration_s=.2, label_source='independent analytical fixture')
        self.assertEqual(result['baseline_y_px'], 30.)
        self.assertEqual(result['gain_mm_mv'], 10.)
        self.assertEqual(result['paper_speed_mm_s'], 25.)
        self.assertFalse(result['waveform_time_zero_inferred'])

    def test_broken_clipped_or_sloped_pulse_is_rejected(self):
        original = pulse_mask()
        broken, clipped, diagonal = original.copy(), original.copy(), original.copy()
        broken[10, 20] = False
        clipped[0, 10] = True
        diagonal[10, 11:30] = False
        for x in range(11, 30):
            diagonal[10 + min(x - 10, 30 - x) // 2, x] = True
        for mask in [broken, clipped, diagonal]:
            with self.assertRaises(ValueError):
                measure_step_pulse(mask, pixels_per_mm_x=4., pixels_per_mm_y=2.,
                    pulse_amplitude_mv=1., pulse_duration_s=.2, label_source='fixture')

    def test_origin_and_baseline_cannot_be_derived_from_visible_trace(self):
        for name in ['time_anchor', 'baseline']:
            evidence = self.evidence()
            evidence[name] = None
            calibration = resolve_document_calibration(**evidence)
            self.assertEqual(calibration['status'], 'unresolved')
            with self.assertRaises(ValueError):
                sample_with_document_calibration([[100., 100.]], [17, 18], calibration,
                    fs=100, samples=2, max_gap_s=.011)
        for name, method in [('time_anchor', 'first_visible_column'), ('baseline', 'trace_mean')]:
            evidence = self.evidence()
            evidence[name]['method'] = method
            self.assertEqual(resolve_document_calibration(**evidence)['status'], 'unresolved')

    def test_frame_or_image_mismatch_is_rejected_including_after_resolution(self):
        for name in ['grid', 'baseline', 'time_anchor']:
            for field in ['frame_id', 'source_id']:
                evidence = self.evidence()
                evidence[name][field] = 'another-frame-or-image'
                with self.assertRaises(ValueError):
                    resolve_document_calibration(**evidence)
        calibration = resolve_document_calibration(**self.evidence())
        calibration['time_anchor']['frame_id'] = 'original-rgb'
        with self.assertRaises(ValueError):
            sample_with_document_calibration([[100., 100.]], [17, 18], calibration,
                fs=100, samples=2, max_gap_s=.011)
        evidence = self.evidence()
        evidence['settings']['source_id'] = '1' * 64
        with self.assertRaises(ValueError):
            resolve_document_calibration(**evidence)

    def test_st_offset_and_peak_time_survive_leading_occlusion(self):
        # Independent physical raster: 4 px/mm x 25 mm/s = 100 columns/s.
        # 0.2 mV maps to 4 vertical pixels at 2 px/mm and 10 mm/mV.
        x = np.arange(17, 1017)
        t = (x - 17) / 100
        y = 100 - 20 * (.2 + np.exp(-((t - 4.) / .03) ** 2))
        y[:100] = np.nan
        evidence = self.evidence()
        pulse = measure_step_pulse(pulse_mask(), pixels_per_mm_x=4., pixels_per_mm_y=2.,
            pulse_amplitude_mv=1., pulse_duration_s=.2, label_source='independent analytic label')
        # The independently specified pulse ROI is 70 px below this row's origin.
        evidence['baseline'].update(y_px=[pulse['baseline_y_px'] + 70.], method='pulse_foot', roi_origin_y_px=70.)
        calibration = resolve_document_calibration(**evidence)
        values, metadata = sample_with_document_calibration(y[None], x, calibration,
            fs=100, samples=1000, max_gap_s=.0101)
        self.assertEqual(np.nanargmax(values[0]) / 100, 4.)
        self.assertTrue(np.isnan(values[0, :100]).all())
        self.assertAlmostEqual(values[0, 200], .2)
        self.assertFalse(metadata['baseline_fitted_to_trace_mean'])
        self.assertFalse(metadata['origin_fitted_to_visible_trace'])

    def test_text_pulse_disagreement_is_unresolved(self):
        evidence = self.evidence()
        pulse = measure_step_pulse(pulse_mask(), pixels_per_mm_x=4., pixels_per_mm_y=2.,
            pulse_amplitude_mv=1., pulse_duration_s=.2, label_source='fixture')
        pulse.update(source_id=evidence['source_id'], frame_id=evidence['frame_id'])
        evidence['pulse'] = pulse
        self.assertEqual(resolve_document_calibration(**evidence)['status'], 'resolved_supplied_evidence')
        wrong = deepcopy(evidence)
        wrong['settings'] = dict(parse_print_settings('50 mm/s 20 mm/mV'), source_id=evidence['source_id'])
        result = resolve_document_calibration(**wrong)
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(len(result['unresolved']), 2)


if __name__ == '__main__':
    unittest.main()
