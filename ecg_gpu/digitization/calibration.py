"""Document calibration from explicit image evidence, with unresolved outcomes.

Text supplies units. Grid measurements supply pixel density. A labelled pulse
can check scale and anchor the baseline. Neither grid nor pulse establishes the
waveform's time zero. Every spatial observation must use the same image frame.
"""

import math
import re
import unicodedata

import cv2
import numpy as np

from .physical_trace import sample_physical_trace


def positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be positive and finite')


def parse_print_settings(text):
    """Parse observed unit-bearing text. Conflicts and missing units abstain.

    This parser is not OCR. Its correctness does not establish correctness of
    a recognizer's text. No 25 mm/s or 10 mm/mV defaults are introduced.
    """
    if not isinstance(text, str):
        raise TypeError('Observed text required')
    normalized = unicodedata.normalize('NFKC', text).replace('\u2215', '/').replace('\u2044', '/')
    number = r'(?<![\w.,+/-])(?P<value>\d+(?:[.,]\d+)?)[ \t]*'
    patterns = [('paper_speed_mm_s', r'mm[ \t]*/[ \t]*s(?:ec(?:ond)?s?)?\b', lambda v: v),
                ('gain_mm_mv', r'mm[ \t]*/[ \t]*mv\b', lambda v: v),
                ('gain_mm_mv', r'mv[ \t]*/[ \t]*cm\b', lambda v: 10 / v),
                ('gain_mm_mv', r'mv[ \t]*/[ \t]*mm\b', lambda v: 1 / v)]
    candidates = []
    for field, units, convert in patterns:
        for match in re.finditer(number + units, normalized, re.IGNORECASE):
            value = float(match['value'].replace(',', '.'))
            if value <= 0:
                continue
            candidates.append({'field': field, 'value': convert(value), 'observed_text': match[0],
                               'text_span': [match.start(), match.end()]})
    result = {'schema': 'ecg-print-settings-text-v1', 'candidates': candidates,
              'ambiguous_fields': [], 'default_values_used': False, 'ocr_accuracy_validated': False}
    for field in ['paper_speed_mm_s', 'gain_mm_mv']:
        values = [c['value'] for c in candidates if c['field'] == field]
        unique = []
        for value in values:
            if not any(math.isclose(value, other, rel_tol=1e-9) for other in unique):
                unique.append(value)
        result[field] = unique[0] if len(unique) == 1 else None
        if len(unique) > 1:
            result['ambiguous_fields'].append(field)
    return result


def measure_step_pulse(mask, *, pixels_per_mm_x, pixels_per_mm_y,
                       pulse_amplitude_mv, pulse_duration_s, label_source):
    """Measure an independently identified pulse in an isolated rectified ROI.

    Input is a binary ink mask, not arbitrary ECG RGB. Amplitude and duration
    must be supplied from independent label/profile evidence. A pulse-shaped
    waveform cannot be distinguished from a calibration mark by this primitive.
    Broken, clipped or nonrectangular marks are rejected without repair.
    Returned coordinates are local to the ROI, not automatically the page.
    """
    for name, value in [('pixels_per_mm_x', pixels_per_mm_x), ('pixels_per_mm_y', pixels_per_mm_y),
                        ('pulse_amplitude_mv', pulse_amplitude_mv), ('pulse_duration_s', pulse_duration_s)]:
        positive(value, name)
    if not isinstance(label_source, str) or not label_source.strip():
        raise ValueError('Independent pulse amplitude and duration provenance required')
    a = np.asarray(mask)
    if a.dtype != np.bool_ or a.ndim != 2 or min(a.shape) < 7 or not a.any():
        raise ValueError('Nonempty isolated binary pulse ROI required')
    if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
        raise ValueError('Pulse touches ROI boundary, clipping cannot be excluded')
    count, _ = cv2.connectedComponents(a.astype(np.uint8), connectivity=8)
    if count != 2:
        raise ValueError('A single connected pulse is required, no gap filling')
    yy, xx = np.nonzero(a)
    height = float(yy.max() - yy.min())
    if height < 4:
        raise ValueError('Insufficient pulse height')
    columns = np.arange(xx.min(), xx.max() + 1)
    low = np.array([np.flatnonzero(a[:, x]).min() if a[:, x].any() else np.nan for x in columns])
    high = np.array([np.flatnonzero(a[:, x]).max() if a[:, x].any() else np.nan for x in columns])
    edge_indices = np.flatnonzero(high - low >= .7 * height)
    groups = np.split(edge_indices, np.flatnonzero(np.diff(edge_indices) != 1) + 1)
    if len(groups) != 2 or any(not len(g) for g in groups):
        raise ValueError('Two distinct vertical pulse edges required')
    left, right = groups
    plateau = np.arange(left[-1] + 1, right[0])
    feet = np.r_[np.arange(left[0]), np.arange(right[-1] + 1, len(columns))]
    if len(plateau) < 3 or len(feet) < 2 or not np.isfinite(low).all():
        raise ValueError('Observed plateau and baseline foot required')
    center = (low + high) / 2
    top, baseline = float(np.median(center[plateau])), float(np.median(center[feet]))
    if (np.max(np.abs(center[plateau] - top)) > 1. or np.max(np.abs(center[feet] - baseline)) > 1.
            or baseline - top < .7 * height):
        raise ValueError('Pulse plateau or baseline is inconsistent')
    left_x, right_x = float(columns[left].mean()), float(columns[right].mean())
    width, amplitude = right_x - left_x, baseline - top
    return {'schema': 'rectified-labelled-pulse-measurement-v1', 'baseline_y_px': baseline,
        'top_y_px': top, 'left_x_px': left_x, 'right_x_px': right_x,
        'height_px': amplitude, 'width_px': width,
        'gain_mm_mv': amplitude / (pixels_per_mm_y * pulse_amplitude_mv),
        'paper_speed_mm_s': width / (pixels_per_mm_x * pulse_duration_s),
        'gain_interval_mm_mv': [max(0., amplitude - 2.) / (pixels_per_mm_y * pulse_amplitude_mv),
                                (amplitude + 2.) / (pixels_per_mm_y * pulse_amplitude_mv)],
        'speed_interval_mm_s': [max(0., width - 2.) / (pixels_per_mm_x * pulse_duration_s),
                                (width + 2.) / (pixels_per_mm_x * pulse_duration_s)],
        'interval_basis': 'two-pixel discretization allowance, not a statistical confidence interval',
        'label_source': label_source, 'pulse_amplitude_mv': pulse_amplitude_mv,
        'pulse_duration_s': pulse_duration_s, 'waveform_time_zero_inferred': False,
        'coordinates': 'local isolated pulse ROI', 'photographic_accuracy_validated': False}


def resolve_document_calibration(settings, *, source_id, frame_id, grid, baseline, time_anchor, pulse=None):
    """Resolve supplied evidence in one frame or return named unresolved fields.

    Origin and baseline are never inferred from a trace mean or visible extent.
    Spatial evidence carries image and frame identity so original-image pixels
    cannot silently be used after resampling, perspective correction or dewarp.
    """
    if not isinstance(source_id, str) or not re.fullmatch(r'[a-f0-9]{64}', source_id) or not isinstance(frame_id, str) or not frame_id:
        raise ValueError('Explicit image SHA256 and spatial frame required')
    if settings.get('schema') != 'ecg-print-settings-text-v1' or settings.get('default_values_used'):
        raise ValueError('Observed unit-bearing print settings required')
    if settings.get('source_id') != source_id:
        raise ValueError('Print settings must belong to the same source image')
    unresolved = []
    for name, evidence in [('grid', grid), ('baseline', baseline), ('time_anchor', time_anchor)]:
        if evidence is None:
            unresolved.append(name)
        elif evidence.get('source_id') != source_id or evidence.get('frame_id') != frame_id:
            raise ValueError('Spatial evidence belongs to another image or coordinate frame')
    for field in ['paper_speed_mm_s', 'gain_mm_mv']:
        if settings.get(field) is None or field in settings.get('ambiguous_fields', []):
            unresolved.append(field)
        else:
            positive(settings[field], field)
    if grid is not None:
        for field in ['pixels_per_mm_x', 'pixels_per_mm_y']:
            positive(grid[field], field)
    if baseline is not None:
        values = np.asarray(baseline['y_px'], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Explicit finite baseline for each observed row required')
        if baseline.get('method') not in ['pulse_foot', 'supplied_iso_line', 'analytic_fixture']:
            unresolved.append('independent_baseline')
        baseline = dict(baseline, y_px=values.tolist())
    if time_anchor is not None:
        if (isinstance(time_anchor['x_px'], bool) or isinstance(time_anchor['time_s'], bool)
                or not math.isfinite(time_anchor['x_px']) or not math.isfinite(time_anchor['time_s'])):
            raise ValueError('Finite observed time marker required')
        if time_anchor.get('method') not in ['printed_time_marker', 'supplied_time_marker', 'analytic_fixture']:
            unresolved.append('independent_time_anchor')
    if pulse is not None:
        if pulse.get('source_id') != source_id or pulse.get('frame_id') != frame_id:
            raise ValueError('Pulse belongs to another image or coordinate frame')
        for field, interval in [('paper_speed_mm_s', 'speed_interval_mm_s'), ('gain_mm_mv', 'gain_interval_mm_mv')]:
            value = settings.get(field)
            if value is not None and not pulse[interval][0] <= value <= pulse[interval][1]:
                unresolved.append('pulse_text_disagreement_' + field)
    return {'schema': 'document-physical-calibration-v1', 'status': 'unresolved' if unresolved else 'resolved_supplied_evidence',
        'source_id': source_id, 'frame_id': frame_id, 'unresolved': sorted(set(unresolved)),
        'settings': settings, 'grid': grid, 'baseline': baseline, 'time_anchor': time_anchor,
        'pulse': pulse, 'reference_signal_received': False, 'photographic_accuracy_validated': False}


def sample_with_document_calibration(y_px, x_px, calibration, *, fs, samples, max_gap_s, start_s=0.):
    calibration = resolve_document_calibration(calibration['settings'], source_id=calibration['source_id'],
        frame_id=calibration['frame_id'], grid=calibration['grid'], baseline=calibration['baseline'],
        time_anchor=calibration['time_anchor'], pulse=calibration['pulse'])
    if calibration.get('status') != 'resolved_supplied_evidence' or calibration['unresolved']:
        raise ValueError('Unresolved document calibration, no waveform is fabricated')
    grid, settings, anchor = (calibration[k] for k in ['grid', 'settings', 'time_anchor'])
    zero = anchor['x_px'] - anchor['time_s'] * grid['pixels_per_mm_x'] * settings['paper_speed_mm_s']
    values, metadata = sample_physical_trace(y_px, x_px, baseline_y_px=calibration['baseline']['y_px'],
        x_origin_px=zero, pixels_per_mm_x=grid['pixels_per_mm_x'], pixels_per_mm_y=grid['pixels_per_mm_y'],
        paper_speed_mm_s=settings['paper_speed_mm_s'], gain_mm_mv=settings['gain_mm_mv'],
        fs=fs, samples=samples, max_gap_s=max_gap_s, start_s=start_s)
    metadata.update(calibration=calibration, origin_fitted_to_visible_trace=False,
                    baseline_fitted_to_trace_mean=False, photographic_accuracy_validated=False)
    return values, metadata
