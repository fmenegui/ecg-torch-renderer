"""Experimental coordinate-preserving resampling of calibrated raster traces.

The caller must supply observed x coordinates, physical calibration and a
baseline for each trace. None are inferred from the extent of visible samples.
This is a building block, not a validated replacement for a native digitizer.
"""

import math

import numpy as np


def sample_physical_trace(y_px, x_px, *, baseline_y_px, x_origin_px,
                          pixels_per_mm_x, pixels_per_mm_y, paper_speed_mm_s,
                          gain_mm_mv, fs, samples, max_gap_s, start_s=0.):
    y, x = np.asarray(y_px, dtype=np.float64), np.asarray(x_px, dtype=np.float64)
    baseline = np.asarray(baseline_y_px, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < 2 or x.shape != (y.shape[1],) or baseline.shape != (y.shape[0],):
        raise ValueError('Explicit [trace, column] coordinates and per-trace baselines required')
    if not y.shape[0] or np.isinf(y).any() or not np.isfinite(x).all() or not np.isfinite(baseline).all() or not (np.diff(x) > 0).all():
        raise ValueError('Finite increasing x coordinates and baselines, y may contain NaN')
    for value in [pixels_per_mm_x, pixels_per_mm_y, paper_speed_mm_s, gain_mm_mv, fs, max_gap_s]:
        if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive explicit physical calibration required')
    if type(samples) is not int or samples < 1 or not math.isfinite(x_origin_px) or not math.isfinite(start_s):
        raise ValueError('Finite public time origin and positive output length required')
    source_t = (x - x_origin_px) / (pixels_per_mm_x * paper_speed_mm_s)
    target_t = start_s + np.arange(samples, dtype=np.float64) / fs
    mv = (baseline[:, None] - y) / (pixels_per_mm_y * gain_mm_mv)
    output = np.full((len(y), samples), np.nan)
    right = np.searchsorted(source_t, target_t)
    safe = np.minimum(right, len(x) - 1)
    exact = (right < len(x)) & np.isclose(source_t[safe], target_t, rtol=0, atol=1e-10)
    output[:, exact] = mv[:, safe[exact]]
    inside = ~exact & (right > 0) & (right < len(x))
    columns = np.flatnonzero(inside)
    lo, hi = right[columns] - 1, right[columns]
    dt = source_t[hi] - source_t[lo]
    usable = (dt <= max_gap_s + 1e-12)[None] & np.isfinite(mv[:, lo]) & np.isfinite(mv[:, hi])
    fraction = (target_t[columns] - source_t[lo]) / dt
    interpolated = mv[:, lo] * (1 - fraction) + mv[:, hi] * fraction
    output[:, columns] = np.where(usable, interpolated, np.nan)
    return output, {'schema': 'physical-trace-resampling-v1', 'fs': float(fs), 'samples': samples,
                    'start_s': float(start_s), 'x_origin_px': float(x_origin_px),
                    'pixels_per_mm_x': float(pixels_per_mm_x), 'pixels_per_mm_y': float(pixels_per_mm_y),
                    'paper_speed_mm_s': float(paper_speed_mm_s), 'gain_mm_mv': float(gain_mm_mv),
                    'max_gap_s': float(max_gap_s), 'calibration_supplied_by_caller': True,
                    'time_fitted_to_visible_extent': False, 'missing_samples': int(np.isnan(output).sum()),
                    'missing_gaps_bridged': False, 'extrapolation': False,
                    'baseline_inferred_from_trace_mean': False, 'validated_on_real_photographs': False}
