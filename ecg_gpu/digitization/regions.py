"""Layout-independent trace decoding after region detection and rectification.

This module does not infer calibration, lead identity or regions from RGB.
Those fields come from a paper profile or a separately trained detector.
Repeated leads and arbitrary numbers of regions remain separate observations.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .signals import LEADS


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class TraceCalibration:
    """Local rectified pixel centers, physical units and the first column's time.

    Separate x/y scales support anisotropic network crops. ``baseline_y`` is in
    the heatmap's own y coordinates. ``polarity=-1`` restores a printed -aVR.
    Time windows must be supplied explicitly, including sequential columns.
    """
    instance_id: str
    lead: str
    pixels_per_mm_x: float
    pixels_per_mm_y: float
    speed_mm_per_s: float
    gain_mm_per_mv: float
    baseline_y: float
    start_s: float = 0.
    polarity: int = 1

    def __post_init__(self):
        if not isinstance(self.instance_id, str) or not self.instance_id.strip() or self.lead not in LEADS:
            raise ValueError('A nonempty instance ID and a standard lead are required')
        for name in ('pixels_per_mm_x', 'pixels_per_mm_y', 'speed_mm_per_s', 'gain_mm_per_mv'):
            _positive(getattr(self, name), name)
        if (isinstance(self.baseline_y, bool) or not isinstance(self.baseline_y, Real)
                or not math.isfinite(self.baseline_y)):
            raise ValueError('baseline_y must be finite')
        if isinstance(self.start_s, bool) or not isinstance(self.start_s, Real) or not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError('start_s must be nonnegative and finite')
        if type(self.polarity) is not int or self.polarity not in (-1, 1):
            raise ValueError('polarity must be +1 or -1')


@dataclass(frozen=True)
class TraceObservation:
    """A printed instance in mV. Missing samples are NaN and remain unobserved.

    A full twelve-lead recording is not implied. No relations are imposed on
    different leads or time windows and no rhythm occurrence overwrites another.
    """
    instance_id: str
    lead: str
    values_mv: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    fs: float
    start_s: float

    def __post_init__(self):
        _positive(self.fs, 'fs')
        if not isinstance(self.instance_id, str) or not self.instance_id.strip() or self.lead not in LEADS:
            raise ValueError('A nonempty instance ID and a standard lead are required')
        if isinstance(self.start_s, bool) or not isinstance(self.start_s, Real) or not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError('start_s must be nonnegative and finite')
        values = np.array(self.values_mv, dtype=np.float64, copy=True)
        observed = np.array(self.observed, copy=True)
        confidence = np.array(self.confidence, dtype=np.float64, copy=True)
        if (values.ndim != 1 or not len(values) or observed.dtype != np.bool_
                or values.shape != observed.shape or values.shape != confidence.shape):
            raise ValueError('Matching nonempty vectors and a boolean observation mask required')
        if not np.array_equal(np.isfinite(values), observed) or np.isinf(values).any():
            raise ValueError('Finite mV samples must match observed, missing samples must be NaN')
        if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
            raise ValueError('Confidence must lie in [0,1]')
        for name, value in (('values_mv', values), ('observed', observed), ('confidence', confidence)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def decode_trace_region(probability, calibration, *, confidence_threshold=.15, ambiguity_ratio=.95):
    """Decode one arbitrary lead-region heatmap with fractional y peaks.

    Peaks at the crop boundary, weak peaks and competing local maxima at least
    five pixels apart remain missing. Thresholds are configurable research
    choices requiring validation on deployment data. No gap filling is performed.
    The native sample rate follows horizontal paper calibration, not a fixed
    ten-second canvas. Probabilities are expected at this calibration's scale.
    """
    if not isinstance(calibration, TraceCalibration):
        raise TypeError('A TraceCalibration is required')
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] < 1:
        raise ValueError('A [height >= 3, width >= 1] heatmap is required')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Finite probabilities in [0,1] are required')
    for value, name in ((confidence_threshold, 'confidence_threshold'), (ambiguity_ratio, 'ambiguity_ratio')):
        _positive(value, name)
        if value > 1:
            raise ValueError(f'{name} must be at most one')
    height, width = p.shape
    peak = p.argmax(axis=0)
    columns = np.arange(width)
    center = p[peak, columns]
    left, right = p[np.maximum(peak-1, 0), columns], p[np.minimum(peak+1, height-1), columns]
    denominator = left-2*center+right
    delta = np.zeros(width)
    concave = denominator < -1e-12
    delta[concave] = np.clip(.5*(left[concave]-right[concave])/denominator[concave], -.5, .5)
    maxima = np.zeros_like(p, dtype=bool)
    maxima[1:-1] = (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:])
    separated = np.abs(np.arange(height)[:, None]-peak[None]) >= 5
    second = np.where(maxima & separated, p, 0).max(axis=0)
    observed = ((center >= confidence_threshold) & (peak > 0) & (peak < height-1)
                & (second < ambiguity_ratio*center))
    y = peak+delta
    scale = calibration.pixels_per_mm_y*calibration.gain_mm_per_mv
    values = calibration.polarity*(calibration.baseline_y-y)/scale
    return TraceObservation(calibration.instance_id, calibration.lead,
        np.where(observed, values, np.nan), observed, center,
        calibration.pixels_per_mm_x*calibration.speed_mm_per_s, calibration.start_s)


def resample_trace(observation, fs):
    """Interpolate inside contiguous observed runs, preserving all missing gaps.

    Do not extrapolate beyond observed sample centers, including the last source
    column. The output spans the same declared half-open acquisition interval.
    A single observed column only transfers to an exactly coincident target time.
    """
    if not isinstance(observation, TraceObservation):
        raise TypeError('A TraceObservation is required')
    _positive(fs, 'fs')
    if fs == observation.fs:
        return observation
    count = max(1, round(len(observation.values_mv)/observation.fs*fs))
    times = np.arange(count)/fs
    source = np.arange(len(observation.values_mv))/observation.fs
    values = np.full(count, np.nan)
    confidence = np.zeros(count)
    ids = np.flatnonzero(observation.observed)
    for run in np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1):
        if not len(run):
            continue
        keep = (times >= source[run[0]]-1e-12) & (times <= source[run[-1]]+1e-12)
        values[keep] = np.interp(times[keep], source[run], observation.values_mv[run])
        confidence[keep] = np.interp(times[keep], source[run], observation.confidence[run])
    return TraceObservation(observation.instance_id, observation.lead, values,
                            np.isfinite(values), confidence, fs, observation.start_s)
