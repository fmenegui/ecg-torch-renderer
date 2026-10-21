"""Subpixel waveform decoding and explicit time/amplitude calibration."""

from dataclasses import dataclass, field
import math

import numpy as np

LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
ROW_LEADS = (('I', 'aVR', 'V1', 'V4'), ('II', 'aVL', 'V2', 'V5'), ('III', 'aVF', 'V3', 'V6'))


@dataclass(frozen=True)
class LeadSignal:
    """Samples in mV with physical sample rate and printed time window."""

    values: np.ndarray
    fs: float
    start_s: float = 0.

    def __post_init__(self):
        a = np.array(self.values, dtype=np.float64, copy=True)
        if a.ndim != 1 or len(a) < 1 or not np.isfinite(a).all():
            raise ValueError('Lead samples must be a nonempty finite vector in mV')
        if isinstance(self.fs, bool) or not math.isfinite(self.fs) or self.fs <= 0:
            raise ValueError('A positive finite sample rate is required')
        if not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError('A nonnegative finite start time is required')
        a.setflags(write=False)
        object.__setattr__(self, 'values', a)


@dataclass
class DigitizationResult:
    """A successful reconstruction. Failures are reported separately, never zeros."""

    leads: dict[str, LeadSignal]
    provenance: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if set(self.leads) != set(LEADS):
            raise ValueError('Exactly the twelve named ECG leads are required')
        if not all(isinstance(v, LeadSignal) for v in self.leads.values()):
            raise TypeError('Every lead must be a LeadSignal')
        if len({v.fs for v in self.leads.values()}) != 1:
            raise ValueError('All leads must share the sample rate')


def decode_waveform_rows(probability, *, zero_mv, pixels_per_mv, length,
                         confidence_threshold=.15, gap_interpolation='linear'):
    """Decode four row heatmaps by quadratic peak refinement, then Fourier resample.

    Return mV rows and the *original column* observation mask. Interpolated
    columns remain explicitly marked as missing. This follows Tang's peak
    decoder, with an explicit mask and rejection of completely absent rows.
    No diagnosis or synthetic beat is introduced to fill a missing trace.
    """
    from scipy.signal import resample

    p = np.asarray(probability)
    z = np.asarray(zero_mv, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] != 4 or min(p.shape[1:]) < 3:
        raise ValueError('Expected a [4, height, width] probability map')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite and lie in [0, 1]')
    if z.shape != (4,) or not np.isfinite(z).all():
        raise ValueError('Four finite zero-mV positions are required')
    if not math.isfinite(pixels_per_mv) or pixels_per_mv <= 0:
        raise ValueError('pixels_per_mv must be positive and finite')
    if type(length) is not int or length < 4:
        raise ValueError('Output length must be an integer >= 4')
    if not math.isfinite(confidence_threshold) or not 0 < confidence_threshold <= 1:
        raise ValueError('confidence_threshold must lie in (0, 1]')
    if gap_interpolation not in ('linear', 'pchip'):
        raise ValueError('gap_interpolation must be linear or pchip')
    height, width = p.shape[1:]
    peak = p.argmax(axis=1)
    def at(index):
        return np.take_along_axis(p, index[:, None, :], axis=1)[:, 0, :]
    center = at(peak)
    left = at(np.maximum(peak-1, 0))
    right = at(np.minimum(peak+1, height-1))
    denominator = left-2*center+right+1e-8
    delta = np.clip(.5*(left-right)/denominator, -.5, .5)
    y = peak.astype(np.float64)+delta
    observed = center >= confidence_threshold
    for row in range(4):
        valid_x = np.flatnonzero(observed[row])
        missing_x = np.flatnonzero(~observed[row])
        if len(valid_x) == 0:
            raise ValueError(f'No observed waveform in row {row}')
        if len(missing_x):
            valid_y = y[row, valid_x]
            if gap_interpolation == 'pchip' and len(valid_x) > 2:
                from scipy.interpolate import PchipInterpolator
                y[row, missing_x] = PchipInterpolator(valid_x, valid_y)(missing_x)
            else:
                y[row, missing_x] = np.interp(missing_x, valid_x, valid_y)
    if length != width:
        y = resample(y, length, axis=-1)
    values = (z[:, None]-y)/pixels_per_mv
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite decoded waveform')
    return values, observed


def split_rows(rows_mv, fs, *, average_ii=True, einthoven=True):
    """Kaggle 3x4 + long-II layout, including Tang's optional lead fusion.

    This calibration assumes ten seconds, sequential 2.5-second short leads.
    Relations between I, II and III use their simultaneous first segment only.
    The returned starts preserve the printed time windows for every lead.
    """
    a = np.asarray(rows_mv, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != 4 or a.shape[1] < 4 or not np.isfinite(a).all():
        raise ValueError('Expected four finite waveform rows')
    if not math.isfinite(fs) or fs <= 0 or a.shape[1] != round(10*fs):
        raise ValueError('Kaggle row length must equal ten seconds at fs')
    n = a.shape[1]
    boundaries = [round(i*n/4) for i in range(5)]
    leads = {}
    starts = {}
    for row, names in enumerate(ROW_LEADS):
        for col, name in enumerate(names):
            leads[name] = a[row, boundaries[col]:boundaries[col+1]].copy()
            starts[name] = boundaries[col]/fs
    short_ii = leads['II']
    leads['II'] = a[3].copy()
    if average_ii:
        leads['II'][:len(short_ii)] = .5*(short_ii+leads['II'][:len(short_ii)])
    if einthoven:
        length = min(len(leads[name]) for name in ('I', 'II', 'III'))
        residual = (leads['II'][:length]-leads['I'][:length]-leads['III'][:length])/3
        leads['I'][:length] += residual
        leads['III'][:length] += residual
        leads['II'][:length] -= residual
    return {name: LeadSignal(leads[name], fs, starts[name]) for name in LEADS}
