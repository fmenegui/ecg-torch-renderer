"""Artificial integration signals, not a validated physiological simulator.

Limb leads obey their voltage identities. Precordial waves deliberately vary
in shape so visual smoke tests do not show twelve copies of one scaled trace.

The defaults reproduce the original strictly periodic waveform. Optional
heart-rate variability, baseline wander and broadband noise make galleries
look less mechanical. They are cosmetic, not models of real acquisition.
"""
import math

import numpy as np
from .schema_v05 import LEADS as STANDARD_LEADS

DEFAULT_RR_S = .85


def _beat_phases(t, heart_rate, hrv, rng):
    """Seconds since the most recent beat onset for every sample time."""
    if heart_rate is None and hrv == 0:
        return t % DEFAULT_RR_S
    rr = DEFAULT_RR_S if heart_rate is None else 60. / heart_rate
    count = int(math.ceil(t[-1] / (rr * (1 - 3 * hrv)))) + 2 if len(t) else 1
    intervals = rr * (1 + hrv * rng.standard_normal(count)).clip(.7, 1.3)
    onsets = np.concatenate([[0.], np.cumsum(intervals)])
    index = np.searchsorted(onsets, t, side='right') - 1
    return t - onsets[index]


def demo_signals(fs=500, duration=10, *, heart_rate=None, hrv=0., baseline_wander_mv=0.,
                 noise_mv=0., seed=None):
    """Return ``(signals[12,T] in mV, fs, lead names)``.

    ``heart_rate`` in beats/min (40..120, default about 71). ``hrv`` is the
    relative RR standard deviation (0..0.1). ``baseline_wander_mv`` and
    ``noise_mv`` add slow drift and white noise at the electrode level, so
    limb-lead identities still hold. ``seed`` makes the variation repeatable.
    """
    if heart_rate is not None and not 40 <= heart_rate <= 120:
        raise ValueError('heart_rate must be in 40..120 beats/min')
    if not 0 <= hrv <= .1:
        raise ValueError('hrv must be in [0, 0.1]')
    if baseline_wander_mv < 0 or noise_mv < 0:
        raise ValueError('baseline_wander_mv and noise_mv must be nonnegative')
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs*duration)) / fs
    phases = _beat_phases(t, heart_rate, hrv, rng)
    p = np.exp(-((phases-.15)/.04)**2)
    q = np.exp(-((phases-.28)/.014)**2)
    r = np.exp(-((phases-.31)/.018)**2)
    s = np.exp(-((phases-.35)/.018)**2)
    twave = np.exp(-((phases-.57)/.08)**2)
    lead_i = .08*p-.08*q+.84*r-.13*s+.20*twave
    lead_ii = .12*p-.10*q+1.05*r-.23*s+.28*twave
    precordial = [.04*p+.20*r-.85*s-.08*twave,
                 .06*p+.42*r-1.00*s+.10*twave,
                 .08*p+.78*r-.75*s+.23*twave,
                 .10*p-.04*q+1.15*r-.38*s+.32*twave,
                 .09*p-.07*q+1.02*r-.18*s+.29*twave,
                 .08*p-.08*q+.84*r-.12*s+.24*twave]
    if baseline_wander_mv or noise_mv:
        def disturbance():
            phase = rng.uniform(0, 2*math.pi)
            frequency = rng.uniform(.15, .35)
            drift = baseline_wander_mv * np.sin(2*math.pi*frequency*t + phase)
            return drift + noise_mv * rng.standard_normal(len(t))
        lead_i = lead_i + disturbance()
        lead_ii = lead_ii + disturbance()
        precordial = [wave + disturbance() for wave in precordial]
    limbs = [lead_i, lead_ii, lead_ii-lead_i, -(lead_i+lead_ii)/2,
             lead_i-lead_ii/2, lead_ii-lead_i/2]
    return np.stack(limbs+precordial), fs, STANDARD_LEADS
