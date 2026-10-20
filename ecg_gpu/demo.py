"""Artificial integration signals, not a validated physiological simulator.

Limb leads obey their voltage identities. Precordial waves deliberately vary
in shape so visual smoke tests do not show twelve copies of one scaled trace.
"""
import numpy as np
from .schema_v05 import LEADS as STANDARD_LEADS

def demo_signals(fs=500, duration=10):
    t = np.arange(int(fs*duration)) / fs
    phases = t % 0.85
    p = np.exp(-((phases-.15)/.04)**2)
    q = np.exp(-((phases-.28)/.014)**2)
    r = np.exp(-((phases-.31)/.018)**2)
    s = np.exp(-((phases-.35)/.018)**2)
    twave = np.exp(-((phases-.57)/.08)**2)
    lead_i = .08*p-.08*q+.84*r-.13*s+.20*twave
    lead_ii = .12*p-.10*q+1.05*r-.23*s+.28*twave
    limbs = [lead_i, lead_ii, lead_ii-lead_i, -(lead_i+lead_ii)/2,
             lead_i-lead_ii/2, lead_ii-lead_i/2]
    precordial = [.04*p+.20*r-.85*s-.08*twave,
                 .06*p+.42*r-1.00*s+.10*twave,
                 .08*p+.78*r-.75*s+.23*twave,
                 .10*p-.04*q+1.15*r-.38*s+.32*twave,
                 .09*p-.07*q+1.02*r-.18*s+.29*twave,
                 .08*p-.08*q+.84*r-.12*s+.24*twave]
    return np.stack(limbs+precordial), fs, STANDARD_LEADS

