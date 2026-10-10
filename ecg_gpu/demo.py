"""Entirely artificial integration examples, no patient signals."""
import numpy as np
from .schema_v05 import LEADS as STANDARD_LEADS

def demo_signals(fs=500, duration=10):
    t = np.arange(int(fs*duration)) / fs
    phases = t % 0.85
    base = (0.10*np.exp(-((phases-0.15)/0.04)**2) - 0.12*np.exp(-((phases-0.28)/0.014)**2)
            + 1.05*np.exp(-((phases-0.31)/0.018)**2) - 0.23*np.exp(-((phases-0.35)/0.018)**2)
            + 0.25*np.exp(-((phases-0.57)/0.08)**2))
    amplitudes = [0.8, 1.0, 0.6, -0.6, 0.45, 0.7, -0.5, 0.7, 1.0, 1.1, 0.9, 0.75]
    return np.stack([a * base for a in amplitudes]), fs, STANDARD_LEADS
