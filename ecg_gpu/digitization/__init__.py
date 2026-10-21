"""ECG image digitization boundaries, independent of the renderer hot path.

Optional reference networks are loaded only when explicitly requested.
"""

from .signals import LeadSignal, DigitizationResult, decode_waveform_rows, split_rows

__all__ = ['LeadSignal', 'DigitizationResult', 'decode_waveform_rows', 'split_rows']
