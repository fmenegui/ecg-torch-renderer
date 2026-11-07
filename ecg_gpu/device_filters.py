"""Recorder display filters applied to the printed trace, never to the source.

Electrocardiographs print a filtered signal: a high-pass for baseline wander
(0.05-0.5 Hz), a low-pass for muscle noise (40-150 Hz) and often a mains notch
(50 or 60 Hz). The printed waveform therefore differs from the stored source
signal. This module produces that printed signal so the image shows what a
recorder would print, while ``RenderBatch.reference`` keeps the untouched
source and ``metadata['printed_signals']`` records the printed version.

Filters are zero-phase magnitude responses applied in the frequency domain
after even-symmetric extension, which avoids wrap-around at the record ends.
They approximate the amplitude behaviour of recorder filters, not the phase
of a specific device.
"""
import torch

HIGHPASS_HZ = (.05, .15, .5)
LOWPASS_HZ = (40., 100., 150.)
NOTCH_HZ = (0., 50., 60.)


def filter_response(freqs, highpass_hz, lowpass_hz, notch_hz, notch_width_hz=1.):
    """Magnitude response ``[B,F]`` for per-record cutoffs ``[B]``.

    High-pass: first-order Butterworth. Low-pass: second-order Butterworth.
    Notch: Gaussian rejection of the given width; 0 Hz disables it.
    """
    f = freqs.clamp_min(1e-9)
    high = highpass_hz[:, None]
    low = lowpass_hz[:, None]
    notch = notch_hz[:, None]
    hp = torch.where(high > 0, 1 / (1 + (high / f).square()).sqrt(), torch.ones_like(f))
    lp = 1 / (1 + (f / low) ** 4).sqrt()
    reject = 1 - torch.exp(-.5 * ((freqs - notch) / notch_width_hz).square())
    nt = torch.where(notch > 0, reject, torch.ones_like(f))
    return hp * lp * nt


def printed_signal(signals, rates, highpass_hz, lowpass_hz, notch_hz):
    """Filter ``[B,C,T]`` mV signals sampled at ``rates`` ``[B]`` Hz per record."""
    length = signals.shape[-1]
    extended = torch.cat([signals, signals.flip(-1)], -1)
    spectrum = torch.fft.rfft(extended, dim=-1)
    freqs = torch.fft.rfftfreq(2 * length, device=signals.device)[None] * rates[:, None]
    response = filter_response(freqs, highpass_hz.to(freqs.dtype), lowpass_hz.to(freqs.dtype),
                               notch_hz.to(freqs.dtype))
    filtered = torch.fft.irfft(spectrum * response[:, None, :], n=2 * length, dim=-1)
    return filtered[..., :length].to(signals.dtype)
