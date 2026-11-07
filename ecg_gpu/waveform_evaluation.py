"""Offline sim-to-real waveform benchmark: SNR of digitized signals vs references.

This is the measurement behind roadmap item 1. A digitizer trained on images
from this generator (or any other) writes one prediction per real image; this
module scores the predictions against reference signals and summarizes them
per capture type with patient-level bootstrap intervals.

Scoring follows the signal-to-noise ratio used by the George B. Moody
PhysioNet Challenge 2024 for reconstruction quality: for each lead,
``10 log10(sum(ref^2) / sum((ref - est)^2))``. Missing samples (NaN) in the
estimate count as zero. Optional vertical-offset removal and a bounded time
shift can be enabled; report which options were used, since they change the
score. Everything here is NumPy on the host, outside any training loop.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

MAX_SNR_DB = 100.


def _as_leads(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2:
        raise ValueError(f'{name} must be [leads, samples]')
    return value


def resample(signal, fs, target_fs, length):
    """Linear resampling of ``[L,T]`` to ``length`` samples at ``target_fs``."""
    signal = _as_leads(signal, 'signal')
    source_t = np.arange(signal.shape[1]) / fs
    target_t = np.arange(length) / target_fs
    out = np.full((signal.shape[0], length), np.nan)
    for lead, values in enumerate(signal):
        valid = np.isfinite(values)
        if valid.sum() >= 2:
            inside = (target_t >= source_t[valid][0]) & (target_t <= source_t[valid][-1])
            out[lead, inside] = np.interp(target_t[inside], source_t[valid], values[valid])
    return out


def lead_snr_db(reference, estimate, *, remove_offset=False, max_shift=0):
    """SNR in dB for one lead; ``max_shift`` in samples, searched both ways."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.nan_to_num(np.asarray(estimate, dtype=np.float64), nan=0.)
    if reference.shape != estimate.shape or reference.ndim != 1:
        raise ValueError('reference and estimate must be 1-D arrays of equal length')
    signal_power = float(np.square(reference).sum())
    if signal_power == 0:
        return float('nan')
    best = -math.inf
    for shift in range(-int(max_shift), int(max_shift) + 1):
        shifted = np.roll(estimate, shift)
        if shift > 0:
            shifted[:shift] = 0.
        elif shift < 0:
            shifted[shift:] = 0.
        if remove_offset:
            shifted = shifted + float(np.mean(reference - shifted))
        noise = float(np.square(reference - shifted).sum())
        snr = MAX_SNR_DB if noise == 0 else min(MAX_SNR_DB, 10 * math.log10(signal_power / noise))
        best = max(best, snr)
    return best


def record_snr_db(reference, estimate, *, remove_offset=False, max_shift=0):
    """Mean SNR over leads with nonzero reference power, and the per-lead values."""
    reference, estimate = _as_leads(reference, 'reference'), _as_leads(estimate, 'estimate')
    if reference.shape != estimate.shape:
        raise ValueError(f'Shape mismatch: reference {reference.shape}, estimate {estimate.shape}')
    per_lead = [lead_snr_db(r, e, remove_offset=remove_offset, max_shift=max_shift)
                for r, e in zip(reference, estimate)]
    finite = [value for value in per_lead if math.isfinite(value)]
    return (float(np.mean(finite)) if finite else float('nan')), per_lead


def patient_bootstrap(values, patients, *, repeats=2000, seed=71):
    """Percentile 95 % interval of the mean, resampling whole patient groups."""
    groups = {}
    for value, patient in zip(values, patients):
        groups.setdefault(patient, []).append(value)
    keys = sorted(groups)
    if len(keys) < 2:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        chosen = rng.integers(len(keys), size=len(keys))
        sample = [value for index in chosen for value in groups[keys[index]]]
        means.append(np.mean(sample))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _load_signal(path):
    with np.load(path, allow_pickle=False) as data:
        if 'signals' not in data or 'fs' not in data:
            raise ValueError(f'{path} must contain "signals" [12,T] and "fs"')
        return np.asarray(data['signals'], dtype=np.float64), float(data['fs'])


def evaluate(manifest, predictions, *, remove_offset=False, max_shift_s=0., repeats=2000):
    """Score every manifest row that has a prediction; missing predictions score as zeros.

    ``manifest`` is a JSON-lines file with ``id``, ``reference`` (an .npz with
    ``signals`` [12,T] mV and ``fs``), ``patient`` and ``capture`` (for example
    ``scan``, ``phone_photo``, ``screen``). ``predictions`` is a directory of
    ``<id>.npz`` files in the same format.
    """
    manifest, predictions = Path(manifest), Path(predictions)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    results = []
    for row in rows:
        for key in ('id', 'reference', 'patient', 'capture'):
            if key not in row:
                raise ValueError(f'Manifest row lacks {key!r}: {row}')
        reference, fs = _load_signal(manifest.parent / row['reference'])
        prediction_path = predictions / f"{row['id']}.npz"
        if prediction_path.exists():
            estimate, estimate_fs = _load_signal(prediction_path)
            estimate = resample(estimate, estimate_fs, fs, reference.shape[1])
            missing = False
        else:
            estimate, missing = np.zeros_like(reference), True
        snr, per_lead = record_snr_db(reference, estimate, remove_offset=remove_offset,
                                      max_shift=round(max_shift_s * fs))
        results.append({'id': row['id'], 'patient': row['patient'], 'capture': row['capture'],
                        'snr_db': snr, 'lead_snr_db': per_lead, 'missing_prediction': missing})
    summary = {}
    for capture in sorted({row['capture'] for row in results} | {'all'}):
        chosen = [row for row in results if capture == 'all' or row['capture'] == capture]
        values = [row['snr_db'] for row in chosen if math.isfinite(row['snr_db'])]
        summary[capture] = {
            'records': len(chosen), 'patients': len({row['patient'] for row in chosen}),
            'missing_predictions': sum(row['missing_prediction'] for row in chosen),
            'mean_snr_db': float(np.mean(values)) if values else float('nan'),
            'bootstrap_95': patient_bootstrap([row['snr_db'] for row in chosen if math.isfinite(row['snr_db'])],
                                              [row['patient'] for row in chosen if math.isfinite(row['snr_db'])],
                                              repeats=repeats)}
    return {'schema': 'ecg-waveform-benchmark-v1',
            'metric': 'per-lead SNR dB, mean over leads, mean over records',
            'options': {'remove_offset': remove_offset, 'max_shift_s': max_shift_s,
                        'missing_samples': 'zero', 'max_snr_db': MAX_SNR_DB,
                        'bootstrap': {'unit': 'patient', 'repeats': repeats}},
            'summary': summary, 'records': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Score digitized ECG signals against references (SNR dB).')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--remove-offset', action='store_true')
    parser.add_argument('--max-shift-s', type=float, default=0.)
    args = parser.parse_args(argv)
    report = evaluate(args.manifest, args.predictions, remove_offset=args.remove_offset,
                      max_shift_s=args.max_shift_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    for capture, values in report['summary'].items():
        print(f"{capture}: {values['mean_snr_db']:.2f} dB over {values['records']} records, "
              f"95 % CI {values['bootstrap_95']}")


if __name__ == '__main__':
    main()
