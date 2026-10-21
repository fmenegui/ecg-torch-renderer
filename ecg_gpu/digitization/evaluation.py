"""Signal-level evaluation using the locked official Kaggle metric.

The official alignment tolerates timing and DC errors. Raw, unaligned errors
are reported alongside it, so a high competition score cannot hide these.
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np

from .artifacts import artifact_lock, sha256_file
from .signals import LEADS, LeadSignal


class OfficialMetric:
    """Use selected functions from the exact official metric v7 source."""

    def __init__(self, assets):
        import pandas as pd
        import scipy.optimize
        import scipy.signal
        from .reference import _definitions

        source = Path(assets)/'official_metric.py'
        item = next(a for a in artifact_lock()['artifacts'] if a['path'] == 'official_metric.py')
        if sha256_file(source) != item['sha256']:
            raise ValueError('Official metric checksum mismatch')
        env = {'np': np, 'pd': pd, 'scipy': scipy, 'Tuple': Tuple, 'LEADS': list(LEADS),
               'MAX_TIME_SHIFT': .2, 'PERFECT_SCORE': 1e6}
        self.functions = _definitions(source,
            ('ParticipantVisibleError', 'compute_power', 'compute_snr', 'align_signals',
             '_calculate_image_score', 'score'), env)
        self.source_sha256 = item['sha256']
        self.pd = pd

    def _frames(self, reference, prediction, image_id):
        if set(reference) != set(LEADS) or set(prediction) != set(LEADS):
            raise ValueError('Exactly twelve leads are required for official scoring')
        if not image_id or '_' in image_id:
            raise ValueError('Metric image ID must be nonempty and contain no underscore')
        fs = reference['II'].fs
        if fs != int(fs) or len(reference['II'].values) != 10*fs:
            raise ValueError('Official metric requires ten seconds of lead II at integer fs')
        truth, pred = [], []
        for name in LEADS:
            a, b = reference[name], prediction[name]
            if a.fs != fs or b.fs != fs or a.start_s != b.start_s or a.values.shape != b.values.shape:
                raise ValueError(f'Mismatched sample rate, time window or length for {name}')
            truth.extend({'id': f'{image_id}_{i}_{name}', 'fs': fs, 'value': float(v)}
                         for i, v in enumerate(a.values))
            pred.extend({'id': f'{image_id}_{i}_{name}', 'value': float(v)}
                        for i, v in enumerate(b.values))
        return self.pd.DataFrame(truth), self.pd.DataFrame(pred)

    def image(self, reference, prediction, *, image_id='image'):
        truth, pred = self._frames(reference, prediction, image_id)
        official = float(self.functions['score'](truth, pred, 'id'))
        per_lead = {}
        signal_power = noise_power = 0.
        for name in LEADS:
            a = reference[name].values
            b = prediction[name].values
            aligned = self.functions['align_signals'](a, b, int(reference[name].fs*.2))
            power, noise = self.functions['compute_power'](a, aligned.copy())
            signal_power += float(power)
            noise_power += float(noise)
            raw_error = b-a
            from scipy.signal import correlate, correlation_lags
            correlation = correlate(a-a.mean(), b-b.mean(), mode='full')
            lags = correlation_lags(len(a), len(b), mode='full')
            allowed = np.abs(lags) <= int(reference[name].fs*.2)
            candidates = np.flatnonzero(allowed & (correlation == correlation[allowed].max()))
            lag = min((int(lags[i]) for i in candidates), key=abs)
            per_lead[name] = {'raw_rmse_mv': float(np.sqrt(np.mean(raw_error**2))),
                'raw_bias_mv': float(raw_error.mean()),
                'raw_max_abs_error_mv': float(np.abs(raw_error).max()),
                'alignment_shift_samples': lag, 'alignment_shift_ms': 1000*lag/reference[name].fs,
                'signal_power': float(power), 'aligned_noise_power': float(noise)}
        ratio = float(self.functions['compute_snr'](signal_power, noise_power))
        if ratio > 0 and not math.isclose(official, 10*math.log10(ratio), abs_tol=1e-9):
            raise RuntimeError('Official per-image metric and power aggregation disagree')
        return {'official_snr_db': official, 'official_power_ratio': ratio,
                'signal_power': signal_power, 'aligned_noise_power': noise_power,
                'per_lead': per_lead}

    def aggregate(self, items):
        """Official aggregation: mean capped LINEAR image ratios, then log10."""
        if not items:
            return None
        ratios = np.asarray([item['official_power_ratio'] for item in items], dtype=np.float64)
        if not np.isfinite(ratios).all() or (ratios < 0).any():
            raise ValueError('Invalid per-image power ratio')
        mean = float(ratios.mean())
        return max(10*math.log10(mean), -384.) if mean > 0 else -384.


def zero_prediction(reference):
    """Explicit failure penalty for scoring only. Never a successful output."""
    return {name: LeadSignal(np.zeros_like(lead.values), lead.fs, lead.start_s)
            for name, lead in reference.items()}
