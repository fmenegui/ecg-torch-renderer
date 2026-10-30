"""Missing-aware signal scores for externally acquired ECG images.

Reference NaNs mark unprinted time, prediction NaNs mark unrecovered samples.
The primary score retains every printed sample. Missing predictions use zero
only as a scoring convention, never as a fabricated waveform. This convention
can reward dropping an erroneous observation. Paired support/error audits are
required for interpreting improvements. Conditional scores and a bounded,
DC-centered alignment are explicitly secondary measurements.
This is not a Kaggle leaderboard metric or a clinical acceptance criterion.
"""

from __future__ import annotations

import math

import numpy as np

from .signals import LEADS


def _snr(power, error):
    if power <= 0:
        return None
    if error <= 0:
        return 120.0
    return float(np.clip(10 * math.log10(power / error), -120.0, 120.0))


def _metrics(x, y):
    if not len(x):
        return None
    error = y - x
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return {
        "samples": len(x),
        "rmse_mv": float(np.sqrt(np.mean(error**2))),
        "bias_mv": float(error.mean()),
        "pcc": float(np.clip(xc @ yc / denominator, -1.0, 1.0))
        if denominator > 0
        else None,
        "snr_db": _snr(float(x @ x), float(error @ error)),
    }


def score_external_signals(
    reference_mv, prediction_mv, fs, *, max_shift_s=0.1, minimum_alignment_coverage=0.9
):
    """Score canonical [12, time] arrays without repairing identity or amplitude.

    The caller places reference and prediction on the same physical time axis.
    Only scoring receives reference signals and printed support. Alignment
    compares reference[t] with prediction[t + shift], removes separate means,
    and fits no amplitude scale. Candidates must cover at least 90% of the
    original printed samples, including losses at shifted boundaries.
    """
    a = np.asarray(reference_mv, dtype=np.float64)
    b = np.asarray(prediction_mv, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != 12 or a.shape != b.shape or not a.shape[1]:
        raise ValueError("Matching nonempty [12, time] arrays required")
    if np.isinf(a).any() or np.isinf(b).any() or not np.isfinite(a).any():
        raise ValueError("Reference/prediction may contain NaN, but no infinity")
    if isinstance(fs, bool) or not math.isfinite(fs) or fs <= 0:
        raise ValueError("Positive finite sample rate required")
    if not math.isfinite(max_shift_s) or not 0 <= max_shift_s <= 1:
        raise ValueError("Alignment radius must be between zero and one second")
    if not 0 < minimum_alignment_coverage <= 1:
        raise ValueError("Alignment coverage must be in (0, 1]")
    per_lead = {}
    all_x, all_y, observed_x, observed_y = [], [], [], []
    covered_total = expected_total = unexpected_total = 0
    n = a.shape[1]
    radius = min(round(max_shift_s * fs), n - 1)
    for name, truth, pred in zip(LEADS, a, b):
        printed = np.isfinite(truth)
        observed = printed & np.isfinite(pred)
        expected, covered = int(printed.sum()), int(observed.sum())
        unexpected = int((~printed & np.isfinite(pred)).sum())
        expected_total += expected
        covered_total += covered
        unexpected_total += unexpected
        x = truth[printed]
        y = np.where(np.isfinite(pred[printed]), pred[printed], 0.0)
        all_x.append(x)
        all_y.append(y)
        observed_x.append(truth[observed])
        observed_y.append(pred[observed])
        best = None
        # Every candidate is checked against ORIGINAL printed support. A shift
        # cannot make a missing lead disappear from the coverage denominator.
        for shift in range(-radius, radius + 1):
            shifted = np.full(n, np.nan)
            if shift >= 0:
                shifted[: n - shift] = pred[shift:]
            else:
                shifted[-shift:] = pred[: n + shift]
            keep = printed & np.isfinite(shifted)
            count = int(keep.sum())
            if count < max(3, math.ceil(expected * minimum_alignment_coverage)):
                continue
            xx, yy = truth[keep], shifted[keep]
            xx, yy = xx - xx.mean(), yy - yy.mean()
            result = _metrics(xx, yy)
            if result["snr_db"] is None:
                continue
            candidate = dict(
                result,
                shift_samples=shift,
                shift_ms=1000 * shift / fs,
                coverage=count / expected,
            )
            if best is None or (candidate["snr_db"], -abs(shift), -shift) > (
                best["snr_db"],
                -abs(best["shift_samples"]),
                -best["shift_samples"],
            ):
                best = candidate
        per_lead[name] = {
            "printed_samples": expected,
            "observed_samples": covered,
            "coverage": covered / expected if expected else None,
            "unsupported_output_samples": unexpected,
            "raw_observed": _metrics(truth[observed], pred[observed]),
            "raw_all_printed_zero_missing": _metrics(x, y),
            "aligned_observed_dc_centered": best,
        }
    return {
        "schema": "ecg-external-signal-scores-v1",
        "missing_data_warning": "The zero-missing scalar error can improve when observations are dropped. Inspect paired support loss/gain and common-support error before claiming improvement.",
        "fs": float(fs),
        "printed_samples": expected_total,
        "observed_samples": covered_total,
        "coverage": covered_total / expected_total,
        "unsupported_output_samples": unexpected_total,
        "raw_observed": _metrics(
            np.concatenate(observed_x), np.concatenate(observed_y)
        ),
        "raw_all_printed_zero_missing": _metrics(
            np.concatenate(all_x), np.concatenate(all_y)
        ),
        "leads_expected": sum(v["printed_samples"] > 0 for v in per_lead.values()),
        "leads_entirely_missing": sum(
            v["printed_samples"] > 0 and v["observed_samples"] == 0
            for v in per_lead.values()
        ),
        "leads_below_90_percent_coverage": sum(
            v["coverage"] is not None and v["coverage"] < 0.9 for v in per_lead.values()
        ),
        "alignment": {
            "max_shift_s": max_shift_s,
            "minimum_original_printed_coverage": minimum_alignment_coverage,
            "dc_centered": True,
            "amplitude_fit": False,
        },
        "snr_numeric_cap_db": [-120.0, 120.0],
        "per_lead": per_lead,
    }
