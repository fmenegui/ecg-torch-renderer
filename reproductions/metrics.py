"""Audited fixed-denominator metrics, separate from each author's native scorer.

No amplitude fitting. Shifts use masked padding, never circular wraparound.
Every metric exposes coverage on ORIGINAL printed support.
"""

from __future__ import annotations

import math
import numpy as np


def shifted(values: np.ndarray, lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    if abs(lag) >= len(values):
        return out
    if lag > 0:
        out[lag:] = values[:-lag]
    elif lag < 0:
        out[:lag] = values[-lag:]
    else:
        out[:] = values
    return out


def _quality(reference: np.ndarray, prediction: np.ndarray) -> dict:
    if not len(reference):
        return {"rmse_mv": None, "snr_db": None, "snr_status": "no_support", "pcc": None}
    error_power = float(np.sum((reference - prediction) ** 2))
    signal_power = float(np.sum(reference ** 2))
    if error_power == 0:
        snr, status = math.inf, "perfect"
    elif signal_power == 0:
        snr, status = -math.inf, "zero_reference"
    else:
        snr, status = 10 * math.log10(signal_power / error_power), "finite"
    pcc = None
    if len(reference) > 1 and np.std(reference) > 0 and np.std(prediction) > 0:
        pcc = float(np.corrcoef(reference, prediction)[0, 1])
    return {"rmse_mv": math.sqrt(error_power / len(reference)), "snr_db": snr, "snr_status": status, "pcc": pcc}


def score_lead(reference, prediction, printed=None, fs=1000, align_ms=0, center=False, min_aligned_coverage=0.9) -> dict:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if reference.ndim != 1 or prediction.shape != reference.shape or not len(reference):
        raise ValueError("Equal, nonempty 1D arrays are required, with explicit time coordinates")
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(align_ms) or align_ms < 0:
        raise ValueError("Invalid time convention")
    if not 0 <= min_aligned_coverage <= 1:
        raise ValueError("Invalid aligned support requirement")
    printed = np.ones(len(reference), dtype=bool) if printed is None else np.asarray(printed, dtype=bool)
    if printed.shape != reference.shape or not printed.any():
        raise ValueError("Explicit, nonempty printed support is required")
    if not np.isfinite(reference[printed]).all():
        raise ValueError("Reference is nonfinite on printed support")
    observed = printed & np.isfinite(prediction)
    n_printed = int(printed.sum())
    scored_prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    all_quality = _quality(reference[printed], scored_prediction[printed])
    result = {
        "coverage_definition": "finite returned predictions on original printed times, not physical visibility or confidence",
        "printed_samples": n_printed,
        "returned_samples": int(observed.sum()),
        "coverage": float(observed.sum() / n_printed),
        "entirely_missing": not bool(observed.any()),
        "all_printed_zero_scoring_only": all_quality,
        "observed": _quality(reference[observed], prediction[observed]),
        "aligned": None,
    }
    best = None
    max_lag = min(len(reference) - 1, int(round(fs * align_ms / 1000)))
    # Prefer the smallest absolute shift on equal error.
    for lag in sorted(range(-max_lag, max_lag + 1), key=lambda k: (abs(k), k)):
        candidate = shifted(prediction, lag)
        support = printed & np.isfinite(candidate)
        coverage = float(support.sum() / n_printed)
        if not support.any() or coverage < min_aligned_coverage:
            continue
        y = reference[support].copy()
        p = candidate[support].copy()
        if center:
            y -= y.mean()
            p -= p.mean()
        quality = _quality(y, p)
        # SNR selection, matching the declared objective. No amplitude scaling.
        snr = quality["snr_db"]
        if best is None or snr > best["snr_db"]:
            best = {**quality, "lag_samples": lag, "coverage_on_original_printed": coverage, "dc_centered": bool(center)}
    result["aligned"] = best
    return result


def patient_macro(records: list[dict], field: str) -> dict:
    """Describe missing metrics without silently discarding failed cases."""
    groups = {}
    for record in records:
        groups.setdefault(record["patient_id"], []).append(record.get(field))
    if not groups:
        raise ValueError("No scheduled patients")
    values = {}
    for patient, items in groups.items():
        finite = [float(x) for x in items if x is not None and np.isfinite(x)]
        values[patient] = {"scheduled": len(items), "defined": len(finite), "mean": float(np.mean(finite)) if finite else None}
    means = [v["mean"] for v in values.values() if v["mean"] is not None]
    return {"patients_scheduled": len(groups), "patients_with_defined_metric": len(means), "mean": float(np.mean(means)) if means else None, "per_patient": values}
