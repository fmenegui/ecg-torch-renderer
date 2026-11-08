"""Reference-assisted time-axis diagnosis, never an inference correction.

Search a wider non-circular lag window after all predictions are frozen.
No waveform, amplitude, time axis, selection policy or primary score is changed.
Periodic rhythms can produce competing peaks, which remain explicitly visible.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags


def lag_diagnostic(reference, prediction, fs, *, maximum_lag_s=.5):
    x, y = np.asarray(reference, float), np.asarray(prediction, float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3:
        raise ValueError("Matching one-dimensional physical time axes required")
    if isinstance(fs, bool) or not math.isfinite(fs) or fs <= 0 or not 0 < maximum_lag_s <= 1:
        raise ValueError("Positive physical sampling metadata and a bounded lag window required")
    a, b = np.isfinite(x), np.isfinite(y)
    expected = int(a.sum())
    observed = int((a & b).sum())
    out = {"reference_assisted_diagnostic": True, "predictions_changed": False,
           "printed_samples": expected, "observed_samples": observed,
           "coverage": observed / expected if expected else 0., "eligible": False}
    if not expected or out["coverage"] < .9:
        return dict(out, reason="below_90_percent_printed_support")
    xx, yy = np.where(a, x, 0), np.where(b, y, 0)
    a, b = a.astype(float), b.astype(float)
    lags = correlation_lags(len(y), len(x), mode="full")
    keep = np.abs(lags) <= round(maximum_lag_s * fs)

    def cross(left, right):
        return correlate(left, right, mode="full", method="fft")[keep]

    n = np.rint(cross(b, a)).clip(0)
    sx, sy = cross(b, xx), cross(yy, a)
    sxx, syy, sxy = cross(b, xx * xx), cross(yy * yy, a), cross(yy, xx)
    safe_n = np.maximum(n, 1)
    vx, vy = (sxx - sx * sx / safe_n).clip(0), (syy - sy * sy / safe_n).clip(0)
    denominator = np.sqrt(vx * vy)
    valid = (n >= max(100, .8 * expected)) & (denominator > 1e-18)
    if not valid.any():
        return dict(out, reason="insufficient_shared_variation")
    pcc = np.full(len(n), np.nan)
    pcc[valid] = np.clip((sxy[valid] - sx[valid] * sy[valid] / n[valid]) / denominator[valid], -1, 1)
    lags = lags[keep]
    index = min(np.flatnonzero(valid), key=lambda i: (-pcc[i], abs(lags[i]), lags[i]))
    lag = int(lags[index])
    zero = int(np.flatnonzero(lags == 0)[0])
    if lag > 0:
        ref, pred = x[:-lag], y[lag:]
    elif lag < 0:
        ref, pred = x[-lag:], y[:lag]
    else:
        ref, pred = x, y
    paired = np.isfinite(ref) & np.isfinite(pred)
    alternative = valid & (np.abs(lags - lag) > round(.04 * fs))
    next_peak = float(np.nanmax(pcc[alternative])) if alternative.any() else None
    out.update(eligible=True, reason="diagnostic_only", predicted_delay_samples=lag,
        predicted_delay_s=lag / fs, delay_sign="positive means prediction features occur later than reference",
        maximum_lag_s=maximum_lag_s, best_pcc=float(pcc[index]),
        zero_lag_pcc=float(pcc[zero]) if np.isfinite(pcc[zero]) else None,
        samples_at_best_lag=int(paired.sum()), raw_rmse_mv_at_best_lag=float(np.sqrt(np.mean((pred[paired] - ref[paired]) ** 2))),
        competing_peak_pcc_outside_40ms=next_peak,
        competing_peak_gap=float(pcc[index] - next_peak) if next_peak is not None else None,
        amplitude_or_dc_fitted=False, circular_shift_used=False)
    return out


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ["manifest", "sampling", "preserve", "landscape", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    configurations = [("native", args.sampling, "npj_native"), ("adaptive_then_native", args.sampling, "adaptive_sampling"),
                      ("downstream_preserve", args.preserve, "external_downstream"),
                      ("downstream_landscape", args.landscape, "external_downstream")]
    records, hashes = {}, {}
    for label, folder, arm in configurations:
        report = json.loads((folder / "report.json").read_text())
        if report["status"] != "completed" or not (folder / "phase1-decisions.json").exists():
            raise ValueError("All inference and selections must be sealed before this diagnosis")
        records[label] = {c["image_id"]: c for c in report["cases"] if c["arm"] == arm}
        hashes[label] = digest(folder / "report.json")
    rows = []
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for case in manifest["cases"]:
        reference = args.manifest.parent / case["reference"]
        if digest(reference) != case["reference_sha256"]:
            raise ValueError("Frozen scoring reference changed")
        with np.load(reference, allow_pickle=False) as z:
            gold = z["values_mv"]
        for label, folder, _ in configurations:
            record = records[label][case["image_id"]]
            path = folder / record["prediction"]
            if digest(path) != record["prediction_sha256"]:
                raise ValueError("Frozen prediction changed")
            with np.load(path, allow_pickle=False) as z:
                values = z["values_mv"]
            for lead, a, b in zip(leads, gold, values):
                rows.append({"image_id": case["image_id"], "patient_id": case["patient_id"],
                    "method": label, "lead": lead, **lag_diagnostic(a, b, 500)})
    summary = {"schema": "reference-assisted-time-axis-diagnosis-v1", "reference_assisted_diagnostic": True,
        "predictions_changed": False, "primary_scores_changed": False, "selection_changed": False,
        "new_external_validation": False, "report_sha256": hashes, "methods": {},
        "warning": "Conditional diagnostic, not a paper or leaderboard metric. Periodic rhythms can make a large-lag match ambiguous."}
    for label, _, _ in configurations:
        group = [r for r in rows if r["method"] == label and r["eligible"]]
        zero_pcc = [r["zero_lag_pcc"] for r in group if r["zero_lag_pcc"] is not None]
        summary["methods"][label] = {"eligible_leads": len(group), "scheduled_leads": 12 * manifest["images"],
            "median_absolute_delay_ms": float(np.median([abs(r["predicted_delay_s"]) * 1000 for r in group])) if group else None,
            "leads_above_50ms_delay": sum(abs(r["predicted_delay_s"]) > .05 for r in group),
            "median_zero_lag_pcc": float(np.median(zero_pcc)) if zero_pcc else None,
            "median_best_pcc": float(np.median([r["best_pcc"] for r in group])) if group else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.out / "leads.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.out / "leads.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["methods"]))


if __name__ == "__main__":
    main()
