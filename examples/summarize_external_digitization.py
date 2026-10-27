"""Patient-balanced external-pilot summaries and paired descriptive intervals."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def summarize(manifest, run, out):
    manifest, run, out = Path(manifest), Path(run), Path(out)
    data = json.loads(manifest.read_text())
    report = json.loads((run / "report.json").read_text())
    expected = {(r["image_id"], a) for r in data["cases"] for a in report["arms"]}
    found = {(r["image_id"], r["arm"]) for r in report["cases"]}
    if (
        report["status"] != "completed"
        or found != expected
        or len(found) != len(report["cases"])
    ):
        raise ValueError("Complete unique scheduled results are required")
    out.mkdir(parents=True, exist_ok=True)
    cases = {r["image_id"]: r for r in data["cases"]}
    rows = []
    expected_layout = {
        "3x4": "standard_3x4",
        "3x4+1R": "standard_3x4_with_r1",
        "6x2": "standard_6x2",
        "12x1": "standard_12x1",
    }
    for record in report["cases"]:
        case = cases[record["image_id"]]
        with np.load(manifest.parent / case["reference"], allow_pickle=False) as z:
            truth = z["values_mv"]
        rms = float(np.sqrt(np.mean(truth[np.isfinite(truth)] ** 2)))
        e = record["evaluation"]
        aligned = [
            v["aligned_observed_dc_centered"]
            for v in e["per_lead"].values()
            if v["aligned_observed_dc_centered"] is not None
        ]
        rows.append(
            {
                "image_id": record["image_id"],
                "patient_id": case["patient_id"],
                "recording_id": case["recording_id"],
                "condition": case["condition"],
                "layout": case["reference_layout_scoring_only"],
                "arm": record["arm"],
                "status": record["status"],
                "predicted_layout": record.get("predicted_layout"),
                "compatible_layout_label": record.get("predicted_layout")
                == expected_layout[case["reference_layout_scoring_only"]],
                "layout_fallback": record.get("layout_fallback"),
                "coverage": e["coverage"],
                "raw_all_printed_rmse_mv": e["raw_all_printed_zero_missing"]["rmse_mv"],
                "raw_all_printed_nrmse": e["raw_all_printed_zero_missing"]["rmse_mv"]
                / rms,
                "raw_all_printed_snr_db": e["raw_all_printed_zero_missing"]["snr_db"],
                "raw_observed_rmse_mv": e["raw_observed"]["rmse_mv"]
                if e["raw_observed"]
                else None,
                "raw_observed_pcc": e["raw_observed"]["pcc"]
                if e["raw_observed"]
                else None,
                "mean_eligible_aligned_lead_snr_db": float(
                    np.mean([v["snr_db"] for v in aligned])
                )
                if aligned
                else None,
                "mean_eligible_aligned_lead_rmse_mv": float(
                    np.mean([v["rmse_mv"] for v in aligned])
                )
                if aligned
                else None,
                "aligned_leads": len(aligned),
                "leads_entirely_missing": e["leads_entirely_missing"],
                "leads_below_90_percent_coverage": e["leads_below_90_percent_coverage"],
                "leads_expected": e["leads_expected"],
                "exact_rgb_cache_hit": record.get(
                    "exact_rgb_prediction_cache_hit", False
                ),
                "wall_seconds": record["inference_wall_seconds"],
                "m6_resampled": record.get("rectification", {})
                .get("complete_geometry", {})
                .get("resampled"),
            }
        )
    numeric = (
        "coverage",
        "raw_all_printed_nrmse",
        "raw_all_printed_rmse_mv",
        "raw_all_printed_snr_db",
        "raw_observed_rmse_mv",
        "raw_observed_pcc",
        "mean_eligible_aligned_lead_snr_db",
        "mean_eligible_aligned_lead_rmse_mv",
    )

    def group_summary(group):
        patient = {}
        for p in sorted({r["patient_id"] for r in group}):
            selected = [r for r in group if r["patient_id"] == p]
            patient[p] = {
                k: float(np.mean([r[k] for r in selected if r[k] is not None]))
                if any(r[k] is not None for r in selected)
                else None
                for k in numeric
            }
        return {
            "images": len(group),
            "patients": len(patient),
            "macro": {
                k: float(np.mean([p[k] for p in patient.values() if p[k] is not None]))
                if any(p[k] is not None for p in patient.values())
                else None
                for k in numeric
            },
            "patients_with_eligible_aligned_lead": sum(
                p["mean_eligible_aligned_lead_snr_db"] is not None
                for p in patient.values()
            ),
            "exceptions": sum(r["status"] == "failed" for r in group),
            "compatible_layout_labels": sum(
                r["compatible_layout_label"] for r in group
            ),
            "layout_fallbacks": sum(r["layout_fallback"] is True for r in group),
            "empty_returned_images": sum(
                r["status"] != "failed" and r["coverage"] == 0 for r in group
            ),
            "lead_coverage_failures": sum(
                r["leads_below_90_percent_coverage"] for r in group
            ),
            "leads_entirely_missing": sum(r["leads_entirely_missing"] for r in group),
            "expected_leads": sum(r["leads_expected"] for r in group),
            "exact_rgb_cache_hits": sum(r["exact_rgb_cache_hit"] for r in group),
            "m6_resampled_images": sum(r["m6_resampled"] is True for r in group),
            "execution_wall_seconds": sum(r["wall_seconds"] for r in group),
            "per_patient": patient,
        }

    summary = {
        "schema": "external-m7-patient-summary-v1",
        "patients": data["patients"],
        "images": data["images"],
        "case_method_results": len(rows),
        "cpu_max_side": report["cpu_max_side"],
        "arms": {},
        "paired": {},
        "warning": "Descriptive development pilot. Eight patients, unknown upstream training overlap. Alignment metrics are conditional and exclude leads below 90% support.",
    }
    for arm in report["arms"]:
        group = [r for r in rows if r["arm"] == arm]
        summary["arms"][arm] = group_summary(group)
        for stratum in ("condition", "layout"):
            summary["arms"][arm]["by_" + stratum] = {
                key: group_summary([r for r in group if r[stratum] == key])
                for key in sorted({r[stratum] for r in group})
            }
    rng = np.random.default_rng(20261026)
    patients = sorted({r["patient_id"] for r in rows})
    draws = rng.integers(len(patients), size=(20000, len(patients)))
    for first, second in itertools.combinations(report["arms"], 2):
        before = summary["arms"][first]["per_patient"]
        after = summary["arms"][second]["per_patient"]
        comparison = {}
        for metric in ("coverage", "raw_all_printed_nrmse"):
            difference = np.array(
                [after[p][metric] - before[p][metric] for p in patients]
            )
            interval = np.quantile(difference[draws].mean(1), [0.025, 0.975])
            comparison[metric] = {
                "delta": float(difference.mean()),
                "descriptive_patient_bootstrap_95": interval.tolist(),
                "per_patient_delta": dict(zip(patients, difference.tolist())),
            }
        summary["paired"][second + " minus " + first] = comparison
    # Common support compares errors on precisely the same printed samples.
    # It is secondary and cannot replace the full denominator primary scores.
    common_rows = []
    for image_id, case in cases.items():
        with np.load(manifest.parent / case["reference"], allow_pickle=False) as z:
            truth = z["values_mv"]
        predictions = {}
        for record in [r for r in report["cases"] if r["image_id"] == image_id]:
            if record["status"] == "failed":
                values = np.full(truth.shape, np.nan)
            else:
                with np.load(run / record["prediction"], allow_pickle=False) as z:
                    values = z["values_mv"]
            predictions[record["arm"]] = values
        mask = np.isfinite(truth)
        for values in predictions.values():
            mask &= np.isfinite(values)
        common_rows.append(
            {
                "image_id": image_id,
                "patient_id": case["patient_id"],
                "coverage": float(mask.sum() / np.isfinite(truth).sum()),
                "raw_rmse_mv": {
                    arm: float(np.sqrt(np.mean((values[mask] - truth[mask]) ** 2)))
                    if mask.any()
                    else None
                    for arm, values in predictions.items()
                },
            }
        )
    summary["common_support_all_arms"] = common_rows
    common_patients = {}
    for patient in patients:
        group = [r for r in common_rows if r["patient_id"] == patient]
        common_patients[patient] = {
            "coverage": float(np.mean([r["coverage"] for r in group])),
            "raw_rmse_mv": {
                arm: float(
                    np.mean(
                        [
                            r["raw_rmse_mv"][arm]
                            for r in group
                            if r["raw_rmse_mv"][arm] is not None
                        ]
                    )
                )
                if any(r["raw_rmse_mv"][arm] is not None for r in group)
                else None
                for arm in report["arms"]
            },
        }
    summary["common_support_summary"] = {
        "patient_macro_coverage": float(
            np.mean([r["coverage"] for r in common_patients.values()])
        ),
        "patients_with_common_samples": sum(
            r["raw_rmse_mv"][report["arms"][0]] is not None
            for r in common_patients.values()
        ),
        "patient_macro_raw_rmse_mv": {
            arm: float(
                np.mean(
                    [
                        r["raw_rmse_mv"][arm]
                        for r in common_patients.values()
                        if r["raw_rmse_mv"][arm] is not None
                    ]
                )
            )
            if any(r["raw_rmse_mv"][arm] is not None for r in common_patients.values())
            else None
            for arm in report["arms"]
        },
        "per_patient": common_patients,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (out / "cases.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    import csv

    with (out / "cases.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({arm: s["macro"] for arm, s in summary["arms"].items()}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    summarize(**vars(parser.parse_args()))
