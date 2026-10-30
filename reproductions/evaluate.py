"""Evaluate every scheduled case after inference, with explicit printed support.

Reference NPZ: values[12,T], printed[12,T], unit is specified by manifest.
No resizing of time axes or estimation of gain from reference data occurs here.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from reproductions.common import new_run, sha256, write_json
from reproductions.manifest import LEADS, UNIT_TO_MV, load_manifest
from reproductions.metrics import patient_macro, score_lead


def kaggle_timeline(series):
    series = np.asarray(series, dtype=float)
    if series.ndim != 2 or series.shape[0] != 4 or series.shape[1] % 4:
        raise ValueError("Four native rows and an exactly divisible time axis required")
    values = np.full((12, series.shape[1]), np.nan)
    layout = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
    quarter = series.shape[1] // 4
    for row, names in enumerate(layout):
        for column, lead in enumerate(names):
            values[LEADS.index(lead), column * quarter:(column + 1) * quarter] = series[row, column * quarter:(column + 1) * quarter]
    values[LEADS.index("II")] = series[3]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", choices=["npj", "kaggle-second"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    _, cases = load_manifest(args.manifest)
    root = Path(args.manifest).resolve().parent
    predictions = Path(args.predictions).resolve()
    native_receipt = json.loads((predictions / "receipt.json").read_text())
    if args.method == "npj":
        identities = {r["case_id"]: r["image_sha256"] for r in native_receipt["results"]}
    else:
        identities = native_receipt["images_sha256"]
    out = new_run(args.out)
    records = []
    for case in cases:
        reference_file = np.load(root / case["reference"], allow_pickle=False)
        reference = np.asarray(reference_file["values"], float) * UNIT_TO_MV[case["reference_unit"]]
        printed = np.asarray(reference_file["printed"], bool)
        if reference.ndim != 2 or reference.shape[0] != 12 or printed.shape != reference.shape:
            raise ValueError("Reference requires canonical lead order and explicit printed support")
        status, error = "returned", None
        try:
            if identities.get(case["case_id"]) != case["image_sha256"]:
                raise ValueError("No receipt linking this input image to the prediction")
            if args.method == "npj":
                file = np.load(predictions / (case["case_id"] + ".npz"), allow_pickle=False)
                unit = str(file["unit"])
                if unit != "uV":
                    raise ValueError("Unexpected native npj unit")
                prediction = np.asarray(file["values"], float) * UNIT_TO_MV[unit]
            else:
                prediction = kaggle_timeline(np.load(predictions / "native/digitalised" / (case["case_id"] + ".series.npy"), allow_pickle=False))
            if prediction.shape != reference.shape:
                raise ValueError("Incompatible time axes. Explicit source preparation is required.")
        except (OSError, ValueError, KeyError) as exception:
            prediction = np.full_like(reference, np.nan)
            status, error = "missing_or_incompatible", str(exception)
        leads = []
        for index, name in enumerate(LEADS):
            if not printed[index].any():
                leads.append({"lead": name, "status": "not_printed"})
                continue
            leads.append({"lead": name, "status": status, **score_lead(reference[index], prediction[index], printed[index], fs=case["fs"], align_ms=100, center=True)})
        total = printed.sum()
        observed = printed & np.isfinite(prediction)
        records.append({"case_id": case["case_id"], "patient_id": case["patient_id"], "acquisition": case["acquisition"], "status": status, "error": error, "coverage": float(observed.sum() / total) if total else 0, "leads": leads})
    write_json(out / "case-results.json", records)
    write_json(out / "summary.json", {"schema": "audited-common-evaluation-v1", "method": args.method, "manifest_sha256": sha256(args.manifest), "native_receipt_sha256": sha256(predictions / "receipt.json"), "scheduled_images": len(cases), "retained_images": len(records), "coverage": patient_macro(records, "coverage"), "same_as_native_paper_or_kaggle_metric": False, "paper_result_reproduced": False})


if __name__ == "__main__":
    main()
