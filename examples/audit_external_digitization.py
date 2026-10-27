"""Independently recount external waveform artifacts without calling the scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(manifest, run, out):
    manifest, run = Path(manifest), Path(run)
    data = json.loads(manifest.read_text())
    report = json.loads((run / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("The complete scheduled run is required")
    if report["manifest_sha256"] != digest(manifest):
        raise ValueError("Manifest hash mismatch")
    cases = {r["image_id"]: r for r in data["cases"]}
    scheduled = {(key, arm) for key in cases for arm in report["arms"]}
    found = {(r["image_id"], r["arm"]) for r in report["cases"]}
    if scheduled != found or len(found) != len(report["cases"]):
        raise ValueError("Missing, duplicated or unexpected scheduled result")
    checks = 0
    decoded_predictions = {}
    cache_replays = 0
    for record in report["cases"]:
        row = cases[record["image_id"]]
        reference = manifest.parent / row["reference"]
        if digest(reference) != row["reference_sha256"]:
            raise ValueError("Reference hash mismatch")
        with np.load(reference, allow_pickle=False) as z:
            truth = z["values_mv"]
        if record["status"] == "failed":
            pred = np.full(truth.shape, np.nan)
        else:
            path = run / record["prediction"]
            if digest(path) != record["prediction_sha256"]:
                raise ValueError("Prediction hash mismatch")
            with np.load(path, allow_pickle=False) as z:
                pred = z["values_mv"]
        if truth.shape != pred.shape or np.isinf(pred).any():
            raise ValueError("Malformed saved prediction")
        if record["status"] != "failed":
            arm_root = run / record["image_id"] / record["arm"]
            if record["arm"] == "npj_native":
                input_path = manifest.parent / row["image"]
            elif record["arm"] == "text_then_npj":
                input_path = arm_root / "oriented-original-rgb.png"
            else:
                input_path = (
                    arm_root / "rectification" / record["rectification"]["output_image"]
                )
            with Image.open(input_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            key = (rgb.shape, hashlib.sha256(rgb.tobytes()).hexdigest())
            assert list(rgb.shape) == record["input_shape"]
            assert key[1] == record["decoded_rgb_sha256"]
            if record["exact_rgb_prediction_cache_hit"]:
                assert key in decoded_predictions
                assert np.array_equal(decoded_predictions[key], pred, equal_nan=True)
                cache_replays += 1
            else:
                decoded_predictions[key] = pred.copy()
            checks += 2
        expected = np.isfinite(truth)
        covered = expected & np.isfinite(pred)
        e = record["evaluation"]
        assert int(expected.sum()) == e["printed_samples"]
        assert int(covered.sum()) == e["observed_samples"]
        assert np.isclose(covered.sum() / expected.sum(), e["coverage"], atol=1e-12)
        penalty = np.where(np.isfinite(pred[expected]), pred[expected], 0.0)
        error = penalty - truth[expected]
        assert np.isclose(
            np.sqrt(np.mean(error**2)),
            e["raw_all_printed_zero_missing"]["rmse_mv"],
            atol=1e-12,
        )
        checks += 4
        order = (
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        )
        for i, name in enumerate(order):
            lead = e["per_lead"][name]
            assert lead["printed_samples"] == int(expected[i].sum())
            assert lead["observed_samples"] == int(covered[i].sum())
            if covered[i].any():
                difference = pred[i, covered[i]] - truth[i, covered[i]]
                assert np.isclose(
                    np.sqrt(np.mean(difference**2)),
                    lead["raw_observed"]["rmse_mv"],
                    atol=1e-12,
                )
                checks += 1
            checks += 2
    result = {
        "schema": "external-m7-independent-audit-v1",
        "passed": True,
        "case_method_results": len(found),
        "images": len(cases),
        "patients": data["patients"],
        "recount_checks": checks,
        "exact_rgb_cache_replays_verified": cache_replays,
        "manifest_sha256": digest(manifest),
        "report_sha256": digest(run / "report.json"),
        "scorer_imported": False,
        "no_scheduled_failure_dropped": True,
    }
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(**vars(parser.parse_args()))
