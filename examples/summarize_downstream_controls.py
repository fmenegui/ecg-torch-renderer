"""Combine frozen development arms with a fixed, input-defined paired subset."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.summarize_external_digitization import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ["manifest", "sampling", "preserve", "landscape", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.manifest.read_text())
    configurations = [("native", args.sampling, "npj_native"), ("adaptive_then_native", args.sampling, "adaptive_sampling"),
        ("downstream_preserve", args.preserve, "external_downstream"),
        ("downstream_landscape", args.landscape, "external_downstream"), ("selective", args.landscape, "selective")]
    records = []
    hashes = {}
    for label, source, arm in configurations:
        report = json.loads((source / "report.json").read_text())
        if report["status"] != "completed":
            raise ValueError("Only complete frozen inference runs can enter this comparison")
        hashes[label] = digest(source / "report.json")
        folder = args.out / label
        folder.mkdir()
        for row in [r for r in report["cases"] if r["arm"] == arm]:
            prediction = source / row["prediction"]
            if digest(prediction) != row["prediction_sha256"]:
                raise ValueError("A prediction changed after its inference receipt")
            path = folder / (row["image_id"] + ".npz")
            path.symlink_to(prediction.resolve())
            records.append({**row, "arm": label, "prediction": str(path.relative_to(args.out))})
    combined = {"schema": "ecg-external-pilot-run-v1", "status": "completed", "cpu_max_side": 1500,
        "arms": [c[0] for c in configurations], "cases": records, "component_report_sha256": hashes,
        "new_external_validation": False, "new_inference_in_this_summary": False}
    (args.out / "report.json").write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n")
    for name in ["references"]:
        (args.out / name).symlink_to((args.manifest.parent / name).resolve(), target_is_directory=True)
    (args.out / "manifest.json").write_bytes(args.manifest.read_bytes())
    summarize(args.out / "manifest.json", args.out, args.out / "all-24")
    frozen = json.loads((args.preserve / "frozen-protocol.json").read_text())
    eligible = {c["image_id"] for c in frozen["inputs"] if c["rectified_rgb_sha256"] is not None}
    paired = args.out / "paired-available-meshes"
    paired.mkdir()
    paired_data = {**data, "cases": [c for c in data["cases"] if c["image_id"] in eligible], "images": len(eligible)}
    paired_report = {**combined, "cases": [r for r in records if r["image_id"] in eligible],
                     "subset_defined_by_inputs_before_inference": True}
    for label, _, _ in configurations:
        (paired / label).symlink_to((args.out / label).resolve(), target_is_directory=True)
    (paired / "references").symlink_to((args.manifest.parent / "references").resolve(), target_is_directory=True)
    (paired / "manifest.json").write_text(json.dumps(paired_data, indent=2) + "\n")
    (paired / "report.json").write_text(json.dumps(paired_report, indent=2, allow_nan=False) + "\n")
    summarize(paired / "manifest.json", paired, paired / "summary")
    # Confirm signal arrays, rather than their NPZ container encoding, have
    # remained identical in both reused direct arms.
    for source in [args.preserve, args.landscape]:
        old = json.loads((source / "report.json").read_text())
        for row in [r for r in old["cases"] if r["arm"] == "npj_native"]:
            with np.load(source / row["prediction"], allow_pickle=False) as a, \
                 np.load(args.sampling / row["image_id"] / "npj_native.npz", allow_pickle=False) as b:
                np.testing.assert_array_equal(a["values_mv"], b["values_mv"])
    print(json.dumps({"scheduled_images": data["images"], "input_defined_paired_images": len(eligible),
                      "baseline_signal_arrays_exactly_preserved": True, "primary_all_24_denominator_retained": True}))


if __name__ == "__main__":
    main()
