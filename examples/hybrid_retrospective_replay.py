"""Freeze and replay an image-only selection policy on retained M7 outputs.

This is development evidence. It is not a new cohort or a new inference run.
The selector receives only inference arrays/metadata and geometry. Reference
waveforms are opened in a separate phase after every decision is saved.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.adaptive_sampling import validate_mesh
from ecg_gpu.digitization.external_evaluation import score_external_signals
from ecg_gpu.digitization.npj_backend import canonical_layout
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.digitization.selective import GeometryEvidence, SelectionPolicy, SignalProposal, select_proposal
from examples.summarize_external_digitization import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inference_record(record):
    allowed = ("image_id", "patient_id", "recording_id", "condition", "image_sha256", "arm",
               "predicted_layout", "layout_fallback", "status", "prediction", "prediction_sha256", "input_shape")
    return {k: record[k] for k in allowed if k in record}


def proposal(record, root):
    path = root / record["prediction"]
    if digest(path) != record["prediction_sha256"]:
        raise ValueError("Changed saved native prediction")
    with np.load(path, allow_pickle=False) as data:
        values, fs = data["values_mv"], float(data["fs"])
    return SignalProposal(values, fs, record["image_sha256"], canonical_layout(record["predicted_layout"]),
                          metadata={"layout_fallback": record.get("layout_fallback", False), "native_layout": record["predicted_layout"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.measurements), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    policy = SelectionPolicy()
    _write_json(out / "frozen-policy.json", {"policy": asdict(policy), "sha256": policy.sha256,
                "selector_source_sha256": digest(Path(__file__).resolve().parents[1] / "ecg_gpu/digitization/selective.py"),
                "development_only": True, "frozen_before_replay": True, "new_external_cohort": False})
    # Strip reference-derived evaluations at the loader boundary.
    old = json.loads((root / "run/report.json").read_text())
    records = {(r["image_id"], r["arm"]): inference_record(r) for r in old["cases"]}
    old_config = {k: old[k] for k in ["upstream_commit", "our_checkpoint_sha256", "cpu_max_side"]}
    del old
    cases = sorted({image for image, arm in records})
    decisions, saved = [], {}
    for image_id in cases:
        first, second = records[image_id, "npj_native"], records[image_id, "m6_then_npj"]
        direct, candidate = proposal(first, root / "run"), proposal(second, root / "run")
        geometry_root = root / "run" / image_id / "m6_then_npj/rectification"
        geometry = json.loads((geometry_root / "report.json").read_text())
        checks = geometry.get("complete_geometry", {})
        valid, reason = False, None
        try:
            with np.load(geometry_root / "full-frame-mesh.npz", allow_pickle=False) as mesh:
                validate_mesh(mesh["source"], mesh["target_mm"], mesh["triangles"], first["input_shape"])
            valid = True
        except (OSError, ValueError) as error:
            reason = str(error)
        evidence = GeometryEvidence(first["image_sha256"], geometry.get("source_for_rectification") == "original RGB",
            bool(valid and checks.get("boundary_self_intersections_checked")), valid, valid,
            float(checks.get("extrapolated_fraction", 1.0)))
        start = time.perf_counter()
        selected, decision = select_proposal(direct, candidate, evidence, policy)
        decision.update(image_id=image_id, geometry_rejection=reason, selector_wall_seconds=time.perf_counter() - start)
        decisions.append(decision)
        folder = out / image_id
        folder.mkdir()
        for arm, value in [("npj_native", direct), ("m6_then_npj", candidate), ("selective", selected)]:
            np.savez_compressed(folder / (arm + ".npz"), values_mv=value.values_mv, fs=value.fs)
            origin = second if arm == "m6_then_npj" or arm == "selective" and selected is candidate else first
            saved[image_id, arm] = {"prediction": str((folder / (arm + ".npz")).relative_to(out)),
                "prediction_sha256": digest(folder / (arm + ".npz")), "predicted_layout": origin["predicted_layout"],
                "layout_fallback": origin.get("layout_fallback"), "status": origin["status"],
                "inference_wall_seconds": decision["selector_wall_seconds"] if arm == "selective" else 0.0}
    _write_json(out / "phase1-decisions.json", {"schema": "hybrid-reference-free-replay-decisions-v1", "decisions": decisions,
                "reference_waveforms_opened": False, "source_report_sha256": digest(root / "run/report.json")})
    # Evaluation starts only after all predictions and decisions are finalized.
    manifest = root / "inputs/manifest.json"
    data = json.loads(manifest.read_text())
    report = {"schema": "ecg-external-pilot-run-v1", "status": "completed", "arms": ["npj_native", "m6_then_npj", "selective"],
              "cpu_max_side": old_config["cpu_max_side"], "cases": [], "new_inference": False,
              "development_replay": True, "policy_sha256": policy.sha256,
              "phase1_sha256": digest(out / "phase1-decisions.json"), "selection_or_tuning_using_test_signals": False,
              "timing_scope": "Selector replay only. Native and geometric inference costs are excluded."}
    for case in data["cases"]:
        path = manifest.parent / case["reference"]
        if digest(path) != case["reference_sha256"]:
            raise ValueError("Changed saved reference")
        with np.load(path, allow_pickle=False) as z:
            reference = z["values_mv"]
        for arm in report["arms"]:
            row = dict(saved[case["image_id"], arm], image_id=case["image_id"], arm=arm)
            with np.load(out / row["prediction"], allow_pickle=False) as z:
                row["evaluation"] = score_external_signals(reference, z["values_mv"], float(z["fs"]))
            report["cases"].append(row)
    _write_json(out / "report.json", report)
    summary = summarize(manifest, out, out / "summary")
    print(json.dumps({"images": len(cases), "candidate_accepted": sum(d["selected"] == "candidate" for d in decisions), "summary_written": str(out / "summary")}))
    return summary


if __name__ == "__main__":
    main()
