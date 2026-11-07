"""Isolate a second native geometry fit on previously inspected M7 images.

The direct and adaptive+native signal arrays are reused exactly from the paired
1500-pixel development study. Only the separately named downstream arm is
new inference. References enter scoring after all decisions have been sealed.
"""

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.external_evaluation import score_external_signals
from ecg_gpu.digitization.npj_backend import NativeNpjBackend, canonical_layout
from ecg_gpu.digitization.rectified_backend import RectifiedNpjBackend
from ecg_gpu.digitization.oriented_rectified_backend import LandscapeRectifiedNpjBackend
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.digitization.selective import GeometryEvidence, SelectionPolicy, SignalProposal, select_proposal
from examples.summarize_external_digitization import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ["manifest", "paired-sampling", "prior-meshes", "upstream", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--orientation", choices=["preserve", "landscape"], default="preserve")
    args = p.parse_args()
    out, prior = args.out, args.paired_sampling
    out.mkdir(parents=True, exist_ok=False)
    policy = SelectionPolicy()
    document = json.loads(args.manifest.read_text())
    # Only image identity reaches the inference loop. Explicitly discard scores
    # that happen to exist in a previously evaluated baseline report.
    image_only = [{k: c[k] for k in ["image_id", "image_sha256"]} for c in document["cases"]]
    old = json.loads((prior / "report.json").read_text())
    safe_keys = ["image_id", "arm", "status", "prediction", "prediction_sha256", "predicted_layout", "layout_fallback", "metadata", "inference_wall_seconds"]
    baselines = {(c["image_id"], c["arm"]): {k: c[k] for k in safe_keys if k in c} for c in old["cases"]}
    del old
    samples = []
    for c in image_only:
        folder = prior / c["image_id"]
        image, sampling = folder / "adaptive-original-rgb.png", folder / "sampling.json"
        samples.append({"image_id": c["image_id"], "original_image_sha256": c["image_sha256"],
            "rectified_rgb_sha256": digest(image) if image.exists() else None,
            "sampling_receipt_sha256": digest(sampling)})
    _write_json(out / "frozen-protocol.json", {"schema": "external-downstream-development-v1",
        "development_only": True, "new_external_cohort": False, "native_max_side": 1500, "device": args.device,
        "policy": asdict(policy), "policy_sha256": policy.sha256,
        "manifest_sha256": digest(args.manifest), "paired_report_sha256": digest(prior / "report.json"),
        "downstream_source_sha256": digest(Path(__file__).resolve().parents[1] / "ecg_gpu/digitization/rectified_backend.py"),
        "orientation_control": args.orientation,
        "landscape_control_source_sha256": digest(Path(__file__).resolve().parents[1] / "ecg_gpu/digitization/oriented_rectified_backend.py") if args.orientation == "landscape" else None,
        "baseline_arms_reused_without_modification": True, "new_geometry_or_grid_training": False,
        "npz_container_byte_identity_claimed": False,
        "changes": ["omit second perspective/crop/TPS", "preserve orientation" if args.orientation == "preserve" else "native aspect heuristic as an exact feature-map quarter turn", "use inferred sampler density instead of native grid-pitch search"],
        "separate_causes_not_fully_isolated": "The downstream composition changes geometry and density together. No claim that perspective alone caused a difference.",
        "inputs": samples})
    torch.set_num_threads(2)
    adapter = RectifiedNpjBackend if args.orientation == "preserve" else LandscapeRectifiedNpjBackend
    backend = adapter(NativeNpjBackend(args.upstream, device=args.device, max_side=1500))
    report = {"schema": "ecg-external-pilot-run-v1", "status": "inference", "cpu_max_side": 1500,
              "arms": ["npj_native", "external_then_native", "external_downstream", "selective"], "cases": []}
    decisions = []
    for case in image_only:
        image_id, identity = case["image_id"], case["image_sha256"]
        folder = out / image_id
        folder.mkdir()
        predictions, statuses, seconds = {}, {}, {}
        for arm, old_arm in [("npj_native", "npj_native"), ("external_then_native", "adaptive_sampling")]:
            record = baselines[image_id, old_arm]
            path = prior / record["prediction"]
            if digest(path) != record["prediction_sha256"]:
                raise ValueError("Saved baseline prediction changed")
            with np.load(path, allow_pickle=False) as z:
                predictions[arm] = SignalProposal(z["values_mv"], float(z["fs"]), identity,
                    canonical_layout(record["predicted_layout"]), metadata=record["metadata"])
            statuses[arm], seconds[arm] = record["status"], record["inference_wall_seconds"]
        direct = predictions["npj_native"]
        started, log = time.perf_counter(), io.StringIO()
        with contextlib.redirect_stdout(log):
            try:
                source = prior / image_id / "adaptive-original-rgb.png"
                if not source.exists():
                    raise FileNotFoundError("No previously accepted mesh for this image")
                rgb = np.asarray(Image.open(source).convert("RGB"))
                sampling = json.loads((prior / image_id / "sampling.json").read_text())
                candidate = backend.predict(rgb, fs=500, samples=5000, source_id=identity,
                                            pixels_per_mm=sampling["effective_pixels_per_mm"])
                statuses["external_downstream"] = "returned"
                old_geometry = json.loads((args.prior_meshes / "run" / image_id / "m6_then_npj/rectification/report.json").read_text())
                evidence = GeometryEvidence(identity, True, sampling["source_frame_covered"],
                    sampling["target_boundary_simple"], sampling["triangle_orientation_consistent"],
                    float(old_geometry.get("complete_geometry", {}).get("extrapolated_fraction", 1)))
                selected, decision = select_proposal(direct, candidate, evidence, policy)
            except Exception as error:
                candidate = SignalProposal(np.full((12, 5000), np.nan), 500, identity, None,
                    metadata={"error_type": type(error).__name__, "error": str(error)})
                statuses["external_downstream"] = "failed"
                selected, decision = direct, {"selected": "direct", "reason": "downstream_branch_failed",
                    "error_type": type(error).__name__, "error": str(error), "reference_used": False, "policy_sha256": policy.sha256}
        (folder / "inference.log").write_text(log.getvalue())
        predictions.update(external_downstream=candidate, selective=selected)
        seconds["external_downstream"] = time.perf_counter() - started
        seconds["selective"] = seconds["npj_native"] + seconds["external_downstream"]
        statuses["selective"] = statuses["npj_native"] if selected is direct else statuses["external_downstream"]
        decision["image_id"] = image_id
        decisions.append(decision)
        _write_json(folder / "decision.json", decision)
        for arm, proposal in predictions.items():
            path = folder / (arm + ".npz")
            np.savez_compressed(path, values_mv=proposal.values_mv, fs=proposal.fs)
            report["cases"].append({"image_id": image_id, "arm": arm, "status": statuses[arm],
                "prediction": str(path.relative_to(out)), "prediction_sha256": digest(path),
                "predicted_layout": proposal.metadata.get("native_layout"), "layout_fallback": proposal.metadata.get("layout_fallback"),
                "metadata": proposal.metadata, "inference_wall_seconds": seconds[arm]})
        _write_json(out / "report.json", report)
        print(json.dumps({"image_id": image_id, "status": statuses["external_downstream"],
                          "seconds": seconds["external_downstream"], "selected": decision["selected"], "reason": decision["reason"]}), flush=True)
    _write_json(out / "phase1-decisions.json", {"decisions": decisions, "inference_received_references": False})
    by_image = {c["image_id"]: c for c in document["cases"]}
    for row in report["cases"]:
        reference = args.manifest.parent / by_image[row["image_id"]]["reference"]
        if digest(reference) != by_image[row["image_id"]]["reference_sha256"]:
            raise ValueError("Scoring reference changed")
        with np.load(reference, allow_pickle=False) as gold, np.load(out / row["prediction"], allow_pickle=False) as prediction:
            row["evaluation"] = score_external_signals(gold["values_mv"], prediction["values_mv"], 500)
    report["status"] = "completed"
    _write_json(out / "report.json", report)
    summarize(args.manifest, out, out / "summary")


if __name__ == "__main__":
    main()
