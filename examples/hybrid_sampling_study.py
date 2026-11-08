"""Paired direct/adaptive/selective inference with a frozen M7 geometry bank.

Changes the sampling stage, not learned points or lattice correspondences.
All images are previously inspected development data. The 3000-pixel direct
arm is re-executed, with no reuse of the earlier 1500-pixel predictions.
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
from ecg_gpu.digitization.adaptive_sampling import render_mesh
from ecg_gpu.digitization.external_evaluation import score_external_signals
from ecg_gpu.digitization.npj_backend import NativeNpjBackend
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.digitization.selective import GeometryEvidence, SelectionPolicy, SignalProposal, select_proposal
from examples.summarize_external_digitization import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(manifest, prior, upstream, out, *, max_side=3000, device="cpu"):
    manifest, prior, out = Path(manifest), Path(prior), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    policy = SelectionPolicy()
    protocol = {"schema": "hybrid-sampling-development-v1", "development_only": True,
                "previously_inspected_m7_patients": True, "new_holdout": False,
                "native_max_side": max_side, "adaptive_canvas_max_side": max_side,
                "device": device, "policy": asdict(policy), "policy_sha256": policy.sha256,
                "policy_calibrated_on_known_development_outputs": True,
                "manifest_sha256": digest(manifest), "new_training": False,
                "learned_grid_and_correspondences": "Frozen M7 outputs, not recomputed or gold-corrected",
                "inference_arguments": "RGB, original-derived mesh, public 500 Hz and 5000 samples only"}
    _write_json(out / "frozen-protocol.json", protocol)
    data = json.loads(manifest.read_text())
    image_only = [{k: c[k] for k in ["image_id", "image", "image_sha256"]} for c in data["cases"]]
    _write_json(out / "image-only-manifest.json", image_only)
    for case in image_only:
        if digest(manifest.parent / case["image"]) != case["image_sha256"]:
            raise ValueError("Original image changed")
    torch.set_num_threads(2)
    backend = NativeNpjBackend(upstream, device=device, max_side=max_side)
    report = {"schema": "ecg-external-pilot-run-v1", "status": "inference", "cpu_max_side": max_side,
              "arms": ["npj_native", "adaptive_sampling", "selective"], "cases": [],
              "new_inference": True, "new_external_cohort": False, "protocol_sha256": digest(out / "frozen-protocol.json"),
              "reference_waveforms_opened_for_inference": False}
    decisions = []
    for case in image_only:
        folder = out / case["image_id"]
        folder.mkdir()
        print(json.dumps({"starting": case["image_id"]}), flush=True)
        image = np.asarray(Image.open(manifest.parent / case["image"]).convert("RGB"))
        log = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(log):
            try:
                direct = backend.predict(image, fs=500, samples=5000, source_id=case["image_sha256"])
                direct_status = "returned"
            except Exception as error:
                direct = SignalProposal(np.full((12, 5000), np.nan), 500, case["image_sha256"], None,
                                        metadata={"error_type": type(error).__name__, "error": str(error)})
                direct_status = "failed"
            direct_seconds = time.perf_counter() - started
            started = time.perf_counter()
            geometry = prior / "run" / case["image_id"] / "m6_then_npj/rectification"
            sampling, candidate, candidate_status, candidate_error = {}, direct, direct_status, None
            try:
                original_receipt = json.loads((geometry / "report.json").read_text())
                if original_receipt["input_sha256"] != case["image_sha256"]:
                    raise ValueError("Saved mesh belongs to another image")
                with np.load(geometry / "full-frame-mesh.npz", allow_pickle=False) as z:
                    corrected, _, sampling = render_mesh(image, z["source"], z["target_mm"], z["triangles"], max_side=max_side)
                Image.fromarray(corrected).save(folder / "adaptive-original-rgb.png")
                candidate = backend.predict(corrected, fs=500, samples=5000, source_id=case["image_sha256"])
                candidate_status = "returned"
                extrapolated = original_receipt.get("complete_geometry", {}).get("extrapolated_fraction", 1.0)
                evidence = GeometryEvidence(case["image_sha256"], True, sampling["source_frame_covered"], sampling["target_boundary_simple"], sampling["triangle_orientation_consistent"], float(extrapolated))
                selected, decision = select_proposal(direct, candidate, evidence, policy)
            except Exception as error:
                selected = direct
                candidate_error = type(error).__name__ + ": " + str(error)
                decision = {"selected": "direct", "reason": "adaptive_branch_rejected", "detail": candidate_error,
                            "policy_sha256": policy.sha256, "reference_used": False}
            candidate_seconds = time.perf_counter() - started
        decision.update(image_id=case["image_id"])
        decisions.append(decision)
        _write_json(folder / "decision.json", decision)
        _write_json(folder / "sampling.json", sampling)
        (folder / "inference.log").write_text(log.getvalue())
        for arm, result, status, seconds in [("npj_native", direct, direct_status, direct_seconds), ("adaptive_sampling", candidate, candidate_status, candidate_seconds), ("selective", selected, candidate_status if selected is candidate else direct_status, direct_seconds + candidate_seconds)]:
            path = folder / (arm + ".npz")
            np.savez_compressed(path, values_mv=result.values_mv, fs=result.fs)
            report["cases"].append({"image_id": case["image_id"], "arm": arm, "status": status,
                "prediction": str(path.relative_to(out)), "prediction_sha256": digest(path),
                "predicted_layout": result.metadata.get("native_layout"), "layout_fallback": result.metadata.get("layout_fallback"),
                "inference_wall_seconds": seconds, "metadata": result.metadata,
                "adaptive_branch_error": candidate_error, "direct_alias": bool(result is direct)})
        _write_json(out / "report.json", report)
        print(json.dumps({"completed": case["image_id"], "selected": decision["selected"], "reason": decision["reason"], "direct_seconds": direct_seconds, "candidate_seconds": candidate_seconds}), flush=True)
    _write_json(out / "phase1-decisions.json", {"decisions": decisions, "all_references_unopened": True})
    # Reference access starts only after every inference output and decision is saved.
    by_image = {c["image_id"]: c for c in data["cases"]}
    for row in report["cases"]:
        case = by_image[row["image_id"]]
        ref = manifest.parent / case["reference"]
        if digest(ref) != case["reference_sha256"]:
            raise ValueError("Original reference changed")
        with np.load(ref, allow_pickle=False) as gold, np.load(out / row["prediction"], allow_pickle=False) as prediction:
            row["evaluation"] = score_external_signals(gold["values_mv"], prediction["values_mv"], 500)
    report["status"] = "completed"
    _write_json(out / "report.json", report)
    summarize(manifest, out, out / "summary")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ["manifest", "prior", "upstream", "out"]:
        p.add_argument("--" + arg, required=True)
    p.add_argument("--max-side", type=int, default=3000)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    run(a.manifest, a.prior, a.upstream, a.out, max_side=a.max_side, device=a.device)
