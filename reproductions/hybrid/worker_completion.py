"""Finish deadline-skipped inference with frozen models and image-only inputs.

No references are acquired or scored in this worker. Its outputs are a separate
continuation, preserving the original budget-limited run and all its failures.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from ecg_gpu.digitization.grid_backend import AdaptiveGridRectifier
from ecg_gpu.digitization.npj_backend import NativeNpjBackend
from ecg_gpu.digitization.selective import SelectionPolicy
from examples.prepare_pm_ecg_external import archive_index, fetch_member
from reproductions.common import environment, sha256, write_json
from .worker import infer_images, native_specialist

ROOT = Path(__file__).parent
SOURCE_PHASE1 = "44662fce73eff571506de1c6debcd787ae55f02c35905be8d9119db7f08e76a8"


def validate_inputs(manifest):
    expected = {"schema", "source_workflow_run", "source_commit", "source_phase1_sha256",
                "frozen_cohort_sha256", "policy_sha256", "specialist_images", "repair_images"}
    if set(manifest) != expected or manifest["source_phase1_sha256"] != SOURCE_PHASE1:
        raise ValueError("Exact sealed continuation manifest required")
    if manifest["policy_sha256"] != SelectionPolicy().sha256:
        raise ValueError("The frozen selection policy cannot change")
    if len(manifest["specialist_images"]) != 6 or len(manifest["repair_images"]) != 1:
        raise ValueError("Only the six sealed eligible images and one deadline-skipped image are scheduled")
    cases = manifest["specialist_images"] + manifest["repair_images"]
    for case in cases:
        if set(case) != {"image_id", "image", "image_sha256"}:
            raise ValueError("Continuation inference accepts image-only fields")
        path = Path(case["image"])
        if path.is_absolute() or ".." in path.parts or len(case["image_sha256"]) != 64:
            raise ValueError("Invalid frozen image path or identity")
    if len({c["image_id"] for c in cases}) != 7 or len({c["image_sha256"] for c in cases}) != 7:
        raise ValueError("Scheduled continuation identities must be unique")
    return cases


def scheduled_inputs(manifest, plan):
    cases = validate_inputs(manifest)
    reuse = plan.get("reuse_repair_from_run")
    if reuse is None:
        return cases, True
    if reuse != 41113820943:
        raise ValueError("Only the already verified repair result may be reused")
    return list(manifest["specialist_images"]), False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    args = p.parse_args()
    root = Path(args.workspace)
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    execution = json.loads((root / "execution.json").read_text())
    deadline = execution["deadline"]
    manifest = json.loads((ROOT / "completion-inputs.json").read_text())
    cases, do_repair = scheduled_inputs(manifest, execution["plan"])
    if not torch.cuda.is_available():
        raise RuntimeError("The continuation requires actual CUDA inference")
    torch.set_num_threads(2)
    receipt = {"schema": "hybrid-inference-completion-v1", "status": "started", "environment": environment(),
               "input_manifest_sha256": sha256(ROOT / "completion-inputs.json"), "stages": [],
               "reference_inputs_received": False, "reference_scoring_performed": False,
               "selection_policy_sha256": SelectionPolicy().sha256, "new_model_training": False,
               "original_run_preserved": True, "clinical_validity_established": False}
    shutil.copyfile(ROOT / "completion-inputs.json", out / "frozen-completion-inputs.json")

    def stage(name, command):
        started = time.perf_counter()
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1, min(420, int(deadline - time.time() - 180))))
        receipt["stages"].append({"name": name, "returncode": result.returncode, "seconds": time.perf_counter() - started})
        write_json(out / "receipt.json", receipt)
        if result.returncode:
            raise RuntimeError(name + " failed. See retained stage log")

    write_json(out / "receipt.json", receipt)
    try:
        if do_repair:
            stage("geometry-dependency", [sys.executable, "-m", "pip", "install", "--no-deps", "shapely==2.1.2"])
            if not shutil.which("tesseract"):
                stage("ocr-catalog", ["apt-get", "update"])
                stage("ocr-install", ["apt-get", "install", "-y", "tesseract-ocr"])
            stage("source-bootstrap", [sys.executable, "-m", "reproductions.bootstrap", "--method", "npj", "--dest", str(root / "sources")])
        stage("kaggle-pinned-assets", [sys.executable, "-m", "reproductions.kaggle.fetch_assets", "--dest", str(root / "kaggle-assets")])
        stage("kaggle-compatible-timm", [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(root / "kaggle-timm-compat"), "timm==1.0.22"])
        data = root / "completion-image-only"
        data.mkdir()
        index = {row["name"]: row for row in archive_index(data)}
        def fetch(case):
            path = fetch_member(data, index[case["image"]])
            if sha256(path) != case["image_sha256"]:
                raise ValueError("Scheduled original image identity changed")
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(fetch, cases))
        eligible = list(manifest["specialist_images"])
        if do_repair:
            backend = NativeNpjBackend(root / "sources/npj", device="cuda", max_side=3000)
            rectifier = AdaptiveGridRectifier(root / "grid-checkpoint.pt", device="cuda", max_side=3000)
            repair = infer_images(manifest["repair_images"], data, out / "repair", backend, rectifier,
                                  lambda *args: {}, SelectionPolicy(), deadline)
            for row in repair["cases"]:
                if row["arm"] == "npj_native" and row["status"] == "returned" and row["predicted_layout"] == "standard_3x4_with_r1" and not row["layout_fallback"]:
                    eligible.extend(c for c in manifest["repair_images"] if c["image_id"] == row["image_id"])
            receipt["repair_phase1_sha256"] = sha256(out / "repair/phase1-decisions.json")
            del backend, rectifier
            gc.collect()
            torch.cuda.empty_cache()
        else:
            receipt["reused_repair_workflow_run"] = execution["plan"]["reuse_repair_from_run"]
        write_json(out / "sealed-specialist-inputs.json", eligible)
        if time.time() >= deadline - 180:
            raise TimeoutError("No bounded time remains for the unchanged native specialist")
        started = time.perf_counter()
        returned = native_specialist(root)(eligible, data, out / "kaggle-native", deadline)
        records = []
        for case in eligible:
            path = out / (case["image_id"] + "-kaggle.npz")
            item = returned.get(case["image_id"])
            if item is None:
                records.append({"image_id": case["image_id"], "status": "missing"})
                continue
            proposal, evidence = item
            np.savez_compressed(path, values_mv=proposal.values_mv, fs=proposal.fs)
            records.append({"image_id": case["image_id"], "status": "returned", "prediction": path.name,
                            "prediction_sha256": sha256(path), "metadata": proposal.metadata,
                            "specialist_evidence": asdict(evidence), "source_id": proposal.source_id,
                            "layout_id": proposal.layout_id})
        write_json(out / "phase1-completion-predictions.json", {"schema": "sealed-completion-predictions-v1",
            "reference_used": False, "source_phase1_sha256": SOURCE_PHASE1, "records": records,
            "eligible_inputs_sha256": sha256(out / "sealed-specialist-inputs.json"),
            "batch_seconds": time.perf_counter() - started})
        receipt.update(status="completed_inference_continuation", specialist_images=len(eligible),
            specialist_returned=len(returned), prediction_seal_sha256=sha256(out / "phase1-completion-predictions.json"))
        if len(returned) != len(eligible):
            native = json.loads((out / "kaggle-native/receipt.json").read_text())
            receipt["native_batch_status"] = native["status"]
            if native["status"] == "completed":
                receipt["status"] = "completed_with_native_failures"
            else:
                receipt["status"] = "completed_with_missing_outputs"
                raise RuntimeError("Native specialist did not return every sealed eligible input")
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        if receipt["status"] == "started":
            receipt["status"] = "failed_with_preserved_outputs"
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
