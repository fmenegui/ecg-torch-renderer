"""Frozen public cohort, native 3000-pixel GPU arms and reference-free selection.

References and true layouts are excluded from the inference function's input.
Every scheduled image remains in the scoring denominator, including failures.
The original six-model notebook is eligible from predicted layout only.
"""

import argparse
import contextlib
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
import torch

from ecg_gpu.digitization.external_evaluation import score_external_signals
from ecg_gpu.digitization.grid_backend import AdaptiveGridRectifier
from ecg_gpu.digitization.npj_backend import NativeNpjBackend
from ecg_gpu.digitization.selective import SelectionPolicy, SignalProposal, SpecialistEvidence, select_proposal
from examples.prepare_pm_ecg_external import prepare
from examples.summarize_external_digitization import summarize
from reproductions.common import environment, sha256, write_json
from reproductions.evaluate import kaggle_timeline

ROOT = Path(__file__).parent
ARMS = ["npj_native", "grid_adaptive", "kaggle_native_eligible", "selective"]
PUBLIC_AXES = {"fs": 500, "samples": 5000, "origin_s": 0.0, "gain_mm_per_mv": 10.0}


def empty(source_id, error, backend):
    return SignalProposal(np.full((12, 5000), np.nan), 500, source_id, None,
                          metadata={"error": str(error), "backend": backend})


def infer_images(image_only, data, out, backend, rectifier, specialist_batch, policy, deadline):
    """This boundary accepts no reference paths, waveforms, masks or true layout."""
    if any(set(c) != {"image_id", "image", "image_sha256"} for c in image_only):
        raise ValueError("Inference accepts only the frozen image-only interface")
    out.mkdir(parents=True, exist_ok=False)
    records, predictions, geometries, times = [], {}, {}, {}
    for case in image_only:
        name, identity = case["image_id"], case["image_sha256"]
        folder = out / name
        folder.mkdir()
        if sha256(data / case["image"]) != identity:
            raise ValueError("Frozen source photograph changed")
        rgb = np.asarray(Image.open(data / case["image"]).convert("RGB"))
        log, started = io.StringIO(), time.perf_counter()
        with contextlib.redirect_stdout(log):
            try:
                if time.time() > deadline - 240:
                    raise TimeoutError("Bounded GPU budget exhausted before this direct inference")
                direct = backend.predict(rgb, fs=500, samples=5000, source_id=identity)
                direct_status = "returned"
            except Exception as error:
                direct, direct_status = empty(identity, error, "npj-native"), "failed"
        direct_seconds = time.perf_counter() - started
        write_json(folder / "direct-metadata.json", direct.metadata)
        candidate, evidence, geometry_error = None, None, None
        # A deadline may prevent __call__, which normally resets these fields.
        # Never attach a previous image's geometry to a skipped candidate.
        rectifier.last_report = None
        rectifier.last_observations = {}
        started = time.perf_counter()
        with contextlib.redirect_stdout(log):
            try:
                if time.time() > deadline - 240:
                    raise TimeoutError("Bounded GPU budget exhausted before this geometric candidate")
                corrected, evidence = rectifier(rgb, source_id=identity)
                Image.fromarray(corrected).save(folder / "adaptive-original-rgb.png")
                candidate = backend.predict(corrected, fs=500, samples=5000, source_id=identity)
                geometry_status = "returned"
            except Exception as error:
                candidate, geometry_status = empty(identity, error, "grid-adaptive"), "failed"
                geometry_error = type(error).__name__ + ": " + str(error)
        (folder / "inference.log").write_text(log.getvalue())
        geometry_receipt = {"report": rectifier.last_report, "error": geometry_error}
        if getattr(rectifier, "last_observations", None):
            path = folder / "geometry-observations.npz"
            np.savez_compressed(path, **rectifier.last_observations)
            geometry_receipt["observations"] = {"file": path.name, "sha256": sha256(path), "reference_used": False}
        write_json(folder / "geometry.json", geometry_receipt)
        predictions[name] = {"npj_native": (direct, direct_status), "grid_adaptive": (candidate, geometry_status)}
        geometries[name] = evidence
        times[name] = {"npj_native": direct_seconds, "grid_adaptive": time.perf_counter() - started}
        # Save every result immediately. These survive a later process failure.
        for arm, (proposal, _) in predictions[name].items():
            np.savez_compressed(folder / (arm + ".npz"), values_mv=proposal.values_mv, fs=proposal.fs)
        print(json.dumps({"image_id": name, "direct": direct_status, "grid": geometry_status,
                          "predicted_layout": direct.layout_id}), flush=True)
    eligible = [c for c in image_only if predictions[c["image_id"]]["npj_native"][0].layout_id == "3x4+1R"
                and not predictions[c["image_id"]]["npj_native"][0].metadata.get("layout_fallback")]
    specialist_started = time.perf_counter()
    specialist_results, specialist_error = {}, None
    if eligible:
        try:
            if time.time() > deadline - 300:
                raise TimeoutError("Insufficient bounded time for the original six-model batch")
            torch.cuda.empty_cache()
            specialist_results = specialist_batch(eligible, data, out / "kaggle-native", deadline)
        except Exception as error:
            specialist_error = type(error).__name__ + ": " + str(error)
    specialist_seconds = time.perf_counter() - specialist_started
    decisions = []
    eligible_ids = {c["image_id"] for c in eligible}
    for case in image_only:
        name, identity = case["image_id"], case["image_sha256"]
        direct, direct_status = predictions[name]["npj_native"]
        candidate, candidate_status = predictions[name]["grid_adaptive"]
        specialist, specialist_evidence = specialist_results.get(name, (None, None))
        if specialist is None:
            reason = specialist_error or ("Native notebook failed or returned no observed series" if name in eligible_ids
                                           else "Unsupported inferred layout. No reference layout used")
            specialist = empty(identity, reason, "kaggle-native-eligible")
            specialist_status = "failed" if name in eligible_ids else "ineligible"
        else:
            specialist_status = "returned"
        selected, decision, selected_status = direct, {"selected": "direct", "reason": "no_accepted_candidate",
            "policy_sha256": policy.sha256, "reference_used": False}, direct_status
        if specialist_evidence is not None:
            selected, decision = select_proposal(direct, specialist, specialist_evidence, policy)
            if selected is specialist:
                selected_status = specialist_status
        if selected is direct:
            specialist_decision = decision
            if geometries[name] is not None and candidate_status == "returned":
                selected, decision = select_proposal(direct, candidate, geometries[name], policy)
                selected_status = candidate_status if selected is candidate else direct_status
                decision["specialist_decision"] = specialist_decision
            else:
                decision["geometry_branch_error"] = candidate.metadata.get("error")
        decision.update(image_id=name, source_id=identity, specialist_eligible=name in eligible_ids,
                        specialist_error=specialist_error)
        decisions.append(decision)
        write_json(out / name / "decision.json", decision)
        rows = {**predictions[name], "kaggle_native_eligible": (specialist, specialist_status),
                "selective": (selected, selected_status)}
        for arm, (proposal, status) in rows.items():
            path = out / name / (arm + ".npz")
            np.savez_compressed(path, values_mv=proposal.values_mv, fs=proposal.fs)
            seconds = times[name].get(arm, specialist_seconds / max(1, len(eligible)))
            if arm == "kaggle_native_eligible" and name not in eligible_ids:
                seconds = 0.0
            if arm == "selective":
                seconds = sum(times[name].values()) + specialist_seconds / max(1, len(eligible)) if name in eligible_ids else sum(times[name].values())
            records.append({"image_id": name, "arm": arm, "status": status, "prediction": str(path.relative_to(out)),
                "prediction_sha256": sha256(path), "inference_wall_seconds": seconds,
                "timing_basis": "native per-image inference, geometry per-image branch, specialist amortized eligible batch, selective all measured branches",
                "predicted_layout": proposal.metadata.get("native_layout"),
                "layout_fallback": proposal.metadata.get("layout_fallback"), "backend_metadata": proposal.metadata})
    write_json(out / "phase1-decisions.json", {"schema": "hybrid-blinded-inference-v1", "decisions": decisions,
        "inference_received_reference_paths": False, "inference_received_true_layout": False,
        "inference_received_reference_waveforms_or_printed_masks": False, "policy_sha256": policy.sha256,
        "specialist_eligible_images": len(eligible), "specialist_error": specialist_error})
    report = {"schema": "ecg-external-pilot-run-v1", "status": "inference_complete_unscored", "cpu_max_side": 3000,
        "inference_device": "cuda", "native_max_side": 3000, "arms": ARMS, "cases": records}
    write_json(out / "report.json", report)
    return report


def native_specialist(root):
    def replay(cases, data, out, deadline):
        import pandas as pd
        source = root / "specialist-image-only"
        (source / "test").mkdir(parents=True)
        rows = []
        for case in cases:
            Image.open(data / case["image"]).convert("RGB").save(source / "test" / (case["image_id"] + ".png"))
            rows.extend({"id": case["image_id"], "lead": lead, "fs": 500,
                         "number_of_rows": 5000 if lead == "II" else 1250}
                        for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])
        pd.DataFrame(rows).to_csv(source / "test.csv", index=False)
        remaining = max(1, min(900, int(deadline - time.time() - 150)))
        env = dict(os.environ, PYTHONPATH=str(root / "kaggle-timm-compat") + os.pathsep + str(Path.cwd()))
        command = [sys.executable, "-m", "reproductions.kaggle.replay", "--notebook", str(root / "kaggle-assets/notebook.api.json"),
            "--assets", str(root / "kaggle-assets"), "--assets-lock", str(root / "kaggle-assets/assets.lock.json"),
            "--competition-data", str(source), "--out", str(out), "--timeout-seconds", str(remaining)]
        with (out.parent / "kaggle-replay.log").open("w") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=remaining + 20)
        receipt_path = out / "receipt.json"
        if not receipt_path.exists():
            raise RuntimeError("Native notebook produced no provenance receipt")
        receipt = json.loads(receipt_path.read_text())
        lock_digest = sha256(root / "kaggle-assets/assets.lock.json")
        if receipt.get("assets_lock_sha256") != lock_digest:
            raise ValueError("Native notebook receipt has a different asset lock")
        for case in cases:
            if receipt.get("images_sha256", {}).get(case["image_id"]) != sha256(source / "test" / (case["image_id"] + ".png")):
                raise ValueError("Native notebook receipt has a different staged original RGB image")
        returned = {}
        for case in cases:
            path = out / "native/digitalised" / (case["image_id"] + ".series.npy")
            if not path.exists():
                continue
            values = kaggle_timeline(np.load(path, allow_pickle=False))
            if values.shape != (12, 5000):
                continue
            identity = case["image_sha256"]
            proposal = SignalProposal(values, 500, identity, "3x4+1R", metadata={"backend": "kaggle-second-final-six",
                "native_layout": "standard_3x4_with_r1", "layout_fallback": False, "layout_is_profile_assumption": True,
                "public_time_axes": PUBLIC_AXES, "assets_lock_sha256": lock_digest,
                "notebook_batch_returncode": result.returncode, "native_series_sha256": sha256(path)})
            returned[case["image_id"]] = (proposal, SpecialistEvidence(identity, "kaggle-second-final-six", lock_digest))
        return returned
    return replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    args = p.parse_args()
    root, policy = Path(args.workspace), SelectionPolicy()
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    execution = json.loads((root / "execution.json").read_text())
    deadline = execution["deadline"]
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if protocol["selection_policy"] != asdict(policy) or protocol["public_axes"] != PUBLIC_AXES:
        raise ValueError("Executable policy differs from the prospective frozen protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires real CUDA inference")
    torch.set_num_threads(2)
    receipt = {"schema": "hybrid-frozen-cohort-gpu-v1", "environment": environment(), "stages": [],
               "protocol_sha256": sha256(ROOT / "protocol.json"), "frozen_cohort_sha256": sha256(ROOT / "frozen-cohort.json"),
               "status": "started", "clinical_validity_established": False, "world_best_established": False}
    shutil.copyfile(ROOT / "protocol.json", out / "frozen-protocol.json")
    shutil.copyfile(ROOT / "frozen-cohort.json", out / "frozen-cohort.json")

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
        stage("geometry-dependency", [sys.executable, "-m", "pip", "install", "--no-deps", "shapely==2.1.2"])
        if not shutil.which("tesseract"):
            stage("ocr-catalog", ["apt-get", "update"])
            stage("ocr-install", ["apt-get", "install", "-y", "tesseract-ocr"])
        stage("source-bootstrap", [sys.executable, "-m", "reproductions.bootstrap", "--method", "npj", "--dest", str(root / "sources")])
        stage("kaggle-pinned-assets", [sys.executable, "-m", "reproductions.kaggle.fetch_assets", "--dest", str(root / "kaggle-assets")])
        stage("kaggle-compatible-timm", [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(root / "kaggle-timm-compat"), "timm==1.0.22"])
        data = root / "frozen-public-inputs"
        data.mkdir()
        csv = data / "ptbxl_database.csv"
        with urllib.request.urlopen("https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/ptbxl_database.csv", timeout=60) as response:
            csv.write_bytes(response.read())
        if sha256(csv) != protocol["ptbxl_metadata_sha256"]:
            raise ValueError("Pinned publisher patient metadata changed")
        manifest = prepare(data, csv, [ROOT / "selection-exclusions.json"], per_layout=2, fetch_missing=True)
        if sha256(data / "manifest.json") != sha256(ROOT / "frozen-cohort.json"):
            raise ValueError("Reacquired public cohort differs from the frozen pre-inference manifest")
        shutil.copytree(data / "references", out / "references")
        image_only = [{k: c[k] for k in ["image_id", "image", "image_sha256"]} for c in manifest["cases"]]
        write_json(out / "image-only-manifest.json", image_only)
        backend = NativeNpjBackend(root / "sources/npj", device="cuda", max_side=3000)
        rectifier = AdaptiveGridRectifier(root / "grid-checkpoint.pt", device="cuda", max_side=3000)
        report = infer_images(image_only, data, out / "run", backend, rectifier, native_specialist(root), policy, deadline)
        # The reference waveforms enter scoring only after every selection is sealed.
        phase1_digest = sha256(out / "run/phase1-decisions.json")
        by_id = {c["image_id"]: c for c in manifest["cases"]}
        for row in report["cases"]:
            case = by_id[row["image_id"]]
            ref = data / case["reference"]
            if sha256(ref) != case["reference_sha256"]:
                raise ValueError("Scoring reference changed")
            with np.load(ref, allow_pickle=False) as gold, np.load(out / "run" / row["prediction"], allow_pickle=False) as prediction:
                row["evaluation"] = score_external_signals(gold["values_mv"], prediction["values_mv"], 500)
        report["status"] = "completed"
        write_json(out / "run/report.json", report)
        summary = summarize(data / "manifest.json", out / "run", out / "summary")
        receipt.update(status="completed_frozen_pilot", phase1_decisions_sha256=phase1_digest,
                       scheduled_images=len(image_only), scheduled_patient_count=manifest["patients"],
                       patient_macro={a: s["macro"] for a, s in summary["arms"].items()},
                       note="Kaggle eligible arm retains unsupported images as NaN, never drops them from the denominator")
    except BaseException as error:
        receipt.update(status="failed_with_preserved_outputs", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
