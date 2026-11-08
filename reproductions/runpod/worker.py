"""Executed GPU pilot. Every adaptation and failure remains in the output bank."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from reproductions.common import environment, run_bounded, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    root = Path(args.workspace)
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The requested GPU worker has no CUDA")
    receipt = {"schema": "ecg-real-gpu-pilot-v1", "environment": environment(), "stages": [], "full_training_completed": False, "author_headline_result_reproduced": False}

    def save():
        write_json(out / "receipt.json", receipt)

    def stage(name, command, timeout=600, env=None):
        started = time.monotonic()
        row = {"name": name, "status": "started", "command": command}
        receipt["stages"].append(row)
        save()
        print("START", name, flush=True)
        try:
            with (out / (name + ".log")).open("w") as log:
                if env is None:
                    code = run_bounded(command, timeout, log)
                else:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=env).returncode
            row.update(status="completed" if code == 0 else "failed", returncode=code)
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
        row["seconds"] = time.monotonic() - started
        save()
        print("END", name, row["status"], flush=True)
        return row["status"] == "completed"

    py = sys.executable
    stage("source-bootstrap", [py, "-m", "reproductions.bootstrap", "--method", "npj", "--dest", str(root / "sources")])
    upstream = root / "sources/npj"
    fixture = upstream / "test/test_data/data/ecg_data"
    stage("npj-native-gpu-fixture", [py, "-m", "reproductions.npj.infer", "--upstream", str(upstream), "--preset", "generic", "--image", str(fixture / "10_1.png"), "--out", str(out / "npj-native-fixture")])
    smoke = root / "training-engineering-fixtures"
    for split, ids in [("train", ["10_1", "11_1"]), ("val", ["12_1"])]:
        (smoke / split).mkdir(parents=True, exist_ok=True)
        for case in ids:
            for suffix in [".png", "_mask.png"]:
                shutil.copyfile(fixture / (case + suffix), smoke / split / (case + suffix))
    stage("npj-main-gpu-engineering-training", [py, "-m", "reproductions.npj.train", "--upstream", str(upstream), "--profile", "paper_table4", "--train-dir", str(smoke / "train"), "--validation-dir", str(smoke / "val"), "--smoke", "--out", str(out / "npj-main-training")])
    stage("npj-lead-gpu-engineering-training", [py, "-m", "reproductions.npj.train", "--upstream", str(upstream), "--profile", "lead_text_public", "--smoke", "--out", str(out / "npj-lead-training")])
    stage("pm-disclosed-geometry-controls", [py, "-m", "reproductions.powerful_medical.controls", "--out", str(out / "pm-controls")])
    try:
        from reproductions.runpod.pilot_data import prepare
        data = root / "public-development-pilot"
        groups = prepare(data)
        shutil.copytree(data, out / "pilot-inputs")
        for family, cases in groups.items():
            command = [py, "-m", "reproductions.npj.infer", "--upstream", str(upstream), "--preset", "george-moody-2024" if family == "d0" else "ahus", "--out", str(out / ("npj-" + family))]
            if family != "d0":
                command += ["--layout-hint", family]
            for case in cases:
                command += ["--image", str(data / case["image"])]
            stage("npj-public-" + family, command)
            stage("npj-audited-" + family, [py, "-m", "reproductions.evaluate", "--manifest", str(data / (family + ".json")), "--method", "npj", "--predictions", str(out / ("npj-" + family)), "--out", str(out / ("evaluation-npj-" + family))])
        stage("kaggle-asset-download", [py, "-m", "reproductions.kaggle.fetch_assets", "--dest", str(root / "kaggle-assets")])
        compat = root / "kaggle-timm-compat"
        stage("kaggle-compatible-timm", [py, "-m", "pip", "install", "--no-deps", "--target", str(compat), "timm==1.0.22"])
        env = dict(os.environ, PYTHONPATH=str(compat))
        stage("kaggle-native-six-gpu", [py, "-m", "reproductions.kaggle.replay", "--notebook", str(root / "kaggle-assets/notebook.api.json"), "--assets", str(root / "kaggle-assets"), "--assets-lock", str(root / "kaggle-assets/assets.lock.json"), "--competition-data", str(data / "kaggle-image-only"), "--out", str(out / "kaggle-d0"), "--timeout-seconds", "600"], timeout=660, env=env)
        stage("kaggle-audited-d0", [py, "-m", "reproductions.evaluate", "--manifest", str(data / "d0.json"), "--method", "kaggle-second", "--predictions", str(out / "kaggle-d0"), "--out", str(out / "evaluation-kaggle-d0")])
        receipt["public_data_acquisition_sha256"] = sha256(data / "acquisition.json")
    except Exception as error:
        receipt["stages"].append({"name": "public-development-pilot", "status": "failed", "error_type": type(error).__name__, "error": str(error)})
    receipt["status"] = "completed_engineering_pilot" if all(r["status"] == "completed" for r in receipt["stages"]) else "completed_with_failures"
    receipt["limitations"] = ["Training is explicitly reduced and not the 45/100 epoch reproduction", "Public scoring uses two source records in development, not original article or hidden leaderboard cohorts", "Dotter annotations/weights are unavailable, geometry control is not learned detection performance", "Device IDs are retained as release variant codes without an unverified device-name crosswalk"]
    save()
    print(json.dumps(receipt), flush=True)
    if receipt["status"] != "completed_engineering_pilot":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
