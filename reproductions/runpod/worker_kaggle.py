"""Finish the native six-model GPU replay and compare explicit public time axes."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from reproductions.common import environment, write_json
from reproductions.runpod.pilot_data import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    root = Path(args.workspace)
    out = root / "outputs"
    receipt = {"schema": "ecg-kaggle-gpu-followup-v1", "environment": environment(), "stages": [], "paper_or_leaderboard_reproduced": False, "cohort": "one PTB-XL development record, potential overlap with original Kaggle training data"}

    def stage(name, command, timeout=600, env=None):
        row = {"name": name, "status": "started", "command": command}
        receipt["stages"].append(row)
        write_json(out / "receipt.json", receipt)
        started = time.perf_counter()
        try:
            with (out / (name + ".log")).open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            row.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
        row["seconds"] = time.perf_counter() - started
        write_json(out / "receipt.json", receipt)
        print(name, row["status"], flush=True)

    py = sys.executable
    data = root / "public-development-pilot"
    prepare(data, include_ahus=False)
    shutil.copytree(data, out / "pilot-inputs")
    stage("source-bootstrap", [py, "-m", "reproductions.bootstrap", "--method", "npj", "--dest", str(root / "sources")])
    stage("npj-public-d0-500hz", [py, "-m", "reproductions.npj.infer", "--upstream", str(root / "sources/npj"), "--preset", "george-moody-2024", "--target-num-samples", "5000", "--image", str(data / "raw/D0_000022_02.png"), "--out", str(out / "npj-d0-500hz")])
    stage("npj-audited-d0", [py, "-m", "reproductions.evaluate", "--manifest", str(data / "d0.json"), "--method", "npj", "--predictions", str(out / "npj-d0-500hz"), "--out", str(out / "evaluation-npj-d0")])
    stage("kaggle-asset-download", [py, "-m", "reproductions.kaggle.fetch_assets", "--dest", str(root / "kaggle-assets")])
    compat = root / "kaggle-timm-compat"
    stage("kaggle-compatible-timm", [py, "-m", "pip", "install", "--no-deps", "--target", str(compat), "timm==1.0.22"])
    env = dict(os.environ, PYTHONPATH=str(compat))
    stage("kaggle-native-six-gpu", [py, "-m", "reproductions.kaggle.replay", "--notebook", str(root / "kaggle-assets/notebook.api.json"), "--assets", str(root / "kaggle-assets"), "--assets-lock", str(root / "kaggle-assets/assets.lock.json"), "--competition-data", str(data / "kaggle-image-only"), "--out", str(out / "kaggle-d0"), "--timeout-seconds", "600"], timeout=660, env=env)
    stage("kaggle-audited-d0", [py, "-m", "reproductions.evaluate", "--manifest", str(data / "d0.json"), "--method", "kaggle-second", "--predictions", str(out / "kaggle-d0"), "--out", str(out / "evaluation-kaggle-d0")])
    stage("full-shape-cuda-capacity", [py, "-m", "reproductions.runpod.capacity_probe", "--upstream", str(root / "sources/npj"), "--out", str(out / "capacity")], timeout=240)
    receipt["status"] = "completed" if all(r["status"] == "completed" for r in receipt["stages"]) else "completed_with_failures"
    write_json(out / "receipt.json", receipt)
    if receipt["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
