"""Full npj training profiles across bounded, durable-volume invocations."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from reproductions.common import environment, sha256, write_json


def completed(path):
    receipt = Path(path) / "receipt.json"
    return receipt.exists() and json.loads(receipt.read_text()).get("status") == "completed_training"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    root = Path(args.workspace)
    output = root / "outputs/full-training"
    output.mkdir(parents=True, exist_ok=True)
    control = json.loads((root / "execution.json").read_text())
    # Leave time for the current dataloader batch, checkpoint collection and SSH.
    deadline = control["deadline"] - 600
    receipt = {"schema": "npj-full-training-worker-v1", "status": "started", "environment": environment(), "stages": [], "paper_result_reproduced": False, "source_profiles": ["paper_table4", "lead_text_public"], "font_bank": "observed container fonts, original historical font bank unavailable"}
    py = sys.executable

    def run(name, command):
        remaining = deadline - time.time()
        if remaining < 60:
            receipt["status"] = "paused_training"
            return False
        row = {"name": name, "command": command, "status": "started"}
        receipt["stages"].append(row)
        write_json(output / "receipt.json", receipt)
        try:
            with (output / (name + ".log")).open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            row.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
            if result.returncode:
                receipt["status"] = "failed"
                raise subprocess.CalledProcessError(result.returncode, command)
            return True
        except subprocess.TimeoutExpired:
            row["status"] = "paused_at_invocation_deadline"
            receipt["status"] = "paused_training"
            return False
        finally:
            write_json(output / "receipt.json", receipt)

    if not run("source-bootstrap", [py, "-m", "reproductions.bootstrap", "--method", "npj", "--dest", str(root / "sources")]):
        return
    capacity = output / ("capacity-" + str(time.time_ns()))
    if not run("original-batch-capacity", [py, "-m", "reproductions.runpod.capacity_probe", "--upstream", str(root / "sources/npj"), "--out", str(capacity)]):
        return
    if not json.loads((capacity / "receipt.json").read_text()).get("full_shape_feasible"):
        receipt.update(status="failed", reason="Original batch/patch capacity control failed. Data download and training were not started.")
        write_json(output / "receipt.json", receipt)
        raise RuntimeError(receipt["reason"])
    data = root / "npj-data"
    extraction = data / "data-receipt.json"
    if not extraction.exists():
        if not run("original-data-download", [py, "-m", "reproductions.npj.fetch_data", "--dest", str(root / "npj-parquets")]):
            return
        command = [py, "-m", "reproductions.npj.prepare_data", "--parquet-root", str(root / "npj-parquets"), "--out", str(data), "--require-complete"]
        if data.exists():
            command.append("--resume")
        if not run("original-data-extraction", command):
            return
    identity = json.loads(extraction.read_text())
    if not identity.get("complete_public_release"):
        raise ValueError("A full original development-data release is required")
    receipt["data_receipt_sha256"] = sha256(extraction)
    shutil.copy2(extraction, output / "data-receipt.json")
    fonts = [{"path": str(p), "sha256": sha256(p)} for p in sorted(Path("/usr/share/fonts").rglob("*.ttf"))]
    write_json(output / "fonts.json", fonts)
    for profile, folder in [("paper_table4", "main-paper-table4"), ("lead_text_public", "lead-names-public")]:
        destination = output / folder
        if completed(destination):
            continue
        command = [py, "-m", "reproductions.npj.train", "--upstream", str(root / "sources/npj"), "--profile", profile, "--out", str(destination), "--invocation-deadline", str(deadline - 60)]
        if profile == "paper_table4":
            command += ["--train-dir", str(data / "train"), "--validation-dir", str(data / "val"), "--data-receipt", str(extraction)]
        if (destination / "last.pt").exists():
            command.append("--resume")
        elif destination.exists():
            # Preserve diagnostics from an attempt that never completed an epoch.
            destination.rename(destination.with_name(destination.name + ".unfinished-" + str(time.time_ns())))
        if not run("train-" + profile, command) or not completed(destination):
            if receipt["status"] == "started":
                receipt["status"] = "paused_training"
            write_json(output / "receipt.json", receipt)
            return
    receipt["status"] = "completed_full_training"
    receipt["matched_cohort_evaluation_completed"] = False
    write_json(output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
