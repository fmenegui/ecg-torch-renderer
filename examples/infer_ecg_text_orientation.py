"""Batch ECG text orientation with source hashes and all OCR alternatives.

The inference loop consumes RGB only. Source layout, quarter-rotation metadata,
grid truth, material fields and query references are not read by this program.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg_gpu.digitization.page_orientation import infer_text_orientation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(bank, schedule, out, split):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        {k: c[k] for k in ("key", "file", "sha256")}
        for c in json.loads(schedule.read_text())["cases"]
        if c["split"] == split
    ]
    protocol = {
        "schema": "ECG-text-orientation-v1",
        "split": split,
        "cases": cases,
        "source_sha256": digest(ROOT / "ecg_gpu/digitization/page_orientation.py"),
        "driver_sha256": digest(__file__),
        "schedule_sha256": digest(schedule),
        "reference_used_for_inference": False,
        "new_training": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for i, case in enumerate(cases):
        started = time.perf_counter()
        filename = bank / case["file"]
        if digest(filename) != case["sha256"]:
            raise ValueError("Capture changed: " + case["key"])
        with np.load(filename, allow_pickle=False) as z:
            image = z["image"].copy()
        folder = out / case["key"]
        result = infer_text_orientation(image, artifact_dir=folder)
        result["source_capture_sha256"] = case["sha256"]
        result["elapsed_seconds"] = time.perf_counter() - started
        (folder / "orientation.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "key": case["key"],
                    "completed": i + 1,
                    "scheduled": len(cases),
                    "quarter": result["quarter_turns_ccw"],
                    "status": result["status"],
                    "seconds": result["elapsed_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "schedule", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    run(**vars(parser.parse_args()))
