"""Execute one bounded batch INSIDE a newly provisioned Runpod training pod.

Provisioning, price/stock reads and external stop verification belong to the
controller. No remote resource is created by this script. A failed stop remains
a failure, never a claim that billing ended. This is an inner guard, not a
guarantee against host failure or a disconnected control plane.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproductions.common import run_bounded, write_json


def preflight():
    return {"pod_id_present": bool(os.getenv("RUNPOD_POD_ID")), "api_key_present": bool(os.getenv("RUNPOD_API_KEY")), "cli_present": bool(shutil.which("runpodctl"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    status = preflight()
    if args.preflight_only:
        write_json(args.out, {"schema": "runpod-preflight-v1", "checks": status, "resources_created": [], "gpu_execution_started": False})
        return
    if not all(status.values()):
        raise RuntimeError("Runpod batch requires a pod identity, scoped control-plane credential and current runpodctl")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout_seconds <= 0:
        raise ValueError("Explicit command and positive timeout are required")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "runpod-batch-v1", "pod_id": os.environ["RUNPOD_POD_ID"], "command": command, "timeout_seconds": args.timeout_seconds, "status": "started", "stop_returncode": None, "billing_end_verified_externally": False}
    write_json(output / "receipt.json", receipt)
    started = time.perf_counter()
    try:
        with (output / "job.log").open("w") as log:
            code = run_bounded(command, args.timeout_seconds, log)
        receipt.update(status="completed" if code == 0 else "failed", returncode=code)
    except subprocess.TimeoutExpired:
        receipt["status"] = "timeout"
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        # Durable output first. The stop may terminate this interpreter.
        write_json(output / "receipt.json", receipt)
        stop = subprocess.run(["runpodctl", "pod", "stop", os.environ["RUNPOD_POD_ID"]], capture_output=True, text=True, timeout=60, check=False)
        receipt["stop_returncode"] = stop.returncode
        write_json(output / "receipt.json", receipt)
    if receipt["status"] != "completed" or receipt["stop_returncode"] != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
