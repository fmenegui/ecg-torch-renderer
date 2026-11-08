"""Read current price, stock and templates without creating paid resources."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from reproductions.common import write_json


def invoke(cli, arguments, key):
    process = subprocess.run([cli, *arguments], capture_output=True, text=True, timeout=120)
    if process.returncode:
        message = process.stderr.replace(key, "[redacted]") if key else process.stderr
        raise RuntimeError(message[:1200])
    return process.stdout


def arrays(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ["gpus", "gpuTypes", "templates", "dataCenters", "data", "results", "items"]:
            if key in value:
                found = arrays(value[key])
                if found:
                    return found
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    key = os.getenv("RUNPOD_API_KEY", "")
    receipt = {"schema": "runpod-live-catalog-v1", "credential_present": bool(key), "authenticated": False, "resources_created": [], "gpu_execution_started": False}
    if not key:
        receipt["status"] = "missing_credential"
        write_json(root / "receipt.json", receipt)
        print(json.dumps(receipt))
        raise SystemExit(2)
    try:
        account = json.loads(invoke(args.cli, ["user"], key))
        receipt.update(authenticated=True, status="catalog_read")
        # Do not preserve names, tokens, keys or unrelated account metadata.
        receipt["account_balance"] = {k: account[k] for k in ["balance", "currentSpendPerHr"] if k in account}
        for name, command in [
            ("gpus", ["gpu", "list", "--include-unavailable"]),
            ("templates", ["template", "search", "pytorch", "--type", "official", "--limit", "20"]),
            ("datacenters", ["datacenter", "list"]),
        ]:
            data = json.loads(invoke(args.cli, command, key))
            write_json(root / (name + ".json"), data)
            rows = arrays(data)
            print(json.dumps({"catalog": name, "root_type": type(data).__name__, "root_keys": list(data)[:20] if isinstance(data, dict) else None, "records": len(rows)}))
            if name == "gpus":
                selected = [r for r in rows if any(s in str(r.get("id", r.get("displayName", ""))) for s in ["4090", "3090", "A40", "A5000", "A6000", "L4"])]
                print(json.dumps({"gpu_candidates": selected[:12]}))
            if name == "templates":
                print(json.dumps({"pytorch_templates": [{k: r[k] for k in ["id", "name", "imageName", "image", "allowedCudaVersions"] if k in r} for r in rows[:8]]}))
        for command in [["pod", "create", "--help"], ["pod", "update", "--help"], ["pod", "--help"], ["ssh", "add-key", "--help"], ["ssh", "remove-key", "--help"], ["ssh", "info", "--help"], ["network-volume", "create", "--help"]]:
            (root / ("help-" + "-".join(command[:-1]) + ".txt")).write_text(invoke(args.cli, command, key))
    except Exception as error:
        receipt.update(status="failed", error=str(error).replace(key, "[redacted]"), error_type=type(error).__name__)
        raise
    finally:
        write_json(root / "receipt.json", receipt)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
