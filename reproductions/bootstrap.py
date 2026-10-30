"""Fetch original sources at exact commits, without vendoring them in our repo."""

import argparse
import json
from pathlib import Path
import subprocess

from reproductions.common import ROOT, verify_source, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--method", choices=["npj", "kaggle_second", "all"], default="all")
    args = parser.parse_args()
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name in ["npj", "kaggle_second"] if args.method == "all" else [args.method]:
        record = lock["repositories"][name]
        path = dest / name
        if not path.exists():
            subprocess.run(["git", "clone", record["url"], str(path)], check=True)
            subprocess.run(["git", "checkout", "--detach", record["commit"]], cwd=path, check=True)
        if name == "npj" and (path / "weights/unet_weights_07072025.pt").stat().st_size < 1000:
            subprocess.run(["git", "lfs", "pull"], cwd=path, check=True)
        receipts.append({"method": name, **verify_source(path, record["commit"])})
    write_json(dest / "source-receipt.json", receipts)


if __name__ == "__main__":
    main()
