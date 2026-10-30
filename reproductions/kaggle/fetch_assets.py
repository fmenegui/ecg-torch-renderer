"""Download version-pinned public assets, verify bytes, retain a sealed receipt."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import urllib.request

from reproductions.common import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--capture-first-digests", action="store_true", help="Initial provenance capture when an upstream digest was not supplied")
    args = parser.parse_args()
    records = json.loads((ROOT / "kaggle/asset-sources.json").read_text())
    root = Path(args.dest).resolve()
    root.mkdir(parents=True, exist_ok=True)
    notebook_path = root / "notebook.api.json"
    if not notebook_path.exists():
        with urllib.request.urlopen("https://www.kaggle.com/api/v1/kernels/pull/takashisomeya/physionet-2nd-place-submission", timeout=30) as response:
            document = json.load(response)
        write_json(notebook_path, document)
    document = json.loads(notebook_path.read_text())
    from reproductions.kaggle.replay import notebook_code
    _, notebook_digest = notebook_code(document)
    if notebook_digest != records["notebook_source_sha256"]:
        raise ValueError("Original submission notebook identity changed")

    def fetch(record):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Asset path escaped staging root")
        expected = record.get("sha256")
        if not expected and not args.capture_first_digests:
            raise ValueError("Unsealed asset identity requires an explicit first observation")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(path.name + ".partial")
            try:
                with urllib.request.urlopen(record["url"], timeout=60) as response, temporary.open("wb") as stream:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
                if record.get("size") is not None and temporary.stat().st_size != record["size"]:
                    raise ValueError(f"Asset size mismatch: {relative}")
                digest = sha256(temporary)
                if expected and digest != expected:
                    raise ValueError(f"Asset digest mismatch: {relative}")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        digest = sha256(path)
        if expected and digest != expected:
            raise ValueError(f"Changed existing asset: {relative}")
        if record.get("size") is not None and path.stat().st_size != record["size"]:
            raise ValueError(f"Changed existing asset size: {relative}")
        print(str(relative), digest, flush=True)
        return str(relative), digest

    with ThreadPoolExecutor(max_workers=2) as pool:
        files = dict(pool.map(fetch, records["files"]))
    write_json(root / "assets.lock.json", {"schema": "kaggle-second-assets-v1", "method": "kaggle-second-final-six", "notebook_source_sha256": records["notebook_source_sha256"], "dataset_versions": records["dataset_versions"], "first_digests_captured_this_invocation": bool(args.capture_first_digests), "files": files})


if __name__ == "__main__":
    main()
