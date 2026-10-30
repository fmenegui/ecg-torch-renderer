"""Acquire the public ECG-Image-Database file inventory, without renting a GPU.

Pages are retained verbatim for replay and resumable acquisition. File bytes
and original scoring-cohort identity still require subsequent validation.
"""

import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

from reproductions.common import sha256, write_json


def describe(files):
    counts = {}
    images = {".png", ".jpg", ".jpeg"}
    for file in files:
        path = Path(file["name"])
        source = path.parts[0] if path.parts else "other"
        if source not in ["D0", "D1", "D2"]:
            source = "other"
        item = counts.setdefault(source, {"files": 0, "image_files": 0, "bytes": 0, "records": set()})
        item["files"] += 1
        item["bytes"] += int(file.get("totalBytes", 0))
        if len(path.parts) > 1:
            item["records"].add(path.parts[1])
        if path.suffix.lower() in images:
            item["image_files"] += 1
    for item in counts.values():
        item["records"] = len(item["records"])
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pages", type=int, default=500)
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = "https://www.kaggle.com/api/v1/datasets/"
    with urllib.request.urlopen(base + "view/physionet/ecg-image-database", timeout=30) as response:
        metadata = json.load(response)
    if metadata.get("currentVersionNumber") != 2:
        raise ValueError("Reviewed dataset version changed, reconcile the source before continuing")
    if (root / "dataset-view.json").exists():
        old = json.loads((root / "dataset-view.json").read_text())
        if old.get("currentVersionNumber") != 2:
            raise ValueError("Cached dataset version differs")
    write_json(root / "dataset-view.json", metadata)
    token, files, pages, names = None, [], [], set()
    complete = False
    for page in range(args.max_pages):
        path = root / f"page-{page:04d}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            url = base + "list/physionet/ecg-image-database?" + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=30) as response:
                result = json.load(response)
            write_json(path, result)
        if result.get("errorMessage"):
            raise ValueError("Dataset catalog API returned an error")
        for file in result.get("datasetFiles", []):
            if file["name"] in names:
                raise ValueError("Repeated file identity in paginated inventory")
            names.add(file["name"])
            files.append(file)
        pages.append({"file": path.name, "sha256": sha256(path)})
        next_token = result.get("nextPageToken")
        complete = not bool(next_token)
        receipt = {"schema": "ecg-image-database-public-catalog-v1", "dataset": "physionet/ecg-image-database", "version": 2, "complete_file_inventory": complete, "pages": pages, "file_count": len(files), "by_source": describe(files), "file_bytes_downloaded": False, "original_cohort_metrics_reproduced": False}
        write_json(root / "receipt.json", receipt)
        if page % 10 == 0 or complete:
            print(f"catalog pages={page + 1} files={len(files)} complete={complete}", flush=True)
        if complete:
            break
        if next_token == token:
            raise ValueError("Non-advancing catalog page")
        token = next_token
    write_json(root / "files.json", files)
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
