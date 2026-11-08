"""Restore exact duplicate paths from a SHA256-deduplicated measurement ZIP.

Extract the ZIP first, then pass the directory containing manifest.json.
Existing different files are never overwritten. No network is used.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def restore(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    verified, created = {}, 0
    for entry in manifest["files"]:
        source, target = root / entry["stored_as"], root / entry["path"]
        if not source.resolve().is_relative_to(
            root
        ) or not target.resolve().is_relative_to(root):
            raise ValueError("Measurement path leaves the extraction directory")
        if source not in verified:
            verified[source] = digest(source)
        if (
            verified[source] != entry["sha256"]
            or source.stat().st_size != entry["bytes"]
        ):
            raise ValueError("Stored measurement differs: " + entry["stored_as"])
        if target.exists():
            if digest(target) != entry["sha256"]:
                raise FileExistsError(
                    "Existing different measurement: " + entry["path"]
                )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        created += 1
    print(
        json.dumps(
            {
                "verified_paths": len(manifest["files"]),
                "created_aliases": created,
                "unique_payloads": len(verified),
                "exact_bytes_restored": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    restore(**vars(parser.parse_args()))
