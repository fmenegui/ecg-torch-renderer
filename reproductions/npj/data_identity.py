"""Verify the official development-data bytes, independently of row counts."""

import json
from pathlib import Path

from reproductions.common import sha256

LOCK_PATH = Path(__file__).with_name("development-data.lock.json")


def load_lock():
    return json.loads(LOCK_PATH.read_text())


def source_identity(lock):
    return {"repository": lock["repository"], "revision": lock["revision"]}


def expected_shards(lock):
    expected = {}
    for item in lock["files"]:
        name = Path(item["path"]).name
        if name in expected:
            raise ValueError("Ambiguous official shard name")
        expected[name] = {"file": name, "sha256": item["sha256"], "bytes": item["bytes"]}
    return expected


def verify_shards(root, lock=None, require_complete=False):
    lock = load_lock() if lock is None else lock
    expected = expected_shards(lock)
    found = {}
    for path in sorted(Path(root).rglob("*.parquet")):
        name = path.name
        if name in found or name not in expected:
            raise ValueError("Duplicate or unrecognized development-data shard")
        identity = {"file": name, "sha256": sha256(path), "bytes": path.stat().st_size}
        if identity != expected[name]:
            raise ValueError("Official development-data shard bytes differ: " + name)
        found[name] = identity
    complete = bool(found) and set(found) == set(expected)
    if require_complete and not complete:
        raise ValueError("All original pinned development-data shards are required")
    return [found[k] for k in sorted(found)], complete


def validate_receipt_source(receipt, lock=None):
    lock = load_lock() if lock is None else lock
    if receipt.get("source_release") != source_identity(lock):
        raise ValueError("Development-data source revision differs")
    expected = expected_shards(lock)
    shards = receipt.get("shards", [])
    supplied = {item["file"]: item for item in shards}
    if len(supplied) != len(shards) or supplied != expected:
        raise ValueError("Receipt does not identify every official development-data shard")
