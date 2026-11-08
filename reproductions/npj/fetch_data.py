"""Download the pinned original development data, then verify all LFS hashes.

Run on the durable Runpod volume with space for 75 GB of Parquet files plus
extracted images/masks. This command does not create or start a paid resource.
"""

import argparse

from reproductions.common import write_json
from reproductions.npj.data_identity import load_lock, source_identity, verify_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    lock = load_lock()
    snapshot_download(repo_id=lock["repository"], repo_type="dataset", revision=lock["revision"], local_dir=args.dest, allow_patterns=[r["path"] for r in lock["files"]])
    shards, complete = verify_shards(args.dest, lock, require_complete=True)
    from pathlib import Path
    write_json(Path(args.dest) / "download-receipt.json", {"schema": "npj-development-data-download-v1", "source_release": source_identity(lock), "shards": shards, "complete": complete, "training_completed": False})


if __name__ == "__main__":
    main()
