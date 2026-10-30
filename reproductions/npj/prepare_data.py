"""Extract both original and T0 views from the authors' Parquet split files.

No new split, renderer or pixel transformation is introduced. The public
release has 4112 training and 217 validation rows, two image/mask pairs per row.
Biological patient independence is not inferred from source row IDs.
"""

import argparse
import io
from pathlib import Path

from reproductions.common import new_run, sha256, write_json
from reproductions.npj.data_identity import load_lock, source_identity, verify_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    import pyarrow.parquet as parquet
    from PIL import Image
    root = Path(args.parquet_root)
    lock = load_lock()
    shards, complete_shards = verify_shards(root, lock, require_complete=args.require_complete)
    out = new_run(args.out)
    ownership = {}
    rows, images = {}, []
    for split in ["train", "val"]:
        folder = out / split
        folder.mkdir()
        rows[split] = 0
        files = sorted(root.rglob(split + "-*.parquet"))
        if not files:
            raise ValueError(f"No original {split} shards supplied")
        for path in files:
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=1, columns=["id", "img", "mask", "img_T0", "mask_T0"]):
                record = batch.to_pylist()[0]
                identity = str(record["id"])
                if identity in ownership:
                    raise ValueError("Repeated source row ID, including possible split overlap")
                ownership[identity] = split
                rows[split] += 1
                for suffix in ["", "_T0"]:
                    stem = f"{rows[split]:06d}{suffix}"
                    dimensions = []
                    for key, name in [("img" + suffix, stem + ".png"), ("mask" + suffix, stem + "_mask.png")]:
                        value = record[key]
                        if not value or value.get("bytes") is None:
                            raise ValueError("Original embedded PNG bytes are required")
                        payload = value["bytes"]
                        with Image.open(io.BytesIO(payload)) as im:
                            im.load()
                            dimensions.append(im.size)
                            if im.format != "PNG":
                                raise ValueError("Native dataset expects PNG, no implicit transcoding")
                        destination = folder / name
                        destination.write_bytes(payload)
                        images.append({"source_id": identity, "source_view": suffix or "original", "split": split, "file": str(destination.relative_to(out)), "sha256": sha256(destination)})
                    if dimensions[0] != dimensions[1]:
                        raise ValueError("Image and mask dimensions differ")
    complete = complete_shards and rows == {"train": 4112, "val": 217} and len(images) == 4 * 4329
    receipt = {"schema": "npj-public-data-extraction-v1", "source_release": source_identity(lock), "rows": rows, "image_mask_pairs": len(images) // 2, "complete_public_release": complete, "patient_independence": "not_established_by_row_ids", "shards": shards, "files": images}
    write_json(out / "data-receipt.json", receipt)
    if args.require_complete and not complete:
        raise ValueError("Incomplete original public dataset, see retained receipt")


if __name__ == "__main__":
    main()
