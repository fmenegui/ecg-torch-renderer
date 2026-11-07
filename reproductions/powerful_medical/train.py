"""Train the underdetermined Dotter hypothesis on explicit annotated pairs.

The original 176-image annotation bank is not public. A different bank is an
adaptation, never an exact reproduction of the commercial model.
"""

import argparse
import csv
from pathlib import Path
import random

import numpy as np
from PIL import Image

from reproductions.common import environment, new_run, require_cuda, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="CSV: image,mask,image_sha256,mask_sha256,patient_id,split")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    require_cuda(args.device)
    import torch
    from reproductions.powerful_medical.dotter import Dotter
    torch.set_num_threads(2)
    torch.manual_seed(1973)
    random.seed(1973)
    records = list(csv.DictReader(Path(args.annotations).open()))
    root = Path(args.annotations).resolve().parent
    ownership = {}
    for row in records:
        if row["split"] not in ["train", "validation"] or not row["patient_id"]:
            raise ValueError("Explicit patient grouping and split required")
        if ownership.setdefault(row["patient_id"], row["split"]) != row["split"]:
            raise ValueError("Patient leakage")
        for field in ["image", "mask"]:
            if sha256(root / row[field]) != row[field + "_sha256"]:
                raise ValueError("Changed annotation input")
    if not all(any(r["split"] == split for r in records) for split in ["train", "validation"]):
        raise ValueError("Both training and validation required")
    out = new_run(args.out)
    model = Dotter().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    receipt = {"schema": "pm-dotter-training-v1", "source_annotations_sha256": sha256(args.annotations), "environment": environment(), "commercial_model_reproduced": False, "engineering_smoke": args.smoke, "documented": {"optimizer": "Adam", "learning_rate": 0.005, "loss": "BCEWithLogits", "epochs": 300, "patch": [256, 256], "activation": "SiLU"}, "choices": {**model.architecture_choices_not_disclosed_by_authors, "batch_size": 1, "patch_sampling": "one_patch_per_supplied_image_per_epoch", "augmentation": "horizontal_flip_only", "validation": "deterministic_center_patch", "mask_rendering": "supplied_annotation_pixels"}, "status": "started"}
    write_json(out / "receipt.json", receipt)
    best = float("inf")
    try:
        for epoch in range(1 if args.smoke else 300):
            metrics = {"epoch": epoch + 1}
            for split in ["train", "validation"]:
                subset = [r for r in records if r["split"] == split]
                if split == "train":
                    random.shuffle(subset)
                if args.smoke:
                    subset = subset[:2]
                model.train(split == "train")
                total = 0
                for row in subset:
                    image = np.asarray(Image.open(root / row["image"]).convert("RGB"))
                    mask = np.asarray(Image.open(root / row["mask"]).convert("L"))
                    if image.shape[:2] != mask.shape or min(mask.shape) < 256:
                        raise ValueError("Aligned annotations at least 256x256 required")
                    y = random.randrange(mask.shape[0] - 255) if split == "train" else (mask.shape[0] - 256) // 2
                    x = random.randrange(mask.shape[1] - 255) if split == "train" else (mask.shape[1] - 256) // 2
                    a = image[y:y + 256, x:x + 256].copy()
                    b = mask[y:y + 256, x:x + 256].copy()
                    if split == "train" and random.random() < 0.5:
                        a, b = a[:, ::-1].copy(), b[:, ::-1].copy()
                    a = torch.from_numpy(a).permute(2, 0, 1).float()[None].to(args.device) / 255
                    b = torch.from_numpy(b).float()[None, None].to(args.device) / 255
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(split == "train"):
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(a), b)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite Dotter loss")
                        if split == "train":
                            loss.backward()
                            optimizer.step()
                    total += float(loss.detach())
                metrics[split + "_loss"] = total / len(subset)
            if metrics["validation_loss"] < best:
                best = metrics["validation_loss"]
                torch.save(model.state_dict(), out / "best.pt")
            receipt.update(completed_epochs=epoch + 1, last_epoch=metrics)
            write_json(out / "receipt.json", receipt)
        receipt["status"] = "completed_engineering_smoke" if args.smoke else "completed_adapted_training"
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
