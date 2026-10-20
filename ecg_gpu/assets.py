"""Explicit construction-time loading of local measured RGB asset banks."""

from pathlib import Path
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from PIL import Image, ImageOps
import torch


def load_rgb_bank(paths, *, height=768, width=1024):
    """Return local sRGB photographs as [K,3,H,W] float32 CPU tensors.

    Resize/crop and disk access happen once before constructing a renderer.
    The renderer stores asset hashes and moves the bank with .to(device).
    Callers remain responsible for asset rights and their train/test splits.
    """
    if type(height) is not int or type(width) is not int or min(height, width) < 8:
        raise ValueError("RGB bank dimensions must be integers at least 8")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("At least one local RGB photograph required")
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = ImageOps.fit(
                source.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS
            )
            images.append(
                torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(
                    2, 0, 1
                )
            )
    return torch.stack(images)


@dataclass(frozen=True)
class SceneBank:
    """Local scene photographs with declared rights and split provenance."""

    images: torch.Tensor
    provenance: tuple

    @classmethod
    def from_json(cls, path, *, split="train", height=768, width=1024):
        path = Path(path)
        data = json.loads(path.read_text())
        if set(data) != {"schema", "assets"} or data["schema"] != "ecg-scene-bank-v1":
            raise ValueError("Expected ecg-scene-bank-v1 manifest")
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown scene asset split")
        selected, provenance, hash_split, family_split = [], [], {}, {}
        for item in data["assets"]:
            required = {"path", "source", "license", "family", "split"}
            if set(item) != required or any(
                not isinstance(item[k], str) or not item[k].strip() for k in required
            ):
                raise ValueError(
                    "Each scene requires path/source/license/family/split strings"
                )
            group = item["split"]
            if group not in ("train", "validation", "test"):
                raise ValueError("Unknown scene asset split")
            local = path.parent / item["path"]
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            if (
                hash_split.get(digest, group) != group
                or family_split.get(item["family"], group) != group
            ):
                raise ValueError("Scene image or family leaks across dataset splits")
            hash_split[digest], family_split[item["family"]] = group, group
            if group == split:
                selected.append(local)
                provenance.append(
                    {k: v for k, v in item.items() if k != "path"} | {"sha256": digest}
                )
        if not selected:
            raise ValueError("No scene assets in requested split")
        return cls(
            load_rgb_bank(selected, height=height, width=width), tuple(provenance)
        )
