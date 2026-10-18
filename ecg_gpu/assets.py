"""Explicit construction-time loading of local measured RGB asset banks."""
from pathlib import Path

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
        raise ValueError('RGB bank dimensions must be integers at least 8')
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError('At least one local RGB photograph required')
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert('RGB'), (width, height), method=Image.Resampling.LANCZOS)
            images.append(torch.from_numpy(np.array(image, dtype=np.float32)/255).permute(2, 0, 1))
    return torch.stack(images)
