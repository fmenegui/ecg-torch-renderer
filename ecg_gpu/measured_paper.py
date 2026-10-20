"""Construction-time loading of authorized paper scans at a declared scale."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PaperScan:
    """A flat-fielded sRGB albedo scan, not an illuminated paper photograph."""

    path: str
    pixels_per_mm: float
    source: str
    license: str
    kind: str = "measured"

    def __post_init__(self):
        if (
            isinstance(self.pixels_per_mm, bool)
            or not math.isfinite(self.pixels_per_mm)
            or self.pixels_per_mm <= 0
        ):
            raise ValueError("Positive finite pixels_per_mm required")
        if self.kind not in ("measured", "illustrative"):
            raise ValueError("Paper scan kind must be measured or illustrative")
        if any(
            not isinstance(x, str) or not x.strip()
            for x in (self.path, self.source, self.license)
        ):
            raise ValueError("Scan path, source and license are required")


class MeasuredPaperBank(nn.Module):
    """Scale-preserving crops. Scans must share shape and scan resolution.

    Samples are never tiled or stretched to cover a larger sheet. Oversized
    requests raise at runtime. Decoding, provenance and hashes are CPU-only
    construction work. Supplied scans must already be flat-field corrected.
    """

    def __init__(self, scans):
        super().__init__()
        from PIL import Image
        import numpy as np

        scans = tuple(scans)
        if not scans or any(not isinstance(s, PaperScan) for s in scans):
            raise ValueError("At least one PaperScan is required")
        images, provenance = [], []
        for scan in scans:
            path = Path(scan.path)
            with Image.open(path) as image:
                images.append(
                    torch.from_numpy(
                        np.array(image.convert("RGB"), dtype=np.float32)
                    ).permute(2, 0, 1)
                    / 255
                )
            provenance.append(
                {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source": scan.source,
                    "license": scan.license,
                    "kind": scan.kind,
                    "pixels_per_mm": scan.pixels_per_mm,
                }
            )
        if (
            len({tuple(x.shape) for x in images}) != 1
            or len({s.pixels_per_mm for s in scans}) != 1
        ):
            raise ValueError("Scans must have identical shape and pixels_per_mm")
        if min(images[0].shape[-2:]) < 8:
            raise ValueError("Paper scans must be at least 8x8")
        self.register_buffer("images", torch.stack(images))
        self.scan_ppm = scans[0].pixels_per_mm
        self.provenance = provenance

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        data = json.loads(path.read_text())
        if set(data) != {"schema", "scans"} or data["schema"] != "ecg-paper-scans-v1":
            raise ValueError("Expected ecg-paper-scans-v1 manifest")
        return cls(
            PaperScan(**{**item, "path": str(path.parent / item["path"])})
            for item in data["scans"]
        )

    def specification(self):
        return {
            "model": "physical-scale-paper-crop-v1",
            "provenance": self.provenance,
            "shape": list(self.images.shape),
        }

    def forward(self, ppm, height, width, *, phase):
        b, _, sh, sw = self.images.shape
        del b
        if (
            ppm.ndim != 1
            or phase.shape != ppm.shape
            or ppm.device != self.images.device
        ):
            raise ValueError("ppm/phase must be [B] on the bank device")
        torch._assert_async(
            torch.isfinite(ppm).all() & (ppm > 0).all() & torch.isfinite(phase).all(),
            "Finite phase and positive finite canonical ppm required",
        )
        ratio = self.scan_ppm / ppm
        extent_x, extent_y = width * ratio / sw, height * ratio / sh
        torch._assert_async(
            ((extent_x <= 1) & (extent_y <= 1)).all(),
            "Paper scan is physically smaller than requested page",
        )
        x = (torch.arange(width, device=ppm.device) + 0.5) * (2 / width) - 1
        y = (torch.arange(height, device=ppm.device) + 0.5) * (2 / height) - 1
        # Reuse an existing per-record phase to keep the geometry RNG stream
        # unchanged when this appearance-only component is enabled.
        ids = (phase.abs() * 997).floor().long().remainder(len(self.images))
        ox, oy = phase.sin() * (1 - extent_x), (phase * 1.7).sin() * (1 - extent_y)
        gx = x[None, None] * extent_x[:, None, None] + ox[:, None, None]
        gy = y[None, :, None] * extent_y[:, None, None] + oy[:, None, None]
        field = torch.stack([gx.expand(-1, height, -1), gy.expand(-1, -1, width)], -1)
        return F.grid_sample(
            self.images[ids], field, align_corners=False, padding_mode="border"
        ), ids
