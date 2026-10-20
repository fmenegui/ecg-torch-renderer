"""Directional paper reflectance and thermal head/feed defects in torch."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ThermalPrintConfig:
    head_strength: float = 0.12
    feed_strength: float = 0.045
    retention_strength: float = 0.08
    fiber_strength: float = 0.012
    head_pitch_mm: float = 0.125
    feed_period_mm: float = 9.0

    def __post_init__(self):
        for key in (
            "head_strength",
            "feed_strength",
            "retention_strength",
            "fiber_strength",
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{key} must be finite in [0,1]")
        for key in ("head_pitch_mm", "feed_period_mm"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite {key} required")


class ThermalPrint(nn.Module):
    """Illustrative material variation. Does not modify geometric supervision.

    Head defects are constant along the feed direction. Feed bands use physical
    millimeters. Fibers are anisotropic correlated reflectance, not raised mesh.
    """

    def __init__(self, height, width, config=None):
        super().__init__()
        self.config = config or ThermalPrintConfig()
        if not isinstance(self.config, ThermalPrintConfig):
            raise TypeError("config must be ThermalPrintConfig")
        self.height, self.width = height, width
        self.register_buffer(
            "x", torch.arange(width, dtype=torch.float32)[None, None, None]
        )
        self.register_buffer(
            "y", torch.arange(height, dtype=torch.float32)[None, None, :, None]
        )

    def specification(self):
        return {
            "model": "illustrative-thermal-head-feed-v1",
            "config": asdict(self.config),
        }

    def forward(self, ppm, generator=None, *, phase=None):
        b, d, c = len(ppm), ppm.device, self.config
        scale = ppm[:, None, None, None]

        # Native head density is integrated by bilinear sampling. The allocated
        # bank has a fixed bound, avoiding device-to-host shape decisions.
        def random(shape, salt):
            if phase is None:
                return torch.rand(shape, device=d, generator=generator)
            count = math.prod(shape[1:])
            coordinate = torch.arange(count, device=d, dtype=torch.float32)[None]
            value = (
                torch.sin(coordinate * 12.9898 + phase[:, None] * 78.233 + salt)
                * 43758.5453
            )
            return value.frac().abs().reshape(shape)

        head = random((b, 1, 1, 8192), 1.0)
        pitch_pixels = scale * c.head_pitch_mm
        head_field = (self.x / pitch_pixels + 0.5) * (2 / 8192) - 1
        torch._assert_async(
            (self.width / pitch_pixels <= 8191).all(),
            "Thermal head bank does not cover this physical sheet width",
        )
        field = torch.stack(
            [head_field[:, 0, 0], torch.zeros(b, self.width, device=d)], -1
        )[:, None]
        head = F.grid_sample(head, field, align_corners=False, padding_mode="border")
        feed_phase = random((b, 1, 1, 1), 2.0) * (2 * math.pi)
        feed = (
            1
            + torch.sin(self.y / scale * (2 * math.pi / c.feed_period_mm) + feed_phase)
        ) / 2
        local = F.interpolate(
            random((b, 1, 12, 16), 3.0),
            (self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        retention = (
            (1 - c.head_strength * head)
            * (1 - c.feed_strength * feed)
            * (1 - c.retention_strength * local)
        )
        fibers = F.interpolate(
            random((b, 1, max(2, self.height // 2), max(2, self.width // 24)), 4.0),
            (self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        return {
            "print_retention": retention,
            "substrate_factor": 1 - c.fiber_strength * fibers,
        }
