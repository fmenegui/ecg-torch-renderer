"""Emissive display observation, independent of paper reflectance and wear."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .gpu_appearance import srgb_to_linear


@dataclass(frozen=True)
class DisplayConfig:
    bezel_fraction: float = 0.035
    native_width: int = 1280
    native_height: int = 800
    subpixel_order: str = "RGB"
    fill_factor: float = 0.88
    peak_radiance: float = 1.0
    black_radiance: float = 0.003
    glass_ior: float = 1.5
    view_cosine: float = 0.9
    reflection_strength: float = 0.08
    refresh_hz: float = 120.0
    duty_cycle: float = 1.0
    exposure_s: float = 1 / 60
    readout_s: float = 0.018
    phase_cycles: float = 0.13
    footprint_samples: int = 2

    def __post_init__(self):
        for key in ("native_width", "native_height"):
            if (
                type(getattr(self, key)) is not int
                or not 16 <= getattr(self, key) <= 16384
            ):
                raise ValueError(f"{key} must be an integer in [16,16384]")
        if self.subpixel_order not in ("RGB", "BGR"):
            raise ValueError("Subpixel order must be RGB or BGR")
        if (
            type(self.footprint_samples) is not int
            or not 1 <= self.footprint_samples <= 4
        ):
            raise ValueError("footprint_samples must be an integer in [1,4]")
        for key in ("fill_factor", "view_cosine", "duty_cycle"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{key} must be finite in (0,1]")
        for key in ("peak_radiance", "glass_ior", "refresh_hz", "exposure_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite {key} required")
        for key in (
            "bezel_fraction",
            "black_radiance",
            "reflection_strength",
            "readout_s",
            "phase_cycles",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Nonnegative finite {key} required")
        if self.bezel_fraction > 0.2:
            raise ValueError("bezel_fraction must be <=.2 of panel width")
        if self.black_radiance >= self.peak_radiance or self.glass_ior < 1:
            raise ValueError("Use black below peak and glass IOR >=1")


class EmissiveDisplay(nn.Module):
    """RGB stripe aperture, pixel footprint, PWM exposure and glass reflection.

    Refresh is modeled as a full-panel rectangular PWM backlight. Frame scanout,
    LCD response and RGB OLED geometry are not inferred from refresh rate.
    Content integration is stratified quadrature. Aperture coverage uses an
    analytical box integral, approximating rotated footprints by their local
    axis-aligned bounds. This is not a diffraction model. The supplied UV map is shared unchanged with geometric targets.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or DisplayConfig()
        if not isinstance(self.config, DisplayConfig):
            raise TypeError("config must be DisplayConfig")
        order = [0, 1, 2] if self.config.subpixel_order == "RGB" else [2, 1, 0]
        self.register_buffer("stripe", torch.tensor(order)[None, :, None, None])

    def specification(self):
        return {
            "model": "emissive-rgb-stripe-pwm-glass-v1",
            "config": asdict(self.config),
        }

    def frame_coverage(self, field):
        b, h, w, _ = field.shape
        return F.grid_sample(
            torch.ones(b, 1, h, w, device=field.device),
            field / (1 + 2 * self.config.bezel_fraction),
            align_corners=False,
        )

    def composite(self, background, emission, panel_coverage, field, *, shadow=0.0):
        """Coverage weights sum to one, with no double alpha at panel edges."""
        frame = torch.maximum(self.frame_coverage(field), panel_coverage)
        bezel = (frame - panel_coverage).clamp(0, 1)
        reflected = (background * (1 - frame) + 0.012 * bezel) * (1 - shadow)
        return reflected + emission * panel_coverage, bezel

    def exposure_fraction(self, start_s):
        """Exact integral of a periodic rectangular pulse over shutter time."""
        c = self.config

        # Integrate in cycles to avoid subtraction of small second-valued times.
        def primitive(cycles):
            return cycles.floor() * c.duty_cycle + cycles.remainder(1).clamp(
                max=c.duty_cycle
            )

        start = start_s * c.refresh_hz + c.phase_cycles
        duration = c.exposure_s * c.refresh_hz
        return ((primitive(start + duration) - primitive(start)) / duration).clamp(0, 1)

    @staticmethod
    def aperture_coverage(center, footprint, offset, width):
        """Exact axis-aligned box integral of a periodic unit-pitch aperture."""

        def primitive(position):
            return position.floor() * width + (position.remainder(1) - offset).clamp(
                0, width
            )

        return (
            (primitive(center + footprint / 2) - primitive(center - footprint / 2))
            / footprint.clamp_min(1e-8)
        ).clamp(0, 1)

    def forward(self, canonical, field, *, reflection=None):
        b, _, h, w = canonical.shape
        if field.shape != (b, h, w, 2) or field.device != canonical.device:
            raise ValueError("Display field must be [B,H,W,2] on the image device")
        c = self.config
        dy, dx = torch.gradient(field, dim=(1, 2))
        linear = srgb_to_linear(canonical)
        result = torch.zeros_like(linear)
        n = c.footprint_samples
        # Integrate each stratum's aperture analytically. Point samples alias
        # unresolved RGB stripes into false rainbow patterns at a distance.
        footprint_x = (
            (dx[..., 0].abs() + dy[..., 0].abs()) * (c.native_width / (2 * n))
        )[:, None]
        footprint_y = (
            (dx[..., 1].abs() + dy[..., 1].abs()) * (c.native_height / (2 * n))
        )[:, None]
        for iy in range(n):
            for ix in range(n):
                uv = (
                    field
                    + dx * ((ix + 0.5) / n - 0.5)
                    + dy * ((iy + 0.5) / n - 0.5)
                    + 1
                ) / 2
                px, py = uv[..., 0] * c.native_width, uv[..., 1] * c.native_height
                center = torch.stack(
                    [
                        (px.floor() + 0.5) * (2 / c.native_width) - 1,
                        (py.floor() + 0.5) * (2 / c.native_height) - 1,
                    ],
                    -1,
                )
                value = F.grid_sample(
                    linear, center, align_corners=False, padding_mode="border"
                )
                aperture_x = self.aperture_coverage(
                    px[:, None],
                    footprint_x,
                    (self.stripe + (1 - c.fill_factor) / 2) / 3,
                    c.fill_factor / 3,
                )
                aperture_y = self.aperture_coverage(
                    py[:, None], footprint_y, (1 - c.fill_factor) / 2, c.fill_factor
                )
                result = result + value * aperture_x * aperture_y * 3 / (
                    c.fill_factor**2
                )
        result = result / (n * n)
        row = torch.arange(h, device=canonical.device)[None, None, :, None] / max(
            h - 1, 1
        )
        # Both black leakage and modulated signal come from the emissive source.
        emission = (
            c.black_radiance + result * (c.peak_radiance - c.black_radiance)
        ) * self.exposure_fraction(row * c.readout_s)
        r0 = ((c.glass_ior - 1) / (c.glass_ior + 1)) ** 2
        fresnel = r0 + (1 - r0) * (1 - c.view_cosine) ** 5
        if reflection is None:
            x = torch.linspace(-1, 1, w, device=canonical.device)[None, None, None]
            y = torch.linspace(-1, 1, h, device=canonical.device)[None, None, :, None]
            reflection = torch.exp(
                -((x + 0.45) / 0.38).pow(8) - ((y + 0.35) / 0.55).pow(8)
            )
        elif (
            reflection.shape != canonical.shape or reflection.device != canonical.device
        ):
            raise ValueError("Reflection must be linear RGB [B,3,H,W] on image device")
        return emission * (1 - fresnel) + reflection * c.reflection_strength * fresnel
