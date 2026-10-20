"""Thin-lens defocus with bounded work and explicit calibration provenance."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .gpu_appearance import CameraAppearance


@dataclass(frozen=True)
class LensConfig:
    focal_length_mm: float = 4.0
    f_number: float = 2.0
    focus_distance_m: float = 0.38
    camera_distance_m: float = 0.38
    pixel_pitch_um: float = 4.0
    max_sigma_pixels: float = 4.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite {key} required")
        if (
            min(self.focus_distance_m, self.camera_distance_m)
            <= self.focal_length_mm * 0.001
        ):
            raise ValueError("Object/focus distances must exceed focal length")
        if self.max_sigma_pixels > 4:
            raise ValueError(
                "Maximum sigma must be <=4 pixels for the bounded PSF bank"
            )


@dataclass(frozen=True)
class SensorConfig:
    """Declared linear RGB sensor parameters. No CFA or measured ISP inferred."""

    full_well_electrons: float = 12000.0
    read_noise_electrons: float = 2.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Nonnegative finite {key} required")
        if self.full_well_electrons <= 0:
            raise ValueError("Positive full_well_electrons required")


class PhysicalCamera(CameraAppearance):
    """A thin lens followed by the existing linear RGB sensor and JPEG model.

    Defocus uses a spatial mixture of normalized Gaussian PSFs. Sigma equals
    one quarter of the circle-of-confusion diameter, matching a uniform disk's
    second moment. This is not an occlusion-aware lens integrator. Large CoCs
    raise an error instead of silently clipping blur. Use offline ray tracing
    for silhouettes requiring lens visibility integration.
    """

    def __init__(self, config=None, *, jpeg_enabled=True, provenance=None, sensor=None):
        super().__init__(mode="sensor", jpeg_enabled=jpeg_enabled)
        self.config = config or LensConfig()
        if not isinstance(self.config, LensConfig):
            raise TypeError("config must be LensConfig")
        if sensor is not None and not isinstance(sensor, SensorConfig):
            raise TypeError("sensor must be SensorConfig")
        self.sensor = sensor
        self.provenance = json.loads(json.dumps(provenance or {"kind": "illustrative"}))
        if self.provenance.get("kind") not in ("illustrative", "measured"):
            raise ValueError("Calibration kind must be illustrative or measured")
        if self.provenance["kind"] == "measured" and any(
            not isinstance(self.provenance.get(k), str)
            or not self.provenance[k].strip()
            for k in ("source", "method", "data_sha256")
        ):
            raise ValueError(
                "Measured calibration requires source, method and data_sha256"
            )
        if self.provenance["kind"] == "measured":
            digest = self.provenance["data_sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Calibration data_sha256 must be lowercase SHA-256")
        self.register_buffer("sigmas", torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0]))
        for index, sigma in enumerate((0.5, 1.0, 2.0, 4.0)):
            radius = math.ceil(3 * sigma)
            coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (coordinate / sigma).square())
            self.register_buffer(f"psf_{index}", kernel / kernel.sum())

    requires_observation = True

    def scene_depth(self, scene, ppm, width, valid):
        depth = torch.full_like(valid, self.config.camera_distance_m)
        if scene is not None:
            scale_m = width / ppm * 0.001 / 2
            depth = (
                depth
                - scene["height"] * scale_m[:, None, None, None] * scene["paper_mask"]
            )
        return depth

    @classmethod
    def from_json(cls, path, *, jpeg_enabled=True):
        """Read calibration once at construction. No I/O occurs in forward."""
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if (
            set(data) - {"schema", "lens", "provenance", "sensor"}
            or not {"schema", "lens", "provenance"} <= data.keys()
            or data["schema"] != "ecg-camera-calibration-v1"
        ):
            raise ValueError(
                "Expected ecg-camera-calibration-v1 with lens and provenance"
            )
        provenance = {
            **data["provenance"],
            "profile_sha256": hashlib.sha256(raw).hexdigest(),
        }
        return cls(
            LensConfig(**data["lens"]),
            jpeg_enabled=jpeg_enabled,
            provenance=provenance,
            sensor=SensorConfig(**data["sensor"])
            if data.get("sensor") is not None
            else None,
        )

    def specification(self):
        return {
            "model": "thin-lens-gaussian-mixture-v1",
            "lens": asdict(self.config),
            "mode": self.mode,
            "jpeg_enabled": self.jpeg_enabled,
            "provenance": self.provenance,
            "sensor": asdict(self.sensor) if self.sensor else None,
        }

    def circle_of_confusion_pixels(self, depth_m):
        c = self.config
        f, focus = c.focal_length_mm * 0.001, c.focus_distance_m
        torch._assert_async(
            torch.isfinite(depth_m).all() & (depth_m > f).all(),
            "Finite object depths beyond focal length required",
        )
        return (
            f
            * f
            * (focus - depth_m).abs()
            / (c.f_number * (focus - f) * depth_m * c.pixel_pitch_um * 1e-6)
        )

    def depth_blur(self, image, depth_m):
        if (
            depth_m.shape != (image.shape[0], 1, *image.shape[-2:])
            or depth_m.device != image.device
        ):
            raise ValueError("depth_m must be [B,1,H,W] on image device")
        sigma = self.circle_of_confusion_pixels(depth_m) / 4
        torch._assert_async(
            (sigma <= self.config.max_sigma_pixels).all(),
            "Defocus exceeds the configured finite PSF support",
        )
        # Piecewise-linear interpolation in variance. Accumulate one blurred image
        # at a time rather than retaining a [levels,B,C,H,W] tensor bank.
        upper = torch.bucketize(sigma.contiguous(), self.sigmas).clamp(
            1, len(self.sigmas) - 1
        )
        lower = upper - 1
        fraction = (sigma.square() - self.sigmas[lower].square()) / (
            self.sigmas[upper].square() - self.sigmas[lower].square()
        )
        result = image * ((lower == 0) * (1 - fraction))
        for index in range(4):
            kernel = getattr(self, f"psf_{index}")
            radius = len(kernel) // 2
            channels = image.shape[1]
            horizontal = kernel[None, None, None].expand(channels, 1, 1, -1)
            vertical = kernel[None, None, :, None].expand(channels, 1, -1, 1)
            blurred = F.conv2d(
                F.pad(image, (radius, radius, 0, 0), mode="replicate"),
                horizontal,
                groups=channels,
            )
            blurred = F.conv2d(
                F.pad(blurred, (0, 0, radius, radius), mode="replicate"),
                vertical,
                groups=channels,
            )
            weight = (lower == index + 1) * (1 - fraction) + (
                upper == index + 1
            ) * fraction
            result = result + blurred * weight
        return result

    def sensor_parameters(self, parameters):
        if self.sensor is None:
            return parameters
        full_well = self.sensor.full_well_electrons
        return {
            **parameters,
            "shot_noise": torch.full_like(
                parameters["shot_noise"], 1 / math.sqrt(full_well)
            ),
            "read_noise": torch.full_like(
                parameters["read_noise"], self.sensor.read_noise_electrons / full_well
            ),
        }

    def forward(self, image, parameters, generator=None, *, depth_m=None):
        return super().forward(
            image, self.sensor_parameters(parameters), generator, depth_m=depth_m
        )

    def optical_blur(self, image, sigma, motion, angle, *, depth_m=None):
        if depth_m is None:
            depth_m = torch.full_like(image[:, :1], self.config.camera_distance_m)
        image = self.depth_blur(image, depth_m)
        return super().optical_blur(image, sigma, motion, angle)
