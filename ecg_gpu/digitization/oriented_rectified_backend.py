"""Aspect-based quarter-turn control, without another perspective fit.

This reuses the native pipeline's landscape heuristic, not a semantic
orientation certificate. Feature maps receive an exact quarter-turn without
interpolation. Physical density is isotropic and is therefore unchanged.
"""

from dataclasses import replace
from types import SimpleNamespace

import torch

from .rectified_backend import RectifiedNpjBackend


class _OrientedModel:
    def __init__(self, model):
        self.original = model
        self.quarter_turns_clockwise = 0

    @property
    def times(self):
        return self.original.times

    @times.setter
    def times(self, value):
        self.original.times = value

    @property
    def identifier(self):
        return self.original.identifier

    @property
    def signal_extractor(self):
        return self.original.signal_extractor

    def min_max_normalize(self, image):
        return self.original.min_max_normalize(image)

    def _get_feature_maps(self, image):
        maps = self.original._get_feature_maps(image)
        self.quarter_turns_clockwise = int(image.shape[2] > image.shape[3])
        if self.quarter_turns_clockwise:
            return tuple(torch.rot90(value, k=3, dims=(2, 3)) for value in maps)
        return maps


class LandscapeRectifiedNpjBackend:
    def __init__(self, native):
        self.native = native

    def predict(self, rgb, *, fs, samples, source_id, pixels_per_mm):
        # A per-call proxy avoids patching a shared native module. Both
        # adapters acquire the same inference lock in RectifiedNpjBackend.
        model = _OrientedModel(self.native.model)
        proxy = SimpleNamespace(model=model, device=self.native.device,
            max_side=self.native.max_side, _lock=self.native._lock)
        result = RectifiedNpjBackend(proxy).predict(rgb, fs=fs, samples=samples, source_id=source_id,
                                                  pixels_per_mm=pixels_per_mm)
        return replace(result, metadata={**result.metadata,
            "backend": "npj-downstream-external-geometry-landscape-control",
            "native_auto_rotation_applied": bool(model.quarter_turns_clockwise),
            "feature_map_quarter_turns_clockwise": model.quarter_turns_clockwise,
            "orientation_rule": "native aspect heuristic, not ECG semantic evidence",
            "semantic_orientation_verified": False, "quarter_turn_interpolates_pixels": False})
