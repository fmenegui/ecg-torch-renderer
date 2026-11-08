"""Experimental native npj downstream modules on an externally rectified RGB.

The published modules and checkpoints remain unchanged. This composition
deliberately omits native perspective, cropping, automatic rotation, TPS and
grid-pitch estimation. The sampler supplies its own inferred physical density.
It is a separately named ablation, never the original article's full pipeline.
"""

import math

import numpy as np
import torch

from .npj_backend import NPJ_COMMIT, NPJ_WEIGHTS, canonical_layout
from .selective import SignalProposal


class RectifiedNpjBackend:
    """Share verified native weights, without applying a second geometry fit."""
    def __init__(self, native):
        self.native = native

    def predict(self, rgb, *, fs, samples, source_id, pixels_per_mm):
        array = np.asarray(rgb)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Externally rectified uint8 RGB required")
        if min(array.shape[:2]) < 2 or max(array.shape[:2]) > self.native.max_side:
            raise ValueError("External canvas exceeds the declared side budget")
        if type(samples) is not int or samples < 1:
            raise ValueError("Explicit positive public sample count required")
        if isinstance(pixels_per_mm, bool) or not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
            raise ValueError("The original-RGB sampler must supply a positive finite density")
        if array.min() == array.max():
            raise ValueError("A constant image cannot be normalized into an ECG")
        model = self.native.model
        with self.native._lock, torch.inference_mode():
            model.times = {}
            model.identifier.target_num_samples = samples
            torch.manual_seed(20261026)
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float().to(self.native.device)
            tensor = model.min_max_normalize(tensor)
            signal_prob, _, text_prob = model._get_feature_maps(tensor)
            signals = model.signal_extractor(signal_prob.squeeze())
            layout = model.identifier(signals, text_prob, float(pixels_per_mm), layout_should_include_substring=None)
            canonical = layout.get("canonical_lines")
            if canonical is None:
                raise ValueError("Native downstream modules returned no canonical waveform")
            values = canonical.detach().cpu().numpy().astype(np.float64)
            if values.ndim == 3 and values.shape[0] == 1:
                values = values[0]
            values = values / 1000.0
            if values.shape != (12, samples):
                raise ValueError("Native downstream modules changed the public time axis")
            raw_layout = layout.get("layout", "Unknown layout")
            cost = float(layout.get("cost", 1.0))
        metadata = {"backend": "npj-downstream-external-geometry", "source_commit": NPJ_COMMIT,
            "published_weights_sha256": NPJ_WEIGHTS, "native_layout": raw_layout,
            "layout_matching_cost": cost if np.isfinite(cost) else None,
            "layout_fallback": not np.isfinite(cost), "device": self.native.device,
            "external_density_pixels_per_mm": float(pixels_per_mm), "density_is_inferred": True,
            "native_perspective_applied": False, "native_crop_applied": False,
            "native_auto_rotation_applied": False, "native_dewarping_applied": False,
            "native_grid_pitch_search_applied": False, "image_resampled_by_downstream_adapter": False,
            "native_output_unit": "uV", "assumed_gain_mm_per_mv": 10.0,
            "reference_used": False, "original_article_full_pipeline": False}
        return SignalProposal(values, fs, source_id, canonical_layout(raw_layout), metadata=metadata)
