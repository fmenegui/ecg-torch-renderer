"""Pinned native Open-ECG-Digitizer behind the common hybrid signal contract."""

import hashlib
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from .selective import SignalProposal

NPJ_COMMIT = "97a15087d4abcda843da8c58ee74b1d8f47e6f9a"
NPJ_WEIGHTS = {
    "unet_weights_07072025.pt": "17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b",
    "lead_name_unet_weights_07072025.pt": "840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2",
}


def canonical_layout(name):
    standard = {"standard_3x4": "3x4", "standard_3x4_with_r1": "3x4+1R",
                "standard_3x4_with_r2": "3x4+2R", "standard_3x4_with_r3": "3x4+3R",
                "standard_6x2": "6x2", "standard_12x1": "12x1"}
    return standard.get(name, name)


class NativeNpjBackend:
    """Serial inference on explicit public time axes, native preset otherwise.

    Sampling metadata is supplied by the caller. No reference signal or true
    layout enters this backend. The model's own missing samples stay missing.
    """
    def __init__(self, upstream, *, device="cpu", max_side=3000):
        root = Path(upstream).resolve()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=root, text=True).splitlines()
        if commit != NPJ_COMMIT or set(changed) - {"weights/" + k for k in NPJ_WEIGHTS}:
            raise ValueError("Unchanged pinned npj source is required")
        for name, expected in NPJ_WEIGHTS.items():
            if hashlib.sha256((root / "weights" / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Native checkpoint hash mismatch")
        if torch.device(device).type not in ["cpu", "cuda"]:
            raise ValueError("CPU or CUDA must be selected explicitly")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, unavailable. No CPU substitution")
        if type(max_side) is not int or not 256 <= max_side <= 4096:
            raise ValueError("A bounded native input side is required")
        limit_file = Path("/sys/fs/cgroup/memory.max")
        if torch.device(device).type == "cpu" and max_side > 2048 and limit_file.exists():
            limit = limit_file.read_text().strip()
            if limit.isdigit() and int(limit) < 12 * 1024**3:
                raise MemoryError("Native 3000-pixel CPU inference exceeded an 8 GiB engineering host. Use CUDA or an explicitly separate smaller CPU profile")
        sys.path.insert(0, str(root))
        from src.config.default import get_cfg
        from src.model.inference_wrapper import InferenceWrapper
        cfg = get_cfg(str(root / "src/config/inference_wrapper.yml"))
        cfg.MODEL.KWARGS.device = device
        cfg.MODEL.KWARGS.resample_size = max_side
        settings = cfg.MODEL.KWARGS.config
        settings.LAYOUT_IDENTIFIER.KWARGS.device = device
        settings.LAYOUT_IDENTIFIER.config_path = str(root / "src/config/lead_layouts_all.yml")
        settings.LAYOUT_IDENTIFIER.unet_config_path = str(root / "src/config/lead_name_unet.yml")
        settings.LAYOUT_IDENTIFIER.unet_weight_path = str(root / "weights/lead_name_unet_weights_07072025.pt")
        settings.SEGMENTATION_MODEL.weight_path = str(root / "weights/unet_weights_07072025.pt")
        self.model = InferenceWrapper(**cfg.MODEL.KWARGS)
        self.max_side, self.device, self.config = max_side, device, cfg
        self._lock = threading.Lock()

    def _validate_image(self, rgb, samples):
        if type(samples) is not int or samples < 1:
            raise ValueError("Explicit positive public output length required")
        array = np.asarray(rgb)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("uint8 RGB required")
        return array

    def _infer(self, array, samples):
        self.model.identifier.target_num_samples = samples
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float()
        torch.manual_seed(20261026)
        result = self.model(tensor, layout_should_include_substring=None)
        return result

    def _proposal(self, result, fs, samples, source_id):
        canonical = result["signal"]["canonical_lines"]
        if canonical is None:
            raise ValueError("Native pipeline returned no canonical waveform")
        values = canonical.detach().cpu().numpy().astype(np.float64).squeeze() / 1000.0
        if values.shape != (12, samples):
            raise ValueError("Native result changed the requested canonical time axis")
        raw_layout = result["layout_name"]
        layout = canonical_layout(raw_layout)
        cost = float(result["signal"]["layout_matching_cost"])
        metadata = {"backend": "npj-native", "source_commit": NPJ_COMMIT,
                    "published_weights_sha256": NPJ_WEIGHTS, "native_layout": raw_layout,
                    "layout_matching_cost": cost if np.isfinite(cost) else None,
                    "layout_fallback": not np.isfinite(cost), "max_side": self.max_side,
                    "device": self.device, "native_output_unit": "uV", "public_sample_count": samples,
                    "assumed_gain_mm_per_mv": 10.0, "print_gain_inferred_from_image": False,
                    "pixel_spacing_mm": {k: float(v) for k, v in result["pixel_spacing_mm"].items()}}
        return SignalProposal(values, fs, source_id, layout, metadata=metadata)

    def predict(self, rgb, *, fs, samples, source_id):
        array = self._validate_image(rgb, samples)
        with self._lock, torch.inference_mode():
            result = self._infer(array, samples)
            return self._proposal(result, fs, samples, source_id)

    def predict_file(self, path, *, fs, samples):
        """Decode EXIF orientation once, then run the unchanged native model.

        Source identity remains the original file hash. The pixel permutation
        is retained as input provenance. No waveform-reference information is
        used to choose orientation, and no output timing origin is inferred.
        """
        from .image_io import decode_ecg_rgb
        rgb, decoding = decode_ecg_rgb(path)
        proposal = self.predict(rgb, fs=fs, samples=samples, source_id=decoding['source_id'])
        proposal.metadata['input_rgb_decoding'] = decoding
        return proposal

    def predict_observed(self, rgb, *, fs, samples, source_id):
        from .npj_trace import NativeTraceObserver
        array = self._validate_image(rgb, samples)
        with self._lock, torch.inference_mode():
            with NativeTraceObserver(self.model) as observer:
                result = self._infer(array, samples)
            direct = self._proposal(result, fs, samples, source_id)
            observer.report['pixel_spacing_mm'] = {k: float(v) for k, v in result['pixel_spacing_mm'].items()}
            observer.arrays['source_points'] = result['source_points'].detach().cpu().numpy().copy()
            observer.report['source_id'] = source_id
            observer.report['observation_shapes'] = {k: list(v.shape) for k, v in observer.arrays.items()}
            return direct, observer
