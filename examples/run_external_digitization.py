"""Frozen external RGB-to-signal pilot with every scheduled failure retained.

Compare a pinned Open-ECG-Digitizer with the same downstream pipeline after
our M6 original-RGB rectifier. Reference waveforms, layouts and printed masks
are loaded only AFTER prediction. No per-image upstream preset is selected.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.external_evaluation import score_external_signals
from ecg_gpu.digitization.page_orientation import infer_text_orientation
from ecg_gpu.digitization.runner import _write_json, _write_prediction
from examples.digitization_npj_probe import UPSTREAM_COMMIT, WEIGHTS
from examples.rectify_denoising_grid import _write_rgb_png, rectify


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native(root, max_side):
    root = Path(root).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=root, text=True
    ).splitlines()
    if commit != UPSTREAM_COMMIT or set(changed) - {
        "weights/" + name for name in WEIGHTS
    }:
        raise ValueError("Require unchanged pinned upstream source and configuration")
    for filename, expected in WEIGHTS.items():
        if digest(root / "weights" / filename) != expected:
            raise ValueError("Upstream weight mismatch: " + filename)
    # Hydrated LFS files legitimately differ from their tracked pointer files.
    # Their full SHA-256 is checked above, never exempted from verification.
    sys.path.insert(0, str(root))
    from src.config.default import get_cfg
    from src.model.inference_wrapper import InferenceWrapper

    cfg = get_cfg(str(root / "src/config/inference_wrapper.yml"))
    cfg.MODEL.KWARGS.device = "cpu"
    cfg.MODEL.KWARGS.resample_size = max_side
    settings = cfg.MODEL.KWARGS.config
    settings.LAYOUT_IDENTIFIER.KWARGS.device = "cpu"
    settings.LAYOUT_IDENTIFIER.KWARGS.target_num_samples = 5000
    settings.LAYOUT_IDENTIFIER.config_path = str(
        root / "src/config/lead_layouts_all.yml"
    )
    settings.LAYOUT_IDENTIFIER.unet_config_path = str(
        root / "src/config/lead_name_unet.yml"
    )
    settings.LAYOUT_IDENTIFIER.unet_weight_path = str(
        root / "weights/lead_name_unet_weights_07072025.pt"
    )
    settings.SEGMENTATION_MODEL.weight_path = str(
        root / "weights/unet_weights_07072025.pt"
    )
    model = InferenceWrapper(**cfg.MODEL.KWARGS)
    if model.segmentation_model.training or model.identifier.unet.training:
        raise RuntimeError("Pretrained models must run in evaluation mode")
    return model, cfg


def predict_native(model, image, cache=None):
    with Image.open(image) as im:
        rgb = np.asarray(im.convert("RGB"))
    key = (rgb.shape, hashlib.sha256(rgb.tobytes()).hexdigest())
    if cache is not None and key in cache:
        values, metadata = cache[key]
        return values.copy(), dict(metadata, exact_rgb_prediction_cache_hit=True)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        torch.manual_seed(20261026)
        result = model(tensor, layout_should_include_substring=None)
    canonical = result["signal"]["canonical_lines"]
    if canonical is None:
        raise ValueError("No canonical waveform returned")
    values = canonical.detach().cpu().numpy().astype(np.float64) / 1000.0
    if values.shape != (12, 5000) or np.isinf(values).any():
        raise ValueError("Malformed canonical waveform")
    cost = float(result["signal"]["layout_matching_cost"])
    metadata = {
        "predicted_layout": result["layout_name"],
        "layout_matching_cost": cost if np.isfinite(cost) else None,
        "layout_fallback": not np.isfinite(cost),
        "stage_seconds": dict(model.times),
        "input_shape": list(rgb.shape),
        "pixel_spacing_mm": {
            k: float(v) for k, v in result["pixel_spacing_mm"].items()
        },
        "decoded_rgb_sha256": key[1],
        "exact_rgb_prediction_cache_hit": False,
    }
    if cache is not None:
        cache[key] = (values.copy(), metadata)
    return values, metadata


def run(manifest, out, upstream, checkpoint, max_side=1500, threads=2):
    manifest, out = Path(manifest).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    data = json.loads(manifest.read_text())
    if data.get("schema") != "pm-ecg-external-pilot-v1" or not data.get("cases"):
        raise ValueError("A prepared external pilot manifest is required")
    for case in data["cases"]:
        if digest(manifest.parent / case["image"]) != case["image_sha256"]:
            raise ValueError("Input image checksum mismatch")
    torch.set_num_threads(threads)
    out.mkdir(parents=True)
    model, config = load_native(upstream, max_side)
    _write_json(
        out / "effective-config.json",
        json.loads(json.dumps(dict(config), default=dict)),
    )
    report = {
        "schema": "ecg-external-pilot-run-v1",
        "status": "running",
        "manifest_sha256": digest(manifest),
        "script_sha256": digest(__file__),
        "scorer_sha256": digest(
            Path(__file__).resolve().parents[1]
            / "ecg_gpu/digitization/external_evaluation.py"
        ),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_weights_sha256": WEIGHTS,
        "our_checkpoint_sha256": digest(checkpoint),
        "cpu_max_side": max_side,
        "threads": threads,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "arms": ["npj_native", "text_then_npj", "m6_then_npj"],
        "inference_inputs": "RGB only, fixed 10 second output at 500 Hz, fixed 10 mm/mV convention",
        "selection_or_tuning_using_test_signals": False,
        "failures": "No reconstructed waveform. Score-only zero penalty on every printed reference sample.",
        "limits": [
            "Small development pilot, no confirmatory holdout or population claim",
            "External image acquisition, unknown upstream training waveform overlap",
            "CPU maximum side differs from published 3000 pixel preset",
            "M6 is a preprocessing ablation with unchanged native downstream geometry/layout/extraction",
            "No execution of the proprietary Powerful Medical system",
            "Coverage failures and SNR use our declared protocol, not published study denominators",
        ],
        "cases": [],
    }
    _write_json(out / "report.json", report)
    native_cache = {}
    for case in data["cases"]:
        image_path = manifest.parent / case["image"]
        for arm in report["arms"]:
            case_out = out / case["image_id"] / arm
            case_out.mkdir(parents=True)
            record = {
                k: case[k]
                for k in (
                    "image_id",
                    "patient_id",
                    "recording_id",
                    "condition",
                    "image_sha256",
                )
            }
            record["arm"] = arm
            print(json.dumps({"starting": case["image_id"], "arm": arm}), flush=True)
            started = time.perf_counter()
            log = io.StringIO()
            try:
                with contextlib.redirect_stdout(log):
                    model_input = image_path
                    if arm == "text_then_npj":
                        with Image.open(image_path) as im:
                            rgb = np.asarray(im.convert("RGB"))
                        orientation = infer_text_orientation(
                            rgb, artifact_dir=case_out / "orientation"
                        )
                        record["orientation"] = orientation
                        rotated = np.ascontiguousarray(
                            np.rot90(rgb, orientation["quarter_turns_ccw"])
                        )
                        model_input = case_out / "oriented-original-rgb.png"
                        _write_rgb_png(rotated, model_input)
                    if arm == "m6_then_npj":
                        geometry = rectify(
                            Path(checkpoint),
                            image_path,
                            case_out / "rectification",
                            device="cpu",
                            pixels_per_mm=4.0,
                            profile="banding",
                            organizer="baseline",
                            page_orientation="text",
                        )
                        record["rectification"] = geometry
                        model_input = (
                            case_out / "rectification" / geometry["output_image"]
                        )
                    prediction, native = predict_native(
                        model, model_input, native_cache
                    )
                    record.update(native)
                _write_prediction(
                    case_out / "prediction.npz",
                    {"values_mv": prediction, "fs": np.asarray(500)},
                )
                record.update(
                    status="returned",
                    prediction=f"{case['image_id']}/{arm}/prediction.npz",
                    prediction_sha256=digest(case_out / "prediction.npz"),
                )
            except Exception as exc:
                prediction = np.full((12, 5000), np.nan)
                record.update(
                    status="failed", error=type(exc).__name__ + ": " + str(exc)
                )
            record["inference_wall_seconds"] = time.perf_counter() - started
            (case_out / "inference.log").write_text(log.getvalue())
            # Gold never reaches either inference arm, including model layout
            # selection, native output support, or our fallback decision.
            reference_path = manifest.parent / case["reference"]
            if digest(reference_path) != case["reference_sha256"]:
                raise ValueError("Reference checksum mismatch")
            with np.load(reference_path, allow_pickle=False) as reference:
                evaluation = score_external_signals(
                    reference["values_mv"], prediction, 500
                )
            record["evaluation"] = evaluation
            record["scoring_zero_missing_penalty_only"] = True
            report["cases"].append(record)
            _write_json(out / "report.json", report)
            print(
                json.dumps(
                    {
                        "completed": case["image_id"],
                        "arm": arm,
                        "status": record["status"],
                        "coverage": evaluation["coverage"],
                        "raw_all_printed_snr_db": evaluation[
                            "raw_all_printed_zero_missing"
                        ]["snr_db"],
                        "seconds": record["inference_wall_seconds"],
                    }
                ),
                flush=True,
            )
    report["status"] = "completed"
    _write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "out", "upstream", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.max_side < 512 or args.threads < 1:
        parser.error("max-side >= 512 and threads >= 1 required")
    run(**vars(args))
