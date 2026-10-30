"""Experimental selective ECG inference with explicit physical output metadata."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .grid_backend import AdaptiveGridRectifier
from .npj_backend import NativeNpjBackend
from .runner import _write_json, _write_prediction
from .selective import SelectiveDigitizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--npj-source", required=True, type=Path)
    parser.add_argument("--grid-checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fs", required=True, type=float)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--gain-mm-per-mv", type=float, default=10.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-side", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in [args.fs, args.duration]) or args.gain_mm_per_mv != 10:
        parser.error("Positive sampling metadata and the currently supported 10 mm/mV convention are required")
    samples = round(args.fs * args.duration)
    if not 1 <= samples <= 120000 or not 1 <= args.threads <= 32:
        parser.error("Requested output or thread count exceeds the bounded interface")
    if args.out.exists():
        parser.error("Choose a new output directory")
    torch.set_num_threads(args.threads)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    backend = NativeNpjBackend(args.npj_source, device=args.device, max_side=args.max_side)
    rectifier = AdaptiveGridRectifier(args.grid_checkpoint, device=args.device, max_side=args.max_side) if args.grid_checkpoint else None
    digitizer = SelectiveDigitizer(backend, rectifier)
    started = time.perf_counter()
    result, decision = digitizer.predict(rgb, fs=args.fs, samples=samples)
    args.out.mkdir(parents=True)
    _write_prediction(args.out / "signal.npz", {"values_mv": result.values_mv, "fs": result.fs,
                                              "origin_s": result.origin_s, "lead_names": np.asarray(result.lead_names), "units": np.asarray("mV")})
    receipt = {"schema": "hybrid-digitizer-cli-v1", "experimental": True, "clinical_validity_established": False,
               "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
               "signal_sha256": hashlib.sha256((args.out / "signal.npz").read_bytes()).hexdigest(),
               "fs": result.fs, "samples": samples, "source_id": result.source_id,
               "finite_output_fraction": float(np.isfinite(result.values_mv).mean()),
               "finite_output_is_not_printed_support_accuracy": True, "policy": asdict(digitizer.policy),
               "decision": decision, "selected_backend_metadata": result.metadata,
               "public_sampling_metadata_supplied_by_caller": True, "assumed_print_gain_mm_per_mv": 10,
               "wall_seconds_including_branches": time.perf_counter() - started}
    if rectifier is not None:
        receipt["geometry"] = rectifier.last_report
        if rectifier.last_observations:
            observations = args.out / "geometry-observations.npz"
            _write_prediction(observations, rectifier.last_observations)
            receipt["geometry_observations"] = {"file": observations.name,
                "sha256": hashlib.sha256(observations.read_bytes()).hexdigest(), "reference_used": False}
    _write_json(args.out / "receipt.json", receipt)
    print(json.dumps({"output": str(args.out), "selected": decision["selected"], "reason": decision["reason"]}))


if __name__ == "__main__":
    main()
