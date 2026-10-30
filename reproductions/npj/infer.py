"""Run the complete native npj chain on immutable image-only inputs.

References are deliberately unavailable to this worker. Output uses native uV.
Native presets are preserved, including layout hints when explicitly supplied.
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from reproductions.common import environment, new_run, require_cuda, sha256, verify_source, write_json
from reproductions.npj.protocol import COMMIT, WEIGHTS, inference_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--preset", choices=["ahus", "george-moody-2024", "generic"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--layout-hint", help="Native public filename hint, never a scoring reference")
    parser.add_argument("--segmentation-checkpoint", help="Explicit retrained main-network ablation")
    parser.add_argument("--lead-checkpoint", help="Explicit retrained lead-name-network ablation")
    args = parser.parse_args()
    upstream = Path(args.upstream).resolve()
    source = verify_source(upstream, COMMIT)
    require_cuda(args.device)
    for name, expected in WEIGHTS.items():
        if sha256(upstream / "weights" / name) != expected:
            raise ValueError("Published checkpoint hash mismatch")
    out = new_run(args.out)
    sys.path.insert(0, str(upstream))
    import torch
    from PIL import Image
    from yacs.config import CfgNode
    from src.model.inference_wrapper import InferenceWrapper
    torch.set_num_threads(args.threads)
    torch.manual_seed(1973)
    config = inference_config(upstream, args.preset, args.device)
    components = config["MODEL"]["KWARGS"]["config"]
    if args.segmentation_checkpoint:
        components["SEGMENTATION_MODEL"]["weight_path"] = str(Path(args.segmentation_checkpoint).resolve())
    if args.lead_checkpoint:
        components["LAYOUT_IDENTIFIER"]["unet_weight_path"] = str(Path(args.lead_checkpoint).resolve())
    active_weights = {"segmentation": sha256(components["SEGMENTATION_MODEL"]["weight_path"]), "lead_names": sha256(components["LAYOUT_IDENTIFIER"]["unet_weight_path"])}
    write_json(out / "effective-config.json", config)
    kwargs = config["MODEL"]["KWARGS"]
    model = InferenceWrapper(config=CfgNode(kwargs.pop("config")), **kwargs)
    results = []
    used = set()
    for image_path in args.image:
        image_path = Path(image_path).resolve()
        name = image_path.stem
        if name in used:
            raise ValueError("Image stems must be unique within a native run")
        used.add(name)
        row = {"case_id": name, "image_sha256": sha256(image_path), "status": "scheduled"}
        start = time.perf_counter()
        try:
            array = np.array(Image.open(image_path).convert("RGB"), copy=True)
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            with torch.inference_mode():
                result = model(tensor, layout_should_include_substring=args.layout_hint)
            native = result["signal"]["canonical_lines"]
            if native is None:
                row["status"] = "no_signal"
            else:
                values = native.detach().cpu().numpy().squeeze()
                np.savez_compressed(out / (name + ".npz"), values=values, unit="uV")
                row.update(status="returned", shape=list(values.shape), finite_fraction=float(np.isfinite(values).mean()), output_sha256=sha256(out / (name + ".npz")))
            row["predicted_layout"] = result["layout_name"]
            aligned = result["aligned"]["image"].squeeze().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray(np.rint(np.clip(aligned, 0, 1) * 255).astype(np.uint8)).save(out / (name + ".aligned.png"))
        except Exception as error:
            row.update(status="exception", error_type=type(error).__name__, error=str(error))
        row["seconds_including_native_forward"] = time.perf_counter() - start
        results.append(row)
        write_json(out / "receipt.json", {"schema": "npj-native-inference-v1", "source": source, "environment": environment(), "preset": args.preset, "layout_hint": args.layout_hint, "active_weight_sha256": active_weights, "retrained_weight_ablation": bool(args.segmentation_checkpoint or args.lead_checkpoint), "output_unit": "uV", "results": results, "scheduled_images": len(args.image), "paper_cohort_reproduction_completed": False})
    if any(r["status"] == "exception" for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
