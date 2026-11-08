"""Run our declared Dotter hypothesis, followed by disclosed-mechanism Gridder.

Tile averaging and widths are engineering choices, not commercial defaults.
"""

import argparse

import numpy as np
from PIL import Image

from reproductions.common import environment, new_run, require_cuda, sha256, write_json
from reproductions.powerful_medical.gridder import extract_centroids, organize, warp_cells


def predict_tiled(model, rgb, device="cpu", tile=256, stride=128):
    import torch
    from torch.nn import functional as F
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not 1 <= stride <= tile:
        raise ValueError("Original uint8 RGB and a complete tile cover required")
    height, width = rgb.shape[:2]
    total = np.zeros((height, width), float)
    counts = np.zeros_like(total)
    model.eval()
    with torch.inference_mode():
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                crop = rgb[y:y + tile, x:x + tile].copy()
                h, w = crop.shape[:2]
                tensor = torch.from_numpy(crop).permute(2, 0, 1).float()[None].to(device) / 255
                tensor = F.pad(tensor, (0, tile - w, 0, tile - h), mode="replicate")
                probability = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()[:h, :w]
                total[y:y + h, x:x + w] += probability
                counts[y:y + h, x:x + w] += 1
    if not (counts > 0).all():
        raise AssertionError("Uncovered source pixels")
    return (total / counts).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    require_cuda(args.device)
    import torch
    from reproductions.powerful_medical.dotter import Dotter
    out = new_run(args.out)
    model = Dotter().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True), strict=True)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    receipt = {"schema": "pm-adapted-pipeline-v1", "commercial_implementation": False, "image_sha256": sha256(args.image), "checkpoint_sha256": sha256(args.checkpoint), "environment": environment(), "dotter_choices": model.architecture_choices_not_disclosed_by_authors, "tiling_choice": {"tile": 256, "stride": 128, "fusion": "mean_probability", "padding": "replicate", "rgb_divisor": 255}, "status": "started", "full_waveform_extraction": False}
    try:
        probability = predict_tiled(model, rgb, args.device)
        np.savez_compressed(out / "dotter-probability.npz", probability=probability)
        Image.fromarray(np.rint(probability * 255).astype(np.uint8)).save(out / "dotter-mask.png")
        grid = organize(extract_centroids(probability))
        result = warp_cells(rgb, grid)
        Image.fromarray(result["rgb"]).save(out / "relative-partial-grid.png")
        np.savez_compressed(out / "geometry.npz", points_xy=grid.points_xy, ids_rc=grid.ids_rc, observed_nodes=grid.observed, support=result["support"], map_x=result["map_x"], map_y=result["map_y"])
        receipt.update(status="returned_relative_partial_geometry", grid=grid.diagnostics)
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
