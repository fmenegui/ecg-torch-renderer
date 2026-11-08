"""Execute the disclosed geometry mechanism from a Dotter mask or point list."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from reproductions.common import environment, new_run, sha256, write_json
from reproductions.powerful_medical.gridder import extract_centroids, infer_missing, organize, warp_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dotter-mask")
    inputs.add_argument("--points", help="Detected xy array, or oracle array in a labeled component diagnostic")
    parser.add_argument("--oracle-points", action="store_true")
    parser.add_argument("--infer-missing", action="store_true")
    parser.add_argument("--pixels-per-cell", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = new_run(args.out)
    mask_path = Path(args.dotter_mask) if args.dotter_mask else None
    points = extract_centroids(np.asarray(Image.open(mask_path).convert("L"))) if mask_path else np.load(args.points, allow_pickle=False)
    receipt = {"schema": "pm-disclosed-mechanism-v1", "commercial_implementation": False, "oracle_points": args.oracle_points, "image_sha256": sha256(args.image), "input_points": len(points), "environment": environment(), "status": "started", "physical_calibration": "not_resolved_from_grid_alone"}
    try:
        grid = organize(points)
        if args.infer_missing:
            grid = infer_missing(grid)
        result = warp_cells(np.asarray(Image.open(args.image).convert("RGB")), grid, args.pixels_per_cell, include_inferred=args.infer_missing)
        Image.fromarray(result["rgb"]).save(out / "relative-partial-grid.png")
        np.savez_compressed(out / "geometry.npz", points_xy=grid.points_xy, ids_rc=grid.ids_rc, observed_nodes=grid.observed, support=result["support"], inferred_support=result["inferred_support"], map_x=result["map_x"], map_y=result["map_y"])
        receipt.update(status="completed_component", diagnostics=grid.diagnostics, output_support_fraction=float(result["support"].mean()), complete_digitization=False)
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
