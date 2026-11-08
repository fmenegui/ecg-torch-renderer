"""Failure-directed exact-area control. Not a fresh held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

from axis_aligned_area import AxisAlignedAreaSampler
from streaming_sampler import OutputGrid, sample_image
from benchmark import inventory
from evaluate import HERE, digest


AXIS_PROTOCOL_SHA = "5aa74013d005426eb1dabfd636d899ed72ec78e9bcca733b085361e1596a03e2"


def cost_worker():
    source = np.random.default_rng(20261023).integers(0, 256, (1100, 1400, 3), np.uint8)
    grid = OutputGrid((768, 1024), (0., 0.), (1., 1.))
    timings, hashes = [], []
    for repeat in range(3):
        started = time.perf_counter()
        sampler = AxisAlignedAreaSampler(source, np.diag([1.1, 1.15]), [40., 50.])
        built = time.perf_counter()
        image, mean, fraction = sampler.sample(grid, max_tile_pixels=16384)
        finished = time.perf_counter()
        timings.append({"build_seconds": built-started, "render_seconds": finished-built,
                        "total_seconds": finished-started})
        hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
        integral_bytes = sampler.integral.nbytes
        del sampler, image, mean, fraction
    print(json.dumps({"timings": timings, "median_total_seconds": float(np.median([x["total_seconds"] for x in timings])),
                      "integral_image_bytes": integral_bytes,
                      "collected_image_mean_and_support_bytes": 768*1024*(3+3*8+8),
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      "repeated_output_sha256": hashes,
                      "all_repeats_identical": len(set(hashes)) == 1}))


def run(out):
    if digest(HERE/"axis_area_protocol.json") != AXIS_PROTOCOL_SHA:
        raise ValueError("frozen axis-area protocol mismatch")
    cv2.setNumThreads(1)
    out.mkdir(parents=True, exist_ok=True)
    records, predictions = [], []
    grid = OutputGrid((1, 1), (0., 0.), (1., 1.))
    arms = ["exact-source-cell-area", "area-bilinear-2", "area-bilinear-4", "area-bilinear-8"]
    for width in (1, 2, 5):
        source = np.full((65, 65, 3), 255, np.uint8)
        source[:, 32:32+width] = 0
        for scale in (2, 4, 8, 16, 32):
            for phase in (0., .25, .5, .75):
                offset = np.array([32.+phase, 32.])
                lower, upper = offset[0]-scale/2, offset[0]+scale/2
                overlap = max(0., min(upper, 31.5+width)-max(lower, 31.5))
                reference = 255*(1-overlap/scale)
                sampler = AxisAlignedAreaSampler(source, np.diag([float(scale)]*2), offset)
                image, mean, fraction = sampler.sample(grid)
                row = {"width_source_pixels": width, "scale_source_pixels_per_output_pixel": scale,
                       "phase_source_pixels": phase, "reference_encoded_mean": reference,
                       "exact_cell_mean": float(mean[0, 0, 0]),
                       "exact_cell_mean_abs_error": abs(float(mean[0, 0, 0])-reference),
                       "exact_cell_u8": int(image[0, 0, 0]),
                       "source_cell_area_fraction": float(fraction[0, 0])}
                pred = [int(image[0, 0, 0])]
                for n in (2, 4, 8):
                    result = sample_image(source, lambda p: p*scale+offset, grid, subgrid=n)
                    value = int(result.image[0, 0, 0])
                    row[f"bilinear_{n}_u8"] = value
                    pred.append(value)
                records.append(row)
                predictions.append(pred)
    cost = json.loads(subprocess.run([sys.executable, str(Path(__file__)), "--cost-worker"],
                                    capture_output=True, text=True, check=True).stdout)
    main = next(row for row in records if row["width_source_pixels"] == 1 and
                row["scale_source_pixels_per_output_pixel"] == 32 and row["phase_source_pixels"] == 0)
    report = {"schema": "m4-exact-axis-area-results-v1", "protocol_sha256": AXIS_PROTOCOL_SHA,
              "source_sha256": {p.name: digest(p) for p in [HERE/"axis_aligned_area.py", Path(__file__)]},
              "development_status": "Positive control developed after the independent fixed-grid counterexample. This is failure-directed development, not an unseen validation bank.",
              "scenes": len(records), "arms": arms, "main_counterexample": main,
              "exact_area_largest_mean_abs_error": max(row["exact_cell_mean_abs_error"] for row in records),
              "exact_area_u8_disagreements_with_rounded_reference": sum(row["exact_cell_u8"] != int(np.rint(row["reference_encoded_mean"])) for row in records),
              "complete_disappearances": {arm: sum(row[key] == 255 and row["reference_encoded_mean"] < 255
                                                    for row in records)
                                          for arm, key in zip(arms, ["exact_cell_u8", "bilinear_2_u8", "bilinear_4_u8", "bilinear_8_u8"])},
              "cost": cost, "hardware": inventory(),
              "scope": "Exact area of a source-cell constant reconstruction under diagonal affine maps only. Different reconstruction model from bilinear. No general rectification or radiometric claim.",
              "numerical_failure_found_and_fixed": "Initial independent-overlap test found prefix-subtraction residue at zero alpha, amplified by unpremultiplication. Initial failing log is retained. A conservative prefix arithmetic guard now rejects unresolved tiny boxes and zeroes alpha only within estimated arithmetic uncertainty. Final independent controls pass.",
              "cases": records}
    (out/"axis-area.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(out/"axis-line-family.npz", arms=np.asarray(arms), predictions_u8=np.array(predictions, np.uint8),
                        width_scale_phase=np.array([[r["width_source_pixels"], r["scale_source_pixels_per_output_pixel"],
                                                      r["phase_source_pixels"]] for r in records]),
                        reference_encoded_mean=np.array([r["reference_encoded_mean"] for r in records]))
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    if report["exact_area_u8_disagreements_with_rounded_reference"]:
        raise SystemExit("positive control failed, retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"results")
    parser.add_argument("--cost-worker", action="store_true")
    args = parser.parse_args()
    if args.cost_worker:
        cost_worker()
    else:
        run(args.out)
