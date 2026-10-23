"""Replay every archived M3 analytical scene without changing its references."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from streaming_sampler import OutputGrid, sample_image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL_SHA = "228329fb3e8f50bcdb39911aa2ab91f3f7613f0ed6411d69ae8bdeff1900043e"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_module():
    path = ROOT/"m3/sampling/sampling_component.py"
    spec = importlib.util.spec_from_file_location("m4_archived_m3_sampling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(out):
    if digest(HERE/"protocol.json") != PROTOCOL_SHA:
        raise ValueError("frozen protocol mismatch")
    cv2.setNumThreads(1)
    out.mkdir(parents=True, exist_ok=True)
    old = old_module()
    m3_protocol = json.loads((ROOT/"m3/sampling/protocol.json").read_text())
    if digest(ROOT/"m3/sampling/protocol.json") != old.PROTOCOL_SHA256:
        raise ValueError("M3 frozen protocol mismatch")
    CompleteMap, _ = old.load_native(ROOT/"ecg-torch-renderer/ecg_gpu/digitization",
                                    m3_protocol["native_source_sha256"])
    grid = OutputGrid((41, 41), (2., 2.), (.25, .25))
    arms = json.loads((HERE/"protocol.json").read_text())["arms"]
    records, quality, sources = [], [], {}
    checked_values, mismatch_values, max_difference, scenes = 0, 0, 0, 0
    started = time.perf_counter()
    for map_spec in m3_protocol["maps"]:
        name = map_spec["id"]
        path = ROOT/f"m3/sampling/results/{name}.npz"
        sources[str(path.relative_to(ROOT))] = digest(path)
        mapping, _, _ = old.make_mapping(CompleteMap,
                                         map_spec["source_pixels_per_physical_mm_matrix"])
        with np.load(path) as archived:
            for index, gray in enumerate(archived["source_gray_u8"]):
                width, angle, phase = archived["scene_width_angle_offset"][index]
                source = np.repeat(gray[..., None], 3, -1)
                scenes += 1
                for arm in arms:
                    _, reconstruction, ntext = arm.split("-")
                    n = int(ntext)
                    archived_arm = list(archived["arms"]).index(arm)
                    expected_gray = 255-archived["darkness_u8"][index, archived_arm]
                    expected = np.repeat(expected_gray[..., None], 3, -1)
                    first = None
                    for budget in (512, 1681):
                        result = sample_image(source, mapping.inverse, grid, subgrid=n,
                                              reconstruction=reconstruction, max_tile_pixels=budget)
                        image = result.image
                        difference = np.abs(image.astype(np.int16)-expected.astype(np.int16))
                        mismatches = int(np.count_nonzero(difference))
                        checked_values += image.size
                        mismatch_values += mismatches
                        maximum = int(difference.max())
                        max_difference = max(max_difference, maximum)
                        tile_mismatches = 0 if first is None else int(np.count_nonzero(image != first))
                        records.append({"map": name, "scene": index, "width_mm": float(width),
                                        "angle": float(angle), "phase": float(phase), "arm": arm,
                                        "tile_pixels": budget, "mismatched_channel_values": mismatches,
                                        "maximum_abs_difference": maximum,
                                        "different_from_other_tile_budget": tile_mismatches,
                                        "all_source_cell_fractions_one": bool(np.all(result.source_cell_fraction == 1)),
                                        "stats": asdict(result.stats)})
                        if first is None:
                            first = image
                            reference = archived["ideal_coverage_f64"][index]
                            metric, _ = old.score_image(image, np.ones((41, 41), bool), reference.ravel(), 4.)
                            quality.append({"map": name, "scene": index, "width_mm": float(width),
                                            "arm": arm, **metric})
                if (index+1) % 35 == 0:
                    print(json.dumps({"map": name, "completed": index+1, "cases": scenes}), flush=True)
    summaries = []
    archived_summaries = json.loads((ROOT/"m3/sampling/results/results.json").read_text())["summaries"]
    for width in (.1, 1.):
        for arm in arms:
            selected = [r for r in quality if r["width_mm"] == width and r["arm"] == arm]
            summary = {"width_mm": width, "arm": arm, "cases": len(selected),
                       "disappearances": sum(r["complete_disappearance"] for r in selected),
                       "mean_pixel_MAE": float(np.mean([r["pixel_MAE"] for r in selected])),
                       "mean_abs_relative_energy_error": float(np.mean([abs(r["relative_line_energy_error"]) for r in selected])),
                       "mean_peak_abs_error": float(np.mean([r["peak_darkness_absolute_error"] for r in selected])),
                       "minimum_peak_darkness": min(r["peak_darkness"] for r in selected)}
            baseline = next(r for r in archived_summaries if r["width_mm"] == width and r["arm"] == arm)
            summary["largest_summary_difference_from_M3"] = max(
                abs(summary[key]-baseline[key]) for key in summary if key not in ("arm", "width_mm"))
            summaries.append(summary)
    report = {"schema": "m4-sampling-parity-results-v1", "protocol_sha256": PROTOCOL_SHA,
              "source_sha256": {p.name: digest(p) for p in [HERE/"streaming_sampler.py", HERE/"pngio.py", Path(__file__)]},
              "archive_sha256": sources, "python": platform.python_version(), "numpy": np.__version__,
              "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
              "scenes": scenes, "arms": len(arms), "tile_budgets": [512, 1681],
              "outputs_compared": len(records), "channel_values_compared": checked_values,
              "mismatched_channel_values": mismatch_values, "maximum_abs_difference": max_difference,
              "tile_invariance_failures": sum(r["different_from_other_tile_budget"] != 0 for r in records),
              "support_failures": sum(not r["all_source_cell_fractions_one"] for r in records),
              "wall_seconds": time.perf_counter()-started,
              "scope": "All archived scored 41x41 ROI pixels, not full-frame parity. No new geometry quality claim.",
              "quality_summaries_reproduced": summaries, "comparisons": records}
    (out/"parity.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))
    if mismatch_values or report["tile_invariance_failures"] or report["support_failures"]:
        raise SystemExit("Parity failure retained in parity.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"results")
    evaluate(parser.parse_args().out)
