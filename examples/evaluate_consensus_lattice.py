"""Paired development evaluation with fixed detections and untouched M3 queries.

The candidate sees only detector coordinates and confidence. Query reference
coordinates are passed to the scorer after inference. Primary scores retain
native orientation. Secondary D4 scores diagnose unresolved page orientation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg_gpu.digitization.consensus_lattice import organize_consensus_lattice
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.local_cells import locally_supported_cells
from ecg_gpu.digitization.page_orientation import rotate_lattice_ids
from examples.diagnose_digitization_stages import predictions_from_cells
from examples.rectification_comparison_metrics import (
    aggregate_patient_backend,
    paired_patient_bootstrap,
    score_predictions,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def failed_predictions(queries):
    return {
        d: {
            "coords": np.full_like(q["source"], np.nan),
            "claimed_eligible": np.zeros(len(q["source"]), bool),
            "native_footprint": np.zeros(len(q["source"]), bool),
        }
        for d, q in queries.items()
    }


def run(
    bank,
    comparison,
    observations,
    out,
    split,
    *,
    render=False,
    orientation_receipts=None,
):
    import cv2

    cv2.setNumThreads(1)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    schedule = json.loads((comparison / "schedule.json").read_text())
    cases = [c for c in schedule["cases"] if c["split"] == split]
    if not cases:
        raise ValueError("No scheduled cases in the selected split")
    bindings = []
    for case in cases:
        point_path = observations / case["key"] / "fresh-observations.npz"
        if not point_path.exists():
            point_path = comparison / "cases" / case["key"] / "fresh-observations.npz"
        if digest(bank / case["file"]) != case["sha256"]:
            raise ValueError("Capture differs: " + case["key"])
        if digest(comparison / case["query_file"]) != case["query_archive_sha256"]:
            raise ValueError("Queries differ: " + case["key"])
        binding = {
            **case,
            "points_path": str(point_path),
            "points_sha256": digest(point_path),
        }
        if orientation_receipts is not None:
            receipt = orientation_receipts / case["key"] / "orientation.json"
            binding.update(
                orientation_receipt=str(receipt), orientation_sha256=digest(receipt)
            )
        bindings.append(binding)
    protocol = {
        "schema": "ECG-consensus-lattice-v1",
        "development_only": True,
        "split": split,
        "cases": bindings,
        "new_training": False,
        "reference_used_for_inference": False,
        "semantic_orientation_method": "ECG text OCR"
        if orientation_receipts
        else "unresolved",
        "native_major_pitch_mm": 5.0,
        "registration": "anchor-only translation, frozen M3 scorer",
        "primary_threshold_mm": 0.5,
        "primary_domain": "paper",
        "render_requested": render,
        "full_refinement_strength": 0.0,
        "full_baseline": "B0, existing CLI default without optional point refinement",
        "full_geometry_seed": 1973,
        "RGB_cache": "within-case exact selected indices, IDs and supported cells, no approximate reuse",
        "candidate_selection": "supported cells minus twice conflicting constraints, no query scoring",
        "source_sha256": {
            p: digest(ROOT / p)
            for p in (
                "ecg_gpu/digitization/consensus_lattice.py",
                "ecg_gpu/digitization/page_orientation.py",
                "examples/evaluate_consensus_lattice.py",
                "examples/rectification_comparison_metrics.py",
                "examples/rectify_denoising_grid.py",
            )
        },
        "schedule_sha256": digest(comparison / "schedule.json"),
    }
    write_json(out / "protocol.json", protocol)
    rows, details = [], []
    for index, case in enumerate(bindings):
        started = time.perf_counter()
        key = case["key"]
        with np.load(case["points_path"], allow_pickle=False) as z:
            xy, confidence = z["xy"].copy(), z["scores"].copy()
        with np.load(comparison / case["query_file"], allow_pickle=False) as z:
            queries = {
                d: {
                    "source": z[d + "_source"].copy(),
                    "reference": z[d + "_reference"].copy(),
                }
                for d in ("anchors", "grid", "paper")
            }
        topologies = {}
        for name in ("baseline-cells", "consensus-cells"):
            try:
                topologies[name] = (
                    locally_supported_cells(xy, organize_lattice(xy, confidence))
                    if name == "baseline-cells"
                    else organize_consensus_lattice(xy, confidence)
                )
            except Exception as exc:
                topologies[name] = {
                    "selected_indices": [],
                    "ids": [],
                    "cells": [],
                    "status": "rejected",
                    "partial_status": "rejected",
                    "inference_failure": type(exc).__name__ + ": " + str(exc),
                }
        if orientation_receipts is not None:
            orientation = json.loads(Path(case["orientation_receipt"]).read_text())
            with np.load(bank / case["file"], allow_pickle=False) as z:
                image_hash = hashlib.sha256(
                    np.ascontiguousarray(z["image"]).tobytes()
                ).hexdigest()
            if orientation.get("rgb_sha256") != image_hash:
                raise ValueError("OCR receipt does not bind the original image: " + key)
            for method in ("baseline", "consensus"):
                topologies[method + "-text-cells"] = rotate_lattice_ids(
                    topologies[method + "-cells"], int(orientation["quarter_turns_ccw"])
                )
        folder = out / "cases" / key
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(bank / case["file"], allow_pickle=False) as z:
            image = z["image"].copy() if render else None
        full_cache = {}
        for name, topology in topologies.items():
            selected = np.asarray(topology["selected_indices"], int)
            ids = np.asarray(topology["ids"], int).reshape(-1, 2)
            write_json(folder / (name + "-topology.json"), topology)
            try:
                predicted, _ = predictions_from_cells(
                    xy[selected], ids, queries, topology["cells"]
                )
            except Exception as exc:
                predicted = failed_predictions(queries)
                details.append(
                    {
                        "key": key,
                        "method": name,
                        "inference_failure": type(exc).__name__ + ": " + str(exc),
                    }
                )
            variants = {name: predicted}
            if render:
                from examples.compare_rectification_methods import OurNativeMap
                from examples.rectify_denoising_grid import write_geometry_output

                target = folder / name.replace("-cells", "-full")
                full_key = hashlib.sha256(
                    selected.astype(np.int64).tobytes()
                    + ids.astype(np.int64).tobytes()
                    + np.asarray(sorted(map(tuple, topology["cells"])), np.int64)
                    .reshape(-1, 2)
                    .tobytes()
                ).hexdigest()
                shared_from = None
                try:
                    if full_key in full_cache:
                        previous, report, full = full_cache[full_key]
                        shutil.copytree(previous, target, copy_function=os.link)
                        shared_from = previous.name
                    else:
                        from examples.compare_rectification_methods import deadline

                        target.mkdir()
                        cv2.setRNGSeed(1973)
                        with deadline(180):
                            report = write_geometry_output(
                                image,
                                xy,
                                confidence,
                                topology,
                                target,
                                pixels_per_mm=4.0,
                            )
                            mapping = OurNativeMap(target, report)
                            full = {}
                            for domain, q in queries.items():
                                coords, eligible, footprint = mapping.predict(
                                    q["source"]
                                )
                                full[domain] = {
                                    "coords": coords,
                                    "claimed_eligible": eligible,
                                    "native_footprint": footprint,
                                }
                        full_cache[full_key] = (target, report, full)
                except Exception as exc:
                    from PIL import Image

                    Image.fromarray(image).save(
                        target / "original-preserved-failure.png"
                    )
                    failure = type(exc).__name__ + ": " + str(exc)
                    report = {
                        "inference_failure": failure,
                        "complete_geometry": {"resampled": False, "reason": failure},
                    }
                    full = failed_predictions(queries)
                write_json(target / "report.json", report)
                variants[name.replace("-cells", "-full")] = full
                details.append(
                    {
                        "key": key,
                        "method": name.replace("-cells", "-full"),
                        "resampled": report["complete_geometry"]["resampled"],
                        "reason": report["complete_geometry"].get("reason"),
                        "shared_exact_geometry_from": shared_from,
                    }
                )
            for method, prediction in variants.items():
                np.savez_compressed(
                    folder / (method + "-predictions.npz"),
                    **{
                        d + "_" + k: v
                        for d, p in prediction.items()
                        for k, v in p.items()
                    },
                )
                score = score_predictions(prediction, queries)
                write_json(folder / (method + "-scores.json"), score)
                for registration in ("primary", "secondary_D4"):
                    for domain in ("paper", "grid"):
                        metrics = score[registration]["domains"][domain]
                        rows.append(
                            {
                                "key": key,
                                "group_id": case["group_id"],
                                "backend": case["backend"],
                                "layout": case["layout"],
                                "split": split,
                                "method": method,
                                "domain": domain,
                                "registration": registration,
                                **{
                                    k: metrics[k]
                                    for k in ("g", "w", "c", "r", "rmse_mm", "p95_mm")
                                },
                            }
                        )
        write_json(out / "cases.json", rows)
        write_json(out / "details.json", details)
        print(
            json.dumps(
                {
                    "key": key,
                    "completed": index + 1,
                    "scheduled": len(cases),
                    "seconds": round(time.perf_counter() - started, 3),
                    "cells": {m: len(t["cells"]) for m, t in topologies.items()},
                }
            ),
            flush=True,
        )
    summary = {}
    for registration in ("primary", "secondary_D4"):
        summary[registration] = {}
        for domain in ("paper", "grid"):
            methods = sorted({r["method"] for r in rows})
            grouped = {
                m: [
                    r
                    for r in rows
                    if r["registration"] == registration
                    and r["domain"] == domain
                    and r["method"] == m
                ]
                for m in methods
            }
            paired = {}
            for suffix in ("cells", "full") if render else ("cells",):
                base = "baseline-" + suffix
                paired[suffix] = paired_patient_bootstrap(
                    {m: grouped[m] for m in methods if m.endswith("-" + suffix)}, base
                )
            summary[registration][domain] = {
                "means": {m: aggregate_patient_backend(v) for m, v in grouped.items()},
                "paired": paired,
            }
    write_json(out / "summary.json", summary)
    import csv

    with (out / "cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "comparison", "observations", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--orientation-receipts", type=Path)
    run(**vars(parser.parse_args()))
