"""Separate-process CPU cost, including all coordinate preparation."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

from streaming_sampler import OutputGrid, sample_image
from evaluate import HERE, PROTOCOL_SHA, digest, old_module


def inverse(points):
    return points @ np.array([[1.1, .05], [-.02, 1.15]]).T+[40., 50.]


def worker(method, n):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261023)
    source = rng.integers(0, 256, (1100, 1400, 3), dtype=np.uint8)
    grid = OutputGrid((768, 1024), (0., 0.), (1., 1.))
    old = old_module()
    samples, digests, memory = [], [], []
    for repeat in range(3):
        gc.collect()
        started = time.perf_counter()
        if method == "tiled":
            result = sample_image(source, inverse, grid, subgrid=n, max_tile_pixels=16384)
            image = result.image
            total = time.perf_counter()-started
            extra = {"stats": asdict(result.stats)}
            samples.append({"total_seconds": total})
        else:
            yy, xx = np.mgrid[:768, :1024]
            physical = np.stack((xx, yy), -1).astype(float)
            class Map:
                pass
            mapping = Map()
            mapping.inverse = inverse
            maps, preparation = old.prepare_area_maps(mapping, physical, 1., n, source.shape)
            prepared = time.perf_counter()
            image = old.area_sample(source, maps, "bilinear")
            finished = time.perf_counter()
            extra = {"map_cache_bytes": maps.nbytes}
            samples.append({"total_seconds": finished-started,
                            "map_seconds": prepared-started,
                            "application_seconds": finished-prepared})
        digests.append(hashlib.sha256(image.tobytes()).hexdigest())
        memory.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        del image
        if method == "tiled":
            del result
        else:
            del maps, physical, xx, yy
    print(json.dumps({"method": method, "subgrid": n, "samples": samples,
                      "median_total_seconds": float(np.median([x["total_seconds"] for x in samples])),
                      "output_pixels_per_second": 768*1024/float(np.median([x["total_seconds"] for x in samples])),
                      "peak_rss_bytes": max(memory), "repeated_output_sha256": digests,
                      "output_hashes_identical": len(set(digests)) == 1,
                      "opencv_threads": cv2.getNumThreads(), **extra}))


def inventory():
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "unknown")
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {"cpu_model": model, "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "opencv": cv2.__version__, "logical_cpu_count": os.cpu_count(),
            "process_affinity_cpu_count": len(os.sched_getaffinity(0)),
            "cgroup_cpu_max": quota.read_text().strip() if quota.exists() else None,
            "threads": {name: os.environ.get(name) for name in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def run(out):
    if digest(HERE/"protocol.json") != PROTOCOL_SHA:
        raise ValueError("frozen protocol mismatch")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for n in (2, 4, 8):
        for method in ("cached-m3", "tiled"):
            command = [sys.executable, str(Path(__file__)), "--worker", method, "--n", str(n)]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            record = json.loads(completed.stdout)
            records.append(record)
            (out/f"benchmark-{method}-{n}.json").write_text(json.dumps(record, indent=2)+"\n")
            print(json.dumps(record), flush=True)
    same_outputs = all(next(x for x in records if x["subgrid"] == n and x["method"] == "cached-m3")
                       ["repeated_output_sha256"][0] ==
                       next(x for x in records if x["subgrid"] == n and x["method"] == "tiled")
                       ["repeated_output_sha256"][0] for n in (2, 4, 8))
    report = {"schema": "m4-sampling-cost-v1", "protocol_sha256": PROTOCOL_SHA,
              "source_sha256": digest(HERE/"streaming_sampler.py"), "hardware": inventory(),
              "source_shape": [1100, 1400, 3], "output_shape": [768, 1024],
              "source_float32_bytes": 1100*1400*3*4,
              "collected_output_and_evidence_bytes": 768*1024*(3+3*4),
              "all_methods_same_output_for_each_n": same_outputs, "measurements": records,
              "scope": "CPU affine callback including conversion, coordinates, sampling and collection. Separate fresh processes, three repeated images. Peak RSS is lifetime high water, including imports and previous repeats. Host is shared. No GPU or production-pipeline timing claim.",
              "large_frame_formula": {"output_pixels": 16000000, "subgrid": 8,
                                      "old_coordinate_cache_bytes": 16000000*64*2*4,
                                      "tile_coordinate_array_bytes_at_budget_16384": 16384*56,
                                      "note": "Only coordinate arrays in the tile formula, not total process/image memory or callback internals."}}
    (out/"benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if not same_outputs:
        raise SystemExit("Large-frame benchmark parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"results")
    parser.add_argument("--worker", choices=("tiled", "cached-m3"))
    parser.add_argument("--n", type=int, choices=(2, 4, 8))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.n)
    else:
        run(args.out)
