"""Independent saved-evidence and selected direct-affine parity audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/"m4/sampling"))
from streaming_sampler import OutputGrid, sample_image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = ROOT/"m4/sampling/results/parity.json"
    report = json.loads(path.read_text())
    protocol = json.loads((ROOT/"m4/sampling/protocol.json").read_text())
    checks = 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)
    check(report["protocol_sha256"] == digest(ROOT/"m4/sampling/protocol.json"), "protocol hash")
    for name, expected in report["source_sha256"].items():
        check(digest(ROOT/"m4/sampling"/name) == expected, "source hash "+name)
    for name, expected in report["archive_sha256"].items():
        check(digest(ROOT/name) == expected, "archive hash "+name)
    check(len(report["comparisons"]) == 280*6*2 == 3360, "full schedule")
    identities = set()
    for row in report["comparisons"]:
        identity = (row["map"], row["scene"], row["arm"], row["tile_pixels"])
        check(identity not in identities, "duplicate comparison")
        identities.add(identity)
        check(row["mismatched_channel_values"] == row["maximum_abs_difference"] == row["different_from_other_tile_budget"] == 0, "reported mismatch")
        check(row["all_source_cell_fractions_one"] is True, "interior source fraction")
        n = int(row["arm"].split("-")[-1])
        stats = row["stats"]
        check(stats["max_query_points"] <= row["tile_pixels"], "query tile budget")
        check(stats["inverse_points"] == 41*41*n*n, "sample count")
        check(stats["inverse_calls"] == stats["tiles"]*n*n, "call count")
        check(stats["max_coordinate_bytes"] <= stats["max_query_points"]*56, "coordinate bound")
    expected_keys = {(name, scene, arm, budget) for name in ["affine-isotropic", "affine-anisotropic-shear"]
                     for scene in range(140) for arm in protocol["arms"] for budget in [512, 1681]}
    check(identities == expected_keys, "schedule identities")
    check(report["channel_values_compared"] == 3360*41*41*3, "channel denominator")
    quality = {}
    selected_indices = [0, 9, 13, 35, 69, 70, 119, 139]
    selected_comparisons = 0
    mismatch_values = 0
    max_direct_difference = 0
    grid = OutputGrid((41, 41), (2., 2.), (.25, .25))
    for name in ["affine-isotropic", "affine-anisotropic-shear"]:
        with np.load(ROOT/f"m3/sampling/results/{name}.npz", allow_pickle=False) as archive:
            source_matrix = archive["source_per_mm"]
            # Direct formula from declared source centre100 at physical10.
            # Does not call M3 CompleteMap or M4 evaluate.old_module.
            inverse = lambda q, a=source_matrix: (q-10.)@a.T+100.
            arms = archive["arms"].tolist()
            for scene, (width, angle, phase) in enumerate(archive["scene_width_angle_offset"]):
                for arm in protocol["arms"]:
                    index = arms.index(arm)
                    dark = archive["darkness_u8"][scene, index].astype(float)/255.
                    reference = archive["ideal_coverage_f64"][scene]
                    values = quality.setdefault((float(width), arm), [])
                    values.append({"disappeared": bool(dark.max() == 0),
                        "mae": float(np.mean(np.abs(dark-reference))),
                        "energy_error": float(abs(dark.sum()/reference.sum()-1.)),
                        "peak_error": float(abs(dark.max()-reference.max())),
                        "peak": float(dark.max())})
                    if scene not in selected_indices:
                        continue
                    n = int(arm.split("-")[-1])
                    mode = arm.split("-")[1]
                    source = np.repeat(archive["source_gray_u8"][scene, ..., None], 3, axis=2)
                    expected = np.repeat((255-archive["darkness_u8"][scene, index])[..., None], 3, axis=2)
                    for budget in [257, 1681]:
                        got = sample_image(source, inverse, grid, subgrid=n,
                            reconstruction=mode, max_tile_pixels=budget)
                        difference = np.abs(got.image.astype(int)-expected.astype(int))
                        maximum = int(difference.max())
                        # Direct algebra can shift final uint8 half-LSB ties
                        # compared with an inverse through fitted triangles.
                        check(maximum <= 1, f"direct affine parity {name}/{scene}/{arm}/{budget}")
                        max_direct_difference = max(max_direct_difference, maximum)
                        mismatch_values += int(np.count_nonzero(difference))
                        selected_comparisons += 1
    for summary in report["quality_summaries_reproduced"]:
        values = quality[(summary["width_mm"], summary["arm"])]
        expected = {"cases": len(values), "disappearances": sum(v["disappeared"] for v in values),
            "mean_pixel_MAE": float(np.mean([v["mae"] for v in values])),
            "mean_abs_relative_energy_error": float(np.mean([v["energy_error"] for v in values])),
            "mean_peak_abs_error": float(np.mean([v["peak_error"] for v in values])),
            "minimum_peak_darkness": min(v["peak"] for v in values)}
        for key, value in expected.items():
            check(bool(np.isclose(value, summary[key], rtol=1e-12, atol=1e-13)), "summary "+key)
    benchmark_file = ROOT/"m4/sampling/results/benchmark.json"
    benchmark = json.loads(benchmark_file.read_text())
    check(benchmark["protocol_sha256"] == report["protocol_sha256"], "cost protocol")
    check(benchmark["all_methods_same_output_for_each_n"] is True, "cost output parity")
    cost_rows = []
    for row in benchmark["measurements"]:
        total = [sample["total_seconds"] for sample in row["samples"]]
        check(len(total) == 3, "cost repeats")
        check(float(np.median(total)) == row["median_total_seconds"], "cost median")
        check(len(set(row["repeated_output_sha256"])) == 1, "cost repeat hash")
        check(row["output_hashes_identical"] is True, "cost self parity")
        cost_rows.append({key: row[key] for key in ["method", "subgrid", "median_total_seconds", "peak_rss_bytes"]})
    receipt = {"schema": "m4-independent-sampling-results-audit-v1", "passed": True,
        "checks": checks, "saved_comparisons_checked": 3360,
        "direct_affine_outputs_rerun": selected_comparisons,
        "direct_affine_scene_indices": selected_indices,
        "direct_affine_max_uint8_difference": max_direct_difference,
        "direct_affine_mismatched_channel_values": mismatch_values,
        "quality_summaries_recomputed": 12,
        "source_report_sha256": digest(path), "benchmark_sha256": digest(benchmark_file),
        "source_bindings": report["source_sha256"], "cost_rows": cost_rows,
        "limits": ["All3360saved comparisons structurally audited,192selectedoutputs rerun from direct affine formula",
            "Benchmark medians and scope audited, peak RSS not independently remeasured",
            "M3 parity is on41x41interiorROI only, not full-frame or nonlinear fidelity",
            "One-level uint8 tolerance acknowledges direct affine versus CellMap tie arithmetic, not geometry tuning"]}
    target = HERE/"receipts/sampling-result-audit.json"
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"receipt": str(target), "checks": checks,
        "rerun_outputs": selected_comparisons, "max_difference": max_direct_difference, "passed": True}))


if __name__ == "__main__":
    main()
