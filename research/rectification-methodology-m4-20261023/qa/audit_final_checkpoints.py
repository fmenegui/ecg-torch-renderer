"""Audit final-source cost/replay receipts without rerunning measurements."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = 0
    def check(condition, note):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(note)
    original_path = ROOT/"m4/sampling/results/benchmark.json"
    final_path = ROOT/"m4/sampling/results/final-cost/benchmark.json"
    original = json.loads(original_path.read_text())
    final = json.loads(final_path.read_text())
    check(final["source_sha256"] == sha(ROOT/"m4/sampling/streaming_sampler.py"), "final sampler identity")
    check(final["protocol_sha256"] == sha(ROOT/"m4/sampling/protocol.json"), "final protocol identity")
    check(final["all_methods_same_output_for_each_n"] is True, "cross-method output parity")
    previous = {(row["method"], row["subgrid"]): row for row in original["measurements"]}
    rows = []
    for row in final["measurements"]:
        key = (row["method"], row["subgrid"])
        check(key in previous, "cost schedule")
        check(len(row["samples"]) == 3, "three repeats")
        check(statistics.median(r["total_seconds"] for r in row["samples"]) == row["median_total_seconds"], "median")
        check(row["repeated_output_sha256"] == previous[key]["repeated_output_sha256"], "original/final output hashes")
        check(len(set(row["repeated_output_sha256"])) == 1, "stable repeated output")
        throughput = 768*1024/row["median_total_seconds"]
        check(abs(throughput-row["output_pixels_per_second"]) <= throughput*1e-12, "throughput arithmetic")
        check(row["peak_rss_bytes"] > 0, "positive RSS")
        if row["method"] == "tiled":
            check(row["stats"]["max_query_points"] <= 16384, "tile point budget")
            check(row["stats"]["max_coordinate_bytes"] == 16384*56, "explicit coordinate bound")
            check(row["stats"]["inverse_points"] == 768*1024*row["subgrid"]**2, "inverse points")
        else:
            check(row["map_cache_bytes"] == 768*1024*row["subgrid"]**2*2*4, "cache byte formula")
        rows.append({"method": row["method"], "subgrid": row["subgrid"],
            "original_median_seconds": previous[key]["median_total_seconds"],
            "final_median_seconds": row["median_total_seconds"],
            "final_peak_rss_bytes": row["peak_rss_bytes"]})
    check(len(rows) == 6, "complete final cost schedule")
    receipt = {"schema": "m4-independent-final-checkpoints-audit-v1", "cost_passed": True,
        "cost_scope": "Audited medians/hashes/budget arithmetic. No timing or RSS remeasurement. Shared host CPU only.",
        "original_cost_sha256": sha(original_path), "final_cost_sha256": sha(final_path),
        "final_sampler_sha256": final["source_sha256"], "cost_rows": rows}
    replay_path = ROOT/"m4/integration/final-sampling-replay/parity.json"
    receipt["final_replay_present"] = replay_path.exists()
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        original_replay_path = ROOT/"m4/sampling/results/parity.json"
        baseline = json.loads(original_replay_path.read_text())
        check(replay["protocol_sha256"] == baseline["protocol_sha256"], "replay protocol")
        check(replay["outputs_compared"] == 3360, "replay outputs")
        check(replay["mismatched_channel_values"] == replay["maximum_abs_difference"] == replay["tile_invariance_failures"] == replay["support_failures"] == 0, "replay parity")
        check(replay["comparisons"] == baseline["comparisons"], "same saved comparisons and deterministic counters")
        check(replay["quality_summaries_reproduced"] == baseline["quality_summaries_reproduced"], "same quality summaries")
        check(replay["source_sha256"] == baseline["source_sha256"], "same recorded component identities")
        for name, value in replay["source_sha256"].items():
            check(sha(ROOT/"m4/sampling"/name) == value, "current final replay identity "+name)
        receipt.update(final_replay_passed=True, final_replay_sha256=sha(replay_path),
            final_replay_outputs=3360, final_replay_note="Full replay on final component identities confirms prior parity. The independent192output rerun remains separately audited.")
    else:
        receipt["final_replay_passed"] = None
        receipt["final_replay_note"] = "Pending, no receipt available. Do not infer success from an active process."
    receipt["checks"] = checks
    target = HERE/"receipts/final-checkpoints-audit.json"
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"receipt": str(target), "checks": checks, "cost_passed": True,
        "final_replay_present": receipt["final_replay_present"], "final_replay_passed": receipt["final_replay_passed"]}))


if __name__ == "__main__":
    main()
