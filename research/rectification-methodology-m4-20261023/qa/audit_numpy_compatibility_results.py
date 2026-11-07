"""Independent byte/record parity audit of both compatibility reruns."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/"receipts/numpy-compat"
OLD_SHA = "0ce6062797c5e25dd9ce3d7b26a6a9fabe7779ca72b5ea4e55fcda564bed9e87"
NEW_SHA = "421ebe98bdd9b5c18daf039837297d8592e9fdb4f937b466311033a9f6be7fe0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    baseline_dir = ROOT/"m4/geometry/results-guard-v2"
    baseline = json.loads((baseline_dir/"results.json").read_text())
    checks = 0
    def check(condition, note):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(note)
    check(sha(ROOT/"m4/geometry/geometry_selector.py") == NEW_SHA, "new source")
    check(sha(ROOT/"m4/geometry/geometry_selector_guard_v2_historical.py") == OLD_SHA, "historical source")
    compared = []
    for folder, version in [("results-numpy-compatible", "2.3.5"), ("results-numpy254", "2.5.4")]:
        directory = ROOT/"m4/geometry"/folder
        result = json.loads((directory/"results.json").read_text())
        check(result["source_sha256"]["m4/geometry/geometry_selector.py"] == NEW_SHA, "recorded source")
        check(result["environment"]["numpy"] == version, "actual experiment NumPy")
        check(result["protocol_sha256"] == baseline["protocol_sha256"], "unchanged protocol")
        check(result["physical_gold_used_in_selection"] is False, "no gold selection")
        check(result["decision_seal_unchanged"] is True, "unchanged decision seal")
        check(result["decision_seal_sha256"] == sha(directory/"decisions-before-physical-scoring.json"), "decision seal digest")
        check(len(result["cases"]) == 19, "scene count")
        arrays = 0
        candidates = 0
        scores = 0
        for before, after in zip(baseline["cases"], result["cases"]):
            check(before["scene"] == after["scene"], "same scene")
            check(before["methods"] == after["methods"], "all four complete method score records")
            scores += len(after["methods"])
            check(before["all_candidates_physical_scores_after_selection"] == after["all_candidates_physical_scores_after_selection"], "candidate scores")
            candidates += len(after["all_candidates_physical_scores_after_selection"])
            first = dict(before["selection"])
            second = dict(after["selection"])
            first.pop("selection_seconds_cpu")
            second.pop("selection_seconds_cpu")
            check(first == second, "all selection fields except elapsed time")
            with np.load(baseline_dir/before["array_file"], allow_pickle=False) as original:
                with np.load(directory/after["array_file"], allow_pickle=False) as updated:
                    check(original.files == updated.files, "archive key order/identity")
                    for key in original.files:
                        a, b = original[key], updated[key]
                        check(a.dtype == b.dtype and a.shape == b.shape, "array type/shape "+key)
                        av = np.ascontiguousarray(a).view(np.uint8)
                        bv = np.ascontiguousarray(b).view(np.uint8)
                        check(bool(np.array_equal(av, bv)), "all array bytes, including NaN and signed zero "+key)
                        arrays += 1
        check(arrays == 722 and candidates == 342 and scores == 76, "complete schedule")
        compared.append({"directory": str(directory), "numpy": version, "result_sha256": sha(directory/"results.json"),
            "scenes": 19, "candidate_outputs": candidates, "method_score_records": scores,
            "arrays_compared_byte_for_byte": arrays, "differences": []})
    receipt = {"schema": "m4-independent-numpy-compatibility-results-audit-v1", "passed": True,
        "checks": checks, "historical_geometry_sha256": OLD_SHA, "new_geometry_sha256": NEW_SHA,
        "baseline_results_sha256": sha(baseline_dir/"results.json"), "reruns": compared,
        "limits": "No new fit, method choice or gold tuning. Saved outputs are identical to the independently audited guard-v2 development experiment."}
    path = OUT/"results-parity-audit.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"receipt": str(path), "passed": True, "checks": checks,
        "arrays_compared": sum(row["arrays_compared_byte_for_byte"] for row in compared)}))


if __name__ == "__main__":
    main()
