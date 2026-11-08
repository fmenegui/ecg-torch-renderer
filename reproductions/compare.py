"""Refuse headline comparisons when cohorts, methods or protocols differ."""

import argparse
import json
import math
from pathlib import Path

from reproductions.common import ROOT, write_json


def compare(expected, observed):
    reasons = []
    for key in ["method_profile", "dataset", "cohort", "metric_protocol", "aggregation", "unit"]:
        if observed.get(key) != expected.get(key):
            reasons.append(f"different_or_missing_{key}")
    if not observed.get("exact_cohort_verified", False):
        reasons.append("original_cohort_identity_not_verified")
    if not observed.get("all_scheduled_outcomes_retained", False):
        reasons.append("failure_denominator_not_verified")
    if observed.get("execution_status") != "completed":
        reasons.append("execution_not_completed")
    values = observed.get("metrics", {})
    for name, value in expected["metrics"].items():
        if isinstance(value, (int, float)) and (not isinstance(values.get(name), (int, float)) or not math.isfinite(values[name])):
            reasons.append(f"undefined_metric_{name}")
    deltas = {}
    if not reasons:
        for name, value in expected["metrics"].items():
            if name in values and isinstance(value, (int, float)):
                deltas[name] = values[name] - value
    return {"benchmark": expected["id"], "directly_comparable": not reasons, "reasons": reasons, "reported": expected["metrics"], "observed": values, "delta_only_if_matched": deltas, "reproduced": False, "interpretation": "Matched measurements still require uncertainty and declared equivalence criteria. No automatic reproduced verdict."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    registered = json.loads((ROOT / "reported-results.json").read_text())["benchmarks"]
    matches = [x for x in registered if x["id"] == args.benchmark]
    if len(matches) != 1:
        raise ValueError("Unknown benchmark identity")
    write_json(args.out, compare(matches[0], json.loads(Path(args.observed).read_text())))


if __name__ == "__main__":
    main()
