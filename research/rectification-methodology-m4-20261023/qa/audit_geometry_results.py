"""Recompute all saved M4 geometry scores without using the solver's scorer."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical(base, scene):
    factor = scene.get("physical_pitch_mm", 5.)/5.
    origin = scene.get("physical_origin_mm", [0., 0.])
    return base*factor+origin


def source_formula(truth, scene):
    # Independent direct formula, not imported from experiment or solver.
    x = truth[:, 0].copy()
    y = truth[:, 1].copy()
    if scene["kind"] == "smooth":
        x = x+scene["amplitudes_mm"][0]*np.sin(truth[:, 1]*(2*np.pi/55))
        y = y+scene["amplitudes_mm"][1]*np.sin(truth[:, 0]*(2*np.pi/75))
    elif scene["kind"] == "aligned":
        y = y+scene["slope"]*np.abs(truth[:, 0]-scene["crease_u_mm"])
    elif scene["kind"] == "oblique":
        x = x+scene["slope"]*np.maximum(truth[:, 1]-.6*truth[:, 0]-scene["line_v_minus_0_6u_mm"], 0)
    elif scene["kind"] != "flat":
        raise AssertionError("Unexpected scene family")
    return np.stack((x*10/(1+scene["k"]*x), y*10/(1+scene["k"]*x)), axis=1)


def quantile_sorted(value, fraction):
    ordered = sorted(float(x) for x in value)
    pos = fraction*(len(ordered)-1)
    lo = int(np.floor(pos))
    hi = min(lo+1, len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)


def recompute(pred, reference):
    finite = np.isfinite(pred).all(1)
    error = np.sqrt(np.sum((pred-reference)**2, axis=1))
    e = error[finite]
    count = int(finite.sum())
    wrong = int(np.sum(e > .5))
    good = count-wrong
    result = {"queries": len(reference), "returned_queries": count,
        "G": good/len(reference), "W": wrong/len(reference), "C": count/len(reference),
        "R": wrong/count if count else None,
        "rmse_mm": float(np.sqrt(np.dot(e, e)/count)) if count else None,
        "p95_mm": quantile_sorted(e, .95) if count else None,
        "p99_mm": quantile_sorted(e, .99) if count else None,
        "max_mm": float(max(e)) if count else None}
    return result, finite, error


def run(folder):
    result_file = folder/"results.json"
    results = json.loads(result_file.read_text())
    seal_file = folder/"decisions-before-physical-scoring.json"
    seal = json.loads(seal_file.read_text())
    protocol_file = ROOT/"m4/geometry/protocol.json"
    protocol = json.loads(protocol_file.read_text())
    checks = 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)
    def compare(a, b, label, tolerance=1e-10):
        if a is None or b is None:
            check(a is None and b is None, label)
        else:
            check(bool(np.isclose(a, b, rtol=1e-10, atol=tolerance)), f"{label}: {a} != {b}")
    check(results["complete"] is True, "incomplete experiment")
    check(results["protocol_sha256"] == digest(protocol_file), "protocol digest")
    check(results["decision_seal_sha256"] == digest(seal_file), "decision seal digest")
    check(results["decision_seal_unchanged"] is True, "decision seal altered")
    check(results["physical_gold_used_in_selection"] is False, "truth used in selection")
    check(seal["selection_used_query_reference"] is False, "seal claims truth selection")
    check(len(results["cases"]) == len(protocol["scenes"]) == 19, "scene schedule")
    check([c["scene"] for c in results["cases"]] == protocol["scenes"], "scene identities/order")
    check([d["scene"] for d in seal["decisions"]] == [s["name"] for s in protocol["scenes"]], "seal schedule")
    hashes = {}
    extrema = []
    for case, decision in zip(results["cases"], seal["decisions"]):
        name = case["scene"]["name"]
        report = case["selection"]
        check(report == decision["report"], f"{name} selection changed after seal")
        check(report["selection_used_physical_reference"] is False, f"{name} reference use")
        check("remain unverified" in report["identity_status"], f"{name} identity overclaim")
        check(list(report["candidates"]) == protocol["candidate_order"], f"{name} candidate schedule")
        means, ses = {}, {}
        for candidate in protocol["candidate_order"]:
            entry = report["candidates"][candidate]
            loss = np.array(entry["fold_loss"])
            check(len(loss) == 4, f"{name}/{candidate} folds")
            check(bool(np.isfinite(loss).all()), f"{name}/{candidate} finite loss")
            means[candidate] = float(sum(loss)/4)
            ses[candidate] = float(np.sqrt(sum((loss-means[candidate])**2)/3)/2)
            compare(means[candidate], entry["mean_loss"], f"{name}/{candidate} mean")
            compare(ses[candidate], entry["standard_error"], f"{name}/{candidate} se")
        best = min(protocol["candidate_order"], key=lambda key: means[key])
        threshold = means[best]+ses[best]
        chosen = next(key for key in protocol["candidate_order"] if means[key] <= threshold+1e-14)
        check(best == report["best_cv_candidate"], f"{name} best CV")
        check(chosen == report["selected_candidate"], f"{name} non-oracle selection")
        compare(threshold, report["one_standard_error_threshold"], f"{name} threshold")
        file = folder/case["array_file"]
        hashes[case["array_file"]] = digest(file)
        with np.load(file, allow_pickle=False) as arrays:
            truth = arrays["physical_queries"]
            np.testing.assert_allclose(truth, physical(arrays["base_queries"], case["scene"]), atol=0, rtol=0)
            checks += 1
            np.testing.assert_allclose(arrays["source_queries"], source_formula(truth, case["scene"]), atol=2e-12, rtol=2e-14)
            checks += 1
            check(truth.shape == (4096, 2), f"{name} fixed query count")
            base = arrays["base_queries"]
            support = ((base >= [0, 0]) & (base <= [75, 55])).all(1)
            gap = case["scene"].get("gap")
            if gap == "rows4to7-cols6to9":
                support &= ~((base[:, 0] > 25) & (base[:, 0] < 50) & (base[:, 1] > 15) & (base[:, 1] < 40))
            elif gap == "columns0to2":
                support &= base[:, 0] >= 15
            np.testing.assert_array_equal(arrays["reference_observed_support"], support)
            checks += 1
            finite_by_method = {}
            errors_by_method = {}
            for method in protocol["comparators"]:
                pred = arrays[method+"_prediction"]
                actual, finite, error = recompute(pred, truth)
                saved = case["methods"][method]
                for key in actual:
                    compare(actual[key], saved[key], f"{name}/{method}/{key}")
                compare(saved["G"]+saved["W"], saved["C"], f"{name}/{method}/G+W=C")
                exterior = arrays[method+"_exterior_prediction"]
                compare(int(np.isfinite(exterior).all(1).sum()), saved["exterior_returned"], f"{name}/{method}/exterior")
                compare(int((finite & ~support).sum()), saved["returned_outside_reference_support"], f"{name}/{method}/outside support")
                finite_by_method[method] = finite
                errors_by_method[method] = error
            direct = "current-direct-M2-cells"
            for method in protocol["comparators"]:
                common = finite_by_method[direct] & finite_by_method[method]
                saved = case["methods"][method]
                compare(int(common.sum()), saved["common_with_direct_queries"], f"{name}/{method}/common count")
                expected = float(np.sqrt(np.mean(errors_by_method[method][common]**2))) if common.any() else None
                base_expected = float(np.sqrt(np.mean(errors_by_method[direct][common]**2))) if common.any() else None
                compare(expected, saved["common_rmse_mm"], f"{name}/{method}/common RMSE")
                compare(base_expected, saved["direct_common_rmse_mm"], f"{name}/{method}/direct common RMSE")
            stack = []
            check(len(case["all_candidates_physical_scores_after_selection"]) == 18, f"{name} candidates all retained")
            for candidate in protocol["candidate_order"]:
                pred = arrays["candidate_"+candidate.replace("/", "__")+"_prediction"]
                actual, _, _ = recompute(pred, truth)
                stored = case["all_candidates_physical_scores_after_selection"][candidate]
                for key in actual:
                    compare(actual[key], stored[key], f"{name}/{candidate}/{key}")
                stack.append(pred)
            # Recompute disagreement point by point without vectorized nansum.
            stack = np.asarray(stack)
            actual_disagreement = np.full(len(truth), np.nan)
            for query in range(len(truth)):
                points = stack[:, query]
                points = points[np.isfinite(points).all(1)]
                if len(points) >= 2:
                    centre = sum(points)/len(points)
                    actual_disagreement[query] = np.sqrt(sum(float(np.dot(p-centre, p-centre)) for p in points)/len(points))
            np.testing.assert_allclose(actual_disagreement, arrays["candidate_disagreement_mm_uncalibrated"], rtol=2e-12, atol=2e-13, equal_nan=True)
            checks += 1
            chosen_array = arrays["candidate_"+chosen.replace("/", "__")+"_prediction"]
            np.testing.assert_allclose(chosen_array, arrays["selected-input-only_prediction"], rtol=0, atol=0, equal_nan=True)
            checks += 1
            ext = case["methods"]["selected-input-only"]["exterior_returned"]
            extrema.append({"scene": name, "selected": chosen, "exterior_returned": ext,
                **{key: case["methods"]["selected-input-only"][key] for key in ["G", "W", "C", "R", "rmse_mm", "p99_mm", "max_mm"]}})
    return {"schema": "m4-independent-geometry-results-audit-v1", "passed": True,
        "source_results": str(result_file), "source_results_sha256": digest(result_file),
        "decision_seal_sha256": digest(seal_file), "source_bindings": results["source_sha256"],
        "checks": checks, "scenes": 19, "primary_method_outputs": 76, "candidate_outputs": 342,
        "archive_sha256": hashes, "selected_scene_results": extrema,
        "limits": ["Recomputes saved predictions and decisions, does not rerun19fits",
            "Source inspection and sealed decisions support truth isolation, not a universal information-flow proof",
            "All scene families are development analytical scenes, not fresh real captures",
            "Supplied lattice IDs, physical pitch and chart origin remain unverified"]}


if __name__ == "__main__":
    folder = ROOT/"m4/geometry/results-guard-v2"
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    receipt = run(folder)
    target = HERE/"receipts/geometry-result-audit.json"
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"receipt": str(target), "checks": receipt["checks"],
        "primary_method_outputs": 76, "candidate_outputs": 342, "passed": True, "sha256": digest(target)}))
