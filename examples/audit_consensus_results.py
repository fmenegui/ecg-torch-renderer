"""Recount saved predictions and audit original-RGB sampling independently.

Does not import the organizer, rectifier, CellMap or metric implementation.
Triangle interpolation and primary score counts are recomputed from saved data.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inverse_triangles(source, target, triangles, query):
    """Independent barycentric lookup on a small audit sample."""
    result = np.full(query.shape, np.nan)
    vertices = target[triangles]
    a = vertices[:, 0]
    b, c = vertices[:, 1] - a, vertices[:, 2] - a
    determinant = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
    assert (abs(determinant) > 1e-14).all()
    for start in range(0, len(query), 32):
        q = query[start : start + 32, None] - a
        u = (q[..., 0] * c[:, 1] - q[..., 1] * c[:, 0]) / determinant
        v = (b[:, 0] * q[..., 1] - b[:, 1] * q[..., 0]) / determinant
        weights = np.stack((1 - u - v, u, v), -1)
        inside = ((weights >= -1e-8) & (weights <= 1 + 1e-8)).all(-1)
        unique = (weights > 1e-8).all(-1).sum(1) <= 1
        keep = inside.any(1) & unique
        chosen = inside.argmax(1)[keep]
        result[start : start + len(q)][keep] = np.einsum(
            "ni,nij->nj",
            weights[np.flatnonzero(keep), chosen],
            source[triangles[chosen]],
        )
    return result


def audit(bank, comparison, evaluation, out):
    from PIL import Image
    import cv2

    protocol = json.loads((evaluation / "protocol.json").read_text())
    rows = json.loads((evaluation / "cases.json").read_text())
    summary = json.loads((evaluation / "summary.json").read_text())
    methods = sorted({r["method"] for r in rows})
    checks, counts, topology_checks, rgb_samples = 0, 0, 0, 0
    output_cases = []
    baseline_replay = []
    for case in protocol["cases"]:
        key = case["key"]
        assert digest(bank / case["file"]) == case["sha256"]
        assert digest(Path(case["points_path"])) == case["points_sha256"]
        assert digest(comparison / case["query_file"]) == case["query_archive_sha256"]
        with np.load(bank / case["file"], allow_pickle=False) as z:
            original = z["image"].copy()
        with np.load(comparison / case["query_file"], allow_pickle=False) as z:
            queries = {
                d: (z[d + "_source"].copy(), z[d + "_reference"].copy())
                for d in ("anchors", "paper", "grid")
            }
        folder = evaluation / "cases" / key
        if "orientation_receipt" in case:
            path = Path(case["orientation_receipt"])
            assert digest(path) == case["orientation_sha256"]
            receipt = json.loads(path.read_text())
            assert (
                receipt["rgb_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
            )
            checks += 2
        for method in methods:
            with np.load(
                folder / (method + "-predictions.npz"), allow_pickle=False
            ) as z:
                prediction = {
                    d: (z[d + "_coords"].copy(), z[d + "_claimed_eligible"].copy())
                    for d in queries
                }
            scored = json.loads((folder / (method + "-scores.json")).read_text())
            for mode in ("primary", "secondary_D4"):
                registration = scored[mode]["registration"]
                matrix = np.asarray(registration["matrix"])
                coords, claimed = prediction["anchors"]
                keep = claimed & np.isfinite(coords).all(1)
                can_register = (
                    keep.sum() >= 3
                    and (np.ptp(queries["anchors"][1][keep], axis=0) >= 1.0).all()
                )
                assert can_register == (
                    registration["status"] == "registered_translation_only"
                )
                checks += 1
                if can_register:
                    offset = np.median(
                        queries["anchors"][1][keep] - coords[keep] @ matrix.T, axis=0
                    )
                    np.testing.assert_allclose(
                        offset, registration["offset_mm"], atol=1e-12, rtol=0
                    )
                    checks += 1
                for domain in ("paper", "grid"):
                    coords, claimed = prediction[domain]
                    ref = queries[domain][1]
                    if can_register:
                        finite = claimed & np.isfinite(coords).all(1)
                        error = np.linalg.norm(
                            coords[finite] @ matrix.T + offset - ref[finite], axis=1
                        )
                        correct = int((error <= 0.5).sum())
                        coverage = int(claimed.sum())
                    else:
                        correct, coverage = 0, 0
                    wrong = coverage - correct
                    expected = {
                        "g": correct / len(ref),
                        "w": wrong / len(ref),
                        "c": coverage / len(ref),
                        "r": wrong / coverage if coverage else None,
                    }
                    stored = scored[mode]["domains"][domain]
                    for name, value in expected.items():
                        if value is None:
                            assert stored[name] is None
                        else:
                            assert abs(value - stored[name]) < 1e-12
                        counts += 1
            if method.endswith("-cells"):
                topology = json.loads(
                    (folder / (method + "-topology.json")).read_text()
                )
                selected = topology["selected_indices"]
                ids = np.asarray(topology["ids"], int).reshape(-1, 2)
                assert len(selected) == len(set(selected)) == len(ids)
                assert len({tuple(k) for k in ids}) == len(ids)
                present = {tuple(k) for k in ids}
                for r, c in topology["cells"]:
                    assert all(
                        k in present
                        for k in ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
                    )
                topology_checks += len(topology["cells"]) + 2
            else:
                target = folder / method
                report = json.loads((target / "report.json").read_text())
                complete = report["complete_geometry"]
                resampled = complete["resampled"]
                if not resampled:
                    filename = (
                        "original-preserved-failure.png"
                        if report.get("inference_failure")
                        else "original-preserved.png"
                    )
                    np.testing.assert_array_equal(
                        np.asarray(Image.open(target / filename)), original
                    )
                    checks += 1
                else:
                    with np.load(target / "full-frame-mesh.npz") as z:
                        source, physical, triangles = (
                            z["source"],
                            z["target_mm"],
                            z["triangles"],
                        )
                    with np.load(target / "geometry-evidence.npz") as z:
                        valid = z["sampling_valid"]
                    candidates = np.flatnonzero(valid.ravel())
                    chosen = candidates[
                        np.linspace(
                            0, len(candidates) - 1, min(256, len(candidates))
                        ).astype(int)
                    ]
                    y, x = np.unravel_index(chosen, valid.shape)
                    points = (
                        np.column_stack((x, y)) / complete["pixels_per_mm"]
                        + complete["output_origin_mm"]
                    )
                    coordinates = inverse_triangles(source, physical, triangles, points)
                    assert np.isfinite(coordinates).all()
                    coordinates = coordinates.clip(
                        [0, 0], [original.shape[1] - 1, original.shape[0] - 1]
                    ).astype(np.float32)
                    expected = cv2.remap(
                        original,
                        coordinates[:, 0:1],
                        coordinates[:, 1:2],
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255),
                    )[:, 0]
                    actual = np.asarray(
                        Image.open(target / "rectified-original-rgb.png")
                    )[y, x]
                    np.testing.assert_array_equal(actual, expected)
                    rgb_samples += len(chosen)
                output_cases.append(
                    {"key": key, "method": method, "resampled": resampled}
                )
                if method == "baseline-full":
                    old = comparison / "cases" / key / "our-B0/query-predictions.npz"
                    with np.load(old, allow_pickle=False) as z:
                        exact = all(
                            np.array_equal(
                                z[d + "_coords"], prediction[d][0], equal_nan=True
                            )
                            and np.array_equal(
                                z[d + "_claimed_eligible"], prediction[d][1]
                            )
                            for d in queries
                        )
                    baseline_replay.append(
                        {"key": key, "coordinates_and_eligibility_exact": exact}
                    )
    for mode in ("primary", "secondary_D4"):
        for domain in ("paper", "grid"):
            for method in methods:
                selected = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["domain"] == domain
                    and r["registration"] == mode
                ]
                assert len(selected) == len(protocol["cases"])
                grouped = {}
                for row in selected:
                    grouped.setdefault((row["group_id"], row["backend"]), []).append(
                        row
                    )
                means = [
                    [np.mean([r[k] for r in group]) for k in ("g", "w", "c")]
                    for group in grouped.values()
                ]
                g, w, c = np.mean(means, axis=0)
                stored = summary[mode][domain]["means"][method]["overall"]
                np.testing.assert_allclose(
                    [g, w, c], [stored[k] for k in ("g", "w", "c")], atol=1e-12, rtol=0
                )
                checks += 1
    result = {
        "passed": True,
        "scheduled_cases": len(protocol["cases"]),
        "methods": methods,
        "coordinate_count_checks": counts,
        "registration_and_binding_checks": checks,
        "observed_cell_checks": topology_checks,
        "original_RGB_sample_checks": rgb_samples,
        "full_frame_cases": output_cases,
        "baseline_replay": baseline_replay,
        "source_sha256": digest(__file__),
        "evaluation_protocol_sha256": digest(evaluation / "protocol.json"),
        "scope": "Independent numerical recount and sampling audit, not independent physical truth or clinical validation",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("full_frame_cases", "baseline_replay")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "comparison", "evaluation", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    audit(**vars(parser.parse_args()))
