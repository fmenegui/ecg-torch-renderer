"""Independent analytical controls, without a trained commercial detector."""

import argparse

import numpy as np

from reproductions.common import environment, new_run, write_json
from reproductions.powerful_medical.gridder import Grid, organize, warp_cells


def project(points, matrix):
    q = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return q[:, :2] / q[:, 2:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from PIL import Image
    out = new_run(args.out)
    row, col = np.mgrid[:8, :11]
    ids = np.column_stack([row.ravel(), col.ravel()])
    canonical = np.column_stack([col.ravel() * 20, row.ravel() * 20]).astype(float)
    h = np.array([[1.05, 0.12, 25], [0.07, 1.12, 20], [0.0010, 0.0005, 1]])
    strong_h = np.array([[1.05, 0.12, 25], [0.07, 1.12, 20], [0.018, 0.005, 1]])
    angle = np.deg2rad(50)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    cases = {
        "flat": canonical + [20, 20],
        "projective": project(canonical, h),
        "smooth_bend": canonical + np.column_stack([np.zeros(len(canonical)), 9 * np.sin(canonical[:, 0] / 60)]) + [20, 20],
        "strong_projective": project(canonical, strong_h),
        "rotated_semantically_unresolved": canonical @ rotation.T + [150, 20],
        "missing_entire_row": canonical + [20, 20],
    }
    yy, xx = np.mgrid[:400, :400]
    image = np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype(np.uint8)
    Image.fromarray(image).save(out / "source-colour-probe.png")
    results = []
    for name, points in cases.items():
        organizer = {"status": "rejected"}
        try:
            visible = ids[:, 0] != 3 if name == "missing_entire_row" else np.ones(len(ids), bool)
            native = organize(points[visible])
            # Points are exact source observations, with no model localization error.
            lookup = {tuple(p): tuple(rc) for p, rc in zip(points, ids)}
            truth = np.array([lookup[tuple(p)] for p in native.points_xy])
            delta = native.ids_rc - truth
            agreement = np.all(delta == delta[0], axis=1).mean()
            organizer = {"status": "returned", "retained_fraction": len(native.points_xy) / len(points), "relative_id_agreement_after_anchor_translation": float(agreement), "diagnostics": native.diagnostics}
        except ValueError as error:
            organizer["reason"] = str(error)
        oracle_grid = Grid(points, ids, np.ones(len(points), bool), {})
        result = warp_cells(image, oracle_grid, pixels_per_cell=20)
        Image.fromarray(result["rgb"]).save(out / (name + "-oracle-cells.png"))
        query_y, query_x = np.mgrid[:140, :200]
        queries = np.column_stack([query_x.ravel(), query_y.ravel()]).astype(float)
        if name in ["flat", "missing_entire_row"]:
            expected = queries + [20, 20]
        elif name == "projective":
            expected = project(queries, h)
        elif name == "smooth_bend":
            expected = queries + np.column_stack([np.zeros(len(queries)), 9 * np.sin(queries[:, 0] / 60)]) + [20, 20]
        elif name == "strong_projective":
            expected = project(queries, strong_h)
        else:
            expected = queries @ rotation.T + [150, 20]
        sampled = np.stack([result["map_x"].ravel(), result["map_y"].ravel()], -1)
        error = np.linalg.norm(sampled - expected, axis=1)
        record = {"case": name, "organizer_with_perfect_points": organizer, "sampler_with_known_ids": {"query_count": len(error), "p95_source_error_px": float(np.quantile(error, 0.95)), "max_source_error_px": float(error.max()), "source_support_fraction": float(result["support"].mean())}}
        results.append(record)
    # Exact global projective mappings should be reproduced by each local H.
    if results[0]["sampler_with_known_ids"]["max_source_error_px"] > 1e-5 or results[1]["sampler_with_known_ids"]["max_source_error_px"] > 2e-4:
        raise AssertionError("Analytical cell mapping control failed")
    write_json(out / "receipt.json", {"schema": "pm-mechanism-controls-v1", "environment": environment(), "trained_detector_executed": False, "commercial_implementation_executed": False, "full_paper_scores_reproduced": False, "cases": results, "limitations": ["global directional clusters are one numerical interpretation", "grid orientation and units remain unresolved", "curvature inside cells is approximated", "partial raster is not full-frame content preservation"]})


if __name__ == "__main__":
    main()
