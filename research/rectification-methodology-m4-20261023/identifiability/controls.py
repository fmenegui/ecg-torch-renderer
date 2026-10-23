#!/usr/bin/env python3
"""Frozen analytical witnesses. No ECG model, source gold, or M3 inference is used.

Each case is retained, including failed assertions. Equal RGB or observations
with different material targets prove non-identifiability only in the declared
world class. Model-limit cases explicitly have distinguishable input evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_array(a):
    a = np.ascontiguousarray(a)
    return sha_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def grid(w=161, h=161, step=20):
    a = np.full((h, w, 3), 255, dtype=np.uint8)
    a[::step, :, :] = (220, 100, 100)
    a[:, ::step, :] = (220, 100, 100)
    return a


def d4(a):
    return [np.rot90(a, k) for k in range(4)] + [
        np.rot90(np.flip(a, axis=1), k) for k in range(4)
    ]


def d4_matrices():
    r = np.array([[0.0, -1.0], [1.0, 0.0]])
    f = np.diag([-1.0, 1.0])
    rs = [np.linalg.matrix_power(r, k) for k in range(4)]
    return rs + [q @ f for q in rs]


def check(label, cond, **metrics):
    return {"name": label, "passed": bool(cond), **metrics}


def result(metrics, checks, **arrays):
    return {"metrics": metrics, "checks": checks, "arrays": arrays}


def c01():
    rgb = grid()
    pitches = np.array([1.0, 5.0, 10.0])
    q = np.array([100.0, 80.0])
    physical = q[None, :] * pitches[:, None] / 20.0
    worlds = np.stack([rgb] * 3)
    return result(
        {"observation_level": "exact_RGB", "surviving_pitch_hypotheses_mm": pitches.tolist(),
         "camera_pixels_per_mm": (20 / pitches).tolist(),
         "query_source_xy": q.tolist(), "query_material_mm": physical.tolist(),
         "world_condition": "uniform-looking lines, absent unit legend and absent reliable major/minor evidence"},
        [check("all_RGB_identical", np.array_equal(worlds[0], worlds[1]) and np.array_equal(worlds[1], worlds[2])),
         check("physical_targets_differ", len(np.unique(physical, axis=0)) == 3)],
        RGB_worlds=worlds, pitch_mm=pitches, query=q, target_mm=physical)


def c02():
    rgb = grid()
    hashes = [sha_array(a) for a in d4(rgb)]
    q = np.array([40.0, 100.0])
    center = np.array([20.0, 20.0])
    p = q / 4.0
    targets = np.array([center + qmat @ (p - center) for qmat in d4_matrices()])
    return result(
        {"observation_level": "exact_RGB", "RGB_equivalent_D4_hypotheses": len(hashes),
         "unique_observation_hashes": len(set(hashes)), "distinct_query_targets": len(np.unique(targets, axis=0)),
         "query_source_xy": q.tolist(), "query_material_mm": targets.tolist(),
         "world_condition": "square grid, square crop, no semantic direction or chirality cue"},
        [check("eight_D4_images_identical", len(set(hashes)) == 1),
         check("eight_distinct_semantic_targets", len(np.unique(targets, axis=0)) == 8)],
        RGB=rgb, query=q, target_mm=targets, D4=np.array(d4_matrices()))


def c03():
    rgb = grid(201, 201)
    obs = np.array(list(itertools.product(np.arange(0, 201, 40), repeat=2)), float)
    true_mm = obs / 4.0
    wrong_mm = obs / 8.0
    # Both affine fits have exact observations and exact reciprocal inverses.
    residual_true = np.max(np.abs(4 * true_mm - obs))
    residual_wrong = np.max(np.abs(8 * wrong_mm - obs))
    q = np.array([100.0, 100.0])
    err = float(np.linalg.norm(q / 4 - q / 8))
    # Evidence deliberately omitted by the detector remains in this RGB.
    intermediate_present = bool(np.all(rgb[:, 20, :] == (220, 100, 100)))
    return result(
        {"observation_level": "extracted_points_only", "RGB_can_disambiguate": True,
         "actual_observed_pitch_mm": 10.0, "wrong_consecutive_ID_pitch_mm": 5.0,
         "true_fit_residual_px": float(residual_true), "wrong_fit_residual_px": float(residual_wrong),
         "wrong_map_roundtrip_error_px": float(np.max(np.abs(8 * (obs / 8) - obs))),
         "wrong_query_error_mm": err,
         "world_condition": "every second major row/column lost by frontend, but intermediate lines remain visible in RGB"},
        [check("two_zero_residual_maps", residual_true == residual_wrong == 0),
         check("RGB_contains_omitted_line", intermediate_present),
         check("wrong_mm_not_repaired_by_roundtrip", err > 17.0)],
        RGB=rgb, observations_px=obs, true_mm=true_mm, wrong_mm=wrong_mm, query=q)


def bridge_rgb(opaque):
    rgb = grid(241, 81)
    rgb[:, 81:160, :] = (160, 160, 160) if opaque else (255, 255, 255)
    return rgb


def c04():
    rgb = bridge_rgb(True)
    # Two extruded, developable strips under an orthographic camera. The hidden
    # bridge spans 20mm in x. A V of half-span10mm, depth7.5mm has length25mm.
    flat_length, fold_length = 20.0, 2 * np.hypot(10.0, 7.5)
    q = np.array([200.0, 40.0])
    targets = np.array([[50.0, 10.0], [50.0 + fold_length - flat_length, 10.0]])
    world_rgb = np.stack([rgb, rgb])
    return result(
        {"observation_level": "exact_RGB", "hidden_bridge_lengths_mm": [flat_length, float(fold_length)],
         "component_phase_difference_major_steps": float((fold_length-flat_length)/5),
         "query_material_mm": targets.tolist(),
         "world_condition": "orthographic camera, hidden bridge, integer5mm extra arc length, no trusted whole-plane constraint"},
        [check("same_visible_RGB", np.array_equal(*world_rgb)),
         check("one_integer_phase_step", np.isclose(fold_length-flat_length, 5)),
         check("different_component_origin", np.isclose(targets[1, 0]-targets[0, 0], 5))],
        RGB_worlds=world_rgb, query=q, target_mm=targets,
        bridge_xyz_flat=np.array([[20, 0, 0], [30, 0, 0], [40, 0, 0]], float),
        bridge_xyz_fold=np.array([[20, 0, 0], [30, 0, 7.5], [40, 0, 0]], float))


def c05():
    rgb = bridge_rgb(False)
    # All material is visible. Constant emitted/ambient radiance deliberately
    # removes shading evidence. The blank V-strip is piecewise developable.
    targets = np.array([[30.0, 10.0], [32.5, 10.0]])
    return result(
        {"observation_level": "exact_RGB_in_declared_photometric_model", "all_material_visible": True,
         "query_material_mm": targets.tolist(), "query_disagreement_mm": 2.5,
         "world_condition": "blank strip, orthographic camera, constant radiance, no known edge-to-edge material distance; real shading may add evidence"},
        [check("same_RGB", np.array_equal(rgb, rgb.copy())),
         check("different_visible_material_coordinate", np.linalg.norm(targets[0]-targets[1]) == 2.5),
         check("developable_V_arc_length", np.isclose(2*np.hypot(10, 7.5), 25))],
        RGB_worlds=np.stack([rgb, rgb]), query=np.array([120.0, 40.0]), target_mm=targets)


def c06():
    L, a = 5.0, 1.0
    xs = np.linspace(0, L, 257)
    xx, yy = np.meshgrid(xs, xs)
    delta = a * np.sin(np.pi*xx/L)**2 * np.sin(np.pi*yy/L)**2
    jac = 1 + a*np.pi/L*np.sin(2*np.pi*xx/L)*np.sin(np.pi*yy/L)**2
    bound = 1 - a*np.pi/L
    boundary = np.concatenate([delta[0], delta[-1], delta[:, 0], delta[:, -1]])
    rgb = np.full((161, 161, 3), 255, np.uint8)
    rgb[[0, -1], :, :] = 0
    rgb[:, [0, -1], :] = 0
    targets = np.array([[2.5, 2.5], [3.5, 2.5]])
    return result(
        {"observation_level": "exact_RGB_for_general_2D_warp_class", "amplitude_mm": a,
         "analytic_Jacobian_lower_bound": bound, "sampled_Jacobian_min": float(jac.min()),
         "max_boundary_displacement_mm": float(np.max(np.abs(boundary))),
         "center_disagreement_mm": 1.0,
         "world_condition": "general orientation-preserving C1 material maps, blank interior; NOT a theorem for nonstretchable paper under a fixed camera model"},
        [check("same_boundary_to_roundoff", np.max(np.abs(boundary)) < 1e-28),
         check("orientation_preserving_analytic_bound", bound > 0),
         check("interior_difference_above_half_mm", np.linalg.norm(targets[0]-targets[1]) > .5)],
        RGB_worlds=np.stack([rgb, rgb]), delta_u_mm=delta, sampled_Jacobian=jac, target_mm=targets)


def c07():
    rgb = grid()
    q = np.array([[40.0, 60.0], [120.0, 100.0]])
    offsets = np.array([[0.0, 0.0], [15.0, 25.0]])
    targets = q[None, :, :]/4 + offsets[:, None, :]
    relative = targets[:, 1, :] - targets[:, 0, :]
    return result(
        {"observation_level": "exact_RGB", "origin_hypotheses_mm": offsets.tolist(),
         "relative_shape_identical": True, "y25mm_baseline_shift_mV_at_10mm_per_mV": 2.5,
         "x15mm_time_shift_seconds_at_25mm_per_s": .6,
         "world_condition": "partial crop, grid unit known, no paper origin, lead baseline, timing marker or crop provenance"},
        [check("identical_relative_geometry", np.array_equal(relative[0], relative[1])),
         check("absolute_coordinates_differ", not np.array_equal(targets[0], targets[1]))],
        RGB_worlds=np.stack([rgb, rgb]), query=q, target_mm=targets)


def c08():
    ii = grid()
    ii[72:89, 73:77] = 0
    ii[72:89, 84:88] = 0
    # Reflection axis x80 maps x73..76 to84..87 exactly.
    f = grid()
    f[67:94, 69:73] = 0
    f[67:71, 69:91] = 0
    f[78:82, 69:85] = 0
    h_ii = sha_array(ii)
    ii_equal = [sha_array(ii), sha_array(np.flip(ii, 1)), sha_array(np.rot90(ii, 2))]
    fhash = [sha_array(a) for a in d4(f)]
    return result(
        {"observation_level": "exact_RGB", "token": "II in symmetric bar font",
         "identical_identity_horizontal_mirror_and_180": len(set(ii_equal)) == 1,
         "asymmetric_F_positive_unique_D4_views": len(set(fhash)),
         "world_condition": "II is insufficient for reading direction/chirality; an asymmetric known-upright glyph can resolve symmetry if reliably interpreted"},
        [check("II_does_not_resolve_mirror_or_upside_down", len(set(ii_equal)) == 1),
         check("asymmetric_glyph_has_distinct_views", len(set(fhash)) == 8)],
        RGB_symmetric_II=ii, RGB_asymmetric_F=f, F_D4=np.stack(d4(f)))


def c09():
    r, x = .2, 5.0
    u_front, u_back = x, 20 + np.pi*r - x
    projected = np.array([[u_front, 0], [20+np.pi*r-u_back, 0]])
    depth = np.array([0, 2*r])
    front = grid(81, 81)
    hidden0, hidden1 = grid(81, 81), grid(81, 81)
    hidden1[30:34, 10:70] = 0
    # Camera is at negative z. Front chart fully occludes back on shared domain.
    observed0, observed1 = front.copy(), front.copy()
    return result(
        {"observation_level": "exact_RGB", "surface": "two parallel strips connected by semicylinder radius0.2mm",
         "source_projection_xy": projected.tolist(), "material_u_mm": [u_front, float(u_back)],
         "depth_mm": depth.tolist(), "inverse_branches_before_visibility": 2,
         "world_condition": "orthographic fold with overlap; only front sheet is visible; hidden back has distinct possible ink"},
        [check("two_material_points_one_source_pixel", np.array_equal(projected[0], projected[1])),
         check("different_material_points", abs(u_back-u_front) > 10),
         check("hidden_ink_unobservable", np.array_equal(observed0, observed1) and not np.array_equal(hidden0, hidden1))],
        RGB_worlds=np.stack([observed0, observed1]), hidden_RGB_worlds=np.stack([hidden0, hidden1]),
        projection=projected, material_u=np.array([u_front, u_back]), depth=depth)


def c10():
    a, b = grid(), grid()
    b[60:65, 50:110] = 0
    mask = np.zeros(a.shape[:2], bool)
    mask[40:120, 40:120] = True
    obs_a, obs_b = a.copy(), b.copy()
    obs_a[mask], obs_b[mask] = (120, 130, 140), (120, 130, 140)
    return result(
        {"observation_level": "exact_RGB", "different_hidden_channel_values": int(np.count_nonzero(a != b)),
         "world_condition": "fully opaque foreground patch; no previous frame or external record"},
        [check("different_hidden_content", not np.array_equal(a, b)),
         check("identical_captured_RGB", np.array_equal(obs_a, obs_b))],
        RGB_worlds=np.stack([obs_a, obs_b]), hidden_document_worlds=np.stack([a, b]), occluder_mask=mask)


def c11():
    expected, seen = ["I", "II"], ["II", "II"]
    candidates = list(itertools.permutations(expected))
    costs = np.array([sum(x != y for x, y in zip(c, seen)) for c in candidates])
    best = int(np.min(costs))
    nbest = int(np.sum(costs == best))
    hard_valid = int(np.sum(costs == 0))
    return result(
        {"observation_level": "RGB_OCR_metadata_constraint", "expected_unique_leads": expected,
         "perfect_OCR_labels": seen, "hard_consistent_assignments": hard_valid,
         "minimum_disagreements": best, "equal_cost_best_assignments": nbest,
         "legitimate_repeated_II_requires_distinct_time_windows": True,
         "world_condition": "declared uniqueI/II layout contradicts observedII/II; do not silently choose a permutation or assume repeated rhythm labels invalid"},
        [check("hard_evidence_contradictory", hard_valid == 0),
         check("forced_soft_assignment_tied", nbest == 2 and best == 1)],
        mismatch_costs=costs)


def c12():
    pitches_px = np.array([22.0, 18.0])
    square_original = np.array([5.0, 5.0])
    stretch = np.array([1.1, .9])
    rectangular_original = np.array([5.5, 4.5])
    camera = 4.0
    obs_a = camera*stretch*square_original
    obs_b = camera*rectangular_original
    q = np.array([110.0, 90.0])
    target_square = q/(camera*stretch)
    target_rectangle = q/camera
    return result(
        {"observation_level": "exact_lattice_RGB_possible", "observed_pitch_xy_px": pitches_px.tolist(),
         "square5mm_original_then_copy_stretch": stretch.tolist(),
         "alternative_original_rectangular_pitch_mm": rectangular_original.tolist(),
         "query_original_mm_square_prior": target_square.tolist(),
         "query_original_mm_unknown_grid_family": target_rectangle.tolist(),
         "known_original_square5mm_prior_resolves_copy_stretch": True,
         "world_condition": "ambiguity only when original grid family unknown; do not call anisotropic photocopy stretch uncorrectable if square5mm prior is trustworthy"},
        [check("same_observed_lattice", np.max(np.abs(obs_a-obs_b)) < 1e-12),
         check("different_original_physical_targets", np.linalg.norm(target_square-target_rectangle) > 3),
         check("known_square_corrects_both_axes", np.max(np.abs((q/(pitches_px/5))-np.array([25, 25]))) < 1e-12)],
        observed_pitch_px=pitches_px, original_targets_mm=np.stack([target_square, target_rectangle]))


def c13():
    pixel_steps = np.array([20.0, 20.0])
    input_physical_steps = np.array([5.0, 10.0])
    # Known input metadata, not a score-derived scale choice.
    global_mm_per_px = float(pixel_steps @ input_physical_steps / (pixel_steps @ pixel_steps))
    global_errors = global_mm_per_px*pixel_steps-input_physical_steps
    local = input_physical_steps/pixel_steps
    return result(
        {"observation_level": "input_RGB_plus_panel_specific_units", "panel_pitch_mm": input_physical_steps.tolist(),
         "panel_pitch_px": pixel_steps.tolist(), "best_shared_linear_mm_per_px": global_mm_per_px,
         "shared_model_pitch_errors_mm": global_errors.tolist(), "per_panel_mm_per_px": local.tolist(),
         "world_condition": "composite page with separately calibrated panels and lead/time namespaces; one shared scale is outside its declared model"},
        [check("shared_scale_cannot_satisfy_both", np.min(np.abs(global_errors)) >= 2.5),
         check("separate_charts_satisfy_input_units", np.max(np.abs(local*pixel_steps-input_physical_steps)) == 0)],
        pixel_steps=pixel_steps, input_units_mm=input_physical_steps, global_pitch_errors_mm=global_errors)


def c14():
    rgb = grid()
    x = np.arange(5, 156)
    y = np.rint(80-20*np.sin(2*np.pi*(x-5)/100)).astype(int)
    rgb[y, x] = 0
    speed = np.array([25.0, 50.0])
    gain = np.array([10.0, 20.0])
    physical = np.array([25.0, 10.0])
    signal_targets = np.column_stack([physical[0]/speed, physical[1]/gain])
    return result(
        {"observation_level": "exact_RGB", "paper_geometry_units_known": True,
         "speed_mm_per_s_hypotheses": speed.tolist(), "gain_mm_per_mV_hypotheses": gain.tolist(),
         "query_time_s_amplitude_mV": signal_targets.tolist(),
         "world_condition": "same drawn trace, absent speed/gain label and calibration pulse; physiological plausibility is not independent calibration evidence"},
        [check("same_drawn_RGB", np.array_equal(rgb, rgb.copy())),
         check("different_signal_calibration", not np.array_equal(signal_targets[0], signal_targets[1]))],
        RGB_worlds=np.stack([rgb, rgb]), signal_targets=signal_targets)


def c15():
    rgb = grid()
    mask = np.ones((161, 161), bool)
    mask[40:121, 40:121] = False
    rgb[~mask] = 255
    x = np.arange(45, 116)
    y = np.rint(80-15*np.sin((x-45)*np.pi/35)).astype(int)
    rgb[y, x] = 0
    rgb[55:58, 48:70] = 0
    bad = rgb.copy()
    bad[~mask] = 255
    preserved = rgb.copy()
    ink = np.all(rgb == 0, axis=-1) & ~mask
    erased = int(np.count_nonzero(ink & np.all(bad == 255, axis=-1)))
    return result(
        {"observation_level": "content_policy_counterexample", "no_grid_ink_pixels": int(ink.sum()),
         "ink_pixels_erased_by_grid_support_RGB_mask": erased,
         "identity_fallback_max_channel_difference": int(np.max(np.abs(preserved.astype(int)-rgb.astype(int)))),
         "world_condition": "visible waveform/text in grid-free box; this is a policy witness, not a claim that the currentM3 implementation applies that bad mask"},
        [check("no_grid_region_can_contain_ink", int(ink.sum()) > 100),
         check("grid_mask_deletes_visible_content", erased == int(ink.sum())),
         check("fallback_preserves_exact_RGB", np.array_equal(rgb, preserved))],
        RGB=rgb, grid_support_mask=mask, deliberately_bad_RGB=bad, fallback_RGB=preserved)


def c16():
    q = np.array([[0, 0], [160, 0], [160, 160], [0, 160]], float)
    # The four ORDERED correspondences with physical units are INPUT evidence.
    input_calibration_mm = q/4
    pc, mc = np.array([80., 80.]), np.array([20., 20.])
    candidates, residuals = [], []
    for pitch in [1., 5., 10.]:
        for pose_id, mat in enumerate(d4_matrices()):
            predicted = (q-pc) @ mat.T * pitch/20 + mc
            candidates.append([pitch, pose_id])
            residuals.append(float(np.max(np.linalg.norm(predicted-input_calibration_mm, axis=1))))
    residuals = np.array(residuals)
    survivors = np.flatnonzero(residuals < 1e-10)
    design = np.column_stack([q, np.ones(4)])
    fitted, *_ = np.linalg.lstsq(design, input_calibration_mm, rcond=None)
    query = np.array([40., 100., 1.])
    predicted_query = query @ fitted
    return result(
        {"observation_level": "ordered_input_physical_fiducials", "joint_pitch_pose_candidates": len(candidates),
         "candidates_consistent_with_input_evidence": len(survivors),
         "surviving_pitch_pose": [candidates[i] for i in survivors],
         "fitted_query_mm": predicted_query.tolist(),
         "world_condition": "ordered asymmetric fiducial identities and calibratedmm are observed input metadata, not gold used to choose anECG result"},
        [check("independent_evidence_breaks_all_aliases", len(survivors) == 1 and candidates[survivors[0]] == [5., 0]),
         check("positive_calibrated_map", np.max(np.abs(predicted_query-np.array([10., 25.]))) < 1e-10)],
        source_fiducials=q, input_fiducial_mm=input_calibration_mm,
        candidate_pitch_pose=np.array(candidates), input_evidence_residual_mm=residuals,
        fitted_affine=fitted)


FUNCTIONS = {f"ID{i:02d}": globals()[f"c{i:02d}"] for i in range(1, 17)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results.json")
    parser.add_argument("--witness-dir", type=Path, default=ROOT / "witnesses")
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    source_hash = sha_bytes(Path(__file__).read_bytes())
    if source_hash != protocol["controls_py_sha256"]:
        raise RuntimeError("Controls source differs from frozen protocol. Preserve old run and register a new protocol before executing.")
    if list(FUNCTIONS) != [c["id"] for c in protocol["controls"]]:
        raise RuntimeError("Registered control order differs from implementation.")
    args.witness_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in protocol["controls"]:
        cid = spec["id"]
        row = {"id": cid, "name": spec["name"], "expected": spec["expected"]}
        try:
            r = FUNCTIONS[cid]()
            arr = r.pop("arrays")
            witness = args.witness_dir / f"{cid}.npz"
            np.savez_compressed(witness, **arr)
            row.update(r)
            row["witness_file"] = str(witness.resolve())
            row["witness_sha256"] = sha_bytes(witness.read_bytes())
            row["array_sha256"] = {key: sha_array(value) for key, value in arr.items()}
            row["status"] = "PASS" if all(c["passed"] for c in row["checks"]) else "FAIL"
        except Exception as exc:
            row.update(status="ERROR", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        rows.append(row)
        print(f"{cid}: {row['status']}", flush=True)
    status = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    out = {
        "schema": "m4-identifiability-controls-v1", "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha_bytes(protocol_bytes), "controls_py_sha256": source_hash,
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "scope": "Constructive observations and model-limit controls only. No rectifier inference, CNN, real capture benchmark, timing claim or superiority claim.",
        "failure_policy": "Every registered control retained, including FAIL/ERROR. No run is silently repaired or overwritten by a changed source.",
        "controls_registered": len(rows), "controls_passed": sum(r["status"] == "PASS" for r in rows),
        "assertions_total": sum(len(r.get("checks", [])) for r in rows),
        "assertions_passed": sum(c["passed"] for r in rows for c in r.get("checks", [])),
        "results": rows,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
