#!/usr/bin/env python3
"""Check saved witnesses without importing or calling controls.py.

Reconstruct physical bridge arclengths from3D polyline segments, then render
material-grid observations through an orthographic camera. This is an ideal
analytic renderer, not an independent photographic capture simulator.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

P = Path(__file__).resolve().parent
checks = []


def verify(name, passed, **details):
    checks.append({"name": name, "passed": bool(passed), **details})


def material_bridge(x, vertices):
    # Inputx is projected distance. Derive material arclength independently
    # from each3D segment, rather than copying hardcoded1.25 slopes.
    segment_length = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    projected_length = np.diff(vertices[:, 0])
    out = np.asarray(x, float).copy()
    left, mid, right = vertices[:, 0]
    first = (x > left) & (x <= mid)
    second = (x > mid) & (x < right)
    after = x >= right
    out[first] = left+(x[first]-left)*segment_length[0]/projected_length[0]
    out[second] = left+segment_length[0]+(x[second]-mid)*segment_length[1]/projected_length[1]
    out[after] = left+segment_length.sum()+(x[after]-right)
    return out, segment_length


def render_world(pitch, pixels_per_mm, shape):
    y, x = np.indices(shape)
    u, v = x/pixels_per_mm, y/pixels_per_mm
    on_grid = (np.abs(u/pitch-np.rint(u/pitch)) < 1e-10) | (np.abs(v/pitch-np.rint(v/pitch)) < 1e-10)
    rgb = np.full((*shape, 3), 255, np.uint8)
    rgb[on_grid] = (220, 100, 100)
    return rgb


def render_bridge(vertices, opaque):
    y, x = np.indices((81, 241))
    x, y = x/4.0, y/4.0
    u, segment_lengths = material_bridge(x, vertices)
    v = y
    on_grid = (np.abs(u/5-np.rint(u/5)) < 1e-10) | (np.abs(v/5-np.rint(v/5)) < 1e-10)
    rgb = np.full((81, 241, 3), 255, np.uint8)
    rgb[on_grid] = (220, 100, 100)
    blank_or_occluder = (x > vertices[0, 0]) & (x < vertices[-1, 0])
    rgb[blank_or_occluder] = (160, 160, 160) if opaque else (255, 255, 255)
    return rgb, u, v, segment_lengths


def main():
    protocol = json.loads((P/"audit-protocol.json").read_text())
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_sha != protocol["audit_source_sha256"]:
        raise RuntimeError("Independent audit source changed after freeze")
    r = json.loads((P/"results.json").read_text())
    receipt = json.loads((P/"protocol-freeze.json").read_text())
    verify("main_protocol_unchanged", hashlib.sha256((P/"protocol.json").read_bytes()).hexdigest() == receipt["protocol_sha256"])
    verify("main_source_unchanged", hashlib.sha256((P/"controls.py").read_bytes()).hexdigest() == receipt["controls_py_sha256"])
    verify("all_registered_results_preserved", [x["id"] for x in r["results"]] == [f"ID{i:02d}" for i in range(1, 17)])
    for row in r["results"]:
        data = (P/"witnesses"/f"{row['id']}.npz").read_bytes()
        verify(row["id"]+"_saved_witness_hash", hashlib.sha256(data).hexdigest() == row["witness_sha256"])
    w = np.load(P/"witnesses/ID01.npz")
    for i, pitch in enumerate(w["pitch_mm"]):
        generated = render_world(float(pitch), 20/float(pitch), w["RGB_worlds"][i].shape[:2])
        verify(f"ID01_world_{i}_independent_material_grid_render", np.array_equal(generated, w["RGB_worlds"][i]))
    flat = np.array([[20., 0, 0], [30., 0, 0], [40., 0, 0]])
    folded = np.array([[20., 0, 0], [30., 0, 7.5], [40., 0, 0]])
    for cid, opaque, query_x in [("ID04", True, 50.), ("ID05", False, 30.)]:
        w = np.load(P/f"witnesses/{cid}.npz")
        images, material_targets = [], []
        for i, vertices in enumerate([flat, folded]):
            rgb, u, v, lengths = render_bridge(vertices, opaque)
            images.append(rgb)
            target_x, _ = material_bridge(np.array([query_x]), vertices)
            target = np.array([target_x[0], 10.])
            material_targets.append(target)
            verify(f"{cid}_world_{i}_3D_render_matches_saved_RGB", np.array_equal(rgb, w["RGB_worlds"][i]), segment_arclengths_mm=lengths.tolist())
            verify(f"{cid}_world_{i}_3D_arclength_matches_target", np.max(np.abs(target-w["target_mm"][i])) < 1e-12)
        verify(cid+"_independently_rendered_worlds_identical", np.array_equal(images[0], images[1]))
    w = np.load(P/"witnesses/ID06.npz")
    coords = np.linspace(0, 5, w["delta_u_mm"].shape[1])
    mapped_x = coords[None, :] + w["delta_u_mm"]
    increments = np.diff(mapped_x, axis=1)
    verify("ID06_saved_rows_strictly_increasing", np.min(increments) > 0, min_step_mm=float(increments.min()))
    verify("ID06_analytic_bound_positive", 1-np.pi/5 > 0)
    w = np.load(P/"witnesses/ID09.npz")
    r_hinge = .2
    u0, u1 = w["material_u"]
    xyz0 = np.array([u0, 0, 0.])
    xyz1 = np.array([10-(u1-10-np.pi*r_hinge), 0, 2*r_hinge])
    verify("ID09_physical_strips_same_orthographic_projection", np.max(np.abs(xyz0[:2]-xyz1[:2])) < 1e-12)
    verify("ID09_physical_back_strip_is_deeper", xyz1[2] > xyz0[2])
    out = {"schema":"m4-identifiability-independent-audit-v1", "created_utc":datetime.now(timezone.utc).isoformat(),
           "source_sha256":source_sha, "protocol_sha256":hashlib.sha256((P/"audit-protocol.json").read_bytes()).hexdigest(),
           "scope":"No controls.py import. Hash-integrity and independent analytic3D arclength/material-grid render checks. No independent camera/photographic realism evidence.",
           "checks_total":len(checks), "checks_passed":sum(c["passed"] for c in checks),
           "status":"PASS" if all(c["passed"] for c in checks) else "FAIL", "checks":checks}
    (P/"independent-audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
    print({k:out[k] for k in ["status","checks_total","checks_passed"]})
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
