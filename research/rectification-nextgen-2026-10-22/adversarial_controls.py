"""Independent fault-injection controls for point-based ECG geometry.

These controls expose an information limit and an assumption violation.
They are not a new benchmark, a clinical acceptance test or a novelty claim.
Run with --geometry-dir pointing to the directory containing the three audited
production modules complete_rectifier.py, frame_mesh.py and cell_mesh.py.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def setup(geometry_dir):
    from audit_loader import load_geometry
    return load_geometry(geometry_dir).fit_complete_map


def grid():
    row, col = np.mgrid[:7, :7]
    ids = np.column_stack((row.ravel(), col.ravel()))
    canonical = ids[:, ::-1]*5.
    return canonical*10, canonical, ids


def same_boundary_alternative(physical):
    """A C1 orientation-preserving map equal to identity outside one cell."""
    output = physical.copy()
    local = (physical-[10., 10.])/5
    active = ((local >= 0) & (local <= 1)).all(1)
    x, y = local[active].T
    output[active, 0] += 1.2*np.sin(np.pi*x)**2*np.sin(np.pi*y)**2
    return output


def run(geometry_dir):
    fit_complete_map = setup(geometry_dir)
    observed, canonical, ids = grid()
    mapping, report = fit_complete_map(observed, canonical, ids, (301, 301), np.ones(len(ids)))
    assert mapping is not None
    query = np.array([[125., 125.], [175., 175.]])
    predicted = mapping.forward(query)
    backward = mapping.inverse(predicted)
    eligible = mapping.eligible_geometry(query)
    assert eligible.all()
    roundtrip = float(np.linalg.norm(backward-query, axis=1).max())
    # Deliberate front-end mistake: intersections of a 1 mm grid are treated
    # as a 5 mm grid. This violates the major-grid assumption supplied to fit.
    true_minor_mm = query/50
    alias_errors = np.linalg.norm(predicted-true_minor_mm, axis=1)
    low_confidence, low_report = fit_complete_map(observed, canonical, ids, (301, 301), np.zeros(len(ids)))
    assert low_confidence is not None
    same_scores = (np.array_equal(mapping.source, low_confidence.source)
                   and np.array_equal(mapping.target, low_confidence.target)
                   and np.array_equal(mapping.triangles, low_confidence.triangles))
    # Alternative truth on the observed cell interior. All grid intersections
    # and every grid line stay fixed, but the center moves by 1.2 mm.
    q = np.array([[125., 125.]])
    base_truth = q/10
    alternative_truth = same_boundary_alternative(base_truth)
    anchor_error = float(np.max(np.abs(same_boundary_alternative(canonical)-canonical)))
    t = np.linspace(0, 1, 1001)
    boundary = np.vstack((np.column_stack((t, np.zeros_like(t))),
                          np.column_stack((t, np.ones_like(t))),
                          np.column_stack((np.zeros_like(t), t)),
                          np.column_stack((np.ones_like(t), t))))*5+[10, 10]
    boundary_error = float(np.max(np.abs(same_boundary_alternative(boundary)-boundary)))
    x, y = np.meshgrid(t, t)
    determinant = 1+(1.2*np.pi/5)*np.sin(2*np.pi*x)*np.sin(np.pi*y)**2
    bound = 1-1.2*np.pi/5
    separation = float(np.linalg.norm(alternative_truth-base_truth, axis=1)[0])
    actual_error = float(np.linalg.norm(mapping.forward(q)-alternative_truth, axis=1)[0])
    assert anchor_error < 1e-12 and boundary_error < 1e-12
    assert determinant.min() >= bound-1e-12 and bound > 0
    assert mapping.eligible_geometry(q).all() and abs(actual_error-1.2) < 1e-12
    assert (alias_errors > .5).all()
    names = ['complete_rectifier.py', 'frame_mesh.py', 'cell_mesh.py']
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy', 'scipy', 'shapely', 'opencv-python-headless']}
    return {
        'audited_commit': 'b9fd04dd6516e3469d1f36593849249c082458a8',
        'source_sha256': {name: hashlib.sha256((geometry_dir/name).read_bytes()).hexdigest() for name in names},
        'versions': versions,
        'supported_opencv_major': versions['opencv-python-headless'].startswith('4.'),
        'environment_scope': 'Analytic controls only. This run is not the project integration suite.',
        'scale_alias': {
            'kind': 'Deliberate 1 mm to 5 mm major-grid assignment fault injection',
            'assumption_violation': True,
            'fit_status': report['status'],
            'inlier_fraction': report['global_inlier_fraction'],
            'predicted_mm': predicted.tolist(), 'true_mm': true_minor_mm.tolist(),
            'errors_mm': alias_errors.tolist(), 'eligible': eligible.tolist(),
            'roundtrip_max_px': roundtrip,
            'zero_confidence_geometry_identical': bool(same_scores),
            'zero_confidence_fit_status': low_report['status'],
            'claim': 'Topology and self-consistency cannot validate an externally supplied physical scale assignment.'},
        'subcell_identifiability': {
            'kind': 'Two C1 2D maps with identical grid intersections and grid boundaries',
            'cell_mm': 5., 'bump_amplitude_mm': 1.2,
            'analytic_minimum_jacobian_determinant': float(bound),
            'sampled_minimum_jacobian_determinant': float(determinant.min()),
            'maximum_anchor_disagreement_mm': anchor_error,
            'maximum_cell_boundary_disagreement_mm': boundary_error,
            'center_truth_separation_mm': separation,
            'point_or_boundary_only_minimax_error_lower_bound_mm': separation/2,
            'current_estimator_error_under_alternative_truth_mm': actual_error,
            'current_estimator_center_eligible': bool(mapping.eligible_geometry(q)[0]),
            'current_evidence_label': int(mapping.evidence(q)[0]),
            'scope': 'Only the unconstrained C1 2D map class. No impossibility claim for calibrated 3D geometry, known isometry/curvature bounds or additional image evidence.',
            'novelty_claim': False}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir', type=Path,
        default=Path(__file__).resolve().parents[2]/'ecg_gpu/digitization')
    parser.add_argument('--out', type=Path,
        default=Path(__file__).with_name('adversarial_controls.json'))
    args = parser.parse_args()
    result = run(args.geometry_dir)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
