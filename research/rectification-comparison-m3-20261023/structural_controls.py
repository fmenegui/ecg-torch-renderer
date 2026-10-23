"""Independent structural controls of exact M2 modules, without neural inference.

These are deliberate observation fault injections and assumption probes. They
are not a population benchmark, native competitor run, or physical realism proof.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import types
import platform

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parent
MODULES = ROOT/'source/ecg_gpu/digitization'
COMMIT = 'f4f6caef3b59b5830e93284cccadbc6855fc701e'
EXPECTED = {
    'lattice.py': '48a3e69fe00c5cfd5319c326ee83a97cf368fe4f879c82f62f7f7450e5a689b6',
    'robust_refinement.py': '0b2d2f15ba3f86e63d7532a47d6f87bccded1b5f65ac56499f2d6ed765241ced',
    'cell_mesh.py': '566253fba786de84475f00d593b80876b3869499d3703c18a71c9a5396cf37d6',
}


def setup(geometry_dir):
    package = types.ModuleType('m2_audit')
    package.__path__ = [str(geometry_dir)]
    sys.modules['m2_audit'] = package
    lattice = importlib.import_module('m2_audit.lattice')
    refine = importlib.import_module('m2_audit.robust_refinement')
    cells = importlib.import_module('m2_audit.cell_mesh')
    return lattice.organize_lattice, refine.refine_grid_points, cells.forward_cell_map


def topology_report(value):
    return {k: v for k, v in value.items() if k not in ('ids', 'cells', 'selected_indices')}


def run(geometry_dir=MODULES):
    geometry_dir = Path(geometry_dir)
    actual = {name: hashlib.sha256((geometry_dir/name).read_bytes()).hexdigest()
              for name in EXPECTED}
    if actual != EXPECTED:
        raise ValueError('Source differs from the frozen M2 implementation')
    organize, refine, forward = setup(geometry_dir)
    # A full regular sheet has major-grid intersections every 5 mm. The
    # observation loses every odd row and column, retaining a perfect subgrid.
    row, col = np.mgrid[:11:2, :11:2]
    true_ids = np.column_stack((row.ravel(), col.ravel()))
    observed = true_ids[:, ::-1]*50.+[10., 10.]
    topology = organize(observed)
    selected = np.asarray(topology['selected_indices'], int)
    ids = np.asarray(topology['ids'], int)
    q = np.array([[260., 260.]])
    truth = np.array([[25., 25.]])
    controls = {}
    for strength in (0., .15, 1.):
        xy, report = refine(observed[selected], ids, strength=strength, crease_guard=True)
        prediction = forward(xy, ids[:, ::-1]*5., ids, q, topology['cells'])
        error = float(np.linalg.norm(prediction-truth, axis=1)[0])
        controls[str(strength)] = {'prediction_mm': prediction.tolist(),
                                  'error_mm': error, 'refinement': report}
    assert topology['status'] == 'accepted'
    assert topology['cycle_conflicts'] == 0
    assert abs(controls['1.0']['error_mm']-np.sqrt(2)*12.5) < 1e-9
    assert controls['1.0']['refinement']['reason'] == 'projective observations already consistent'

    # Two visible regular regions belong to one planar sheet, separated by
    # four absent grid columns. No wrong point is injected.
    r, c = np.mgrid[:7, :7]
    left = np.column_stack((c.ravel(), r.ravel()))*40.+[10., 10.]
    right = left+[440., 0.]
    separated = organize(np.vstack((left, right)))
    assert separated['components'] == 2
    assert separated['component_points'] == len(left)

    # Perfect points from a single finite homography. The transformation has
    # no pole within the visible sheet and no noise or occlusion.
    r, c = np.mgrid[:12, :16]
    flat = np.column_stack((c.ravel(), r.ravel()))*30.
    perspective = []
    for coefficient in (0., .001, .002, .004, .008):
        denominator = 1+coefficient*flat[:, 0]
        xy = flat/denominator[:, None]+[10., 10.]
        value = organize(xy)
        perspective.append({'projective_coefficient_per_px': coefficient,
                            'denominator_min': float(denominator.min()),
                            'denominator_max': float(denominator.max()),
                            'topology': topology_report(value),
                            'supported_cells': len(value['cells'])})
    # Geometry-only oracle-ID control. Even exact projective anchors do not
    # make a piecewise affine triangle interpolant equal to a homography at
    # cell interiors. No lattice inference or detector is involved here.
    exact_ids = np.column_stack((r.ravel(), c.ravel()))
    canonical = exact_ids[:, ::-1]*5.
    known_cells = [[row, col] for row in range(11) for col in range(15)]
    rng = np.random.default_rng(1973)
    physical_queries = rng.uniform([.001, .001], [74.999, 54.999], (4096, 2))
    geometry = []
    for coefficient in (0., .01, .02, .04, .08):
        def project(value):
            return value*10/(1+coefficient*value[:, :1])
        xy = project(canonical)
        source_queries = project(physical_queries)
        value = forward(xy, canonical, exact_ids, source_queries, known_cells)
        error = np.linalg.norm(value-physical_queries, axis=1)
        exact_inverse = source_queries/(10-coefficient*source_queries[:, :1])
        fitted_homography, _ = cv2.findHomography(xy, canonical, method=0)
        homogeneous_queries = np.column_stack((source_queries, np.ones(len(source_queries))))
        projected = homogeneous_queries@fitted_homography.T
        fitted_prediction = projected[:, :2]/projected[:, 2:]
        fitted_error = np.linalg.norm(fitted_prediction-physical_queries, axis=1)
        physical_probe = np.array([[2.5, 2.5]])
        probe = forward(xy, canonical, exact_ids, project(physical_probe), known_cells)
        # A physically straight segment crosses the first cell diagonal at
        # u=v=2.5 mm, with true slope 0.5. Tangents on either side should agree.
        epsilon = 1e-5
        along = np.array([-epsilon, 0., epsilon])
        straight = physical_probe+along[:, None]*[1., .5]
        mapped = forward(xy, canonical, exact_ids, project(straight), known_cells)
        a, b = mapped[1]-mapped[0], mapped[2]-mapped[1]
        angle = np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1)))
        assert np.isfinite(value).all()
        geometry.append({'projective_coefficient_per_mm': coefficient,
                         'canonical_grid_step_mm': 5., 'queries': len(value),
                         'piecewise_affine_rmse_mm': float(np.sqrt(np.mean(error**2))),
                         'piecewise_affine_p95_mm': float(np.quantile(error,.95)),
                         'piecewise_affine_max_mm': float(error.max()),
                         'exact_inverse_homography_max_mm': float(np.linalg.norm(exact_inverse-physical_queries,axis=1).max()),
                         'homography_fitted_same_anchors_rmse_mm': float(np.sqrt(np.mean(fitted_error**2))),
                         'homography_fitted_same_anchors_max_mm': float(fitted_error.max()),
                         'probe_physical_mm': physical_probe.tolist(),
                         'probe_piecewise_affine_mm': probe.tolist(),
                         'probe_error_mm': float(np.linalg.norm(probe-physical_probe)),
                         'straight_segment_diagonal_kink_degrees': float(angle)})
    names = ('lattice.py', 'robust_refinement.py', 'cell_mesh.py')
    return {
        'schema': 'm3-independent-structural-controls-v1',
        'audited_commit': COMMIT,
        'runtime': {'python': platform.python_version(),
                    'numpy': importlib.metadata.version('numpy'),
                    'scipy': importlib.metadata.version('scipy'),
                    'opencv': cv2.__version__},
        'scope': 'Input-only topology/refinement controls, no image detector, no full-frame RGB sampler, no competitor execution',
        'source_sha256': {name: actual[name] for name in names},
        'missing_every_second_grid_row_column': {
            'interpretation': 'Deliberate coherent detection omission and pitch ambiguity',
            'true_major_grid_pitch_mm': 5.,
            'retained_observation_pitch_mm': 10.,
            'assumed_inferred_step_mm': 5.,
            'topology': topology_report(topology),
            'query_px': q.tolist(), 'truth_mm': truth.tolist(),
            'arms': controls,
            'inference': 'Low localization noise and exact projective residual do not identify the missing integer lattice steps',
        },
        'two_visible_regions_same_sheet': {
            'observed_points': len(left)+len(right),
            'topology': topology_report(separated),
            'retained_cells': len(separated['cells']),
            'inference': 'A full visible region is discarded from coordinate inference because only the largest connected component is retained. This does not mean original RGB is erased.',
        },
        'perfect_projective_points': perspective,
        'flat_projective_geometry_oracle_ids': {
            'scope': 'Geometry-only interpolation with exact known IDs and correspondences, not native end-to-end performance',
            'inference': 'Exact anchors do not imply exact projective interior under piecewise affine interpolation. Shared edges provide C0 continuity, not C1 continuity.',
            'arms': geometry},
        'claim_limits': ['Not an estimate of how often a trained detector makes these omissions',
                         'The scale control violates the adjacent-major-grid assumption and tests whether the pipeline detects that violation',
                         'Perspective controls probe the documented moderate-perspective assumption, not a failure of projective geometry itself',
                         'No clinical or SOTA claim'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir', type=Path, default=MODULES)
    parser.add_argument('--out', type=Path, default=ROOT/'structural-controls.json')
    args = parser.parse_args()
    result = run(args.geometry_dir)
    output = args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'out': str(output), 'scope': result['scope'],
                      'alias_error_mm': result['missing_every_second_grid_row_column']['arms']['1.0']['error_mm'],
                      'disconnected_retained_fraction': result['two_visible_regions_same_sheet']['topology']['retained_point_fraction'],
                      'perspective_retained_fractions': [v['topology'].get('retained_point_fraction') for v in result['perfect_projective_points']]}))
