"""Independent analytic physical-query controls for fixed-ID grid refinement.

These are synthetic known-camera controls, not CNN or real-patient results.
The negative unguarded crease results remain alongside guarded results.
Run with OMP/MKL/OPENBLAS_NUM_THREADS=2 to bound concurrent CPU work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = (0., .05, .15, .5, 1.)
SOURCE_FILES = (
    'examples/refinement_control_study.py',
    'ecg_gpu/digitization/robust_refinement.py',
    'ecg_gpu/digitization/cell_mesh.py',
)
CAMERA = np.array([[2.2, .4, 31.], [-.15, 2.7, 26.], [.0015, -.001, 1.]])
CASES = (
    {'key': 'projective-noisy-7919', 'kind': 'projective', 'seed': 7919, 'sigma_px': .7, 'outliers': 10},
    {'key': 'projective-noisy-1973', 'kind': 'projective', 'seed': 1973, 'sigma_px': .7, 'outliers': 10},
    {'key': 'projective-noisy-104729', 'kind': 'projective', 'seed': 104729, 'sigma_px': .7, 'outliers': 10},
    {'key': 'smooth-clean', 'kind': 'smooth', 'seed': 104729, 'amplitude_px': 6., 'rotation': 0.},
    {'key': 'crease-clean', 'kind': 'crease', 'seed': 104729, 'amplitude_px': 6., 'location_mm': 25., 'rotation': 0.},
    {'key': 'smooth-rotated', 'kind': 'smooth', 'seed': 65537, 'amplitude_px': 6., 'rotation': .61},
    {'key': 'crease-rotated-relocated', 'kind': 'crease', 'seed': 65537, 'amplitude_px': 3., 'location_mm': 20., 'rotation': .61},
    {'key': 'crease-weak', 'kind': 'crease', 'seed': 65537, 'amplitude_px': .2, 'location_mm': 30., 'rotation': -.27},
    {'key': 'crease-noisy', 'kind': 'crease', 'seed': 65537, 'amplitude_px': 6., 'location_mm': 25., 'rotation': 0., 'sigma_px': .7},
    {'key': 'crease-oblique', 'kind': 'oblique', 'seed': 65537, 'amplitude_px': 6., 'location_mm': 35., 'rotation': .31},
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix+'.part')
    staged.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    staged.replace(path)


def analytic_camera(points, case):
    if case['kind'] == 'projective':
        homogeneous = np.column_stack((points, np.ones(len(points))))@CAMERA.T
        return homogeneous[:, :2]/homogeneous[:, 2:]
    u, v = points.T
    if case['kind'] == 'smooth':
        displacement = case['amplitude_px']*np.sin(u/15)*np.sin(v/14)
    else:
        argument = u if case['kind'] == 'crease' else u+.5*v
        displacement = case['amplitude_px']*np.maximum(argument-case['location_mm'], 0)/5
    observed = np.column_stack((u*2.5+30, v*2.5+20+displacement))
    angle = case.get('rotation', 0.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return observed@rotation.T+[70., 35.] if angle else observed


def fixture(case, queries=1500):
    row, col = np.mgrid[:10, :12]
    ids = np.column_stack((row.ravel(), col.ravel()))
    material = ids[:, ::-1]*5.
    truth = analytic_camera(material, case)
    rng = np.random.default_rng(case['seed'])
    observed = truth.copy()
    sigma = case.get('sigma_px', 0.)
    if sigma:
        observed += rng.normal(0, sigma, observed.shape)
    scores = np.ones(len(ids))
    if case.get('outliers', 0):
        # Ranking is deliberately controlled in this observation experiment.
        # It is not an empirically calibrated CNN confidence distribution.
        outliers = rng.choice(len(ids), case['outliers'], replace=False)
        observed[outliers] += rng.normal(0, 3, (len(outliers), 2))
        scores = rng.uniform(.8, 1., len(ids))
        scores[outliers] = rng.uniform(.2, .4, len(outliers))
    query_mm = rng.uniform([5., 5.], [50., 40.], (queries, 2))
    query = analytic_camera(query_mm, case)
    cells = [[r, c] for r in range(9) for c in range(11)]
    return ids, material, truth, observed, scores, query, query_mm, cells


def metrics(predicted, reference, fixed_mask):
    accepted = fixed_mask & np.isfinite(predicted).all(1)
    error = np.linalg.norm(predicted-reference, axis=1)
    denominator = int(fixed_mask.sum())
    good = accepted & (error <= .5)
    wrong = accepted & (error > .5)
    values = error[accepted]
    return {
        'denominator_queries': denominator,
        'accepted_queries': int(accepted.sum()),
        'G': float(good.sum()/denominator) if denominator else 0.,
        'W': float(wrong.sum()/denominator) if denominator else 0.,
        'C': float(accepted.sum()/denominator) if denominator else 0.,
        'R': float(wrong.sum()/accepted.sum()) if accepted.any() else None,
        'physical_rmse_mm': float(np.sqrt(np.mean(values**2))) if len(values) else None,
        'physical_p95_mm': float(np.quantile(values, .95)) if len(values) else None,
        'physical_p99_mm': float(np.quantile(values, .99)) if len(values) else None,
    }


def run(out):
    sys.path.insert(0, str(ROOT))
    from ecg_gpu.digitization.cell_mesh import forward_cell_map
    from ecg_gpu.digitization.robust_refinement import refine_grid_points

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: digest(ROOT/name) for name in SOURCE_FILES}
    protocol = {
        'scope': 'independent analytic camera controls, fixed known IDs, no detector or training',
        'cases': CASES, 'strengths': STRENGTHS, 'guard_modes': [False, True],
        'grid_shape': [10, 12], 'physical_spacing_mm': 5., 'queries_per_case': 1500,
        'query_domain_mm': [[5., 5.], [50., 40.]], 'error_threshold_mm': .5,
        'score_contract': 'controlled ranking for location outliers, unit weights in other cases',
        'main_eligibility': 'B0 finite coordinate support, fixed across refinements',
        'additional_domain': 'all predefined physical queries, with missing outputs in denominator',
        'evaluation_alignment': 'none, analytic camera defines absolute physical coordinates',
        'cell_topology': 'same two fixed triangles per 5 mm cell in B0 and all candidate arms',
        'guard_caveat': 'Conservative prior-conflict heuristic. Coherent detection error can trigger and noisy/oblique bends may escape.',
        'design_status': 'development controls, not a previously untouched image benchmark',
        'statistical_scope': 'No independence or population inference from these synthetic scenes',
        'source_sha256': source_hashes,
    }
    protocol_path = out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Control protocol or source changed. Use a new result directory.')
    write_json(protocol_path, protocol)
    records = []
    (out/'arrays').mkdir(exist_ok=True)
    for case in CASES:
        ids, material, truth, observed, scores, query, reference, cells = fixture(case)
        baseline = forward_cell_map(observed, material, ids, query, cells)
        fixed = np.isfinite(baseline).all(1)
        arrays = {'ids': ids, 'canonical_mm': material, 'true_xy': truth, 'observed_xy': observed,
                  'scores': scores, 'query_xy': query, 'reference_mm': reference,
                  'fixed_B0_eligible': fixed, 'cells': np.asarray(cells)}
        record = {'case': case, 'methods': {}, 'baseline_support_queries': int(fixed.sum())}
        for strength in STRENGTHS:
            guards = (False,) if strength == 0 else (False, True)
            for guard in guards:
                name = 'B0' if strength == 0 else f'soft-{strength}-guard-{str(guard).lower()}'
                started = time.perf_counter()
                refined, report = refine_grid_points(observed, ids, scores,
                    strength=strength, crease_guard=guard)
                failure = None
                try:
                    predicted = forward_cell_map(refined, material, ids, query, cells)
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    failure = type(error).__name__+': '+str(error)
                    predicted = np.full(query.shape, np.nan)
                row = {
                    'fixed_B0_support': metrics(predicted, reference, fixed),
                    'all_queries': metrics(predicted, reference, np.ones(len(query), bool)),
                    'refinement_report': report, 'mapping_failure': failure,
                    'node_rmse_px': float(np.sqrt(np.mean(np.sum((refined-truth)**2, axis=1)))),
                    'seconds_component_control': time.perf_counter()-started,
                }
                common = fixed & np.isfinite(predicted).all(1)
                row['common_with_B0'] = {
                    'queries': int(common.sum()),
                    'B0': metrics(baseline, reference, common),
                    'candidate': metrics(predicted, reference, common),
                }
                record['methods'][name] = row
                arrays[name+'_refined_xy'] = refined
                arrays[name+'_predicted_mm'] = predicted
        path = out/'arrays'/(case['key']+'.npz')
        np.savez_compressed(path, **arrays)
        record['arrays'] = str(path.relative_to(out))
        record['arrays_sha256'] = digest(path)
        records.append(record)
        write_json(out/'results.json', {'protocol_sha256': digest(protocol_path),
            'cases': records, 'complete': len(records) == len(CASES)})
        print(json.dumps({'control_complete': case['key'], 'count': len(records)}), flush=True)
    if source_hashes != {name: digest(ROOT/name) for name in SOURCE_FILES}:
        raise ValueError('A control source changed during the run')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'build/refinement-controls')
    run(parser.parse_args().out)
