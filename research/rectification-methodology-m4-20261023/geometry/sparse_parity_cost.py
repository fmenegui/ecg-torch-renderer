"""Parity and CPU cost of identical dense versus sparse residual equations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import geometry_selector as dense
import geometry_selector_sparse as sparse
from run_experiment import make_inputs

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SPARSE_PROTOCOL_SHA = '348d586f1f583f331ec7891faeb36bad08ff69f18912133c2a434f4c56b56d1b'


def worker(implementation, nr, nc):
    r, c = np.mgrid[:nr, :nc]
    ids = np.c_[r.ravel(), c.ravel()]
    fn_lattice = dense.lattice if implementation == 'dense' else sparse.lattice_sparse
    fn_fit = dense.fit_field if implementation == 'dense' else sparse.fit_field_sparse
    started = time.perf_counter()
    all_ids, index, penalty = fn_lattice(ids)
    assembly = time.perf_counter()-started
    random = np.random.default_rng(96023)
    residual = np.c_[.3*np.sin(ids[:, 0]/7), .5*np.cos(ids[:, 1]/9)]
    residual += random.normal(0, .03, residual.shape)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        value = fn_fit(residual, index, len(all_ids), penalty, 3.)
        times.append(time.perf_counter()-start)
    result = {'implementation': implementation, 'shape': [nr,nc], 'nodes': len(ids),
        'assembly_seconds': assembly, 'four_irls_solve_seconds': times,
        'median_four_irls_solve_seconds': float(np.median(times)),
        'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'penalty_stored_entries': penalty.size if implementation == 'dense' else penalty.nnz,
        'penalty_bytes': penalty.nbytes if implementation == 'dense' else penalty.data.nbytes+penalty.indices.nbytes+penalty.indptr.nbytes,
        'finite_solution': bool(np.isfinite(value).all()),
        'scope': 'one field fit with four IRLS updates, excludes H, CV, candidate guards and geometry queries'}
    print(json.dumps(result))


def coordinate_parity(a, b):
    fa, fb = np.isfinite(a).all(1), np.isfinite(b).all(1)
    common = fa & fb
    return {'finite_mask_difference': int(np.sum(fa != fb)),
            'max_difference': float(np.max(np.linalg.norm(a[common]-b[common], axis=1))) if common.any() else None,
            'common': int(common.sum())}


def run(out):
    if hashlib.sha256((ROOT/'sparse-protocol.json').read_bytes()).hexdigest() != SPARSE_PROTOCOL_SHA:
        raise RuntimeError('Frozen sparse protocol changed')
    out.mkdir(parents=True, exist_ok=True)
    parity = []
    for scene in dense.PROTOCOL['scenes']:
        observed, ids, allowed = make_inputs(scene)
        md, rd, models_d = dense.fit_select(observed, ids, allowed)
        ms, rs, models_s = sparse.fit_select(observed, ids, allowed)
        entries = []
        for name in dense.CANDIDATES:
            a, b = models_d.get(name), models_s.get(name)
            record = {'candidate': name,
                'full_status_same': rd['candidates'][name]['full_fit_status'] == rs['candidates'][name]['full_fit_status'],
                'max_cv_loss_difference': float(np.max(np.abs(np.asarray(rd['candidates'][name]['fold_loss'])-np.asarray(rs['candidates'][name]['fold_loss']))))}
            if a is not None and b is not None:
                record['max_field_difference_mm'] = float(np.max(np.linalg.norm(a.field_source-b.field_source, axis=1)))
                query = np.r_[observed, observed[a.triangles].mean(1)]
                forward_a, forward_b = a.forward(query), b.forward(query)
                record['forward_parity'] = coordinate_parity(forward_a, forward_b)
                valid = np.isfinite(forward_a).all(1) & np.isfinite(forward_b).all(1)
                if valid.any():
                    record['inverse_parity'] = coordinate_parity(a.inverse(forward_a[valid]), b.inverse(forward_a[valid]))
            entries.append(record)
        parity.append({'scene': scene['name'], 'selected_dense': rd['selected_candidate'],
            'selected_sparse': rs['selected_candidate'], 'selected_same': rd['selected_candidate'] == rs['selected_candidate'],
            'dense_seconds': rd['selection_seconds_cpu'], 'sparse_seconds': rs['selection_seconds_cpu'],
            'candidates': entries})
        print(json.dumps({'parity_scene': scene['name'], 'selected_same': parity[-1]['selected_same']}), flush=True)
    inventory = []
    bank = WORKSPACE/'ecg-torch-renderer/build/m3-comparison/cases'
    for path in sorted(bank.glob('*/our-M2/observed-grid-geometry.npz')):
        with np.load(path) as z:
            ids, observed = z['ids'], z['xy']
            extent = ids.max(0)-ids.min(0)+1 if len(ids) else np.array([0,0])
            nodes = int(np.prod(extent))
            inventory.append({'case': path.parent.parent.name, 'observations': len(ids),
                'complete_rectangular_shape': extent.astype(int).tolist(),
                'complete_rectangular_nodes': nodes, 'allowed_cells': len(z['cells']),
                'dense_under_hard_limit': nodes <= 2000,
                'sparse_under_hard_limit': nodes <= 20000,
                'dense_penalty_bytes_estimate': int(8*nodes*nodes),
                'dense_stencil_bytes_upper_estimate': int(24*nodes*nodes),
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'executed_full_selection': False})
    costs = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for nr,nc in [[12,16],[18,28],[35,53],[69,105]]:
        for implementation in ('dense','sparse'):
            if implementation == 'dense' and nr*nc > 2000:
                costs.append({'implementation': implementation, 'shape': [nr,nc], 'nodes': nr*nc,
                    'skipped': 'over frozen 2000-node dense limit', 'dense_penalty_bytes_estimate': 8*(nr*nc)**2})
                continue
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', implementation, str(nr), str(nc)]
            completed = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
            cost = json.loads(completed.stdout)
            costs.append(cost)
            print(json.dumps({'cost': implementation, 'nodes': nr*nc, 'seconds': cost['median_four_irls_solve_seconds']}), flush=True)
    candidate_entries = [x for case in parity for x in case['candidates']]
    forward_entries = [x['forward_parity'] for x in candidate_entries if 'forward_parity' in x]
    inverse_entries = [x['inverse_parity'] for x in candidate_entries if 'inverse_parity' in x]
    summary = {
        'selected_same_all': all(case['selected_same'] for case in parity),
        'full_fit_status_same_all': all(x['full_status_same'] for x in candidate_entries),
        'max_cv_loss_difference': max(x['max_cv_loss_difference'] for x in candidate_entries),
        'max_field_difference_mm': max((x['max_field_difference_mm'] for x in candidate_entries if 'max_field_difference_mm' in x), default=0.),
        'forward_finite_mask_differences': sum(x['finite_mask_difference'] for x in forward_entries),
        'inverse_finite_mask_differences': sum(x['finite_mask_difference'] for x in inverse_entries),
        'max_forward_difference_mm': max((x['max_difference'] for x in forward_entries if x['max_difference'] is not None), default=0.),
        'max_inverse_difference_px': max((x['max_difference'] for x in inverse_entries if x['max_difference'] is not None), default=0.)}
    summary['parity_pass'] = (summary['selected_same_all'] and summary['full_fit_status_same_all']
        and summary['max_cv_loss_difference'] < 1e-8 and summary['max_field_difference_mm'] < 1e-6
        and summary['forward_finite_mask_differences'] == 0 and summary['inverse_finite_mask_differences'] == 0
        and summary['max_forward_difference_mm'] < 1e-6 and summary['max_inverse_difference_px'] < 1e-5)
    result = {'protocol_sha256': SPARSE_PROTOCOL_SHA,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'geometry_selector.py', ROOT/'geometry_selector_sparse.py', ROOT/'sparse-protocol.json']},
        'summary': summary, 'parity_cases': parity, 'costs': costs, 'frozen_real_input_inventory': inventory,
        'no_full_real_selection_or_physical_benchmark': True,
        'limitation': 'Sparse implementation only improves stencil storage and field solve. Full selector still refits many candidates and performs a conservative all-triangle overlap guard with quadratic worst-case cost.',
        'complete': True}
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'complete': True, 'summary': summary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'sparse-results')
    parser.add_argument('--worker', nargs=3)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    else:
        run(args.out)
