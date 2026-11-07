"""Replay frozen ECG geometry and bounded oracle interventions for development.

This is a diagnostic experiment, not a new automatic rectifier. E2 explicitly
uses gold association AND removes unmatched detections, so it is not ID-only.
E2-control uses the exact same matched coordinates with inferred identities.
Counterfactual cell maps return partial coordinates, never a claim of rectified
RGB. All scheduled cases, empty maps and exceptions remain in the denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.cell_mesh import CellMap, cell_triangles
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.local_cells import locally_supported_cells
from examples.rectification_comparison_metrics import (
    aggregate_patient_backend, paired_patient_bootstrap, score_predictions,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def associate(xy, truth, tolerance=2.):
    """Deterministic closest-first unique matches, with every exclusion counted."""
    if not len(xy) or not len(truth):
        return np.empty(0, int), np.empty(0, int)
    distances, nearest = cKDTree(truth).query(xy)
    used, pred, ref = set(), [], []
    for i in np.argsort(distances, kind='stable'):
        j = int(nearest[i])
        if distances[i] <= tolerance and j not in used:
            used.add(j)
            pred.append(int(i))
            ref.append(j)
    return np.asarray(pred, int), np.asarray(ref, int)


def inferred(xy, scores):
    topology = locally_supported_cells(xy, organize_lattice(xy, scores))
    chosen = np.asarray(topology['selected_indices'], int)
    ids = np.asarray(topology['ids'], int).reshape(-1, 2)
    return xy[chosen], ids, topology['cells'], {
        'input_points': len(xy), 'retained_points': len(chosen),
        'components': topology.get('components'),
        'cells': len(topology['cells']), 'status': topology.get('status'),
    }


def predictions_from_cells(xy, ids, queries, cells=None):
    canonical = np.asarray(ids)[:, ::-1]*5.
    triangles = cell_triangles(xy, canonical, ids, cells)
    mapping = CellMap(xy, canonical, triangles)
    prediction = {}
    for domain, q in queries.items():
        coords = mapping(q['source'])
        prediction[domain] = {'coords': coords,
            'claimed_eligible': np.isfinite(coords).all(1),
            'native_footprint': np.zeros(len(coords), bool)}
    return prediction, {'triangles': len(triangles),
        'scope': 'partial coordinates only, RGB not rectified'}


def archived_predictions(folder):
    with np.load(folder/'query-predictions.npz', allow_pickle=False) as a:
        return {d: {'coords': a[d+'_coords'].copy(),
                    'claimed_eligible': a[d+'_claimed_eligible'].copy(),
                    'native_footprint': a[d+'_native_footprint'].copy()}
                for d in ('anchors', 'grid', 'paper')}


def run(bank, comparison, out, *, split='test', fresh_detector=None, limit=None):
    schedule = json.loads((comparison/'schedule.json').read_text())
    selected = [r for r in schedule['cases'] if r['split'] == split]
    if limit is not None:
        selected = selected[:limit]
    methods = ['E0-native-M2', 'E1-orientation-oracle', 'E0-native-cell-map',
               'E2-matched-inferred-IDs', 'E2-oracle-association-IDs',
               'E3-perfect-points-inferred-IDs', 'E4-perfect-points-perfect-IDs']
    protocol = {'schema': 'ECG-stage-diagnostic-v1', 'date': '2026-10-26',
        'development_only': True, 'cases': [r['key'] for r in selected],
        'methods': methods, 'matching_radius_source_px': 2.,
        'E2_scope': 'Gold nearest unique matches plus false-positive exclusion. Compare matched-inferred-IDs to isolate identity on the same retained observations.',
        'E3_E4_scope': 'Identical perfect observations, inferred versus gold IDs. No RGB success claim.',
        'E1_scope': 'Gold D4 orientation diagnostic, never automatic performance.',
        'scoring': 'Frozen M3 queries, native physical scale, anchor-only translation, 0.5 mm.',
        'bootstrap': '20000 draws, patient/backend balanced, descriptive development intervals.',
        'reference_M3_lock_sha256': digest(comparison/'lock.json'),
        'schedule_sha256': digest(comparison/'schedule.json'),
        'code_sha256': digest(__file__), 'new_training': False,
        'signal_experiments': 'Separate conditional study, never inferred from geometry scores.'}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Existing diagnostic protocol differs')
    else:
        write_json(out/'protocol.json', protocol)
    detector = None
    if fresh_detector:
        import cv2
        import torch
        from examples.compare_rectification_methods import OurDetector
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        detector = OurDetector(fresh_detector)
    all_rows, details = [], []
    for index, case in enumerate(selected):
        start = time.perf_counter()
        key = case['key']
        if digest(bank/case['file']) != case['sha256']:
            raise ValueError('Capture differs: '+key)
        if digest(comparison/case['query_file']) != case['query_archive_sha256']:
            raise ValueError('Query archive differs: '+key)
        with np.load(bank/case['file'], allow_pickle=False) as a:
            image = a['image'].copy()
            truth_xy, truth_ids = a['xy'].copy(), a['ids'].copy()
        with np.load(comparison/case['query_file'], allow_pickle=False) as a:
            queries = {d: {'source': a[d+'_source'].copy(),
                           'reference': a[d+'_reference'].copy()}
                       for d in ('anchors', 'grid', 'paper')}
        case_folder = comparison/'cases'/key
        with np.load(case_folder/'fresh-observations.npz', allow_pickle=False) as a:
            xy, scores = a['xy'].copy(), a['scores'].copy()
        fresh = None
        if detector:
            fresh_dir = out/'fresh-detector'/key
            fresh_dir.mkdir(parents=True, exist_ok=True)
            _, _, _, fresh = detector.infer(image, fresh_dir, case_folder/'fresh-observations.npz')
            write_json(fresh_dir/'receipt.json', fresh)
        pidx, tidx = associate(xy, truth_xy)
        preparations = {}
        for name, obs, confidence in [
                ('E0-native-cell-map', xy, scores),
                ('E2-matched-inferred-IDs', xy[pidx], scores[pidx]),
                ('E3-perfect-points-inferred-IDs', truth_xy, np.ones(len(truth_xy)))]:
            preparations[name] = inferred(obs, confidence)
        preparations['E2-oracle-association-IDs'] = (
            xy[pidx], truth_ids[tidx], None,
            {'input_points': len(xy), 'matched_points': len(pidx),
             'excluded_unmatched_or_duplicate_detections': len(xy)-len(pidx),
             'missing_truth_points': len(truth_xy)-len(pidx), 'gold_used': True})
        preparations['E4-perfect-points-perfect-IDs'] = (
            truth_xy, truth_ids, None, {'input_points': len(truth_xy), 'gold_used': True})
        native_scores = score_predictions(archived_predictions(case_folder/'our-M2'), queries)
        for method in methods:
            info, failed = {}, None
            if method == 'E0-native-M2':
                scored = native_scores['primary']['domains']
            elif method == 'E1-orientation-oracle':
                scored = native_scores['secondary_D4']['domains']
            else:
                obs, identities, cells, info = preparations[method]
                try:
                    prediction, map_info = predictions_from_cells(obs, identities, queries, cells)
                    info = {**info, **map_info}
                except (ValueError, np.linalg.LinAlgError) as exc:
                    failed = type(exc).__name__+': '+str(exc)
                    prediction = {d: {'coords': np.full_like(q['source'], np.nan),
                                     'claimed_eligible': np.zeros(len(q['source']), bool),
                                     'native_footprint': np.zeros(len(q['source']), bool)}
                                  for d, q in queries.items()}
                scored = score_predictions(prediction, queries)['primary']['domains']
                target = out/'predictions'/key
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target/(method+'.npz'),
                    **{d+'_'+k: v for d, p in prediction.items() for k, v in p.items()})
            for domain in ('grid', 'paper'):
                all_rows.append({'key': key, 'group_id': case['group_id'],
                    'backend': case['backend'], 'layout': case['layout'],
                    'method': method, 'domain': domain,
                    'g': scored[domain]['g'], 'w': scored[domain]['w'],
                    'c': scored[domain]['c'], 'r': scored[domain]['r'],
                    'rmse_mm': scored[domain]['rmse_mm'],
                    'p95_mm': scored[domain]['p95_mm'], 'failure': failed})
            details.append({'key': key, 'method': method, 'details': info, 'failure': failed})
        write_json(out/'cases.json', all_rows)
        write_json(out/'details.json', details)
        print(json.dumps({'case': key, 'completed': index+1, 'scheduled': len(selected),
                          'seconds': round(time.perf_counter()-start, 3),
                          'fresh_detector_exact': fresh['M2_replay_comparison']['coordinates_exact'] if fresh else None}), flush=True)
    summary = {'protocol': protocol, 'domains': {},
               'fresh_inference_requested': detector is not None}
    for domain in ('grid', 'paper'):
        grouped = {m: [r for r in all_rows if r['method'] == m and r['domain'] == domain] for m in methods}
        summary['domains'][domain] = {'means': {m: aggregate_patient_backend(rows) for m, rows in grouped.items()},
            'paired_vs_native': paired_patient_bootstrap(grouped, 'E0-native-M2'),
            'paired_matched_IDs': paired_patient_bootstrap(
                {m: grouped[m] for m in ('E2-matched-inferred-IDs', 'E2-oracle-association-IDs')},
                'E2-matched-inferred-IDs'),
            'paired_perfect_points': paired_patient_bootstrap(
                {m: grouped[m] for m in ('E3-perfect-points-inferred-IDs', 'E4-perfect-points-perfect-IDs')},
                'E3-perfect-points-inferred-IDs')}
    write_json(out/'summary.json', summary)
    import csv
    with (out/'cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--fresh-detector', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(**vars(args))
