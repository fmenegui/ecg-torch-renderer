"""Paired image-only soft-grid refinement study with patient-grouped selection.

Gold material fields enter scoring only. Fixed-ID refinement shares the B0
evaluation gauge. Native-context detection is a separate diagnostic arm.
No waveform, physical scale or affine transform is fitted by the evaluator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.cell_mesh import forward_cell_map
from ecg_gpu.digitization.complete_rectifier import fit_complete_map
from ecg_gpu.digitization.denoising_grid import DenoisingGridNet
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.local_cells import locally_supported_cells
from ecg_gpu.digitization.observation import remove_periodic_banding
from ecg_gpu.digitization.robust_refinement import refine_grid_points
from ecg_gpu.grid_evaluation import match_grid_points, summarize_grid_matches
from examples.diverse_grid_study import detect
from examples.evaluate_rectification_diagnostic import aligned_mm, alignment


ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = (0., .05, .15, .5, 1.)
CHECKPOINT_SHA = '8115e99c3517dc0b99974b9e64744aab9eb961fd0da06f1e84a65a1e45c3dbf8'
FROZEN_PATHS = (
    'examples/evaluate_nextgen_refinement.py',
    'ecg_gpu/digitization/robust_refinement.py',
    'ecg_gpu/digitization/complete_rectifier.py',
    'ecg_gpu/digitization/frame_mesh.py',
    'ecg_gpu/digitization/cell_mesh.py',
    'ecg_gpu/digitization/lattice.py',
    'ecg_gpu/digitization/local_cells.py',
    'ecg_gpu/digitization/observation.py',
    'ecg_gpu/digitization/denoising_grid.py',
    'ecg_gpu/digitization/field_rectifier.py',
    'ecg_gpu/grid_point_learning.py',
    'examples/diverse_grid_study.py',
    'examples/evaluate_rectification_diagnostic.py',
    'ecg_gpu/grid_evaluation.py',
)
SELECTION_RULE = {
    'primary_domain': 'grid', 'error_threshold_mm': .5,
    'eligibility_radius': .75, 'minimum_delta_G': 0.,
    'crease_guard': True,
    'maximum_delta_W': .001, 'maximum_delta_R': .001,
    'maximum_backend_delta_W': .002, 'maximum_backend_delta_R': .002,
    'minimum_backend_delta_G': -.005,
    'tie_break': 'smaller strength',
    'interpretation': 'development operating gates, not confidence bounds',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(ROOT/name) for name in FROZEN_PATHS}


def development_manifest_hash(manifest):
    rows = [r for r in manifest if r['split'] == 'validation']
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def method_key(strength):
    return 'B0' if strength == 0 else 'soft-'+str(strength)


def fixed_queries(data, seed, count=4096):
    material, valid, ids = data['reference_mm'], data['valid'], data['ids']
    physical = (valid >= .99) & np.isfinite(material).all(-1)
    if len(ids):
        low, high = ids[:, ::-1].min(0)*5., ids[:, ::-1].max(0)*5.
        grid = physical & (material >= low).all(-1) & (material <= high).all(-1)
    else:
        grid = np.zeros_like(physical)
    output = {}
    for index, (name, domain) in enumerate((('grid', grid), ('paper', physical))):
        pixels = np.flatnonzero(domain)
        rng = np.random.default_rng(seed+1907+index*7919)
        selected = rng.choice(pixels, size=min(count, len(pixels)), replace=False)
        y, x = np.unravel_index(selected, domain.shape)
        output[name] = {
            'query': np.column_stack((x, y)).astype(float),
            'reference': material[y, x].astype(float),
            'visible_domain_pixels': len(pixels),
            'query_sha256': hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
        }
    return output


def frozen_detection(model, image, checkpoint_sha, cache, key, threshold):
    # Early caches were produced while the frozen detect helper was audited.
    # Verify that exact Git blob on every cache read, not just the model heads.
    helper = (ROOT/'examples/diverse_grid_study.py').read_bytes()
    helper_blob = hashlib.sha1(b'blob '+str(len(helper)).encode()+b'\0'+helper).hexdigest()
    if helper_blob != '8195dab780caa5e526159218b0f7298b422d51c3':
        raise ValueError('The B0 detect helper changed, use a new inference cache')
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/(key+'.npz')
    image_sha = hashlib.sha256(image.tobytes()).hexdigest()
    meta = {'image_array_sha256': image_sha, 'checkpoint_sha256': checkpoint_sha,
            'threshold': threshold, 'profile': 'banding',
            'inference_source_hashes': {name: digest(ROOT/name) for name in (
                'ecg_gpu/digitization/denoising_grid.py',
                'ecg_gpu/digitization/field_rectifier.py',
                'ecg_gpu/grid_point_learning.py',
                'ecg_gpu/digitization/observation.py')},
            'torch_version': torch.__version__}
    if path.exists() and path.with_suffix('.json').exists():
        previous = json.loads(path.with_suffix('.json').read_text())
        if previous['input_contract'] == meta:
            with np.load(path) as loaded:
                previous['verified_detect_helper_git_blob'] = helper_blob
                return loaded['xy'], loaded['scores'], previous
        raise ValueError('Detection cache source or image changed: '+key)
    start = time.perf_counter()
    xy, scores, output = detect(model, image, device='cpu', threshold=threshold)
    del output
    filtered, banding = remove_periodic_banding(image, xy)
    if banding['applied']:
        xy, scores, output = detect(model, filtered, device='cpu', threshold=threshold)
        del output
    metadata = {'input_contract': meta, 'seconds': time.perf_counter()-start,
                'banding': banding, 'cached': False,
                'verified_detect_helper_git_blob': helper_blob}
    np.savez_compressed(path, xy=xy, scores=scores)
    write_json(path.with_suffix('.json'), metadata)
    return xy, scores, metadata


def build_geometry(xy, scores, topology, shape, strength):
    chosen = np.asarray(topology['selected_indices'], int)
    ids = np.asarray(topology['ids'], int).reshape(-1, 2)
    original = xy[chosen]
    started = time.perf_counter()
    refined, report = refine_grid_points(original, ids, scores[chosen], strength=strength,
                                         crease_guard=True)
    used = refined
    failure = None
    cv2.setRNGSeed(1973)
    try:
        mapping, fitted = fit_complete_map(used, ids[:, ::-1]*5., ids, shape,
                                          scores[chosen], topology['cells'])
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        mapping, fitted = None, {'status': 'fit-error', 'reason': str(exc)}
    # Fallback depends only on geometry construction, never scoring truth.
    if strength and mapping is None:
        used = original
        failure = fitted
        cv2.setRNGSeed(1973)
        mapping, fitted = fit_complete_map(used, ids[:, ::-1]*5., ids, shape,
                                          scores[chosen], topology['cells'])
    details = {
        'map_fit': fitted, 'refinement': report,
        'refined_coordinates': bool(strength and failure is None and report['status'] == 'refined'),
        'detection_coordinates_preserved_exactly': bool(np.array_equal(used, original)),
        'map_fit_observations': 'refined points' if not np.array_equal(used, original) else 'original detections',
        'construction_fallback_to_B0': failure is not None,
        'refined_map_rejection': failure, 'geometry_seconds': time.perf_counter()-started,
        'full_map': mapping is not None,
    }
    return mapping, used, ids, details


def project_geometry(mapping, used, ids, topology, query):
    if mapping is None:
        predicted = forward_cell_map(used, ids[:, ::-1]*5., ids, query, topology['cells'])
        eligible = np.isfinite(predicted).all(1)
    else:
        predicted = mapping.forward(query)
        eligible = mapping.eligible_geometry(query, radius=.75, mapped=predicted)
    return predicted, eligible


def evaluate_prediction(predicted, eligible, reference):
    n = len(reference)
    finite = np.isfinite(predicted).all(1)
    accepted = eligible & finite
    error = np.linalg.norm(predicted-reference, axis=1)
    good, wrong = accepted & (error <= .5), accepted & (error > .5)
    values = error[accepted]
    result = {'queries': n, 'g': float(good.mean()) if n else 0.,
              'w': float(wrong.mean()) if n else 0.,
              'c': float(accepted.mean()) if n else 0.,
              'r': float(wrong.sum()/accepted.sum()) if accepted.any() else None,
              'p95_mm': float(np.quantile(values, .95)) if len(values) else None,
              'p99_mm': float(np.quantile(values, .99)) if len(values) else None,
              'rmse_mm': float(np.sqrt(np.mean(values**2))) if len(values) else None,
              'unrestricted_good': float((finite & (error <= .5)).mean()) if n else 0.,
              'unrestricted_wrong': float((finite & (error > .5)).mean()) if n else 0.}
    return result, accepted, error


def aggregate(cases, method, domain):
    by_backend = {}
    for backend in sorted({r['backend'] for r in cases}):
        patient = {}
        for row in cases:
            if row['backend'] == backend:
                patient.setdefault(row['group_id'], []).append(row['methods'][method]['domains'][domain])
        means = [{k: float(np.mean([v[k] for v in values])) for k in ('g', 'w', 'c')}
                 for values in patient.values()]
        value = {k: float(np.mean([p[k] for p in means])) for k in ('g', 'w', 'c')}
        value['r'] = value['w']/value['c'] if value['c'] else None
        by_backend[backend] = dict(value, patients=len(patient))
    overall = {k: float(np.mean([v[k] for v in by_backend.values()])) for k in ('g', 'w', 'c')}
    overall['r'] = overall['w']/overall['c'] if overall['c'] else None
    return {'overall': overall, 'by_backend': by_backend}


def summarize(cases, strengths):
    return {method_key(s): {domain: aggregate(cases, method_key(s), domain)
                           for domain in ('grid', 'paper')} for s in strengths}


def choose(summary):
    baseline = summary['B0']['grid']
    chosen, candidates = 0., {}
    for strength in STRENGTHS:
        key = method_key(strength)
        values = summary[key]['grid']
        base, current = baseline['overall'], values['overall']
        delta = {k: current[k]-base[k] for k in ('g', 'w', 'c')}
        delta['r'] = ((current['r'] or 0)-(base['r'] or 0))
        passed = (delta['g'] >= 0 and delta['w'] <= .001 and delta['r'] <= .001)
        for backend, b in baseline['by_backend'].items():
            a = values['by_backend'][backend]
            passed &= (a['g']-b['g'] >= -.005 and a['w']-b['w'] <= .002
                       and (a['r'] or 0)-(b['r'] or 0) <= .002)
        candidates[key] = {'delta': delta, 'passes_development_gates': bool(passed)}
        if passed and current['g'] > summary[method_key(chosen)]['grid']['overall']['g']+1e-12:
            chosen = strength
    return chosen, candidates


def run(bank, checkpoint, out, split='validation', selection=None):
    out.mkdir(parents=True, exist_ok=True)
    actual_checkpoint = digest(checkpoint)
    if actual_checkpoint != CHECKPOINT_SHA:
        raise ValueError('Checkpoint differs from the previously selected B0')
    manifest = json.loads((bank/'manifest.json').read_text())
    attempts = json.loads((bank/'attempts.json').read_text())
    # Selection is frozen before assessment rows or query fields are opened.
    lock = None
    if split == 'test':
        if selection is None:
            raise ValueError('New-patient assessment needs a frozen development selection')
        lock = json.loads(selection.read_text())
        if lock['source_hashes'] != source_hashes():
            raise ValueError('Inference or evaluator changed after selection')
        if lock['checkpoint_sha256'] != actual_checkpoint or lock['data_protocol_sha256'] != digest(bank/'protocol.json'):
            raise ValueError('Data selection or checkpoint changed')
        if lock['development_manifest_sha256'] != development_manifest_hash(manifest):
            raise ValueError('Development capture manifest changed after selection')
        if lock['validation_sha256'] != digest(selection.parent/'validation.json'):
            raise ValueError('Development scores changed after selection')
        strengths = tuple(sorted({0., float(lock['selected_strength'])}))
    else:
        strengths = STRENGTHS
    selected = [r for r in manifest if r['split'] == split]
    expected = 48 if split == 'validation' else 64
    if len(selected) != expected:
        raise ValueError(f'Study split incomplete: {len(selected)}/{expected} captures')
    failed = [r for r in attempts if r.get('split') == split and r.get('status') == 'failed']
    if failed:
        raise RuntimeError('Rendering failures require explicit denominator resolution: '+json.dumps(failed))
    if not selected:
        raise ValueError('No study images for split')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = DenoisingGridNet(width=state['width']).eval()
    model.load_state_dict(state['state_dict'])
    records = []
    maps = out/'query-maps'
    maps.mkdir(exist_ok=True)
    for row in selected:
        if digest(bank/row['file']) != row['sha256']:
            raise ValueError('Bank archive changed: '+row['key'])
        with np.load(bank/row['file']) as archive:
            data = {k: archive[k] for k in ('image', 'xy', 'ids', 'valid', 'reference_mm')}
        xy, scores, detected = frozen_detection(model, data['image'], actual_checkpoint,
            bank/'detections-B0', row['key'], float(state['threshold']))
        topology = locally_supported_cells(xy, organize_lattice(xy, scores))
        chosen = np.asarray(topology['selected_indices'], int)
        ids = np.asarray(topology['ids'], int).reshape(-1, 2)
        pairs = match_grid_points(xy[chosen], data['xy'], radius=2.)
        gauge = alignment(ids[pairs['prediction_indices']], data['ids'][pairs['reference_indices']])
        queries = fixed_queries(data, int(row['seed']))
        record = {k: row[k] for k in ('key', 'group_id', 'backend', 'split', 'layout')}
        record.update(detection=detected, points=summarize_grid_matches([
            match_grid_points(xy, data['xy'], radius=2.)]),
            evaluation_gauge=gauge,
            topology={k: v for k, v in topology.items() if k not in ('selected_indices', 'ids', 'cells')},
            topology_points=len(chosen), topology_cells=len(topology['cells']),
            query_contract={n: {k: v for k, v in q.items() if k not in ('query', 'reference')}
                            for n, q in queries.items()}, methods={})
        saved = {n+'_'+k: q[k] for n, q in queries.items() for k in ('query', 'reference')}
        saved.update(detected_xy=xy, detected_scores=scores, selected=chosen,
                     lattice_ids=ids, cells=np.asarray(topology['cells'], int).reshape(-1, 2))
        base_accepted, base_error = {}, {}
        for strength in strengths:
            key = method_key(strength)
            mapping, used, identities, details = build_geometry(xy, scores, topology, data['image'].shape, strength)
            details['domains'] = {}
            for domain, q in queries.items():
                predicted, eligible = project_geometry(mapping, used, identities, topology, q['query'])
                if gauge['matched']:
                    predicted = aligned_mm(predicted, gauge)
                else:
                    predicted[:] = np.nan
                value, accepted, error = evaluate_prediction(predicted, eligible, q['reference'])
                if strength == 0:
                    base_accepted[domain], base_error[domain] = accepted, error
                common = base_accepted[domain] & accepted
                value['common_with_B0_queries'] = int(common.sum())
                value['common_rmse_mm'] = float(np.sqrt(np.mean(error[common]**2))) if common.any() else None
                value['B0_common_rmse_mm'] = float(np.sqrt(np.mean(base_error[domain][common]**2))) if common.any() else None
                details['domains'][domain] = value
                saved[key+'_'+domain+'_predicted'] = predicted
                saved[key+'_'+domain+'_eligible'] = eligible
            record['methods'][key] = details
        np.savez_compressed(maps/(row['key']+'.npz'), **saved)
        records.append(record)
        report = {'split': split, 'cases': records, 'strengths': strengths,
            'data_protocol_sha256': digest(bank/'protocol.json'), 'checkpoint_sha256': actual_checkpoint,
            'source_hashes': source_hashes(), 'gold_used_for_inference': False,
            'evaluation_gauge': 'B0 point-based signed axes and integer origin, fixed across strengths',
            'clinical_or_external_real_image_validation': False,
            'selection_rule': SELECTION_RULE, 'new_neural_training': False}
        write_json(out/(split+'.json'), report)
        print(json.dumps({'key': row['key'], 'done': len(records), 'total': len(selected),
            'B0_G': record['methods']['B0']['domains']['grid']['g']}), flush=True)
    report['summary'] = summarize(records, strengths)
    write_json(out/(split+'.json'), report)
    if split == 'validation':
        strength, candidates = choose(report['summary'])
        write_json(out/'selection.json', {'selected_strength': strength, 'candidates': candidates,
            'source_hashes': source_hashes(), 'checkpoint_sha256': actual_checkpoint,
            'data_protocol_sha256': digest(bank/'protocol.json'), 'selection_rule': SELECTION_RULE,
            'development_manifest_sha256': development_manifest_hash(manifest),
            'validation_sha256': digest(out/'validation.json'),
            'test_opened_by_selection': False})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('bank', 'checkpoint', 'out'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run(**vars(args))
