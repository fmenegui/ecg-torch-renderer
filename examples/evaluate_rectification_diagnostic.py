"""Separate exact geometry, anonymous gold points and frozen network detections.

Physical errors use a discrete grid-origin/axis alignment, with no fitted scale.
Dense material coordinates come from generator fields independent of this mesh.
They are never passed to either point detector or image-only grid organizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.cell_mesh import forward_cell_map
from ecg_gpu.digitization.cycle_lattice import organize_cycle_lattice
from ecg_gpu.digitization.denoising_grid import DenoisingGridNet
from ecg_gpu.digitization.observation import remove_periodic_banding
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.local_cells import locally_supported_cells
from ecg_gpu.digitization.support import cell_support_summary
from ecg_gpu.grid_evaluation import match_grid_points, summarize_grid_matches
from examples.diverse_grid_study import detect
from examples.legacy_rectification_maps import legacy_forward_supported


def complete_cells(ids):
    present = {tuple(k) for k in np.asarray(ids, int)}
    return [[r, c] for r, c in sorted(present)
            if all(k in present for k in ((r, c+1), (r+1, c+1), (r+1, c)))]


def alignment(predicted_ids, reference_ids):
    """Evaluation-only signed permutation and integer origin, never scale/shear."""
    best = {'consistent': 0, 'matched': len(predicted_ids), 'swap': False,
            'sign': [1, 1], 'origin': [0, 0]}
    for swap in (False, True):
        p = predicted_ids[:, ::-1] if swap else predicted_ids
        for sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if not len(p):
                continue
            offsets, count = np.unique(reference_ids-p*sign, axis=0, return_counts=True)
            index = int(count.argmax())
            if int(count[index]) > best['consistent']:
                best.update(consistent=int(count[index]), swap=swap, sign=list(sign), origin=offsets[index].tolist())
    best['fraction'] = best['consistent']/len(predicted_ids) if len(predicted_ids) else 0.
    return best


def aligned_mm(mapped, transform):
    ids = mapped[:, ::-1]/5
    if transform['swap']:
        ids = ids[:, ::-1]
    ids = ids*np.asarray(transform['sign'])+np.asarray(transform['origin'])
    return ids[:, ::-1]*5


def measure(xy, scores, topology, truth, truth_ids, query, reference, mapper):
    chosen = np.asarray(topology['selected_indices'], int)
    ids = np.asarray(topology['ids'], int).reshape(-1, 2)
    output = np.full(query.shape, np.nan)
    raw = output.copy()
    result = {'status': 'rejected', 'cell_support': cell_support_summary(topology),
              'identity_agreement': 0., 'covered_query_fraction': 0.,
              'coverage_within_0_5mm': 0., 'coverage_within_1mm': 0.,
              'conditional_p95_mm': None, 'conditional_rmse_mm': None,
              'roundtrip_p95_px': None, 'roundtrip_supported_fraction': 0.}
    if len(chosen) < 4 or not len(topology['cells']):
        return result, output
    paired = match_grid_points(xy[chosen], truth, radius=2.)
    transform = alignment(ids[paired['prediction_indices']], truth_ids[paired['reference_indices']])
    result['identity_agreement'] = transform['fraction']
    result['evaluation_origin_alignment'] = transform
    try:
        raw = mapper(xy[chosen], ids[:, ::-1]*5., ids, query, topology['cells'])
        if transform['matched']:
            output = aligned_mm(raw, transform)
        covered = np.isfinite(output).all(1)
        error = np.linalg.norm(output[covered]-reference[covered], axis=1)
        result.update(status='partial' if covered.any() else 'rejected',
                      covered_query_fraction=float(covered.mean()),
                      coverage_within_0_5mm=float((error <= .5).sum()/len(query)),
                      coverage_within_1mm=float((error <= 1.).sum()/len(query)),
                      conditional_p95_mm=float(np.quantile(error, .95)) if len(error) else None,
                      conditional_rmse_mm=float(np.sqrt(np.mean(error**2))) if len(error) else None)
        keep = np.isfinite(raw).all(1)
        if keep.any():
            reverse = mapper(ids[:, ::-1]*5., xy[chosen], ids, raw[keep], topology['cells'])
            valid = np.isfinite(reverse).all(1)
            roundtrip = np.linalg.norm(reverse[valid]-query[keep][valid], axis=1)
            result['roundtrip_p95_px'] = float(np.quantile(roundtrip, .95)) if len(roundtrip) else None
            result['roundtrip_supported_fraction'] = float(valid.sum()/len(query))
    except (ValueError, RuntimeError) as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    return result, output


def evaluate(root, checkpoint, out, split='validation',
             variants=('legacy', 'cells', 't70', 'clean85', 'clean70', 'anchor50'), selection=None):
    frozen = None
    if split == 'test':
        if selection is None:
            raise ValueError('Reserved assessment requires a frozen validation selection')
        frozen = json.loads(selection.read_text())
        if frozen['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('Frozen checkpoint changed')
        for name, expected in frozen['source_code_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen inference source changed: '+name)
        variants = tuple(dict.fromkeys(('legacy', 'cells', frozen['candidate'])))
    manifest = json.loads((root/'manifest.json').read_text())
    selected = [r for r in manifest if r['split'] == split]
    if not selected:
        raise ValueError('No frozen cases for requested split')
    out.mkdir(parents=True, exist_ok=True)
    cache = root/'detections'
    cache.mkdir(exist_ok=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = DenoisingGridNet(width=state['width']).eval()
    model.load_state_dict(state['state_dict'])
    records = []
    output_file = out/(split+'.json')
    report = None
    if output_file.exists():
        report = json.loads(output_file.read_text())
        records = report['cases']
    by_key = {r['key']: r for r in records}
    arms_by_variant = {v: ('oracle', 'anonymous', 'network') if v in ('legacy', 'cells')
                       else ('network',) for v in variants}
    completed = {r['key'] for r in records if all(v+'-'+a in r['results']
                 for v in variants for a in arms_by_variant[v])}
    for row in selected:
        if row['key'] in completed:
            continue
        with np.load(root/row['file']) as data:
            rgb, truth, truth_ids = data['image'], data['xy'], data['ids']
            valid, material = data['valid'], data['reference_mm']
        p = cache/(row['key']+'.npz')
        if p.exists():
            with np.load(p) as detection:
                xy, scores = detection['xy'], detection['scores']
        else:
            xy, scores, _ = detect(model, rgb, device='cpu', threshold=float(state['threshold']))
            np.savez_compressed(p, xy=xy, scores=scores)
        point_metric = summarize_grid_matches([match_grid_points(xy, truth, radius=2.)])
        extra = any(v not in ('legacy', 'cells', 'cycle') for v in variants)
        low_xy, low_scores, clean_xy, clean_scores = xy, scores, xy, scores
        preprocessing = {'applied': False}
        if extra:
            low_file = cache/(row['key']+'-low.npz')
            if low_file.exists():
                with np.load(low_file) as detection:
                    low_xy, low_scores = detection['xy'], detection['scores']
            else:
                low_xy, low_scores, _ = detect(model, rgb, device='cpu', threshold=.3)
                np.savez_compressed(low_file, xy=low_xy, scores=low_scores)
            clean_rgb, preprocessing = remove_periodic_banding(rgb, xy)
            clean_xy, clean_scores = low_xy, low_scores
            if preprocessing['applied']:
                clean_file = cache/(row['key']+'-clean-low.npz')
                if clean_file.exists():
                    with np.load(clean_file) as detection:
                        clean_xy, clean_scores = detection['xy'], detection['scores']
                else:
                    clean_xy, clean_scores, _ = detect(model, clean_rgb, device='cpu', threshold=.3)
                    np.savez_compressed(clean_file, xy=clean_xy, scores=clean_scores)
        low, high = truth_ids[:, ::-1].min(0)*5., truth_ids[:, ::-1].max(0)*5.
        domain = ((valid >= .99) & np.isfinite(material).all(-1)
                  & (material >= low).all(-1) & (material <= high).all(-1))
        pixels = np.flatnonzero(domain)
        if not len(pixels):
            raise ValueError('No visible gold material domain')
        rng = np.random.default_rng(int(row['seed'])+1907)
        selected_pixels = rng.choice(pixels, size=min(4096, len(pixels)), replace=False)
        y, x = np.unravel_index(selected_pixels, domain.shape)
        query, reference = np.column_stack((x, y)).astype(float), material[y, x].astype(float)
        permutation = rng.permutation(len(truth))
        oracle = {'selected_indices': list(range(len(truth))), 'ids': truth_ids.tolist(),
                  'cells': complete_cells(truth_ids)}
        arms = {'oracle': (truth, np.ones(len(truth)), oracle),
                'anonymous': (truth[permutation], np.ones(len(truth)), None),
                'network': (xy, scores, None)}
        case = by_key.get(row['key'], {k: row[k] for k in ('key', 'group_id', 'split', 'backend', 'layout')})
        case.update(points=point_metric, evaluation_queries=len(query), visible_grid_interior_pixels=len(pixels),
                    domain='visible material pixels within true visible major-node extent')
        case.setdefault('results', {})
        case['observation_preprocessing'] = preprocessing
        case.setdefault('variant_inference', {})
        topologies = {}
        for variant in variants:
            mapper = legacy_forward_supported if variant == 'legacy' else forward_cell_map
            organizer = organize_cycle_lattice if variant in ('cycle', 'anchor50') else organize_lattice
            for arm, (points, confidence, known) in arms.items():
                if arm not in arms_by_variant[variant]:
                    continue
                result_key = variant+'-'+arm
                if result_key in case['results']:
                    continue
                if variant in ('t70', 'clean70', 'clean85', 'anchor50'):
                    source_xy, source_scores = (low_xy, low_scores) if variant == 't70' else (clean_xy, clean_scores)
                    threshold = .85 if variant == 'clean85' else .5 if variant == 'anchor50' else .7
                    points, confidence = source_xy[source_scores >= threshold], source_scores[source_scores >= threshold]
                    case['variant_inference'][variant] = {'threshold': threshold,
                        'background_banding_removed': variant != 't70' and preprocessing['applied'],
                        'points': summarize_grid_matches([match_grid_points(points, truth, radius=2.)])}
                topology = known
                if topology is None:
                    topokey = (organizer.__name__, arm, variant)
                    if topokey not in topologies:
                        if variant == 'anchor50':
                            high = clean_scores >= .85
                            hint = organize_lattice(clean_xy[high], clean_scores[high])
                            if hint.get('spacing_px'):
                                native = organizer(points, confidence, geometry_hint=hint)
                            else:
                                points, confidence = xy, scores
                                native = organize_lattice(points, confidence)
                                case['variant_inference'][variant]['fallback'] = 'original pilot, unavailable geometry hint'
                        else:
                            native = organizer(points, confidence)
                        topologies[topokey] = locally_supported_cells(points, native)
                    topology = topologies[topokey]
                measured, prediction = measure(points, confidence, topology, truth, truth_ids, query, reference, mapper)
                case['results'][variant+'-'+arm] = measured
                folder = out/'query-maps'
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/(row['key']+'-'+variant+'-'+arm+'.npz'),
                                    query=query, reference=reference, predicted=prediction)
                if arm == 'network':
                    folder = out/'topologies'
                    folder.mkdir(exist_ok=True)
                    np.savez_compressed(folder/(row['key']+'-'+variant+'.npz'), xy=points, scores=confidence,
                        selected=np.asarray(topology['selected_indices']), ids=np.asarray(topology['ids']).reshape(-1, 2),
                        cells=np.asarray(topology['cells']).reshape(-1, 2))
        by_key[row['key']] = case
        records = list(by_key.values())
        report = {'schema': 'three-arm-rectification-diagnostic-v1', 'split': split, 'cases': records,
                  'patients': len({r['group_id'] for r in records}), 'checkpoint_threshold': state['threshold'],
                  'evaluation_alignment': 'signed axis permutation and integer origin only, no fitted scale',
                  'ground_truth_passed_to_network': False, 'ground_truth_passed_to_organizer': False,
                  'variant_arms': arms_by_variant,
                  'query_generation': 'up to 4096 independent observed pixels with dense material truth',
                  'observation': 'new CPU protocol replay, not original CUDA bank bytes',
                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                  'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest() if frozen else None}
        temporary = output_file.with_suffix('.json.part')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output_file)
        print(json.dumps({'key': row['key'], 'completed': len(records), 'total': len(selected),
                          'network_good_coverage': {v: round(case['results'][v+'-network']['coverage_within_0_5mm'], 4)
                                                    for v in variants}}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'checkpoint', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(**vars(args))
