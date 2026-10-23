"""Freeze all input-only decisions, then score against analytical references."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import cv2
import numpy as np
import scipy
from geometry_selector import (ROOT, WORKSPACE, CELL_PATH, CELL_SHA, PROTOCOL, PROTOCOL_SHA,
    CANDIDATES, cells, fit_h, fit_select, project)

OLD_PATH = WORKSPACE/'m3/our-critique/composition_probe.py'
spec = importlib.util.spec_from_file_location('m3_frozen_composition', OLD_PATH)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def forward_reference(physical, scene):
    """Reference analytic image formation, used to make observations and score.

    These are the same development scene families seen in M3, with new seeds.
    This function is never called by the selector and is not a clinical reference.
    """
    u, v = physical[:, 0].copy(), physical[:, 1].copy()
    kind = scene['kind']
    if kind == 'smooth':
        u += scene['amplitudes_mm'][0]*np.sin(2*np.pi*physical[:, 1]/55)
        v += scene['amplitudes_mm'][1]*np.sin(2*np.pi*physical[:, 0]/75)
    elif kind == 'aligned':
        v += scene['slope']*np.abs(physical[:, 0]-scene['crease_u_mm'])
    elif kind == 'oblique':
        u += scene['slope']*np.maximum(physical[:, 1]-.6*physical[:, 0]-scene['line_v_minus_0_6u_mm'], 0)
    elif kind != 'flat':
        raise ValueError(kind)
    den = 1+scene['k']*u
    return np.c_[10*u/den, 10*v/den]


def physical_coordinates(base, scene):
    return base*(scene.get('physical_pitch_mm', 5.)/5.)+np.asarray(scene.get('physical_origin_mm', [0., 0.]))


def make_inputs(scene):
    rr, cc = np.mgrid[:12, :16]
    ids = np.c_[rr.ravel(), cc.ravel()]
    canonical = ids[:, ::-1]*5.
    physical = physical_coordinates(canonical, scene)
    observed = forward_reference(physical, scene)
    random = np.random.default_rng(scene['seed'])
    observed += random.normal(0, scene['noise_px'], observed.shape)
    if scene.get('outlier_fraction'):
        chosen = random.choice(len(ids), round(len(ids)*scene['outlier_fraction']), replace=False)
        observed[chosen] += random.normal(0, scene['outlier_sigma_px'], (len(chosen), 2))
    if scene.get('fault') == 'row-sine-5px':
        observed[:, 0] += 5*np.sin(2*np.pi*ids[:, 0]/11)
    if scene.get('fault') == 'half-shift-13px':
        observed[canonical[:, 0] >= 25, 0] += 13
    keep = np.ones(len(ids), bool)
    if scene.get('gap') == 'rows4to7-cols6to9':
        keep &= ~((ids[:, 0] >= 4) & (ids[:, 0] <= 7) & (ids[:, 1] >= 6) & (ids[:, 1] <= 9))
    elif scene.get('gap') == 'columns0to2':
        keep &= ids[:, 1] > 2
    allowed = np.array([[r, c] for r in range(11) for c in range(15)])
    return observed[keep], ids[keep], allowed


def expected_observed_support(base_queries, scene):
    # Independent rectangle-level support reference for each predeclared gap.
    inside = ((base_queries >= [0, 0]) & (base_queries <= [75, 55])).all(1)
    if scene.get('gap') == 'rows4to7-cols6to9':
        hole = ((base_queries[:, 0] > 25) & (base_queries[:, 0] < 50)
            & (base_queries[:, 1] > 15) & (base_queries[:, 1] < 40))
        inside &= ~hole
    elif scene.get('gap') == 'columns0to2':
        inside &= base_queries[:, 0] >= 15
    return inside


def measure(prediction, truth):
    finite = np.isfinite(prediction).all(1)
    errors = np.linalg.norm(prediction-truth, axis=1)
    e = errors[finite]
    return {
        'queries': len(truth), 'returned_queries': int(finite.sum()),
        'G': float(np.mean(finite & (errors <= .5))),
        'W': float(np.mean(finite & (errors > .5))),
        'C': float(finite.mean()),
        'R': float(np.mean(e > .5)) if len(e) else None,
        'rmse_mm': float(np.sqrt(np.mean(e**2))) if len(e) else None,
        'p95_mm': float(np.quantile(e, .95)) if len(e) else None,
        'p99_mm': float(np.quantile(e, .99)) if len(e) else None,
        'max_mm': float(e.max()) if len(e) else None,
    }, finite, errors


def barycentric(point, source, target):
    a = np.stack([source[0]-source[2], source[1]-source[2]], axis=1)
    first = np.linalg.solve(a, point-source[2])
    return np.r_[first, 1-first.sum()]@target


def seam_measure(model):
    if model is None:
        return None
    if hasattr(model, 'field_source'):
        if not model.residual:
            return {'shared_edges': 0, 'forward_seam_max_mm': 0., 'inverse_seam_max_px': 0.}
        source, target = model.field_source, model.canonical
    else:
        return old.seams(model)
    adjacency = {}
    for i, t in enumerate(model.triangles):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            adjacency.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    forward, inverse = [], []
    for edge, adjacent in adjacency.items():
        if len(adjacent) != 2:
            continue
        p = source[list(edge)].mean(0)
        q = target[list(edge)].mean(0)
        answers, backs = [], []
        for i in adjacent:
            tri = model.triangles[i]
            answers.append(barycentric(p, source[tri], target[tri]))
            z = barycentric(q, target[tri], source[tri])
            backs.append(project(z[None], model.h_inverse)[0])
        forward.append(float(np.linalg.norm(answers[0]-answers[1])))
        inverse.append(float(np.linalg.norm(backs[0]-backs[1])))
    return {'shared_edges': len(forward), 'forward_seam_max_mm': max(forward, default=0.),
            'inverse_seam_max_px': max(inverse, default=0.)}


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    (out/'arrays').mkdir(exist_ok=True)
    source_hashes = {str(path.relative_to(WORKSPACE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), ROOT/'geometry_selector.py', ROOT/'protocol.json', OLD_PATH, CELL_PATH]}
    fits = []
    # This phase cannot evaluate physical query error because those queries do not
    # exist yet. The selector accepts only observed source points and integer IDs.
    for scene in PROTOCOL['scenes']:
        observed, ids, allowed = make_inputs(scene)
        selected, report, candidates = fit_select(observed, ids, allowed)
        fits.append((scene, observed, ids, allowed, selected, report, candidates))
        print(json.dumps({'fit': scene['name'], 'selected': report['selected_candidate'],
                          'status': report['selected_geometry_status'], 'seconds': report['selection_seconds_cpu']}), flush=True)
    decision_seal = {'protocol_sha256': PROTOCOL_SHA, 'source_sha256': source_hashes,
        'selection_used_query_reference': False,
        'decisions': [{'scene': scene['name'], 'report': report}
            for scene, _, _, _, _, report, _ in fits]}
    seal_path = out/'decisions-before-physical-scoring.json'
    seal_path.write_text(json.dumps(decision_seal, indent=2, allow_nan=False)+'\n')
    seal_hash = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    random = np.random.default_rng(PROTOCOL['queries']['seed'])
    base_queries = random.uniform([.001, .001], [74.999, 54.999], (4096, 2))
    # Rejection sample a fixed exterior ring, rather than counting interior points
    # in an expanded rectangle as purported extrapolation checks.
    exterior = []
    while len(exterior) < 256:
        p = random.uniform([-10, -10], [85, 65], (512, 2))
        exterior.extend(p[~((p >= 0).all(1) & (p <= [75, 55]).all(1))].tolist())
    exterior = np.asarray(exterior[:256])
    cases, rows = [], []
    for scene, observed, ids, allowed, selected, report, candidates in fits:
        truth = physical_coordinates(base_queries, scene)
        query = forward_reference(truth, scene)
        exterior_physical = physical_coordinates(exterior, scene)
        exterior_source = forward_reference(exterior_physical, scene)
        canonical = ids[:, ::-1]*5.
        mappings = {'selected-input-only': selected}
        rejected = {}
        for name in ('fixed-H-LS', 'current-direct-M2-cells', 'old-unregularized-composition-LS'):
            try:
                if name == 'current-direct-M2-cells':
                    mappings[name] = old.Geometry(cells, observed, canonical, ids, allowed)
                else:
                    h, _ = old.homography(observed, canonical, 'LS')
                    if h is None:
                        raise ValueError('old homography fit rejected')
                    mappings[name] = old.Geometry(cells, observed, canonical, ids, allowed,
                        h=h, residual=name != 'fixed-H-LS')
            except (ValueError, np.linalg.LinAlgError) as error:
                mappings[name] = None
                rejected[name] = str(error)
        saved = {'physical_queries': truth, 'source_queries': query, 'base_queries': base_queries,
            'observed': observed, 'ids': ids, 'canonical': canonical, 'cells': allowed,
            'reference_observed_support': expected_observed_support(base_queries, scene),
            'exterior_physical': exterior_physical, 'exterior_source': exterior_source}
        predictions, results = {}, {}
        for name in PROTOCOL['comparators']:
            model = mappings[name]
            pred = model.forward(query) if model is not None else np.full(truth.shape, np.nan)
            result, finite, error = measure(pred, truth)
            predictions[name] = (finite, error)
            saved[name+'_prediction'] = pred
            result['seams'] = seam_measure(model)
            result['rejected'] = rejected.get(name)
            if name == 'selected-input-only':
                result['selected_candidate'] = report['selected_candidate']
                result['geometry_status'] = report['selected_geometry_status']
                if model is not None:
                    _, statuses = model.forward(query, return_status=True)
                else:
                    statuses = np.full(len(query), 'geometry-rejected', dtype='<U48')
                saved['selected_status'] = statuses
                result['query_status_counts'] = {s: int(np.sum(statuses == s)) for s in np.unique(statuses)}
            ext_pred = model.forward(exterior_source) if model is not None else np.full(exterior_source.shape, np.nan)
            result['exterior_returned'] = int(np.isfinite(ext_pred).all(1).sum())
            saved[name+'_exterior_prediction'] = ext_pred
            support = saved['reference_observed_support']
            result['queries_in_reference_support'] = int(support.sum())
            result['returned_outside_reference_support'] = int((finite & ~support).sum())
            if model is not None and finite.any():
                back = model.inverse(pred[finite])
                good = np.isfinite(back).all(1)
                result['roundtrip_returned'] = int(good.sum())
                result['roundtrip_max_px'] = float(np.linalg.norm(back[good]-query[finite][good], axis=1).max()) if good.any() else None
            else:
                result['roundtrip_returned'] = 0
                result['roundtrip_max_px'] = None
            results[name] = result
        direct_finite, direct_error = predictions['current-direct-M2-cells']
        for name, result in results.items():
            finite, error = predictions[name]
            common = finite & direct_finite
            result['common_with_direct_queries'] = int(common.sum())
            result['common_rmse_mm'] = float(np.sqrt(np.mean(error[common]**2))) if common.any() else None
            result['direct_common_rmse_mm'] = float(np.sqrt(np.mean(direct_error[common]**2))) if common.any() else None
            rows.append({'scene': scene['name'], 'method': name,
                **{k: result[k] for k in ['G','W','C','R','rmse_mm','p95_mm','p99_mm','max_mm','exterior_returned','common_rmse_mm','direct_common_rmse_mm']},
                'selected_candidate': report['selected_candidate'] if name == 'selected-input-only' else ''})
        # Retain all finite candidates for negative-outcome inspection, but never
        # change the already sealed selection using these physical errors.
        candidate_scores, stack = {}, []
        for name in CANDIDATES:
            model = candidates.get(name)
            p = model.forward(query) if model is not None else np.full(truth.shape, np.nan)
            candidate_scores[name] = measure(p, truth)[0]
            saved['candidate_'+name.replace('/','__')+'_prediction'] = p
            stack.append(p)
        stack = np.asarray(stack)
        counts = np.isfinite(stack).all(2).sum(0)
        total = np.nansum(stack, axis=0)
        means = total/np.maximum(counts[:, None], 1)
        distances = np.nansum(np.sum((stack-means[None])**2, axis=2), axis=0)
        uncertainty = np.sqrt(distances/np.maximum(counts, 1))
        uncertainty[counts < 2] = np.nan
        saved['candidate_disagreement_mm_uncalibrated'] = uncertainty
        np.savez_compressed(out/'arrays'/(scene['name']+'.npz'), **saved)
        cases.append({'scene': scene, 'selection': report, 'methods': results,
            'all_candidates_physical_scores_after_selection': candidate_scores,
            'array_file': 'arrays/'+scene['name']+'.npz'})
    result = {'protocol_sha256': PROTOCOL_SHA, 'source_sha256': source_hashes,
        'decision_seal_sha256': seal_hash,
        'decision_seal_unchanged': hashlib.sha256(seal_path.read_bytes()).hexdigest() == seal_hash,
        'physical_gold_used_in_selection': False,
        'query_note': 'Base coordinates use the frozen 75x55mm rectangle. Alias and origin controls transform both anchors and physical queries by their explicitly declared physical pitch/origin, while the selector still assumes supplied 5mm IDs.',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__, 'device': 'CPU', 'openblas_threads': 1},
        'claim': 'Analytical development only. Scene families already seen in M3. No independent clinical holdout, neural detector, RGB delivery or external method ranking.',
        'cases': cases, 'complete': True}
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (out/'measurements.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'complete': True, 'cases': len(cases), 'arms': len(rows), 'out': str(out)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results')
    run(parser.parse_args().out)
