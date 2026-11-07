"""Audit full-frame maps on fixed queries, retaining every geometric failure."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.complete_rectifier import fit_complete_map
from examples.evaluate_rectification_diagnostic import aligned_mm


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATHS = ('ecg_gpu/digitization/complete_rectifier.py',
    'ecg_gpu/digitization/frame_mesh.py', 'ecg_gpu/digitization/cell_mesh.py',
    'examples/rectify_denoising_grid.py', 'examples/rectify_field_image.py',
    'examples/evaluate_complete_rectification.py', 'pyproject.toml')


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FROZEN_PATHS}


def evaluate(bank, previous, out, *, split='validation', selection=None):
    frozen = None
    if split == 'test':
        if selection is None:
            raise ValueError('Retrospective test requires a validation selection frozen first')
        frozen = json.loads(selection.read_text())
        if frozen['source_hashes'] != source_hashes():
            raise ValueError('Sources differ from the validation-frozen configuration')
        manifest_hash = hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()
        if manifest_hash != frozen['input_manifest_sha256']:
            raise ValueError('Input manifest changed after selection')
    out.mkdir(parents=True, exist_ok=True)
    before = json.loads((previous/(split+'.json')).read_text())
    manifest = {r['key']: r for r in json.loads((bank/'manifest.json').read_text())}
    records = []
    for case in before['cases']:
        key = case['key']
        with np.load(bank/manifest[key]['file']) as data:
            shape = data['image'].shape
        with np.load(previous/'topologies'/(key+'-clean85.npz')) as d:
            xy, scores, chosen, ids, cells = [d[k] for k in ('xy', 'scores', 'selected', 'ids', 'cells')]
        with np.load(previous/'query-maps'/(key+'-clean85-network.npz')) as data:
            query, reference, baseline_prediction = data['query'], data['reference'], data['predicted']
        started = time.perf_counter()
        mapping, fit = fit_complete_map(xy[chosen], ids[:, ::-1]*5., ids, shape,
                                       scores[chosen], cells)
        # If the complete image map is rejected, retain the original observed-
        # cell coordinate field as a separate measurement, without erasing RGB.
        predicted = baseline_prediction.copy() if mapping is None else np.full(query.shape, np.nan)
        if mapping is not None:
            transform = case['results']['clean85-network'].get('evaluation_origin_alignment')
            if transform and transform['matched']:
                predicted = aligned_mm(mapping.forward(query), transform)
        covered = np.isfinite(predicted).all(1)
        errors = np.linalg.norm(predicted[covered]-reference[covered], axis=1)
        result = {'covered_fraction': float(covered.mean()),
            'good_coverage': float((errors <= .5).sum()/len(query)),
            'wrong_covered_fraction': float((errors > .5).sum()/len(query)),
            'conditional_p95_mm': float(np.quantile(errors, .95)) if len(errors) else None}
        if mapping is not None:
            reverse = mapping.inverse(mapping.forward(query))
            delta = np.linalg.norm(reverse-query, axis=1)
            result['roundtrip_max_px'] = float(np.nanmax(delta))
            evidence = mapping.evidence(query)
            result['evidence_query_fractions'] = {name: float((evidence == value).mean())
                for value, name in ((1, 'extrapolated'), (2, 'interpolated'), (3, 'observed_cells'))}
            result['eligibility'] = {}
            for radius in (.25, .5, .75, 1., 1.25):
                eligible = mapping.eligible_geometry(query, radius=radius)
                all_errors = np.linalg.norm(predicted-reference, axis=1)
                result['eligibility'][str(radius)] = {'covered_fraction': float(eligible.mean()),
                    'good_coverage': float((eligible & (all_errors <= .5)).mean()),
                    'wrong_covered_fraction': float((eligible & (all_errors > .5)).mean())}
        else:
            result['eligibility'] = {str(radius): dict(result) for radius in (.25, .5, .75, 1., 1.25)}
        record = {k: case[k] for k in ('key', 'group_id', 'backend', 'split')}
        old = case['results']['clean85-network']
        record.update(fit=fit, result=result, baseline={'good_coverage': old['coverage_within_0_5mm'],
            'wrong_covered_fraction': old['covered_query_fraction']-old['coverage_within_0_5mm']},
            seconds=time.perf_counter()-started,
            coordinate_mode='complete-map' if mapping is not None else 'previous-observed-cell-field')
        records.append(record)
        folder = out/'query-maps'
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(key+'.npz'), query=query, reference=reference, predicted=predicted)
        if mapping is not None:
            folder = out/'meshes'
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/(key+'.npz'), source=mapping.source, target=mapping.target,
                ids=mapping.ids, triangles=mapping.triangles)
        report = {'split': split, 'cases': records, 'method': 'constrained full-frame cell completion',
            'source_hashes': source_hashes(),
            'fixed_query_domain': True, 'gold_used_for_fit': False,
            'test_previously_inspected': split == 'test', 'new_independent_test': False}
        if frozen is not None:
            report['selected_radius'] = frozen['selected_radius']
        path = out/(split+'.json')
        tmp = path.with_suffix('.json.part')
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(path)
        print(json.dumps({'key': key, 'status': fit['status'],
            'old_good': round(record['baseline']['good_coverage'], 4),
            'new_good': round(result['good_coverage'], 4),
            'new_wrong': round(result['wrong_covered_fraction'], 4)}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'previous', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--selection', type=Path)
    evaluate(**vars(parser.parse_args()))
