"""Select on validation, then summarize retrospective fixed-query assessment."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.complete_rectifier import CompleteMap
from examples.evaluate_complete_rectification import source_hashes


def average(cases, getter):
    backends = sorted({c['backend'] for c in cases})
    result = {}
    for backend in backends:
        subset = [c for c in cases if c['backend'] == backend]
        patients = sorted({c['group_id'] for c in subset})
        result[backend] = {metric: float(np.mean([
            np.mean([getter(c)[metric] for c in subset if c['group_id'] == patient])
            for patient in patients])) for metric in ('good_coverage', 'wrong_covered_fraction')}
        result[backend]['covered_fraction'] = sum(result[backend].values())
    result['backend_balanced'] = {metric: float(np.mean([result[b][metric] for b in backends]))
        for metric in ('good_coverage', 'wrong_covered_fraction', 'covered_fraction')}
    return result


def select(root, bank):
    validation = root/'validation.json'
    cases = json.loads(validation.read_text())['cases']
    if len(cases) != 48:
        raise ValueError('All 48 validation cases are required')
    baseline = average(cases, lambda c: c['baseline'])
    candidates = []
    for radius in (.25, .5, .75, 1., 1.25):
        metrics = average(cases, lambda c: c['result']['eligibility'][str(radius)])
        acceptable = all(metrics[b]['wrong_covered_fraction'] <= baseline[b]['wrong_covered_fraction']
            + (.005 if b == 'backend_balanced' else .01) for b in metrics)
        candidates.append({'radius': radius, 'metrics': metrics, 'eligible': acceptable})
    chosen = max((r for r in candidates if r['eligible']),
                 key=lambda r: (r['metrics']['backend_balanced']['good_coverage'], -r['radius']))
    if inspect.signature(CompleteMap.eligible_geometry).parameters['radius'].default != chosen['radius']:
        raise ValueError('CLI default radius must match validation selection before freezing')
    digest = hashlib.sha256()
    for path in sorted((bank.parent/'diagnostic-evaluation/topologies').glob('*-clean85.npz')):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    result = {'selected_radius': chosen['radius'], 'candidates': candidates, 'baseline': baseline,
        'selection_rule': 'Maximize correct coverage, wrong coverage <= baseline +0.005 overall and +0.01 per backend',
        'engineering_error_threshold_mm': .5, 'clinical_acceptance_claim': False,
        'source_hashes': source_hashes(), 'validation_sha256': hashlib.sha256(validation.read_bytes()).hexdigest(),
        'input_manifest_sha256': hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),
        'cached_topologies_aggregate_sha256': digest.hexdigest(),
        'checkpoint_sha256': '8115e99c3517dc0b99974b9e64744aab9eb961fd0da06f1e84a65a1e45c3dbf8',
        'detector_inference': 'Reused frozen cached detections, no network re-execution or training',
        'test_previously_inspected': True, 'new_independent_test': False,
        'fallback': 'Original RGB retained, existing observed-cell coordinate field measured separately'}
    (root/'selection.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def summarize(root):
    selection = json.loads((root/'selection.json').read_text())
    radius = str(selection['selected_radius'])
    summary = {'selected_radius': float(radius), 'engineering_error_threshold_mm': .5,
        'new_independent_test': False, 'physical_scale_fitted_for_evaluation': False,
        'rgb_preservation': 'Full-frame sampled RGB on accepted maps, exact original RGB on rejection',
        'coordinate_fallback': 'Previous partial coordinate field retained independently of visible RGB',
        'splits': {}}
    for split in ('validation', 'test'):
        data = json.loads((root/(split+'.json')).read_text())
        cases = data['cases']
        if len(cases) != 48:
            raise ValueError('All 48 cases must stay in each denominator')
        baseline = average(cases, lambda c: c['baseline'])
        eligible = average(cases, lambda c: c['result']['eligibility'][radius])
        full = average(cases, lambda c: c['result'])
        patients = sorted({c['group_id'] for c in cases})
        delta = np.array([[average([c for c in cases if c['group_id'] == patient],
            lambda c: c['result']['eligibility'][radius])['backend_balanced'][metric]
            - average([c for c in cases if c['group_id'] == patient],
            lambda c: c['baseline'])['backend_balanced'][metric]
            for metric in ('good_coverage', 'wrong_covered_fraction')] for patient in patients])
        rng = np.random.default_rng(1973)
        bootstrap = delta[rng.integers(len(patients), size=(20000, len(patients)))].mean(1)
        intervals = np.quantile(bootstrap, [.025, .975], axis=0)
        by_reason = {}
        for c in cases:
            reason = c['fit']['reason'] or 'complete map accepted'
            by_reason[reason] = by_reason.get(reason, 0)+1
        summary['splits'][split] = {'images': len(cases), 'patients': len(patients),
            'baseline': baseline, 'eligible_coordinates': eligible, 'all_proposed_coordinates': full,
            'full_map_proposals': sum(c['coordinate_mode'] == 'complete-map' for c in cases),
            'original_frame_fallbacks': sum(c['coordinate_mode'] != 'complete-map' for c in cases),
            'reasons': by_reason, 'paired_patient_bootstrap_95_delta': {
                metric: intervals[:, i].tolist() for i, metric in enumerate(('good_coverage', 'wrong_covered_fraction'))},
            'roundtrip_max_px': max(c['result'].get('roundtrip_max_px', 0) for c in cases),
            'case_gains_above_1_percentage_point': sum(c['result']['eligibility'][radius]['good_coverage']
                - c['baseline']['good_coverage'] > .01 for c in cases),
            'case_losses_below_minus_1_percentage_point': sum(c['result']['eligibility'][radius]['good_coverage']
                - c['baseline']['good_coverage'] < -.01 for c in cases)}
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--select', action='store_true')
    args = parser.parse_args()
    result = select(args.root, args.bank) if args.select else summarize(args.root)
    print(json.dumps(result, indent=2))
