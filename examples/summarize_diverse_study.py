"""Patient-cluster summaries for the diverse rectification study.

Model selection uses validation fields only. Test cases and failures remain in
the denominator. Bootstrap intervals describe test-patient sampling, not random
initialization uncertainty, which is shown through separate seed trials.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.runner import _write_json


def f1(counts):
    tp, fp, fn = np.asarray(counts, float).sum(axis=0)
    denominator = 2*tp+fp+fn
    return float(2*tp/denominator) if denominator else None


def cluster_counts(cases, groups):
    index = {group: i for i, group in enumerate(groups)}
    values = np.zeros((len(groups), 3), float)
    for row in cases:
        values[index[row['group_id']]] += [row['points'][k] for k in ('tp', 'fp', 'fn')]
    return values


def paired_delta(left, right, draws=2000, seed=20261022):
    a = {row['key']: row for row in left}
    b = {row['key']: row for row in right}
    if a.keys() != b.keys() or any(a[k]['group_id'] != b[k]['group_id'] for k in a):
        raise ValueError('Exactly matched cases and patients required')
    groups = sorted({row['group_id'] for row in left})
    x, y = cluster_counts(left, groups), cluster_counts(right, groups)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        chosen = rng.integers(len(groups), size=len(groups))
        fx, fy = f1(x[chosen]), f1(y[chosen])
        if fx is not None and fy is not None:
            differences.append(fy-fx)
    return {'patients': len(groups), 'cases': len(left), 'delta_f1_right_minus_left': f1(y)-f1(x),
            'ci95': np.quantile(differences, [.025, .975]).tolist() if differences else None,
            'bootstrap_draws': draws, 'cluster': 'patient'}


def main(root):
    trials = json.loads((root/'trials.json').read_text())
    if len(trials) != 4 or not all('validation' in row for row in trials):
        raise ValueError('Four frozen trial selections required')
    selected_detector = max(trials, key=lambda row: row['validation']['backend_macro_f1'])
    selected_joint = max([row for row in trials if row['joint']], key=lambda row: row['validation']['backend_macro_f1'])
    summary = {'selection': {'detector': selected_detector['name'], 'joint_restoration': selected_joint['name'],
                            'criterion': 'validation backend-macro F1 only'}, 'trials': [], 'paired_seed_comparisons': []}
    reports = {}
    for trial in trials:
        report = json.loads((root/('test-'+trial['name']+'.json')).read_text())
        reports[trial['name']] = report
        cases = report['cases']
        if not cases:
            raise ValueError('Empty test cohort')
        restoration = [row for row in cases if row['restoration_visible_mse_before'] is not None]
        support = [row.get('local_geometry', {}).get('point_support_fraction', 0.) for row in cases]
        residuals = [row['local_geometry']['p95_mm'] for row in cases if row.get('local_geometry', {}).get('p95_mm') is not None]
        identity = [row['relative_identity'] for row in cases if row.get('relative_identity')]
        summary['trials'].append({'name': trial['name'], 'joint': trial['joint'], 'seed': trial['seed'],
            'by_backend': report['by_backend'], 'held_out_noise': report['held_out_noise'],
            'test_cases': len(cases), 'patients': len({row['group_id'] for row in cases}),
            'native_topology_accepted': sum(row['topology_status'] == 'accepted' for row in cases),
            'partial_local_outputs': sum(row['local_status'] == 'partial' for row in cases),
            'mean_all_case_true_grid_node_support': float(np.mean(support)),
            'conditional_median_case_p95_mm': float(np.median(residuals)) if residuals else None,
            'cases_with_geometric_residual': len(residuals),
            'cases_with_at_least_99pct_matched_node_identity_agreement': sum((row['fraction'] or 0) >= .99 for row in identity),
            'mean_per_case_restoration_mse_before': float(np.mean([r['restoration_visible_mse_before'] for r in restoration])) if restoration else None,
            'mean_per_case_restoration_mse_after': float(np.mean([r['restoration_visible_mse_after'] for r in restoration])) if restoration else None,
            'restoration_head_supervised': trial['joint']})
    for seed in sorted({row['seed'] for row in trials}):
        left = next(row for row in trials if row['seed'] == seed and not row['joint'])
        right = next(row for row in trials if row['seed'] == seed and row['joint'])
        for backend in ('torch-cached', 'offline'):
            x = [r for r in reports[left['name']]['cases'] if r['backend'] == backend]
            y = [r for r in reports[right['name']]['cases'] if r['backend'] == backend]
            summary['paired_seed_comparisons'].append(dict(seed=seed, backend=backend, **paired_delta(x, y)))
    summary['limits'] = ['Patient bootstrap conditional on each trained model',
        'Partial cell support does not certify correct identities or global injectivity',
        'Synthetic grid support is not complete page or waveform coverage',
        'Color reconstruction error is not clinical waveform preservation',
        'The user photograph is development material and was never training data']
    _write_json(root/'summary-diverse.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
