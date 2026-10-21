"""Paired frozen-model geometry ablation, with all failed cases retained.

This tests whether disabling upstream nonlinear dewarping explains the folded
paper failure. It does not fit geometry or select settings from ground truth.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.runner import _write_json
from examples.digitization_npj_profiles import run


CASES = ('six-rows', 'portrait', 'offline-flat', 'offline-fold')


def paired_summary(enabled, disabled):
    """Only form quality deltas when both paired calls returned valid scores."""
    if enabled['input_manifest_sha256'] != disabled['input_manifest_sha256']:
        raise ValueError('Paired input manifests must match')
    for key in ('upstream_commit', 'device', 'precision', 'torch'):
        if enabled[key] != disabled[key]:
            raise ValueError(f'Paired {key} must match')
    left = {c['name']: c for c in enabled['cases']}
    right = {c['name']: c for c in disabled['cases']}
    if left.keys() != right.keys():
        raise ValueError('Paired case names must match')
    records = []
    for name, a in left.items():
        b = right[name]
        if a['image_sha256'] != b['image_sha256']:
            raise ValueError('Paired image hashes must match')
        record = {'name': name}
        for label, case in (('enabled', a), ('disabled', b)):
            evaluation = case.get('evaluation', {})
            record[label] = {key: case.get(key) for key in
                             ('status', 'error', 'predicted_layout', 'layout_matches_expected')}
            record[label].update({key: evaluation.get(key) for key in
                ('mean_observed_lead_snr_db', 'printed_time_coverage',
                 'raw_overlap_rmse_mv', 'unsupported_output_fraction')})
        valid_pair = all(c.get('status') == 'returned' for c in (a, b))
        x = record['enabled']['mean_observed_lead_snr_db']
        y = record['disabled']['mean_observed_lead_snr_db']
        record['disabled_minus_enabled_snr_db'] = y-x if valid_pair and x is not None and y is not None else None
        records.append(record)
    return {'schema': 'ecg-dewarping-ablation-v1', 'cases': records,
            'training_performed': False, 'automatic_setting_selection': False,
            'limits': ['One artificial signal, four diagnostic cases',
                       'Conditional SNR can improve while coverage deteriorates',
                       'No real-image or clinical validation']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    torch.set_num_threads(2)
    reports = [run(args.inputs, args.out/label, args.source, device=args.device,
                   arms=('configured',), dewarping=value, case_names=CASES)
               for label, value in (('enabled', True), ('disabled', False))]
    _write_json(args.out/'paired-summary.json', paired_summary(*reports))
