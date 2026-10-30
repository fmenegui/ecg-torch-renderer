"""Complete saved inference with the frozen selector, then score separately.

The primary report remains unchanged. Reusing its outputs is verified by
replaying every original selection before applying continuation predictions.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ecg_gpu.digitization.external_evaluation import score_external_signals
from ecg_gpu.digitization.npj_backend import canonical_layout
from ecg_gpu.digitization.selective import GeometryEvidence, SelectionPolicy, SignalProposal, SpecialistEvidence, select_proposal
from examples.summarize_external_digitization import summarize
from reproductions.common import sha256, write_json
from reproductions.hybrid.worker import empty

ARMS = ['npj_native', 'grid_adaptive', 'kaggle_native_eligible', 'selective']


def proposal(root, row, identity):
    p = root / row['prediction']
    if sha256(p) != row['prediction_sha256']:
        raise ValueError('Saved prediction changed')
    with np.load(p, allow_pickle=False) as z:
        values, fs = z['values_mv'].copy(), float(z['fs'])
    metadata = row['backend_metadata']
    return SignalProposal(values, fs, identity, canonical_layout(metadata.get('native_layout')),
                          metadata=metadata), row['status']


def geometry(root, name, identity):
    report = json.loads((root / name / 'geometry.json').read_text())['report']
    if not report or report.get('original_source_id') != identity or report.get('status') != 'rectified':
        return None
    return GeometryEvidence(identity, True, True, True, True, report['outside_observed_hull_fraction'])


def choose(direct, candidate, evidence, specialist, specialist_evidence, policy):
    d, direct_status = direct
    c, candidate_status = candidate
    selected, status = d, direct_status
    decision = {'selected': 'direct', 'reason': 'no_accepted_candidate',
                'policy_sha256': policy.sha256, 'reference_used': False}
    if specialist_evidence is not None:
        selected, decision = select_proposal(d, specialist, specialist_evidence, policy)
        if selected is specialist:
            status = 'returned'
    if selected is d:
        prior = decision
        if evidence is not None and candidate_status == 'returned':
            selected, decision = select_proposal(d, c, evidence, policy)
            status = candidate_status if selected is c else direct_status
            decision['specialist_decision'] = prior
        else:
            decision['geometry_branch_error'] = c.metadata.get('error')
    return selected, status, decision


def build(image_only, primary, repair, specialist_root, out):
    if any(set(c) != {'image_id', 'image', 'image_sha256'} for c in image_only):
        raise ValueError('Selection accepts the image-only manifest')
    policy = SelectionPolicy()
    original = json.loads((primary / 'run/report.json').read_text())
    old = {(r['image_id'], r['arm']): {k: v for k, v in r.items() if k != 'evaluation'} for r in original['cases']}
    phase1 = json.loads((primary / 'run/phase1-decisions.json').read_text())
    if phase1['policy_sha256'] != policy.sha256:
        raise ValueError('Policy changed after primary inference')
    # Independent reconstruction must reproduce every original output first.
    for case in image_only:
        name, identity = case['image_id'], case['image_sha256']
        d = proposal(primary / 'run', old[name, 'npj_native'], identity)
        c = proposal(primary / 'run', old[name, 'grid_adaptive'], identity)
        selected, _, _ = choose(d, c, geometry(primary / 'run', name, identity), None, None, policy)
        with np.load(primary / 'run' / old[name, 'selective']['prediction'], allow_pickle=False) as z:
            np.testing.assert_array_equal(selected.values_mv, z['values_mv'])
    patched = json.loads((repair / 'repair/report.json').read_text())
    replacements = {(r['image_id'], r['arm']): r for r in patched['cases']}
    special = {}
    batch_seconds = 0.
    if specialist_root is not None:
        seal = json.loads((specialist_root / 'phase1-completion-predictions.json').read_text())
        receipt = json.loads((specialist_root / 'receipt.json').read_text())
        if sha256(specialist_root / 'phase1-completion-predictions.json') != receipt['prediction_seal_sha256']:
            raise ValueError('Continuation prediction seal changed')
        special = {r['image_id']: r for r in seal['records']}
        batch_seconds = seal['batch_seconds']
    out.mkdir(parents=True, exist_ok=False)
    records, decisions = [], []
    for case in image_only:
        name, identity = case['image_id'], case['image_sha256']
        data, times, chosen_rows = {}, {}, {}
        repaired = (name, 'npj_native') in replacements
        source = repair / 'repair' if repaired else primary / 'run'
        for arm in ['npj_native', 'grid_adaptive']:
            row = replacements[name, arm] if repaired else old[name, arm]
            data[arm] = proposal(source, row, identity)
            times[arm] = row['inference_wall_seconds']
            chosen_rows[arm] = row
        spec, evidence = None, None
        record = special.get(name)
        if record and record['status'] == 'returned':
            p = specialist_root / record['prediction']
            if sha256(p) != record['prediction_sha256'] or record['source_id'] != identity:
                raise ValueError('Specialist output identity changed')
            with np.load(p, allow_pickle=False) as z:
                spec = SignalProposal(z['values_mv'], float(z['fs']), identity, record['layout_id'], metadata=record['metadata'])
            evidence = SpecialistEvidence(**record['specialist_evidence'])
        eligible = data['npj_native'][0].layout_id == '3x4+1R' and not data['npj_native'][0].metadata.get('layout_fallback')
        selected, selected_status, decision = choose(data['npj_native'], data['grid_adaptive'],
            geometry(source, name, identity), spec, evidence, policy)
        if spec is None:
            spec = empty(identity, 'Native specialist failed or abstained' if eligible else 'Unsupported inferred layout', 'kaggle-native-eligible')
            spec_status = 'failed' if eligible else 'ineligible'
        else:
            spec_status = 'returned'
        data.update(kaggle_native_eligible=(spec, spec_status), selective=(selected, selected_status))
        times['kaggle_native_eligible'] = batch_seconds / max(1, len(special)) if eligible else 0.
        times['selective'] = sum(times.values())
        folder = out / name
        folder.mkdir()
        for arm, (p, status) in data.items():
            dest = folder / (arm + '.npz')
            np.savez_compressed(dest, values_mv=p.values_mv, fs=p.fs)
            records.append({'image_id': name, 'arm': arm, 'status': status,
                'prediction': str(dest.relative_to(out)), 'prediction_sha256': sha256(dest),
                'predicted_layout': p.metadata.get('native_layout'), 'layout_fallback': p.metadata.get('layout_fallback'),
                'backend_metadata': p.metadata, 'inference_wall_seconds': times[arm],
                'timing_basis': 'Reused measured native branches plus separately measured eligible specialist batch'})
        decision.update(image_id=name, source_id=identity, specialist_eligible=eligible)
        decisions.append(decision)
    write_json(out / 'phase2-decisions.json', {'schema': 'post-primary-frozen-policy-completion-v1',
        'original_selection_replay_verified': True, 'reference_received_by_selection': False,
        'policy': asdict(policy), 'policy_sha256': policy.sha256, 'decisions': decisions})
    report = {'schema': original['schema'], 'status': 'inference_complete_unscored', 'cpu_max_side': 3000,
              'inference_device': 'cuda', 'native_max_side': 3000, 'arms': ARMS, 'cases': records,
              'primary_run_preserved': True, 'continuation_after_primary_results_opened': True,
              'phase2_decisions_sha256': sha256(out / 'phase2-decisions.json')}
    write_json(out / 'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['primary', 'repair', 'data', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--specialist', type=Path)
    args = p.parse_args()
    image_only = json.loads((args.primary / 'image-only-manifest.json').read_text())
    report = build(image_only, args.primary, args.repair, args.specialist, args.out / 'run')
    frozen = json.loads((args.data / 'manifest.json').read_text())
    by_id = {c['image_id']: c for c in frozen['cases']}
    for row in report['cases']:
        case = by_id[row['image_id']]
        if sha256(args.data / case['reference']) != case['reference_sha256']:
            raise ValueError('Scoring reference changed')
        with np.load(args.data / case['reference'], allow_pickle=False) as gold, np.load(args.out / 'run' / row['prediction'], allow_pickle=False) as pred:
            row['evaluation'] = score_external_signals(gold['values_mv'], pred['values_mv'], 500)
    report['status'] = 'completed'
    write_json(args.out / 'run/report.json', report)
    summarize(args.data / 'manifest.json', args.out / 'run', args.out / 'summary')


if __name__ == '__main__':
    main()
