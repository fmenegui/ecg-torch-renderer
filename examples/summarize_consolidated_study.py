"""Patient-bootstrap summaries, paired raw support and honest failure denominators."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.digitization.signals import LEADS
from examples.digitization_layout_probe import printed_windows, support_mask


def f1(cases):
    counts = np.array([[c['detection'][k] for k in ('tp','fp','fn')] for c in cases]).sum(0)
    tp, fp, fn = counts
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.


def point_summary(trials, repeats=2000):
    rng = np.random.default_rng(1973)
    records = []
    for trial in trials:
        for backend in ('online','offline'):
            cases = [c for c in trial['cases'] if c['backend']==backend]
            groups = sorted({c['group_id'] for c in cases})
            clusters = {g:[c for c in cases if c['group_id']==g] for g in groups}
            draws = [f1([c for g in rng.choice(groups,len(groups)) for c in clusters[g]]) for _ in range(repeats)]
            accepted = [c for c in cases if c['topology_status']=='accepted']
            records.append({'name':trial['name'],'backend':backend,'patients':len(groups),
                'detection':trial['by_backend'][backend],'f1_patient_bootstrap_95':np.quantile(draws,[.025,.975]).tolist(),
                'topology_accepted':len(accepted),'images':len(cases),
                'accepted_with_relative_identity_at_least_99pct':sum(c['identity'] is not None and (c['identity']['fraction'] or 0)>=.99 for c in accepted),
                'relative_identity_scope':'selected nodes matched within 1.5px, not a certificate of the full warp'})
    return records


def point_deltas(trials, repeats=2000):
    """Paired patient bootstrap with the same seed, cases and training budget."""
    records = []
    rng = np.random.default_rng(20261021)
    for trial in trials:
        variant, seed = trial['name'].rsplit('-', 1)
        if variant == 'raw-point':
            continue
        baseline = next(t for t in trials if t['name'] == 'raw-point-'+seed)
        for backend in ('online', 'offline'):
            cases = [c for c in trial['cases'] if c['backend'] == backend]
            raw = [c for c in baseline['cases'] if c['backend'] == backend]
            groups = sorted({c['group_id'] for c in cases})
            intervention = {g:[c for c in cases if c['group_id'] == g] for g in groups}
            control = {g:[c for c in raw if c['group_id'] == g] for g in groups}
            differences = []
            for _ in range(repeats):
                selected = rng.choice(groups, len(groups))
                differences.append(f1([c for g in selected for c in intervention[g]])-
                                   f1([c for g in selected for c in control[g]]))
            records.append({'name':trial['name'], 'baseline':baseline['name'], 'backend':backend,
                            'f1_delta':f1(cases)-f1(raw),
                            'paired_patient_bootstrap_95':np.quantile(differences,[.025,.975]).tolist()})
    return records


def signal_summary(root):
    pairs=json.loads((root/'signal-inputs.json').read_text())
    results=json.loads((root/'signal-results.json').read_text())
    records=[]
    totals={arm:{'attempted':0,'returned':0,'correct_layout':0,'coverage':[],'conditional_snr_db':[]} for arm in ('original','oracle','learned','resize-2x')}
    for patient in pairs:
        source=root/patient['original']
        manifest=json.loads((source/'manifest.json').read_text())
        with np.load(source/'source.npz') as data:
            truth,fs=data['values_mv'],int(data['fs'])
        for entry in manifest['cases']:
            windows=printed_windows(entry['metadata'])
            printed=np.stack([support_mask(np.arange(truth.shape[1])/fs,windows[lead]) for lead in LEADS])
            record={'source_id':patient['source_id'],'group_id':patient['group_id'],'backend':entry['name'],'printed_samples':int(printed.sum()),'arms':{}}
            predictions={}
            for arm, count in totals.items():
                trial=next((x for x in results if x['source_id']==patient['source_id'] and x['arm']==arm),None)
                case=next((x for x in trial['cases'] if x['name']==entry['name']),None) if trial else None
                count['attempted']+=1
                valid=case is not None and case.get('status')=='returned'
                matching=valid and case.get('layout_matches_expected',False)
                count['returned']+=int(valid)
                count['correct_layout']+=int(matching)
                evaluation=case.get('evaluation',{}) if case else {}
                # Rejections remain zero coverage in the all-case population.
                count['coverage'].append(evaluation.get('printed_time_coverage',0.) if matching else 0.)
                if matching and evaluation.get('mean_observed_lead_snr_db') is not None:
                    count['conditional_snr_db'].append(evaluation['mean_observed_lead_snr_db'])
                record['arms'][arm]={'status':case.get('status') if case else 'geometry_rejected', 'layout_correct':bool(matching),'evaluation':evaluation}
                path=root/'signal-results'/patient['source_id']/arm/'configured'/(entry['name']+'-prediction.npz')
                if matching and path.exists():
                    with np.load(path) as data:predictions[arm]=data['values_mv'].copy()
            # Report pairwise original-vs-intervention support, so a missing third arm cannot erase valid pairs.
            for alternative in ('oracle','learned'):
                common=printed.copy()
                if 'original' not in predictions or alternative not in predictions:
                    common[:]=False
                else:
                    common &= np.isfinite(predictions['original']) & np.isfinite(predictions[alternative])
                pair={'samples':int(common.sum())}
                for arm in ('original',alternative):
                    if arm in predictions and common.any():
                        delta=predictions[arm][common]-truth[common]
                        pair[arm+'_raw_rmse_mv']=float(np.sqrt(np.mean(delta**2)))
                record['original_vs_'+alternative]=pair
            records.append(record)
    for counts in totals.values():
        counts['all_case_mean_coverage']=float(np.mean(counts.pop('coverage')))
        snr=counts.pop('conditional_snr_db')
        counts['mean_snr_correct_layout_returned_only_db']=float(np.mean(snr)) if snr else None
        counts['snr_cases']=len(snr)
    return {'totals':totals,'cases':records}


def reference_summary(root):
    path = root/'reference-results.json'
    if not path.exists():
        return None
    references = json.loads(path.read_text())
    totals = {}
    pairs = []
    for method in ('npj_dewarping_enabled', 'kaggle'):
        cases = [c for patient in references for c in patient[method]]
        valid = lambda c: c['status'] == 'returned' and c.get(
            'layout_matches_expected', c.get('layout_assumption_matches_input', False))
        compatible = [c for c in cases if c.get('layout_assumption_matches_input', True)]
        totals[method] = {'attempted':len(cases), 'returned':sum(c['status']=='returned' for c in cases),
            'correct_or_compatible_layout_returns':sum(valid(c) for c in cases),
            'layout_compatible_inputs':len(compatible),
            'all_case_mean_coverage':float(np.mean([c.get('evaluation',{}).get('printed_time_coverage',0.) if valid(c) else 0. for c in cases])),
            'layout_compatible_input_mean_coverage':float(np.mean([c.get('evaluation',{}).get('printed_time_coverage',0.) if valid(c) else 0. for c in compatible])) if compatible else None}
    patients = json.loads((root/'signal-inputs.json').read_text())
    main_results = json.loads((root/'signal-results.json').read_text())
    from itertools import combinations
    methods = ('original','oracle','learned','resize-2x','npj_dewarping_enabled','kaggle')
    for patient in patients:
        original = root/patient['original']
        manifest = json.loads((original/'manifest.json').read_text())
        with np.load(original/'source.npz') as data:
            truth, fs = data['values_mv'], int(data['fs'])
        reference = next(r for r in references if r['source_id'] == patient['source_id'])
        for entry in manifest['cases']:
            windows = printed_windows(entry['metadata'])
            printed = np.stack([support_mask(np.arange(truth.shape[1])/fs,windows[lead]) for lead in LEADS])
            predictions = {}
            for method in methods:
                if method in ('original','oracle','learned','resize-2x'):
                    trial = next((r for r in main_results if r['source_id'] == patient['source_id'] and r['arm'] == method), None)
                    cases = trial['cases'] if trial else []
                    filename = root/'signal-results'/patient['source_id']/method/'configured'/(entry['name']+'-prediction.npz')
                else:
                    cases = reference[method]
                    folder = 'npj-dewarping-enabled/configured' if method == 'npj_dewarping_enabled' else 'kaggle'
                    filename = root/'reference-results'/patient['source_id']/folder/(entry['name']+'-prediction.npz')
                case = next((c for c in cases if c['name'] == entry['name']), None)
                if case and case['status'] == 'returned' and case.get('layout_matches_expected',case.get('layout_assumption_matches_input',False)) and filename.exists():
                    with np.load(filename) as data:
                        predictions[method] = data['values_mv'].copy()
            for first, second in combinations(methods,2):
                common = printed.copy()
                if first not in predictions or second not in predictions:
                    common[:] = False
                else:
                    common &= np.isfinite(predictions[first]) & np.isfinite(predictions[second])
                record = {'source_id':patient['source_id'],'group_id':patient['group_id'],'backend':entry['name'],
                          'first':first,'second':second,'common_printed_samples':int(common.sum()),'printed_samples':int(printed.sum())}
                if common.any():
                    for method in (first,second):
                        delta = predictions[method][common]-truth[common]
                        record[method+'_raw_rmse_mv'] = float(np.sqrt(np.mean(delta**2)))
                pairs.append(record)
    pooled = []
    for first, second in combinations(methods,2):
        rows = [r for r in pairs if r['first']==first and r['second']==second]
        count = sum(r['common_printed_samples'] for r in rows)
        printed = sum(r['printed_samples'] for r in rows)
        record = {'first':first,'second':second,'common_samples':count,
                  'cases_with_common_support':sum(r['common_printed_samples']>0 for r in rows),
                  'total_cases':len(rows),'common_printed_support_fraction':count/printed if printed else None}
        for method in (first,second):
            squared = sum(r['common_printed_samples']*r[method+'_raw_rmse_mv']**2 for r in rows if r['common_printed_samples'])
            record[method+'_pooled_raw_rmse_mv'] = float(np.sqrt(squared/count)) if count else None
        pooled.append(record)
    return {'totals':totals,'pairwise_raw_errors':pairs,'pooled_pairwise_raw_errors':pooled}


def main(root):
    report={'schema':'ecg-consolidated-summary-v1','points':point_summary(json.loads((root/'test-results.json').read_text())),
        'paired_point_deltas':point_deltas(json.loads((root/'test-results.json').read_text())),
        'signals':signal_summary(root), 'references':reference_summary(root), 'limits':['Synthetic captures only, original patient waveforms','Two seeds, eight held-out patients','Paper gain supplied to frozen npj','Conditional SNR is not the Kaggle competition metric','Oracle supplies known geometry only, not a guaranteed upper bound on extraction because output density and crop differ','Grid association does not certify global injectivity','Train and test share capture families and geometry ranges','Resize 2x does not exactly match every warped output density, so the comparison evaluates the pipeline rather than isolating geometry']}
    _write_json(root/'consolidated-summary.json',report)
    print(json.dumps(report['signals']['totals'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
