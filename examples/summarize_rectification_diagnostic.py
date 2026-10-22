"""Patient-level summaries and paired bootstrap for frozen diagnostic replays."""
import argparse
import json
from pathlib import Path

import numpy as np


def patient_values(cases, method, backend=None):
    patients = sorted({c['group_id'] for c in cases})
    values = []
    for patient in patients:
        means = []
        for name in ((backend,) if backend else ('torch-cached', 'offline')):
            rows = [c['results'][method] for c in cases
                    if c['group_id'] == patient and c['backend'] == name]
            if rows:
                means.append(np.mean([[r['coverage_within_0_5mm'],
                    r['covered_query_fraction']-r['coverage_within_0_5mm'],
                    r['covered_query_fraction']] for r in rows], axis=0))
        values.append(np.mean(means, axis=0))
    return np.asarray(values)


def describe(cases, method, backend=None):
    chosen = [c for c in cases if backend is None or c['backend'] == backend]
    values = patient_values(chosen, method, backend)
    errors = [c['results'][method]['conditional_p95_mm'] for c in chosen
              if c['results'][method]['conditional_p95_mm'] is not None]
    return dict(zip(('good_coverage', 'wrong_covered_fraction', 'covered_fraction'),
                    map(float, values.mean(0))),
        images=len(chosen), patients=len(values),
        median_image_conditional_p95_mm=float(np.median(errors)) if errors else None,
        images_with_conditional_error=len(errors),
        images_below_10pct_good_coverage=sum(c['results'][method]['coverage_within_0_5mm'] < .1 for c in chosen),
        images_below_50pct_good_coverage=sum(c['results'][method]['coverage_within_0_5mm'] < .5 for c in chosen))


def comparison(cases, baseline, selected, backend=None):
    delta = patient_values(cases, selected, backend)-patient_values(cases, baseline, backend)
    rng = np.random.default_rng(1973)
    sample = rng.integers(0, len(delta), (20000, len(delta)))
    boot = delta[sample].mean(1)
    result = {}
    for column, metric in enumerate(('good_coverage', 'wrong_covered_fraction', 'covered_fraction')):
        result[metric] = {'delta': float(delta[:, column].mean()),
            'paired_patient_bootstrap_ci95': np.quantile(boot[:, column], [.025, .975]).tolist(),
            'patient_deltas': delta[:, column].tolist()}
    chosen = [c for c in cases if backend is None or c['backend'] == backend]
    gains = np.asarray([c['results'][selected]['coverage_within_0_5mm']
                        -c['results'][baseline]['coverage_within_0_5mm'] for c in chosen])
    result.update(images_improved_over_1pp=int((gains > .01).sum()),
                  images_regressed_over_1pp=int((gains < -.01).sum()))
    return result


def common_domain(cases, maps, baseline, selected, backend=None):
    rows = []
    for case in cases:
        if backend is not None and case['backend'] != backend:
            continue
        a = np.load(maps/(case['key']+'-'+baseline+'.npz'))
        b = np.load(maps/(case['key']+'-'+selected+'.npz'))
        if not np.array_equal(a['query'], b['query']) or not np.array_equal(a['reference'], b['reference']):
            raise ValueError('Query-domain parity lost')
        keep = np.isfinite(a['predicted']).all(1) & np.isfinite(b['predicted']).all(1)
        errors = [np.linalg.norm(p['predicted'][keep]-p['reference'][keep], axis=1) for p in (a, b)]
        rows.append({'key': case['key'], 'queries': len(keep), 'common_queries': int(keep.sum()),
                     'baseline_squared_error_sum': float((errors[0]**2).sum()),
                     'selected_squared_error_sum': float((errors[1]**2).sum())})
        a.close()
        b.close()
    count = sum(r['common_queries'] for r in rows)
    return {'pooled_common_query_fraction': count/sum(r['queries'] for r in rows),
            'baseline_pooled_common_rmse_mm': float(np.sqrt(sum(r['baseline_squared_error_sum'] for r in rows)/count)) if count else None,
            'selected_pooled_common_rmse_mm': float(np.sqrt(sum(r['selected_squared_error_sum'] for r in rows)/count)) if count else None,
            'weighting': 'pooled image query points, conditional on common finite support'}


def summarize(root, out):
    selection = json.loads((root/'selection.json').read_text())
    selected = selection['candidate']+'-network'
    result = {'selected': selected, 'cli_profile': selection['cli_profile'],
              'good_definition': 'fixed gold query domain covered with material error <=0.5 mm',
              'wrong_definition': 'fixed domain covered with material error >0.5 mm',
              'aggregation': 'patient means, equal backend weights for overall results',
              'bootstrap': '20000 paired patient resamples, seed 1973, six patients per split',
              'inference_scope': 'conditional on these six replay patients, not a population or clinical claim',
              'splits': {}}
    for split in ('validation', 'test'):
        cases = json.loads((root/(split+'.json')).read_text())['cases']
        if len(cases) != 48:
            raise ValueError('Completed 48-image split required')
        methods = ('legacy-oracle', 'cells-oracle', 'legacy-anonymous', 'cells-anonymous',
                   'legacy-network', 'cells-network', selected)
        summary = {'by_method': {}, 'paired': {}, 'common_domain': {},
                   'patient_ids': sorted({c['group_id'] for c in cases})}
        for backend in (None, 'torch-cached', 'offline'):
            name = backend or 'backend-balanced'
            summary['by_method'][name] = {m: describe(cases, m, backend) for m in methods}
            summary['paired'][name] = {m: comparison(cases, m, selected, backend)
                                       for m in ('legacy-network', 'cells-network')}
            summary['common_domain'][name] = {m: common_domain(cases, root/'query-maps', m, selected, backend)
                                             for m in ('legacy-network', 'cells-network')}
        result['splits'][split] = summary
    out.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    summarize(**vars(parser.parse_args()))
