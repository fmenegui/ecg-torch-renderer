"""Independent patient-balanced point and paired-bootstrap summary audit."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXECUTION = ROOT/'ecg-torch-renderer/build/m3-comparison'
SUMMARY = ROOT/'ecg-torch-renderer/build/m3-summary'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    summary = json.loads((SUMMARY/'summary.json').read_text())
    receipt = json.loads((SUMMARY/'postprocessing-receipt.json').read_text())
    schedule = [r for r in json.loads((EXECUTION/'schedule.json').read_text())['cases'] if r['split'] == 'test']
    methods = summary['methods']
    patients = sorted({r['group_id'] for r in schedule})
    backends = sorted({r['backend'] for r in schedule})
    results = {(r['key'], m): json.loads((EXECUTION/'cases'/r['key']/m/'result.json').read_text())
               for r in schedule for m in methods}
    checks, failures = 0, []

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(str(label))

    def same(a, b, label):
        check(a is None if b is None else a is not None and np.isclose(a, b, rtol=1e-10, atol=1e-12), label)

    check(receipt['expected_case_method_results'] == receipt['actual_case_method_results'] == 160, 'full diagnostic receipt')
    check(receipt['case_method_track_domain_rows'] == 1280, 'full32×5×4×2CSV')
    for filename, expected in receipt['output_sha256'].items():
        check(digest(SUMMARY/filename) == expected, 'summary output hash '+filename)
    for item in receipt['result_files']:
        check(digest(EXECUTION/item['path']) == item['sha256'], 'source result hash '+item['path'])
    check(set(summary['splits']) == {'test'}, 'no incomplete development aggregate')
    part = summary['splits']['test']
    for track in ('primary', 'secondary_D4', 'unrestricted_finite_field', 'operational_RGB'):
        for domain in ('paper', 'grid'):
            tensor = np.empty((5, 8, 2, 3))
            for im, method in enumerate(methods):
                for ip, patient in enumerate(patients):
                    for ib, backend in enumerate(backends):
                        values = []
                        for row in schedule:
                            if row['group_id'] != patient or row['backend'] != backend:
                                continue
                            result = results[row['key'], method]
                            name = 'unrestricted_finite_field' if track == 'operational_RGB' else track
                            metric = result['scores'][name]['domains'][domain]
                            values.append([metric[k] for k in ('g', 'w', 'c')] if track != 'operational_RGB' or result['rgb_resampled'] else [0., 0., 0.])
                        tensor[im, ip, ib] = np.mean(values, axis=0)
            point = tensor.mean(axis=(1, 2))
            indices = np.random.default_rng(1973).integers(0, 8, size=(20000, 8))
            draws = tensor[:, indices].mean(axis=(2, 3))
            ratios = np.divide(draws[..., 1], draws[..., 2], out=np.full(draws.shape[:2], np.nan), where=draws[..., 2] > 0)
            draws = np.concatenate((draws, ratios[..., None]), axis=2)
            baseline = methods.index('our-B0')
            bootstrap = part['tracks'][track][domain]['bootstrap']
            check(bootstrap['repetitions'] == 20000 and bootstrap['seed'] == 1973, (track, domain, 'draw protocol'))
            check(bootstrap['patients'] == patients and bootstrap['backends'] == backends, (track, domain, 'paired strata'))
            for im, method in enumerate(methods):
                expected_point = list(point[im])+[point[im, 1]/point[im, 2] if point[im, 2] else None]
                original = bootstrap['methods'][method]
                for ik, name in enumerate(('g', 'w', 'c', 'r')):
                    same(original['metrics'][name], expected_point[ik], (track, domain, method, 'point '+name))
                    delta = draws[im, :, ik]-draws[baseline, :, ik]
                    defined = delta[np.isfinite(delta)]
                    interval = original['delta_95_descriptive_interval'][name]
                    check(interval['defined_draws'] == len(defined), (track, domain, method, name, 'defined draws'))
                    same(interval['lower'], float(np.quantile(defined, .025)) if len(defined) else None,
                         (track, domain, method, name, 'lower'))
                    same(interval['upper'], float(np.quantile(defined, .975)) if len(defined) else None,
                         (track, domain, method, name, 'upper'))
    review = {'status': 'fail' if failures else 'pass', 'checks': checks, 'failures': failures,
              'summary_sha256': digest(SUMMARY/'summary.json'), 'postprocessing_receipt_sha256': digest(SUMMARY/'postprocessing-receipt.json'),
              'scope': 'All160 source-result hashes,1280CSV-row count, all patient/backend point estimates and all20000 paired patient-bootstrap draws recomputed independently for5methods×4tracks×2domains×4metrics. No inference or selection changed.',
              'independent_patients': len(patients), 'methods': len(methods), 'repetitions': 20000,
              'undefined_Kaggle_R_draws': 20000,
              'limit': 'A reproduced descriptive interval does not confer external or confirmatory validity.'}
    (Path(__file__).parent/'postprocessing-results-review.json').write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps(review, indent=2))
    return review


if __name__ == '__main__':
    result = run()
    raise SystemExit(1 if result['failures'] else 0)
