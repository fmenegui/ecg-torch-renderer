"""Independently recount saved counterfactual geometry without the metric core."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(comparison, diagnostic, out):
    rows = json.loads((diagnostic/'cases.json').read_text())
    checks = 0
    cases = sorted({r['key'] for r in rows})
    methods = sorted({r['method'] for r in rows if r['method'] not in
                      ('E0-native-M2', 'E1-orientation-oracle')})
    for key in cases:
        with np.load(comparison/'queries'/(key+'.npz')) as q:
            for method in methods:
                with np.load(diagnostic/'predictions'/key/(method+'.npz')) as p:
                    keep = p['anchors_claimed_eligible'] & np.isfinite(p['anchors_coords']).all(1)
                    reference = q['anchors_reference'][keep]
                    registered = len(reference) >= 3 and (np.ptp(reference, axis=0) >= 1.).all()
                    offset = np.median(reference-p['anchors_coords'][keep], axis=0) if registered else np.zeros(2)
                    for domain in ('paper', 'grid'):
                        row = next(r for r in rows if r['key'] == key and r['method'] == method and r['domain'] == domain)
                        coords = p[domain+'_coords']
                        accepted = p[domain+'_claimed_eligible'].astype(bool)
                        if not registered:
                            accepted[:] = False
                        finite = np.isfinite(coords).all(1)
                        error = np.linalg.norm(coords[finite]+offset-q[domain+'_reference'][finite], axis=1)
                        correct = np.zeros(len(coords), bool)
                        correct[finite] = error <= .5
                        values = {'g': np.count_nonzero(accepted & correct)/len(coords),
                                  'w': np.count_nonzero(accepted & ~correct)/len(coords),
                                  'c': np.count_nonzero(accepted)/len(coords)}
                        for name, value in values.items():
                            assert value == row[name], (key, method, domain, name)
                            checks += 1
                        assert not p[domain+'_native_footprint'].any()
                        checks += 1
    result = {'independent_counter_checks': checks, 'cases': len(cases),
              'counterfactual_methods': len(methods), 'passed': True,
              'method': 'Direct finite masks, anchor medians and Euclidean errors. No scoring-core imports.',
              'scope': 'Saved component geometry only, not independent clinical observations.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'diagnostic', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(audit(**vars(parser.parse_args()))))
