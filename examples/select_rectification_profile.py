"""Freeze a geometry profile using validation patients, never test outcomes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

CANDIDATES = ('cells', 't70', 'clean85', 'clean70', 'anchor50')
PROFILES = {'cells': 'baseline', 't70': 'threshold-70', 'clean85': 'banding',
            'clean70': 'banding-recall', 'anchor50': 'geometry-recall'}


def summarize(cases, method):
    patients = sorted({c['group_id'] for c in cases})
    backend = {}
    for name in ('torch-cached', 'offline'):
        values = []
        for patient in patients:
            selected = [c['results'][method+'-network'] for c in cases
                        if c['group_id'] == patient and c['backend'] == name]
            if selected:
                good = np.mean([r['coverage_within_0_5mm'] for r in selected])
                wrong = np.mean([r['covered_query_fraction']-r['coverage_within_0_5mm'] for r in selected])
                values.append((good, wrong))
        if not values:
            raise ValueError('Both capture backends must be represented')
        backend[name] = {'good_coverage': float(np.mean([v[0] for v in values])),
                         'wrong_covered_fraction': float(np.mean([v[1] for v in values]))}
    return {'patients': len(patients), 'images': len(cases), 'by_backend': backend,
            'backend_macro_good_coverage': float(np.mean([b['good_coverage'] for b in backend.values()])),
            'backend_macro_wrong_coverage': float(np.mean([b['wrong_covered_fraction'] for b in backend.values()]))}


def choose(cases):
    if any(c['split'] != 'validation' for c in cases):
        raise ValueError('Profile selection accepts validation cases only')
    summaries = {m: summarize(cases, m) for m in CANDIDATES}
    baseline = summaries['cells']
    eligible = []
    for name, row in summaries.items():
        passes = row['backend_macro_wrong_coverage'] <= baseline['backend_macro_wrong_coverage']+.005
        passes &= all(row['by_backend'][b]['wrong_covered_fraction']
                      <= baseline['by_backend'][b]['wrong_covered_fraction']+.01
                      for b in baseline['by_backend'])
        row['eligible'] = bool(passes)
        if passes:
            eligible.append(name)
    selected = max(eligible, key=lambda m: summaries[m]['backend_macro_good_coverage'])
    return {'candidate': selected, 'cli_profile': PROFILES[selected], 'validation': summaries,
            'selection_rule': 'maximize backend-balanced patient-mean coverage within 0.5 mm, '
                'subject to wrong covered fraction <= baseline +0.005 overall and +0.01 per backend',
            'clinical_acceptance_rule': False, 'test_outcomes_used': False}


def main(validation, out, checkpoint):
    report = json.loads(validation.read_text())
    if report['split'] != 'validation' or len(report['cases']) != 48:
        raise ValueError('Completed frozen 48-image validation diagnostic required')
    if out.exists():
        raise FileExistsError('Do not overwrite an already frozen selection')
    result = choose(report['cases'])
    result['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['validation_sha256'] = hashlib.sha256(validation.read_bytes()).hexdigest()
    result['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result['source_code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path('ecg_gpu/digitization/cell_mesh.py'), Path('ecg_gpu/digitization/cycle_lattice.py'),
        Path('ecg_gpu/digitization/observation.py'), Path('ecg_gpu/digitization/lattice.py'),
        Path('ecg_gpu/digitization/local_cells.py'), Path('ecg_gpu/digitization/denoising_grid.py'),
        Path('ecg_gpu/grid_point_learning.py'), Path('examples/diverse_grid_study.py'),
        Path('examples/evaluate_rectification_diagnostic.py'), Path('examples/rectify_denoising_grid.py')]}
    out.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('validation', 'out', 'checkpoint'):
        parser.add_argument('--'+key, type=Path, required=True)
    print(json.dumps(main(**vars(parser.parse_args()))))
