"""Compare compatibility-only retest against the approved guard-v2 receipts."""
import ast
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(after_dir, receipt):
    historical = ROOT/'geometry_selector_guard_v2_historical.py'
    current = ROOT/'geometry_selector.py'
    original_line = '    cross = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])\n'
    replacement = ('    edge_a = vertices[:, 1]-vertices[:, 0]\n'
                   '    edge_b = vertices[:, 2]-vertices[:, 0]\n'
                   '    cross = edge_a[:, 0]*edge_b[:, 1]-edge_a[:, 1]*edge_b[:, 0]\n')
    exact_source_change = (historical.read_text().count(original_line) == 1
        and historical.read_text().replace(original_line, replacement) == current.read_text())
    paths = [ROOT/'geometry_selector.py', ROOT/'geometry_selector_sparse.py',
        ROOT/'run_experiment.py', ROOT/'check_results.py', ROOT/'sparse_parity_cost.py',
        WORKSPACE/'m3/our-critique/composition_probe.py',
        WORKSPACE/'ecg-torch-renderer/ecg_gpu/digitization/cell_mesh.py']
    cross_calls = []
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'np'
                and node.func.attr == 'cross'):
                cross_calls.append({'path': str(path.relative_to(WORKSPACE)), 'line': node.lineno})
    before_dir = ROOT/'results-guard-v2'
    before = json.loads((before_dir/'results.json').read_text())
    after = json.loads((after_dir/'results.json').read_text())
    checks = 0
    differences = []
    candidate_arrays = 0
    primary_records = 0
    array_records = 0
    scene_records = []
    assert len(before['cases']) == len(after['cases']) == 19
    for a,b in zip(before['cases'], after['cases']):
        scene = a['scene']['name']
        checks += 1
        assert scene == b['scene']['name']
        selected_same = a['selection']['selected_candidate'] == b['selection']['selected_candidate']
        checks += 1
        if not selected_same:
            differences.append({'scene': scene, 'kind': 'selected name'})
        for name in a['selection']['candidates']:
            old = a['selection']['candidates'][name]
            new = b['selection']['candidates'][name]
            for field in ['fold_loss','fold_rmse_mm','fold_status','mean_loss','standard_error','full_fit_status']:
                checks += 1
                if old[field] != new[field]:
                    differences.append({'scene': scene, 'candidate': name, 'selection_field': field})
        for name, record in a['methods'].items():
            checks += 1
            primary_records += 1
            if record != b['methods'][name]:
                differences.append({'scene': scene, 'primary_record': name})
        for name, record in a['all_candidates_physical_scores_after_selection'].items():
            checks += 1
            if record != b['all_candidates_physical_scores_after_selection'][name]:
                differences.append({'scene': scene, 'candidate_physical_record': name})
        with np.load(before_dir/a['array_file']) as old, np.load(after_dir/b['array_file']) as new:
            assert old.files == new.files
            for key in old.files:
                checks += 1
                array_records += 1
                if key.startswith('candidate_') and key.endswith('_prediction'):
                    candidate_arrays += 1
                same = np.array_equal(old[key],new[key],equal_nan=True) if old[key].dtype.kind in 'fc' else np.array_equal(old[key],new[key])
                if not same:
                    differences.append({'scene': scene, 'array': key})
        scene_records.append({'scene': scene,'same_selected_candidate': selected_same})
    result = {'schema': 'm4-numpy2d-cross-compatibility-regression-v1',
        'compatibility_revision_sha256': sha(ROOT/'numpy-compatibility-revision.json'),
        'historical_geometry_source_sha256': sha(historical),
        'new_geometry_source_sha256': sha(current),
        'source_change_exactly_requested_scalar_determinant': exact_source_change,
        'source_diff': ''.join(difflib.unified_diff(historical.read_text().splitlines(True), current.read_text().splitlines(True), fromfile=historical.name, tofile=current.name)),
        'audited_current_path_sha256': {str(p.relative_to(WORKSPACE)):sha(p) for p in paths},
        'np_cross_calls_in_current_component_path': cross_calls,
        'historical_snapshots_intentionally_excluded_from_current_path': ['geometry_selector_v1.py','geometry_selector_guard_v2_historical.py'],
        'same_selected_names_19': all(x['same_selected_candidate'] for x in scene_records),
        'candidate_arrays_compared': candidate_arrays,
        'all_saved_arrays_compared': array_records,
        'primary_score_records_compared': primary_records,
        'check_count': checks,
        'differences': differences,
        'scenes': scene_records,
        'decision_seal_unchanged_after_compatible_run': after['decision_seal_unchanged'],
        'physical_gold_used_in_selection': after['physical_gold_used_in_selection'],
        'local_environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'actual_external_ci_not_executed_by_this_local_receipt': True,
        'after_results_directory': str(after_dir),
        'source_sha256': sha(Path(__file__))}
    result['pass'] = (exact_source_change and not cross_calls and not differences
        and candidate_arrays == 342 and primary_records == 76
        and after['decision_seal_unchanged'] and not after['physical_gold_used_in_selection'])
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    if not result['pass']:
        raise AssertionError('NumPy compatibility regression check failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-dir', type=Path, default=ROOT/'results-numpy-compatible')
    parser.add_argument('--receipt', type=Path, default=ROOT/'numpy-compatibility-retest.json')
    args = parser.parse_args()
    run(args.after_dir, args.receipt)
