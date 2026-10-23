"""Replay M4 component tests in a disposable, correctly shaped workspace.

The frozen study scripts retain their original relative paths and hashes.
This launcher copies those exact bytes, links the existing repository read-only
by convention, and writes all new receipts in the requested workspace. It never
replaces the archived experiment results or requires neural checkpoints.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def locate():
    here = Path(__file__).resolve().parent
    bundle = next((p for p in (here, here.parent)
                   if (p/'geometry/protocol.json').is_file()), None)
    if bundle is None:
        raise FileNotFoundError('M4 geometry/protocol.json not found beside launcher')
    candidates = [bundle.parent/'ecg-torch-renderer', *bundle.parents]
    repo = next((p for p in candidates
                 if (p/'ecg_gpu/digitization/cell_mesh.py').is_file()), None)
    if repo is None:
        raise FileNotFoundError('Run from an intact ecg-torch-renderer source checkout')
    return bundle, repo


def make_workspace(destination: Path):
    bundle, repo = locate()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('geometry', 'sampling', 'identifiability', 'qa'):
        source = bundle/name
        target = destination/'m4'/name
        target.mkdir(parents=True)
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in ('.py', '.json'):
                shutil.copyfile(path, target/path.name)
    (destination/'m4/qa/receipts').mkdir()
    (destination/'ecg-torch-renderer').symlink_to(repo, target_is_directory=True)
    old = repo/'research/rectification-comparison-m3-20261023/composition_probe.py'
    if not old.is_file():
        raise FileNotFoundError('Pinned M3 composition comparator is absent')
    old_target = destination/'m3/our-critique'
    old_target.mkdir(parents=True)
    shutil.copyfile(old, old_target/old.name)
    return destination


def command(task, workspace):
    tasks = {
        'qa': [sys.executable, 'm4/qa/independent_component_tests.py'],
        'qa-area': [sys.executable, 'm4/qa/independent_axis_area_tests.py'],
        'sampling-tests': [sys.executable, '-m', 'unittest', 'discover', '-s',
                           'm4/sampling', '-p', 'test_sampling_export.py', '-v'],
        'area-tests': [sys.executable, '-m', 'unittest', 'discover', '-s',
                      'm4/sampling', '-p', 'test_axis_area.py', '-v'],
        'geometry': [sys.executable, 'm4/geometry/run_experiment.py', '--out',
                     str(workspace/'m4/geometry/replayed-results')],
        'identifiability': [sys.executable, 'm4/identifiability/controls.py', '--out',
                            str(workspace/'m4/identifiability/replayed-results.json'),
                            '--witness-dir', str(workspace/'m4/identifiability/replayed-witnesses')],
    }
    return tasks[task]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['qa', 'qa-area', 'sampling-tests', 'area-tests',
                                         'geometry', 'identifiability'], default='qa')
    parser.add_argument('--workspace', type=Path,
                        help='New directory for a retained replay. Must not already exist.')
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.workspace:
        work = make_workspace(args.workspace.absolute())
        return subprocess.call(command(args.task, work), cwd=work, env=env)
    with tempfile.TemporaryDirectory(prefix='ecg-m4-replay-') as directory:
        work = make_workspace(Path(directory)/'workspace')
        return subprocess.call(command(args.task, work), cwd=work, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
