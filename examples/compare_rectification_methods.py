"""Frozen M3 diagnostic comparison of native ECG rectification stages.

Gold arrays are confined to query preparation and the independent scorer.
Original comparator code/checkpoints remain external and pinned. The existing
M2 captures are retrospective diagnostic data, not a new unseen holdout.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg_gpu.digitization.cell_mesh import CellMap, cell_triangles
from ecg_gpu.digitization.complete_rectifier import CompleteMap
from ecg_gpu.digitization.denoising_grid import DenoisingGridNet
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.local_cells import locally_supported_cells
from ecg_gpu.digitization.observation import remove_periodic_banding
from examples.diverse_grid_study import detect
from examples.evaluate_nextgen_refinement import FROZEN_PATHS, CHECKPOINT_SHA
from examples.rectify_denoising_grid import write_geometry_output

METHODS = ('our-B0', 'our-M2', 'npj-generic', 'npj-george-moody-2024', 'kaggle-third')
DOMAINS = ('anchors', 'grid', 'paper')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def deadline(seconds):
    def expired(_signum, _frame):
        raise TimeoutError(f'Predeclared {seconds}s operation budget exceeded')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class OurNativeMap:
    """Reconstruct exactly the map archived by the unchanged production CLI."""
    def __init__(self, folder, report):
        self.rgb_resampled = bool(report['complete_geometry']['resampled'])
        with np.load(folder/'observed-grid-geometry.npz') as z:
            observed, canonical = z['xy'].copy(), z['canonical_mm'].copy()
            identities, cells = z['ids'].copy(), z['cells'].copy()
        self.mapping = None
        if self.rgb_resampled:
            with np.load(folder/'full-frame-mesh.npz') as z:
                self.mapping = CompleteMap(z['source'].copy(), z['target_mm'].copy(),
                    np.empty((0, 2), int), z['triangles'].copy(),
                    report['complete_geometry'], observed, canonical, identities, cells)
        triangles = cell_triangles(observed, canonical, identities, cells)
        self.partial = CellMap(observed, canonical, triangles)

    def predict(self, query):
        if self.mapping is None:
            coords = self.partial(query)
            return coords, np.isfinite(coords).all(1), np.zeros(len(query), bool)
        coords = self.mapping.forward(query)
        eligible = self.mapping.eligible_geometry(query, radius=.75, mapped=coords)
        return coords, eligible, np.isfinite(coords).all(1)


class OurDetector:
    def __init__(self, checkpoint):
        if digest(checkpoint) != CHECKPOINT_SHA:
            raise ValueError('Frozen M2 checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.threshold = float(state['threshold'])
        self.model = DenoisingGridNet(width=state['width']).eval()
        self.model.load_state_dict(state['state_dict'], strict=True)

    def infer(self, image, folder, previous_cache):
        started = time.perf_counter()
        with torch.inference_mode():
            xy, scores, network = detect(self.model, image, device='cpu', threshold=self.threshold)
        del network
        prepared, banding = remove_periodic_banding(image, xy)
        if banding['applied']:
            with torch.inference_mode():
                xy, scores, network = detect(self.model, prepared, device='cpu', threshold=self.threshold)
            del network
        seconds = time.perf_counter()-started
        topology_start = time.perf_counter()
        topology = locally_supported_cells(xy, organize_lattice(xy, scores))
        topology_seconds = time.perf_counter()-topology_start
        np.savez_compressed(folder/'fresh-observations.npz', xy=xy, scores=scores)
        replay = {}
        if previous_cache.exists():
            with np.load(previous_cache) as z:
                old_xy, old_scores = z['xy'], z['scores']
                same_shape = old_xy.shape == xy.shape
                replay = {'old_points': len(old_xy), 'fresh_points': len(xy),
                          'coordinates_exact': bool(np.array_equal(old_xy, xy)),
                          'scores_exact': bool(np.array_equal(old_scores, scores)),
                          'maximum_coordinate_difference_px': float(np.max(np.abs(old_xy-xy)))
                          if same_shape and len(xy) else None}
        return xy, scores, topology, {
            'fresh_detector_seconds': seconds, 'threshold': self.threshold,
            'topology_seconds': topology_seconds,
            'checkpoint_sha256': CHECKPOINT_SHA, 'banding': banding,
            'gold_used_for_inference': False, 'M2_replay_comparison': replay}


def load_queries(path):
    with np.load(path) as z:
        return {d: {'source': z[d+'_source'].copy(),
                    'reference': z[d+'_reference'].copy()} for d in DOMAINS}


def failed_predictions(queries):
    return {d: {'coords': np.full(queries[d]['source'].shape, np.nan),
                'claimed_eligible': np.zeros(len(queries[d]['source']), bool),
                'native_footprint': np.zeros(len(queries[d]['source']), bool)} for d in DOMAINS}


def freeze(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'lock.json').exists():
        raise ValueError('Study already frozen, do not overwrite its lock')
    sources = [ROOT/p for p in FROZEN_PATHS]
    sources += [Path(__file__).resolve(), args.metric_core, args.npj_adapter, args.kaggle_adapter, args.summarizer,
                ROOT/'examples/rectify_denoising_grid.py', args.protocol, args.schedule]
    sources += list((args.npj_repo/'src').rglob('*.py'))
    sources += list(args.npj_repo.rglob('*.yml'))
    sources += list(args.kaggle_repo.rglob('*.py'))
    weights = [args.checkpoint, args.npj_repo/'weights/unet_weights_07072025.pt',
               args.kaggle_assets/'weight/stage0-last.checkpoint.pth',
               args.kaggle_assets/'weight/stage1-last.checkpoint.pth',
               args.kaggle_assets/'640106434-0001.gridpoint_xy.npy']
    schedule = json.loads(args.schedule.read_text())
    if digest(args.bank/'manifest.json') != schedule['manifest_sha256']:
        raise ValueError('Restored capture manifest mismatch')
    queries = out/'queries'
    queries.mkdir(exist_ok=True)
    for case in schedule['cases']:
        p = args.schedule.parent/case['query_file']
        if digest(p) != case['query_archive_sha256']:
            raise ValueError('Prepared query archive mismatch')
        shutil.copy2(p, queries/(case['key']+'.npz'))
    shutil.copy2(args.protocol, out/'protocol.json')
    shutil.copy2(args.schedule, out/'schedule.json')
    lock = {'schema': 'M3-execution-lock-v1', 'date': '2026-10-23',
            'methods': METHODS, 'captures': len(schedule['cases']),
            'source_sha256': {str(p.resolve()): digest(p) for p in sources},
            'owned_file_sha256': {name: digest(out/name) for name in ('protocol.json', 'schedule.json')},
            'weight_sha256': {str(p.resolve()): digest(p) for p in weights},
            'query_sha256': {p.name: digest(p) for p in queries.glob('*.npz')},
            'our_commit': 'f4f6caef3b59b5830e93284cccadbc6855fc701e',
            'new_training': False, 'unseen_holdout_claim': False,
            'runtime': {'torch': torch.__version__, 'opencv': cv2.__version__,
                        'numpy': np.__version__, 'device': 'cpu', 'threads': 2,
                        'packages': {name: version(name) for name in
                                     ('torchvision', 'Pillow', 'scipy', 'timm', 'shapely', 'connected-components-3d')}},
            'runtime_paths': {name: str(getattr(args, name).resolve()) for name in
                              ('bank', 'checkpoint', 'metric_core', 'npj_adapter', 'npj_repo',
                               'kaggle_adapter', 'kaggle_repo', 'kaggle_assets', 'protocol', 'schedule', 'summarizer')},
            'upstream_commits': {name: subprocess.check_output(
                ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
                for name, path in (('npj', args.npj_repo), ('kaggle', args.kaggle_repo))},
            'RGB_output_budget_pixels': 16000000, 'case_timeout_seconds': 180,
            'timeout_mechanism': 'Cooperative SIGALRM, native calls can defer Python handling; elapsed and overruns retained',
            'primary_execution_split': 'test', 'primary_scheduled_captures': 32,
            'detector_reexecution': 'Fresh original CNN per capture shared by B0/M2, never historical latency',
            'RGB_operation': 'CLI native4px/mm for our method, actual native output for competitors'}
    write_json(out/'lock.json', lock)
    print(json.dumps({'frozen': True, 'cases': lock['captures'], 'methods': METHODS,
                      'lock_sha256': digest(out/'lock.json')}), flush=True)


def verify_lock(out):
    lock = json.loads((out/'lock.json').read_text())
    for group in ('source_sha256', 'weight_sha256'):
        for path, expected in lock[group].items():
            if digest(path) != expected:
                raise ValueError('Source or weights changed after freeze: '+path)
    for name, expected in lock['query_sha256'].items():
        if digest(out/'queries'/name) != expected:
            raise ValueError('Queries changed after freeze')
    for name, expected in lock['owned_file_sha256'].items():
        if digest(out/name) != expected:
            raise ValueError('Lock-owned protocol or schedule changed')
    return lock


def execute(args):
    lock = verify_lock(args.out)
    for name, expected in lock['runtime_paths'].items():
        if str(getattr(args, name).resolve()) != expected:
            raise ValueError('Runtime path differs from frozen binding: '+name)
    metric = load_module('m3_metric_core', args.metric_core)
    npj = load_module('m3_npj_adapter', args.npj_adapter)
    kaggle = load_module('m3_kaggle_adapter', args.kaggle_adapter)
    selected_methods = list(METHODS) if args.method == 'all' else [args.method]
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    load_errors, competitors, our = {}, {}, None
    if any(m.startswith('our-') for m in selected_methods):
        try:
            with deadline(300):
                our = OurDetector(args.checkpoint)
        except Exception as exc:
            load_errors['our-B0'] = load_errors['our-M2'] = type(exc).__name__+': '+str(exc)
    for name in selected_methods:
        try:
            with deadline(300):
                if name.startswith('npj-'):
                    competitors[name] = npj.NPJPerspectiveRectifier(args.npj_repo, preset=name[4:])
                elif name == 'kaggle-third':
                    competitors[name] = kaggle.KaggleThirdPlaceAdapter(args.kaggle_repo, args.kaggle_assets, threads=2)
        except Exception as exc:
            load_errors[name] = type(exc).__name__+': '+str(exc)
    schedule = json.loads((args.out/'schedule.json').read_text())['cases']
    rows = [r for r in schedule if args.split == 'all' or r['split'] == args.split]
    patients = sorted({r['group_id'] for r in rows})
    if args.limit:
        rows = rows[:args.limit]
    completed = 0
    for row in rows:
        if digest(args.bank/row['file']) != row['sha256']:
            raise ValueError('Input capture changed')
        with np.load(args.bank/row['file']) as z:
            image = z['image'].copy()
        queries = load_queries(args.out/'queries'/(row['key']+'.npz'))
        case_folder = args.out/'cases'/row['key']
        case_folder.mkdir(parents=True, exist_ok=True)
        observations, observation_error = None, None
        methods = selected_methods if patients.index(row['group_id']) % 2 == 0 else selected_methods[::-1]
        for method in methods:
            folder = case_folder/method
            folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists() and not args.no_resume:
                previous = json.loads((folder/'result.json').read_text())
                if previous['lock_sha256'] != digest(args.out/'lock.json'):
                    raise ValueError('Existing result belongs to a different execution lock')
                if previous['predictions_sha256'] != digest(folder/'query-predictions.npz'):
                    raise ValueError('Existing prediction archive changed before resume')
                if previous['key'] != row['key'] or previous['method'] != method:
                    raise ValueError('Existing result identity differs from scheduled case/method')
                for name in ('group_id', 'split', 'backend', 'layout', 'view'):
                    if previous[name] != row[name]:
                        raise ValueError('Existing result metadata differs from scheduled case')
                completed += 1
                continue
            started, metadata = time.perf_counter(), {}
            predictions = failed_predictions(queries)
            status, rgb_resampled = 'failed', False
            try:
                with deadline(lock['case_timeout_seconds']):
                    if method in load_errors:
                        metadata['execution_stage'] = 'dependency_or_model_load_failure'
                        raise RuntimeError('Model-load failure: '+load_errors[method])
                    if method.startswith('our-'):
                        if observation_error is not None:
                            raise RuntimeError('Shared detection already failed: '+observation_error)
                        if observations is None:
                            try:
                                observations = our.infer(image, case_folder,
                                    args.bank/'detections-B0'/(row['key']+'.npz'))
                            except Exception as exc:
                                observation_error = type(exc).__name__+': '+str(exc)
                                raise
                        xy, scores, topology, detection_meta = observations
                        cv2.setRNGSeed(1973)
                        geometry_start = time.perf_counter()
                        report = write_geometry_output(image, xy, scores, topology, folder,
                            pixels_per_mm=4., refinement_strength=0. if method == 'our-B0' else 1.)
                        metadata = {'detection': detection_meta, 'native_output': report,
                                    'geometry_and_RGB_seconds': time.perf_counter()-geometry_start}
                        mapping = OurNativeMap(folder, report)
                        rgb_resampled = mapping.rgb_resampled
                        with Image.open(folder/report['output_image']) as output_image:
                            if output_image.width*output_image.height > lock['RGB_output_budget_pixels']:
                                raise ValueError('Native RGB exceeds the frozen output pixel budget')
                        metadata['inference_and_RGB_seconds'] = (
                            detection_meta['fresh_detector_seconds']+detection_meta['topology_seconds']
                            +metadata['geometry_and_RGB_seconds'])
                        for domain in DOMAINS:
                            coords, eligible, footprint = mapping.predict(queries[domain]['source'])
                            predictions[domain] = {'coords': coords, 'claimed_eligible': eligible,
                                                   'native_footprint': footprint}
                        status = 'completed' if rgb_resampled else 'original_preserved_partial_coordinates'
                    elif method.startswith('npj-'):
                        mapping, rgb, metadata = competitors[method].infer(image)
                        output = np.clip(np.rint(rgb*255), 0, 255).astype(np.uint8)
                        if output.shape[0]*output.shape[1] > lock['RGB_output_budget_pixels']:
                            raise ValueError('Native RGB exceeds the frozen output pixel budget')
                        Image.fromarray(output).save(folder/'rectified-native.png')
                        np.savez_compressed(folder/'native_geometry.npz', source_to_mm=mapping.source_to_mm,
                                            source_to_pixels=mapping.source_to_pixels,
                                            mm_per_pixel=mapping.mm_per_pixel,
                                            source_shape=mapping.source_shape, output_shape=mapping.output_shape)
                        metadata['inference_and_RGB_seconds'] = time.perf_counter()-started
                        rgb_resampled = True
                        for domain in DOMAINS:
                            coords, eligible = mapping.predict(queries[domain]['source'])
                            predictions[domain] = {'coords': coords, 'claimed_eligible': eligible,
                                                   'native_footprint': eligible.copy()}
                        status = 'completed'
                    else:
                        mapping = competitors[method].infer(image, folder)
                        metadata = mapping.report
                        rgb_resampled = metadata['status'] == 'completed'
                        metadata['inference_and_RGB_seconds'] = time.perf_counter()-started
                        for domain in DOMAINS:
                            coords, eligible = mapping.forward(queries[domain]['source'])
                            predictions[domain] = {'coords': coords, 'claimed_eligible': eligible,
                                                   'native_footprint': eligible.copy()}
                        status = 'completed' if rgb_resampled else 'failed'
                        if str(metadata.get('error', '')).startswith('TimeoutError:'):
                            status = 'timeout'
            except Exception as exc:
                metadata = {**metadata, 'execution_error': type(exc).__name__+': '+str(exc)}
                predictions = failed_predictions(queries)
                status = 'timeout' if isinstance(exc, TimeoutError) else 'failed'
                rgb_resampled = False
            elapsed = time.perf_counter()-started
            scores = metric.score_predictions(predictions, queries)
            saved = {d+'_'+k: value for d, p in predictions.items() for k, value in p.items()}
            np.savez_compressed(folder/'query-predictions.npz', **saved)
            result = {k: row[k] for k in ('key', 'group_id', 'split', 'backend', 'layout', 'view')}
            result.update(method=method, status=status, rgb_resampled=rgb_resampled,
                          wall_seconds_including_forward_query_evaluation=elapsed,
                          case_budget_overrun_seconds=max(0., elapsed-lock['case_timeout_seconds']),
                          scores=scores, metadata=metadata,
                          input_archive_sha256=row['sha256'],
                          queries_archive_sha256=lock['query_sha256'][row['key']+'.npz'],
                          lock_sha256=digest(args.out/'lock.json'),
                          predictions_sha256=digest(folder/'query-predictions.npz'))
            write_json(folder/'result.json', result)
            completed += 1
            primary = scores['primary']['domains']['paper']
            print(json.dumps({'key': row['key'], 'method': method, 'status': status,
                              'paper_g': primary['g'], 'paper_c': primary['c'],
                              'seconds': elapsed, 'completed_this_invocation': completed}), flush=True)
            gc.collect()
        del image, observations
        gc.collect()
    verify_lock(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run'))
    for name in ('bank', 'checkpoint', 'metric-core', 'npj-adapter', 'npj-repo',
                 'kaggle-adapter', 'kaggle-repo', 'kaggle-assets', 'protocol', 'schedule', 'summarizer', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--method', choices=('all', *METHODS), default='all')
    parser.add_argument('--split', choices=('all', 'validation', 'test'), default='all')
    parser.add_argument('--limit', type=int, default=0,
                        help='Explicit feasibility-only prefix, never a complete-panel estimate')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    (freeze if args.command == 'freeze' else execute)(args)


if __name__ == '__main__':
    main()
