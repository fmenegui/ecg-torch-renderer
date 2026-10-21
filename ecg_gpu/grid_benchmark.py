"""Reproducible cached-render grid-detector experiment, outside the hot path.

Run ``python -m ecg_gpu.grid_benchmark --help``. Renderer comparisons use equal
source/view keys, initialization, crop sequence and optimizer steps. Online
denotes the torch image distribution, not on-the-fly timing in this experiment.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

from . import __version__
from .gpu_module import ECGImageLayer
from .grid_points import GridPointConfig, rasterize_grid_points
from .grid_point_learning import grid_point_loss, decode_grid_points
from .grid_evaluation import match_grid_points, summarize_grid_matches, rectification_diagnostic
from .scenarios import capture_scenario
from .schema_v05 import LEADS


class GridUNet(nn.Module):
    """Small full-resolution baseline with bounded nearest-pixel offsets."""
    def __init__(self, width=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.low = nn.Sequential(nn.Conv2d(width, 2*width, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(2*width, 2*width, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(3*width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, 3, 1))

    def forward(self, image):
        skip = self.encoder((image-.5)*2)
        low = self.low(F.max_pool2d(skip, 2))
        output = self.head(torch.cat((skip, F.interpolate(low, size=skip.shape[-2:], mode='bilinear', align_corners=False)), 1))
        return output[:, :1], .5*output[:, 1:].tanh()


@dataclass(frozen=True)
class BenchmarkConfig:
    height: int = 384
    width: int = 512
    views: int = 2
    steps: int = 300
    batch_size: int = 4
    crop_size: int = 192
    spp: int = 16
    seed: int = 1973
    sigma: float = 1.2
    mask_shape: str = 'gaussian'
    radius: float | None = None
    offsets: bool = True
    match_radius: float = 1.5
    device: str = 'cpu'

    def __post_init__(self):
        for key in ('height', 'width', 'views', 'steps', 'batch_size', 'crop_size', 'spp'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if min(self.height, self.width) < 64 or self.crop_size < 16 or self.crop_size > min(self.height, self.width):
            raise ValueError('Output dimensions >=64 and crop_size in [16,min(H,W)] required')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be an integer in [0,2^31)')
        if not isinstance(self.offsets, bool) or self.device not in ('cpu', 'cuda'):
            raise ValueError('Boolean offsets and cpu/cuda device required')
        if not math.isfinite(self.match_radius) or self.match_radius <= 0:
            raise ValueError('match_radius must be positive and finite')
        GridPointConfig(sigma_pixels=self.sigma, mask_shape=self.mask_shape, radius_pixels=self.radius)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_dependencies(*, rendering):
    required = ('scipy', 'mitsuba') if rendering else ('scipy',)
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing benchmark dependencies {missing}. Install ecg-torch-renderer[benchmark,pathtrace]")


def load_sources(path):
    """NPZ mV records with explicit patient/source IDs and preassigned splits.

    Require all twelve standard leads in order. Exact duplicate waveforms,
    sources and patient groups cannot cross partitions. Group correctness is
    still the curator's responsibility. No physiological augmentation is used.
    """
    with np.load(path, allow_pickle=False) as data:
        required = {'signals', 'fs', 'source_ids', 'group_ids', 'splits', 'lead_names', 'units'}
        if required-set(data.files):
            raise ValueError(f'Missing source fields: {sorted(required-set(data.files))}')
        signals = np.asarray(data['signals'], dtype=np.float32)
        fs = float(data['fs'])
        names = data['lead_names'].tolist()
        if names != list(LEADS) or str(data['units']) != 'mV':
            raise ValueError('Supply standard ordered lead_names and units=mV')
        ids = [data[key].tolist() for key in ('source_ids', 'group_ids', 'splits')]
    if signals.ndim != 3 or signals.shape[1] != 12 or signals.shape[2] < 2 or not np.isfinite(signals).all():
        raise ValueError('signals must be finite [N,12,T] physical mV')
    if not math.isfinite(fs) or fs <= 0 or any(len(value) != len(signals) for value in ids):
        raise ValueError('Positive fs and one ID/split per source required')
    if set(ids[2]) != {'train', 'validation', 'test'}:
        raise ValueError('Nonempty train, validation and test partitions required')
    if len(set(ids[0])) != len(signals):
        raise ValueError('source_ids must be unique')
    owners, fingerprints, records = {}, {}, []
    for signal, sid, gid, split in zip(signals, *ids):
        if any(not isinstance(value, str) or not value.strip() for value in (sid, gid, split)):
            raise ValueError('Nonempty string source/group IDs required')
        digest = hashlib.sha256(signal.astype('<f4').tobytes()).hexdigest()
        if owners.setdefault(gid, split) != split or fingerprints.setdefault(digest, split) != split:
            raise ValueError('Patient or waveform leakage across partitions')
        records.append({'signal': torch.from_numpy(signal.copy())[None], 'source_id': sid,
                        'group_id': gid, 'split': split, 'signal_sha256': digest})
    return records, fs


def _capture(record, view, index, fs, config, backend):
    seed = config.seed+index*1009+view*47
    scenarios = ('flatbed_scan', 'phone_photo', 'folded_photo', 'aged_scan')
    scenario = scenarios[(index+view) % len(scenarios)]
    render_config, parameters = capture_scenario(scenario, height=config.height, width=config.width, severity=.45)
    render_config = replace(render_config, surface_resolution=32, shadow_samples=0, antialias_samples=2)
    parameters.update(layout=('3x4+r1', '6x2', '12x1')[(index+view) % 3],
                      grid_phase_x=(seed % 997)/997*5, grid_phase_y=(seed % 991)/991*5)
    point_config = GridPointConfig(rows=64, columns=96)
    if backend == 'online':
        layer = ECGImageLayer(fs=fs, config=render_config, parameters=parameters, seed=seed)
        rendered = layer.render_batch(record['signal'], return_grid_points=True, grid_point_config=point_config)
        image, targets = rendered.image, rendered.targets
        metadata = {'scenario': scenario, 'render_config': asdict(render_config), 'seed': seed}
    else:
        from .pathtrace import PathTraceConfig, pathtrace_capture
        clean, printing = capture_scenario('clean_print', height=config.height*2, width=config.width*2)
        printing.update(layout=parameters['layout'], grid_phase_x=parameters['grid_phase_x'], grid_phase_y=parameters['grid_phase_y'])
        layer = ECGImageLayer(fs=fs, config=replace(clean, surface_resolution=32, shadow_samples=0), parameters=printing, seed=seed)
        canonical = layer.render_batch(record['signal'], photo=False, stage='all', as_texture=True, return_instances=True)
        surface = ('flat', 'flat', 'fold', 'curl')[(index+view) % 4]
        capture_config = PathTraceConfig(height=config.height, width=config.width, samples_per_pixel=config.spp,
            surface=surface, bend_radians=.25+.1*((index+view) % 4), geometry_seed=seed,
            camera_origin=(.025*((index % 3)-1), -.07-.015*view, .38+.01*(index % 3)),
            key_radiance=4.+.3*(index % 4), environment_radiance=.15, mesh_columns=65, mesh_rows=33)
        capture = pathtrace_capture(canonical, config=capture_config, seed=seed,
                                   return_grid_points=True, grid_point_config=point_config)
        image, targets = capture['image'], capture
        metadata = capture['metadata']
    accepted = targets['grid_points_valid'][0]
    return {'image': image.cpu(), 'valid': targets['valid'].cpu(),
            'xy': targets['grid_points_xy'][:, accepted].cpu(),
            'canonical': targets['grid_points_canonical_xy'][0, accepted].cpu(),
            'ids': targets['grid_points_ids'][0, accepted].cpu(),
            'backend': backend, 'view': view, 'scenario': scenario,
            'source_id': record['source_id'], 'group_id': record['group_id'], 'split': record['split'],
            'signal_sha256': record['signal_sha256'], 'metadata': metadata}


def prepare_captures(records, fs, out, config):
    """Generate both distributions once, keeping sampling outside train timing."""
    import drjit as dr
    dr.set_thread_count(2)
    result, manifest = [], []
    cache = Path(out)/'captures'
    cache.mkdir()
    for index, record in enumerate(records):
        for view in range(config.views):
            for backend in ('online', 'offline'):
                started = time.perf_counter()
                sample = _capture(record, view, index, fs, config, backend)
                key = f'{index:05d}-{view}-{backend}'
                sample['key'] = key
                torch.save(sample, cache/(key+'.pt'))
                result.append(sample)
                manifest.append({k: sample[k] for k in ('key', 'backend', 'view', 'source_id', 'group_id', 'split', 'signal_sha256', 'scenario')}
                                | {'file_sha256': _hash(cache/(key+'.pt')), 'generation_seconds': time.perf_counter()-started,
                                   'accepted_points': sample['xy'].shape[1]})
        print(json.dumps({'prepared_sources': index+1, 'sources': len(records)}), flush=True)
    (Path(out)/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return result


def _targets(samples, config):
    point_config = GridPointConfig(sigma_pixels=config.sigma, mask_shape=config.mask_shape,
                                  radius_pixels=config.radius, return_offsets=True)
    for sample in samples:
        sample['targets'] = rasterize_grid_points(sample['xy'], torch.ones(sample['xy'].shape[:2], dtype=torch.bool),
                                                  sample['valid'], config=point_config)


def _training_samples(samples, regime):
    train = [s for s in samples if s['split'] == 'train']
    if regime in ('online', 'offline'):
        return [s for s in train if s['backend'] == regime]
    keys = sorted({(s['source_id'], s['view']) for s in train})
    selection = {key: ('online' if i % 2 == 0 else 'offline') for i, key in enumerate(keys)}
    return [s for s in train if s['backend'] == selection[(s['source_id'], s['view'])]]


def _predict(model, sample, device, offsets):
    with torch.inference_mode():
        heat, delta = model(sample['image'].to(device))
    return heat.sigmoid().cpu(), delta.cpu() if offsets else None


def _decode(heat, delta, threshold):
    # Score every peak. A small fixed capacity can inflate precision by hiding
    # false positives and suppress recall when a page contains >2000 nodes.
    decoded = decode_grid_points(heat, delta, threshold=threshold, max_points=heat.shape[-2]*heat.shape[-1])
    return decoded['xy'][0, decoded['valid'][0]].numpy(), bool(decoded['truncated'][0])


def train_model(samples, regime, config):
    """Same initialization, crop RNG and number of examples for every regime."""
    torch.manual_seed(config.seed)
    model = GridUNet().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train = _training_samples(samples, regime)
    train.sort(key=lambda s: (s['source_id'], s['view']))
    rng = np.random.default_rng(config.seed)
    synchronize = torch.cuda.synchronize if config.device == 'cuda' else lambda: None
    synchronize()
    started = time.perf_counter()
    history = []
    for step in range(config.steps):
        images, batches = [], []
        for _ in range(config.batch_size):
            sample = train[int(rng.integers(len(train)))]
            y = int(rng.integers(config.height-config.crop_size+1))
            x = int(rng.integers(config.width-config.crop_size+1))
            images.append(sample['image'][..., y:y+config.crop_size, x:x+config.crop_size])
            batches.append({k: sample['targets'][k][..., y:y+config.crop_size, x:x+config.crop_size]
                            for k in ('grid_points_mask', 'grid_points_offsets', 'grid_points_offset_valid')})
        image = torch.cat(images).to(config.device)
        batch = {key: torch.cat([b[key] for b in batches]).to(config.device) for key in batches[0]}
        heat, delta = model(image)
        losses = grid_point_loss(heat, delta if config.offsets else None, batch, pos_weight=10.)
        optimizer.zero_grad(set_to_none=True)
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 0 or (step+1) % 50 == 0 or step+1 == config.steps:
            history.append({'step': step+1, 'loss': float(losses['loss'].detach())})
            if not math.isfinite(history[-1]['loss']):
                raise RuntimeError('Nonfinite training loss')
    synchronize()
    seconds = time.perf_counter()-started
    model.eval()
    choices, chosen = select_threshold(model, samples, config)
    return model, {'regime': regime, 'threshold': chosen['threshold'], 'validation': choices,
                   'training_images': len(train), 'training_steps': config.steps,
                   'training_examples_seen': config.steps*config.batch_size,
                   'training_seconds': seconds, 'history': history}


def select_threshold(model, samples, config):
    """Select one threshold from validation only, without a detection cap."""
    validation = [s for s in samples if s['split'] == 'validation']
    predictions = [(s, *_predict(model, s, config.device, config.offsets)) for s in validation]
    choices = []
    for threshold in (.15, .3, .5, .7, .85):
        matches = [match_grid_points(_decode(h, d, threshold)[0], s['xy'][0].numpy(), radius=config.match_radius)
                   for s, h, d in predictions]
        score = summarize_grid_matches(matches)
        choices.append({'threshold': threshold, **score})
    chosen = max(choices, key=lambda m: (m['f1'] or 0., -(m['median_error_px'] if m['median_error_px'] is not None else float('inf'))))
    return choices, chosen


def _overlay(sample, predicted, path):
    image = Image.fromarray((sample['image'][0].permute(1, 2, 0).numpy().clip(0, 1)*255).round().astype('uint8'))
    draw = ImageDraw.Draw(image)
    for x, y in sample['xy'][0].numpy():
        draw.ellipse((x-2, y-2, x+2, y+2), outline='#00b8d4')
    for x, y in predicted:
        draw.line((x-2, y, x+2, y), fill='#ff3854')
        draw.line((x, y-2, x, y+2), fill='#ff3854')
    image.save(path)


def evaluate_model(model, samples, config, threshold, out, name):
    rows, grouped = [], {}
    for sample in samples:
        if sample['split'] != 'test':
            continue
        heat, delta = _predict(model, sample, config.device, config.offsets)
        predicted, truncated = _decode(heat, delta, threshold)
        match = match_grid_points(predicted, sample['xy'][0].numpy(), radius=config.match_radius)
        grouped.setdefault(sample['backend'], []).append(match)
        row = {key: sample[key] for key in ('key', 'source_id', 'group_id', 'backend', 'scenario')}
        reference = match['reference_indices']
        diagnostic = rectification_diagnostic(predicted[match['prediction_indices']],
            sample['canonical'][reference].numpy(), sample['ids'][reference].numpy())
        row.update(summarize_grid_matches([match]), truncated=truncated, rectification=diagnostic)
        rows.append(row)
        # Export every test prediction independently of targets for later review.
        np.savez_compressed(Path(out)/(name+'-'+sample['key']+'-prediction.npz'), xy=predicted,
                            threshold=threshold, truncated=truncated)
        if sample['view'] == 0 and len(grouped[sample['backend']]) == 1:
            _overlay(sample, predicted, Path(out)/(name+'-'+sample['backend']+'-overlay.png'))
    return {'by_backend': {key: summarize_grid_matches(value) for key, value in grouped.items()}, 'per_image': rows}


def load_external(path, config):
    """JSON manifest with image paths/hashes and optional complete-ROI points.

    Unannotated images receive predictions only. Annotations use ORIGINAL
    pixel centers. A complete annotation region [x0,y0,x1,y1] limits scoring
    to its interior. Coordinates are transformed with pixel-center scaling.
    """
    manifest = json.loads(Path(path).read_text())
    samples = []
    for row in manifest:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['id']) or row['id'] in {s['key'] for s in samples}:
            raise ValueError('Unique safe external IDs required')
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ('source_id', 'group_id')):
            raise ValueError('External source/group IDs required')
        image_path = Path(path).parent/row['image']
        if _hash(image_path) != row['sha256']:
            raise ValueError('External image SHA256 mismatch')
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        ratio = min(config.width/original_width, config.height/original_height)
        w, h = round(original_width*ratio), round(original_height*ratio)
        resized = image.resize((w, h), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (config.width, config.height), 'white')
        canvas.paste(resized, (0, 0))
        scale = np.array([w/original_width, h/original_height])
        xy = np.asarray(row.get('points_xy', []), dtype=np.float32).reshape(-1, 2)
        if not np.isfinite(xy).all() or ((xy < 0) | (xy > [original_width-1, original_height-1])).any():
            raise ValueError('External points must lie inside the original image')
        original_xy = xy.copy()
        xy = (xy+.5)*scale-.5
        if len(xy) and ('annotation_region' not in row or not row.get('annotation_method')):
            raise ValueError('Annotated external images require a complete region and method')
        if len(xy):
            region = np.asarray(row['annotation_region'], dtype=float)
            if (region.shape != (4,) or not np.isfinite(region).all()
                    or (region[:2] >= region[2:]).any()
                    or (region[:2] < 0).any() or (region[2:] > [original_width-1, original_height-1]).any()
                    or ((original_xy < region[:2]) | (original_xy > region[2:])).any()):
                raise ValueError('Valid complete annotation region enclosing every reference point required')
        samples.append({'image': torch.from_numpy(np.array(canvas).copy()).permute(2, 0, 1)[None].float()/255,
            'xy': torch.from_numpy(xy.astype(np.float32))[None], 'metadata': row, 'scale': scale,
            'key': row['id'], 'split': 'external', 'source_id': row['source_id'], 'group_id': row['group_id']})
    return samples


def evaluate_external(model, external_samples, trial, threshold, out, name):
    """Inference-only external evaluation with validation-selected threshold."""
    real_results = []
    for sample in external_samples:
        heat, delta = _predict(model, sample, trial.device, trial.offsets)
        predicted, truncated = _decode(heat, delta, threshold)
        np.savez_compressed(out/(name+'-'+sample['key']+'-prediction.npz'), xy=predicted,
                            original_xy=(predicted+.5)/sample['scale']-.5)
        _overlay(sample, predicted, out/(name+'-'+sample['key']+'-overlay.png'))
        result = {'id': sample['key'], 'predictions': len(predicted), 'truncated': truncated,
                  'image_sha256': sample['metadata']['sha256'], 'metrics': None}
        if sample['xy'].shape[1]:
            region = np.asarray(sample['metadata']['annotation_region']).reshape(2, 2)
            region = (region+.5)*sample['scale']-.5
            inside = ((predicted >= region[0]) & (predicted <= region[1])).all(-1)
            result['metrics'] = summarize_grid_matches([match_grid_points(predicted[inside], sample['xy'][0].numpy(), radius=trial.match_radius)])
            result['annotation_method'] = sample['metadata']['annotation_method']
        real_results.append(result)
    return real_results


def run(signals, out, *, config=None, cache=None, external=None, ablations=False):
    config = BenchmarkConfig() if config is None else config
    if not isinstance(config, BenchmarkConfig):
        raise TypeError('config must be BenchmarkConfig')
    _require_dependencies(rendering=cache is None)
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    records, fs = load_sources(signals)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if cache is None:
        samples = prepare_captures(records, fs, out, config)
    else:
        previous = json.loads((Path(cache)/'report.json').read_text())
        if previous['signals_sha256'] != _hash(signals):
            raise ValueError('Cache has different source data')
        for key in ('height', 'width', 'views', 'spp', 'seed'):
            if previous['configuration'][key] != asdict(config)[key]:
                raise ValueError(f'Cache configuration differs: {key}')
        manifest = json.loads((Path(cache)/'manifest.json').read_text())
        samples = []
        for entry in manifest:
            path = Path(cache)/'captures'/(entry['key']+'.pt')
            if _hash(path) != entry['file_sha256']:
                raise ValueError('Capture cache SHA256 mismatch')
            samples.append(torch.load(path, weights_only=True))
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    external_samples = load_external(external, config) if external else []
    source_ids = {r['source_id'] for r in records}
    groups = {r['group_id'] for r in records}
    if any(s['source_id'] in source_ids or s['group_id'] in groups for s in external_samples):
        raise ValueError('External source/group overlaps synthetic source partitions')
    report = {'schema': 'ecg-grid-benchmark-v1', 'status': 'running', 'planned_models': 7 if ablations else 3,
        'package_version': __version__,
        'configuration': asdict(config), 'signals_sha256': _hash(signals), 'fs': fs,
        'lead_names': list(LEADS), 'source_records': len(records),
        'patient_groups': len(groups), 'source_code_sha256': _hash(__file__),
        'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__, 'python': platform.python_version(),
                    'device': config.device, 'cuda_measured': config.device == 'cuda', 'threads': 2},
        'protocol': {'sampling': 'both renderers cached before training',
                     'comparison': 'equal source/view keys, initialization, crop RNG and optimizer budget',
                     'threshold_selection': 'validation only, fixed grid of five thresholds',
                     'matching': 'maximum cardinality then minimum distance within radius',
                     'renderer_confounding': 'two recipe distributions, not causal isolation of the rendering algorithm',
                     'real_images_used_for_training_or_tuning': False,
                     'repeated_training_seeds': 1, 'study_level': 'small exploratory pilot, not a definitive ranking'},
        'models': [], 'external': {'images': len(external_samples), 'annotated_images': sum(bool(s['xy'].shape[1]) for s in external_samples)},
        'limitations': ['Single model seed and small cohort', 'No end-to-end waveform reconstruction or SNR result',
                        'Rectification diagnostic uses oracle correspondence, grid ordering remains unresolved',
                        'Point visibility is geometric, optical readability is not an input label']}
    settings = [('baseline', config)]
    if ablations:
        settings += [('sigma06', replace(config, sigma=.6)), ('sigma20', replace(config, sigma=2.)),
                     ('nooffset', replace(config, offsets=False)),
                     ('disk', replace(config, mask_shape='disk', radius=2.5))]
    for setting, trial in settings:
        _targets(samples, trial)
        for regime in (('online', 'offline', 'mixed') if setting == 'baseline' else ('mixed',)):
            name = setting+'-'+regime
            print(json.dumps({'training': name}), flush=True)
            model, training = train_model(samples, regime, trial)
            torch.save({'state_dict': model.cpu().state_dict(), 'configuration': asdict(trial),
                        'threshold': training['threshold'], 'signals_sha256': _hash(signals)}, out/(name+'.pt'))
            model.to(trial.device)
            evaluated = evaluate_model(model, samples, trial, training['threshold'], out, name)
            real_results = evaluate_external(model, external_samples, trial, training['threshold'], out, name)
            report['models'].append({'name': name, 'target': asdict(trial), **training, **evaluated, 'external': real_results})
            (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    report['status'] = 'completed'
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_metrics(report, out)
    return report


def write_metrics(report, out):
    """Export aggregate measurements without rounding their stored precision."""
    with (out/'metrics.csv').open('w', newline='') as stream:
        fields = ['model', 'backend', 'images', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'median_error_px', 'p95_error_px']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model in report['models']:
            for backend, metrics in model['by_backend'].items():
                writer.writerow({k: v for k, v in {'model': model['name'], 'backend': backend, **metrics}.items() if k in fields})


def rescore(previous, out, *, external=None):
    """Retune on validation and evaluate saved weights, without optimizer steps.

    Use the original capture directory. Capture hashes are checked. This permits
    correcting evaluation independently of training, with explicit provenance.
    """
    _require_dependencies(rendering=False)
    previous, out = Path(previous), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    report = json.loads((previous/'report.json').read_text())
    if report.get('status') == 'running':
        raise ValueError('Wait for the original experiment to complete before rescoring')
    report['status'] = 'running'
    manifest = json.loads((previous/'manifest.json').read_text())
    samples = []
    for entry in manifest:
        path = previous/'captures'/(entry['key']+'.pt')
        if _hash(path) != entry['file_sha256']:
            raise ValueError('Capture cache SHA256 mismatch')
        samples.append(torch.load(path, weights_only=True))
    config = BenchmarkConfig(**report['configuration'])
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for this experiment configuration')
    external_samples = load_external(external, config) if external else []
    if any(s['source_id'] in {r['source_id'] for r in samples} or s['group_id'] in {r['group_id'] for r in samples}
           for s in external_samples):
        raise ValueError('External source/group overlaps synthetic source partitions')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report['training_driver_sha256'] = report.get('training_driver_sha256', report['source_code_sha256'])
    report['source_code_sha256'] = _hash(__file__)
    report['training_package_version'] = report['package_version']
    report['package_version'] = __version__
    report['rescoring'] = {'previous_report_sha256': _hash(previous/'report.json'), 'optimizer_steps': 0,
                           'capacity': 'all H*W candidates, no peak truncation',
                           'reason': 'independent validation calibration and test evaluation of saved weights'}
    report['external'] = {'images': len(external_samples), 'annotated_images': sum(bool(s['xy'].shape[1]) for s in external_samples)}
    for model_report in report['models']:
        name = model_report['name']
        print(json.dumps({'rescoring': name}), flush=True)
        trial = BenchmarkConfig(**model_report['target'])
        checkpoint = torch.load(previous/(name+'.pt'), weights_only=True)
        if checkpoint['signals_sha256'] != report['signals_sha256'] or checkpoint['configuration'] != model_report['target']:
            raise ValueError('Checkpoint does not match experiment provenance')
        model = GridUNet().to(trial.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        choices, selected = select_threshold(model, samples, trial)
        model_report.update(threshold=selected['threshold'], validation=choices)
        model_report.update(evaluate_model(model, samples, trial, selected['threshold'], out, name))
        model_report['external'] = evaluate_external(model, external_samples, trial, selected['threshold'], out, name)
        checkpoint['threshold'] = selected['threshold']
        torch.save(checkpoint, out/(name+'.pt'))
        model_report['checkpoint_sha256'] = _hash(out/(name+'.pt'))
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(previous/'manifest.json', out/'manifest.json')
    report['status'] = 'completed'
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_metrics(report, out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--signals', type=Path)
    source.add_argument('--rescore', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--external', type=Path)
    parser.add_argument('--ablations', action='store_true')
    for key in ('height', 'width', 'views', 'steps', 'batch_size', 'crop_size', 'spp', 'seed'):
        parser.add_argument('--'+key.replace('_', '-'), type=int, default=getattr(BenchmarkConfig(), key))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    options = vars(parser.parse_args())
    previous = options.pop('rescore')
    if previous:
        if options['cache'] or options['ablations']:
            parser.error('--rescore does not use --cache or --ablations')
        rescore(previous, options['out'], external=options['external'])
        return
    names = ('signals', 'out', 'cache', 'external', 'ablations')
    arguments = {key: options.pop(key) for key in names}
    run(**arguments, config=BenchmarkConfig(**options))


if __name__ == '__main__':
    main()
