"""Locked GPU study: fresh patients, matched ablations and signal-level checks.

All test outputs are evaluated after model/threshold selection on validation.
The script is a research driver, not a clinical digitization application.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGImageLayer, capture_scenario
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.field_rectifier import GridFieldNet, chart_lattice, chart_loss
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.gpu_cli import plain
from ecg_gpu.grid_evaluation import match_grid_points, summarize_grid_matches
from ecg_gpu.grid_point_learning import decode_grid_points, grid_point_loss
from ecg_gpu.grid_points import GridPointConfig, rasterize_grid_points
from ecg_gpu.pathtrace import PathTraceConfig, pathtrace_capture
from examples.digitization_grid_rectification import grid_remap, save_rgb

VARIANTS = ('raw-point', 'contrast-point', 'contrast-global')
SEEDS = (1973, 20261021)
THRESHOLDS = (.3, .5, .7, .85, .93)


def progress(root, phase, **values):
    record = {'phase': phase, 'time': time.time(), **values}
    _write_json(root/'progress.json', record)
    print(json.dumps(record), flush=True)


def capture_dataset(records, fs, root, *, variant='cuda_ad_rgb'):
    """Store uint8 RGB and geometric supervision, without changing waveforms."""
    import drjit as dr
    dr.set_thread_count(6)
    directory = root/'captures'
    directory.mkdir(exist_ok=True)
    manifest = []
    for index, record in enumerate(records):
        for view in range(2):
            seed = 20261021+index*1009+view*47
            layout = ('3x4+r1', '6x2', '12x1')[(index+view) % 3]
            for backend in ('online', 'offline'):
                scenario = ('flatbed_scan', 'phone_photo', 'folded_photo', 'aged_scan')[(index+view) % 4]
                config, parameters = capture_scenario(scenario if backend == 'online' else 'clean_print',
                    width=768 if backend == 'online' else 1536,
                    height=512 if backend == 'online' else 1024, severity=.35+.1*view)
                parameters.update(layout=layout, speed_index=1, gain_index=1, font=0,
                    font_size=16. if backend == 'online' else 32.,
                    grid_phase_x=(seed % 997)/997*5, grid_phase_y=(seed % 991)/991*5)
                layer = ECGImageLayer(fs=fs, config=config, parameters=parameters, seed=seed,
                    randomize=False, photo=backend == 'online' and scenario != 'flatbed_scan').eval()
                signal = record['signal']
                with torch.no_grad():
                    if backend == 'online':
                        batch = layer.render_batch(signal, stage='all', return_grid_points=True,
                            grid_point_config=GridPointConfig(rows=64, columns=96))
                        image, targets, metadata = batch.image, batch.targets, plain(batch.metadata)
                        canonical_size = [512, 768]
                        capture_config = None
                    else:
                        batch = layer.render_batch(signal, stage='all', photo=False, as_texture=True, return_instances=True)
                        surface = ('flat', 'fold', 'curl', 'multifold')[(index+view) % 4]
                        capture_config = PathTraceConfig(width=768, height=512, samples_per_pixel=128,
                            surface=surface, bend_radians=.25+.15*view, geometry_seed=seed,
                            camera_origin=(.015*((index % 3)-1), -.065-.015*view, .36+.015*(index % 3)),
                            environment_radiance=.15, key_radiance=4., variant=variant,
                            missing_corner=.04 if index % 7 == 0 else 0., occluder=index % 11 == 0)
                        captured = pathtrace_capture(batch, config=capture_config, seed=seed,
                            return_grid_points=True, grid_point_config=GridPointConfig(rows=64, columns=96))
                        image, targets, metadata = captured['image'], captured, plain(batch.metadata)
                        canonical_size = list(batch.image.shape[-2:])
                visible = targets['grid_points_valid'][0].cpu().numpy()
                rgb = (image[0].permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8)
                key = f'{index:03d}-{view}-{backend}'
                path = directory/(key+'.npz')
                np.savez_compressed(path, image=rgb, valid=targets['valid'][0, 0].cpu().numpy().astype(np.float16),
                    xy=targets['grid_points_xy'][0].cpu().numpy()[visible],
                    ids=targets['grid_points_ids'][0].cpu().numpy()[visible].astype(np.int16),
                    canonical=targets['grid_points_canonical_xy'][0].cpu().numpy()[visible])
                manifest.append({'key': key, 'file': str(path.relative_to(root)), 'sha256': sha256_file(path),
                    'source_id': record['source_id'], 'group_id': record['group_id'], 'split': record['split'],
                    'source_index': index, 'view': view, 'backend': backend, 'seed': seed, 'layout': layout,
                    'render_config': asdict(config), 'capture_config': asdict(capture_config) if capture_config else None,
                    'paper_metadata': metadata, 'canonical_size': canonical_size,
                    'points': int(visible.sum())})
                _write_json(root/'capture-manifest.json', manifest)
                progress(root, 'rendering', completed=len(manifest), total=len(records)*4)
                del layer, batch, targets, image
                if backend == 'offline':
                    del captured
    return manifest


def load_samples(root, manifest):
    samples = []
    owners = {}
    for row in manifest:
        path = root/row['file']
        if sha256_file(path) != row['sha256']:
            raise ValueError('Capture bytes differ from locked manifest')
        if owners.setdefault(row['group_id'], row['split']) != row['split']:
            raise ValueError('Patient leakage')
        with np.load(path) as data:
            image = torch.from_numpy(data['image'].copy()).permute(2, 0, 1)[None].float()/255
            valid = torch.from_numpy(data['valid'].astype(np.float32))[None, None]
            xy = torch.from_numpy(data['xy'].copy())[None].float()
            ids = torch.from_numpy(data['ids'].copy()).float()
        target = rasterize_grid_points(xy, torch.ones(xy.shape[:2], dtype=torch.bool), valid,
            config=GridPointConfig(sigma_pixels=1., return_offsets=True))
        height, width = image.shape[-2:]
        samples.append(dict(row, image=image, visible=valid, xy=xy, ids=ids, target=target,
            chart_xy=(xy+.5)/torch.tensor([width, height])*2-1,
            chart_ids=ids.flip(-1)/torch.tensor([96., 64.])))
    return samples


def predict(model, sample, *, threshold=.3):
    with torch.inference_mode():
        output = model(sample['image'].cuda())
        prediction = decode_grid_points(output['heat'].sigmoid(), output['offsets'],
            threshold=threshold, max_points=sample['image'].shape[-2]*sample['image'].shape[-1])
        keep = prediction['valid'][0]
        xy = prediction['xy'][0, keep]
        scores = prediction['scores'][0, keep]
        height, width = sample['image'].shape[-2:]
        grid = (xy+.5)/xy.new_tensor([width, height])*2-1
        chart = F.grid_sample(output['chart'].float(), grid[None, None], align_corners=False)[0, :, 0].T
    return xy.cpu().numpy(), scores.cpu().numpy(), chart.cpu().numpy(), output['visible'].sigmoid().cpu()


def select_threshold(model, samples):
    predicted = [(s, *predict(model, s)[:2]) for s in samples if s['split'] == 'validation']
    choices = []
    for threshold in THRESHOLDS:
        grouped = {}
        for sample, xy, confidence in predicted:
            match = match_grid_points(xy[confidence >= threshold], sample['xy'][0].numpy(), radius=1.5)
            grouped.setdefault(sample['backend'], []).append(match)
        scores = {k: summarize_grid_matches(v) for k, v in grouped.items()}
        macro = float(np.mean([v['f1'] or 0. for v in scores.values()]))
        choices.append({'threshold': threshold, 'backend_macro_f1': macro, 'by_backend': scores})
    return max(choices, key=lambda x: x['backend_macro_f1']), choices


def train(root, samples, variant, seed, steps):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GridFieldNet(contrast=variant != 'raw-point').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    training = [s for s in samples if s['split'] == 'train']
    rng = np.random.default_rng(seed)
    history = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(steps):
        chosen = [training[int(rng.integers(len(training)))] for _ in range(2)]
        image = torch.cat([s['image'] for s in chosen]).cuda()
        brightness = .8+.4*torch.rand(2, 1, 1, 1, device='cuda')
        image = (image*brightness).clamp(0, 1)
        target = {key: torch.cat([s['target'][key] for s in chosen]).cuda()
                  for key in ('grid_points_mask', 'grid_points_offsets', 'grid_points_offset_valid')}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(image)
        loss = grid_point_loss(output['heat'], output['offsets'], target, pos_weight=10.)['loss']
        visibility = F.binary_cross_entropy_with_logits(output['visible'].float(), torch.cat([s['visible'] for s in chosen]).cuda())
        auxiliary = loss*0
        if variant == 'contrast-global':
            for i, sample in enumerate(chosen):
                present = torch.ones(sample['chart_xy'].shape[:2], device='cuda')
                auxiliary = auxiliary+chart_loss(output['chart'][i:i+1], sample['chart_xy'].cuda(),
                    sample['chart_ids'][None].cuda(), present)/2
        total = loss+.1*visibility+10*auxiliary
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 0 or (step+1) % 100 == 0:
            value = float(total.detach())
            if not np.isfinite(value):
                raise RuntimeError('Nonfinite GPU training loss')
            history.append({'step': step+1, 'total_loss': value, 'point_loss': float(loss.detach()),
                            'chart_loss': float(auxiliary.detach())})
            progress(root, 'training', variant=variant, seed=seed, step=step+1, steps=steps, loss=value)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    model.eval()
    selected, calibration = select_threshold(model, samples)
    directory = root/'models'
    directory.mkdir(exist_ok=True)
    name = variant+'-'+str(seed)
    path = directory/(name+'.pt')
    torch.save({'state_dict': model.cpu().state_dict(), 'variant': variant, 'seed': seed,
                'threshold': selected['threshold'], 'steps': steps}, path)
    result = {'name': name, 'variant': variant, 'seed': seed, 'steps': steps,
              'training_groups': len({s['group_id'] for s in training}),
              'training_images': len(training), 'seconds': seconds, 'peak_gpu_bytes': peak,
              'validation': selected, 'calibration': calibration, 'history': history,
              'checkpoint': str(path.relative_to(root)), 'checkpoint_sha256': sha256_file(path)}
    _write_json(directory/(name+'.json'), result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def test_models(root, samples, trials):
    """Read the untouched test split only after every model is frozen."""
    results = []
    for trial in trials:
        checkpoint = torch.load(root/trial['checkpoint'], map_location='cpu', weights_only=True)
        model = GridFieldNet(contrast=trial['variant'] != 'raw-point').cuda().eval()
        model.load_state_dict(checkpoint['state_dict'])
        grouped, cases = {}, []
        for sample in samples:
            if sample['split'] != 'test':
                continue
            xy, scores, chart, _ = predict(model, sample, threshold=trial['validation']['threshold'])
            match = match_grid_points(xy, sample['xy'][0].numpy(), radius=1.5)
            grouped.setdefault(sample['backend'], []).append(match)
            topology = chart_lattice(xy, scores, chart) if trial['variant'] == 'contrast-global' else organize_lattice(xy, scores)
            identity = None
            if topology['status'] == 'accepted':
                ids = np.asarray(topology['ids'])
                selected = np.asarray(topology['selected_indices'])
                paired = match_grid_points(xy[selected], sample['xy'][0].numpy(), radius=1.5)
                inferred = ids[paired['prediction_indices']]
                truth = sample['ids'].numpy()[paired['reference_indices']]
                best = 0
                for swap in (False, True):
                    p = inferred[:, ::-1] if swap else inferred
                    for sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        if len(p):
                            _, counts = np.unique(truth-p*sign, axis=0, return_counts=True)
                            best = max(best, int(counts.max()))
                identity = {'matched_nodes': len(inferred), 'consistent_nodes': best,
                            'fraction': best/len(inferred) if len(inferred) else None}
            cases.append({'key': sample['key'], 'group_id': sample['group_id'], 'backend': sample['backend'],
                'detection': summarize_grid_matches([match]), 'topology_status': topology['status'],
                'cells': len(topology['cells']), 'identity': identity})
        results.append({'name': trial['name'], 'by_backend': {k: summarize_grid_matches(v) for k, v in grouped.items()}, 'cases': cases})
        _write_json(root/'test-results.json', results)
        progress(root, 'test-evaluation', completed=len(results), total=len(trials))
        del model
        torch.cuda.empty_cache()
    return results


def prepare_pipeline_inputs(root, samples, records, fs):
    """Each patient has its own truth signal and matched original/oracle inputs."""
    from scipy.signal import resample_poly
    pairs = []
    for index, record in enumerate(records):
        if record['split'] != 'test':
            continue
        original = root/'signal-inputs'/record['source_id']/'original'
        oracle = root/'signal-inputs'/record['source_id']/'oracle'
        original.mkdir(parents=True)
        oracle.mkdir(parents=True)
        values = resample_poly(record['signal'][0].numpy(), 500, int(fs), axis=-1).astype(np.float32)
        np.savez_compressed(original/'source.npz', values_mv=values, fs=500)
        (oracle/'source.npz').write_bytes((original/'source.npz').read_bytes())
        manifest = {'source_sha256': sha256_file(original/'source.npz'), 'cases': []}
        oracle_manifest = dict(manifest, cases=[])
        for sample in samples:
            if sample['source_id'] != record['source_id'] or sample['view'] != 0:
                continue
            with np.load(root/sample['file']) as data:
                rgb, xy, canonical, ids = data['image'], data['xy'], data['canonical'], data['ids']
            name = sample['backend']
            image = name+'.png'
            digest = save_rgb(original/image, rgb)
            entry = {'name': name, 'image': image, 'image_sha256': digest,
                     'layout': sample['layout'], 'metadata': sample['paper_metadata']}
            manifest['cases'].append(entry)
            warped, _support = grid_remap(rgb, xy, canonical, ids, tuple(sample['canonical_size']))
            oracle_entry = dict(entry, image_sha256=save_rgb(oracle/image, warped))
            oracle_manifest['cases'].append(oracle_entry)
        _write_json(original/'manifest.json', manifest)
        _write_json(oracle/'manifest.json', oracle_manifest)
        pairs.append({'source_id': record['source_id'], 'group_id': record['group_id'],
                      'original': str(original.relative_to(root)), 'oracle': str(oracle.relative_to(root))})
    _write_json(root/'signal-inputs.json', pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signals', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--render-variant', default='cuda_ad_rgb')
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('This consolidated run requires a real CUDA GPU')
    from ecg_gpu.grid_benchmark import load_sources
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, fs = load_sources(args.signals)
    protocol = {'variants': VARIANTS, 'seeds': SEEDS, 'steps': args.steps, 'batch_size': 2,
        'image_size': [512, 768], 'physical_spp': 128, 'thresholds': THRESHOLDS,
        'selection': 'validation macro F1 across rendering backends, test read after all models freeze',
        'data_sha256': sha256_file(args.signals), 'gpu': torch.cuda.get_device_name(),
        'torch': str(torch.__version__), 'precision': 'bfloat16 forward, float32 losses',
        'render_variant': args.render_variant, 'real_image_validation': False}
    _write_json(root/'locked-protocol.json', protocol)
    manifest = capture_dataset(records, fs, root, variant=args.render_variant)
    samples = load_samples(root, manifest)
    trials = []
    for variant in VARIANTS:
        for seed in SEEDS:
            trials.append(train(root, samples, variant, seed, args.steps))
            _write_json(root/'trials.json', trials)
    winner = max(trials, key=lambda t: t['validation']['backend_macro_f1'])
    _write_json(root/'selected-model.json', winner)
    test_models(root, samples, trials)
    prepare_pipeline_inputs(root, samples, records, fs)
    progress(root, 'detector-study-completed', winner=winner['name'])


if __name__ == '__main__':
    main()
