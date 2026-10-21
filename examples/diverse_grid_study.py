"""Offline diverse capture bank, multitask training and locked evaluations.

All corruptions and clean targets are cached before training. Patient splits
are fixed by the source cohort. Real photographs are never training inputs.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGImageLayer, capture_scenario
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.denoising_grid import DenoisingGridNet, restoration_loss
from ecg_gpu.digitization.field_rectifier import GridFieldNet
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.local_cells import locally_supported_cells
from ecg_gpu.digitization.hybrid_rectifier import forward_supported, similarity_residual
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.grid_evaluation import match_grid_points, summarize_grid_matches
from ecg_gpu.grid_point_learning import decode_grid_points, grid_point_loss
from ecg_gpu.grid_points import GridPointConfig, rasterize_grid_points
from ecg_gpu.offline_noise import FAMILIES, corrupt_observation
from ecg_gpu.pathtrace import PathTraceConfig, pathtrace_capture
from ecg_gpu.scenarios import CAPTURE_SCENARIOS
from ecg_gpu.schema_v05 import LAYOUTS
from examples.digitization_grid_rectification import grid_remap

TRAIN_NOISE = FAMILIES[:10]
OOD_NOISE = FAMILIES[10:]
WIDTH, HEIGHT, VIEWS = 1024, 704, 8
SEEDS = (1973, 20261021)
THRESHOLDS = (.3, .5, .7, .85, .93)


def progress(root, phase, **values):
    report = {'phase': phase, 'time': time.time(), **values}
    _write_json(root/'progress.json', report)
    print(json.dumps(report), flush=True)


def uint8(image):
    return (image[0].detach().cpu().permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)


def clean_parameters(parameters):
    """Remove observation damage while retaining exact print geometry/content."""
    result = dict(parameters)
    for key in ('paper_age', 'coffee_strength', 'stain_strength', 'bleed_strength', 'thermal_fade',
                'fiber_strength', 'foxing_strength', 'water_strength', 'thermal_darkening_strength',
                'edge_wear_strength', 'band_strength', 'pen_active', 'artifact_present',
                'angle', 'quarter', 'translate_x', 'translate_y', 'perspective_x', 'perspective_y',
                'radial', 'curl_x', 'curl_y', 'crease_warp', 'surface_strength', 'asset_active'):
        result[key] = 0.
    result.update(light_x=0., light_y=0., vignette=0., shadow_strength=0., paper_noise=0.,
                  tear_active=0., cut_active=0., hole_active=0., missing_corner_active=0., sensor_active=0., jpeg_active=0., read_noise=0., shot_noise=0., blur=0., motion=0.,
                  exposure=1., white_balance=[1., 1., 1.], gamma=1., zoom=1., paper_color=[1., 1., 1.], paper_palette=0, paper_tint=0.,
                  ink_density=1., ink_color=[.035, .035, .035])
    return result


def render_one(signal, fs, index, view, split, *, variant='cuda_ad_rgb'):
    seed = 20261022+index*1009+view*47
    rng = np.random.default_rng(seed)
    offline = view in (2, 6)
    scenario = CAPTURE_SCENARIOS[(index*5+view*3) % len(CAPTURE_SCENARIOS)]
    layout = LAYOUTS[1:][(index+view*7) % (len(LAYOUTS)-1)]
    severity = float(rng.uniform(.15, .85))
    cw, ch = (1536, 1056) if offline else (WIDTH, HEIGHT)
    config, parameters = capture_scenario('clean_print' if offline else scenario, width=cw, height=ch, severity=severity)
    config = replace(config, max_font_size=48. if offline else 32.)
    parameters.update(layout=layout, speed_index=int(rng.integers(3)), gain_index=int(rng.integers(3)),
                      font=int(rng.integers(2)), font_size=float(rng.uniform(16, 24))*(1.5 if offline else 1),
                      grid_phase_x=float(rng.uniform(0, 5)), grid_phase_y=float(rng.uniform(0, 5)),
                      grid_strength=float(rng.uniform(.18, .65)), minor_strength=float(rng.uniform(.15, .5)),
                      grid_style=0, line_width=float(rng.uniform(.7, 1.6)),
                      header_present=float(rng.random() > .15), label_below=bool(rng.random() > .7))
    if offline:
        parameters.update(paper_age=float(rng.uniform(0, .2)), coffee_strength=float(rng.uniform(0, .2)))
    layer = ECGImageLayer(fs=fs, config=config, parameters=parameters, seed=seed,
                          randomize=False, photo=not offline).to(signal.device).eval()
    with torch.no_grad():
        batch = layer.render_batch(signal, return_instances=offline, return_field=not offline,
                                   return_grid_points=not offline,
                                   grid_point_config=GridPointConfig(rows=128, columns=192),
                                   as_texture=offline, photo=not offline)
        clean_config, _ = capture_scenario('clean_print', width=cw, height=ch, severity=0.)
        clean_config = replace(clean_config, max_font_size=48. if offline else 32.)
        clean_layer = ECGImageLayer(fs=fs, config=clean_config, parameters=parameters,
                                   seed=seed, randomize=False, photo=False).to(signal.device).eval()
        clean_batch = clean_layer.render_batch(signal, parameters=clean_parameters(batch.metadata['parameters']),
                                              photo=False, stage='image', as_texture=offline)
        if not torch.allclose(batch.metadata['ppm'], clean_batch.metadata['ppm']):
            raise ValueError('Paired clean target changed physical calibration')
        if offline:
            surfaces = ('flat', 'curl', 'fold', 'multifold', 'wrinkled', 'crumpled')
            capture = PathTraceConfig(width=WIDTH, height=HEIGHT, samples_per_pixel=64,
                label_samples_per_axis=1, surface=surfaces[(index+view//4) % len(surfaces)],
                bend_radians=float(rng.uniform(.1, .85)), geometry_seed=seed,
                camera_origin=(float(rng.uniform(-.04, .04)), float(rng.uniform(-.12, -.03)), float(rng.uniform(.34, .43))),
                key_radiance=float(rng.uniform(1.5, 6.)), environment_radiance=float(rng.uniform(.04, .3)),
                key_origin=(float(rng.uniform(-.25, .2)), -.18, .4),
                missing_corner=.08 if index % 11 == 0 else 0., edge_tear=.04 if index % 13 == 0 else 0.,
                hole_radius=.03 if index % 17 == 0 else 0., occluder=index % 19 == 0, variant=variant)
            result = pathtrace_capture(batch, config=capture, seed=seed, return_geometry=True,
                return_grid_points=True, grid_point_config=GridPointConfig(rows=128, columns=192))
            image, valid = result['image'], result['valid']
            field = result['geometry']['inverse_field']
            clean = F.grid_sample(clean_batch.image.cpu(), field, align_corners=False, padding_mode='border')
            metadata = {'capture': asdict(capture), 'surface': capture.surface}
        else:
            result = batch.targets
            image, valid = batch.image, result['valid']
            clean = F.grid_sample(clean_batch.image, batch.inverse_field, align_corners=False, padding_mode='border')
            metadata = {'scenario': scenario, 'render': asdict(config)}
        active = result['grid_points_valid'][0].cpu().numpy()
        xy = result['grid_points_xy'][0].cpu().numpy()[active]
        ids = result['grid_points_ids'][0].cpu().numpy()[active]
        support = valid[0, 0].cpu().numpy().astype(np.float32)
        noisy = uint8(image)
        clean_rgb = uint8(clean)
        # A source-independent noise assignment covers every family and severity.
        family = 'none' if view == 0 else TRAIN_NOISE[(index*7+view) % len(TRAIN_NOISE)]
        noisy, support, noise = corrupt_observation(noisy, support, family=family, severity=severity, seed=seed+23)
        quarter = 0 if view < 4 else (index+view) % 4
        old_h, old_w = noisy.shape[:2]
        for _ in range(quarter):
            xy = np.column_stack((xy[:, 1], old_w-1-xy[:, 0]))
            old_h, old_w = old_w, old_h
        noisy = np.rot90(noisy, quarter).copy()
        clean_rgb = np.rot90(clean_rgb, quarter).copy()
        support = np.rot90(support, quarter).copy()
        ratio = min(WIDTH/old_w, HEIGHT/old_h)
        new_w, new_h = round(old_w*ratio), round(old_h*ratio)
        if (new_h, new_w) != (HEIGHT, WIDTH):
            canvas = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
            target_canvas = canvas.copy()
            mask_canvas = np.zeros((HEIGHT, WIDTH), np.float32)
            canvas[:new_h, :new_w] = np.asarray(Image.fromarray(noisy).resize((new_w, new_h), Image.Resampling.BILINEAR))
            target_canvas[:new_h, :new_w] = np.asarray(Image.fromarray(clean_rgb).resize((new_w, new_h), Image.Resampling.BILINEAR))
            mask_canvas[:new_h, :new_w] = np.asarray(Image.fromarray(support).resize((new_w, new_h), Image.Resampling.BILINEAR))
            noisy, clean_rgb, support = canvas, target_canvas, mask_canvas
            xy = (xy+.5)*[new_w/old_w, new_h/old_h]-.5
        metadata['post_rotation_quarters'] = quarter
        metadata['letterbox_size'] = [new_w, new_h]
        if len(xy):
            pixel = np.round(xy).astype(int)
            keep = support[pixel[:, 1].clip(0, HEIGHT-1), pixel[:, 0].clip(0, WIDTH-1)] >= .99
            xy, ids = xy[keep], ids[keep]
    return {'image': noisy, 'clean': clean_rgb, 'valid': support, 'xy': xy, 'ids': ids,
            'metadata': {'seed': seed, 'backend': 'offline' if offline else 'torch-cached', 'layout': layout,
                         'severity': severity, 'noise': noise, **metadata}}


def render_bank(cohort, root):
    import drjit as dr
    dr.set_thread_count(6)
    with np.load(cohort/'signals.npz') as data:
        signals = data['signals'].copy()
        fs = float(data['fs'])
        source_ids, groups, splits = data['source_ids'].copy(), data['group_ids'].copy(), data['splits'].copy()
    source = json.loads((cohort/'sources.json').read_text())
    if sha256_file(cohort/'signals.npz') != source['signals_sha256']:
        raise ValueError('Cohort checksum mismatch')
    owners = {}
    for patient, split in zip(groups, splits):
        if owners.setdefault(str(patient), str(split)) != str(split):
            raise ValueError('Patient leakage')
    folder = root/'bank'
    folder.mkdir(exist_ok=True)
    manifest_path = root/'bank-manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    journal = root/'bank-journal.jsonl'
    if journal.exists():
        known = {r['key'] for r in manifest}
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            if r['key'] not in known:
                manifest.append(r)
                known.add(r['key'])
    cached = {r['key']: r for r in manifest}
    expected = len(signals)*VIEWS
    for index in range(len(signals)):
        signal = torch.from_numpy(signals[index:index+1]).cuda()
        for view in range(VIEWS):
            key = f'{index:05d}-{view}'
            if key in cached:
                if sha256_file(root/cached[key]['file']) != cached[key]['sha256']:
                    raise ValueError('Cached capture checksum mismatch')
                continue
            capture = render_one(signal, fs, index, view, str(splits[index]))
            path = folder/(key+'.npz')
            np.savez_compressed(path, image=capture['image'], clean=capture['clean'],
                valid=capture['valid'].astype(np.float16), xy=capture['xy'].astype(np.float32), ids=capture['ids'].astype(np.int16))
            row = {'key': key, 'file': str(path.relative_to(root)), 'sha256': sha256_file(path),
                   'source_id': str(source_ids[index]), 'group_id': str(groups[index]), 'split': str(splits[index]),
                   'source_index': index, 'view': view, 'points': len(capture['xy']), **capture['metadata']}
            manifest.append(row)
            with journal.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if len(manifest) % 64 == 0 or len(manifest) == expected:
                _write_json(manifest_path, manifest)
            progress(root, 'offline-bank', completed=len(manifest), total=expected)
        del signal
    if len(manifest) != expected:
        raise ValueError('Incomplete bank')
    return manifest


class CachedCrops(Dataset):
    """CPU worker I/O, deterministic sampling. Supervision stays on CUDA."""
    def __init__(self, root, rows, *, seed, count, crop=384):
        self.root, self.rows, self.seed, self.count, self.crop = root, rows, seed, count, crop

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed+index*997)
        row = self.rows[int(rng.integers(len(self.rows)))]
        with np.load(self.root/row['file']) as d:
            h, w = d['image'].shape[:2]
            y, x = int(rng.integers(h-self.crop+1)), int(rng.integers(w-self.crop+1))
            image, clean, valid = d['image'][y:y+self.crop, x:x+self.crop].copy(), d['clean'][y:y+self.crop, x:x+self.crop].copy(), d['valid'][y:y+self.crop, x:x+self.crop].copy()
            xy = d['xy'].copy()-[x, y]
        keep = ((xy >= 0) & (xy <= self.crop-1)).all(1)
        return image, clean, valid, xy[keep].astype(np.float32)


def collate(rows):
    images = torch.from_numpy(np.stack([r[0] for r in rows])).permute(0, 3, 1, 2)
    clean = torch.from_numpy(np.stack([r[1] for r in rows])).permute(0, 3, 1, 2)
    valid = torch.from_numpy(np.stack([r[2] for r in rows]).astype(np.float32))[:, None]
    n = max(1, max(len(r[3]) for r in rows))
    xy = torch.zeros(len(rows), n, 2)
    present = torch.zeros(len(rows), n, dtype=torch.bool)
    for i, r in enumerate(rows):
        xy[i, :len(r[3])], present[i, :len(r[3])] = torch.from_numpy(r[3]), True
    return images, clean, valid, xy, present


def detect(model, rgb, device='cuda', threshold=.3):
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device)/255
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        output = model(tensor)
    pred = decode_grid_points(output['heat'].float().sigmoid(), output['offsets'].float(),
                              threshold=threshold, max_points=16384)
    keep = pred['valid'][0]
    return pred['xy'][0, keep].cpu().numpy(), pred['scores'][0, keep].cpu().numpy(), output


def selection(model, rows, root):
    matches = {t: {} for t in THRESHOLDS}
    for row in rows:
        with np.load(root/row['file']) as d:
            rgb, truth = d['image'].copy(), d['xy'].copy()
        xy, score, _ = detect(model, rgb)
        for t in THRESHOLDS:
            matches[t].setdefault(row['backend'], []).append(match_grid_points(xy[score >= t], truth, radius=2.))
    candidates = []
    for t in THRESHOLDS:
        metrics = {k: summarize_grid_matches(v) for k, v in matches[t].items()}
        candidates.append({'threshold': t, 'backend_macro_f1': float(np.mean([v['f1'] or 0 for v in metrics.values()])), 'by_backend': metrics})
    return max(candidates, key=lambda x: x['backend_macro_f1']), candidates


def train(root, manifest, *, joint, seed, steps, batch_size=4, resume=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DenoisingGridNet().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    rows = [r for r in manifest if r['split'] == 'train']
    ds = CachedCrops(root, rows, seed=seed, count=steps*batch_size)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, collate_fn=collate, pin_memory=True)
    name = f'{"joint" if joint else "point"}-{seed}'
    folder = root/'models'
    folder.mkdir(exist_ok=True)
    history = []
    resumed_steps = 0
    signature = {'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                 'seed': seed, 'joint': joint, 'steps': steps, 'batch_size': batch_size,
                 'crop': 384, 'width': 24, 'learning_rate': .001}
    recovery = folder/(name+'-recovery.pt')
    if resume and recovery.exists():
        saved = torch.load(recovery, map_location='cpu', weights_only=True)
        if saved.get('training_signature') != signature:
            raise ValueError('Recovery checkpoint does not match this bank and training protocol')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        resumed_steps = int(saved['steps'])
        if not 0 <= resumed_steps <= steps:
            raise ValueError('Invalid recovery step')
        history_path = folder/(name+'-history.json')
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        subset = torch.utils.data.Subset(ds, range(resumed_steps*batch_size, len(ds)))
        loader = DataLoader(subset, batch_size=batch_size, num_workers=2, collate_fn=collate, pin_memory=True)
        progress(root, 'training-resumed', name=name, step=resumed_steps, steps=steps)
    started = time.time()
    for step, (images, clean, valid, xy, present) in enumerate(loader, resumed_steps+1):
        image, reference = images.cuda(non_blocking=True).float()/255, clean.cuda(non_blocking=True).float()/255
        valid, xy, present = valid.cuda(non_blocking=True), xy.cuda(non_blocking=True), present.cuda(non_blocking=True)
        target = rasterize_grid_points(xy, present, valid, config=GridPointConfig(sigma_pixels=1., return_offsets=True))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(image)
        points = grid_point_loss(output['heat'].float(), output['offsets'].float(), target, pos_weight=10.)['loss']
        restoration = restoration_loss(output['clean'], reference, valid)
        loss = points+.1*F.binary_cross_entropy_with_logits(output['visible'].float(), valid)
        if joint:
            loss = loss+.2*restoration['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 1 or step % 1000 == 0 or step == steps:
            history.append({'step': step, 'loss': float(loss.detach()), 'point_loss': float(points.detach()),
                            'restoration_loss': float(restoration['loss'].detach()), 'elapsed_seconds': time.time()-started})
            temporary = recovery.with_suffix('.pt.part')
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'seed': seed, 'joint': joint, 'steps': step, 'threshold': None, 'width': 24,
                        'model_class': 'DenoisingGridNet', 'training_sampling': 'index-deterministic',
                        'training_signature': signature}, temporary)
            temporary.replace(recovery)
            _write_json(folder/(name+'-history.json'), history)
            progress(root, 'training', name=name, step=step, steps=steps)
    model.eval()
    best, choices = selection(model, [r for r in manifest if r['split'] == 'validation'], root)
    path = folder/(name+'.pt')
    torch.save({'state_dict': model.cpu().state_dict(), 'seed': seed, 'joint': joint, 'steps': steps,
                'threshold': best['threshold'], 'width': 24, 'model_class': 'DenoisingGridNet'}, path)
    return {'name': name, 'checkpoint': str(path.relative_to(root)), 'sha256': sha256_file(path),
            'joint': joint, 'seed': seed, 'steps': steps, 'validation': best, 'threshold_candidates': choices}


def evaluate(root, manifest, trial):
    state = torch.load(root/trial['checkpoint'], map_location='cpu', weights_only=True)
    model = DenoisingGridNet(width=state['width']).cuda().eval()
    model.load_state_dict(state['state_dict'])
    threshold = float(state['threshold'])
    cases, matches, ood = [], {}, {}
    selected = [r for r in manifest if r['split'] == 'test']
    predictions = root/'test-predictions'/trial['name']
    predictions.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(selected):
        with np.load(root/row['file']) as d:
            rgb, clean, valid, truth = d['image'].copy(), d['clean'].copy(), d['valid'].astype(np.float32), d['xy'].copy()
            truth_ids = d['ids'].copy()
        xy, score, output = detect(model, rgb, threshold=threshold)
        metric = match_grid_points(xy, truth, radius=2.)
        matches.setdefault(row['backend'], []).append(metric)
        topology = organize_lattice(xy, score)
        local = locally_supported_cells(xy, topology)
        geometry = {'point_support_fraction': 0., 'p95_mm': None}
        identity = None
        if local['partial_status'] == 'partial':
            ids = np.asarray(local['ids'])
            chosen = np.asarray(local['selected_indices'])
            mapped = forward_supported(xy[chosen], ids[:, ::-1]*5., ids, truth, local['cells'])
            geometry = similarity_residual(mapped, truth_ids[:, ::-1]*5.)
            paired = match_grid_points(xy[chosen], truth, radius=2.)
            inferred, actual = ids[paired['prediction_indices']], truth_ids[paired['reference_indices']]
            best = 0
            for swap in (False, True):
                p = inferred[:, ::-1] if swap else inferred
                for sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    if len(p):
                        _, counts = np.unique(actual-p*sign, axis=0, return_counts=True)
                        best = max(best, int(counts.max()))
            identity = {'matched_nodes': len(inferred), 'consistent_nodes': best,
                        'fraction': best/len(inferred) if len(inferred) else None}
        restored = output['clean'][0].float().cpu().permute(1, 2, 0).numpy()
        mask = valid >= .99
        before = float(np.mean((rgb[mask]/255-clean[mask]/255)**2)) if mask.any() else None
        after = float(np.mean((restored[mask]-clean[mask]/255)**2)) if mask.any() else None
        case = {'key': row['key'], 'group_id': row['group_id'], 'backend': row['backend'],
                'layout': row['layout'], 'noise': row['noise'], 'points': summarize_grid_matches([metric]),
                'topology_status': topology['status'], 'cells': len(topology['cells']),
                'local_status': local['partial_status'], 'local_cells': len(local['cells']),
                'local_geometry': geometry, 'relative_identity': identity,
                'restoration_visible_mse_before': before, 'restoration_visible_mse_after': after,
                'restoration_is_waveform_accuracy': False}
        # Save every test detection, not only selected gallery successes.
        np.savez_compressed(predictions/(row['key']+'.npz'), xy=xy, scores=score,
                            selected=np.asarray(topology['selected_indices']), ids=np.asarray(topology['ids']), cells=np.asarray(local['cells']))
        if i < 32:
            Image.fromarray((restored*255).round().clip(0, 255).astype(np.uint8)).save(predictions/(row['key']+'-restored.png'))
            if local['partial_status'] == 'partial':
                ids = np.asarray(topology['ids'])
                canonical = ids[:, ::-1]*20.
                size = (canonical.max(0)+1).astype(int)[::-1]
                if max(size) <= 4096:
                    warp, support = grid_remap(rgb, xy[topology['selected_indices']], canonical, ids,
                                              tuple(size), allowed_cells=local['cells'])
                    Image.fromarray(warp).save(predictions/(row['key']+'-rectified.png'))
                    np.savez_compressed(predictions/(row['key']+'-support.npz'), valid=support)
        # Two held-out corruption families. They never enter selection or training.
        if row['view'] == 0:
            for family in OOD_NOISE:
                damaged, support, _ = corrupt_observation(clean, valid, family=family, severity=.7, seed=row['seed']+811)
                ox, oscore, _ = detect(model, damaged, threshold=threshold)
                ood.setdefault(family, []).append(match_grid_points(ox, truth, radius=2.))
        cases.append(case)
        if (i+1) % 64 == 0:
            progress(root, 'evaluation', name=trial['name'], completed=i+1, total=len(selected))
    report = {'trial': trial, 'cases': cases,
              'by_backend': {k: summarize_grid_matches(v) for k, v in matches.items()},
              'held_out_noise': {k: summarize_grid_matches(v) for k, v in ood.items()},
              'limits': ['Synthetic patient-disjoint test', 'Physical families overlap across splits',
                         'Point match radius 2 pixels at 1024x704', 'Restoration MSE is not clinical or waveform validation',
                         'No generator coordinates used by detector or topology inference']}
    _write_json(root/('test-'+trial['name']+'.json'), report)
    return report



def evaluate_previous(root, manifest, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = GridFieldNet(contrast=state['variant'] != 'raw-point').cuda().eval()
    model.load_state_dict(state['state_dict'])
    grouped, cases = {}, []
    for row in manifest:
        if row['split'] != 'test':
            continue
        with np.load(root/row['file']) as data:
            rgb, truth = data['image'].copy(), data['xy'].copy()
        h, w = rgb.shape[:2]
        ratio = min(768/w, 512/h)
        rw, rh = round(w*ratio), round(h*ratio)
        canvas = np.full((512, 768, 3), 255, np.uint8)
        canvas[:rh, :rw] = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR))
        xy, scores, _ = detect(model, canvas, threshold=float(state['threshold']))
        inside = ((xy >= 0) & (xy <= [rw-1, rh-1])).all(1)
        xy = (xy[inside]+.5)/[rw/w, rh/h]-.5
        match = match_grid_points(xy, truth, radius=2.)
        grouped.setdefault(row['backend'], []).append(match)
        cases.append({'key': row['key'], 'group_id': row['group_id'], 'points': summarize_grid_matches([match])})
    _write_json(root/'previous-baseline.json', {'checkpoint_sha256': sha256_file(checkpoint),
        'native_input': 'aspect-preserving 768x512, pixel-center coordinates returned to 1024x704',
        'threshold': state['threshold'], 'by_backend': {k: summarize_grid_matches(v) for k, v in grouped.items()},
        'cases': cases, 'limits': ['Architecture, resolution, training cohort and training budget all differ']})

def main(cohort, root, steps=10000, batch_size=4, render_only=False, resume=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required')
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root/'protocol.json', {'width': WIDTH, 'height': HEIGHT, 'views_per_source': VIEWS,
        'train_noise': TRAIN_NOISE, 'held_out_noise': OOD_NOISE, 'steps': steps, 'batch_size': batch_size,
        'seeds': SEEDS, 'clean_targets': 'same-view clean print projected with the identical inverse field',
        'final_rectification_samples': 'original RGB', 'real_images_in_training': False})
    manifest = render_bank(cohort, root)
    if render_only:
        return
    trials = []
    # All choices are frozen before opening any synthetic test predictions.
    for seed in SEEDS:
        for joint in (False, True):
            trials.append(train(root, manifest, joint=joint, seed=seed, steps=steps, batch_size=batch_size, resume=resume))
            _write_json(root/'trials.json', trials)
            torch.cuda.empty_cache()
    selected = max(trials, key=lambda t: t['validation']['backend_macro_f1'])
    _write_json(root/'selected.json', selected)
    for trial in trials:
        evaluate(root, manifest, trial)
        torch.cuda.empty_cache()
    progress(root, 'completed', selected=selected['name'], bank_images=len(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    main(**vars(args))
