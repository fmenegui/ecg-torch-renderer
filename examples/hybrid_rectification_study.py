"""Stream a larger synthetic cohort and train point-only rectifier controls.

Example (CUDA required):
python examples/hybrid_rectification_study.py --cohort build/hybrid-cohort --out build/hybrid-study

Real images never enter this script. Geometric targets are rasterized on CUDA
after selecting train-only samples. The old absolute chart head is not trained.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.field_rectifier import GridFieldNet
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.grid_point_learning import grid_point_loss
from ecg_gpu.grid_points import GridPointConfig, rasterize_grid_points
from examples.consolidated_grid_study import capture_dataset, progress, select_threshold


def read_capture(root, row):
    with np.load(root/row['file']) as data:
        image = torch.from_numpy(data['image'].copy()).permute(2, 0, 1)[None].float()/255
        xy = torch.from_numpy(data['xy'].copy())[None].float()
        valid = torch.from_numpy(data['valid'].astype(np.float32))[None, None]
    return dict(row, image=image, xy=xy, visible=valid)


def verify_cohort(cohort):
    source = json.loads((cohort/'sources.json').read_text())
    if sha256_file(cohort/'signals.npz') != source['signals_sha256']:
        raise ValueError('Source checksum differs')
    with np.load(cohort/'signals.npz') as data:
        owners = {}
        records = []
        for i, (patient, split) in enumerate(zip(data['group_ids'], data['splits'])):
            if owners.setdefault(str(patient), str(split)) != str(split):
                raise ValueError('Patient leakage')
            records.append({'signal': torch.from_numpy(data['signals'][i:i+1].copy()).cuda(),
                            'source_id': str(data['source_ids'][i]), 'group_id': str(patient), 'split': str(split)})
        return records, float(data['fs'])


def train(root, manifest, *, variant, seed, steps, batch_size):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = GridFieldNet(contrast=variant == 'contrast-point').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    training = [r for r in manifest if r['split'] == 'train']
    if not training:
        raise ValueError('No training images')
    model_dir = root/'models'
    model_dir.mkdir(exist_ok=True)
    history = []
    started = time.time()
    for step in range(steps):
        chosen = [read_capture(root, training[int(rng.integers(len(training)))]) for _ in range(batch_size)]
        image = torch.cat([s['image'] for s in chosen]).cuda()
        image = (image*(.8+.4*torch.rand(batch_size, 1, 1, 1, device='cuda'))).clamp(0, 1)
        visible = torch.cat([s['visible'] for s in chosen]).cuda()
        maximum = max(s['xy'].shape[1] for s in chosen)
        xy = torch.zeros(batch_size, maximum, 2, device='cuda')
        present = torch.zeros(batch_size, maximum, dtype=torch.bool, device='cuda')
        for i, s in enumerate(chosen):
            n = s['xy'].shape[1]
            xy[i, :n], present[i, :n] = s['xy'][0].cuda(), True
        targets = rasterize_grid_points(xy, present, visible,
            config=GridPointConfig(sigma_pixels=1., return_offsets=True))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(image)
        point = grid_point_loss(output['heat'], output['offsets'], targets, pos_weight=10.)['loss']
        loss = point+.1*F.binary_cross_entropy_with_logits(output['visible'].float(), visible)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if (step+1) % 1000 == 0 or step == 0:
            history.append({'step': step+1, 'loss': float(loss.detach()), 'elapsed_seconds': time.time()-started})
            name = f'{variant}-{seed}'
            torch.save({'state_dict': model.state_dict(), 'variant': variant, 'seed': seed,
                        'steps': step+1, 'threshold': None, 'selection_status': 'not-selected'}, model_dir/(name+'-resume.pt'))
            _write_json(root/(name+'-history.json'), history)
            progress(root, 'training', variant=variant, seed=seed, step=step+1, steps=steps)
    model.eval()
    # Read validation only after completing the fixed budget. Never open test files here.
    validation = [read_capture(root, r) for r in manifest if r['split'] == 'validation']
    best, choices = select_threshold(model, validation)
    path = model_dir/f'{variant}-{seed}.pt'
    torch.save({'state_dict': model.cpu().state_dict(), 'variant': variant, 'seed': seed,
                'steps': steps, 'threshold': best['threshold'], 'selection_status': 'validation-only'}, path)
    return {'name': path.stem, 'variant': variant, 'seed': seed, 'checkpoint': str(path.relative_to(root)),
            'checkpoint_sha256': sha256_file(path), 'validation': best, 'threshold_choices': choices}


def main(cohort, out, steps=20000, batch_size=2, render_only=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. No CPU substitution for the GPU experiment')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'trials.json').exists():
        raise FileExistsError('Do not overwrite completed trials')
    records, fs = verify_cohort(cohort)
    manifest_path = out/'capture-manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = len(records)*6
        if len(manifest) != expected:
            raise ValueError('Partial capture cache. Recover or regenerate before training')
        for row in manifest:
            if sha256_file(out/row['file']) != row['sha256']:
                raise ValueError('Capture checksum differs')
    else:
        manifest = capture_dataset(records, fs, out, views=3, samples_per_pixel=128)
    del records
    torch.cuda.empty_cache()
    if render_only:
        return
    trials = []
    for seed in (1973, 20261021):
        for variant in ('raw-point', 'contrast-point'):
            trials.append(train(out, manifest, variant=variant, seed=seed, steps=steps, batch_size=batch_size))
            _write_json(out/'trials.json', trials)
            torch.cuda.empty_cache()
    selected = max(trials, key=lambda row: row['validation']['backend_macro_f1'])
    _write_json(out/'selected.json', selected)
    progress(out, 'trained', selected=selected['name'], test_evaluated=False, real_images_used=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cohort', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('Positive steps and batch size required')
    torch.set_num_threads(4)
    main(**vars(args))
