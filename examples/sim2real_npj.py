"""Sim-to-real comparison of ECG image generators through one fixed digitizer.

The Open-ECG-Digitizer segmentation network (npj Digital Medicine, pinned in
``reproductions/npj``) is trained with the same budget on pairs from each
generator recipe, then the native npj pipeline digitizes the same real
ECG-Image-Database images and the signals are scored against PTB-XL
references. Only the training images change between arms.

Subcommands (run in order; ``notebooks/sim2real_benchmark.ipynb`` drives them):

``select``  patient-disjoint PTB-XL training/validation records and real test
            records present in ECG-Image-Database D0.
``render``  generator training pairs for a named recipe from PTB-XL signals.
``train``   fixed-step training of the upstream UNet with its native loss,
            optimizer family and augmentations.
``score``   score native npj outputs against printed reference windows.

Scores use ``ecg_gpu.digitization.external_evaluation`` on the printed 3x4
windows plus the lead II rhythm strip, with patient-level bootstrap
intervals. This compares generators for one digitizer and one cohort; it is
not a clinical validation or a leaderboard result.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


# ---------------------------------------------------------------- select

# File suffixes of ECG-Image-Database D0 captures, as in examples/prepare_digitization_reference.py.
D0_KINDS = {'digital': '_02.png', 'color-scan': '_04.jpeg', 'gray-scan': '_05.jpeg', 'phone-photo': '_06.jpeg'}


def d0_records(files_json, kinds=tuple(D0_KINDS)):
    """PTB-XL record IDs whose D0 folder has a WFDB reference and every requested capture kind."""
    present = {}
    for item in json.loads(Path(files_json).read_text()):
        name = item['name'] if isinstance(item, dict) else item
        match = re.match(r'D0/D0_(\d{6})/(D0_\d{6}.*)$', name)
        if match:
            present.setdefault(int(match.group(1)), set()).add(match.group(2)[len('D0_000000'):])
    wanted = {D0_KINDS[kind] for kind in kinds} | {'.hea', '.dat'}
    return {record for record, suffixes in present.items() if wanted <= suffixes}


def select(ptbxl_csv, available_test, *, test_records, train_records, val_records, seed=1973):
    """Patient-disjoint split: test from PTB-XL fold 10 in D0, validation fold 9, training folds 1-8.

    Every patient whose record is tested is removed from training and
    validation, so no person is seen by both sides.
    """
    rows = list(csv.DictReader(Path(ptbxl_csv).open()))
    rng = random.Random(seed)
    candidates = sorted(int(float(r['ecg_id'])) for r in rows
                        if int(float(r['strat_fold'])) == 10 and int(float(r['ecg_id'])) in available_test)
    if len(candidates) < test_records:
        raise ValueError(f'Only {len(candidates)} fold-10 records have D0 images; asked for {test_records}')
    patient = {int(float(r['ecg_id'])): str(int(float(r['patient_id']))) for r in rows}
    path = {int(float(r['ecg_id'])): r['filename_hr'] for r in rows}
    # One record per patient in the test set keeps bootstrap units honest.
    rng.shuffle(candidates)
    test, seen = [], set()
    for record in candidates:
        if patient[record] not in seen:
            test.append(record)
            seen.add(patient[record])
        if len(test) == test_records:
            break
    excluded = {patient[r] for r in test}
    def pool(folds, count):
        chosen = [int(float(r['ecg_id'])) for r in rows if int(float(r['strat_fold'])) in folds
                  and str(int(float(r['patient_id']))) not in excluded]
        rng.shuffle(chosen)
        if len(chosen) < count:
            raise ValueError(f'Only {len(chosen)} records available for folds {sorted(folds)}')
        return sorted(chosen[:count])
    train = pool(set(range(1, 9)), train_records)
    val = pool({9}, val_records)
    overlap = ({patient[r] for r in train} | {patient[r] for r in val}) & excluded
    if overlap:
        raise AssertionError('Patient leakage between training and test')
    return {'schema': 'sim2real-npj-split-v1', 'seed': seed,
            'test': [{'record': r, 'patient': patient[r]} for r in sorted(test)],
            'train': [{'record': r, 'path': path[r]} for r in train],
            'val': [{'record': r, 'path': path[r]} for r in val],
            'excluded_patients': len(excluded)}


# ---------------------------------------------------------------- render

def recipe_config(name, height, width):
    """Renderer settings per arm. Only these differ between generator arms."""
    from ecg_gpu import RenderConfig
    base = dict(height=height, width=width, profile='mixed', text_target=True, calibration_target=True)
    if name == 'legacy':
        return RenderConfig(**base)
    if name == 'full':
        return RenderConfig(**base, page_model='physical', stroke='isotropic', lead_marks=True, scan_modes=True,
                            device_filters=True, report_text=True, phone_isp=True, print_technology=True,
                            archive_damage=True, office_marks=True, perspective_relief=True)
    if name == 'full-v2':
        # Footprint-integrated page warp. The Mitsuba geometry bank is passed
        # separately to the renderer (see renderer_assets).
        from dataclasses import replace
        return replace(recipe_config('full', height, width), prefilter_warp=True)
    if name == 'full-prefilter':
        from dataclasses import replace
        return replace(recipe_config('full', height, width), prefilter_warp=True)
    if name.startswith('full-no-'):
        flag = name[len('full-no-'):].replace('-', '_')
        config = recipe_config('full', height, width)
        values = {'page_model': 'legacy', 'stroke': 'legacy'}
        if not hasattr(config, flag):
            raise ValueError(f'Unknown ablation flag {flag!r}')
        from dataclasses import replace
        return replace(config, **{flag: values.get(flag, False)})
    raise ValueError(f'Unknown recipe {name!r}. Use legacy, full, full-prefilter, full-v2 or full-no-<flag>')


def renderer_assets(name):
    """Optional construction-time assets per recipe (never in the batch loop)."""
    if name != 'full-v2':
        return {}
    import os
    from ecg_gpu import DocumentAssets
    path = os.environ.get('ECG_GEOMETRY_BANK')
    if not path:
        raise ValueError('full-v2 needs ECG_GEOMETRY_BANK=<bank.npz> from examples/build_geometry_bank.py')
    return {'document_assets': DocumentAssets.from_npz(path)}


def load_ptbxl(root, relative):
    """[12, T] mV signals and fs from a PTB-XL 16-bit WFDB record."""
    from ecg_gpu.digitization.data import read_wfdb16
    leads = read_wfdb16(Path(root) / (relative + '.hea'))
    fs = leads['II'].fs
    return np.stack([np.asarray(leads[name].values, dtype=np.float32) for name in LEADS]), fs


def render(split_json, ptbxl_root, recipe, out, *, height=1200, width=1600, batch=4, seed=1973,
           pulse_class='signal', device=None):
    import torch
    from ecg_gpu import ECGRenderer
    from ecg_gpu.npj_export import npj_mask, write_npj_pairs
    split = json.loads(Path(split_json).read_text())
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = ECGRenderer(recipe_config(recipe, height, width), **renderer_assets(recipe)).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    out = Path(out)
    receipt = {'schema': 'sim2real-npj-render-v1', 'recipe': recipe, 'height': height, 'width': width,
               'pulse_class': pulse_class, 'seed': seed, 'specification': renderer.specification(), 'splits': {}}
    for name in ('train', 'val'):
        rows = split[name]
        index = 1
        start = time.time()
        for offset in range(0, len(rows), batch):
            chunk = rows[offset:offset + batch]
            loaded = [load_ptbxl(ptbxl_root, row['path']) for row in chunk]
            fs = {value for _, value in loaded}
            if len(fs) != 1:
                raise ValueError('Mixed sampling rates in one batch')
            signals = torch.tensor(np.stack([value for value, _ in loaded]), device=device)
            result = renderer(signals, fs.pop(), stage='all', generator=generator)
            write_npj_pairs(result.image, npj_mask(result.targets, pulse_class=pulse_class), out / name, index)
            index += len(chunk)
        receipt['splits'][name] = {'pairs': index - 1, 'seconds': time.time() - start}
        print(json.dumps({'recipe': recipe, 'split': name, 'pairs': index - 1}), flush=True)
    (out / 'render-receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    return receipt


# ---------------------------------------------------------------- train

def train(upstream, train_dir, val_dir, out, *, steps=3000, batch_size=4, crop=1024, lr=.0037,
          weight_decay=.001, seed=1973, device='cuda', log_every=50):
    """Fixed-step training with the upstream UNet, DiceFocalLoss, optimizer and augmentations.

    Native transforms come from the upstream ``unet.yml``; only the crop size,
    step budget and batch size are set here so every arm trains identically.
    """
    import torch
    import yaml
    upstream = Path(upstream).resolve()
    sys.path.insert(0, str(upstream))
    from src.dataset.ecg_scan import ECGScanDataset
    from src.loss.loss import DiceFocalLoss
    from src.model.unet import UNet
    from src.optimizer.adammuon import AdamMuon
    from src.transform.vision import ComposedTransform
    from src.utils import CosineToConstantLR
    config = yaml.safe_load((upstream / 'src/config/unet.yml').read_text())
    transforms = config['DATASET']['TRAIN']['TRANSFORM']['KWARGS']['transform_config']
    for item in transforms:
        if item['class_path'].endswith('RandomCrop'):
            item['KWARGS']['size'] = [crop, crop]
    val_transforms = [{'class_path': 'src.transform.vision.RandomCrop', 'KWARGS': {'size': [crop, crop]}},
                      {'class_path': 'src.transform.vision.RefineMask', 'KWARGS': {}}]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_set = ECGScanDataset(str(Path(train_dir).resolve()), transform=ComposedTransform(transforms))
    val_set = ECGScanDataset(str(Path(val_dir).resolve()), transform=ComposedTransform(val_transforms))
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                         drop_last=True, pin_memory=device == 'cuda')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=2)
    model = UNet(**config['MODEL']['KWARGS']).to(device)
    criterion = DiceFocalLoss(**config['TRAIN']['CRITERION']['KWARGS'])
    optimizer = AdamMuon(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineToConstantLR(optimizer, T_max=steps, eta_min_divisor=10)
    scaler = torch.amp.GradScaler(enabled=device == 'cuda')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log, step, start = [], 0, time.time()
    model.train()
    while step < steps:
        for image, mask in loader:
            image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                loss = criterion(model(image), mask)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % log_every == 0:
                log.append({'step': step, 'loss': float(loss), 'seconds': time.time() - start})
                print(json.dumps(log[-1]), flush=True)
            if step >= steps:
                break
    model.eval()
    losses = []
    with torch.no_grad(), torch.autocast(device_type=device, enabled=device == 'cuda'):
        for image, mask in val_loader:
            losses.append(float(criterion(model(image.to(device)), mask.to(device))))
    torch.save(model.state_dict(), out / 'segmentation.pt')
    receipt = {'schema': 'sim2real-npj-train-v1', 'steps': steps, 'batch_size': batch_size, 'crop': crop,
               'lr': lr, 'weight_decay': weight_decay, 'seed': seed, 'train_pairs': len(train_set),
               'val_pairs': len(val_set), 'val_loss': float(np.mean(losses)) if losses else None,
               'seconds': time.time() - start, 'log': log}
    (out / 'train-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


# ---------------------------------------------------------------- score

def printed_reference(header):
    """[12, T] mV reference with NaN outside the printed 3x4 windows; lead II full length."""
    from ecg_gpu.digitization.data import read_wfdb16, sequential_reference
    full = read_wfdb16(header)
    windows = sequential_reference(full)
    fs = full['II'].fs
    length = len(full['II'].values)
    reference = np.full((12, length), np.nan)
    for row, name in enumerate(LEADS):
        lead = windows[name]
        start = round(lead.start_s * fs)
        reference[row, start:start + len(lead.values)] = lead.values
    return reference, fs


def native_prediction(npz, length):
    """npj ``canonical_lines`` in uV, [12, N], resampled to ``length`` samples over the same 10 s."""
    with np.load(npz) as data:
        values = np.asarray(data['values'], dtype=np.float64)
        unit = str(data['unit']) if 'unit' in data else 'uV'
    if values.ndim != 2 or 12 not in values.shape:
        raise ValueError(f'Expected [12, N] canonical lines, got {values.shape}')
    if values.shape[0] != 12:
        values = values.T
    values = values / 1000. if unit == 'uV' else values
    source = np.linspace(0, 1, values.shape[1], endpoint=False)
    target = np.linspace(0, 1, length, endpoint=False)
    out = np.full((12, length), np.nan)
    for row in range(12):
        finite = np.isfinite(values[row])
        if finite.sum() >= 2:
            interpolated = np.interp(target, source[finite], values[row, finite], left=np.nan, right=np.nan)
            # Do not bridge gaps longer than two source samples.
            gaps = np.interp(target, source, (~finite).astype(float))
            interpolated[gaps > 0] = np.nan
            out[row] = interpolated
    return out


def score(manifest, predictions, out, *, arm, bootstrap=2000, seed=71):
    from ecg_gpu.digitization.external_evaluation import score_external_signals
    manifest = Path(manifest)
    cases = json.loads(manifest.read_text())['cases']
    predictions = Path(predictions)
    rows = []
    for case in cases:
        reference, fs = printed_reference(manifest.parent / case['reference_header'])
        path = predictions / (Path(case['image']).stem + '.npz')
        if path.exists():
            prediction, status = native_prediction(path, reference.shape[1]), 'returned'
        else:
            prediction, status = np.full_like(reference, np.nan), 'missing'
        result = score_external_signals(reference, prediction, fs)
        rows.append({'image_id': case['image_id'], 'patient_id': case['patient_id'],
                     'image_type': case['image_type'], 'status': status, 'metrics': result})
    summary = summarize(rows, bootstrap=bootstrap, seed=seed)
    report = {'schema': 'sim2real-npj-score-v1', 'arm': arm, 'metric': 'external_evaluation primary SNR (dB)',
              'cases': rows, 'summary': summary}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + '\n')
    return report


def _primary_snr(metrics):
    """Record SNR over every printed sample, missing predictions scored as zero."""
    primary = metrics.get('raw_all_printed_zero_missing') or {}
    value = primary.get('snr_db')
    return float(value) if isinstance(value, (int, float)) else float('nan')


def summarize(rows, *, bootstrap=2000, seed=71):
    rng = np.random.default_rng(seed)
    summary = {}
    for kind in sorted({row['image_type'] for row in rows}) + ['all']:
        chosen = [row for row in rows if kind == 'all' or row['image_type'] == kind]
        groups = {}
        for row in chosen:
            groups.setdefault(row['patient_id'], []).append(_primary_snr(row['metrics']))
        values = [v for group in groups.values() for v in group if math.isfinite(v)]
        keys = sorted(groups)
        interval = [float('nan'), float('nan')]
        if len(keys) >= 2:
            means = []
            for _ in range(bootstrap):
                pick = rng.integers(len(keys), size=len(keys))
                sample = [v for i in pick for v in groups[keys[i]] if math.isfinite(v)]
                if sample:
                    means.append(np.mean(sample))
            interval = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        coverage = [row['metrics']['coverage'] for row in chosen]
        summary[kind] = {'images': len(chosen), 'patients': len(keys),
                         'returned': sum(row['status'] == 'returned' for row in chosen),
                         'mean_printed_coverage': float(np.mean(coverage)) if coverage else float('nan'),
                         'mean_snr_db': float(np.mean(values)) if values else float('nan'),
                         'patient_bootstrap_95': interval}
    return summary


# ---------------------------------------------------------------- cli

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    s = commands.add_parser('select')
    s.add_argument('--ptbxl-csv', required=True)
    s.add_argument('--d0-files', required=True, help='files.json from reproductions.fetch_catalog')
    s.add_argument('--kinds', nargs='+', choices=tuple(D0_KINDS), default=list(D0_KINDS))
    s.add_argument('--test-records', type=int, default=40)
    s.add_argument('--train-records', type=int, default=1500)
    s.add_argument('--val-records', type=int, default=150)
    s.add_argument('--seed', type=int, default=1973)
    s.add_argument('--out', required=True)
    r = commands.add_parser('render')
    r.add_argument('--split', required=True)
    r.add_argument('--ptbxl-root', required=True)
    r.add_argument('--recipe', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--height', type=int, default=1200)
    r.add_argument('--width', type=int, default=1600)
    r.add_argument('--batch', type=int, default=4)
    r.add_argument('--seed', type=int, default=1973)
    r.add_argument('--pulse-class', choices=('signal', 'text', 'background'), default='signal')
    t = commands.add_parser('train')
    t.add_argument('--upstream', required=True)
    t.add_argument('--train-dir', required=True)
    t.add_argument('--val-dir', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--steps', type=int, default=3000)
    t.add_argument('--batch-size', type=int, default=4)
    t.add_argument('--crop', type=int, default=1024)
    t.add_argument('--seed', type=int, default=1973)
    t.add_argument('--device', default='cuda')
    c = commands.add_parser('score')
    c.add_argument('--manifest', required=True)
    c.add_argument('--predictions', required=True)
    c.add_argument('--arm', required=True)
    c.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'select':
        result = select(args.ptbxl_csv, d0_records(args.d0_files, args.kinds), test_records=args.test_records,
                        train_records=args.train_records, val_records=args.val_records, seed=args.seed)
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in result.items()}))
    elif args.command == 'render':
        render(args.split, args.ptbxl_root, args.recipe, args.out, height=args.height, width=args.width,
               batch=args.batch, seed=args.seed, pulse_class=args.pulse_class)
    elif args.command == 'train':
        train(args.upstream, args.train_dir, args.val_dir, args.out, steps=args.steps,
              batch_size=args.batch_size, crop=args.crop, seed=args.seed, device=args.device)
    else:
        report = score(args.manifest, args.predictions, args.out, arm=args.arm)
        for kind, values in report['summary'].items():
            print(f"{args.arm} {kind}: {values['mean_snr_db']:.2f} dB, {values['returned']}/{values['images']} "
                  f"returned, 95 % CI {values['patient_bootstrap_95']}")


if __name__ == '__main__':
    main()
