"""Bounded sim-to-real baseline on one cheap rented GPU.

Runs the steps of notebooks/sim2real_benchmark.ipynb without Colab: PTB-XL
signals, a patient-disjoint ECG-Image-Database D0 test cohort, then for every
generator recipe the same render -> train -> digitize -> score chain with the
same budget. Adds a KID distance between rendered and real images. Results are
written as they arrive, so a deadline still leaves every finished arm.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
import urllib.request

PTBXL_URL = ('https://physionet.org/static/published-projects/ptb-xl/'
             'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip')


class Run:
    def __init__(self, workspace, plan, deadline):
        self.workspace, self.plan, self.deadline = Path(workspace), plan, deadline
        self.code = self.workspace / 'ecg-code'
        self.data = self.workspace / 'data'
        self.out = self.workspace / 'outputs' / 'sim2real'
        self.runs = self.workspace / 'runs'
        for path in (self.data, self.out, self.runs):
            path.mkdir(parents=True, exist_ok=True)
        self.report = {'schema': 'ecg-sim2real-runpod-v1', 'config': plan['sim2real'], 'steps': [], 'arms': {}}

    def left(self):
        return self.deadline - time.time()

    def save(self):
        (self.out / 'report.json').write_text(json.dumps(self.report, indent=2, default=str))

    def sh(self, args, name, timeout=None):
        begin = time.time()
        limit = max(60, min(timeout or 1e9, self.left() - 180))
        print('>>', name, ' '.join(map(str, args))[:400], flush=True)
        result = subprocess.run([str(a) for a in args], cwd=self.code, capture_output=True, text=True, timeout=limit)
        log = self.out / 'logs' / (name.replace('/', '_') + '.log')
        log.parent.mkdir(exist_ok=True)
        log.write_text(result.stdout[-200000:] + '\n--- stderr ---\n' + result.stderr[-200000:])
        self.report['steps'].append({'name': name, 'returncode': result.returncode, 'seconds': time.time() - begin})
        self.save()
        print(result.stdout[-1500:], result.stderr[-1500:], flush=True)
        return result

    def prepare(self):
        c = self.plan['sim2real']
        ptbxl = self.data / 'ptbxl'
        if not ptbxl.exists():
            archive = self.data / 'ptbxl.zip'
            begin = time.time()
            urllib.request.urlretrieve(PTBXL_URL, archive)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(self.data)
            (self.data / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3').rename(ptbxl)
            archive.unlink()
            self.report['steps'].append({'name': 'ptbxl', 'seconds': time.time() - begin})
        python = sys.executable
        if self.sh([python, '-m', 'reproductions.bootstrap', '--dest', self.workspace / 'sources', '--method', 'npj'],
                   'bootstrap-npj').returncode:
            raise RuntimeError('npj bootstrap failed')
        if self.sh([python, '-m', 'reproductions.fetch_catalog', '--out', self.data / 'catalog'], 'catalog').returncode:
            raise RuntimeError('D0 catalog failed')
        if self.sh([python, 'examples/sim2real_npj.py', 'select', '--ptbxl-csv', ptbxl / 'ptbxl_database.csv',
                    '--d0-files', self.data / 'catalog' / 'files.json', '--kinds', *c['kinds'],
                    '--test-records', c['test_records'], '--train-records', c['train_records'],
                    '--val-records', c['val_records'], '--seed', c['seed'], '--out', self.data / 'split.json'],
                   'select').returncode:
            raise RuntimeError('split selection failed')
        split = json.loads((self.data / 'split.json').read_text())
        ids = [str(row['record']) for row in split['test']]
        if self.sh([python, 'examples/prepare_digitization_reference.py', '--out', self.data / 'real',
                    '--metadata', ptbxl / 'ptbxl_database.csv', '--kinds', *c['kinds'], '--records', *ids],
                   'prepare-real').returncode:
            raise RuntimeError('real test images failed')
        shutil.copy(self.data / 'split.json', self.out / 'split.json')
        self.manifest = json.loads((self.data / 'real' / 'manifest.json').read_text())
        shutil.copy(self.data / 'real' / 'manifest.json', self.out / 'real-manifest.json')
        self.report['real_images'] = len(self.manifest['cases'])
        self.report['real_patients'] = len({case['patient_id'] for case in self.manifest['cases']})
        self.save()

    def geometry_bank(self):
        """Offline Mitsuba geometry for the full-v2 recipe, built once on this pod's CPUs."""
        c = self.plan['sim2real']
        bank = self.data / 'geometry-bank.npz'
        if self.sh([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q', 'mitsuba==3.9.1'],
                   'install-mitsuba').returncode:
            raise RuntimeError('mitsuba install failed')
        if self.sh([sys.executable, 'examples/build_geometry_bank.py', '--out', bank, '--count', c.get('bank_entries', 64),
                    '--width', 640, '--height', 480, '--seed', c['seed']], 'geometry-bank').returncode:
            raise RuntimeError('geometry bank failed')
        shutil.copy(bank.with_suffix('.json'), self.out / 'geometry-bank.json')
        os.environ['ECG_GEOMETRY_BANK'] = str(bank)

    def infer_and_score(self, arm, checkpoint=None):
        """Native npj inference is CPU-bound per image; run disjoint shards concurrently."""
        images = [str(self.data / 'real' / case['image']) for case in self.manifest['cases']]
        predictions = self.runs / arm / 'infer'
        shutil.rmtree(predictions, ignore_errors=True)
        predictions.mkdir(parents=True)
        shards = max(1, min(self.plan['sim2real'].get('infer_shards', 4), (os.cpu_count() or 2) // 2, len(images)))
        processes = []
        begin = time.time()
        for index in range(shards):
            command = [sys.executable, '-m', 'reproductions.npj.infer', '--upstream', self.workspace / 'sources' / 'npj',
                       '--preset', 'george-moody-2024', '--out', predictions / f'shard-{index}', '--threads', 2]
            command += [arg for image in images[index::shards] for arg in ('--image', image)]
            if checkpoint:
                command += ['--segmentation-checkpoint', checkpoint]
            log = (self.out / 'logs' / f'{arm}_infer_{index}.log')
            log.parent.mkdir(exist_ok=True)
            processes.append((subprocess.Popen([str(a) for a in command], cwd=self.code, stdout=log.open('w'),
                                               stderr=subprocess.STDOUT), log))
        limit = max(60, self.left() - 180)
        codes = []
        for process, _ in processes:
            try:
                codes.append(process.wait(timeout=max(1, limit - (time.time() - begin))))
            except subprocess.TimeoutExpired:
                process.kill()
                codes.append('timeout')
        receipts = []
        for index in range(shards):
            shard = predictions / f'shard-{index}'
            for file in shard.glob('*.npz'):
                shutil.move(str(file), predictions / file.name)
            if (shard / 'receipt.json').exists():
                receipts.append(json.loads((shard / 'receipt.json').read_text()))
        (predictions / 'receipt.json').write_text(json.dumps({'shards': receipts}, default=str))
        self.report['steps'].append({'name': f'{arm}/infer', 'returncode': codes, 'shards': shards,
                                     'cpu_count': os.cpu_count(), 'seconds': time.time() - begin})
        self.save()
        target = self.out / arm / 'scores.json'
        target.parent.mkdir(exist_ok=True)
        self.sh([sys.executable, 'examples/sim2real_npj.py', 'score', '--manifest', self.data / 'real' / 'manifest.json',
                 '--predictions', predictions, '--arm', arm, '--out', target], f'{arm}/score')
        if (predictions / 'receipt.json').exists():
            shutil.copy(predictions / 'receipt.json', self.out / arm / 'infer-receipt.json')
        if target.exists():
            self.report['arms'].setdefault(arm, {})['summary'] = json.loads(target.read_text()).get('summary')
        self.save()

    def arm(self, arm):
        c = self.plan['sim2real']
        work = self.runs / arm
        shutil.rmtree(work, ignore_errors=True)
        (self.out / arm).mkdir(exist_ok=True)
        python = sys.executable
        begin = time.time()
        if self.sh([python, 'examples/sim2real_npj.py', 'render', '--split', self.data / 'split.json',
                    '--ptbxl-root', self.data / 'ptbxl', '--recipe', arm, '--out', work / 'pairs',
                    '--height', c['height'], '--width', c['width'], '--seed', c['seed']], f'{arm}/render').returncode:
            raise RuntimeError(f'{arm} render failed')
        self.report['arms'].setdefault(arm, {})['render_seconds'] = time.time() - begin
        samples = self.out / arm / 'samples'
        samples.mkdir(exist_ok=True)
        for stem in ['000001', '000002', '000003', '000004', '000005', '000006']:
            for suffix in ['.png', '_mask.png']:
                source = work / 'pairs' / 'train' / (stem + suffix)
                if source.exists():
                    shutil.copy(source, samples / (stem + suffix))
        for name in ['render-receipt.json']:
            if (work / 'pairs' / name).exists():
                shutil.copy(work / 'pairs' / name, self.out / arm / name)
        self.report['arms'][arm]['kid'] = self.kid(sorted((work / 'pairs' / 'train').glob('[0-9]*[0-9].png'))[:c['kid_images']])
        self.save()
        begin = time.time()
        if self.sh([python, 'examples/sim2real_npj.py', 'train', '--upstream', self.workspace / 'sources' / 'npj',
                    '--train-dir', work / 'pairs' / 'train', '--val-dir', work / 'pairs' / 'val', '--out', work / 'train',
                    '--steps', c['steps'], '--batch-size', c['batch'], '--crop', c['crop'], '--seed', c['seed']],
                   f'{arm}/train').returncode:
            raise RuntimeError(f'{arm} training failed')
        self.report['arms'][arm]['train_seconds'] = time.time() - begin
        for name in ['train-receipt.json']:
            if (work / 'train' / name).exists():
                shutil.copy(work / 'train' / name, self.out / arm / name)
        shutil.rmtree(work / 'pairs', ignore_errors=True)
        self.infer_and_score(arm, work / 'train' / 'segmentation.pt')

    def kid(self, fake_paths):
        """Kernel Inception Distance, overall and per real capture kind."""
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.models import inception_v3, Inception_V3_Weights
        if not hasattr(self, 'inception'):
            model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True).eval().cuda()
            model.fc = torch.nn.Identity()
            self.inception = model
        mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
        std = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]

        def features(paths):
            rows = []
            with torch.no_grad():
                for start in range(0, len(paths), 16):
                    batch = []
                    for path in paths[start:start + 16]:
                        image = Image.open(path).convert('RGB').resize((299, 299), Image.BICUBIC)
                        batch.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255)
                    value = (torch.stack(batch).cuda() - mean) / std
                    rows.append(self.inception(value).double().cpu())
            return torch.cat(rows).numpy()

        def kid_value(x, y, subsets=50, size=None):
            size = size or min(len(x), len(y), 100)
            rng = np.random.default_rng(0)
            dimension = x.shape[1]
            values = []
            for _ in range(subsets):
                a = x[rng.choice(len(x), size, replace=False)]
                b = y[rng.choice(len(y), size, replace=False)]
                kxx = (a @ a.T / dimension + 1) ** 3
                kyy = (b @ b.T / dimension + 1) ** 3
                kxy = (a @ b.T / dimension + 1) ** 3
                m = size
                values.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                              - 2 * kxy.mean())
            return {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'subset_size': int(size)}

        if not hasattr(self, 'real_features'):
            self.real_features = {}
            for case in self.manifest['cases']:
                self.real_features.setdefault(case.get('image_type', 'all'), []).append(self.data / 'real' / case['image'])
            self.real_features = {kind: features(paths) for kind, paths in self.real_features.items()}
        fake = features(fake_paths)
        real_all = np.concatenate(list(self.real_features.values()))
        result = {'fake_images': len(fake_paths), 'all_real': kid_value(fake, real_all)}
        for kind, value in self.real_features.items():
            result[kind] = kid_value(fake, value)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()
    execution = json.loads((Path(args.workspace) / 'execution.json').read_text())
    plan = execution['plan']
    run = Run(args.workspace, plan, execution['deadline'] - 120)
    try:
        run.prepare()
        if 'full-v2' in plan['sim2real']['arms']:
            run.geometry_bank()
        if plan['sim2real'].get('published_control', False):
            run.infer_and_score('published')
        for arm in plan['sim2real']['arms']:
            if run.left() < plan['sim2real']['minimum_seconds_per_arm']:
                run.report['arms'][arm] = {'skipped': 'deadline'}
                run.save()
                continue
            try:
                run.arm(arm)
            except Exception:
                run.report['arms'].setdefault(arm, {})['error'] = traceback.format_exc()[-3000:]
                run.save()
    except Exception:
        run.report['error'] = traceback.format_exc()[-4000:]
        run.save()
        raise
    run.report['finished'] = True
    run.save()


if __name__ == '__main__':
    main()
