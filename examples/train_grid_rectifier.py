"""Train a small subpixel grid detector on verified, patient-separated captures.

Cached images are fixed before this experiment. Test waveforms and generator
grid identities never enter training or confidence-threshold selection.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.grid_benchmark import BenchmarkConfig, _targets, train_model, evaluate_model


def verified_captures(cache):
    manifest = json.loads((cache/'manifest.json').read_text())
    owners, fingerprints, samples = {}, {}, []
    for row in manifest:
        path = cache/'captures'/(row['key']+'.pt')
        if sha256_file(path) != row['file_sha256']:
            raise ValueError('Cached capture hash mismatch')
        # Explicitly trusted local cache produced by this renderer, not imports.
        sample = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('key', 'source_id', 'group_id', 'split', 'signal_sha256'):
            if sample[key] != row[key]:
                raise ValueError('Capture identity mismatch')
        for mapping, key in ((owners, 'group_id'), (fingerprints, 'signal_sha256')):
            if mapping.setdefault(sample[key], sample['split']) != sample['split']:
                raise ValueError('Patient or signal crosses data partitions')
        samples.append(sample)
    if {s['split'] for s in samples} != {'train', 'validation', 'test'}:
        raise ValueError('Three nonempty source partitions required')
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    config = BenchmarkConfig(steps=args.steps)
    samples = verified_captures(args.cache)
    _targets(samples, config)
    print(json.dumps({'training': True, 'steps': config.steps, 'captures': len(samples)}), flush=True)
    model, training = train_model(samples, 'mixed', config)
    torch.save({'state_dict': model.state_dict(), 'configuration': asdict(config),
                'threshold': training['threshold']}, args.out/'model.pt')
    _write_json(args.out/'training.json', {'configuration': asdict(config), 'training': training,
        'capture_manifest_sha256': sha256_file(args.cache/'manifest.json'),
        'checkpoint_sha256': sha256_file(args.out/'model.pt'),
        'source_groups': {split: sorted({s['group_id'] for s in samples if s['split'] == split})
                          for split in ('train', 'validation', 'test')}})
    result = evaluate_model(model, samples, config, training['threshold'], args.out, 'learned')
    _write_json(args.out/'test-detection.json', result)
    print(json.dumps({'status': 'completed', 'checkpoint_sha256': sha256_file(args.out/'model.pt'),
                      'test': result['by_backend']}), flush=True)
