"""Small all-artificial integration dataset. Supply real sources through ecg-dataset.

Run: python examples/export_paired_dataset.py --out build/paired-demo
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ecg_gpu.demo import demo_signals
from ecg_gpu.paired_cli import generate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=384)
    args = parser.parse_args()
    torch.set_num_threads(2)
    inputs = args.out.parent / (args.out.name + '-artificial-inputs')
    if inputs.exists() and any(inputs.iterdir()):
        raise FileExistsError('Artificial input directory already contains files')
    inputs.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(4):
        values, rate, _ = demo_signals(fs=100, duration=10)
        # Explicitly artificial sources, not augmented patient recordings.
        values = values * (1 + index * .1)
        filename = f'artificial-{index}.npz'
        np.savez(inputs / filename, signals=values, fs=rate, units='mV')
        records.append({'input': filename, 'source_id': f'artificial-source-{index}',
                        'group_id': f'artificial-group-{index}'})
    manifest = inputs / 'sources.json'
    manifest.write_text(json.dumps({'records': records}, indent=2) + '\n', encoding='utf-8')
    report = generate_dataset(manifest, args.out, views=2, width=args.width, height=args.height,
                              layouts=('3x4+r1', '12x1'), gain_mm_per_mv=20)
    print(json.dumps({'source': 'entirely artificial integration examples', **report['counts']}))


if __name__ == '__main__':
    main()
