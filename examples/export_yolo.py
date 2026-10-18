"""Export an artificial grouped ECG dataset for YOLO detection.

Example: python examples/export_yolo.py --out build/yolo --groups 4 --views 2
Replace artificial signals with patient ECGs for research. Use patient IDs as
split groups. Different source recordings from one patient share one group.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, RenderConfig, Choice
from ecg_gpu.dataset_export import samples_from_batch, export_yolo_dataset
from ecg_gpu.demo import demo_signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.groups < 2 or args.views < 1:
        parser.error('At least 2 groups and 1 view required')
    torch.set_num_threads(2)
    values, fs, _ = demo_signals(fs=100, duration=10)
    layer = ECGSupervisionLayer(fs=fs,
        config=RenderConfig(height=args.height, width=args.width, surface_resolution=64, shadow_samples=0),
        parameters={'layout': Choice(['3x4+r1', '6x2', '12x1']), 'lead_present': 1.},
        supervision=SupervisionConfig(outputs=('instance_boxes',)), seed=1701).to(args.device)
    records = []
    for group in range(args.groups):
        # Unique amplitudes make distinct source bytes, not independent patients.
        signal = torch.tensor(values * (1 + group*.03), device=args.device, dtype=torch.float32)
        source_hash = hashlib.sha256(signal.detach().cpu().numpy().tobytes()).hexdigest()
        for view in range(args.views):
            batch = layer(signal[None])
            records.extend(samples_from_batch(batch, source_ids=[f'artificial-{group}'],
                group_ids=[f'artificial-group-{group}'], sample_ids=[f'g{group:04d}-v{view:03d}'],
                provenance=[{'artificial': True, 'source_sha256': source_hash, 'sample_rate_hz': fs,
                             'view': view, 'render_seed': layer.seed, 'render_call': layer._calls,
                             'renderer': layer.renderer.specification()}]))
    manifest = export_yolo_dataset(records, args.out)
    print({'samples': len(manifest['samples']), 'groups': len(manifest['groups']),
           'dataset': str(args.out/'dataset.yaml'), 'clinical_validation': False})


if __name__ == '__main__':
    main()
