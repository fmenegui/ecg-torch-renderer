"""Command line for explicit artifact setup and reference digitization."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('fetch', 'check', 'run'):
        p = commands.add_parser(name)
        p.add_argument('--assets', type=Path, required=True)
        p.add_argument('--profile', choices=('single', 'full'), default='single')
        if name == 'run':
            p.add_argument('--manifest', type=Path, required=True)
            p.add_argument('--out', type=Path, required=True)
            p.add_argument('--device', default='cpu')
            p.add_argument('--precision', choices=('float32', 'bfloat16', 'float16'), default='float32')
            p.add_argument('--geometry', choices=('reference', 'composed'), default='reference')
            p.add_argument('--waveform-tta', action='store_true')
            p.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    from .artifacts import fetch_assets, verify_assets
    if args.command in ('fetch', 'check'):
        op = fetch_assets if args.command == 'fetch' else verify_assets
        lock = op(args.assets, profile=args.profile)
        print(json.dumps({'status': 'verified', 'profile': args.profile,
                          'lock_sha256': lock['lock_sha256']}))
        return 0
    if args.threads < 1:
        parser.error('--threads must be positive')
    # Validate paths/manifests before importing optional model dependencies.
    from .data import load_manifest
    load_manifest(args.manifest)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be new or empty')
    import torch
    from .reference import HengTangDigitizer
    from .evaluation import OfficialMetric
    from .runner import run_manifest
    torch.set_num_threads(args.threads)
    digitizer = HengTangDigitizer(args.assets, device=args.device, precision=args.precision,
        geometry=args.geometry, waveform_tta=args.waveform_tta, profile=args.profile)
    metric = OfficialMetric(args.assets)
    report = run_manifest(args.manifest, args.out, digitizer, metric,
                          progress=lambda item: print(json.dumps(item), flush=True))
    print(json.dumps({'status': report['status'], 'successes': report['successes'],
        'failures': report['failures'], 'official_snr_db': report['official_snr_db']}))
    return int(report['failures'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
