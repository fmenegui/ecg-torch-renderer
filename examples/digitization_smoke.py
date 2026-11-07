"""Execute real public checkpoints and compare the recovered waveform to truth.

Use a declared development manifest. The loose SNR floor is an integration
regression guard on this fixture, not a clinical generalization requirement.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='float32')
    args = parser.parse_args()
    from ecg_gpu.digitization.cli import main as run
    status = run(['run', '--assets', str(args.assets), '--manifest', str(args.manifest),
                  '--out', str(args.out), '--precision', args.precision, '--threads', '2'])
    report = json.loads((args.out/'report.json').read_text())
    if status or report['successes'] != report['cases_total'] or report['scored_cases'] != report['cases_total']:
        raise RuntimeError('Reference inference failed or silently omitted a scored image')
    if report['official_snr_db'] is None or report['official_snr_db'] < 15:
        raise RuntimeError('Integration SNR fell below the declared development-fixture floor')
    print('Real checkpoint, geometry and waveform integration passed')


if __name__ == '__main__':
    main()
