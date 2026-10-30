"""Read-only status of this conversation's exact original-training execution."""

import argparse
import json
import os
import re
import subprocess
from datetime import UTC, datetime

from reproductions.common import write_json

TRAINING_RUN = 41070920613
TRAINING_NAME = 'ecg-repro-41070920613-1'


def select_training_pod(rows):
    matches = [row for row in rows if row.get('name') == TRAINING_NAME]
    if len(matches) > 1:
        raise ValueError('Ambiguous original-training execution identity')
    if not matches:
        return None
    identity = matches[0].get('id')
    if not isinstance(identity, str) or not re.fullmatch(r'[a-zA-Z0-9]+', identity):
        raise ValueError('Verified resource identity required')
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    def read(arguments):
        completed = subprocess.run([args.cli, 'pod', *arguments], capture_output=True, text=True, timeout=60, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stderr.replace(os.environ['RUNPOD_API_KEY'], '[redacted]')[:2000])
        return json.loads(completed.stdout)

    identity = select_training_pod(read(['list']))
    result = {'schema': 'exact-owned-training-status-v1', 'training_run_id': TRAINING_RUN,
        'expected_name': TRAINING_NAME, 'read_at_utc': datetime.now(UTC).isoformat(),
        'mutations': [], 'matching_pod_present': identity is not None,
        'optimizer_progress_verified': False,
        'limitation': 'Resource status cannot establish completed optimizer epochs. Last verified checkpoint remains separate.'}
    if identity:
        pod = read(['get', identity])
        if pod.get('id') != identity or pod.get('name') != TRAINING_NAME:
            raise ValueError('Read-back belongs to another resource')
        result['pod'] = {key: pod[key] for key in ['id', 'name', 'status', 'gpu', 'cost', 'dataCenterId', 'createdAt'] if key in pod}
    write_json(args.out, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
