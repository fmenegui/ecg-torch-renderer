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
TRAINING_VOLUME = 'wbbmw9kpkg'


def verify_training_volume(volume):
    if (volume.get('id') != TRAINING_VOLUME or volume.get('name') != TRAINING_NAME
            or volume.get('size') != 220 or volume.get('dataCenter') != 'US-KS-2'):
        raise ValueError('Original training volume identity or placement changed')
    return {key: volume[key] for key in ['id', 'name', 'size', 'dataCenter']}


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
        completed = subprocess.run([args.cli, *arguments], capture_output=True, text=True, timeout=60, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stderr.replace(os.environ['RUNPOD_API_KEY'], '[redacted]')[:2000])
        return json.loads(completed.stdout)

    identity = select_training_pod(read(['pod', 'list', '--all']))
    result = {'schema': 'exact-owned-training-status-v1', 'training_run_id': TRAINING_RUN,
        'expected_name': TRAINING_NAME, 'read_at_utc': datetime.now(UTC).isoformat(),
        'mutations': [], 'all_pod_states_queried': True, 'matching_pod_present': identity is not None,
        'optimizer_progress_verified': False,
        'limitation': 'Resource status cannot establish completed optimizer epochs. Last verified checkpoint remains separate.'}
    if identity:
        pod = read(['pod', 'get', identity])
        if pod.get('id') != identity or pod.get('name') != TRAINING_NAME:
            raise ValueError('Read-back belongs to another resource')
        result['pod'] = {key: pod[key] for key in ['id', 'name', 'status', 'desiredStatus', 'runtimeStatus',
            'runtimeStatusReason', 'gpu', 'cost', 'dataCenterId', 'createdAt'] if key in pod}
    write_json(args.out, result)
    try:
        result['training_volume'] = verify_training_volume(read(['network-volume', 'get', TRAINING_VOLUME]))
        result['training_volume_metadata_verified'] = True
    except RuntimeError as error:
        result['training_volume_metadata_verified'] = False
        result['training_volume_read_error'] = str(error)
    write_json(args.out, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
