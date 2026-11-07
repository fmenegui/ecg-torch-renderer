"""Independent point-spacing controls against the pinned public decoder.

No learned model is run. These controls establish decoder behavior only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


def run(decoder):
    content = decoder.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if blob != '790a040fa23d8ac488afee643445d803ab0c4c6b':
        raise ValueError('Decoder snapshot differs from pinned Git blob')
    spec = importlib.util.spec_from_file_location('pinned_grid_decoder', decoder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for spacing, nms_radius, capacity in ((2, 2, 16), (2, 1, 16), (3, 2, 16), (3, 2, 1)):
        heat = torch.zeros(1, 1, 15, 15)
        heat[0, 0, 7, 5] = .95
        heat[0, 0, 7, 5+spacing] = .90
        offsets = torch.zeros(1, 2, 15, 15)
        result = module.decode_grid_points(heat, offsets, threshold=.85,
                                          max_points=capacity, nms_radius=nms_radius)
        keep = result['valid'][0]
        records.append({'peak_spacing_px': spacing, 'nms_radius_px': nms_radius,
            'max_points': capacity, 'detected_points': int(keep.sum()),
            'xy': result['xy'][0, keep].tolist(), 'scores': result['scores'][0, keep].tolist(),
            'truncated': bool(result['truncated'][0])})
    assert [r['detected_points'] for r in records] == [1, 2, 2, 1]
    assert records[-1]['truncated']
    return {'repository': 'fmenegui/ecg-torch-renderer',
        'commit': 'b9fd04dd6516e3469d1f36593849249c082458a8',
        'path': 'ecg_gpu/grid_point_learning.py', 'git_blob_sha': blob,
        'pytorch_version': torch.__version__, 'device': 'cpu',
        'neural_model_executed': False, 'records': records,
        'limits': ['Synthetic heat peaks, not real CNN outputs',
                   'No accuracy improvement demonstrated',
                   'Smaller NMS may also increase false positives and needs validation']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder', type=Path, default=Path(__file__).resolve().parents[2]/'ecg_gpu/grid_point_learning.py')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('nms_resolution_control.json'))
    args = parser.parse_args()
    report = run(args.decoder)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
