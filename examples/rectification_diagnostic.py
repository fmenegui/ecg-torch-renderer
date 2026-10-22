"""CPU diagnostic replays with public original signals and dense material truth.

Validation and test patient identities come from the previously frozen cohort.
Replay on LLVM/CPU is a new observation, not byte parity with the CUDA bank.
Gold fields are exported for evaluation only and never passed to inference.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.data import read_wfdb16
from ecg_gpu.schema_v05 import LEADS
from examples.diverse_grid_study import render_one


def prepare(sources, root, patients=6):
    source = json.loads(sources.read_text())
    selected = []
    for split in ('validation', 'test'):
        candidates = [(int(row.get('source_index', i)), row) for i, row in enumerate(source['records'])
                      if row['split'] == split]
        rows = candidates[:patients] if split == 'validation' else candidates[-patients:]
        if len(rows) != patients:
            raise ValueError('Insufficient frozen patients')
        selected.extend(rows)
    root.mkdir(parents=True, exist_ok=True)
    raw = root/'wfdb'
    raw.mkdir(exist_ok=True)

    def download(pair):
        _, row = pair
        for suffix, key in (('.hea', 'header_sha256'), ('.dat', 'dat_sha256')):
            path = raw/(Path(row['filename_lr']).name+suffix)
            if not path.exists():
                url = 'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'+row['filename_lr']+suffix
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read(200001)
                if len(data) > 200000:
                    raise ValueError('Unexpected public record length')
                path.write_bytes(data)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[key]:
                raise ValueError('Original public-source checksum mismatch')
        return row['source_id']

    with ThreadPoolExecutor(6) as pool:
        for name in pool.map(download, selected):
            print(json.dumps({'download_verified': name}), flush=True)
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    selected_indices = {i for i, _ in selected}
    manifest = [r for r in manifest if r['source_index'] in selected_indices]
    known = {r['key'] for r in manifest
             if (root/r['file']).is_file()
             and hashlib.sha256((root/r['file']).read_bytes()).hexdigest() == r['sha256']}
    manifest = [r for r in manifest if r['key'] in known]
    bank = root/'bank'
    bank.mkdir(exist_ok=True)
    import drjit as dr
    dr.set_thread_count(4)
    for index, row in selected:
        record = read_wfdb16(raw/(Path(row['filename_lr']).name+'.hea'))
        values = np.stack([record[name].values for name in LEADS]).astype(np.float32)
        signal = torch.from_numpy(values)[None]
        for view in range(8):
            key = f'{index:05d}-{view}'
            if key in known:
                continue
            capture = render_one(signal, 100., index, view, row['split'],
                                 variant='llvm_ad_rgb', export_geometry=True)
            path = bank/(key+'.npz')
            temporary = path.with_suffix('.npz.part')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **{k: v for k, v in capture.items() if k != 'metadata'})
            temporary.replace(path)
            Image.fromarray(capture['image']).save(bank/(key+'.png'))
            manifest.append({'key': key, 'file': str(path.relative_to(root)),
                             'source_id': row['source_id'], 'group_id': row['group_id'],
                             'split': row['split'], 'source_index': index, 'view': view,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'observation': 'CPU float32 and LLVM replay, not original CUDA bank bytes',
                             **capture['metadata']})
            staging = manifest_path.with_suffix('.json.part')
            staging.write_text(json.dumps(manifest, indent=2)+'\n')
            staging.replace(manifest_path)
            print(json.dumps({'rendered': key, 'split': row['split'], 'completed': len(manifest),
                              'total': len(selected)*8, 'points': len(capture['xy'])}), flush=True)
    (root/'provenance.json').write_text(json.dumps({'source': source['source'], 'license': source['license'],
        'original_source_hashes_verified': True, 'patients_per_split': patients,
        'selected_records': [{'source_index': i, **row} for i, row in selected],
        'gold_usage': 'evaluation only', 'new_training': False, 'cuda_execution': False}, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--patients', type=int, default=6)
    args = parser.parse_args()
    if args.patients < 1:
        parser.error('Positive patient count required')
    torch.set_num_threads(4)
    prepare(**vars(args))
