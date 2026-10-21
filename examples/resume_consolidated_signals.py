"""Finish frozen-model extraction from verified, already prepared study inputs."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.runner import _write_json
from examples.consolidated_grid_study import progress
from examples.digitization_grid_rectification import save_rgb
from examples.digitization_npj_profiles import run


def main(root, source):
    trials=json.loads((root/'trials.json').read_text())
    for trial in trials:
        if sha256_file(root/trial['checkpoint'])!=trial['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint hash mismatch')
    pairs=json.loads((root/'signal-inputs.json').read_text())
    results=[]
    for i,patient in enumerate(pairs):
        original=root/patient['original']
        manifest=json.loads((original/'manifest.json').read_text())
        resized=original.parent/'resize-2x'
        resized.mkdir()
        (resized/'source.npz').write_bytes((original/'source.npz').read_bytes())
        resized_manifest=dict(manifest,cases=[])
        for entry in manifest['cases']:
            if sha256_file(original/entry['image'])!=entry['image_sha256']:
                raise ValueError('Frozen original RGB hash mismatch')
            with Image.open(original/entry['image']) as image:
                rgb=np.array(image.convert('RGB').resize((image.width*2,image.height*2),Image.Resampling.BILINEAR))
            copied=copy.deepcopy(entry)
            copied['image_sha256']=save_rgb(resized/entry['image'],rgb)
            resized_manifest['cases'].append(copied)
        _write_json(resized/'manifest.json',resized_manifest)
        for arm in ('original','oracle','learned','resize-2x'):
            inputs=original.parent/arm
            prepared=json.loads((inputs/'manifest.json').read_text())
            if prepared['cases']:
                report=run(inputs,root/'signal-results'/patient['source_id']/arm,source,
                           device='cuda',arms=('configured',),dewarping=False)
                results.append(dict(patient,arm=arm,cases=report['cases'],status='completed'))
            else:
                results.append(dict(patient,arm=arm,cases=[],status='no accepted geometry'))
            _write_json(root/'signal-results.json',results)
            progress(root,'signal-evaluation',patient=i+1,total=len(pairs),arm=arm)
    progress(root,'resumed-signal-study-completed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    main(a.root,a.source)
