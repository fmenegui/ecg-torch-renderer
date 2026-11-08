"""Prepare an inference/annotation queue from explicitly listed public captures.

Select patients BEFORE inference. Never turn detector outputs into references.
Originals stay in build/, without redistribution in reports or galleries.
"""
import argparse
import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


def prepare(listing, metadata, cohort_sources, out, *, patients=4):
    if type(patients) is not int or patients < 1:
        raise ValueError('Positive patient count required')
    listing, metadata, cohort_sources, out = map(Path, (listing, metadata, cohort_sources, out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    with metadata.open() as stream:
        records = {int(row['ecg_id']): row for row in csv.DictReader(stream)}
    excluded = {r['group_id'] for r in json.loads(cohort_sources.read_text())['records']}
    files = {row['name']: row for row in json.loads(listing.read_text())['datasetFiles']}
    candidates = sorted({int(m[1]) for name in files if (m := re.fullmatch(r'D0/D0_(\d{6})/D0_\1_04.jpeg', name))})
    selected, groups = [], set()
    # Keep the original pilot patient out of this expansion.
    for index in candidates:
        group = 'ptbxl-patient-'+records[index]['patient_id']
        names = [f'D0/D0_{index:06d}/D0_{index:06d}_{suffix}.jpeg' for suffix in ('04', '05', '06')]
        if index == 22 or group in excluded | groups or any(name not in files for name in names):
            continue
        selected.append((index, group, names))
        groups.add(group)
        if len(selected) == patients:
            break
    if len(selected) != patients:
        raise ValueError('Not enough listed, nonoverlapping patient groups')
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, group, names in selected:
        for name in names:
            url = ('https://www.kaggle.com/api/v1/datasets/download/physionet/ecg-image-database/'
                   +urllib.parse.quote(name, safe='')+'?datasetVersionNumber=2')
            path = out/Path(name).name
            with urllib.request.urlopen(url, timeout=45) as response:
                content = response.read(20_000_001)
            if len(content) > 20_000_000 or len(content) != files[name]['totalBytes']:
                raise ValueError('Unexpected file size for '+name)
            path.write_bytes(content)
            with Image.open(path) as image:
                image.verify()
            suffix = name.rsplit('_', 1)[1].split('.')[0]
            manifest.append({'id': f'physical-{index:06d}-{suffix}', 'image': path.name,
                'sha256': hashlib.sha256(content).hexdigest(), 'source_id': f'ptbxl-{index}', 'group_id': group,
                'source': 'https://www.kaggle.com/datasets/physionet/ecg-image-database', 'dataset_version': 2,
                'source_file': name, 'license': 'CC BY-ND 4.0, originals and overlays excluded from redistribution',
                'physical_capture': True, 'clinical_original_paper': False,
                'capture_kind': {'04': 'color scan', '05': 'grayscale scan', '06': 'phone photograph'}[suffix],
                'annotation_status': 'unannotated, inference and review only'})
            print(json.dumps({'downloaded': path.name, 'bytes': len(content)}), flush=True)
    (out/'external.json').write_text(json.dumps(manifest, indent=2)+'\n')
    protocol = {'selection': f'First {patients} complete scan/gray/photo triplets in the archived lexical file listing, excluding original pilot and all synthetic cohort patients. Convenience sample, not representative',
        'patients': len(groups), 'images': len(manifest), 'selected_before_inference': True,
        'used_for_training_or_threshold_tuning': False, 'annotations_created': False,
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (listing, metadata, cohort_sources)},
        'limits': 'Different patients do not establish different devices or sites. Grid references must be created independently of predictions'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('listing', 'metadata', 'cohort-sources', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--patients', type=int, default=4)
    prepare(**vars(parser.parse_args()))
