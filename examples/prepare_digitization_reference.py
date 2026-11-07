"""Download a declared ECG-Image-Database v2 development cohort.

This is a research smoke set, not a new held-out clinical benchmark. Training
overlap with the upstream public checkpoints is unknown. No source image is
redistributed by this script or by the public evaluation report.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.data import read_wfdb16

KINDS = {'digital': ('02', '.png'), 'color-scan': ('04', '.jpeg'),
         'gray-scan': ('05', '.jpeg'), 'phone-photo': ('06', '.jpeg')}


def prepare(out, records, kinds, *, metadata=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    if not records or len(set(records)) != len(records) or any(type(i) is not int or not 1 <= i <= 999999 for i in records):
        raise ValueError('Distinct positive record IDs required')
    if not kinds or len(set(kinds)) != len(kinds) or any(k not in KINDS for k in kinds):
        raise ValueError('Distinct known capture kinds required')
    out.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata = out/'ptbxl_database.csv'
        url = 'https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
        with urllib.request.urlopen(url, timeout=45) as response:
            metadata.write_bytes(response.read(10_000_001))
        if metadata.stat().st_size > 10_000_000:
            raise ValueError('Unexpected PTB-XL metadata size')
    with Path(metadata).open() as stream:
        patients = {int(row['ecg_id']): row['patient_id'] for row in csv.DictReader(stream)}
    cases, sources = [], []
    for record in records:
        if record not in patients:
            raise ValueError('Record is absent from PTB-XL metadata')
        stem = f'D0_{record:06d}'
        names = [stem+'.hea', stem+'.dat']+[stem+'_'+KINDS[k][0]+KINDS[k][1] for k in kinds]
        for name in names:
            remote = f'D0/{stem}/{name}'
            url = ('https://www.kaggle.com/api/v1/datasets/download/physionet/ecg-image-database/'
                   +urllib.parse.quote(remote, safe='')+'?datasetVersionNumber=2')
            with urllib.request.urlopen(url, timeout=45) as response:
                body = response.read(20_000_001)
            if len(body) > 20_000_000:
                raise ValueError('Image or signal file exceeds size limit')
            (out/name).write_bytes(body)
            sources.append({'file': name, 'url': url, 'bytes': len(body),
                            'sha256': hashlib.sha256(body).hexdigest()})
            print(json.dumps({'downloaded': name, 'bytes': len(body)}), flush=True)
        header = out/(stem+'.hea')
        full = read_wfdb16(header)
        for kind in kinds:
            suffix, extension = KINDS[kind]
            image = out/(stem+'_'+suffix+extension)
            cases.append({'image_id': image.stem, 'patient_id': patients[record],
                'split': 'development', 'image_type': kind, 'image': image.name,
                'image_sha256': sha256_file(image), 'fs': int(full['II'].fs),
                'reference_header': header.name, 'reference_header_sha256': sha256_file(header),
                'reference_dat_sha256': sha256_file(header.with_suffix('.dat')),
                'source': {'url': 'https://www.kaggle.com/datasets/physionet/ecg-image-database',
                    'dataset_version': 2, 'record': record, 'source_file': f'D0/{stem}/{image.name}',
                    'license': 'CC BY-ND 4.0, source images excluded from redistribution',
                    'upstream_training_overlap': 'unknown'}})
    manifest = {'schema_version': 1, 'selection': 'Declared development record IDs, no held-out claim',
        'metadata_sha256': sha256_file(metadata), 'cases': cases}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--records', nargs='+', type=int, default=[22, 66, 77, 85, 107])
    parser.add_argument('--kinds', nargs='+', choices=tuple(KINDS), default=list(KINDS))
    parser.add_argument('--metadata', type=Path)
    result = prepare(**vars(parser.parse_args()))
    print(json.dumps({'cases': len(result['cases']), 'patients': len({r['patient_id'] for r in result['cases']})}))
