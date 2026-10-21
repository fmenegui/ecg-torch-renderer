"""Validated research manifests and restricted WFDB16 reference loading."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .artifacts import sha256_file
from .signals import LEADS, ROW_LEADS, LeadSignal


@dataclass(frozen=True)
class ImageCase:
    image_id: str
    patient_id: str
    image_type: str
    image: Path
    image_sha256: str
    fs: int
    reference: dict[str, LeadSignal] | None
    source: dict


def read_wfdb16(header):
    """Read the fixed, interleaved 16-bit mV records supplied with ECG images.

    Deliberately reject unsupported WFDB features rather than interpreting
    other encodings, multisegment records or unknown units approximately.
    """
    header = Path(header)
    lines = header.read_text().splitlines()
    if len(lines) < 13:
        raise ValueError('Expected a twelve-lead WFDB header')
    first = lines[0].split()
    if len(first) < 4 or first[1] != '12':
        raise ValueError('Expected twelve channels')
    fs, length = int(first[2]), int(first[3])
    if fs <= 0 or length != 10*fs:
        raise ValueError('Expected a ten-second reference at positive integer fs')
    channels = [line.split() for line in lines[1:13]]
    if any(len(c) != 9 or c[1] != '16' or c[3] != '16' or c[7] != '0' for c in channels):
        raise ValueError('Only interleaved WFDB16 without block encoding is supported')
    if len({c[0] for c in channels}) != 1:
        raise ValueError('One interleaved signal file is required')
    name = channels[0][0]
    if Path(name).name != name:
        raise ValueError('WFDB data filename must be a basename')
    dat = header.parent/name
    if dat.stat().st_size != length*12*2:
        raise ValueError('WFDB byte length disagrees with header')
    adc = np.fromfile(dat, dtype='<i2').reshape(length, 12)
    if (adc == -32768).any():
        raise ValueError('WFDB missing samples are unsupported')
    names, values = [], []
    for col, parts in enumerate(channels):
        gain = re.fullmatch(r'([0-9.]+)\((-?[0-9]+)\)/mV', parts[2])
        if gain is None or float(gain[1]) <= 0:
            raise ValueError('A positive gain and explicit baseline in mV are required')
        if int(adc[0, col]) != int(parts[5]) or int(adc[:, col].astype(np.int64).sum()) % 65536 != int(parts[6]) % 65536:
            raise ValueError('WFDB initial value or checksum mismatch')
        names.append(next((n for n in LEADS if n.upper() == parts[-1].upper()), ''))
        values.append((adc[:, col].astype(np.float64)-int(gain[2]))/float(gain[1]))
    if set(names) != set(LEADS):
        raise ValueError('Missing or duplicated ECG channel names')
    return {name: LeadSignal(v, fs) for name, v in zip(names, values)}


def sequential_reference(full):
    """Select the printed Kaggle 3x4 windows from simultaneous 10-second leads."""
    if set(full) != set(LEADS):
        raise ValueError('Twelve simultaneous reference leads required')
    fs = full['II'].fs
    length = len(full['II'].values)
    if any(v.fs != fs or len(v.values) != length or v.start_s != 0 for v in full.values()):
        raise ValueError('Full reference channels must share their time axis')
    reference = {'II': full['II']}
    boundaries = [round(i*length/4) for i in range(5)]
    for row in ROW_LEADS:
        for col, name in enumerate(row):
            if name == 'II':
                continue
            reference[name] = LeadSignal(full[name].values[boundaries[col]:boundaries[col+1]],
                                         fs, boundaries[col]/fs)
    return reference


def _relative_file(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Manifest input paths must stay inside the manifest directory')
    path = root/p
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or escaping input path: {name}')
    return path


def load_manifest(path):
    """Load hash-pinned images with patient identity and optional waveform truth.

    The manifest may contain train/validation/test splits, but a patient must
    never cross those partitions. Upstream-model training overlap is separate
    provenance and is never inferred from this check.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('Nonempty schema_version=1 cases manifest required')
    cases, ids, patient_splits = [], set(), {}
    for row in data['cases']:
        required = {'image_id', 'patient_id', 'image_type', 'image', 'image_sha256', 'fs', 'source'}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError('Case is missing required manifest fields')
        image_id = row['image_id']
        if not isinstance(image_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', image_id) or '..' in image_id:
            raise ValueError('Invalid image ID')
        if image_id in ids:
            raise ValueError('Duplicate image ID')
        ids.add(image_id)
        patient = row['patient_id']
        if not isinstance(patient, str) or not patient:
            raise ValueError('Explicit nonempty patient identity required')
        split = row.get('split', 'development')
        if split not in ('train', 'validation', 'test', 'development'):
            raise ValueError('Unknown split')
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError('Patient crosses manifest partitions')
        patient_splits[patient] = split
        image = _relative_file(path.parent, row['image'])
        if sha256_file(image) != row['image_sha256']:
            raise ValueError(f'Image checksum mismatch: {image_id}')
        if type(row['fs']) is not int or row['fs'] <= 0:
            raise ValueError('Positive integer fs required')
        reference = None
        if row.get('reference_header') is not None:
            if not {'reference_header_sha256', 'reference_dat_sha256'}.issubset(row):
                raise ValueError('Reference header and data hashes are required')
            header = _relative_file(path.parent, row['reference_header'])
            if sha256_file(header) != row['reference_header_sha256']:
                raise ValueError('Reference header checksum mismatch')
            data_name = header.read_text().splitlines()[1].split()[0]
            dat = _relative_file(header.parent, data_name)
            if sha256_file(dat) != row['reference_dat_sha256']:
                raise ValueError('Reference data checksum mismatch')
            reference = sequential_reference(read_wfdb16(header))
            if reference['II'].fs != row['fs']:
                raise ValueError('Manifest sample rate disagrees with reference')
        source = row.get('source', {})
        if not isinstance(source, dict) or not source.get('url') or not source.get('license'):
            raise ValueError('Explicit source URL and license provenance required')
        cases.append(ImageCase(image_id, patient, str(row['image_type']), image,
                               row['image_sha256'], row['fs'], reference, source))
    return cases
