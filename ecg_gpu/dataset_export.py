"""Explicit offline PNG/YOLO export with patient/source grouping provenance.

This module does CPU transfers and disk I/O. Use ``ecg_gpu.training`` for the
on-device hot path. Dense waveform masks are deliberately not coerced into YOLO
segmentation polygons: disconnected traces are not a single filled contour.
"""
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image
import torch

from .schema_v05 import LEADS
from .training import detection_targets


def samples_from_batch(batch, *, source_ids, group_ids, sample_ids, provenance=None):
    """Wrap a supervision batch as export records without copying image tensors.

    ``group_ids`` must identify the patient when available, otherwise the source
    recording. All augmentations/windows from that patient must share one ID.
    ``source_ids`` identify original recordings, never a randomized rendering.
    Provenance can hold hashes, sampling rate, renderer specification and seeds.
    """
    count = batch['image'].shape[0]
    if any(len(value) != count for value in (source_ids, group_ids, sample_ids)):
        raise ValueError('Exactly one source/group/sample ID required per image')
    if provenance is not None and len(provenance) != count:
        raise ValueError('Exactly one provenance mapping required per image')
    required = ('image', 'instance_boxes', 'instance_box_valid', 'instance_active', 'instance_lead_ids')
    if 'instance_visible_fraction' in batch:
        required += ('instance_visible_fraction',)
    return [dict({key: batch[key][i:i+1] for key in required},
                 source_id=source_ids[i], group_id=group_ids[i], sample_id=sample_ids[i],
                 provenance=(provenance[i] if provenance is not None else {})) for i in range(count)]


def export_yolo_dataset(samples, out, *, val_fraction=.2, seed=1973, min_visible_fraction=0.):
    """Write an Ultralytics *detection* dataset and auditable JSON manifest.

    Samples are records from ``samples_from_batch``. No model downloads happen.
    Existing nonempty output directories are refused. A source ID or supplied
    source SHA256 cannot occur under two group IDs. Splitting occurs by group,
    before image variants are written. This cannot detect falsely supplied
    patient IDs, so grouping remains the data curator's responsibility.
    """
    if not math.isfinite(val_fraction) or not 0 <= val_fraction < 1:
        raise ValueError('val_fraction must be finite and in [0,1)')
    if not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    records = list(samples)
    if not records:
        raise ValueError('At least one sample is required')
    groups, sources, hashes, sample_ids = set(), {}, {}, set()
    for record in records:
        for key in ('source_id', 'group_id', 'sample_id'):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f'{key} must be a nonempty string')
        sid, gid, name = record['source_id'], record['group_id'], record['sample_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in ('.', '..'):
            raise ValueError('sample_id must be a safe filename stem')
        if name in sample_ids:
            raise ValueError('sample_id must be unique')
        sample_ids.add(name)
        groups.add(gid)
        if sid in sources and sources[sid] != gid:
            raise ValueError('Source leakage: one source_id has multiple group_ids')
        sources[sid] = gid
        if not isinstance(record.get('provenance', {}), dict):
            raise ValueError('provenance must be a JSON-serializable mapping')
        source_hash = record.get('provenance', {}).get('source_sha256')
        if source_hash:
            if not re.fullmatch('[0-9a-fA-F]{64}', source_hash):
                raise ValueError('source_sha256 must be a SHA256 hexadecimal string')
            source_hash = source_hash.lower()
            if source_hash in hashes and hashes[source_hash] != gid:
                raise ValueError('Source leakage: one source_sha256 has multiple group_ids')
            hashes[source_hash] = gid
        # Preflight shapes/labels and provenance before creating any files.
        if record['image'].shape[0] != 1 or record['image'].shape[1] != 3:
            raise ValueError('Each export record must contain image [1,3,H,W]')
        detection_targets(record, min_visible_fraction=min_visible_fraction)
        json.dumps(record.get('provenance', {}), allow_nan=False)
        image = record['image']
        if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError('Export RGB pixels must be finite and in [0,1]')
    if val_fraction > 0 and len(groups) < 2:
        raise ValueError('At least two independent groups required for train/val split')
    ordered = sorted(groups, key=lambda gid: hashlib.sha256(f'{seed}:{gid}'.encode()).hexdigest())
    nval = min(len(groups)-1, max(1, round(len(groups)*val_fraction))) if val_fraction else 0
    validation = set(ordered[:nval])
    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Export output must be absent or an empty directory')
    for split in ('train', 'val'):
        (out/'images'/split).mkdir(parents=True, exist_ok=True)
        (out/'labels'/split).mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'ecg-yolo-detection-v1', 'task': 'detect',
                'class_names': LEADS, 'seed': seed, 'requested_val_fraction': val_fraction,
                'min_visible_fraction': min_visible_fraction,
                'box_definition': 'visible trace instance, xyxy exclusive pixel edges before normalized xywh',
                'split_unit': 'caller-supplied patient or source recording group_id',
                'groups': {gid: 'val' if gid in validation else 'train' for gid in sorted(groups)},
                'samples': []}
    for record in records:
        split = 'val' if record['group_id'] in validation else 'train'
        name = record['sample_id']
        image_path, label_path = Path('images')/split/(name+'.png'), Path('labels')/split/(name+'.txt')
        pixels = (record['image'][0].detach().float().cpu().permute(1, 2, 0)*255).round().to(torch.uint8).numpy()
        Image.fromarray(pixels).save(out/image_path)
        target = detection_targets(record, min_visible_fraction=min_visible_fraction)
        classes = target['cls'].detach().cpu().flatten().tolist()
        boxes = target['bboxes'].detach().cpu().tolist()
        slots = target['instance_idx'].detach().cpu().tolist()
        labels = ''.join(str(cls)+' '+ ' '.join(f'{value:.9f}' for value in box)+'\n'
                         for cls, box in zip(classes, boxes))
        (out/label_path).write_text(labels, encoding='utf-8')
        manifest['samples'].append({'sample_id': name, 'source_id': record['source_id'],
            'group_id': record['group_id'], 'split': split, 'image': image_path.as_posix(),
            'labels': label_path.as_posix(), 'image_sha256': hashlib.sha256((out/image_path).read_bytes()).hexdigest(),
            'label_sha256': hashlib.sha256(labels.encode()).hexdigest(), 'instance_slots': slots,
            'num_objects': len(classes), 'provenance': record.get('provenance', {})})
    config = {'path': str(out), 'train': 'images/train', 'val': 'images/val',
              'nc': len(LEADS), 'names': LEADS}
    # JSON is a YAML 1.2 subset and avoids a separate YAML runtime dependency.
    (out/'dataset.yaml').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return manifest

