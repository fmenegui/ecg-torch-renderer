"""Offline paired ECG datasets for digitization, detection and segmentation.

Inspired by the task organization of SynthECG (Rahimi et al., 2025/2026).
This is an independent implementation using this project's rendered targets.
All CPU transfers, image encoding and file operations happen outside training.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image
import torch

from .gpu_renderer import _conditional_visibility
from .gpu_supervision import mask_bounding_boxes
from .schema_v05 import LEADS
from .training import detection_targets, lead_name_detection_targets


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _slice(value, index, count):
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
        return value[index:index + 1]
    if isinstance(value, dict):
        return {key: _slice(item, index, count) for key, item in value.items()}
    return value


def _source_fingerprint(reference, rate, length):
    from .gpu_cli import normalized_input_fingerprint
    values = reference[..., :length].detach().cpu().float().numpy()
    return normalized_input_fingerprint(values, np.asarray(rate), None)['sha256']


def paired_samples_from_render(rendered, *, source_ids, group_ids, sample_ids,
                               provenance=None, threshold=.05):
    """Wrap an existing RenderBatch without rerendering any target.

    Render with ``return_instances=True, return_name_instances=True`` and
    stage ``trace`` or ``all``. ``return_field=True`` additionally preserves
    the inverse geometry for photograph-to-time coordinate recovery.
    All recordings/views of a patient must share one caller-supplied group ID.
    References remain original mV, including Cabrera's inverted aVR.
    """
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError('threshold must be finite and in [0,1)')
    count = rendered.image.shape[0]
    if any(len(ids) != count for ids in (source_ids, group_ids, sample_ids)):
        raise ValueError('Exactly one source/group/sample ID required per image')
    if provenance is not None and len(provenance) != count:
        raise ValueError('Exactly one provenance mapping required per image')
    required = {'valid', 'instances', 'name_instances'}
    if required - rendered.targets.keys():
        raise ValueError('Render with stage=trace, return_instances=True and return_name_instances=True')
    info = rendered.metadata
    visibility = _conditional_visibility(rendered.targets['valid'], rendered.targets.get('frame_coverage'))
    instances = rendered.targets['instances'] * visibility
    names = rendered.targets['name_instances'] * visibility
    boxes, present = mask_bounding_boxes(instances, threshold)
    name_boxes, name_present = mask_bounding_boxes(names, threshold)
    tensors = {
        'image': rendered.image, 'valid': rendered.targets['valid'],
        'instance_masks': instances, 'instance_boxes': boxes, 'instance_box_valid': present,
        'instance_active': info['active'], 'instance_lead_ids': info['lead_id'],
        'instance_windows_s': torch.stack((info['start_s'], info['end_s']), -1),
        'instance_amplitude_sign': info['amplitude_sign'], 'name_masks': names,
        'name_boxes': name_boxes, 'name_box_valid': name_present,
        'name_active': info['active'] & (info['parameters']['label_present'] > 0),
        'name_lead_ids': info['lead_id'], 'reference': rendered.reference,
        'sample_rate_hz': info['sample_rate_hz'], 'source_length': info['source_length'],
    }
    records = []
    for index in range(count):
        record = {key: tensor[index:index + 1] for key, tensor in tensors.items()}
        record.update(source_id=source_ids[index], group_id=group_ids[index], sample_id=sample_ids[index],
                      threshold=threshold, metadata=_plain(_slice(info, index, count)),
                      inverse_field=None if rendered.inverse_field is None else rendered.inverse_field[index:index + 1])
        supplied = dict(provenance[index]) if provenance is not None else {}
        fingerprint = _source_fingerprint(record['reference'], float(record['sample_rate_hz'][0]),
                                          int(record['source_length'][0]))
        if supplied.get('source_sha256', fingerprint).lower() != fingerprint:
            raise ValueError('source_sha256 differs from the actual normalized source')
        supplied['source_sha256'] = fingerprint
        record['provenance'] = supplied
        records.append(record)
    return records


def _partitions(records, val_fraction, test_fraction, seed):
    if (not all(math.isfinite(x) and 0 <= x < 1 for x in (val_fraction, test_fraction))
            or val_fraction + test_fraction >= 1):
        raise ValueError('Split fractions must be finite, nonnegative and sum to less than one')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    groups, sources, hashes, names = set(), {}, {}, set()
    for record in records:
        for key in ('source_id', 'group_id', 'sample_id'):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f'{key} must be a nonempty string')
        sid, gid, name = (record[key] for key in ('source_id', 'group_id', 'sample_id'))
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in names:
            raise ValueError('sample_id must be a unique safe filename stem')
        names.add(name)
        groups.add(gid)
        source_hash = record.get('provenance', {}).get('source_sha256')
        if not isinstance(source_hash, str) or not re.fullmatch('[0-9a-f]{64}', source_hash):
            raise ValueError('A normalized source_sha256 is required in each source plan entry')
        for identity, mapping in ((sid, sources), (source_hash, hashes)):
            if identity in mapping and mapping[identity] != gid:
                raise ValueError('Source leakage: a source or source hash has multiple group_ids')
            mapping[identity] = gid
    required = 1 + int(val_fraction > 0) + int(test_fraction > 0)
    if len(groups) < required:
        raise ValueError('Not enough independent groups for the requested train/val/test split')
    ordered = sorted(groups, key=lambda gid: hashlib.sha256(f'{seed}:{gid}'.encode()).hexdigest())
    nval = min(len(groups) - 1 - int(test_fraction > 0), max(1, round(len(groups) * val_fraction))) if val_fraction else 0
    ntest = min(len(groups) - nval - 1, max(1, round(len(groups) * test_fraction))) if test_fraction else 0
    return {gid: 'val' if i < nval else 'test' if i < nval + ntest else 'train'
            for i, gid in enumerate(ordered)}


def _validate(record):
    image, valid, masks = record['image'], record['valid'], record['instance_masks']
    if image.ndim != 4 or image.shape[:2] != (1, 3) or min(image.shape[-2:]) < 1:
        raise ValueError('Each image must be [1,3,H,W]')
    if valid.shape != (1, 1, *image.shape[-2:]):
        raise ValueError('valid must be [1,1,H,W]')
    if masks.ndim != 4 or masks.shape[0] != 1 or masks.shape[-2:] != image.shape[-2:]:
        raise ValueError('instance_masks must be [1,I,H,W]')
    if record['name_masks'].shape != masks.shape:
        raise ValueError('name_masks must match instance_masks')
    for value in (image, valid, masks, record['name_masks']):
        if value.device != image.device or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError('Image and masks must share a device and contain finite values in [0,1]')
    detection_targets(record)
    lead_name_detection_targets(record)
    if record['instance_boxes'].shape[1] != masks.shape[1]:
        raise ValueError('Mask and box instance counts differ')
    if not torch.equal(record['name_lead_ids'], record['instance_lead_ids']):
        raise ValueError('Lead-name and waveform instance identities differ')
    ids = record['instance_lead_ids']
    if ((ids < 0) | (ids >= len(LEADS))).any():
        raise ValueError('Canonical lead IDs must be in 0..11')
    for prefix in ('instance', 'name'):
        expected, present = mask_bounding_boxes(record[prefix + '_masks'], record['threshold'])
        if not torch.equal(expected, record[prefix + '_boxes']) or not torch.equal(present, record[prefix + '_box_valid']):
            raise ValueError('Boxes must match the supplied masks and threshold')
    reference = record['reference']
    rate, length = float(record['sample_rate_hz'][0]), int(record['source_length'][0])
    if (reference.ndim != 3 or reference.shape[:2] != (1, 12) or not torch.isfinite(reference).all()
            or not math.isfinite(rate) or rate <= 0 or not 2 <= length <= reference.shape[-1]):
        raise ValueError('Finite original [1,12,T] reference, source length and positive rate required')
    windows, signs = record['instance_windows_s'], record['instance_amplitude_sign']
    if (windows.shape != (1, masks.shape[1], 2) or not torch.isfinite(windows).all()
            or (windows < 0).any() or (windows[..., 1] < windows[..., 0]).any()
            or (windows > (length - 1) / rate + 1e-4).any()):
        raise ValueError('Instance windows must stay within the original recording')
    if signs.shape != windows.shape[:2] or not ((signs == 1) | (signs == -1)).all():
        raise ValueError('Printed amplitude signs must be +1 or -1')
    if _source_fingerprint(reference, rate, length) != record['provenance']['source_sha256']:
        raise ValueError('Source fingerprint differs from the actual reference')
    field = record['inverse_field']
    if field is not None and (field.shape != (1, *image.shape[-2:], 2) or not torch.isfinite(field).all()):
        raise ValueError('inverse_field must be finite [1,H,W,2]')
    json.dumps({'metadata': record['metadata'], 'provenance': record['provenance']}, allow_nan=False)


def _png(path, value, *, rgb=False):
    data = value.detach().float().cpu()
    if rgb:
        data = data.permute(1, 2, 0)
    Image.fromarray((data.clamp(0, 1) * 255).round().to(torch.uint8).numpy()).save(path)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_paired_dataset(samples, out, *, val_fraction=.2, test_fraction=.2, seed=1973,
                          crop_padding=4, source_plan=None):
    """Export all three tasks with one patient-group partition and provenance.

    Detection has 13 classes: waveform region 0 and names I..V6 as 1..12.
    Segmentation crops retain a single clean instance mask, a validity mask,
    a binary BMP and original, unresampled time samples in JSON. A crop is
    ``with-overlap`` when another instance has ink inside that crop. Exact
    crossing pixels are counted separately. This is not optical readability.
    Existing nonempty outputs are refused. All in-memory records are checked
    before export. For bounded-memory generation, supply a lightweight
    ``source_plan`` with sample/source/group IDs and provenance.source_sha256.
    Each yielded capture is checked against that plan. Output is staged and
    published only after every planned sample succeeds.
    """
    if type(crop_padding) is not int or crop_padding < 0:
        raise ValueError('crop_padding must be a nonnegative integer')
    if source_plan is None:
        records = list(samples)
        for record in records:
            _validate(record)
        plan = records
    else:
        plan = list(source_plan)
        records = samples
    if not plan:
        raise ValueError('At least one sample required')
    partitions = _partitions(plan, val_fraction, test_fraction, seed)
    expected = {row['sample_id']: (row['source_id'], row['group_id'], row['provenance']['source_sha256'])
                for row in plan}

    def checked_records():
        remaining = dict(expected)
        for record in records:
            if source_plan is not None:
                _validate(record)
            name = record['sample_id']
            identity = (record['source_id'], record['group_id'], record['provenance']['source_sha256'])
            if remaining.pop(name, None) != identity:
                raise ValueError('Capture differs from source_plan or was yielded more than once')
            yield record
        if remaining:
            raise ValueError('Missing captures from source_plan')

    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Export output must be absent or an empty directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-staging-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        manifest = _write_dataset(checked_records(), stage, logical_root=out, partitions=partitions,
            val_fraction=val_fraction, test_fraction=test_fraction, seed=seed, crop_padding=crop_padding)
        # Recheck before publication, preserving an output populated by another writer.
        if out.exists():
            if not out.is_dir() or any(out.iterdir()):
                raise FileExistsError('Export output changed while generation was running')
            out.rmdir()
        stage.rename(out)
    return manifest


def _write_dataset(records, out, *, logical_root, partitions, val_fraction, test_fraction, seed, crop_padding):
    for split in ('train', 'val', 'test'):
        for task in ('digitization', 'segmentation', 'detection/images', 'detection/labels'):
            (out / task / split).mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'ecg-paired-datasets-v1', 'seed': seed, 'groups': partitions,
                'split_unit': 'caller-supplied patient or source recording group_id',
                'requested_fractions': {'val': val_fraction, 'test': test_fraction},
                'detection_classes': ['waveform', *LEADS], 'crop_padding': crop_padding,
                'overlap_definition': 'another printed instance has geometric ink inside the crop',
                'reference_scope': 'original printed window, not a guarantee of visible/recoverable samples',
                'samples': [], 'crops': [], 'omitted_instances': []}
    for record in records:
        split, name = partitions[record['group_id']], record['sample_id']
        root = out / 'digitization' / split / name
        root.mkdir()
        _png(root / 'image.png', record['image'][0], rgb=True)
        _png(root / 'valid.png', record['valid'][0, 0])
        length = int(record['source_length'][0])
        rate = float(record['sample_rate_hz'][0])
        source_reference = record['reference'][0, :, :length].detach().cpu()
        # NumPy has no bfloat16. Every bfloat16 value is exactly representable
        # as float32. Preserve float16/32/64 references in their original dtype.
        if source_reference.dtype == torch.bfloat16:
            source_reference = source_reference.float()
        reference = source_reference.numpy()
        np.savez_compressed(root / 'signal.npz', signals=reference, fs=rate,
                            units='mV', lead_names=np.asarray(LEADS))
        _json(root / 'signal.json', {'lead_names': LEADS, 'sample_rate_hz': rate,
                                    'units': 'mV', 'signals': reference.tolist()})
        _json(root / 'metadata.json', {'render': record['metadata'], 'provenance': record['provenance'],
                                      'source_id': record['source_id'], 'group_id': record['group_id'],
                                      'source_tensor_dtype': str(record['reference'].dtype),
                                      'reference_storage_dtype': str(reference.dtype)})
        if record['inverse_field'] is not None:
            np.savez_compressed(root / 'geometry.npz', inverse_field=record['inverse_field'][0].detach().cpu().numpy(),
                                coordinate_convention='align_corners=False normalized canonical xy')
        # The same encoded page is reused for detection rather than rendered again.
        detection_image = out / 'detection/images' / split / (name + '.png')
        detection_image.write_bytes((root / 'image.png').read_bytes())
        waves, names = detection_targets(record), lead_name_detection_targets(record)
        labels = ''.join('0 ' + ' '.join(f'{x:.9f}' for x in box) + '\n'
                         for box in waves['bboxes'].detach().cpu().tolist())
        labels += ''.join(str(int(cls) + 1) + ' ' + ' '.join(f'{x:.9f}' for x in box) + '\n'
                          for cls, box in zip(names['cls'].flatten().detach().cpu().tolist(),
                                              names['bboxes'].detach().cpu().tolist()))
        label_path = out / 'detection/labels' / split / (name + '.txt')
        label_path.write_text(labels, encoding='utf-8')
        entry = {key: record[key] for key in ('source_id', 'group_id', 'sample_id', 'threshold', 'provenance')}
        entry.update(split=split, waveform_slots=waves['instance_idx'].detach().cpu().tolist(),
                     name_slots=names['instance_idx'].detach().cpu().tolist(),
                     files={str(p.relative_to(out)): _hash(p) for p in sorted(root.iterdir())},
                     detection_image=str(detection_image.relative_to(out)),
                     detection_image_sha256=_hash(detection_image),
                     detection_labels=str(label_path.relative_to(out)), detection_labels_sha256=_hash(label_path))
        manifest['samples'].append(entry)
        masks = record['instance_masks'][0].detach().cpu()
        occupied = masks > record['threshold']
        active = record['instance_active'][0].detach().cpu().bool()
        boxes = record['instance_boxes'][0].detach().cpu()
        h, w = record['image'].shape[-2:]
        for slot in range(masks.shape[0]):
            if not active[slot] or not record['instance_box_valid'][0, slot]:
                manifest['omitted_instances'].append({'sample_id': name, 'slot': slot,
                    'reason': 'inactive' if not active[slot] else 'no_visible_trace_above_threshold'})
                continue
            x0, y0, x1, y1 = boxes[slot].tolist()
            x0, y0 = max(0, x0 - crop_padding), max(0, y0 - crop_padding)
            x1, y1 = min(w, x1 + crop_padding), min(h, y1 + crop_padding)
            other = occupied[:, y0:y1, x0:x1].clone()
            other[slot] = False
            foreign_slots = other.flatten(1).any(1).nonzero().flatten().tolist()
            foreign = other.any(0)
            crossing_count = int((foreign & occupied[slot, y0:y1, x0:x1]).sum())
            group = 'with-overlap' if foreign_slots else 'without-overlap'
            crop = out / 'segmentation' / split / group / f'{name}-slot{slot:02d}'
            crop.mkdir(parents=True)
            _png(crop / 'image.png', record['image'][0, :, y0:y1, x0:x1], rgb=True)
            _png(crop / 'mask.png', masks[slot, y0:y1, x0:x1])
            _png(crop / 'valid.png', record['valid'][0, 0, y0:y1, x0:x1])
            Image.fromarray(occupied[slot, y0:y1, x0:x1].numpy()).save(crop / 'mask.bmp')
            start, end = record['instance_windows_s'][0, slot].detach().cpu().tolist()
            indices = np.arange(length)[(np.arange(length) / rate >= start) & (np.arange(length) / rate < end)]
            lead = int(record['instance_lead_ids'][0, slot])
            sign = float(record['instance_amplitude_sign'][0, slot])
            signal = {'source_id': record['source_id'], 'lead_name': LEADS[lead], 'lead_id': lead,
                      'instance_slot': slot, 'window_s': [start, end], 'window_convention': '[start,end)',
                      'sample_rate_hz': rate, 'units': 'mV', 'source_indices': indices.tolist(),
                      'time_s': (indices / rate).tolist(), 'signal_mv': reference[lead, indices].tolist(),
                      'printed_amplitude_sign': sign, 'printed_signal_mv': (sign * reference[lead, indices]).tolist(),
                      'crop_box_xyxy': [x0, y0, x1, y1],
                      'reference_scope': manifest['reference_scope'],
                      'page_metadata': str((root / 'metadata.json').relative_to(out)),
                      'inverse_field_available': record['inverse_field'] is not None}
            _json(crop / 'signal.json', signal)
            manifest['crops'].append({'sample_id': name, 'group_id': record['group_id'], 'split': split,
                'slot': slot, 'lead_id': lead, 'lead_name': LEADS[lead], 'category': group,
                'foreign_instance_slots': foreign_slots, 'foreign_trace_pixels': int(foreign.sum()),
                'crossing_pixels': crossing_count, 'crop_box_xyxy': [x0, y0, x1, y1],
                'files': {str(p.relative_to(out)): _hash(p) for p in sorted(crop.iterdir())}})
    _json(out / 'detection/dataset.yaml', {'path': str(logical_root / 'detection'), 'train': 'images/train',
          'val': 'images/val', 'test': 'images/test', 'nc': 13, 'names': manifest['detection_classes']})
    manifest['counts'] = {'pages': len(manifest['samples']), 'crops': len(manifest['crops']),
        'with_overlap': sum(crop['category'] == 'with-overlap' for crop in manifest['crops']),
        'without_overlap': sum(crop['category'] == 'without-overlap' for crop in manifest['crops'])}
    _json(out / 'manifest.json', manifest)
    return manifest
