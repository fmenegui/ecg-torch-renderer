"""On-device adapters for detection and independent-channel segmentation.

No file I/O, host copies, NumPy or scalar reads occur in these helpers.
The padded adapter has fixed shapes. Packing a variable number of detections
uses PyTorch dynamic indexing, which may synchronize CUDA internally. YOLO
labels describe *printed trace instances*, so an extra rhythm strip is another
object with the same canonical lead class. These are detection targets, never
polygons for disconnected waveform segmentation.
"""
import math

import torch
from torch.nn import functional as F

from .schema_v05 import LEADS


def padded_detection_targets(batch, *, min_visible_fraction=0.):
    """Return clipped pixel/normalized boxes and validity without host reads.

    Requires ``instance_boxes``, ``instance_box_valid``, ``instance_active`` and
    ``instance_lead_ids`` from :class:`ECGSupervisionLayer`. Pixel xyxy uses
    exclusive right/bottom edges. Normalized xywh follows Ultralytics. An object
    whose visible trace is empty, cropped out or inactive is excluded. Optional
    visibility filtering requires an explicit ``instance_visible_fraction``.
    This fraction is not inferred from the sparsity of an ECG waveform.
    """
    if not math.isfinite(min_visible_fraction) or not 0 <= min_visible_fraction <= 1:
        raise ValueError('min_visible_fraction must be finite and in [0,1]')
    required = {'image', 'instance_boxes', 'instance_box_valid', 'instance_active', 'instance_lead_ids'}
    if required - batch.keys():
        raise ValueError(f'Missing detection fields: {sorted(required - batch.keys())}')
    image, boxes = batch['image'], batch['instance_boxes']
    if image.ndim != 4 or image.shape[-2] < 1 or image.shape[-1] < 1:
        raise ValueError('image must be nonempty BCHW')
    if boxes.ndim != 3 or boxes.shape[0] != image.shape[0] or boxes.shape[-1] != 4:
        raise ValueError('instance_boxes must have shape [B,I,4]')
    ids = batch['instance_lead_ids']
    flags = [batch['instance_box_valid'], batch['instance_active'], ids]
    if any(value.shape != boxes.shape[:2] for value in flags):
        raise ValueError('Instance flags and class IDs must have shape [B,I]')
    if any(value.device != image.device for value in [boxes, *flags]):
        raise ValueError('All detection tensors must share the image device')
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('instance_lead_ids must be integer canonical class IDs')
    h, w = image.shape[-2:]
    scale = boxes.new_tensor([w, h, w, h], dtype=torch.float32)
    pixel = boxes.to(torch.float32).clamp_min(0).minimum(scale)
    size = pixel[..., 2:] - pixel[..., :2]
    valid = (batch['instance_box_valid'].bool() & batch['instance_active'].bool()
             & (ids >= 0) & (ids < len(LEADS)) & torch.isfinite(boxes).all(-1)
             & (size > 0).all(-1))
    if min_visible_fraction > 0:
        if 'instance_visible_fraction' not in batch:
            raise ValueError('Visibility filtering requires instance_visible_fraction')
        fraction = batch['instance_visible_fraction']
        if fraction.shape != ids.shape or fraction.device != ids.device:
            raise ValueError('instance_visible_fraction must match class IDs and device')
        valid = valid & torch.isfinite(fraction) & (fraction >= min_visible_fraction)
    xywh = torch.cat(((pixel[..., :2] + pixel[..., 2:]) * .5, size), -1) / scale
    return {'boxes': torch.where(valid[..., None], pixel, 0.),
            'bboxes': torch.where(valid[..., None], xywh, 0.),
            'labels': ids.to(torch.int64), 'valid': valid}


def detection_targets(batch, *, format='yolo', min_visible_fraction=0., label_offset=None):
    """Convert a renderer batch to YOLO or torchvision detection targets.

    YOLO returns ``batch_idx[N]``, ``cls[N,1]``, ``bboxes[N,4]`` plus instance
    slot indices. All are tensors on the image device, including empty batches.
    Torchvision returns one dict per image with pixel ``boxes``,
    ``labels``, ``area`` and ``iscrowd``. Torchvision labels default to 1..12
    because 0 is background. Pass ``label_offset=0`` for a custom detector
    accepting canonical 0..11 IDs. YOLO always uses canonical 0..11 IDs.
    """
    if format not in ('yolo', 'torchvision'):
        raise ValueError("format must be 'yolo' or 'torchvision'")
    if label_offset is None:
        label_offset = 1 if format == 'torchvision' else 0
    if label_offset not in (0, 1) or (format == 'yolo' and label_offset != 0):
        raise ValueError('label_offset must be 0 for YOLO, or 0/1 for torchvision')
    target = padded_detection_targets(batch, min_visible_fraction=min_visible_fraction)
    valid = target['valid']
    if format == 'yolo':
        indices = valid.nonzero(as_tuple=False)
        return {'batch_idx': indices[:, 0], 'instance_idx': indices[:, 1],
                'cls': target['labels'][valid][:, None], 'bboxes': target['bboxes'][valid]}
    result = []
    for index in range(batch['image'].shape[0]):
        boxes = target['boxes'][index, valid[index]]
        size = boxes[:, 2:] - boxes[:, :2]
        labels = target['labels'][index, valid[index]] + label_offset
        result.append({'boxes': boxes, 'labels': labels, 'area': size.prod(-1),
                       'iscrowd': torch.zeros_like(labels)})
    return result


def masked_segmentation_loss(logits, targets, valid, *, channel_valid=None, pos_weight=None):
    """BCE for independent lead masks, ignoring invalid/occluded pixels.

    ``targets`` and ``logits`` are BCHW, ``valid`` is B1HW or BCHW. Optional
    ``channel_valid[B,C]`` ignores channels for task-specific missing targets.
    An entirely ignored image has finite zero loss and zero model gradients.
    Using renderer ``valid`` restricts loss to observable paper and ignores
    the surrounding table. Supply an explicit all-ones or task-specific support
    if background negatives should also contribute. This is a task choice.
    Keep the renderer float32. The network may run under mixed precision.
    """
    if logits.ndim != 4 or logits.shape != targets.shape:
        raise ValueError('logits and targets must have matching BCHW shapes')
    if valid.shape not in (logits.shape, (logits.shape[0], 1, *logits.shape[-2:])):
        raise ValueError('valid must have shape B1HW or BCHW')
    if any(value.device != logits.device for value in (targets, valid)):
        raise ValueError('Segmentation tensors must share one device')
    support = valid.to(torch.float32).expand_as(logits)
    if channel_valid is not None:
        if channel_valid.shape != logits.shape[:2] or channel_valid.device != logits.device:
            raise ValueError('channel_valid must be [B,C] on the logits device')
        support = support * channel_valid[..., None, None]
    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, device=logits.device, dtype=torch.float32)
        if pos_weight.ndim == 1 and pos_weight.shape[0] == logits.shape[1]:
            pos_weight = pos_weight[:, None, None]
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets.float(),
                                                reduction='none', pos_weight=pos_weight)
    return (losses * support).sum() / support.sum().clamp_min(1.)


def lead_name_detection_targets(batch, *, format='yolo', label_offset=None):
    """Detect each printed lead-name occurrence, including repeated rhythm labels.

    Classes are canonical 0..11 in YOLO and 1..12 in torchvision. Inverted
    Cabrera labels retain canonical aVR identity. ``name_amplitude_sign``
    records their printed polarity. Boxes describe geometric glyph support,
    not optical readability. No host copies occur here.
    """
    required = {'image', 'name_boxes', 'name_box_valid', 'name_active', 'name_lead_ids'}
    if required - batch.keys():
        raise ValueError(f'Missing lead-name fields: {sorted(required - batch.keys())}')
    mapped = {'image': batch['image'], 'instance_boxes': batch['name_boxes'],
              'instance_box_valid': batch['name_box_valid'],
              'instance_active': batch['name_active'], 'instance_lead_ids': batch['name_lead_ids']}
    return detection_targets(mapped, format=format, label_offset=label_offset)
