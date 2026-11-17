"""Training pairs for the Open-ECG-Digitizer segmentation network (npj Digital Medicine).

The upstream network (Ahus-AIM/Open-ECG-Digitizer, pinned in
``reproductions/npj``) trains on ``NNNNNN.png`` images with ``NNNNNN_mask.png``
masks. Mask channels are soft labels in [0, 1]: red = grid, green = text,
blue = ECG signal; background is the remainder (``src/loss/loss.py``,
``rgb_to_one_hot``). Upstream ``RefineMask`` clears grid and text wherever
the signal is present, so overlaps resolve to signal.

This module converts a renderer batch into those pairs so the same network can
be trained on images from this generator and compared with other generators on
real images. Rendering must request ``stage='all'`` with
``RenderConfig(text_target=True)`` so grid and full text targets exist.
Calibration pulses are drawn with the recorder pen; whether the upstream
training data labels them as signal is not documented, so ``pulse_class``
makes that choice explicit and recordable.
"""
from pathlib import Path

import numpy as np

PULSE_CLASSES = ('signal', 'text', 'background')


def npj_mask(targets, *, pulse_class='signal'):
    """Return a ``[B,3,H,W]`` float mask (grid, text, signal) from renderer targets."""
    import torch
    if pulse_class not in PULSE_CLASSES:
        raise ValueError(f'pulse_class must be one of {PULSE_CLASSES}')
    required = ['traces', 'grid', 'text']
    if pulse_class != 'background':
        required.append('calibration_pulses')
    missing = [name for name in required if name not in targets]
    if missing:
        raise ValueError(f'Missing targets {missing}. Render with stage="all" and RenderConfig(text_target=True'
                         + (', calibration_target=True' if 'calibration_pulses' in missing else '') + ')')
    signal = targets['traces'].amax(1, keepdim=True)
    text = targets['text']
    grid = targets['grid'][:, :1]
    if pulse_class == 'signal':
        signal = torch.maximum(signal, targets['calibration_pulses'])
    elif pulse_class == 'text':
        text = torch.maximum(text, targets['calibration_pulses'])
    mask = torch.cat([grid, text, signal], 1).clamp(0, 1)
    if 'valid' in targets:
        # Outside the visible sheet everything is background.
        mask = mask * targets['valid'][:, :1].clamp(0, 1)
    return mask


def _to_uint8(array):
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def write_npj_pairs(images, masks, out, start_index=1, compress_level=None):
    """Write ``[B,3,H,W]`` images and masks as ``NNNNNN.png`` / ``NNNNNN_mask.png``.

    Returns the written stems. Existing files are never overwritten.
    """
    from PIL import Image
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = images.detach().float().cpu().numpy()
    masks = masks.detach().float().cpu().numpy()
    if images.shape != masks.shape or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f'Matching [B,3,H,W] images and masks required, got {images.shape} and {masks.shape}')
    stems = []
    for offset, (image, mask) in enumerate(zip(images, masks)):
        stem = f'{start_index + offset:06d}'
        image_path, mask_path = out / f'{stem}.png', out / f'{stem}_mask.png'
        if image_path.exists() or mask_path.exists():
            raise FileExistsError(f'Refusing to overwrite {image_path}')
        options = {} if compress_level is None else {'compress_level': compress_level}
        Image.fromarray(_to_uint8(image.transpose(1, 2, 0))).save(image_path, **options)
        Image.fromarray(_to_uint8(mask.transpose(1, 2, 0))).save(mask_path, **options)
        stems.append(stem)
    return stems
