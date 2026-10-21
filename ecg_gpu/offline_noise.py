"""Deterministic observation-only corruptions for cached research datasets.

These parameterized stress recipes are not calibrated camera models. They do
not modify waveform truth, image coordinates or grid identities. Opaque patches
also remove supervision support so hidden content is not a restoration target.
"""
import io

import numpy as np
from PIL import Image, ImageFilter

FAMILIES = ('none', 'sensor', 'blur_motion', 'jpeg_resolution', 'illumination',
            'faded_print', 'color_gamma', 'scanner_banding', 'stain', 'occlusion',
            'impulse', 'moire')


def corrupt_observation(rgb, valid, *, family, severity, seed):
    if family not in FAMILIES or not 0 <= severity <= 1:
        raise ValueError('Known family and severity in [0,1] required')
    rgb, valid = np.asarray(rgb), np.asarray(valid, np.float32).copy()
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or valid.shape != rgb.shape[:2] or rgb.dtype != np.uint8:
        raise ValueError('uint8 RGB and corresponding visibility required')
    rng = np.random.default_rng(seed)
    s = float(severity)
    image = rgb.astype(np.float32)/255
    h, w = valid.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    x, y = x/max(1, w-1), y/max(1, h-1)
    if family == 'sensor':
        image += rng.normal(size=image.shape)*np.sqrt((.004+.045*s)**2+image*.008*s)
    elif family == 'blur_motion':
        image = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(.2+1.8*s))).astype(np.float32)/255
        import cv2
        size = 3+2*int(s*3)
        kernel = np.zeros((size, size), np.float32)
        kernel[size//2] = 1/size
        image = cv2.filter2D(image, -1, kernel)
    elif family == 'jpeg_resolution':
        small = Image.fromarray(rgb).resize((max(32, round(w*(.9-.5*s))), max(32, round(h*(.9-.5*s)))), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        small.save(buffer, format='JPEG', quality=round(95-65*s))
        image = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB').resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32)/255
    elif family == 'illumination':
        center = rng.uniform(-.2, 1.2, 2)
        falloff = np.clip(1-s*.8*((x-center[0])**2+(y-center[1])**2), .12, 1)
        boundary = rng.uniform(.2, .8)
        shadow = 1-s*.65/(1+np.exp(np.clip((x+y-boundary)*25, -50, 50)))
        image *= (falloff*shadow)[..., None]
    elif family == 'faded_print':
        image = 1-(1-image)*(1-.75*s)
    elif family == 'color_gamma':
        image = np.clip(image*rng.uniform(.75, 1.18, (1, 1, 3)), 0, 1)**(.65+1.1*s)
    elif family == 'scanner_banding':
        image += (.01+.1*s)*np.sin(y[..., None]*rng.uniform(15, 110)*2*np.pi)
    elif family == 'stain':
        cx, cy = rng.uniform(.15, .85, 2)
        stain = np.exp(-((x-cx)**2/.08+(y-cy)**2/.035))*.7*s
        image *= 1-stain[..., None]*np.array([.12, .5, .8])
    elif family == 'occlusion':
        cx, cy = rng.uniform(.1, .8, 2)
        bw, bh = .03+.15*s, .03+.08*s
        mask = (x > cx) & (x < cx+bw) & (y > cy) & (y < cy+bh)
        image[mask] = rng.uniform(.02, .9, 3)
        valid[mask] = 0
    elif family == 'impulse':
        mask = rng.random((h, w)) < .002+.025*s
        image[mask] = rng.integers(0, 2, (int(mask.sum()), 1))
    elif family == 'moire':
        wave = np.sin((x*rng.uniform(80, 190)+y*rng.uniform(5, 30))*2*np.pi)
        image += (.02+.08*s)*wave[..., None]*np.array([1., -.5, .5])
    # All recipes include a small, explicitly recorded sensor perturbation.
    if family != 'none':
        image += rng.normal(0, .001+.004*s, image.shape)
    return (np.clip(image, 0, 1)*255).round().astype(np.uint8), valid, {
        'family': family, 'severity': s, 'seed': int(seed),
        'coordinates_changed': False, 'physiology_changed': False,
        'opaque_support_removed': family == 'occlusion', 'calibrated_camera_model': False}
