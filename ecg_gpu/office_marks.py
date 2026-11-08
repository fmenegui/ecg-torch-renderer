"""Office marks on archived ECG pages: barcodes, QR-style codes and ink stamps.

Hospital printouts carry patient barcodes and QR labels, and archived copies
are often stamped ("received", department seals) in red, blue or violet ink,
frequently across the traces. These are painted on the canonical page in
physical millimeters before reproduction and capture. Codes encode random
bits only and are not decodable; QR-style codes carry the three finder
patterns that detectors key on. The union of marks is returned as an
``office_marks`` target; waveform targets are unchanged.
"""
import torch
from torch.nn import functional as F

BARCODE_MODULES = 72
QR_MODULES = 25
STAMP_SEGMENTS = 36
STAMP_INKS = ((.72, .12, .16), (.14, .22, .62), (.42, .18, .55))


def _coords(h, w, device, dtype):
    ys = torch.arange(h, device=device, dtype=dtype)[None, :, None]
    xs = torch.arange(w, device=device, dtype=dtype)[None, None, :]
    return ys, xs


def barcode(shape, active, x, y, width, height, bits):
    """1-D bar pattern; ``bits`` ``[B,M]`` in {0,1}; geometry in pixels ``[B]``."""
    b, h, w = shape
    ys, xs = _coords(h, w, bits.device, x.dtype)
    u = (xs - x[:, None, None]) / width[:, None, None]
    v = (ys - y[:, None, None]) / height[:, None, None]
    inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    module = (u.clamp(0, .9999) * bits.shape[1]).long().expand(b, h, w)
    bar = bits.gather(1, module.reshape(b, -1)).reshape(b, h, w)
    return (inside & (bar > 0)).to(x.dtype) * active[:, None, None]


def qr_code(shape, active, x, y, size, bits):
    """Square code with three finder patterns; ``bits`` ``[B,N,N]``."""
    b, h, w = shape
    n = bits.shape[-1]
    ys, xs = _coords(h, w, bits.device, x.dtype)
    u = (xs - x[:, None, None]) / size[:, None, None]
    v = (ys - y[:, None, None]) / size[:, None, None]
    inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    i = (v.clamp(0, .9999) * n).long().expand(b, h, w)
    j = (u.clamp(0, .9999) * n).long().expand(b, h, w)
    value = bits.reshape(b, -1).gather(1, (i * n + j).reshape(b, -1)).reshape(b, h, w)
    for oi, oj in ((0, 0), (0, n - 7), (n - 7, 0)):
        di, dj = i - oi, j - oj
        region = (di >= 0) & (di < 8) & (dj >= 0) & (dj < 8)
        ring = torch.maximum((di - 3).abs(), (dj - 3).abs())
        finder = ((ring == 3) | (ring <= 1)) & (di < 7) & (dj < 7)
        value = torch.where(region, finder.long(), value)
    return (inside & (value > 0)).to(x.dtype) * active[:, None, None]


def stamp(shape, active, x, y, radius, segments, generator=None):
    """Double ring with a patterned band between rings; patchy ink coverage."""
    b, h, w = shape
    ys, xs = _coords(h, w, segments.device, x.dtype)
    dx, dy = xs - x[:, None, None], ys - y[:, None, None]
    r = radius[:, None, None]
    distance = (dx.square() + dy.square()).sqrt()
    stroke = (r * .045).clamp_min(.8)
    rings = ((stroke - (distance - r).abs()).clamp(0, 1)
             + (stroke - (distance - .74 * r).abs()).clamp(0, 1)
             + (stroke - (distance - .2 * r).abs()).clamp(0, 1)).clamp(0, 1)
    angle = torch.atan2(dy, dx) / (2 * torch.pi) + .5
    index = (angle.clamp(0, .9999) * segments.shape[1]).long()
    band = segments.gather(1, index.reshape(b, -1)).reshape(b, h, w)
    letters = ((distance > .8 * r) & (distance < .94 * r) & (band > 0)).to(x.dtype)
    noise = torch.rand(b, 1, max(4, h // 8), max(4, w // 8), device=segments.device, generator=generator)
    patchy = .55 + .45 * F.interpolate(noise, (h, w), mode='bilinear', align_corners=False)[:, 0]
    return (torch.maximum(rings, letters) * patchy).clamp(0, 1) * active[:, None, None]


def sample_office_marks(b, rand):
    """Per-record controls; ``rand(*shape)`` draws uniforms on device."""
    return {
        'barcode_active': (rand(b) < .3).float(), 'barcode_x': .55 + rand(b) * .25, 'barcode_y': .01 + rand(b) * .05,
        'barcode_width_mm': 35 + rand(b) * 25, 'barcode_height_mm': 7 + rand(b) * 6,
        'barcode_bits': (rand(b, BARCODE_MODULES) < .5).long(),
        'qr_active': (rand(b) < .15).float(), 'qr_x': .82 + rand(b) * .1, 'qr_y': .01 + rand(b) * .05,
        'qr_size_mm': 14 + rand(b) * 8, 'qr_bits': (rand(b, QR_MODULES, QR_MODULES) < .5).long(),
        'stamp_active': (rand(b) < .25).float(), 'stamp_x': .1 + rand(b) * .8, 'stamp_y': .15 + rand(b) * .7,
        'stamp_radius_mm': 12 + rand(b) * 8, 'stamp_segments': (rand(b, STAMP_SEGMENTS) < .6).long(),
        'stamp_ink': (rand(b) * len(STAMP_INKS)).long().clamp(max=len(STAMP_INKS) - 1),
        'stamp_strength': .45 + rand(b) * .4,
    }


def apply_office_marks(page, p, ppm, generator=None):
    """Paint marks on ``page`` ``[B,3,H,W]``; return the page and the union alpha."""
    b, _, h, w = page.shape
    shape = (b, h, w)
    codes = torch.maximum(
        barcode(shape, p['barcode_active'], p['barcode_x'] * w, p['barcode_y'] * h,
                p['barcode_width_mm'] * ppm, p['barcode_height_mm'] * ppm, p['barcode_bits']),
        qr_code(shape, p['qr_active'], p['qr_x'] * w, p['qr_y'] * h, p['qr_size_mm'] * ppm, p['qr_bits']))
    stamped = stamp(shape, p['stamp_active'], p['stamp_x'] * w, p['stamp_y'] * h,
                    p['stamp_radius_mm'] * ppm, p['stamp_segments'], generator)
    page = page * (1 - .92 * codes[:, None])
    ink = p['stamp_ink'][:, None, None, None]
    channels = []
    for channel in range(3):
        values = [color[channel] for color in STAMP_INKS]
        channels.append(torch.where(ink == 0, values[0], torch.where(ink == 1, values[1], values[2])))
    color = torch.cat(channels, 1).to(page.dtype)
    alpha = (stamped * p['stamp_strength'][:, None, None])[:, None]
    # Stamp ink is translucent: it multiplies the page, so traces show through.
    page = page * torch.lerp(torch.ones_like(page), color.expand_as(page), alpha)
    return page, torch.maximum(codes, stamped)[:, None]
