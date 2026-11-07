"""Archive damage observed in real ECG collections: mold colonies and scorch.

ECG-Image-Database documents printouts that grew mold after weeks in humid
storage and that were scorched brown when wet batches were ironed. These are
painted on the canonical page in material coordinates, so they move with the
paper under every later geometric warp. They change appearance only; waveform
targets keep their geometric ownership, as with the existing liquid stains.
"""
import torch
from torch.nn import functional as F

MOLD_RGB = (.22, .27, .20)
SCORCH_RGB = (.55, .33, .14)


def _solid(page, rgb):
    """Constant color image like ``page``, filled on device without a host copy."""
    return torch.cat([torch.full_like(page[:, :1], value) for value in rgb], 1)


def _grid(h, w, device, dtype):
    ys = (torch.arange(h, device=device, dtype=dtype)[:, None] + .5) / w
    xs = (torch.arange(w, device=device, dtype=dtype)[None, :] + .5) / w
    return ys, xs


def mold(page, active, center_x, center_y, radius, density, generator=None):
    """Clustered fuzzy colonies inside a soft circular region; returns the page and alpha."""
    b, _, h, w = page.shape
    ys, xs = _grid(h, w, page.device, page.dtype)
    cy = (center_y * h / w)[:, None, None]
    region = torch.exp(-((xs - center_x[:, None, None]).square() + (ys - cy).square())
                       / (2 * radius[:, None, None].square()))
    coarse = torch.rand(b, 1, max(4, h // 24), max(4, w // 24), device=page.device, generator=generator)
    fine = torch.rand(b, 1, max(4, h // 6), max(4, w // 6), device=page.device, generator=generator)
    noise = (.65 * F.interpolate(coarse, (h, w), mode='bilinear', align_corners=False)
             + .35 * F.interpolate(fine, (h, w), mode='bilinear', align_corners=False))[:, 0]
    colonies = ((noise - (1 - density[:, None, None])) * 6).clamp(0, 1)
    alpha = (colonies * region * active[:, None, None]).clamp(0, 1)[:, None]
    color = _solid(page, MOLD_RGB)
    return torch.lerp(page, color, .85 * alpha), alpha


def scorch(page, active, center_x, center_y, width, height, angle, strength):
    """Soft rounded-triangle brown burn, darker at its core; multiplies the page."""
    b, _, h, w = page.shape
    ys, xs = _grid(h, w, page.device, page.dtype)
    dx = xs - center_x[:, None, None]
    dy = ys - (center_y * h / w)[:, None, None]
    c, s = angle.cos()[:, None, None], angle.sin()[:, None, None]
    u = (dx * c + dy * s) / width[:, None, None]
    v = (-dx * s + dy * c) / height[:, None, None]
    # An iron sole: narrower toward the tip (positive u).
    taper = 1 - .45 * (u.clamp(-1, 1) + 1) / 2
    distance = (u.square() + (v / taper).square()).sqrt()
    core = (1 - distance).clamp(0, 1)
    alpha = (core.sqrt() * strength[:, None, None] * active[:, None, None]).clamp(0, 1)[:, None]
    tint = _solid(page, SCORCH_RGB)
    darker = torch.lerp(torch.ones_like(page), tint, alpha) * (1 - .35 * alpha * core[:, None])
    return page * darker, alpha
