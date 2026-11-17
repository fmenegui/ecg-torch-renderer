"""Footprint-aware resampling of the canonical page (trilinear mip pyramid).

The canonical document is drawn at output resolution, then warped by the
capture field. Plain bilinear grid_sample point-samples it: where the page is
minified (perspective, a 90-degree rotation fitted into the frame, zoom-out)
one output pixel covers several canonical pixels and fine grid lines alias
into moire or vanish. A camera pixel integrates its whole footprint instead.

This module estimates the footprint from the field Jacobian and blends
bilinear samples of an antialiased pyramid. With no minification the result is
exactly the plain bilinear sample. It changes the RGB observation only.
"""
import torch
from torch.nn import functional as F


def pyramid(image, levels):
    """[B,C,H,W] -> list of progressively half-resolution antialiased images."""
    result = [image]
    for _ in range(levels):
        h, w = result[-1].shape[-2:]
        if min(h, w) < 4:
            break
        result.append(F.interpolate(result[-1], size=((h + 1) // 2, (w + 1) // 2), mode='bilinear',
                                    antialias=True, align_corners=False))
    return result


def footprint_level(field, canonical_size):
    """log2 of the canonical-pixel footprint per output pixel, [B,1,H,W] >= 0.

    The field maps output pixels to normalized canonical coordinates. Its
    Jacobian determinant times the canonical pixel count per normalized area
    gives canonical pixels covered by one output pixel.
    """
    h, w = canonical_size
    dy, dx = torch.gradient(field, dim=(1, 2))
    du = torch.stack((dx[..., 0] * w / 2, dx[..., 1] * h / 2), -1)
    dv = torch.stack((dy[..., 0] * w / 2, dy[..., 1] * h / 2), -1)
    # Isotropic estimate: geometric mean of the two axis footprints.
    area = (du[..., 0] * dv[..., 1] - du[..., 1] * dv[..., 0]).abs()
    # A small dead zone keeps unit-scale rotations exactly bilinear despite rounding.
    return (area.clamp_min(1e-12).sqrt().log2() - 1e-3).clamp_min(0)[:, None]


def prefiltered_sample(image, field, *, levels=3, padding_mode='border'):
    """Trilinear mip sampling, identical to bilinear where nothing is minified."""
    stack = pyramid(image, levels)
    lod = footprint_level(field, image.shape[-2:]).clamp(max=len(stack) - 1)
    samples = [F.grid_sample(level, field, align_corners=False, padding_mode=padding_mode) for level in stack]
    result = samples[0]
    for index in range(1, len(samples)):
        weight = (lod - (index - 1)).clamp(0, 1)
        result = torch.lerp(result, samples[index], weight)
    return result
