"""Observed positions of the page corners, for rectification models.

The capture field maps every output pixel to a canonical page coordinate
(normalized, ``align_corners=False``: -1 and 1 are the outer page edges).
The page corners are the canonical points (-1,-1), (1,-1), (1,1) and (-1,1).
Their observed positions are found by a dense nearest-pixel search followed by
Newton steps on the bilinearly interpolated field, entirely on device.

Positions use pixel-center coordinates: pixel ``i`` spans ``[i-0.5, i+0.5]``,
so an unwarped page has corners at (-0.5,-0.5) and (W-0.5,H-0.5). A corner is
``in_frame`` when it lies inside the image and the solve converged. Occlusion
by objects or missing-corner damage is not considered: the corner is the
geometric corner of the sheet.
"""
import torch
from torch.nn import functional as F

CANONICAL_CORNERS = ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))


def _sample(field_chw, x, y):
    """Field values ``[B,K,2]`` at pixel-center positions ``x, y`` ``[B,K]``."""
    _, _, h, w = field_chw.shape
    grid = torch.stack([(x + .5) / w * 2 - 1, (y + .5) / h * 2 - 1], -1)[:, :, None]
    return F.grid_sample(field_chw, grid, mode='bilinear', padding_mode='border',
                         align_corners=False)[..., 0].transpose(1, 2)


def page_corners(field, iterations=3, step=.5):
    """Return ``(corners [B,4,2] xy pixels, in_frame [B,4] bool, residual [B,4])``.

    ``residual`` is the remaining canonical error in output-pixel units along
    the field's local scale, so values well below 1 mean a subpixel solve.
    """
    b, h, w, _ = field.shape
    target = field.new_tensor(CANONICAL_CORNERS)[None].expand(b, -1, -1)
    distance = (field[:, None] - target[:, :, None, None]).square().sum(-1)
    index = distance.flatten(2).argmin(-1)
    y = (index // w).to(field.dtype)
    x = (index % w).to(field.dtype)
    chw = field.permute(0, 3, 1, 2)
    for _ in range(iterations):
        value = _sample(chw, x, y)
        dx = (_sample(chw, x + step, y) - _sample(chw, x - step, y)) / (2 * step)
        dy = (_sample(chw, x, y + step) - _sample(chw, x, y - step)) / (2 * step)
        jacobian = torch.stack([dx, dy], -1)
        error = (target - value)[..., None]
        determinant = jacobian.det()
        safe = determinant.abs() > 1e-12
        jac = torch.where(safe[..., None, None], jacobian, torch.eye(2, device=field.device, dtype=field.dtype))
        delta = torch.linalg.solve(jac, error)[..., 0]
        delta = torch.where(safe[..., None], delta, torch.zeros_like(delta)).clamp(-4, 4)
        x, y = x + delta[..., 0], y + delta[..., 1]
    value = _sample(chw, x, y)
    dx = (_sample(chw, x + step, y) - _sample(chw, x - step, y)) / (2 * step)
    dy = (_sample(chw, x, y + step) - _sample(chw, x, y - step)) / (2 * step)
    scale = .5 * (dx.norm(dim=-1) + dy.norm(dim=-1)).clamp_min(1e-12)
    residual = (target - value).norm(dim=-1) / scale
    inside = (x >= -.5) & (x <= w - .5) & (y >= -.5) & (y <= h - .5)
    return torch.stack([x, y], -1), inside & (residual < .5), residual
