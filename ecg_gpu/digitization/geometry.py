"""Explicit pixel-center conventions for sampling ECG paper once."""

import torch
import torch.nn.functional as F


def compose_grid_to_source(grid_xy, homography, *, output_size):
    """Map rectified pixel centers to the rotated source image.

    grid_xy is [rows, columns, 2] in Heng's normalized-image pixel coordinates.
    homography maps rotated-source pixel coordinates to the normalized image.
    Interpolate in that image's space BEFORE applying the inverse projective
    transform. Interpolating projected corners instead is not equivalent.
    """
    grid = torch.as_tensor(grid_xy)
    if grid.dtype not in (torch.float32, torch.float64):
        grid = grid.float()
    h = torch.as_tensor(homography, device=grid.device, dtype=grid.dtype)
    if grid.ndim != 3 or grid.shape[-1] != 2 or min(grid.shape[:2]) < 2:
        raise ValueError('Expected a [rows >= 2, columns >= 2, 2] grid')
    if h.shape != (3, 3) or not torch.isfinite(h).all() or not torch.isfinite(grid).all():
        raise ValueError('Finite grid and 3x3 homography required')
    if len(output_size) != 2 or any(type(v) is not int or v < 2 for v in output_size):
        raise ValueError('Output height and width must be integers >= 2')
    dense = F.interpolate(grid.permute(2, 0, 1)[None], size=output_size,
                          mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
    xy1 = torch.cat((dense, torch.ones_like(dense[..., :1])), -1)
    try:
        source = xy1 @ torch.linalg.inv(h).T
    except RuntimeError as exc:
        raise ValueError('Singular homography') from exc
    if (source[..., 2].abs() < torch.finfo(source.dtype).eps).any():
        raise ValueError('Projective mapping crosses infinity')
    return source[..., :2]/source[..., 2:]


def sample_source(image, source_xy):
    """Sample uint8/float HWC RGB at absolute source pixel centers.

    Return float HWC RGB and a geometric validity mask. Out-of-image positions
    are border filled for reference compatibility and remain invalid.
    """
    rgb = torch.as_tensor(image)
    xy = torch.as_tensor(source_xy, device=rgb.device, dtype=torch.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError('Expected HWC RGB with height and width >= 2')
    if not torch.isfinite(rgb).all():
        raise ValueError('RGB pixels must be finite')
    if xy.ndim != 3 or xy.shape[-1] != 2 or not torch.isfinite(xy).all():
        raise ValueError('Finite HWC sampling coordinates required')
    height, width = rgb.shape[:2]
    valid = ((xy[..., 0] >= 0) & (xy[..., 0] <= width-1)
             & (xy[..., 1] >= 0) & (xy[..., 1] <= height-1))
    scale = xy.new_tensor((width-1, height-1))
    normalized = 2*xy/scale-1
    warped = F.grid_sample(rgb.permute(2, 0, 1)[None].float(), normalized[None],
                           mode='bilinear', padding_mode='border', align_corners=True)
    return warped[0].permute(1, 2, 0), valid
