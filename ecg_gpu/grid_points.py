"""Identified 5 mm lattice points from the observation's inverse material map.

Coordinates are pixel centers, with (0, 0) at the top-left pixel center.
No detection from RGB is involved. Inverting a sampled map is approximate, so
every accepted point must pass an explicit material-space round-trip test.
All batch work stays in torch on the input device, with fixed padded shapes.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class GridPointConfig:
    """Capacity and observation-space heatmap settings for a 5 mm lattice.

    Increase capacity for unusually large sheets. Capacity overflow raises,
    never silently truncates. ``sigma_pixels`` describes the training target,
    not printed dots or an optical PSF. Visibility is geometric, not readability.
    """
    rows: int = 128
    columns: int = 128
    sigma_pixels: float = .8
    max_error_pixels: float = .01
    min_visibility: float = .99
    mask_shape: str = 'gaussian'
    radius_pixels: float | None = None
    refinement_steps: int = 4
    max_image_error_pixels: float = .01
    return_offsets: bool = False

    def __post_init__(self):
        for name in ('rows', 'columns'):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 1024:
                raise ValueError('Grid point rows/columns must be integers in 1..1024')
        for name in ('sigma_pixels', 'max_error_pixels', 'min_visibility', 'max_image_error_pixels'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.sigma_pixels > 4 or self.min_visibility > 1:
            raise ValueError('Use sigma_pixels <=4 and min_visibility <=1')
        if self.mask_shape not in ('gaussian', 'disk'):
            raise ValueError('mask_shape must be gaussian or disk')
        if self.radius_pixels is not None and (isinstance(self.radius_pixels, bool)
                or not math.isfinite(self.radius_pixels) or not 0 < self.radius_pixels <= 16):
            raise ValueError('radius_pixels must be finite and in (0,16]')
        if self.mask_shape == 'disk' and self.radius_pixels is None:
            raise ValueError('A disk requires an explicit radius_pixels')
        if type(self.refinement_steps) is not int or not 0 <= self.refinement_steps <= 8:
            raise ValueError('refinement_steps must be an integer in 0..8')
        if type(self.return_offsets) is not bool:
            raise ValueError('return_offsets must be bool')


def identity_field(batch, height, width, *, device):
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                          torch.arange(width, device=device, dtype=torch.float32), indexing='ij')
    return torch.stack(((x+.5)*2/width-1, (y+.5)*2/height-1), -1)[None].expand(batch, -1, -1, -1)


def rasterize_grid_points(xy, present, valid, *, config=None):
    """Draw Gaussian/disc targets around floating centers without moving them.

    Returns a mask plus optional sparse nearest-pixel offset targets. Offset
    collisions are ignored explicitly. Work is bounded to 64 offsets per
    splat chunk, rather than allocating B*N*(2*radius+1)^2 full patches.
    This helper permits changing the target size without rerendering RGB.
    """
    config = config or GridPointConfig()
    if not isinstance(config, GridPointConfig):
        raise TypeError('config must be GridPointConfig')
    if xy.ndim != 3 or xy.shape[-1] != 2 or present.shape != xy.shape[:2]:
        raise ValueError('xy/present must have shapes [B,N,2] and [B,N]')
    if valid.ndim != 4 or valid.shape[:2] != (xy.shape[0], 1) or min(valid.shape[-2:]) < 1:
        raise ValueError('valid must have shape [B,1,H,W]')
    if xy.device != valid.device or present.device != xy.device:
        raise ValueError('Point tensors must share one device')
    if xy.dtype != torch.float32 or valid.dtype != torch.float32 or present.dtype != torch.bool:
        raise ValueError('Use float32 xy/valid and boolean present')
    b, _, h, w = valid.shape
    torch._assert_async((torch.isfinite(xy).all(-1) | ~present).all(), 'Visible point coordinates must be finite')
    present = present & (xy[..., 0] >= 0) & (xy[..., 0] <= w-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h-1)
    xy = torch.where(present[..., None], xy, -1.)
    radius = math.ceil(config.radius_pixels if config.radius_pixels is not None else 3*config.sigma_pixels)
    offsets = torch.arange(-radius, radius+1, device=xy.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
    offset = torch.stack((ox.flatten(), oy.flatten()), -1)
    result = torch.zeros(b, h*w, device=xy.device)
    anchor = xy.round().long()
    for first in range(0, len(offset), 64):
        pixel = anchor[..., None, :] + offset[first:first+64]
        distance2 = (pixel.to(xy.dtype) - xy[..., None, :]).square().sum(-1)
        value = (torch.exp(-distance2/(2*config.sigma_pixels**2)) if config.mask_shape == 'gaussian'
                 else torch.ones_like(distance2))
        if config.radius_pixels is not None:
            value = value*(distance2 <= config.radius_pixels**2)
        x, y = pixel.unbind(-1)
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h) & present[..., None]
        index = y.clamp(0, h-1)*w + x.clamp(0, w-1)
        result.scatter_reduce_(1, index.flatten(1), (value*inside).flatten(1), reduce='amax')
    outputs = {'grid_points_mask':result.reshape(b, 1, h, w)*valid}
    if config.return_offsets:
        index = anchor[..., 1].clamp(0, h-1)*w + anchor[..., 0].clamp(0, w-1)
        counts = torch.zeros(b, h*w, device=xy.device, dtype=torch.long)
        counts.scatter_add_(1, index, present.long())
        usable = present & (counts.gather(1, index) == 1)
        support = valid.flatten(1).gather(1, index) >= config.min_visibility
        usable = usable & support
        offsets = torch.zeros(b, 2, h*w, device=xy.device)
        values = (xy-anchor.float())*usable[..., None]
        offsets.scatter_add_(2, index[:, None].expand(-1, 2, -1), values.transpose(1, 2))
        offset_valid = torch.zeros(b, h*w, device=xy.device)
        offset_valid.scatter_add_(1, index, usable.float())
        outputs.update(grid_points_offsets=offsets.reshape(b, 2, h, w),
                       grid_points_offset_valid=offset_valid.reshape(b, 1, h, w),
                       grid_points_anchor_xy=torch.where(usable[..., None], anchor, -1),
                       grid_points_anchor_valid=usable)
    return outputs


def _grid_lattice(canonical_size, ppm, phase_x, phase_y, top, grid_present, config):
    ch, cw = canonical_size
    b = len(ppm)
    phase = torch.stack((phase_x, phase_y), -1).remainder(5)
    extent = torch.tensor([cw-1, ch-1], device=ppm.device)/ppm[:, None]
    capacity = torch.tensor([config.columns, config.rows], device=ppm.device)
    torch._assert_async(((extent-phase)/5 < capacity).all(), 'Grid point capacity exceeded. Increase rows/columns')
    row, col = torch.meshgrid(torch.arange(config.rows, device=ppm.device),
                              torch.arange(config.columns, device=ppm.device), indexing='ij')
    ids = torch.stack((row.flatten(), col.flatten()), -1)[None].expand(b, -1, -1)
    material_mm = ids.flip(-1).float()*5 + phase[:, None]
    canonical = material_mm*ppm[:, None, None]
    active = ((canonical[..., 0] <= cw-1) & (canonical[..., 1] <= ch-1)
              & (canonical[..., 1] >= top[:, None]*.8) & (grid_present[:, None] > 0))
    return {'grid_points_active':active, 'grid_points_ids':ids,
            'grid_points_canonical_xy':canonical, 'grid_points_mm':material_mm}


def _bilinear_with_jacobian(source, xy):
    """Evaluate the exact local bilinear interpolant and its two derivatives."""
    b, h, w, _ = source.shape
    safe = torch.nan_to_num(xy, nan=0., posinf=0., neginf=0.)
    ix = safe[..., 0].floor().long().clamp(0, w-2)
    iy = safe[..., 1].floor().long().clamp(0, h-2)
    f = safe-torch.stack((ix, iy), -1)
    flat = source.reshape(b, h*w, 2)

    def read(dx, dy):
        index = (iy+dy)*w+ix+dx
        return flat.gather(1, index[..., None].expand(-1, -1, 2))

    a, c, d, e = read(0, 0), read(1, 0), read(0, 1), read(1, 1)
    fx, fy = f[..., :1], f[..., 1:]
    value = a*(1-fx)*(1-fy)+c*fx*(1-fy)+d*(1-fx)*fy+e*fx*fy
    return value, (c-a)*(1-fy)+(e-d)*fy, (d-a)*(1-fx)+(e-c)*fx


def _inverse_step(residual, gx, gy):
    det = gx[..., 0]*gy[..., 1]-gx[..., 1]*gy[..., 0]
    safe_det = torch.where(det.abs() > 1e-8, det, torch.ones_like(det))
    step = torch.stack((residual[..., 0]*gy[..., 1]-residual[..., 1]*gy[..., 0],
                        gx[..., 0]*residual[..., 1]-gx[..., 1]*residual[..., 0]), -1)/safe_det[..., None]
    return step, det


@torch.no_grad()
def grid_point_targets(field, valid, *, canonical_size, ppm, phase_x, phase_y,
                       top, grid_present, config=None):
    """Export visible major-grid nodes, IDs, canonical positions and heatmap.

    ``field[B,H,W,2]`` maps observed pixel centers to normalized canonical
    material coordinates, align_corners=False. ``valid[B,1,H,W]`` is the same
    geometric visibility used by the image/trace targets. Canonical dimensions
    can differ from the observed dimensions, as in the offline backend.

    Seed inversion uses one nearest camera pixel per material lattice ID,
    followed by exact-bilinear Newton refinement. Points without a visible seed,
    at discontinuities, or failing the round trip are explicitly invalid.
    This does not claim exact mesh projection or recover hidden nodes.
    """
    config = config or GridPointConfig()
    if not isinstance(config, GridPointConfig):
        raise TypeError('config must be GridPointConfig')
    if field.ndim != 4 or field.shape[-1] != 2 or min(field.shape[1:3]) < 2:
        raise ValueError('field must be [B,H,W,2], H/W >=2')
    b, h, w, _ = field.shape
    ch, cw = canonical_size
    if type(ch) is not int or type(cw) is not int or min(ch, cw) < 2:
        raise ValueError('canonical_size must contain integer height/width >=2')
    if valid.shape != (b, 1, h, w) or valid.device != field.device:
        raise ValueError('valid must be [B,1,H,W] on the field device')
    if field.dtype != torch.float32 or valid.dtype != torch.float32:
        raise ValueError('field and valid must be float32, matching renderer geometry')
    for value in (ppm, phase_x, phase_y, top, grid_present):
        if value.shape != (b,) or value.device != field.device:
            raise ValueError('Grid metadata must be [B] on the field device')
    torch._assert_async((torch.isfinite(ppm) & (ppm > 0)).all(), 'Positive finite ppm required')
    torch._assert_async((torch.isfinite(phase_x) & torch.isfinite(phase_y)
                         & torch.isfinite(top) & torch.isfinite(grid_present)).all(), 'Finite grid metadata required')
    lattice = _grid_lattice(canonical_size, ppm, phase_x, phase_y, top, grid_present, config)
    canonical, active = lattice['grid_points_canonical_xy'], lattice['grid_points_active']
    phase = torch.stack((phase_x, phase_y), -1).remainder(5)
    n = config.rows*config.columns
    scale = torch.tensor([cw/2, ch/2], device=field.device)
    source = (field+1)*scale-.5
    nearest = ((source/ppm[:, None, None, None]-phase[:, None, None])/5).round().long()
    ix, iy = nearest.unbind(-1)
    candidate = iy*config.columns+ix
    usable = ((ix >= 0) & (ix < config.columns) & (iy >= 0) & (iy < config.rows)
              & (valid[:, 0] >= config.min_visibility) & torch.isfinite(source).all(-1))
    candidate = candidate.clamp(0, n-1)
    desired = (nearest.float()*5+phase[:, None, None])*ppm[:, None, None, None]
    score = torch.where(usable, (source-desired).square().sum(-1), torch.inf).flatten(1)
    best = torch.full((b, n), torch.inf, device=field.device)
    best.scatter_reduce_(1, candidate.flatten(1), score, reduce='amin')
    index = torch.arange(h*w, device=field.device)[None].expand(b, -1)
    chosen = torch.where(usable.flatten(1) & (score == best.gather(1, candidate.flatten(1))), index, h*w)
    seeds = torch.full((b, n), h*w, device=field.device, dtype=torch.long)
    seeds.scatter_reduce_(1, candidate.flatten(1), chosen, reduce='amin')
    found = seeds < h*w
    seeds = seeds.clamp_max(h*w-1)

    def gather(value):
        return value.reshape(b, h*w, -1).gather(1, seeds[..., None].expand(-1, -1, value.shape[-1]))

    dy, dx = torch.gradient(source, dim=(1, 2))
    gx, gy = gather(dx), gather(dy)
    residual = canonical-gather(source)
    delta, det = _inverse_step(residual, gx, gy)
    seed_xy = torch.stack((seeds.remainder(w), seeds.div(w, rounding_mode='floor')), -1).float()
    xy = seed_xy+delta
    for _ in range(config.refinement_steps):
        value, gx, gy = _bilinear_with_jacobian(source, xy)
        step, _ = _inverse_step(canonical-value, gx, gy)
        xy = xy+step.clamp(-.5, .5)
    round_trip, gx, gy = _bilinear_with_jacobian(source, xy)
    image_step, det = _inverse_step(canonical-round_trip, gx, gy)
    image_error = image_step.square().sum(-1).sqrt()
    sample_grid = ((xy+.5)/torch.tensor([w/2, h/2], device=field.device)-1)[:, None]
    support = F.grid_sample(valid, sample_grid, align_corners=False)[:, 0, 0]
    error = (round_trip-canonical).square().sum(-1).sqrt()
    visible = (active & found & (det.abs() > 1e-8) & ((xy-seed_xy).abs().amax(-1) <= 1.5)
               & (xy[..., 0] >= 0) & (xy[..., 0] <= w-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h-1)
               & (support >= config.min_visibility) & (error <= config.max_error_pixels)
               & (image_error <= config.max_image_error_pixels))
    xy = torch.where(visible[..., None], xy, -1.)
    return {**lattice, **rasterize_grid_points(xy, visible, valid, config=config),
            'grid_points_xy': xy, 'grid_points_valid': visible,
            'grid_points_error_px': torch.where(visible, error, -1.),
            'grid_points_error_image_px': torch.where(visible, image_error, -1.)}
