"""Material-coordinate paper geometry and deterministic scene kernels.

Units are half the sheet width. Cylindrical bends integrate unit tangents,
so bending shortens the projection without stretching the printed material.
Localized crumples use rounded, finite roof creases with a small-slope
in-plane shortening approximation. This is a bounded procedural model,
not a general thin-shell dynamics or self-intersection solver.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def derivatives(value, aspect):
    """Central interior and one-sided edge derivatives on pixel-center grids."""
    h, w = value.shape[-2:]
    dx = torch.cat((value[..., 1:2] - value[..., :1],
                    (value[..., 2:] - value[..., :-2]) * .5,
                    value[..., -1:] - value[..., -2:-1]), -1) * (w / 2)
    dy = torch.cat((value[..., 1:2, :] - value[..., :1, :],
                    (value[..., 2:, :] - value[..., :-2, :]) * .5,
                    value[..., -1:, :] - value[..., -2:-1, :]), -2) * (h / (2 * aspect))
    return dx, dy


def centered_integral(value, step, dim):
    """Trapezoidal integration anchored at the center of a material row."""
    left, right = value.narrow(dim, 0, value.shape[dim] - 1), value.narrow(dim, 1, value.shape[dim] - 1)
    increments = (left + right) * (.5 * step)
    zero = torch.zeros_like(value.narrow(dim, 0, 1))
    result = torch.cat((zero, increments.cumsum(dim)), dim)
    n = result.shape[dim]
    center = (result.narrow(dim, (n - 1) // 2, 1) + result.narrow(dim, n // 2, 1)) * .5
    return result - center


def newton_step(uv, wanted, values):
    """One bounded 2x2 Newton update, independent of RNG and sampling."""
    fx = uv[..., 0] + values[:, 0] - wanted[..., 0]
    fy = uv[..., 1] + values[:, 1] - wanted[..., 1]
    a, b, c, d = (values[:, index] for index in range(2, 6))
    determinant = a * d - b * c
    safe = determinant.abs().clamp_min(.12) * torch.where(determinant < 0, -1., 1.)
    delta = torch.stack(((d * fx - b * fy) / safe, (a * fy - c * fx) / safe), -1)
    return uv - delta.clamp(-.20, .20)


class PaperSurface(nn.Module):
    """Generate a small material mesh shared by projection and illumination."""
    def __init__(self, rows, columns, aspect, compile_scene=False):
        super().__init__()
        self.aspect = aspect
        y, x = torch.meshgrid((torch.arange(rows) + .5) * (2 * aspect / rows) - aspect,
                              (torch.arange(columns) + .5) * (2 / columns) - 1, indexing='ij')
        self.register_buffer('material', torch.stack((x, y), 0)[None])
        extent = math.sqrt(1 + aspect * aspect)
        line = (torch.arange(columns * 2) + .5) * (2 * extent / (columns * 2)) - extent
        self.register_buffer('line', line[None, None, None])
        self.extent = extent
        self.register_buffer('crease_ids', torch.arange(8)[None, :, None, None])
        self._geometry_kernel = torch.compile(self._geometry, fullgraph=True, dynamic=True) if compile_scene else self._geometry

    def forward(self, parameters, generator=None):
        # Random draws remain outside compiled regions to retain the public
        # generator stream and checkpoint replay across eager/compiled modes.
        noise = torch.randn(len(parameters['surface_mode']), 1, 9, 13,
                            device=self.material.device, generator=generator)
        return self._geometry_kernel(parameters, noise)

    def _geometry(self, parameters, noise):
        p = parameters
        b = len(p['surface_mode'])
        x, y = self.material[:, :1].expand(b, -1, -1, -1), self.material[:, 1:].expand(b, -1, -1, -1)
        mode = p['surface_mode'][:, None, None, None]
        ca, sa = p['surface_axis'].cos()[:, None, None, None], p['surface_axis'].sin()[:, None, None, None]
        # A lifted corner is another developable bend along a diagonal hinge.
        ca = torch.where(mode == 7, p['corner_position'][:, :1, None, None] / math.sqrt(2), ca)
        sa = torch.where(mode == 7, p['corner_position'][:, 1:, None, None] / math.sqrt(2), sa)
        u, v = x * ca + y * sa, -x * sa + y * ca
        t = self.line
        phase = p['surface_phase'][:, None, None, None]
        frequency = p['surface_frequency'][:, None, None, None]
        position = p['fold_position'][:, :1, None, None]
        width = p['fold_width'][:, :1, None, None].clamp_min(.015)
        angle = (.62 * t) * (mode == 1)
        angle = angle + .58 * torch.tanh((t - position) / width) * (mode == 2)
        angle = angle + .34 * torch.tanh(torch.sin(t * frequency * math.pi + phase) / .25) * (mode == 3)
        angle = angle + (.07 * torch.sin(t * 4.7 + phase) + .035 * t) * (mode == 4)
        angle = angle + (.08 * torch.sin(t * 7.3 + phase) + .035 * torch.sin(t * 3.1 - phase)) * (mode == 5)
        angle = angle + 1.12 * torch.sigmoid((t - .20) / .16) * (mode == 6)
        corner_hinge = (1 + self.aspect - p['corner_size'][:, None, None, None]) / math.sqrt(2)
        angle = angle + .90 * torch.sigmoid((t - corner_hinge) / width) * (mode == 7)
        angle = angle * p['surface_strength'][:, None, None, None]
        step = 2 * self.extent / self.line.shape[-1]
        bend_x = centered_integral(angle.cos(), step, -1)
        bend_z = centered_integral(angle.sin(), step, -1)
        curve = torch.cat((bend_x, bend_z), 1)
        field = torch.stack((u[:, 0] / self.extent, torch.zeros_like(u[:, 0])), -1)
        bent = F.grid_sample(curve, field, padding_mode='border', align_corners=False)
        base_x, base_y = bent[:, :1] * ca - v * sa, bent[:, :1] * sa + v * ca
        base_x, base_y = torch.where(mode == 0, x, base_x), torch.where(mode == 0, y, base_y)

        # Finite, signed roof creases form angular facets instead of Gaussian
        # bubbles. Width controls crease rounding, span its surrounding facet.
        azimuth = p['fold_angle'][:, :, None, None]
        c, s = azimuth.cos(), azimuth.sin()
        distance = x * c + y * s - p['fold_position'][:, :, None, None]
        along = -x * s + y * c
        span = p['fold_span'][:, :, None, None]
        rounding = p['fold_width'][:, :, None, None]
        roof = (torch.sqrt(span.square() + rounding.square()) - torch.sqrt(distance.square() + rounding.square())).clamp_min(0) / span
        envelope = (1 - (along / p['fold_length'][:, :, None, None]).abs()).clamp(0, 1)
        envelope = envelope.square() * (3 - 2 * envelope)
        active = self.crease_ids < p['crease_count'][:, None, None, None]
        local = (roof * envelope * active * p['fold_amplitude'][:, :, None, None] * p['fold_sign'][:, :, None, None]).sum(1, keepdim=True)
        local = local * ((mode == 4) + (mode == 5) * .25)
        # Small nonperiodic wrinkles add detail, not the large shape.
        noise = F.interpolate(noise, size=x.shape[-2:], mode='bicubic', align_corners=False)
        local = (local + noise * .0007 * ((mode == 4) + (mode == 5))) * p['surface_strength'][:, None, None, None]
        dzdu, dzdv = derivatives(local, self.aspect)
        shorten_x = centered_integral(torch.rsqrt(1 + dzdu.square()) - 1, 2 / x.shape[-1], -1)
        shorten_y = centered_integral(torch.rsqrt(1 + dzdv.square()) - 1, 2 * self.aspect / x.shape[-2], -2)
        z = bent[:, 1:] + local
        z = z - z.amin((-1, -2), keepdim=True)
        mesh = torch.cat((base_x + shorten_x, base_y + shorten_y, z), 1)
        du, dv = derivatives(mesh, self.aspect)
        normals = F.normalize(torch.linalg.cross(du, dv, dim=1), dim=1)
        metric = torch.cat((du.square().sum(1, keepdim=True).sqrt() - 1,
                            dv.square().sum(1, keepdim=True).sqrt() - 1,
                            (du * dv).sum(1, keepdim=True)), 1)
        return mesh, normals, metric
