"""Material-coordinate paper geometry and deterministic scene kernels.

Units are half the sheet width. Cylindrical bends integrate unit tangents,
so bending shortens the projection without stretching the printed material.
Localized crumples use branched, asymmetric finite creases with a small-slope
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
        # Nonuniform fold spacing is a monotonic material-coordinate remap.
        # Integrating unit tangents afterwards still preserves arc length.
        fold_phase = t * frequency * math.pi + phase
        fold_phase = fold_phase + .38 * torch.sin(.61 * fold_phase + phase)
        # A real opened fold has unequal panels and a rounded hinge. The
        # slow angular drift changes curvature without stretching the paper.
        hinge = (t - position) / width
        angle = angle + (.54 * torch.tanh(hinge) + .055 * torch.sin(t * 2.3 + phase)) * (mode == 2)
        angle = angle + .34 * torch.tanh(torch.sin(fold_phase) / (.23 + .055 * torch.cos(phase))) * (mode == 3)
        angle = angle + (.028 * torch.sin(t * 2.3 + phase) + .020 * t) * (mode == 4)
        angle = angle + (.08 * torch.sin(t * 7.3 + phase) + .035 * torch.sin(t * 3.1 - phase)) * (mode == 5)
        angle = angle + 1.12 * torch.sigmoid((t - .20) / .16) * (mode == 6)
        corner_hinge = (1 + self.aspect - p['corner_size'][:, None, None, None]) / math.sqrt(2)
        corner_curve = (.78 * torch.sigmoid((t - corner_hinge) / width)
                        + .12 * torch.sigmoid((t - corner_hinge + .07) / (.08 + width)))
        angle = angle + corner_curve * (mode == 7)
        angle = angle * p['surface_strength'][:, None, None, None]
        step = 2 * self.extent / self.line.shape[-1]
        bend_x = centered_integral(angle.cos(), step, -1)
        bend_z = centered_integral(angle.sin(), step, -1)
        curve = torch.cat((bend_x, bend_z), 1)
        field = torch.stack((u[:, 0] / self.extent, torch.zeros_like(u[:, 0])), -1)
        bent = F.grid_sample(curve, field, padding_mode='border', align_corners=False)
        base_x, base_y = bent[:, :1] * ca - v * sa, bent[:, :1] * sa + v * ca
        base_x, base_y = torch.where(mode == 0, x, base_x), torch.where(mode == 0, y, base_y)

        # Each pair shares a junction. The second ridge branches from an
        # offset point on the first instead of independently crossing the
        # sheet center. Existing latent parameters encode all variability.
        pair = torch.arange(8, device=x.device).div(2, rounding_mode='floor') * 2
        parent_angle = p['fold_angle'][:, pair]
        raw_angle = p['fold_angle']
        branch = (self.crease_ids % 2).to(x.dtype)
        branch_turn = (.38 + .58 * raw_angle.cos().abs()) * p['fold_sign']
        azimuth = (parent_angle + branch[:, :, 0, 0] * branch_turn)[:, :, None, None]
        c, s = azimuth.cos(), azimuth.sin()
        parent_c, parent_s = parent_angle.cos(), parent_angle.sin()
        transverse = p['fold_position'][:, pair] * .74
        longitudinal = p['fold_position'].roll(1, dims=1)[:, pair] * .54
        junction_shift = branch[:, :, 0, 0] * p['fold_length'][:, pair] * .23 * raw_angle.sin()
        along_center = longitudinal + junction_shift
        center_x = (transverse * parent_c - along_center * parent_s)[:, :, None, None]
        center_y = (transverse * parent_s + along_center * parent_c)[:, :, None, None]
        distance = (x - center_x) * c + (y - center_y) * s
        along = -(x - center_x) * s + (y - center_y) * c
        # Branches inherit scale and sign but are narrower and shorter. The
        # rounded roof returns smoothly to the neighboring facet, avoiding a
        # second sharp crease at the support boundary.
        scale = 1 - branch * .36
        span = p['fold_span'][:, pair, None, None] * scale
        rounding = p['fold_width'][:, pair, None, None] * (1 - branch * .28)
        asymmetry = .20 * raw_angle.sin()[:, :, None, None]
        side_span = span * (1 + asymmetry * torch.tanh(distance / rounding))
        roof = (torch.sqrt(side_span.square() + rounding.square()) - torch.sqrt(distance.square() + rounding.square())).clamp_min(0) / side_span
        transition = (roof / .18).clamp(0, 1)
        roof = roof * transition.square() * (3 - 2 * transition)
        length = p['fold_length'][:, pair, None, None] * (1 - branch * .40)
        along_scaled = along / (length * (1 + .27 * raw_angle.cos()[:, :, None, None] * torch.tanh(along / length)))
        envelope = (1 - along_scaled.square()).clamp(0, 1).square()
        active = self.crease_ids < p['crease_count'][:, None, None, None]
        amplitude = p['fold_amplitude'][:, pair, None, None] * (1 - branch * .46)
        local = (roof * envelope * active * amplitude * p['fold_sign'][:, pair, None, None]).sum(1, keepdim=True)
        local = local * ((mode == 4) + (mode == 5) * .25)
        # Directional microbuckles follow the ridge network, with a smooth
        # finite support. Noise changes their phase, not isotropic height.
        ripple_phase = noise[:, :, :1, :1]
        micro = (torch.sin(distance / rounding * 2.1 + ripple_phase) * envelope
                 * torch.exp(-(distance / (span * .8)).square()) * active).sum(1, keepdim=True)
        micro = micro * ((mode == 4) + (mode == 5))
        local = (local * .66 + micro * .00045) * p['surface_strength'][:, None, None, None]
        if 'history_warp_strength' in p:
            # Liquid, deposits and cockling share the same material center and
            # coffee_radius. Geometry and normals are updated before projection.
            cx = p['stain_x'][:, None, None, None] * 2 - 1
            cy = (p['stain_y'][:, None, None, None] * 2 - 1) * self.aspect
            radius = p['coffee_radius'][:, None, None, None] * 2
            distance_liquid = ((x - cx).square() + (y - cy).square()) / radius.square()
            local = local + p['history_warp_strength'][:, None, None, None] * torch.exp(-distance_liquid) * (
                .65 + .35 * torch.cos((x - cx) / radius * 9))
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

