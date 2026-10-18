"""Material-coordinate paper aging and liquid deposition in linear light.

The output is an appearance prior, not a cellulose chemistry or fluid solver.
All random fields remain attached to the sheet before its geometric warp.
Nothing in this module changes signal samples, trace masks or visibility.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PaperMaterial(nn.Module):
    """Return multiplicative linear-light material factors for a batch.

    ``substrate_factor`` multiplies unprinted paper reflectance.
    ``deposit_transmission`` multiplies the completed paper/ink composite,
    representing a thin colored deposit. ``print_retention`` attenuates
    thermal print coverage before ink compositing, never target coverage.

    Existing controls are paper_age, stain_x/y/strength, coffee_strength,
    coffee_radius, thermal_fade and fade_position/width. Optional controls
    are foxing_strength, fiber_strength, water_strength and
    thermal_darkening_strength, all default zero. Thermal burn models developed
    background coating separately from loss of existing print contrast.
    Missing center/radius controls default to .5/.15. Strengths are [0,1].
    Zero strengths produce exact ones, including with nonzero random fields.
    Replay uses a caller-owned device-compatible torch.Generator.
    """

    def __init__(self, height, width):
        super().__init__()
        if min(height, width) < 16:
            raise ValueError('Paper material requires dimensions at least 16')
        self.height, self.width = int(height), int(width)
        self.aspect = height / width
        y, x = torch.meshgrid((torch.arange(height) + .5) / height,
                              (torch.arange(width) + .5) / width, indexing='ij')
        self.register_buffer('uv', torch.stack((x, y), 0)[None])
        self.register_buffer('oxidation_absorption', torch.tensor([.16, .48, 1.25])[None, :, None, None])
        self.register_buffer('deposit_absorption', torch.tensor([.34, .70, 1.10])[None, :, None, None])
        self.register_buffer('water_absorption', torch.tensor([.15, .18, .22])[None, :, None, None])
        offsets = torch.arange(-3, 4, dtype=torch.float32)
        gaussian = torch.exp(-.5 * (offsets / 1.05).square())
        kernel = gaussian[:, None] * gaussian[None, :]
        self.register_buffer('spot_kernel', (kernel / kernel.max())[None, None])

    def forward(self, parameters, generator=None):
        """Parameters are batch-shaped tensors on the module's device."""
        if not parameters:
            raise ValueError('At least one batch-shaped parameter is required')
        first = next(iter(parameters.values()))
        if not isinstance(first, torch.Tensor) or first.ndim != 1:
            raise ValueError('Material parameters must include a batch vector')
        b = first.shape[0]
        device, dtype = self.uv.device, self.uv.dtype

        def column(name, default=0.):
            value = parameters.get(name)
            if value is None:
                return torch.full((b, 1, 1, 1), default, device=device, dtype=dtype)
            if value.shape != (b,):
                raise ValueError(f'{name} must have shape [B]')
            return value[:, None, None, None]

        def field(rows, columns):
            value = torch.randn(b, 1, rows, columns, device=device,
                                dtype=dtype, generator=generator)
            # Bounded without data-dependent extrema or resolution-dependent
            # sample normalization. Bicubic detail has no tiled repeats.
            return F.interpolate(value, size=(self.height, self.width),
                                 mode='bicubic', align_corners=False).tanh()

        broad, medium, capillary = field(5, 7), field(17, 23), field(43, 59)
        phase = torch.rand(b, 6, 1, 1, device=device, dtype=dtype,
                           generator=generator) * (2 * math.pi)
        x, y = self.uv[:, :1], self.uv[:, 1:]
        age = column('paper_age').clamp(0, 1)
        # Unequal edge penetration and broad humidity history avoid a uniform
        # yellow wash or a symmetric rectangular vignette.
        edge_depth = .018 + .024 * (medium + 1) * .5
        left = torch.exp(-x / edge_depth)
        right = torch.exp(-(1 - x) / (edge_depth * 1.30))
        top = torch.exp(-y * self.aspect / (edge_depth * .75))
        bottom = torch.exp(-(1 - y) * self.aspect / (edge_depth * 1.10))
        edge = (left * (.7 + .3 * phase[:, :1].cos()) + right * .75
                + top * .50 + bottom * .85).clamp_max(1.8)
        oxidation = age * (.38 + .16 * broad + .09 * medium + .72 * edge)

        # Sparse seeds are spatially modulated by broad moisture history.
        # Their multiscale envelopes produce clustered, unequal foxing marks
        # rather than independent full-frame sensor noise or identical dots.
        seeds = torch.rand(b, 1, 61, 83, device=device, dtype=dtype, generator=generator)
        humidity = F.interpolate((broad + 1) * .5, size=(61, 83), mode='bilinear', align_corners=False)
        seeds = (seeds > .993 - .007 * humidity).to(dtype) * (seeds - .985).clamp_min(0) * 65
        spots = F.conv2d(seeds, self.spot_kernel, padding=3) * .65 + seeds * .35
        spots = F.interpolate(spots, size=(self.height, self.width), mode='bilinear', align_corners=False)
        spots = spots * (.70 + .30 * capillary)
        foxing = column('foxing_strength').clamp(0, 1) * spots

        # Fine anisotropic fiber-density texture is reflective material detail,
        # deliberately much weaker than print and independent of sensor noise.
        fibers = field(min(self.height, 129), min(self.width, 193))
        fibers = F.avg_pool2d(fibers, (1, 9), stride=1, padding=(0, 4))
        fiber_density = .005 * column('fiber_strength').clamp(0, 1) * (fibers + 1)
        substrate = torch.exp(-(oxidation + foxing * .80) * self.oxidation_absorption - fiber_density)

        dx = x - column('stain_x', .5)
        dy = (y - column('stain_y', .5)) * self.aspect
        theta = torch.atan2(dy, dx)
        # A changing contact line with anisotropic wicking and pinned sectors.
        # Harmonics are deliberately low order and perturbed by a nonperiodic
        # field, so the visible boundary is not an exact ellipse or circle.
        eccentricity = .83 + .19 * phase[:, :1].cos()
        radius = torch.sqrt(dx.square() + (dy / eccentricity).square() + 1e-12)
        contour = (1 + .10 * torch.sin(3 * theta + phase[:, :1])
                   + .045 * torch.sin(7 * theta + phase[:, 1:2])
                   + .035 * medium + .012 * capillary).clamp_min(.65)
        wet_radius = column('coffee_radius', .15).clamp(.01, .50)
        normalized = radius / (wet_radius * contour)
        # Pixel-aware transition width antialiases the deposition footprint.
        transition = (.018 + .012 * (capillary + 1)).clamp_min(1 / (self.width * wet_radius))
        interior = torch.sigmoid((1 - normalized) / transition)
        rim = torch.exp(-((normalized - 1) / (transition * 1.35)).square())
        pinning = (.65 + .25 * torch.sin(2 * theta + phase[:, 2:3])
                   + .20 * medium).clamp(.15, 1.1)
        tide = torch.exp(-((normalized - .86 - .025 * medium) / (transition * .75)).square())
        coffee = column('coffee_strength').clamp(0, 1) * (
            interior * (.055 + .025 * capillary) + rim * pinning * .95 + tide * .14)

        # Absorbed diffuse stains use the same capillary history but a broader,
        # asymmetric footprint and a visible, low-contrast drying front.
        stain_distance = torch.sqrt((dx / .22).square() + (dy / (.15 * self.aspect)).square() + 1e-12)
        stain_boundary = 1 + .13 * medium + .08 * torch.sin(3 * theta + phase[:, 3:4])
        stain_fill = torch.sigmoid((stain_boundary - stain_distance) / .13)
        stain_tide = torch.exp(-((stain_distance - stain_boundary) / .055).square())
        stain = column('stain_strength').clamp(0, 1) * (
            stain_fill * (.18 + .045 * broad + .025 * capillary) + .15 * stain_tide)
        water = column('water_strength').clamp(0, 1) * (
            stain_fill * .035 + stain_tide * (.15 + .045 * medium))
        deposit = torch.exp(-(coffee * 1.5 + stain * 1.7) * self.deposit_absorption
                            - water * self.water_absorption)

        # A damaged thermal coating has a smooth exposure footprint with
        # local mottling and an oblique drift, rather than a perfect band.
        fade_center = column('fade_position', .5) + .025 * medium + .035 * (y - .5) * phase[:, 4:5].sin()
        fade_width = column('fade_width', .22).clamp_min(1 / self.width)
        footprint = torch.exp(-((x - fade_center) / fade_width).square())
        thermal = column('thermal_fade').clamp(0, 1)
        retention = (1 - thermal * footprint * (.82 + .12 * broad + .06 * medium)).clamp(0, 1)
        development = column('thermal_darkening_strength').clamp(0, 1) * footprint * (
            3.6 + .65 * broad + .25 * medium)
        substrate = substrate * torch.exp(-development)
        return {'substrate_factor': substrate,
                'deposit_transmission': deposit,
                'print_retention': retention}
