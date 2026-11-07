"""Phone camera processing artifacts applied to the final observation.

Phone photographs of ECG printouts show effects that the base camera model
lacks: lateral chromatic aberration (color fringes that grow toward the
corners), aggressive sharpening with halos around dark traces, and specular
glare from glossy thermal paper or a flash. This stage adds them to records
captured photographically. It changes the observation only; geometric targets
are untouched.

It runs after the base camera (including its noise and JPEG simulation), so it
approximates the order of a real image signal processor rather than
reproducing it.
"""
import torch
from torch.nn import functional as F

from .scan_modes import _separable_blur


def chromatic_aberration(image, shift_px):
    """Scale red outward and blue inward by ``shift_px`` at the image corners."""
    b, _, h, w = image.shape
    ys = torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype)
    xs = torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    half_diagonal = .5 * (h * h + w * w) ** .5
    k = (shift_px / half_diagonal)[:, None, None]
    channels = []
    for channel, sign in ((0, 1.), (1, 0.), (2, -1.)):
        scale = 1 / (1 + sign * k)
        grid = torch.stack([gx[None] * scale, gy[None] * scale], -1)
        channels.append(F.grid_sample(image[:, channel:channel + 1], grid, mode='bilinear',
                                      padding_mode='border', align_corners=True))
    shifted = torch.cat(channels, 1)
    return torch.where((shift_px == 0)[:, None, None, None], image, shifted)


def sharpen(image, amount, sigma, kernel_size=9):
    """Unsharp mask; ``amount`` 0 is the identity."""
    blurred = _separable_blur(image, sigma, kernel_size)
    return (image + amount[:, None, None, None] * (image - blurred)).clamp(0, 1)


def glare(image, strength, center_x, center_y, radius):
    """Additive near-white Gaussian highlight; center and radius as image fractions."""
    b, _, h, w = image.shape
    ys = (torch.arange(h, device=image.device, dtype=image.dtype)[None, :, None] + .5) / w
    xs = (torch.arange(w, device=image.device, dtype=image.dtype)[None, None, :] + .5) / w
    cy = (center_y * h / w)[:, None, None]
    cx = center_x[:, None, None]
    falloff = torch.exp(-((xs - cx).square() + (ys - cy).square()) / (2 * radius[:, None, None].square()))
    highlight = strength[:, None, None, None] * falloff[:, None]
    return image + (1 - image) * highlight.clamp(0, 1)


def phone_observation(image, p):
    """Apply aberration, sharpening and glare with per-record controls in ``p``."""
    image = chromatic_aberration(image, p['ca_px'])
    image = sharpen(image, p['sharpen_amount'], p['sharpen_sigma'])
    return glare(image, p['glare_strength'], p['glare_x'], p['glare_y'], p['glare_radius']).clamp(0, 1)
