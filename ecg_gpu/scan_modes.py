"""Flatbed scanner observation modes: color, grayscale and bilevel at a given dpi.

Real ECG archives are often scanned in grayscale or black-and-white and at
low resolution (ECG-Image-Database scans at 600 dpi in all three modes; older
archives and fax copies are far coarser). This stage changes only the
observation. Geometric targets are untouched, as for optical blur.
"""
import torch
from torch.nn import functional as F

COLOR, GRAYSCALE, BILEVEL = 0, 1, 2
LUMA = (.2126, .7152, .0722)


def _gaussian_rows(sigma, size):
    """[B,size] normalized 1-D Gaussian weights; sigma 0 gives a delta."""
    offsets = torch.arange(size, device=sigma.device, dtype=sigma.dtype) - size // 2
    safe = sigma.clamp_min(1e-3)[:, None]
    weights = torch.exp(-.5 * (offsets[None] / safe).square())
    delta = (offsets == 0).to(sigma.dtype)[None].expand_as(weights)
    weights = torch.where(sigma[:, None] > 1e-3, weights, delta)
    return weights / weights.sum(1, keepdim=True)


def _separable_blur(image, sigma, size):
    b, c, h, w = image.shape
    weights = _gaussian_rows(sigma, size).repeat_interleave(c, 0)
    flat = image.reshape(1, b * c, h, w)
    pad = size // 2
    flat = F.conv2d(F.pad(flat, (pad, pad, 0, 0), mode='replicate'), weights[:, None, None, :], groups=b * c)
    flat = F.conv2d(F.pad(flat, (0, 0, pad, pad), mode='replicate'), weights[:, None, :, None], groups=b * c)
    return flat.reshape(b, c, h, w)


def scan_resolution(image, scale, kernel_size):
    """Simulate capture at ``scale`` scanner pixels per output pixel (<= 1).

    The image is integrated over each scanner pixel (Gaussian approximation of
    a box of width 1/scale) and held constant across it, which reproduces the
    blockiness of an upscaled low-dpi scan. Scale 1 returns the input.
    """
    b, _, h, w = image.shape
    scale = scale.clamp(1e-3, 1)
    sigma = ((scale.reciprocal().square() - 1).clamp_min(0) / 12).sqrt()
    blurred = _separable_blur(image, sigma, kernel_size)
    ys = torch.arange(h, device=image.device, dtype=image.dtype)[None, :]
    xs = torch.arange(w, device=image.device, dtype=image.dtype)[None, :]
    # Center of the scanner pixel that covers each output pixel center.
    cy = ((ys + .5) * scale[:, None]).floor().add(.5) / scale[:, None] - .5
    cx = ((xs + .5) * scale[:, None]).floor().add(.5) / scale[:, None] - .5
    grid = torch.stack(torch.broadcast_tensors(
        ((cx + .5) / w * 2 - 1)[:, None, :], ((cy + .5) / h * 2 - 1)[:, :, None]), -1)
    held = F.grid_sample(blurred, grid, mode='nearest', padding_mode='border', align_corners=False)
    return torch.where((scale >= 1)[:, None, None, None], image, held)


def lid_shadow(image, strength, edge, width):
    """Darken toward one page edge, as when the scanner lid does not press the page flat.

    ``edge`` 0/1/2/3 is left/top/right/bottom, ``width`` the falloff as a
    fraction of the image width, ``strength`` the darkening at the edge.
    """
    b, _, h, w = image.shape
    ys = (torch.arange(h, device=image.device, dtype=image.dtype)[None, :, None] + .5) / w
    xs = (torch.arange(w, device=image.device, dtype=image.dtype)[None, None, :] + .5) / w
    edge = edge.long()[:, None, None]
    distance = torch.where(edge == 0, xs, torch.where(edge == 1, ys, torch.where(edge == 2, 1 - xs, h / w - ys)))
    falloff = torch.exp(-distance / width.clamp_min(1e-3)[:, None, None])
    return image * (1 - strength[:, None, None, None] * falloff[:, None])


def scan_observation(image, ppm, mode, dpi, threshold, kernel_size=15, lid=None):
    """Apply per-record scanner mode and resolution to sRGB ``image`` in [0,1].

    ``ppm`` is output pixels per paper millimeter, ``mode`` 0/1/2 is
    color/grayscale/bilevel, ``dpi`` the scanner resolution and ``threshold``
    the bilevel luminance cut. All are ``[B]`` tensors on the image device.
    """
    if lid is not None:
        image = lid_shadow(image, *lid)
    scale = (dpi / 25.4 / ppm).clamp(max=1)
    resampled = scan_resolution(image, scale, kernel_size)
    gray = (LUMA[0] * resampled[:, :1] + LUMA[1] * resampled[:, 1:2]
            + LUMA[2] * resampled[:, 2:3]).expand_as(resampled)
    bilevel = (gray > threshold[:, None, None, None]).to(image.dtype)
    mode = mode.long()[:, None, None, None]
    return torch.where(mode == COLOR, resampled, torch.where(mode == GRAYSCALE, gray, bilevel))
