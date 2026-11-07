"""Print technologies and document reproduction applied to the canonical page.

Archived ECGs are often not the recorder's original print: they are laser
reprints with halftoned grids, photocopies (sometimes copies of copies) or
fax printouts. These change the document before it is photographed or
scanned. Waveform geometry and targets are untouched; only appearance changes.

``print_tech`` per record: 0 original print, 1 laser reprint (ordered-dither
halftone of the colored grid), 2 photocopy (grayscale, contrast curve, toner
speckle and background lift per generation), 3 fax (bilevel at the standard
204 x 98 dpi fax resolution, then printed).
"""
import torch
from torch.nn import functional as F

ORIGINAL, LASER, PHOTOCOPY, FAX = 0, 1, 2, 3
LUMA = (.2126, .7152, .0722)


def halftone(alpha, cell_px):
    """Ordered-dither a coverage map with a 4x4 Bayer screen of ``cell_px`` pixels per dot."""
    b, _, h, w = alpha.shape
    ys = torch.arange(h, device=alpha.device, dtype=alpha.dtype)[None, :, None]
    xs = torch.arange(w, device=alpha.device, dtype=alpha.dtype)[None, None, :]
    cell = cell_px.clamp_min(1)[:, None, None]
    iy = (ys / cell).floor().long() % 4
    ix = (xs / cell).floor().long() % 4
    # 4x4 Bayer index by bit interleaving, computed on device.
    bayer2 = lambda i, j: 2 * (i ^ j) + i
    index = 4 * bayer2(iy & 1, ix & 1) + bayer2((iy >> 1) & 1, (ix >> 1) & 1)
    threshold = (index.to(alpha.dtype) + .5) / 16
    return (alpha > threshold[:, None]).to(alpha.dtype)


def _gray(image):
    return LUMA[0] * image[:, :1] + LUMA[1] * image[:, 1:2] + LUMA[2] * image[:, 2:3]


def photocopy(page, generations, generator=None):
    """Grayscale copy with ``generations`` (1-3) passes of blur, contrast and toner speckle."""
    b, _, h, w = page.shape
    gray = _gray(page)
    for generation in range(1, 4):
        active = (generations >= generation).to(page.dtype)[:, None, None, None]
        copied = F.avg_pool2d(gray, 3, 1, 1, count_include_pad=False)
        copied = torch.sigmoid((copied - .62) * 9)
        background = .05 + .05 * torch.rand(b, 1, 1, 1, device=page.device, generator=generator)
        copied = copied * (1 - background)
        speckle = (torch.rand(b, 1, h, w, device=page.device, generator=generator) < .002).to(page.dtype)
        copied = copied * (1 - .8 * speckle)
        gray = torch.lerp(gray, copied, active)
    return gray.expand(-1, 3, -1, -1)


def fax(page, ppm, threshold=.55):
    """Bilevel at 204 x 98 dpi (8.03 x 3.86 dots/mm), held over each fax dot."""
    b, _, h, w = page.shape
    gray = _gray(page)
    sx = (8.03 / ppm).clamp(max=1)[:, None]
    sy = (3.86 / ppm).clamp(max=1)[:, None]
    ys = torch.arange(h, device=page.device, dtype=page.dtype)[None]
    xs = torch.arange(w, device=page.device, dtype=page.dtype)[None]
    cx = ((xs + .5) * sx).floor().add(.5) / sx - .5
    cy = ((ys + .5) * sy).floor().add(.5) / sy - .5
    grid = torch.stack(torch.broadcast_tensors(((cx + .5) / w * 2 - 1)[:, None, :],
                                               ((cy + .5) / h * 2 - 1)[:, :, None]), -1)
    smoothed = F.avg_pool2d(gray, 3, 1, 1, count_include_pad=False)
    held = F.grid_sample(smoothed, grid, mode='nearest', padding_mode='border', align_corners=False)
    return (held > threshold).to(page.dtype).expand(-1, 3, -1, -1)


def reproduce(page, tech, ppm, generations, generator=None):
    """Apply photocopy or fax per record; other records pass through unchanged."""
    tech = tech.long()[:, None, None, None]
    copied = photocopy(page, generations, generator)
    faxed = fax(page, ppm)
    return torch.where(tech == PHOTOCOPY, copied, torch.where(tech == FAX, faxed, page))
