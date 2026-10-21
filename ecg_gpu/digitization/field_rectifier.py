"""Trainable point/visibility/chart heads for the consolidated research study.

No image or file operations occur in the network's forward path. The chart
predicts major-grid coordinates, supervised only during training. Inference
receives RGB. It does not infer gain, paper speed or lead/time ownership.
"""
from itertools import pairwise

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, source, target):
        super().__init__(nn.Conv2d(source, target, 3, padding=1), nn.GroupNorm(4, target), nn.SiLU(),
                         nn.Conv2d(target, target, 3, padding=1), nn.GroupNorm(4, target), nn.SiLU())


class GridFieldNet(nn.Module):
    """Identical architecture for raw, contrast and auxiliary-chart ablations."""
    def __init__(self, *, contrast=True, width=16):
        super().__init__()
        self.contrast = contrast
        widths = (width, width*2, width*4, width*6)
        self.encoders = nn.ModuleList([Block(6, widths[0])]+[Block(a, b) for a, b in pairwise(widths)])
        self.context = nn.Conv2d(widths[-1], widths[-1], 1)
        self.decoders = nn.ModuleList([Block(widths[3]+widths[2], widths[2]),
                                       Block(widths[2]+widths[1], widths[1]),
                                       Block(widths[1]+widths[0], widths[0])])
        self.head = nn.Conv2d(width, 6, 1)

    def forward(self, image):
        b, _, h, w = image.shape
        gray = image.mean(dim=1, keepdim=True)
        rgb = image
        if self.contrast:
            mean = F.avg_pool2d(image, 31, 1, 15, count_include_pad=False)
            variance = (F.avg_pool2d(image.square(), 31, 1, 15, count_include_pad=False)-mean.square()).clamp_min(.0004)
            rgb = (.5+.15*(image-mean)/variance.sqrt()).clamp(0, 1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype),
                                torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype), indexing='ij')
        coords = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
        x = torch.cat((rgb, gray, coords), 1)
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(F.avg_pool2d(x, 2) if index else x)
            skips.append(x)
        x = x+self.context(F.adaptive_avg_pool2d(x, 1))
        for decoder, skip in zip(self.decoders, skips[-2::-1]):
            x = decoder(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip), 1))
        value = self.head(x)
        return {'heat': value[:, :1], 'offsets': .5*value[:, 1:3].tanh(),
                'chart': value[:, 3:5].sigmoid(), 'visible': value[:, 5:6]}


def chart_loss(predicted, observed_normalized, ids_normalized, present):
    """Pointwise continuous-coordinate loss, no rounding of subpixel locations."""
    sampled = F.grid_sample(predicted.float(), observed_normalized[:, None].float(),
                            align_corners=False)[:, :, 0].transpose(1, 2)
    error = F.smooth_l1_loss(sampled, ids_normalized.float(), reduction='none', beta=.02)
    return (error*present[..., None]).sum()/(2*present.sum()).clamp_min(1)


def chart_lattice(xy, scores, chart_values, *, residual_limit=.3):
    """Infer relative identities from the predicted chart, then check cells.

    This is an experimental global association prior, not a correctness
    certificate. Duplicate identities retain their strongest RGB detection.
    No ground-truth coordinates enter this function.
    """
    xy, scores, chart = np.asarray(xy), np.asarray(scores), np.asarray(chart_values)
    if xy.shape != chart.shape or xy.ndim != 2 or xy.shape[1] != 2 or scores.shape != (len(xy),):
        raise ValueError('Corresponding point, score and chart arrays required')
    if not all(np.isfinite(a).all() for a in (xy, scores, chart)):
        raise ValueError('Finite predicted coordinates required')
    if not len(xy):
        return {'status': 'rejected', 'reason': 'no points', 'selected_indices': [], 'ids': [], 'cells': []}
    coordinates = chart*np.array([96., 64.])
    phase = np.angle(np.mean(np.exp(2j*np.pi*coordinates), axis=0))/(2*np.pi)
    integer = np.round(coordinates-phase).astype(int)
    residual = np.abs(coordinates-phase-integer).max(axis=1)
    chosen = {}
    for index in np.argsort(-scores, kind='stable'):
        if residual[index] <= residual_limit:
            chosen.setdefault(tuple(integer[index, ::-1]), int(index))
    pairs = sorted(chosen.items())
    if len(pairs) < 4:
        return {'status': 'rejected', 'reason': 'too few chart-consistent points',
                'selected_indices': [], 'ids': [], 'cells': []}
    ids = np.array([p[0] for p in pairs])
    ids -= ids.min(axis=0)
    selected = np.array([p[1] for p in pairs])
    lookup = {tuple(identity): index for index, identity in enumerate(ids)}
    cells, invalid = [], 0
    for row, col in ids:
        corners = ((row, col), (row, col+1), (row+1, col+1), (row+1, col))
        if not all(c in lookup for c in corners):
            continue
        quad = xy[selected[[lookup[c] for c in corners]]]
        edge = np.roll(quad, -1, axis=0)-quad
        following = np.roll(edge, -1, axis=0)
        area = edge[:, 0]*following[:, 1]-edge[:, 1]*following[:, 0]
        if (area > .5).all():
            cells.append([int(row), int(col)])
        else:
            invalid += 1
    return {'status': 'accepted' if len(cells) >= 16 else 'rejected',
            'reason': None if len(cells) >= 16 else 'insufficient chart-consistent cells',
            'selected_indices': selected.tolist(), 'ids': ids.tolist(), 'cells': cells,
            'invalid_cells': invalid, 'input_points': len(xy),
            'chart_residual_median_cells': float(np.median(residual)),
            'oracle_correspondence': False, 'global_injectivity_checked': False}
