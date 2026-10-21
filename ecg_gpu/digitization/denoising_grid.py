"""Local multitask ECG-grid detector with an auxiliary RGB restoration head.

Restored RGB is an experimental diagnostic output. Rectification samples the
original image, so a learned restoration cannot silently replace the waveform.
No absolute grid IDs or page coordinates are used by this network.
"""
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from .field_rectifier import Block


class DenoisingGridNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        widths = (width, width*2, width*4, width*6)
        self.encoders = nn.ModuleList([Block(7, width)]+[Block(a, b) for a, b in pairwise(widths)])
        self.decoders = nn.ModuleList([Block(widths[3]+widths[2], widths[2]),
                                      Block(widths[2]+widths[1], widths[1]),
                                      Block(widths[1]+widths[0], widths[0])])
        self.head = nn.Conv2d(width, 7, 1)

    def forward(self, image):
        mean = F.avg_pool2d(image, 31, 1, 15, count_include_pad=False)
        variance = (F.avg_pool2d(image.square(), 31, 1, 15, count_include_pad=False)-mean.square()).clamp_min(.0004)
        contrast = ((image-mean)/variance.sqrt()*.15+.5).clamp(0, 1)
        x = torch.cat((image, contrast, image.mean(1, keepdim=True)), 1)
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(F.avg_pool2d(x, 2) if i else x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, skips[-2::-1]):
            x = decoder(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip), 1))
        value = self.head(x)
        return {'heat': value[:, :1], 'offsets': .5*value[:, 1:3].tanh(),
                'visible': value[:, 3:4], 'clean': (image+.75*value[:, 4:7].tanh()).clamp(0, 1)}


def restoration_loss(predicted, reference, valid):
    """Masked color and edge losses, in the same observed pixel coordinates."""
    if predicted.shape != reference.shape or valid.shape != predicted[:, :1].shape:
        raise ValueError('Matching RGB images and a single-channel visibility mask required')
    predicted, reference, valid = predicted.float(), reference.float(), valid.float()
    absolute = ((predicted-reference).abs()*valid).sum()/(3*valid.sum()).clamp_min(1)
    edge = absolute*0
    for axis in (-1, -2):
        a, b = predicted.diff(dim=axis), reference.diff(dim=axis)
        visibility = torch.minimum(valid.narrow(axis, 0, valid.shape[axis]-1),
                                   valid.narrow(axis, 1, valid.shape[axis]-1))
        edge = edge+((a-b).abs()*visibility).sum()/(3*visibility.sum()).clamp_min(1)
    return {'color_l1': absolute, 'edge_l1': edge, 'loss': absolute+.25*edge}
