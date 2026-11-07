"""Original, trainable networks. These are not proprietary PM checkpoints."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .schema_v05 import SPEEDS, GAINS, LAYOUTS


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        groups = math.gcd(8, outgoing)
        self.body = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outgoing))
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)

    def forward(self, image):
        return F.silu(self.body(image) + self.skip(image))


class ResidualUNet(nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        if width < 4 or depth not in {2, 3, 4, 5}:
            raise ValueError('width >=4 and depth 2..5 required')
        dims = [min(width * 2**i, 384) for i in range(depth + 1)]
        self.encoders = nn.ModuleList([ResidualBlock(3 if i == 0 else dims[i-1], d) for i, d in enumerate(dims)])
        self.decoders = nn.ModuleList([ResidualBlock(dims[i+1] + dims[i], dims[i]) for i in range(depth-1, -1, -1)])
        self.bottom_channels = dims[-1]
        self.width, self.depth = width, depth

    def forward(self, image):
        encoded = []
        value = image
        for i, layer in enumerate(self.encoders):
            if i:
                value = F.avg_pool2d(value, 2)
            value = layer(value)
            encoded.append(value)
        bottom = value
        for layer, skip in zip(self.decoders, reversed(encoded[:-1])):
            value = layer(torch.cat([F.interpolate(value, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], dim=1))
        return value, bottom


class GridVertexNet(nn.Module):
    """Major/minor vertex heatmaps, grid lines and visible-pixel masks."""
    output_names = ['vertices', 'grid', 'visibility']

    def __init__(self, width=32, depth=4):
        super().__init__()
        self.backbone = ResidualUNet(width, depth)
        self.head = nn.Conv2d(width, 4, 1)

    def forward(self, image):
        dense, _ = self.backbone(image)
        logits = self.head(dense)
        return {'vertices': logits[:, :2], 'grid': logits[:, 2:3], 'visibility': logits[:, 3:4]}


class LeadTraceNet(nn.Module):
    """Independent lead masks preserve simultaneous ownership at crossings."""
    output_names = ['traces', 'baselines']

    def __init__(self, width=32, depth=4):
        super().__init__()
        self.backbone = ResidualUNet(width, depth)
        self.head = nn.Conv2d(width, 24, 1)

    def forward(self, image):
        dense, _ = self.backbone(image)
        logits = self.head(dense)
        return {'traces': logits[:, :12], 'baselines': logits[:, 12:]}


class PageMetadataNet(nn.Module):
    """Lead-name masks and page-level orientation/layout/unit hypotheses."""
    output_names = ['labels', 'orientation', 'speed', 'gain', 'layout']

    def __init__(self, width=32, depth=4):
        super().__init__()
        self.backbone = ResidualUNet(width, depth)
        self.labels = nn.Conv2d(width, 12, 1)
        self.global_heads = nn.ModuleDict({
            'orientation': nn.Linear(self.backbone.bottom_channels, 4),
            'speed': nn.Linear(self.backbone.bottom_channels, len(SPEEDS)),
            'gain': nn.Linear(self.backbone.bottom_channels, len(GAINS)),
            'layout': nn.Linear(self.backbone.bottom_channels, len(LAYOUTS))})

    def forward(self, image):
        dense, bottom = self.backbone(image)
        pooled = bottom.mean(dim=(-2, -1))
        return {'labels': self.labels(dense), **{name: head(pooled) for name, head in self.global_heads.items()}}


MODELS = {'geometry': GridVertexNet, 'trace': LeadTraceNet, 'metadata': PageMetadataNet}


def build_model(stage, width=32, depth=4):
    if stage not in MODELS:
        raise ValueError('stage must be geometry, trace or metadata')
    return MODELS[stage](width, depth)


def masked_focal_dice(logits, target, valid):
    valid = valid.expand_as(target).to(logits.dtype)
    positive = (target * valid).sum().detach()
    negative = ((1-target) * valid).sum().detach()
    weight = (negative / positive.clamp_min(1)).clamp(1, 80)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    probability = logits.sigmoid()
    pt = probability*target + (1-probability)*(1-target)
    focal = (bce * (1-pt).square() * (1+(weight-1)*target) * valid).sum() / valid.sum().clamp_min(1)
    axes = (0, 2, 3)
    intersection = (probability * target * valid).sum(axes)
    denominator = ((probability + target) * valid).sum(axes)
    present = (target*valid).sum(axes) > 0
    dice = 1-(2*intersection+1)/(denominator+1)
    return focal + (dice[present].mean() if present.any() else dice.mean()*0)


def task_loss(stage, predictions, targets):
    if stage == 'geometry':
        return (masked_focal_dice(predictions['vertices'], targets['vertices'], targets['valid'])
                + masked_focal_dice(predictions['grid'], targets['grid'], targets['valid'])
                + .2*F.binary_cross_entropy_with_logits(predictions['visibility'], targets['valid']))
    if stage == 'trace':
        return (masked_focal_dice(predictions['traces'], targets['traces'], targets['valid'])
                + .3*masked_focal_dice(predictions['baselines'], targets['baselines'], targets['valid']))
    return masked_focal_dice(predictions['labels'], targets['labels'], targets['valid']) + sum(
        F.cross_entropy(predictions[name], targets[name]) for name in ['orientation', 'speed', 'gain', 'layout'])
