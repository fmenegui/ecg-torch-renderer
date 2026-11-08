"""ResBlock U-Net with SiLU, an explicitly underdetermined Dotter hypothesis.

The paper does not publish channel widths, exact blocks, trained weights or
the 176 annotated source images. This architecture cannot be called identical.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, padding=1), nn.GroupNorm(1, channels_out), nn.SiLU(), nn.Conv2d(channels_out, channels_out, 3, padding=1), nn.GroupNorm(1, channels_out))
        self.skip = nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, x):
        return F.silu(self.main(x) + self.skip(x))


class Dotter(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        self.architecture_choices_not_disclosed_by_authors = {"widths": list(widths), "normalization": "GroupNorm(1,C)", "upsampling": "bilinear"}
        self.encoders = nn.ModuleList([ResidualBlock(3 if i == 0 else widths[i - 1], width) for i, width in enumerate(widths)])
        self.decoders = nn.ModuleList([ResidualBlock(widths[i + 1] + widths[i], widths[i]) for i in range(len(widths) - 2, -1, -1)])
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if i != len(self.encoders) - 1:
                x = F.max_pool2d(x, 2)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.output(x)
