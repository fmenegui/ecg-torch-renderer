"""Batched paper/camera simulation. The forward path uses only torch operations.

JPEG simulation implements 8x8 DCT quantization and 4:2:0 chroma subsampling.
It models decoded compression artifacts, not an entropy-coded JPEG file.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CameraAppearance(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.arange(8, dtype=torch.float32)[:, None]
        n = torch.arange(8, dtype=torch.float32)[None]
        dct = torch.cos(math.pi * (2 * n + 1) * k / 16) * math.sqrt(2 / 8)
        dct[0] /= math.sqrt(2)
        self.register_buffer('dct', dct)
        luma = [[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
                [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
                [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
                [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]
        chroma = [[17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],
                  [24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],
                  *[[99] * 8 for _ in range(4)]]
        self.register_buffer('quantization', torch.tensor([luma, chroma], dtype=torch.float32))
        yy, xx = torch.meshgrid(torch.arange(-4, 5), torch.arange(-4, 5), indexing='ij')
        self.register_buffer('kernel_x', xx.float())
        self.register_buffer('kernel_y', yy.float())
        self.register_buffer('rgb_to_ycc', torch.tensor([
            [.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]]))
        self.register_buffer('ycc_to_rgb', torch.tensor([
            [1., 0., 1.402], [1., -.344136, -.714136], [1., 1.772, 0.]]))

    def _codec_plane(self, plane, quality, table):
        b, c, h, w = plane.shape
        value = F.pad(plane, (0, (-w) % 8, 0, (-h) % 8), mode='replicate') - 128
        hh, ww = value.shape[-2:]
        blocks = value.reshape(b, c, hh // 8, 8, ww // 8, 8).permute(0, 1, 2, 4, 3, 5)
        coefficients = self.dct @ blocks @ self.dct.T
        scale = torch.where(quality < 50, 5000 / quality, 200 - 2 * quality)
        quant = ((table[None] * scale[:, None, None] + 50) / 100).floor().clamp(1, 255)
        quant = quant[:, None, None, None]
        reconstructed = self.dct.T @ ((coefficients / quant).round() * quant) @ self.dct
        output = reconstructed.permute(0, 1, 2, 4, 3, 5).reshape(b, c, hh, ww) + 128
        return output[..., :h, :w]

    def jpeg(self, image, quality):
        quality = quality.float().clamp(1, 100)
        b, _, h, w = image.shape
        ycc = torch.einsum('ij,bjhw->bihw', self.rgb_to_ycc, image * 255)
        y = self._codec_plane(ycc[:, :1], quality, self.quantization[0])
        chroma = ycc[:, 1:] + 128
        chroma = F.avg_pool2d(F.pad(chroma, (0, w % 2, 0, h % 2), mode='replicate'), 2)
        chroma = self._codec_plane(chroma, quality, self.quantization[1])
        chroma = F.interpolate(chroma, size=(h + h % 2, w + w % 2), mode='bilinear', align_corners=False)[..., :h, :w] - 128
        return (torch.einsum('ij,bjhw->bihw', self.ycc_to_rgb, torch.cat([y, chroma], 1)) / 255).clamp(0, 1)

    def optical_blur(self, image, sigma, motion, angle):
        """Defocus convolved with a finite, constant-velocity shutter path.

        ``sigma`` and ``motion`` are in output pixels. The 9x9 support covers
        the renderer's sampled range (sigma <= 1.08, motion <= 2.8). A uniform
        exposure path has variance motion**2 / 12, rather than the unbounded
        Gaussian tails of an elongated defocus approximation. The kernel is
        sampled at pixel centers and normalized after its finite truncation.
        """
        b, c, h, w = image.shape
        x, y = self.kernel_x[None], self.kernel_y[None]
        ca, sa = angle.cos()[:, None, None], angle.sin()[:, None, None]
        u, v = x * ca + y * sa, -x * sa + y * ca
        sigma = sigma[:, None, None].clamp_min(.18)
        length = motion[:, None, None].clamp_min(0)
        # Integrate the Gaussian along the shutter segment analytically. Near
        # zero length use its Gaussian limit to avoid subtractive cancellation.
        scale = math.sqrt(2) * sigma
        along = ((u + length / 2) / scale).erf() - ((u - length / 2) / scale).erf()
        along *= sigma * math.sqrt(math.pi / 2) / length.clamp_min(.01)
        along = torch.where(length < .01, torch.exp(-.5 * (u / sigma).square()), along)
        kernel = along.clamp_min(0) * torch.exp(-.5 * (v / sigma).square())
        kernel = kernel / kernel.sum((-1, -2), keepdim=True)
        kernel = kernel[:, None].expand(b, c, 9, 9).reshape(b * c, 1, 9, 9)
        padded = F.pad(image, (4, 4, 4, 4), mode='replicate').reshape(1, b * c, h + 8, w + 8)
        return F.conv2d(padded, kernel, groups=b * c).reshape(b, c, h, w)

    def forward(self, image, parameters, generator=None):
        b, _, h, w = image.shape
        yy = torch.linspace(-1, 1, h, device=image.device)[None, None, :, None]
        xx = torch.linspace(-1, 1, w, device=image.device)[None, None, None, :]
        p = lambda name: parameters[name][:, None, None, None]
        light = (p('exposure') + p('light_x') * xx + p('light_y') * yy)
        light -= p('vignette') * (xx.square() + yy.square())
        distance = xx * p('shadow_cos') + yy * p('shadow_sin') - p('shadow_position')
        light *= 1 - p('shadow_strength') * torch.sigmoid(-distance / .07)
        image = (image * light * parameters['white_balance'][:, :, None, None]).clamp(0, 1)
        # Finite shutter motion and defocus share a normalized optical PSF.
        blurred = self.optical_blur(image, parameters['blur'], parameters['motion'], parameters['motion_angle'])
        image = torch.lerp(image, blurred, p('sensor_active'))
        noise = torch.randn(image.shape, device=image.device, generator=generator)
        # Independent read and signal-dependent shot noise add in variance.
        # shot_noise is a standard-deviation coefficient in observation units,
        # not an electron count. This remains a Gaussian sensor approximation.
        variance = p('read_noise').square() + p('shot_noise').square() * image.clamp_min(0)
        image = image + noise * variance.sqrt()
        image = image.clamp(0, 1).pow(p('gamma'))
        compressed = self.jpeg(image, parameters['jpeg_quality'])
        image = torch.where(p('jpeg_active').bool(), compressed, image)
        # Quantization occurs only in the observation, never in signal references.
        return ((image.clamp(0, 1) * 255).round() / 255).contiguous()

