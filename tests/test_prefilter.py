import math
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_gpu import ECGRenderer, RenderConfig  # noqa: E402
from ecg_gpu.prefilter import footprint_level, prefiltered_sample  # noqa: E402


def scaled_field(h, w, scale, angle=0.):
    """Output pixel -> canonical normalized coordinates, uniform zoom by `scale`."""
    ys, xs = torch.meshgrid((torch.arange(h) + .5) / h * 2 - 1, (torch.arange(w) + .5) / w * 2 - 1, indexing='ij')
    c, s = math.cos(angle), math.sin(angle)
    u, v = (c * xs - s * ys) * scale, (s * xs + c * ys) * scale
    return torch.stack((u, v), -1)[None]


class PrefilterTests(unittest.TestCase):
    def test_footprint_matches_uniform_zoom(self):
        field = scaled_field(64, 96, 3.)
        lod = footprint_level(field, (64, 96))
        self.assertTrue(torch.allclose(lod, torch.full_like(lod, math.log2(3.)), atol=2e-3))
        self.assertTrue((footprint_level(scaled_field(64, 96, .5), (64, 96)) == 0).all())

    def test_identity_and_magnification_equal_plain_bilinear(self):
        image = torch.rand(2, 3, 40, 56, generator=torch.Generator().manual_seed(1))
        for scale in (1., .6):
            field = scaled_field(40, 56, scale, .3).expand(2, -1, -1, -1)
            plain = F.grid_sample(image, field, align_corners=False, padding_mode='border')
            self.assertTrue(torch.equal(prefiltered_sample(image, field), plain))

    def test_minified_fine_grid_does_not_alias(self):
        # One-pixel lines every two pixels, minified 4x: the true footprint
        # average is uniform 0.5. Point sampling picks whole lines or gaps.
        h, w = 256, 256
        lines = (torch.arange(w) % 2 == 0).float().expand(h, w)[None, None]
        field = scaled_field(64, 64, 4., .07)
        plain = F.grid_sample(lines, field, align_corners=False, padding_mode='border')
        filtered = prefiltered_sample(lines, field)
        self.assertGreater(float(plain.std()), .25)
        self.assertLess(float(filtered.std()), .08)
        self.assertLess(abs(float(filtered.mean()) - .5), .05)

    def test_renderer_flag_is_opt_in_and_keeps_targets(self):
        signals = torch.sin(torch.linspace(0, 60, 2500))[None, None].expand(2, 12, -1).contiguous()
        base = dict(height=96, width=128, camera_mode='direct', jpeg_enabled=False)
        plain = ECGRenderer(RenderConfig(**base))
        filtered = ECGRenderer(RenderConfig(**base, prefilter_warp=True))
        self.assertNotIn('prefilter_warp', plain.specification()['config'])
        self.assertIn('prefilter_warp', filtered.specification()['config'])
        params = plain.sample_parameters(2, generator=torch.Generator().manual_seed(4))
        params.update(photo=torch.ones(2), quarter=torch.ones(2, dtype=torch.long), angle=torch.full((2,), -math.pi / 2))
        a = plain(signals, 250., parameters=params, generator=torch.Generator().manual_seed(8))
        b = filtered(signals, 250., parameters=params, generator=torch.Generator().manual_seed(8))
        for name, value in a.targets.items():
            if torch.is_tensor(value) and value.is_floating_point():
                self.assertTrue(torch.equal(value, b.targets[name]), name)
        self.assertFalse(torch.equal(a.image, b.image))


if __name__ == '__main__':
    unittest.main()
