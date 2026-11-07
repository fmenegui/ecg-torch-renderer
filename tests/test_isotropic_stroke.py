"""Perpendicular stroke width of RenderConfig(stroke='isotropic').

The reference is analytic: a straight antialiased line of width w covers w
pixels per unit length, measured perpendicular to the line. The legacy kernel
thins steep lines; the isotropic kernel must not.
"""
import importlib.util
import math
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.gpu_raster import curve_coverage, curve_coverage_isotropic, raster_kernels

SIZE, SAMPLES, LINE_WIDTH = 120, 4, 3.


def straight_line_inputs(slope):
    xx = torch.arange(SIZE, dtype=torch.float32)[None, None, None, :]
    yy = torch.arange(SIZE, dtype=torch.float32)[None, None, :, None]
    axes = torch.stack([xx[0, 0, 0] - .5 + sample / SAMPLES for sample in range(SAMPLES)])
    px = (xx - axes[:, None, None, :])[:, 0]
    curve = lambda x: 60 + slope * (x - 60)
    ya = curve(axes)[None, None]
    yb = curve(axes + 1 / SAMPLES)[None, None]
    within = torch.ones(1, 1, 1, SIZE, dtype=torch.bool)
    thickness = torch.full((1, 1, 1, 1), LINE_WIDTH / 2 + .5)
    return ya, yb, px, yy, thickness, within


def perpendicular_width(mask, slope):
    columns = slice(58, 63)
    return float(mask[0, 0, :, columns].sum()) / (5 * math.hypot(1, slope))


@unittest.skipUnless(TORCH, 'Torch required')
class IsotropicStrokeTests(unittest.TestCase):
    def test_width_matches_line_width_at_every_slope(self):
        for slope in (0., 1., 3., 10.):
            with self.subTest(slope=slope):
                inputs = straight_line_inputs(slope)
                iso = curve_coverage_isotropic(*inputs, None, None, False, False, radius=3)
                self.assertAlmostEqual(perpendicular_width(iso, slope), LINE_WIDTH, delta=.05 * LINE_WIDTH)

    def test_legacy_kernel_is_unchanged_and_still_thins_steep_lines(self):
        inputs = straight_line_inputs(10.)
        legacy = curve_coverage(*inputs, None, None, False, False)
        self.assertLess(perpendicular_width(legacy, 10.), .5 * LINE_WIDTH)
        flat = straight_line_inputs(0.)
        torch.testing.assert_close(curve_coverage_isotropic(*flat, None, None, False, False, radius=3),
                                   curve_coverage(*flat, None, None, False, False))

    def test_extremum_envelope_is_isotropic_too(self):
        ya, yb, px, yy, thickness, within = straight_line_inputs(0.)
        low = torch.full((1, 1, SIZE), float('inf'))
        high = torch.full((1, 1, SIZE), -float('inf'))
        low[..., 60], high[..., 60] = 20., 59.
        mask = curve_coverage_isotropic(ya, yb, px, yy, thickness, within, low, high, True, False, radius=3)
        # A vertical spike one column wide is drawn with the full stroke width.
        row = mask[0, 0, 40]
        self.assertAlmostEqual(float(row.sum()), LINE_WIDTH, delta=.05 * LINE_WIDTH)

    def test_kernel_selection_and_config(self):
        self.assertIs(raster_kernels(False)[0], curve_coverage)
        with self.assertRaisesRegex(ValueError, 'stroke'):
            raster_kernels(False, stroke='round')
        with self.assertRaisesRegex(ValueError, 'stroke_radius'):
            RenderConfig(stroke='isotropic', stroke_radius=0)
        legacy = ECGRenderer(RenderConfig(height=64, width=96))
        self.assertNotIn('stroke', legacy.specification()['config'])

    def test_renderer_draws_isotropic_traces(self):
        config = dict(height=96, width=128, profile='clean', antialias_samples=2)
        signals = torch.zeros(1, 12, 1000)
        signals[..., ::50] = 1.5
        parameters = {'layout': '3x4', 'line_width': 2.5, 'lead_present': 1.}
        masks = {}
        for stroke in ('legacy', 'isotropic'):
            renderer = ECGRenderer(RenderConfig(**config, stroke=stroke))
            batch = renderer(signals, 100, photo=False, stage='trace', parameters=parameters,
                             generator=torch.Generator().manual_seed(2))
            masks[stroke] = batch.targets['traces']
        self.assertGreater(float(masks['isotropic'].sum()), float(masks['legacy'].sum()))
        self.assertTrue(bool((masks['isotropic'] >= masks['legacy'] - 1e-6).all()))


if __name__ == '__main__':
    unittest.main()
