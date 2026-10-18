"""Independent physical checks for optical blur and sensor observations."""
import math
import unittest

import torch

from ecg_gpu import ECGRenderer, RenderConfig
from ecg_gpu.gpu_appearance import CameraAppearance


class OpticalRealismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def impulse_response(self, sigma, motion, angle):
        size = len(sigma)
        image = torch.zeros(size, 3, 25, 25)
        image[:, :, 12, 12] = 1
        return CameraAppearance().optical_blur(
            image, torch.tensor(sigma), torch.tensor(motion), torch.tensor(angle))

    def test_finite_shutter_matches_independent_numerical_exposure(self):
        sigma, length, angle = .83, 2.6, .63
        response = self.impulse_response([sigma], [length], [angle])[0, 0, 8:17, 8:17]
        # Independently integrate snapshots of a moving Gaussian point source.
        # This reference uses midpoint quadrature, not the renderer's erf model.
        y, x = torch.meshgrid(torch.arange(-4, 5), torch.arange(-4, 5), indexing='ij')
        shutter = (torch.arange(4096, dtype=torch.float64) + .5) / 4096 * length - length / 2
        dx = x[..., None] - shutter * math.cos(angle)
        dy = y[..., None] - shutter * math.sin(angle)
        reference = torch.exp(-(dx.square() + dy.square()) / (2 * sigma**2)).mean(-1)
        reference /= reference.sum()
        torch.testing.assert_close(response.double(), reference, rtol=2e-5, atol=3e-8)

    def test_shutter_variance_and_orientation_obey_uniform_exposure(self):
        sigma, motion = .7, 2.8
        response = self.impulse_response([sigma]*3, [motion]*3, [0., math.pi/2, math.pi/4])[:, 0]
        y, x = torch.meshgrid(torch.arange(-12, 13), torch.arange(-12, 13), indexing='ij')
        torch.testing.assert_close(response.sum((-1, -2)), torch.ones(3), rtol=0, atol=2e-7)
        # A uniform line exposure adds L^2/12 to variance along its direction.
        along, across = sigma**2 + motion**2/12, sigma**2
        self.assertAlmostEqual(float((response[0] * x.square()).sum()), along, delta=.012)
        self.assertAlmostEqual(float((response[0] * y.square()).sum()), across, delta=.012)
        torch.testing.assert_close(response[0].T, response[1], rtol=0, atol=4e-8)
        self.assertAlmostEqual(float((response[2] * x * y).sum()), motion**2/24, delta=.012)

    def test_zero_motion_recovers_isotropic_defocus_and_flat_fields_stay_flat(self):
        response = self.impulse_response([.75, .75], [0., .0001], [.13, 1.2])
        torch.testing.assert_close(response[0], response[1], rtol=0, atol=3e-8)
        constant = torch.tensor([.2, .5, .8])[:, None, None].expand(3, 19, 23)[None].expand(2, -1, -1, -1)
        blurred = CameraAppearance().optical_blur(
            constant, torch.tensor([.18, 1.08]), torch.tensor([0., 2.8]), torch.tensor([0., .72]))
        torch.testing.assert_close(blurred, constant, rtol=0, atol=2e-7)

    def test_sensor_noise_variance_adds_independent_read_and_shot_terms(self):
        camera = CameraAppearance()
        b = 2
        parameters = {key: torch.zeros(b) for key in [
            'light_x', 'light_y', 'vignette', 'shadow_position', 'shadow_strength',
            'motion', 'motion_angle', 'sensor_active', 'jpeg_active']}
        parameters.update(exposure=torch.ones(b), shadow_cos=torch.ones(b), shadow_sin=torch.zeros(b),
                          white_balance=torch.ones(b, 3), blur=torch.full((b,), .18),
                          read_noise=torch.full((b,), .03), shot_noise=torch.full((b,), .04),
                          gamma=torch.ones(b), jpeg_quality=torch.full((b,), 95.))
        levels = torch.tensor([.2, .7])
        image = levels[:, None, None, None].expand(b, 3, 192, 192)
        first = camera(image, parameters, torch.Generator().manual_seed(811))
        replay = camera(image, parameters, torch.Generator().manual_seed(811))
        self.assertTrue(torch.equal(first, replay))
        variance = (first-image).flatten(1).var(1)
        expected = .03**2 + .04**2 * levels + 1/(12 * 255**2)
        torch.testing.assert_close(variance, expected, rtol=.02, atol=0)
        self.assertEqual(first.dtype, torch.float32)

    def test_observation_degradation_preserves_signal_and_geometric_targets(self):
        renderer = ECGRenderer(RenderConfig(height=64, width=96, profile='clean', antialias_samples=1))
        signal = torch.sin(torch.arange(400)[None, None] / 12).expand(1, 12, -1) * .2
        common = {'sensor_active':1., 'jpeg_active':0., 'artifact_present':0., 'pen_active':0.,
                  'occlusion_active':0., 'gamma':1., 'read_noise':0., 'shot_noise':0.}
        clean = renderer(signal, 100, photo=False, stage='all', parameters={**common, 'blur':.18, 'motion':0.},
                         generator=torch.Generator().manual_seed(37))
        adverse = renderer(signal, 100, photo=False, stage='all',
                           parameters={**common, 'blur':1., 'motion':2.8, 'motion_angle':.7, 'read_noise':.02},
                           generator=torch.Generator().manual_seed(37))
        self.assertGreater(float((clean.image-adverse.image).abs().mean()), .005)
        self.assertTrue(torch.equal(adverse.reference, signal))
        for key in clean.targets:
            torch.testing.assert_close(clean.targets[key], adverse.targets[key], rtol=0, atol=0)


class RendererInputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.renderer = ECGRenderer(RenderConfig(height=64, width=96, profile='clean', antialias_samples=1))

    def test_fractional_and_nonfinite_lengths_cannot_change_time_windows_silently(self):
        signals = torch.zeros(1, 12, 100)
        for value in [99.5, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'integer sample counts'):
                self.renderer(signals, 100, photo=False, lengths=[value])

    def test_sample_rate_shape_has_a_useful_error(self):
        signals = torch.zeros(2, 12, 100)
        for rates in [[[100.], [100.]], [100., 100., 100.]]:
            with self.subTest(rates=rates), self.assertRaisesRegex(ValueError, 'fs must be'):
                self.renderer(signals, rates, photo=False)

    def test_integral_float_lengths_retain_exact_source_duration(self):
        batch = self.renderer(torch.zeros(2, 12, 100), [100., 200.], lengths=[90., 80.], photo=False,
                              stage='image', generator=torch.Generator().manual_seed(19))
        self.assertEqual(batch.metadata['source_length'].tolist(), [90, 80])
        torch.testing.assert_close(batch.metadata['duration_s'], torch.tensor([.89, .395]))

    def test_raster_precision_rejects_squared_overflow_without_clinical_clipping(self):
        unsafe = torch.full((1, 12, 20), 1e38)
        with self.assertRaisesRegex(RuntimeError, 'safe float32 raster precision'):
            self.renderer(unsafe, 100, photo=False, stage='trace')
        # A very large but numerically representable waveform is left intact.
        representable = torch.full((1, 12, 20), 1e8)
        batch = self.renderer(representable, 100, photo=False, stage='trace',
                              generator=torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(batch.reference, representable))
        self.assertTrue(torch.isfinite(batch.image).all())
        self.assertTrue(torch.isfinite(batch.targets['traces']).all())


if __name__ == '__main__':
    unittest.main()
