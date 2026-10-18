"""Radiometric references independent of the implementation's color helpers."""
import math
import unittest

import torch

from ecg_gpu.gpu_appearance import CameraAppearance, linear_to_srgb, srgb_to_linear


def parameters(batch=1, **overrides):
    result = {key: torch.zeros(batch) for key in [
        'light_x', 'light_y', 'vignette', 'shadow_position', 'shadow_strength',
        'motion', 'motion_angle', 'read_noise', 'shot_noise', 'jpeg_active']}
    result.update(exposure=torch.ones(batch), shadow_cos=torch.ones(batch),
                  shadow_sin=torch.zeros(batch), white_balance=torch.ones(batch, 3),
                  blur=torch.full((batch,), .18), sensor_active=torch.ones(batch),
                  gamma=torch.ones(batch), jpeg_quality=torch.full((batch,), 95.))
    for key, value in overrides.items():
        result[key] = torch.as_tensor(value).expand_as(result[key])
    return result


def encode_scalar(value):
    return 12.92 * value if value <= .0031308 else 1.055 * value ** (1 / 2.4) - .055


def decode_reference(value):
    return torch.where(value <= .04045, value / 12.92,
                       ((value + .055) / 1.055) ** 2.4)


class LinearCapturePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_srgb_transfer_reference_points_and_hdr_headroom(self):
        encoded = torch.tensor([0., .04045, .5, 1.], dtype=torch.float64)
        expected = torch.tensor([0., .0031308049535603713, .21404114048223255, 1.], dtype=torch.float64)
        torch.testing.assert_close(srgb_to_linear(encoded), expected, rtol=0, atol=2e-16)
        linear = torch.tensor([0., .0031308, .18, .5, 1., 2.], dtype=torch.float64)
        reference = torch.tensor([encode_scalar(float(v)) for v in linear], dtype=torch.float64)
        torch.testing.assert_close(linear_to_srgb(linear), reference, rtol=0, atol=2e-16)
        self.assertGreater(float(linear_to_srgb(linear)[-1]), 1.)
        torch.testing.assert_close(srgb_to_linear(linear_to_srgb(linear)), linear, rtol=0, atol=3e-9)

    def test_one_stop_reduces_irradiance_not_display_code(self):
        output = CameraAppearance()(torch.ones(1, 3, 15, 15), parameters(exposure=.5))
        expected = round(.7353569830524495 * 255) / 255
        torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=1e-7)

    def test_blur_adds_photons_across_black_white_edge(self):
        image = torch.zeros(1, 3, 25, 25)
        image[..., 12:] = 1
        output = CameraAppearance()(image, parameters(blur=.7))
        # Independent discrete Gaussian at x=12. White samples occupy x>=12.
        weights = [math.exp(-.5 * (offset / .7) ** 2) for offset in range(-4, 5)]
        irradiance = sum(weights[:5]) / sum(weights)
        expected = round(encode_scalar(irradiance) * 255) / 255
        self.assertAlmostEqual(float(output[0, 0, 12, 12]), expected, delta=1e-7)
        self.assertGreater(expected - irradiance, .05)

    def test_highlight_energy_spreads_before_sensor_saturation(self):
        image = torch.zeros(1, 3, 25, 25)
        image[..., 12, 12] = 1
        output = CameraAppearance()(image, parameters(exposure=4., blur=.7))
        weights = [math.exp(-.5 * (offset / .7) ** 2) for offset in range(-4, 5)]
        irradiance = 4 * weights[4] * weights[5] / sum(weights) ** 2
        expected = round(encode_scalar(irradiance) * 255) / 255
        self.assertAlmostEqual(float(output[0, 0, 12, 13]), expected, delta=1e-7)
        self.assertGreater(float(output[0, 0, 12, 13]), .6)

    def test_linear_sensor_noise_and_post_sensor_white_balance(self):
        level, read, shot = .6, .012, .020
        gains = torch.tensor([.7, .9, 1.1])
        camera = CameraAppearance()
        image = torch.full((1, 3, 256, 256), level)
        p = parameters(read_noise=read, shot_noise=shot, white_balance=gains)
        output = camera(image, p, torch.Generator().manual_seed(702))
        replay = camera(image, p, torch.Generator().manual_seed(702))
        self.assertTrue(torch.equal(output, replay))
        measured = decode_reference(output).var((-1, -2))[0]
        reference_linear = ((level + .055) / 1.055) ** 2.4
        expected = (read ** 2 + shot ** 2 * reference_linear) * gains ** 2
        torch.testing.assert_close(measured, expected, rtol=.025, atol=0)

    def test_inactive_sensor_retains_document_display_path(self):
        image = torch.full((1, 3, 13, 17), .8)
        output = CameraAppearance()(image, parameters(sensor_active=0., exposure=.5, blur=1., motion=2.))
        torch.testing.assert_close(output, torch.full_like(output, .4), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
