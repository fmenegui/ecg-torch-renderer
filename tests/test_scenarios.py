import unittest

import torch

from ecg_gpu import ECGImageLayer
from ecg_gpu.scenarios import CAPTURE_SCENARIOS, capture_scenario


class CaptureScenariosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_every_scenario_renders_and_replays_with_all_twelve_leads(self):
        signals = torch.zeros(1, 12, 101)
        for name in CAPTURE_SCENARIOS:
            with self.subTest(name=name):
                config, params = capture_scenario(name, height=96, width=128)
                params['layout'] = '3x4+r1'
                layer = ECGImageLayer(fs=100, config=config, parameters=params, seed=83).eval()
                a = layer.render_batch(signals, stage='trace', return_instances=True)
                b = layer.render_batch(signals, stage='trace', return_instances=True)
                self.assertTrue(torch.equal(a.image, b.image))
                self.assertTrue(torch.isfinite(a.image).all())
                self.assertGreaterEqual(a.image.min().item(), 0.)
                self.assertLessEqual(a.image.max().item(), 1.)
                self.assertTrue(a.metadata['active'][0, :12].all())
                self.assertTrue(torch.equal(a.targets['valid'], b.targets['valid']))
                torch.testing.assert_close(a.metadata['canonical_pixels_per_second'],
                                           a.metadata['ppm'] * a.metadata['speed'])
                torch.testing.assert_close(a.metadata['source_samples_per_canonical_pixel'],
                                           100 / a.metadata['canonical_pixels_per_second'])
                torch.testing.assert_close(a.metadata['canonical_millivolts_per_pixel'],
                                           1 / (a.metadata['ppm'] * a.metadata['gain']))

    def test_scans_have_no_camera_geometry_or_motion(self):
        for name in ('clean_print', 'flatbed_scan', 'aged_scan'):
            config, params = capture_scenario(name)
            self.assertEqual(config.surface_mode, 'flat')
            for key in ('photo', 'surface_strength', 'motion', 'perspective_x', 'perspective_y', 'radial',
                        'light_energy', 'flash_strength', 'specular_strength', 'cast_shadow_strength'):
                self.assertEqual(params[key], 0.)
            self.assertEqual(params['ambient_light'], 1.)

    def test_recipes_are_fresh_and_validate_name_and_severity(self):
        _, a = capture_scenario('phone_photo')
        _, b = capture_scenario('phone_photo')
        a['ink_color'][0] = .9
        self.assertNotEqual(a['ink_color'], b['ink_color'])
        for bad in (-.1, 1.1, float('nan'), float('inf'), True, 'severe'):
            with self.assertRaises(ValueError):
                capture_scenario('phone_photo', severity=bad)
        with self.assertRaises(ValueError):
            capture_scenario('unknown')


if __name__ == '__main__':
    unittest.main()
