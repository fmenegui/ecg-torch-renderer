"""Convenience helpers, actionable error messages and demo signal options."""
import importlib.util
import unittest

import numpy as np

from ecg_gpu.demo import demo_signals

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import (ECGImageLayer, ECGRenderer, RenderConfig, LEADS, capture_scenario,
                         describe_parameters, prepare_signals, render_ecg, to_pil)


class DemoSignalTests(unittest.TestCase):
    def test_defaults_stay_strictly_periodic(self):
        signals, fs, leads = demo_signals(fs=500, duration=4)
        period = int(.85 * fs)
        np.testing.assert_allclose(signals[:, :period], signals[:, period:2*period], atol=1e-12)
        self.assertEqual(signals.shape, (12, 2000))

    def test_variability_keeps_limb_identities_and_seed(self):
        a, _, _ = demo_signals(heart_rate=80, hrv=.05, baseline_wander_mv=.1, noise_mv=.01, seed=3)
        b, _, _ = demo_signals(heart_rate=80, hrv=.05, baseline_wander_mv=.1, noise_mv=.01, seed=3)
        np.testing.assert_array_equal(a, b)
        lead_i, lead_ii, lead_iii, avr, avl, avf = a[:6]
        np.testing.assert_allclose(lead_iii, lead_ii - lead_i, atol=1e-12)
        np.testing.assert_allclose(avr, -(lead_i + lead_ii) / 2, atol=1e-12)
        np.testing.assert_allclose(avl, lead_i - lead_ii / 2, atol=1e-12)
        np.testing.assert_allclose(avf, lead_ii - lead_i / 2, atol=1e-12)

    def test_heart_rate_sets_beat_count(self):
        signals, _, _ = demo_signals(heart_rate=80, duration=10)
        lead_ii = signals[1]
        peaks = (lead_ii[1:-1] > lead_ii[:-2]) & (lead_ii[1:-1] >= lead_ii[2:]) & (lead_ii[1:-1] > .8)
        self.assertEqual(int(peaks.sum()), 13)

    def test_rejects_out_of_range_options(self):
        with self.assertRaisesRegex(ValueError, 'heart_rate'):
            demo_signals(heart_rate=200)
        with self.assertRaisesRegex(ValueError, 'hrv'):
            demo_signals(hrv=.5)


@unittest.skipUnless(TORCH, 'Torch required')
class ConvenienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = RenderConfig(height=64, width=96, profile='photo', surface_resolution=32,
                                  shadow_samples=0, antialias_samples=1)
        cls.values = np.sin(np.arange(400) / 13)[None].repeat(12, 0) * .12

    def test_prepare_signals_batches_reorders_and_scales(self):
        values = np.arange(12)[:, None] * np.ones((12, 5))
        shuffled = list(reversed(LEADS))
        prepared = prepare_signals(values[::-1].copy() * 1000, lead_names=[n.lower() for n in shuffled],
                                   units='uV')
        self.assertEqual(tuple(prepared.shape), (1, 12, 5))
        self.assertEqual(prepared.dtype, torch.float32)
        torch.testing.assert_close(prepared[0, :, 0], torch.arange(12, dtype=torch.float32))

    def test_prepare_signals_explains_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'transpose'):
            prepare_signals(np.zeros((500, 12)))
        with self.assertRaisesRegex(ValueError, "Did you mean .*'aVR'"):
            prepare_signals(np.zeros((12, 5)), lead_names=['avx', *LEADS[1:]])
        with self.assertRaisesRegex(ValueError, 'each standard lead once'):
            prepare_signals(np.zeros((12, 5)), lead_names=['I'] * 12)
        with self.assertRaisesRegex(ValueError, 'units'):
            prepare_signals(np.zeros((12, 5)), units='mmHg')
        with self.assertRaisesRegex(ValueError, 'NaN'):
            prepare_signals(np.full((12, 5), np.nan))

    def test_render_ecg_is_reproducible_and_converts_to_pil(self):
        first = render_ecg(self.values, fs=100, height=64, width=96, seed=5, device='cpu')
        second = render_ecg(self.values, fs=100, height=64, width=96, seed=5, device='cpu')
        self.assertEqual(tuple(first.shape), (1, 3, 64, 96))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        images = to_pil(first)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (96, 64))
        self.assertEqual(images[0].mode, 'RGB')

    def test_render_ecg_scenario_and_overrides(self):
        image = render_ecg(self.values, fs=100, scenario='clean_print', height=64, width=96,
                           device='cpu', parameters={'layout': '6x2'})
        self.assertEqual(tuple(image.shape), (1, 3, 64, 96))
        with self.assertRaisesRegex(ValueError, "Did you mean .*'phone_photo'"):
            render_ecg(self.values, fs=100, scenario='phone_foto', height=64, width=96, device='cpu')

    def test_describe_parameters_lists_every_layer_parameter(self):
        described = describe_parameters()
        for name in ('layout', 'angle', 'line_width', 'rhythm_leads', 'photo'):
            self.assertIn(name, described)
        self.assertEqual(described['rhythm_leads']['shape'], (3,))
        for name in described:
            self.assertIsInstance(described[name]['dtype'], str)

    def test_error_messages_suggest_fixes(self):
        layer = ECGImageLayer(fs=100, config=self.config)
        signals = prepare_signals(self.values)
        with self.assertRaisesRegex(ValueError, "Unknown rendering parameter: angel.*Did you mean .*'angle'"):
            ECGImageLayer(config=self.config, parameters={'angel': .1})
        with self.assertRaisesRegex(ValueError, "Did you mean .*'3x4\\+r1'"):
            ECGImageLayer(config=self.config, parameters={'layout': '3x4+rl'})
        with self.assertRaisesRegex(ValueError, 'Unknown surface_mode'):
            ECGImageLayer(config=self.config, parameters={'surface_mode': 'crumpledd'})
        with self.assertRaisesRegex(ValueError, "Did you mean .*'aVF'"):
            ECGImageLayer(config=self.config, parameters={'rhythm_leads': ['II', 'V1', 'avg']})
        with self.assertRaisesRegex(ValueError, 'got \\(12, 400\\).*signals\\[None\\]'):
            layer(signals[0])
        with self.assertRaisesRegex(ValueError, 'transpose'):
            layer(signals.transpose(1, 2).contiguous())
        with self.assertRaisesRegex(ValueError, 'signals.float'):
            layer(signals.double().to(torch.int64))
        with self.assertRaisesRegex(TypeError, 'prepare_signals'):
            layer(self.values[None])
        with self.assertRaisesRegex(ValueError, "Unknown rendering parameters.*Did you mean .*'zoom'"):
            ECGRenderer(self.config)(signals, 100, parameters={'zooom': 1.})

    def test_lowercase_rhythm_lead_names_resolve(self):
        layer = ECGImageLayer(fs=100, config=self.config, parameters={'rhythm_leads': ['ii', 'v1', 'V5']})
        resolved = layer.fixed_rhythm_leads
        self.assertEqual(resolved.reshape(-1).tolist(), [1, 6, 10])

    def test_scenario_overrides_still_work_through_helper(self):
        config, parameters = capture_scenario('phone_photo', height=64, width=96)
        parameters['layout'] = '3x4+r1'
        layer = ECGImageLayer(fs=100, config=config, parameters=parameters, randomize=False).eval()
        self.assertEqual(tuple(layer(prepare_signals(self.values)).shape), (1, 3, 64, 96))


if __name__ == '__main__':
    unittest.main()
