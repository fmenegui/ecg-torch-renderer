"""Recorder display filters: printed trace differs from the untouched source."""
import importlib.util
import math
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.device_filters import printed_signal


def amplitude(signal):
    middle = signal[..., 1000:4000]
    return float((middle.amax(-1) - middle.amin(-1)).mean() / 2)


@unittest.skipUnless(TORCH, 'Torch required')
class FilterResponseTests(unittest.TestCase):
    def setUp(self):
        self.t = torch.arange(5000, dtype=torch.float64) / 500
        self.rates = torch.tensor([500.], dtype=torch.float64)

    def run_filter(self, frequency, highpass, lowpass, notch):
        x = torch.sin(2 * math.pi * frequency * self.t)[None, None]
        y = printed_signal(x, self.rates, torch.tensor([highpass]), torch.tensor([lowpass]), torch.tensor([notch]))
        return amplitude(y) / amplitude(x)

    def test_passband_notch_lowpass_and_highpass(self):
        self.assertAlmostEqual(self.run_filter(10., .05, 150., 0.), 1., delta=.01)
        self.assertLess(self.run_filter(50., .05, 150., 50.), .01)
        self.assertAlmostEqual(self.run_filter(100., .05, 40., 0.), 1 / math.sqrt(1 + 2.5 ** 4), delta=.01)
        self.assertLess(self.run_filter(.1, .5, 150., 0.), .25)

    def test_dc_offset_is_removed_by_highpass(self):
        x = torch.ones(1, 1, 5000, dtype=torch.float64)
        y = printed_signal(x, self.rates, torch.tensor([.05]), torch.tensor([150.]), torch.tensor([0.]))
        self.assertLess(float(y[..., 1000:4000].abs().mean()), 1e-6)

    def test_per_record_settings(self):
        x = torch.sin(2 * math.pi * 50 * self.t)[None, None].expand(2, 1, -1)
        y = printed_signal(x, self.rates.expand(2), torch.tensor([.05, .05]), torch.tensor([150., 150.]),
                           torch.tensor([0., 50.]))
        self.assertGreater(amplitude(y[0]) / amplitude(x[0]), .98)
        self.assertLess(amplitude(y[1]) / amplitude(x[1]), .01)


@unittest.skipUnless(TORCH, 'Torch required')
class RendererFilterTests(unittest.TestCase):
    def test_image_prints_filtered_signal_and_reference_is_source(self):
        base = dict(height=96, width=128, profile='clean', antialias_samples=1)
        t = torch.arange(1000) / 100
        signals = (torch.sin(2 * math.pi * 1.2 * t) * .5 + 1.)[None, None].expand(1, 12, -1).contiguous()
        renderer = ECGRenderer(RenderConfig(**base, device_filters=True))
        parameters = {'filter_highpass_hz': .5, 'filter_lowpass_hz': 40., 'filter_notch_hz': 0., 'filter_clip_mv': 0.,
                      'layout': '3x4'}
        batch = renderer(signals, 100, photo=False, stage='trace', parameters=parameters,
                         generator=torch.Generator().manual_seed(0))
        printed = batch.metadata['printed_signals']
        torch.testing.assert_close(batch.reference, signals)
        # The 1 mV offset is removed in print; the 1.2 Hz wave survives.
        self.assertLess(float(printed[..., 200:800].mean().abs()), .05)
        middle = printed[..., 200:800]
        self.assertGreater(float((middle.amax(-1) - middle.amin(-1)).mean() / 2), .3)
        plain = ECGRenderer(RenderConfig(**base))(signals, 100, photo=False, stage='trace',
                                                  parameters={'layout': '3x4'},
                                                  generator=torch.Generator().manual_seed(0))
        self.assertFalse(torch.equal(batch.targets['traces'], plain.targets['traces']))
        self.assertNotIn('printed_signals', plain.metadata)
        self.assertNotIn('device_filters', ECGRenderer(RenderConfig(**base)).specification()['config'])


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(TORCH, 'Torch required')
class ClippingAndPulseTests(unittest.TestCase):
    def test_clip_printed_per_record(self):
        from ecg_gpu.device_filters import clip_printed
        x = torch.linspace(-4, 4, 81)[None, None].expand(2, 1, -1)
        y = clip_printed(x, torch.tensor([0., 2.]))
        torch.testing.assert_close(y[0], x[0])
        self.assertAlmostEqual(float(y[1].max()), 2.)
        self.assertAlmostEqual(float(y[1].min()), -2.)

    def test_renderer_clips_print_not_reference_and_returns_pulses(self):
        t = torch.arange(1000) / 100
        signals = (3 * torch.sin(2 * math.pi * 1.2 * t))[None, None].expand(1, 12, -1).contiguous()
        renderer = ECGRenderer(RenderConfig(height=96, width=128, profile='clean', antialias_samples=1,
                                            device_filters=True, calibration_target=True))
        parameters = {'filter_highpass_hz': .05, 'filter_lowpass_hz': 150., 'filter_notch_hz': 0.,
                      'filter_clip_mv': 1.5, 'pulse_present': 1., 'layout': '3x4'}
        batch = renderer(signals, 100, photo=False, stage='trace', parameters=parameters,
                         generator=torch.Generator().manual_seed(0))
        self.assertLessEqual(float(batch.metadata['printed_signals'].abs().max()), 1.5 + 1e-6)
        torch.testing.assert_close(batch.reference, signals)
        pulses = batch.targets['calibration_pulses']
        self.assertGreater(float(pulses.sum()), 0)
        # Pulses sit left of the first column's trace start.
        first_x0 = float(batch.metadata['x0'][0, 0])
        columns = (pulses[0, 0] > .5).any(0).nonzero().flatten()
        self.assertTrue(bool((columns.float() < first_x0 + 1).all()))
        self.assertNotIn('calibration_target', ECGRenderer(RenderConfig(height=64, width=96)).specification()['config'])
