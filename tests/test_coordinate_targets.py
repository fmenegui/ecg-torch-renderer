"""Dense per-pixel time and voltage targets, checked against the source signal."""
import importlib.util
import math
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig


@unittest.skipUnless(TORCH, 'Torch required')
class CoordinateTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.fs = 100
        t = torch.arange(1000) / cls.fs
        cls.signals = (.5 * torch.sin(2 * math.pi * .5 * t))[None, None].expand(1, 12, -1).contiguous()
        renderer = ECGRenderer(RenderConfig(height=480, width=640, profile='clean', antialias_samples=2,
                                            coordinate_targets=True))
        parameters = {'layout': '3x4', 'speed_index': 1, 'gain_index': 1, 'line_width': 1.5,
                      'lead_present': 1., 'sequential': True}
        cls.batch = renderer(cls.signals, cls.fs, photo=False, stage='trace', parameters=parameters,
                             generator=torch.Generator().manual_seed(4))

    def test_voltage_on_trace_pixels_matches_source(self):
        targets, meta = self.batch.targets, self.batch.metadata
        on_trace = (targets['traces'][0, 0] > .9) & (targets['coordinate_valid'][0, 0] > 0)
        x0, x1 = float(meta['x0'][0, 0]), float(meta['x1'][0, 0])
        columns = torch.arange(on_trace.shape[1])
        on_trace &= ((columns >= x0) & (columns < x1))[None]
        self.assertGreater(int(on_trace.sum()), 50)
        time = targets['time_s'][0, 0][on_trace]
        voltage = targets['voltage_mv'][0, 0][on_trace]
        source = .5 * torch.sin(2 * math.pi * .5 * time)
        error = (voltage - source).abs()
        self.assertLess(float(error.mean()), .06)
        self.assertLess(float(error.max()), .12)

    def test_time_is_linear_in_x_within_a_cell(self):
        targets, meta = self.batch.targets, self.batch.metadata
        ppm, speed = float(meta['ppm'][0]), float(meta['speed'][0])
        lead = 3  # aVR, second column
        x0, start = float(meta['x0'][0, lead]), float(meta['start_s'][0, lead])
        y = int(round(float(meta['baseline'][0, lead])))
        x = int(math.ceil(x0)) + 5
        expected = start + (x - x0) / (speed * ppm)
        self.assertAlmostEqual(float(targets['time_s'][0, 0, y, x]), expected, places=4)
        self.assertAlmostEqual(float(targets['voltage_mv'][0, 0, y, x]),
                               (float(meta['baseline'][0, lead]) - y) / (float(meta['gain'][0]) * ppm), places=4)

    def test_maps_are_absent_by_default(self):
        plain = ECGRenderer(RenderConfig(height=96, width=128, profile='clean'))
        batch = plain(self.signals, self.fs, photo=False, stage='trace')
        self.assertNotIn('time_s', batch.targets)
        self.assertNotIn('coordinate_targets', plain.specification()['config'])


if __name__ == '__main__':
    unittest.main()
