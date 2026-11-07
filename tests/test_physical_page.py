"""Analytic contracts for RenderConfig(page_model='physical').

The physical page keeps a fixed plotted paper width, so paper speed selects the
shown time window, stroke and grid widths are millimeters, and a paper-feed
error scales the trace against the pre-printed grid. Legacy behaviour, including
its random stream and specification, must not change.
"""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGImageLayer, ECGRenderer, RenderConfig, capture_scenario, render_ecg
    from ecg_gpu.schema_v05 import SPEEDS

MARGIN_MM = 21.


@unittest.skipUnless(TORCH, 'Torch required')
class PhysicalPageGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.renderer = ECGRenderer(RenderConfig(height=384, width=640, profile='clean',
                                                antialias_samples=2, page_model='physical'))

    def parameters(self, count, layout='3x4+r1'):
        p = self.renderer.sample_parameters(count, photo=False, generator=torch.Generator().manual_seed(9))
        p['layout'].fill_(self.renderer.layout_names.index(layout))
        p['sequential'].fill_(True)
        p['lead_present'].fill_(1)
        p['feed_error'].zero_()
        return p

    def test_speed_sets_time_window_on_fixed_paper(self):
        p = self.parameters(3)
        p['speed_index'] = torch.arange(3)
        data = self.renderer._instances(p, torch.full((3,), 30.))
        expected_ppm = (640 - 1) / (250. + MARGIN_MM)
        torch.testing.assert_close(data['ppm'], torch.full((3,), expected_ppm))
        speeds = torch.tensor(SPEEDS[1:])
        # The rhythm strip spans the whole 250 mm plot: 20, 10 and 5 seconds.
        torch.testing.assert_close(data['end_s'][:, 12], 250. / speeds)
        # Each 3x4 column covers a quarter of the plot.
        torch.testing.assert_close(data['end_s'][:, 0] - data['start_s'][:, 0], 62.5 / speeds)
        printed_s = (data['x1'] - data['x0']) / (data['ppm'] * data['trace_speed'])[:, None]
        torch.testing.assert_close(printed_s, data['end_s'] - data['start_s'])

    def test_short_record_ends_early_and_later_columns_are_empty(self):
        p = self.parameters(1)
        p['speed_index'].fill_(1)
        data = self.renderer._instances(p, torch.tensor([4.]))
        ppm = data['ppm'][0]
        torch.testing.assert_close(data['end_s'][0, 12], torch.tensor(4.))
        torch.testing.assert_close(data['x1'][0, 12] - data['x0'][0, 12], 4. * 25. * ppm)
        columns = data['column'][0, :12]
        active = data['active'][0, :12]
        self.assertTrue(bool(active[columns == 0].all()) and bool(active[columns == 1].all()))
        self.assertFalse(bool(active[columns >= 2].any()))
        torch.testing.assert_close(data['end_s'][0, :12][columns == 1], torch.full((3,), 4.))

    def test_feed_error_scales_trace_but_not_grid(self):
        p = self.parameters(2)
        p['speed_index'].fill_(1)
        p['feed_error'] = torch.tensor([0., .02])
        data = self.renderer._instances(p, torch.full((2,), 30.))
        torch.testing.assert_close(data['ppm'][0], data['ppm'][1])
        torch.testing.assert_close(data['trace_speed'], torch.tensor([25., 25.5]))
        torch.testing.assert_close(data['end_s'][1, 12], data['end_s'][0, 12] / 1.02)
        torch.testing.assert_close(data['x1'][1, 12], data['x1'][0, 12])

    def test_controls_are_physical_only_and_legacy_stream_is_unchanged(self):
        legacy = ECGRenderer(RenderConfig(height=384, width=640, profile='clean', antialias_samples=2))
        legacy_p = legacy.sample_parameters(2, generator=torch.Generator().manual_seed(4))
        physical_p = self.renderer.sample_parameters(2, generator=torch.Generator().manual_seed(4))
        for name in ('trace_width_mm', 'grid_line_mm', 'feed_error'):
            self.assertNotIn(name, legacy_p)
            self.assertIn(name, physical_p)
        for name, value in legacy_p.items():
            torch.testing.assert_close(physical_p[name], value, rtol=0, atol=0, msg=name)
        self.assertNotIn('page_model', legacy.specification()['config'])
        self.assertEqual(self.renderer.specification()['config']['page_model'], 'physical')

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, 'page_model'):
            RenderConfig(page_model='paper')
        with self.assertRaisesRegex(ValueError, 'plot_width_mm'):
            RenderConfig(page_model='physical', plot_width_mm=10.)


@unittest.skipUnless(TORCH, 'Torch required')
class PhysicalStrokeWidthTests(unittest.TestCase):
    def render_flat(self, width):
        renderer = ECGRenderer(RenderConfig(height=width * 3 // 4, width=width, profile='clean',
                                            antialias_samples=2, page_model='physical'))
        signals = torch.zeros(1, 12, 5000)
        parameters = {'layout': '3x4', 'trace_width_mm': .25, 'grid_line_mm': .08, 'feed_error': 0.,
                      'speed_index': 1, 'gain_index': 1, 'lead_present': 1., 'sequential': True}
        batch = renderer(signals, 500, photo=False, stage='trace', parameters=parameters,
                         generator=torch.Generator().manual_seed(3))
        return batch

    def test_trace_and_grid_widths_are_millimeters_at_any_resolution(self):
        for width in (1200, 1600):
            with self.subTest(width=width):
                batch = self.render_flat(width)
                meta = batch.metadata
                ppm = meta['ppm'][0]
                column = int(((meta['x0'][0, 0] + meta['x1'][0, 0]) / 2).round())
                # Vertical coverage integral of a flat antialiased line equals its width.
                profile = batch.targets['traces'][0, 0, :, column]
                torch.testing.assert_close(profile.sum() / ppm, torch.tensor(.25), rtol=.01, atol=0)
                torch.testing.assert_close(meta['parameters']['grid_width'][0], .08 * ppm)

    def test_pixel_width_overrides_are_rejected(self):
        renderer = ECGRenderer(RenderConfig(height=96, width=128, page_model='physical'))
        with self.assertRaisesRegex(ValueError, 'trace_width_mm'):
            renderer(torch.zeros(1, 12, 500), 100, parameters={'line_width': 1.})


@unittest.skipUnless(TORCH, 'Torch required')
class PhysicalRecipeTests(unittest.TestCase):
    def test_scenario_and_render_ecg_accept_physical_pages(self):
        config, parameters = capture_scenario('phone_photo', height=96, width=128, page_model='physical')
        self.assertEqual(config.page_model, 'physical')
        self.assertNotIn('line_width', parameters)
        self.assertIn('trace_width_mm', parameters)
        signals = torch.sin(torch.arange(1500) / 13)[None, None].expand(1, 12, -1) * .2
        layer = ECGImageLayer(100, config=config, parameters=parameters, randomize=False).eval()
        self.assertEqual(tuple(layer(signals).shape), (1, 3, 96, 128))
        image = render_ecg(signals, 100, height=96, width=128, device='cpu', page_model='physical')
        self.assertEqual(tuple(image.shape), (1, 3, 96, 128))


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(TORCH, 'Torch required')
class TraceSkewTests(unittest.TestCase):
    def test_flat_trace_drifts_against_grid_by_the_skew(self):
        renderer = ECGRenderer(RenderConfig(height=480, width=640, profile='clean', antialias_samples=2,
                                            page_model='physical', coordinate_targets=True))
        parameters = {'layout': '3x4+r1', 'trace_width_mm': .3, 'feed_error': 0., 'trace_skew': .01,
                      'speed_index': 1, 'gain_index': 1, 'lead_present': 1., 'sequential': True}
        batch = renderer(torch.zeros(1, 12, 1000), 100, photo=False, stage='trace', parameters=parameters,
                         generator=torch.Generator().manual_seed(0))
        meta = batch.metadata
        rhythm = batch.targets['traces'][0, 1]  # lead II rhythm strip spans the page
        rows = torch.arange(rhythm.shape[0], dtype=torch.float32)
        lower = rows > float(meta['baseline'][0, 12]) - 20
        def centroid(x):
            column = rhythm[:, x] * lower
            return float((column * rows).sum() / column.sum())
        x1, x2 = int(meta['x0'][0, 12]) + 20, int(meta['x1'][0, 12]) - 20
        self.assertAlmostEqual(centroid(x2) - centroid(x1), .01 * (x2 - x1), delta=.6)
        on_trace = (batch.targets['traces'][0, 1] > .9) & lower[:, None]
        self.assertLess(float(batch.targets['voltage_mv'][0, 0][on_trace].abs().max()), .2)
        torch.testing.assert_close(meta['trace_skew'], torch.tensor([.01]))
