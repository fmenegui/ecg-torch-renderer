"""Physical geometry and independent numerical reference checks."""
import unittest
from dataclasses import replace
import torch
from torch.nn import functional as F

from ecg_gpu import ECGRenderer, RenderConfig, DocumentAssets, ECGImageLayer
from ecg_gpu.gpu_document import SURFACES


class MaterialPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def renderer(self, **config):
        return ECGRenderer(RenderConfig(height=128, width=192, profile='photo',
                                       antialias_samples=1, surface_resolution=96,
                                       shadow_samples=4, **config))

    def parameters(self, renderer):
        p = renderer.sample_parameters(8, generator=torch.Generator().manual_seed(7103))
        p['surface_mode'] = torch.arange(8)
        p['surface_strength'].fill_(.9)
        p['relief_view'][:] = torch.tensor([.38, -.25])
        return p

    def test_flat_sheet_identity_and_unit_normals_on_pixel_centers(self):
        r = self.renderer()
        p = self.parameters(r)
        p['surface_mode'].zero_()
        p['paper_gap'].zero_()
        mesh, normals, metric = r.document.surface(p, torch.Generator().manual_seed(13))
        expected = r.document.surface_model.material.expand(8, -1, -1, -1)
        torch.testing.assert_close(mesh[:, :2], expected, rtol=0, atol=3e-7)
        self.assertLess(float(mesh[:, 2].abs().max()), 1e-7)
        self.assertLess(float(metric.abs().max()), 3e-5)
        torch.testing.assert_close(normals[:, 2], torch.ones_like(normals[:, 2]), rtol=0, atol=2e-6)

    def test_bending_preserves_material_lengths_and_crumples_remain_bounded(self):
        r = self.renderer()
        p = self.parameters(r)
        mesh, normals, metric = r.document.surface(p, torch.Generator().manual_seed(13))
        self.assertTrue(torch.isfinite(mesh).all())
        torch.testing.assert_close(normals.square().sum(1), torch.ones_like(normals[:, 0]), rtol=0, atol=3e-6)
        # Trapezoidal integration and mesh differentiation have bounded
        # discretization error at sharp folds. This is not a zero-strain solver.
        for mode in range(8):
            self.assertLess(float(metric[mode].abs().mean()), .004, SURFACES[mode])
            self.assertLess(float(torch.quantile(metric[mode].abs().flatten(), .99)), .025, SURFACES[mode])
        for mode in [1, 6]:
            self.assertLess(float(metric[mode].abs().max()), .002, SURFACES[mode])

    def test_coarse_solver_matches_dense_reference_inside_visible_paper(self):
        r = self.renderer()
        dense = ECGRenderer(replace(r.config, geometry_solver='dense', inverse_iterations=12))
        p = self.parameters(r)
        # Distinct camera transforms exercise the inverse field independently
        # of the shared source projection and interpolation code.
        p['angle'] = torch.linspace(-.12, .14, 8)
        p['quarter'].zero_()
        p['zoom'].fill_(.85)
        p['curl_x'].zero_()
        p['curl_y'].zero_()
        base = r._field(p)
        a = r.document.deform(base, p, torch.Generator().manual_seed(13))
        b = dense.document.deform(base, p, torch.Generator().manual_seed(13))
        visible = ((a['field'].abs() < .96).all(-1) & (b['field'].abs() < .96).all(-1)
                   & a['solver_valid'][:, 0].bool() & b['solver_valid'][:, 0].bool())
        error = (a['field'] - b['field']).abs() * torch.tensor([192/2, 128/2])
        self.assertGreater(float(visible.float().mean()), .5)
        self.assertLess(float(error[visible].max()), .15)
        for mode in range(8):
            self.assertLess(float(a['projection_residual'][mode, 0][visible[mode]].max() * 192/2), .15)

    def test_grazing_light_shadow_sampling_is_finite_and_flat_surface_unblocked(self):
        r = self.renderer()
        p = self.parameters(r)
        p['surface_mode'].zero_()
        p['light_direction'][:] = torch.tensor([[1., .2, .02], [-.3, .6, .8], [.1, -.1, 1.]])
        mesh, normals, _ = r.document.surface(p, torch.Generator().manual_seed(13))
        direction = F.normalize(p['light_direction'], dim=-1)
        visibility = r.document._light_visibility(mesh[:, 2:], direction)
        self.assertTrue(torch.isfinite(visibility).all())
        self.assertGreater(float(visibility.min()), .70)
        lighting = r.document.illumination(mesh[:, 2:], normals, p)
        self.assertTrue(torch.isfinite(lighting).all())
        self.assertGreater(float(lighting.min()), .8)

    def test_external_bank_does_not_repeat_visibility_beyond_its_camera(self):
        h, w = 32, 48
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        field = torch.stack((2*(xx+.5)/w-1, 2*(yy+.5)/h-1), -1)[None]
        assets = DocumentAssets(field, torch.ones(1, 1, h, w))
        r = ECGRenderer(RenderConfig(height=128, width=192, surface_mode='off', shadow_samples=0), document_assets=assets)
        b = r(torch.zeros(1, 12, 1000), 100, stage='geometry', return_scene=True,
              parameters={'asset_active':1., 'zoom':.6, 'quarter':0, 'angle':0.,
                          'perspective_x':0., 'perspective_y':0., 'translate_x':0., 'translate_y':0., 'radial':0.},
              generator=torch.Generator().manual_seed(2))
        self.assertEqual(float(b.scene['paper_mask'][0, 0, 0, 0]), 0.)
        self.assertEqual(float(b.scene['paper_mask'][0, 0, 64, 96]), 1.)

    def test_checkpoint_rejects_changed_scene_backend(self):
        cfg = RenderConfig(height=64, width=96, surface_resolution=32, shadow_samples=0)
        a = ECGImageLayer(config=cfg)
        b = ECGImageLayer(config=cfg, compile_scene=True)
        with self.assertRaisesRegex(ValueError, 'Scene backend'):
            b.load_state_dict(a.state_dict())

    def test_compiled_scene_keeps_the_rng_stream_and_geometric_maps(self):
        self.check_compiled_scene(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable in this environment')
    def test_cuda_compiled_scene_preserves_maps_and_generator_under_amp(self):
        with torch.autocast('cuda', dtype=torch.float16):
            self.check_compiled_scene(torch.device('cuda'))

    def check_compiled_scene(self, device):
        cfg = RenderConfig(height=64, width=96, profile='photo', surface_resolution=32,
                           antialias_samples=1, shadow_samples=2)
        eager = ECGRenderer(cfg).to(device)
        compiled = ECGRenderer(cfg, compile_scene=True).to(device)
        signals = torch.zeros(2, 12, 1000, device=device)
        fixed = {'surface_mode':'crumpled', 'jpeg_active':0., 'quarter':0,
                 'angle':.08, 'zoom':.8, 'artifact_present':0., 'occlusion_active':0.}
        ga, gb = torch.Generator(device=device).manual_seed(772), torch.Generator(device=device).manual_seed(772)
        a = eager(signals, 100, parameters=fixed, return_scene=True, return_field=True, stage='all', generator=ga)
        b = compiled(signals, 100, parameters=fixed, return_scene=True, return_field=True, stage='all', generator=gb)
        self.assertTrue(torch.equal(ga.get_state(), gb.get_state()))
        torch.testing.assert_close(a.inverse_field, b.inverse_field, rtol=0, atol=3e-6)
        torch.testing.assert_close(a.scene['height'], b.scene['height'], rtol=0, atol=3e-6)
        torch.testing.assert_close(a.targets['traces'], b.targets['traces'], rtol=0, atol=3e-5)
        # Camera rounding can cross one 8-bit code at a float32 threshold.
        torch.testing.assert_close(a.image, b.image, rtol=0, atol=1/255+2e-6)
        self.assertEqual(b.image.dtype, torch.float32)
        self.assertEqual(b.inverse_field.device, signals.device)


if __name__ == '__main__':
    unittest.main()
