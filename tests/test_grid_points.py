"""Analytical lattice geometry, visibility, correspondence and both backends."""
from dataclasses import replace
import importlib.util
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from ecg_gpu import ECGSupervisionLayer, RenderConfig, SupervisionConfig, GridPointConfig
from ecg_gpu.grid_points import grid_point_targets, identity_field


class GridPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def project(self, field=None, valid=None, **options):
        field = identity_field(1, 90, 120, device='cpu') if field is None else field
        b, h, w, _ = field.shape
        valid = torch.ones(b, 1, h, w) if valid is None else valid
        arguments = dict(canonical_size=(90, 120), ppm=torch.full((b,), 2.),
                         phase_x=torch.full((b,), .3), phase_y=torch.full((b,), .7),
                         top=torch.full((b,), 12.), grid_present=torch.ones(b),
                         config=GridPointConfig(rows=16, columns=16))
        arguments.update(options)
        return grid_point_targets(field, valid, **arguments)

    def test_identity_coordinates_match_independent_five_mm_lattice(self):
        result = self.project()
        visible = result['grid_points_valid'][0]
        ids = result['grid_points_ids'][0, visible]
        expected = torch.stack((ids[:, 1]*10+.6, ids[:, 0]*10+1.4), -1)
        torch.testing.assert_close(result['grid_points_xy'][0, visible], expected, atol=2e-5, rtol=0)
        self.assertEqual(int(visible.sum()), 8*12)
        self.assertTrue(torch.equal(visible, result['grid_points_active'][0]))
        self.assertTrue((result['grid_points_xy'][~result['grid_points_valid']] == -1).all())
        self.assertLess(float(result['grid_points_error_px'].max()), 3e-5)
        mask = result['grid_points_mask']
        self.assertEqual(tuple(mask.shape), (1, 1, 90, 120))
        self.assertEqual(float(mask[0, 0, :8].max()), 0.)
        # A major grid line midpoint must not become a line mask.
        self.assertLess(float(mask[0, 0, 11, 6]), .001)

    def test_affine_rotation_translation_and_scale_match_forward_reference(self):
        # Independent forward projection: camera = A * material + shift.
        h, w = 150, 180
        theta = torch.tensor(.19)
        matrix = torch.stack((torch.stack((theta.cos(), -theta.sin())),
                              torch.stack((theta.sin(), theta.cos()))))*.82
        shift = torch.tensor([35., 20.])
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        material = (torch.stack((xx, yy), -1).float()-shift) @ torch.linalg.inv(matrix).T
        field = ((material+.5)/torch.tensor([60., 45.])-1)[None]
        valid = ((material[..., 0] >= 0) & (material[..., 0] <= 119)
                 & (material[..., 1] >= 0) & (material[..., 1] <= 89))[None, None].float()
        result = self.project(field, valid)
        visible = result['grid_points_valid'][0]
        expected = result['grid_points_canonical_xy'][0] @ matrix.T + shift
        self.assertGreater(int(visible.sum()), 80)
        torch.testing.assert_close(result['grid_points_xy'][0, visible], expected[visible], atol=3e-5, rtol=0)

    def test_nonlinear_inverse_map_round_trip_and_identity_ids(self):
        field = identity_field(1, 90, 120, device='cpu').clone()
        field[..., 0] += .08*torch.sin(field[..., 1]*4)
        result = self.project(field)
        visible = result['grid_points_valid']
        self.assertGreater(int(visible.sum()), 80)
        position = (result['grid_points_xy']+.5)/torch.tensor([60., 45.])-1
        uv = F.grid_sample(field.permute(0, 3, 1, 2), position[:, None], align_corners=False)[:, :, 0].transpose(1, 2)
        canonical = (uv+1)*torch.tensor([60., 45.])-.5
        self.assertLess(float((canonical[visible]-result['grid_points_canonical_xy'][visible]).abs().max()), .03)
        self.assertTrue(torch.equal(result['grid_points_ids'], self.project()['grid_points_ids']))

    def test_refined_coordinates_match_independent_bilinear_forward_equations(self):
        import numpy as np
        yy, xx = torch.meshgrid(torch.arange(90), torch.arange(120), indexing='ij')
        source = torch.stack((xx+.004*yy.square(), yy+.002*xx.square()), -1).float()
        field = ((source+.5)/torch.tensor([60., 45.])-1)[None]
        c = GridPointConfig(rows=16, columns=16, max_image_error_pixels=.0001)
        result = self.project(field, config=c)
        visible = result['grid_points_valid'][0]
        desired = result['grid_points_canonical_xy'][0, visible].double().numpy()
        reference = desired.copy()
        # Independent double-precision solve of the piecewise-linear sampled
        # squared-coordinate functions. This includes raster interpolation.
        for _ in range(12):
            x, y = reference.T
            fx, fy = np.floor(x), np.floor(y)
            sx = fx**2+(x-fx)*(2*fx+1)
            sy = fy**2+(y-fy)*(2*fy+1)
            rx, ry = desired[:, 0]-x-.004*sy, desired[:, 1]-y-.002*sx
            a, b = .004*(2*fy+1), .002*(2*fx+1)
            reference += np.stack(((rx-a*ry)/(1-a*b), (ry-b*rx)/(1-a*b)), -1)
        self.assertGreater(int(visible.sum()), 60)
        actual = result['grid_points_xy'][0, visible].double().numpy()
        np.testing.assert_allclose(actual, reference, atol=5e-5, rtol=0)
        self.assertLess(float(result['grid_points_error_image_px'][0, visible].max()), 1e-4)

    def test_occlusion_cuts_missing_grid_and_absent_header_region(self):
        valid = torch.ones(1, 1, 90, 120)
        valid[..., 30:61, 30:71] = 0
        result = self.project(valid=valid)
        xy = result['grid_points_canonical_xy'][0]
        covered = (xy[:, 0] > 30) & (xy[:, 0] < 70) & (xy[:, 1] > 30) & (xy[:, 1] < 60)
        self.assertTrue(result['grid_points_active'][0, covered].all())
        self.assertFalse(result['grid_points_valid'][0, covered].any())
        self.assertFalse(result['grid_points_mask'][..., 30:61, 30:71].any())
        self.assertTrue((result['grid_points_mask'] <= valid).all())
        missing = self.project(grid_present=torch.zeros(1))
        self.assertFalse(missing['grid_points_valid'].any())
        self.assertFalse(missing['grid_points_active'].any())
        self.assertFalse(missing['grid_points_mask'].any())
        hidden = self.project(valid=torch.zeros_like(valid))
        self.assertFalse(hidden['grid_points_valid'].any())

    def test_batched_phases_and_padding_are_fixed_shape(self):
        field = identity_field(2, 90, 120, device='cpu')
        result = self.project(field, phase_x=torch.tensor([.3, 6.3]), ppm=torch.tensor([2., 3.]))
        self.assertEqual(tuple(result['grid_points_xy'].shape), (2, 256, 2))
        self.assertTrue((result['grid_points_mm'][1, :, 0]-result['grid_points_mm'][0, :, 0]-1).abs().max() < 1e-5)

    def test_capacity_failure_and_configuration_validation(self):
        with self.assertRaisesRegex(RuntimeError, 'capacity exceeded'):
            self.project(config=GridPointConfig(rows=2, columns=2))
        for options in ({'rows': 0}, {'columns': True}, {'sigma_pixels': 0}, {'sigma_pixels': 5},
                        {'max_error_pixels': float('nan')}, {'min_visibility': 1.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                GridPointConfig(**options)
        with self.assertRaisesRegex(ValueError, 'visible_only'):
            SupervisionConfig(outputs=('grid_points',), visible_only=False)

    def test_invalid_map_and_metadata_contracts(self):
        invalid = (({'config':object()}, TypeError), ({'canonical_size':(1, 120)}, ValueError),
                   ({'phase_x':torch.zeros(2)}, ValueError))
        for options, error in invalid:
            with self.subTest(options=options), self.assertRaises(error):
                self.project(**options)
        with self.assertRaises(ValueError):
            self.project(field=torch.zeros(1, 1, 120, 2))
        with self.assertRaises(ValueError):
            self.project(valid=torch.ones(1, 2, 90, 120))
        with self.assertRaisesRegex(ValueError, 'float32'):
            self.project(field=identity_field(1, 90, 120, device='cpu').double())

    def test_mask_only_binary_and_joint_trace_supervision(self):
        source = torch.zeros(1, 12, 1001)
        layer = self.layer()
        layer.supervision = SupervisionConfig(outputs=('grid_points_mask',), binary=True, threshold=.25)
        with mock.patch.object(layer.renderer, '_curves', wraps=layer.renderer._curves) as curves:
            batch = layer(source)
        self.assertEqual(curves.call_args.args[5], 'image')
        self.assertEqual(set(batch), {'image', 'valid', 'grid_points_mask'})
        self.assertTrue(((batch['grid_points_mask'] == 0) | (batch['grid_points_mask'] == 1)).all())
        layer.supervision = SupervisionConfig(outputs=('lead_masks', 'grid_points'))
        joint = layer(source)
        self.assertIn('lead_masks', joint)
        self.assertIn('grid_points_xy', joint)
        self.assertTrue(torch.equal(batch['grid_points_mask'], (joint['grid_points_mask'] > .25).float()))

    def layer(self, photo=False):
        return ECGSupervisionLayer(fs=100, photo=photo, randomize=False, seed=27,
            config=RenderConfig(width=256, height=192, profile='clean', surface_mode='off',
                                camera_mode='direct', jpeg_enabled=False, shadow_samples=0),
            parameters={'layout':'3x4+r1', 'speed_index':1, 'grid_present':1., 'grid_strength':.4,
                        'grid_phase_x':.3, 'grid_phase_y':.7, 'occlusion_active':0., 'artifact_present':0.,
                        'pen_active':0., 'angle':0., 'quarter':0, 'zoom':1., 'perspective_x':0.,
                        'perspective_y':0., 'radial':0., 'translate_x':0., 'translate_y':0.,
                        'curl_x':0., 'curl_y':0., 'crease_warp':0.},
            supervision=SupervisionConfig(outputs=('grid_points',)))

    def test_supervision_opt_in_parity_and_no_host_work(self):
        source = torch.zeros(1, 12, 1001)
        layer = self.layer()
        reference = layer.render_batch(source, stage='all', photo=False)
        with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU transfer')), \
             mock.patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy')), \
             mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar sync')):
            batch = layer(source)
        self.assertTrue(torch.equal(batch['image'], reference.image))
        self.assertTrue(torch.equal(source, reference.reference))
        self.assertGreater(int(batch['grid_points_valid'].sum()), 100)
        # Same capture with identity camera must agree with rectified mode.
        other = self.layer(photo=True)(source)
        common = batch['grid_points_valid'] & other['grid_points_valid']
        self.assertGreater(int(common.sum()), 100)
        torch.testing.assert_close(batch['grid_points_xy'][common], other['grid_points_xy'][common], atol=5e-4, rtol=0)
        plain = layer.render_batch(source, stage='trace', photo=False)
        self.assertNotIn('grid_points_mask', plain.targets)
        with self.assertRaisesRegex(ValueError, 'return_grid_points'):
            layer.supervise(plain)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_outputs_stay_on_device(self):
        layer = self.layer(photo=True).cuda()
        layer.supervision = SupervisionConfig(outputs=('grid_points',),
            grid_point_config=GridPointConfig(return_offsets=True))
        result = layer(torch.zeros(1, 12, 1001, device='cuda'))
        self.assertTrue(all(value.device.type == 'cuda' for value in result.values()))

    @unittest.skipUnless(importlib.util.find_spec('mitsuba'), 'Optional Mitsuba unavailable')
    def test_offline_frontoparallel_points_match_independent_pinhole_projection(self):
        import drjit as dr
        from ecg_gpu.pathtrace import pathtrace_capture, PathTraceConfig
        dr.set_thread_count(2)
        source = torch.zeros(1, 12, 1001)
        layer = self.layer()
        canonical = layer.render_batch(source, stage='all', photo=False, as_texture=True, return_instances=True)
        config = PathTraceConfig(width=256, height=192, samples_per_pixel=1,
            label_samples_per_axis=1, surface='flat', bend_radians=0,
            camera_origin=(0., 0., .4), camera_target=(0., 0., 0.), field_of_view=49.)
        plain = pathtrace_capture(canonical, config=config, seed=72)
        result = pathtrace_capture(canonical, config=config, seed=72, return_grid_points=True)
        self.assertTrue(torch.equal(plain['image'], result['image']))
        self.assertTrue(torch.equal(plain['lead_masks'], result['lead_masks']))
        visible = result['grid_points_valid'][0]
        material = result['grid_points_canonical_xy'][0]
        ppm = canonical.metadata['ppm'][0]
        width_m, height_m = 256/ppm*.001, 192/ppm*.001
        world_x = ((material[:, 0]+.5)/256-.5)*width_m
        world_y = (.5-(material[:, 1]+.5)/192)*height_m
        # Mitsuba's perspective sensor uses an x-axis field of view and
        # look_at right = cross(direction, up), here +X.
        import math
        focal = 256/(2*math.tan(math.radians(49)/2))
        # The generated flat sheet is 80 micrometers above the table.
        expected = torch.stack((127.5+focal*world_x/(.4-.00008), 95.5-focal*world_y/(.4-.00008)), -1)
        self.assertGreater(int(visible.sum()), 100)
        torch.testing.assert_close(result['grid_points_xy'][0, visible], expected[visible], atol=.00015, rtol=0)
        self.assertTrue((result['grid_points_mask'] <= result['valid']).all())
        covered = pathtrace_capture(canonical, config=replace(config, occluder=True), seed=72, return_grid_points=True)
        lost = result['grid_points_valid'] & ~covered['grid_points_valid']
        self.assertGreater(int(lost.sum()), 0)
        self.assertTrue(torch.equal(result['grid_points_ids'], covered['grid_points_ids']))

    @unittest.skipUnless(importlib.util.find_spec('mitsuba'), 'Optional Mitsuba unavailable')
    def test_offline_oblique_fold_matches_independent_triangle_and_camera_projection(self):
        import math
        import numpy as np
        from ecg_gpu.pathtrace import pathtrace_capture, PathTraceConfig, paper_mesh
        layer = self.layer()
        canonical = layer.render_batch(torch.zeros(1, 12, 1001), stage='all', photo=False,
                                        as_texture=True, return_instances=True)
        config = PathTraceConfig(width=256, height=192, samples_per_pixel=1,
            surface='fold', bend_radians=.65, mesh_columns=33, mesh_rows=17,
            camera_origin=(.04, -.12, .4), camera_target=(0., 0., .008), camera_up=(.1, 1., 0.))
        result = pathtrace_capture(canonical, config=config, seed=72, return_grid_points=True,
            grid_point_config=GridPointConfig(return_offsets=True))
        selected = result['grid_points_valid'][0]
        uv = ((result['grid_points_canonical_xy'][0, selected]+.5)/torch.tensor([256, 192])).double().numpy()
        ppm = float(canonical.metadata['ppm'][0])
        vertices, _, _ = paper_mesh(256/ppm*.001, 192/ppm*.001, 'fold', .65, columns=33, rows=17)
        sheet = vertices.reshape(17, 33, 3).astype(np.float64)
        cell = uv*np.array([32, 16])
        ij = np.floor(cell).astype(int)
        fx, fy = (cell-ij).T
        col, row = ij.T
        a, b, c, d = sheet[row, col], sheet[row+1, col], sheet[row, col+1], sheet[row+1, col+1]
        first = a*(1-fx-fy)[:, None]+b*fy[:, None]+c*fx[:, None]
        second = c*(1-fy)[:, None]+b*(1-fx)[:, None]+d*(fx+fy-1)[:, None]
        world = np.where((fx+fy <= 1)[:, None], first, second)
        origin = np.array(config.camera_origin)
        forward = np.array(config.camera_target)-origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, config.camera_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        offset = world-origin
        z = offset@forward
        focal = config.width/(2*math.tan(math.radians(config.field_of_view)/2))
        expected = np.stack((127.5+focal*(offset@right)/z, 95.5-focal*(offset@up)/z), -1)
        self.assertGreater(int(selected.sum()), 100)
        np.testing.assert_allclose(result['grid_points_xy'][0, selected].numpy(), expected, atol=.0002, rtol=0)
        self.assertLess(float(result['grid_points_error_image_px'][0, selected].max()), .0002)


if __name__ == '__main__':
    unittest.main()
