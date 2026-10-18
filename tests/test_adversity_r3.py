"""Independent round-three adversity acceptance checks.

These establish geometric, material and replay contracts. They are not a
photographic-equivalence benchmark and use no production box-reduction oracle.
"""
import copy
import importlib.util
import json
import unittest

import numpy as np
import torch

from ecg_gpu import ECGSupervisionLayer, RenderConfig, SupervisionConfig
from ecg_gpu.gpu_material import PaperMaterial
from ecg_gpu.pathtrace import PathTraceConfig, paper_mesh, pathtrace_capture


class AdversityRoundThreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        time = torch.arange(1001) / 100
        cls.signals = torch.stack([
            (.12 + .006 * lead) * torch.sin(time * (3.2 + .08 * lead))
            for lead in range(12)
        ])[None]

    @staticmethod
    def parameters(**overrides):
        # Isolate geometry/damage from random camera and foreground artifacts.
        result = dict(layout='3x4+r3', angle=0., zoom=1., quarter=0,
            translate_x=0., translate_y=0., perspective_x=0., perspective_y=0.,
            radial=0., curl_x=0., curl_y=0., crease_warp=0., surface_strength=0.,
            tear_active=0., occlusion_active=0., pen_active=0., artifact_present=0.,
            asset_active=0., lead_present=1., paper_age=0., thermal_fade=0.,
            coffee_strength=0., stain_strength=0., sensor_active=0., exposure=1.,
            light_x=0., light_y=0., vignette=0., shadow_strength=0.,
            white_balance=[1., 1., 1.], gamma=1., jpeg_active=0.,
            read_noise=0., shot_noise=0., band_strength=0., paper_noise=0.)
        result.update(overrides)
        return result

    def layer(self, **parameters):
        return ECGSupervisionLayer(fs=100,
            config=RenderConfig(height=96, width=160, profile='clean',
                surface_mode='off', shadow_samples=0, antialias_samples=1),
            parameters=self.parameters(**parameters), randomize=False, seed=202,
            supervision=SupervisionConfig(outputs=(
                'lead_masks', 'lead_boxes', 'instance_masks', 'instance_boxes')))

    def assert_boxes_by_enumeration(self, masks, boxes, validity):
        for index in range(masks.shape[1]):
            coordinates = torch.nonzero(masks[0, index] > .05)
            self.assertEqual(bool(validity[0, index]), bool(len(coordinates)))
            if not len(coordinates):
                self.assertEqual(boxes[0, index].tolist(), [-1] * 4)
                continue
            y, x = coordinates.unbind(1)
            self.assertEqual(boxes[0, index].tolist(), [
                int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1])

    def assert_metadata_has_no_dense_images(self, value, path='metadata'):
        if isinstance(value, dict):
            for key, child in value.items():
                self.assert_metadata_has_no_dense_images(child, path + '.' + str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                self.assert_metadata_has_no_dense_images(child, f'{path}[{index}]')
        elif isinstance(value, torch.Tensor):
            self.assertLessEqual(value.numel(), 512, path)
            self.assertFalse(value.ndim >= 2 and tuple(value.shape[-2:]) == (96, 160), path)

    def test_projective_geometry_has_nonconstant_scale_unlike_a_crop(self):
        layer = self.layer(perspective_x=.24, perspective_y=-.12)
        rendered = layer.render_batch(self.signals, stage='all', return_field=True, return_instances=True)
        field = rendered.inverse_field[0]
        # Independently evaluate three projective point correspondences in
        # pixel-center coordinates. A crop/zoom has no position-varying ratio.
        for y_pixel, x_pixel in ((10, 20), (47, 79), (80, 135)):
            x = (x_pixel + .5 - 80) / 80
            y = (y_pixel + .5 - 48) / 80
            denominator = 1 + .24 * x - .12 * y
            expected = torch.tensor([x / denominator, y / denominator * 160 / 96])
            torch.testing.assert_close(field[y_pixel, x_pixel], expected, rtol=0, atol=2e-7)
        horizontal_steps = field[47, 1:, 0] - field[47, :-1, 0]
        self.assertGreater(float(horizontal_steps.max() / horizontal_steps.min()), 1.5)
        crop = self.layer(zoom=.8).render_batch(self.signals, stage='all', return_field=True)
        crop_steps = crop.inverse_field[0, 47, 1:, 0] - crop.inverse_field[0, 47, :-1, 0]
        self.assertLess(float(crop_steps.max() - crop_steps.min()), 2e-7)
        batch = layer.supervise(rendered)
        self.assertEqual(tuple(batch['lead_masks'].shape), (1, 12, 96, 160))
        self.assertEqual(tuple(batch['instance_masks'].shape), (1, 15, 96, 160))
        self.assert_boxes_by_enumeration(batch['lead_masks'], batch['lead_boxes'], batch['lead_box_valid'])
        self.assert_boxes_by_enumeration(batch['instance_masks'], batch['instance_boxes'], batch['instance_box_valid'])

    def test_projective_horizon_is_rejected_before_division(self):
        # At the left page edge, denominator is about -0.19. A finite-clamped
        # grid would hide a physically invalid camera rather than fix it.
        with self.assertRaisesRegex(RuntimeError, 'camera horizon'):
            self.layer(perspective_x=1.2).render_batch(self.signals, stage='all')

    def test_edge_fast_path_rejects_advanced_damage_instead_of_ignoring_it(self):
        for overrides in (dict(hole_active=[1., 0., 0.]),
                          dict(cut_active=[1., 0., 0.]),
                          dict(missing_corner_active=1.),
                          dict(edge_wear_strength=.5), dict(tear_count=2)):
            with self.subTest(parameters=overrides):
                layer = ECGSupervisionLayer(fs=100, config=RenderConfig(
                    height=96, width=160, profile='clean', surface_mode='off',
                    shadow_samples=0, antialias_samples=1, damage_detail='edge'),
                    parameters=self.parameters(**overrides), seed=202, randomize=False)
                with self.assertRaisesRegex(RuntimeError, 'requires damage_detail=rich'):
                    layer.render_batch(self.signals, stage='trace')

    def test_zero_damage_sizes_are_valid_only_for_inactive_slots(self):
        # Empty padded slots may have zero dimensions, as in detector targets.
        zeros = dict(hole_active=[0., 0., 0.], hole_radius=[[0., 0.]] * 3,
            cut_active=[0., 0., 0.], cut_length=[0.] * 3, cut_width=[0.] * 3,
            missing_corner_active=0., missing_corner_size=[0., 0.])
        clean = self.layer().render_batch(self.signals, stage='trace')
        padded = self.layer(**zeros).render_batch(self.signals, stage='trace')
        torch.testing.assert_close(padded.targets['valid'], clean.targets['valid'], rtol=0, atol=0)
        for activation, message in [(dict(hole_active=[1., 0., 0.]), 'hole_radius'),
                                    (dict(cut_active=[1., 0., 0.]), 'cut_length'),
                                    (dict(missing_corner_active=1.), 'missing_corner_size')]:
            with self.subTest(activation=activation):
                invalid = {**zeros, **activation}
                with self.assertRaisesRegex(RuntimeError, message):
                    self.layer(**invalid).render_batch(self.signals, stage='trace')

    def test_geometric_removal_flags_are_binary_not_partial_alpha(self):
        for overrides in (dict(tear_active=.5), dict(hole_active=[.5, 0., 0.]),
                          dict(cut_active=[0., .5, 0.]), dict(missing_corner_active=.5)):
            with self.subTest(parameters=overrides):
                with self.assertRaisesRegex(RuntimeError, 'must be binary'):
                    self.layer(**overrides).render_batch(self.signals, stage='trace')

    def test_material_zero_strength_identity_and_selective_blue_absorption(self):
        material = PaperMaterial(96, 160)
        zero = material({'paper_age': torch.zeros(1)}, torch.Generator().manual_seed(17))
        for value in zero.values():
            torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)
        aged = material({'paper_age': torch.ones(1)}, torch.Generator().manual_seed(17))
        factor = aged['substrate_factor']
        self.assertTrue((factor[:, 0] >= factor[:, 1]).all())
        self.assertTrue((factor[:, 1] >= factor[:, 2]).all())
        self.assertLess(float(factor[:, 2].mean()), float(factor[:, 0].mean()) - .1)
        # Aged substrate does not silently erase print or add an opaque hole.
        torch.testing.assert_close(aged['print_retention'], torch.ones_like(aged['print_retention']))
        torch.testing.assert_close(aged['deposit_transmission'], torch.ones_like(aged['deposit_transmission']))

    def test_material_replay_and_spatially_nonuniform_deposition(self):
        material = PaperMaterial(96, 160)
        parameters = {key: torch.tensor([value]) for key, value in dict(
            paper_age=.65, coffee_strength=.8, coffee_radius=.22,
            stain_strength=.4, stain_x=.45, stain_y=.52, foxing_strength=.6,
            fiber_strength=.8, water_strength=.5, thermal_fade=.75).items()}
        first = material(parameters, torch.Generator().manual_seed(37))
        replay = material(parameters, torch.Generator().manual_seed(37))
        changed = material(parameters, torch.Generator().manual_seed(38))
        for key, value in first.items():
            torch.testing.assert_close(value, replay[key], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(((value >= 0) & (value <= 1)).all())
            self.assertGreater(float((value - changed[key]).abs().max()), 1e-4)
        # An entire annulus is not constrained to one analytical color. Check
        # azimuths at approximately equal physical distance from the center.
        deposit = first['deposit_transmission'][0, 2]
        values = torch.tensor([deposit[50, 106], deposit[75, 78], deposit[50, 36], deposit[19, 78]])
        self.assertGreater(float(values.max() - values.min()), .025)

    def test_integrated_aging_and_liquids_change_rgb_without_erasing_geometric_gt(self):
        clean = self.layer().render_batch(self.signals, stage='all', return_instances=True)
        aged = self.layer(paper_age=.9, coffee_strength=.9, coffee_radius=.24,
            stain_strength=.7, stain_x=.47, stain_y=.52, thermal_fade=.85,
            foxing_strength=.7, water_strength=.7, fiber_strength=.8).render_batch(
                self.signals, stage='all', return_instances=True)
        self.assertGreater(float((clean.image - aged.image).abs().mean()), .025)
        for key in ('traces', 'instances', 'valid', 'frame_coverage'):
            torch.testing.assert_close(clean.targets[key], aged.targets[key], rtol=0, atol=0)
        torch.testing.assert_close(aged.reference, self.signals, rtol=0, atol=0)

    def test_adversity_replays_without_changing_reference_or_dense_metadata(self):
        layer = self.layer(perspective_x=.21, perspective_y=.13, paper_age=.8,
                           coffee_strength=.8, stain_strength=.6, thermal_fade=.4)
        original = self.signals.clone()
        first = layer.render_batch(self.signals, stage='all', return_instances=True)
        replay = layer.render_batch(self.signals, stage='all', return_instances=True)
        torch.testing.assert_close(first.image, replay.image, rtol=0, atol=0)
        for key in first.targets:
            torch.testing.assert_close(first.targets[key], replay.targets[key], rtol=0, atol=0)
        torch.testing.assert_close(self.signals, original, rtol=0, atol=0)
        torch.testing.assert_close(first.reference, original, rtol=0, atol=0)
        self.assert_metadata_has_no_dense_images(first.metadata)
        # The layer's random-call counter survives checkpoint/resume. Use
        # isolated state copies rather than one advanced generator instance.
        layer.randomize = True
        layer(self.signals)
        state = copy.deepcopy(layer.state_dict())
        expected = layer(self.signals)
        restored = self.layer(perspective_x=.21, perspective_y=.13, paper_age=.8,
                              coffee_strength=.8, stain_strength=.6, thermal_fade=.4)
        restored.randomize = True
        restored.load_state_dict(state)
        actual = restored(self.signals)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_interior_elliptical_hole_has_independent_material_metric_oracle(self):
        parameters = dict(tear_raggedness=0.,
            hole_active=[1., 0., 0.],
            hole_center=[[.10, .20], [0., 0.], [0., 0.]],
            hole_radius=[[.24, .18], [.1, .1], [.1, .1]],
            hole_angle=[0., 0., 0.])
        layer = self.layer(**parameters)
        rendered = layer.render_batch(self.signals, stage='all', return_instances=True,
                                      return_field=True)
        field = rendered.inverse_field
        # In a 160x96 page, physical y is normalized by width, not height.
        x = field[..., 0] - .10
        y = (field[..., 1] - .20) * (96 / 160)
        expected = ((x / .24).square() + (y / .18).square() >= 1).float()[:, None]
        torch.testing.assert_close(rendered.targets['valid'], expected, rtol=0, atol=0)
        clean = self.layer().render_batch(self.signals, photo=False, stage='all', return_instances=True)
        batch = layer.supervise(rendered)
        self.assertGreater(int((expected == 0).sum()), 300)
        for source_key, output_key in [('traces', 'lead_masks'), ('instances', 'instance_masks')]:
            # Identity projection permits an exact source-derived alpha oracle.
            reference = clean.targets[source_key] * expected
            torch.testing.assert_close(batch[output_key], reference, rtol=0, atol=8e-6)
            self.assertGreater(float((clean.targets[source_key] - batch[output_key]).sum()), 1.)
            prefix = 'lead' if source_key == 'traces' else 'instance'
            self.assert_boxes_by_enumeration(reference, batch[prefix + '_boxes'], batch[prefix + '_box_valid'])

    def test_cut_capsule_and_missing_corner_are_removals_not_camera_crops(self):
        layer = self.layer(tear_raggedness=0., cut_active=[1., 0., 0.],
            cut_center=[[.08, -.15], [0., 0.], [0., 0.]],
            cut_length=[.85, .2, .2], cut_width=[.045, .01, .01], cut_angle=[0., 0., 0.])
        rendered = layer.render_batch(self.signals, stage='all', return_instances=True, return_field=True)
        field = rendered.inverse_field
        # A finite slit is a capsule, not a full-frame band or translated crop.
        dx = (field[..., 0] - .08).abs() - .85 / 2
        dy = (field[..., 1] + .15) * (96 / 160)
        distance = torch.sqrt(dx.clamp_min(0).square() + dy.square())
        reference = (distance >= .045 / 2).float()[:, None]
        torch.testing.assert_close(rendered.targets['valid'], reference, rtol=0, atol=0)
        self.assertTrue((reference[..., 0, :] == 1).all())
        self.assertTrue((reference[..., -1, :] == 1).all())
        self.assertGreater(int((reference == 0).sum()), 50)
        corner_layer = self.layer(tear_raggedness=0., missing_corner_active=1.,
                                 missing_corner_index=0, missing_corner_size=[.36, .28])
        corner = corner_layer.render_batch(self.signals, stage='all', return_instances=True)
        self.assertEqual(float(corner.targets['valid'][0, 0, 0, 0]), 0.)
        self.assertEqual(float(corner.targets['valid'][0, 0, 48, 80]), 1.)
        self.assertEqual(float(corner.targets['valid'][0, 0, -1, -1]), 1.)
        for owner, render in [(layer, rendered), (corner_layer, corner)]:
            batch = owner.supervise(render)
            self.assertEqual(batch['lead_masks'].shape[1], 12)
            self.assertEqual(batch['instance_masks'].shape[1], 15)
            for prefix in ('lead', 'instance'):
                masks = batch[prefix + '_masks']
                self.assertFalse((masks * (batch['valid'] == 0)).any())
                self.assert_boxes_by_enumeration(masks, batch[prefix + '_boxes'], batch[prefix + '_box_valid'])

    def test_irregular_damage_replays_and_follows_rotated_material_coordinates(self):
        layer = self.layer(angle=.27, zoom=.83, perspective_x=.18,
            tear_raggedness=1., tear_active=1., tear_count=3, tear_size=.15,
            edge_wear_strength=.8, edge_wear_depth=.018,
            hole_active=[1., 1., 0.],
            hole_center=[[-.3, -.4], [.28, .3], [0., 0.]],
            hole_radius=[[.14, .12], [.13, .18], [.1, .1]],
            cut_active=[1., 0., 0.], cut_length=[.6, .2, .2],
            cut_width=[.04, .01, .01], cut_angle=[.6, 0., 0.],
            missing_corner_active=1., missing_corner_index=2,
            missing_corner_size=[.3, .25])
        original = self.signals.clone()
        first = layer.render_batch(self.signals, stage='all', return_instances=True, return_field=True)
        replay = layer.render_batch(self.signals, stage='all', return_instances=True, return_field=True)
        torch.testing.assert_close(first.image, replay.image, rtol=0, atol=0)
        torch.testing.assert_close(first.targets['valid'], replay.targets['valid'], rtol=0, atol=0)
        torch.testing.assert_close(self.signals, original, rtol=0, atol=0)
        self.assert_metadata_has_no_dense_images(first.metadata)
        batch = layer.supervise(first)
        # A point well inside each hole must disappear even after rotation and
        # perspective. The material centers are far from all other boundaries.
        field = first.inverse_field[0]
        for cx, cy in ((-.3, -.4), (.28, .3)):
            near_center = (field[..., 0] - cx).square() + ((field[..., 1] - cy) * .6).square() < .03 ** 2
            self.assertGreater(int(near_center.sum()), 2)
            self.assertFalse(first.targets['valid'][0, 0, near_center].any())
        for prefix in ('lead', 'instance'):
            masks = batch[prefix + '_masks']
            self.assertFalse((masks * (batch['valid'] == 0)).any())
            self.assert_boxes_by_enumeration(masks, batch[prefix + '_boxes'], batch[prefix + '_box_valid'])

    def test_multifold_metric_and_damaged_mesh_replay(self):
        kwargs = dict(surface='multifold', bend_radians=1.1, geometry_seed=41,
                      fold_count=7, surface_strength=1.2)
        clean_vertices, uv, clean_faces = paper_mesh(.3, .18, **kwargs)
        damaged_vertices, damaged_uv, damaged_faces = paper_mesh(.3, .18,
            missing_corner=.22, edge_tear=.14, hole_radius=.07, **kwargs)
        replay = paper_mesh(.3, .18, missing_corner=.22, edge_tear=.14, hole_radius=.07, **kwargs)
        for actual, expected in zip(replay, (damaged_vertices, damaged_uv, damaged_faces)):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(damaged_vertices, clean_vertices)
        np.testing.assert_array_equal(damaged_uv, uv)
        self.assertLess(len(damaged_faces), len(clean_faces) - 50)
        self.assertTrue(set(map(tuple, damaged_faces)).issubset(set(map(tuple, clean_faces))))
        # Material-coordinate edge lengths, independently reconstructed from
        # UV and meter dimensions, verify ruled multifold developability.
        for a, b in ((0, 1), (1, 2), (2, 0)):
            actual = np.linalg.norm(clean_vertices[clean_faces[:, a]] - clean_vertices[clean_faces[:, b]], axis=1)
            reference = np.linalg.norm((uv[clean_faces[:, a]] - uv[clean_faces[:, b]]) * [.3, .18], axis=1)
            self.assertLess(float(np.max(np.abs(actual / reference - 1))), 1e-4)
        # A finite hole is actually absent from topology, not a dark texture.
        centroid = damaged_uv[damaged_faces].mean(1)
        radius = np.hypot((centroid[:, 0] - .67) * .3, (centroid[:, 1] - .34) * .18)
        self.assertFalse((radius < .07 * .18 * .90).any())

    @unittest.skipUnless(importlib.util.find_spec('mitsuba'), 'Optional Mitsuba dependency unavailable')
    def test_actual_crumpled_capture_preserves_fifteen_instance_targets_and_replay(self):
        canonical = self.layer().render_batch(self.signals, photo=False, stage='all', return_instances=True)
        config = PathTraceConfig(width=96, height=64, samples_per_pixel=4,
            surface='crumpled', bend_radians=1.1, surface_strength=1.2,
            geometry_seed=41, fold_count=7, missing_corner=.22,
            edge_tear=.14, hole_radius=.07, camera_up=(.07, 1., .02))
        original = canonical.reference.clone()
        first = pathtrace_capture(canonical, config=config, seed=611)
        replay = pathtrace_capture(canonical, config=config, seed=611)
        self.assertEqual(tuple(first['lead_masks'].shape), (1, 12, 64, 96))
        self.assertEqual(tuple(first['instance_masks'].shape), (1, 15, 64, 96))
        # LLVM parallel radiance accumulation can differ by a few FP32 ULPs.
        # Meshes and stratified geometric labels below remain bit-exact.
        torch.testing.assert_close(first['image'], replay['image'], rtol=0, atol=3e-7)
        torch.testing.assert_close(canonical.reference, original, rtol=0, atol=0)
        self.assertEqual(first['metadata']['mesh_sha256'], replay['metadata']['mesh_sha256'])
        for prefix in ('lead', 'instance'):
            masks = first[prefix + '_masks']
            self.assertTrue(torch.isfinite(masks).all())
            self.assertFalse((masks * (first['paper_mask'] == 0)).any())
            torch.testing.assert_close(masks, replay[prefix + '_masks'], rtol=0, atol=0)
            self.assert_boxes_by_enumeration(masks, first[prefix + '_boxes'], first[prefix + '_box_valid'])
        self.assert_metadata_has_no_dense_images(first['metadata'])
        # Exported provenance must be strict interoperable JSON. NaN or Inf
        # mesh metrics are invalid even when image pixels happen to be finite.
        json.dumps(first['metadata'], allow_nan=False)


if __name__ == '__main__':
    unittest.main()
