"""Independent primary-ray geometry and color-fiducial QA for offline capture."""
from dataclasses import asdict, replace
import hashlib
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ecg_gpu.pathtrace import PathTraceConfig, _visible_samples, paper_mesh, pathtrace_capture


MITSUBA = importlib.util.find_spec('mitsuba') is not None


class PaperMeshQualityTests(unittest.TestCase):
    def test_meter_scale_arc_length_and_explicit_top_left_uv(self):
        for surface in ('flat', 'curl', 'fold'):
            with self.subTest(surface=surface):
                vertices, uv, faces = paper_mesh(.30, .20, surface, bend_radians=1.)
                sheet = vertices.reshape(65, 129, 3)
                lengths = np.linalg.norm(np.diff(sheet, axis=1), axis=-1).sum(1)
                np.testing.assert_allclose(lengths, .30, rtol=0, atol=1e-7)
                np.testing.assert_allclose(sheet[0, :, 1] - sheet[-1, :, 1], .20,
                                           rtol=0, atol=2e-8)
                np.testing.assert_array_equal(uv[[0, 128, -129, -1]],
                                               [[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
                triangle = vertices[faces]
                normals = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
                self.assertTrue((normals[:, 2] > 0).all())

    def test_local_triangle_edge_strain_under_maximum_fold_and_curl(self):
        for surface in ('flat', 'curl', 'fold'):
            with self.subTest(surface=surface):
                vertices, uv, faces = paper_mesh(.30, .20, surface, bend_radians=1.2)
                for first, last in ((0, 1), (1, 2), (2, 0)):
                    edges = vertices[faces[:, last]] - vertices[faces[:, first]]
                    material = (uv[faces[:, last]] - uv[faces[:, first]]) * np.array([.30, .20])
                    reference = np.linalg.norm(material, axis=-1)
                    strain = np.linalg.norm(edges, axis=-1) / reference - 1
                    self.assertLess(float(np.max(np.abs(strain))), 1e-4)


@unittest.skipUnless(MITSUBA, 'Optional Mitsuba dependency unavailable')
class PathTraceQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mitsuba as mi
        import drjit as dr
        mi.set_variant('llvm_ad_rgb')
        dr.set_thread_count(2)
        torch.set_num_threads(2)
        cls.mi = mi

    @staticmethod
    def canonical_fiducials():
        image = torch.ones(1, 3, 80, 120)
        masks = torch.zeros(1, 12, 80, 120)
        # Each color owns a different asymmetric position and canonical ID.
        for channel, (y, x) in enumerate(((10, 15), (10, 80), (50, 15))):
            image[:, :, y:y+20, x:x+25] = 0.
            image[:, channel, y:y+20, x:x+25] = 1.
            masks[:, channel, y:y+20, x:x+25] = 1.
        return SimpleNamespace(image=image, reference=torch.arange(120, dtype=torch.float32).reshape(1, 12, 10), targets={
            'traces': masks, 'instances': masks.clone(), 'valid': torch.ones(1, 1, 80, 120)},
            metadata={'ppm': torch.tensor([.4]), 'lead_id': torch.arange(12)[None],
                      'active': torch.ones(1, 12, dtype=torch.bool)})

    def test_bitmap_colors_match_target_fiducials_without_vertical_or_channel_flip(self):
        config = PathTraceConfig(width=96, height=64, samples_per_pixel=16, surface='flat',
            camera_origin=(0., 0., .40), camera_target=(0., 0., 0.),
            key_radiance=1., environment_radiance=.3)
        capture = pathtrace_capture(self.canonical_fiducials(), config=config, seed=149)
        self.assertAlmostEqual(capture['metadata']['width_m'], .30, delta=1e-7)
        self.assertAlmostEqual(capture['metadata']['height_m'], .20, delta=1e-7)
        for lead in range(3):
            interior = capture['lead_masks'][0, lead] > .99
            self.assertGreater(int(interior.sum()), 15)
            rgb = capture['image'][0, :, interior].mean(1)
            self.assertGreater(float(rgb[lead]), float(rgb[(lead+1) % 3]) * 2)
            self.assertGreater(float(rgb[lead]), float(rgb[(lead+2) % 3]) * 2)
            coordinates = torch.nonzero(capture['lead_masks'][0, lead] > .05)
            y, x = coordinates.unbind(1)
            expected = [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]
            self.assertEqual(capture['lead_boxes'][0, lead].tolist(), expected)
        self.assertFalse(capture['lead_box_valid'][0, 3:].any())
        self.assertTrue((capture['lead_boxes'][0, 3:] == -1).all())
        self.assertFalse((capture['lead_masks'] * (capture['paper_mask'] == 0)).any())

    def test_public_occluder_branch_renders_and_removes_only_covered_paper_targets(self):
        canonical = self.canonical_fiducials()
        config = PathTraceConfig(width=96, height=64, samples_per_pixel=8,
            surface='fold', bend_radians=.65, occluder=True,
            camera_origin=(.025, -.115, .38), camera_target=(0., 0., .008),
            field_of_view=49., key_radiance=1.1, environment_radiance=.21,
            background_extent_m=1.7)
        # This exercises the production cylinder BSDF construction. A custom
        # analytical-occluder scene would not detect a malformed RGB property.
        covered = pathtrace_capture(canonical, config=config, seed=17)
        uncovered = pathtrace_capture(canonical, config=replace(config, occluder=False), seed=17)
        self.assertTrue(torch.isfinite(covered['image']).all())
        removed = uncovered['paper_mask'] - covered['paper_mask']
        self.assertGreater(float(removed.sum()), 1.)
        self.assertGreaterEqual(float(removed.min()), 0.)
        self.assertTrue((covered['lead_masks'] <= uncovered['lead_masks'] + 1e-7).all())
        metadata = covered['metadata']
        self.assertEqual(metadata['configuration'], asdict(config))
        self.assertEqual(metadata['seed'], 17)
        self.assertEqual(metadata['canonical_reference_shape'], [1, 12, 10])
        self.assertEqual(metadata['canonical_reference_dtype'], '<f4')
        self.assertEqual(metadata['canonical_reference_sha256'], hashlib.sha256(
            canonical.reference.numpy().astype('<f4').tobytes()).hexdigest())
        for key in ('implementation_sha256', 'canonical_texture_sha256'):
            self.assertRegex(metadata[key], r'^[0-9a-f]{64}$')
        self.assertFalse(metadata['external_real_image_validation'])

    def ray_scene(self, config, *, occluder=False):
        mi = self.mi
        transform = mi.ScalarTransform4f
        scene = {'type': 'scene',
            'sensor': {'type': 'perspective', 'fov': config.field_of_view,
                'to_world': transform.look_at(origin=config.camera_origin,
                    target=config.camera_target, up=[0, 1, 0]),
                'film': {'type': 'hdrfilm', 'width': config.width, 'height': config.height,
                         'rfilter': {'type': 'box'}}},
            'paper': {'type': 'rectangle', 'to_world': transform.scale([.1, .1, 1]),
                      'bsdf': {'type': 'twosided', 'bsdf': {'type': 'diffuse'}}}}
        if occluder:
            scene['occluder'] = {'type': 'rectangle',
                'to_world': transform.translate([.05, 0., .1]).scale([.05, .1, 1]),
                'bsdf': {'type': 'diffuse'}}
        scene = mi.load_dict(scene)
        return scene, next(shape for shape in scene.shapes() if shape.id() == 'paper')

    def test_visible_samples_match_analytic_plane_occlusion_with_actual_sensor_rays(self):
        mi = self.mi
        config = PathTraceConfig(width=64, height=64, samples_per_pixel=1,
            camera_origin=(0., 0., .4), camera_target=(0., 0., 0.), surface='flat')
        scene, paper = self.ray_scene(config, occluder=True)
        samples = list(_visible_samples(mi, scene, config, paper))
        yy, xx = np.mgrid[:64, :64]
        for index, (field, visible) in enumerate(samples):
            iy, ix = divmod(index, 2)
            position = mi.Point2f(mi.Float(((xx+(ix+.5)/2)/64).astype(np.float32).ravel()),
                                  mi.Float(((yy+(iy+.5)/2)/64).astype(np.float32).ravel()))
            ray, _ = scene.sensors()[0].sample_ray(0., .5, position, mi.Point2f(.5))
            origin, direction = ray.o.numpy().T, ray.d.numpy().T
            target = origin + (-origin[:, 2]/direction[:, 2])[:, None]*direction
            upper = origin + ((.1-origin[:, 2])/direction[:, 2])[:, None]*direction
            on_paper = (np.abs(target[:, :2]) < .1).all(1)
            blocked = ((upper[:, 0] >= 0) & (upper[:, 0] <= .1) & (np.abs(upper[:, 1]) <= .1))
            expected = (on_paper & ~blocked).reshape(64, 64)
            np.testing.assert_array_equal(visible[0, 0].numpy().astype(bool), expected)
            # Mitsuba rectangle UV, independently mapped from world x/y.
            projected = field[0].numpy().reshape(-1, 2)
            np.testing.assert_allclose(projected[expected.ravel(), 0], target[expected.ravel(), 0]/.1,
                                       rtol=0, atol=2e-6)

    def test_backside_primary_hits_never_receive_front_print_labels(self):
        config = PathTraceConfig(width=64, height=64, samples_per_pixel=1,
            camera_origin=(0., 0., -.4), camera_target=(0., 0., 0.), surface='flat')
        scene, paper = self.ray_scene(config)
        visible = list(_visible_samples(self.mi, scene, config, paper))
        self.assertEqual(len(visible), 4)
        self.assertTrue(all(not support.any() for _, support in visible))


if __name__ == '__main__':
    unittest.main()
