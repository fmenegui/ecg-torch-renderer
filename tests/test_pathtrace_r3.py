"""Independent metric and actual-ray tests for irregular and damaged 3D paper."""
from dataclasses import replace
import importlib.util
import json
import warnings
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ecg_gpu.pathtrace import (PathTraceConfig, paper_mesh, pathtrace_capture,
    _refine_paper, _clip_paper_support)


class IrregularPaperMeshTests(unittest.TestCase):
    def test_seeded_relief_metric_and_finite_front_facing_facets(self):
        for surface in ('multifold', 'wrinkled', 'crumpled'):
            args = dict(surface=surface, bend_radians=1.2, surface_strength=1.5,
                        fold_count=7, geometry_seed=41)
            vertices, uv, faces = paper_mesh(.30, .20, **args)
            repeated = paper_mesh(.30, .20, **args)
            for first, second in zip((vertices, uv, faces), repeated):
                np.testing.assert_array_equal(first, second)
            different = paper_mesh(.30, .20, **dict(args, geometry_seed=42))[0]
            self.assertGreater(float(np.max(np.abs(vertices-different))), .0001)
            metric = np.linalg.norm((uv[faces[:, [1, 2, 0]]]-uv[faces])*[.3, .2], axis=-1)
            lengths = np.linalg.norm(vertices[faces[:, [1, 2, 0]]]-vertices[faces], axis=-1)
            self.assertLess(float(np.max(np.abs(lengths/metric-1))), .03)
            normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                               vertices[faces[:, 2]]-vertices[faces[:, 0]])
            self.assertTrue(np.isfinite(vertices).all())
            self.assertTrue((normals[:, 2] > 0).all())
            self.assertGreaterEqual(float(vertices[:, 2].min()), 0.)
            self.assertGreater(float(np.ptp(vertices[:, 2])), .001)

    def test_damage_is_removed_topology_not_a_material_texture(self):
        vertices, uv, intact = paper_mesh(.30, .20, 'flat')
        removed_vertices, removed_uv, damaged = paper_mesh(.30, .20, 'flat',
            missing_corner=.30, edge_tear=.22, hole_radius=.13)
        np.testing.assert_array_equal(vertices, removed_vertices)
        np.testing.assert_array_equal(uv, removed_uv)
        self.assertLess(len(damaged), len(intact)*.94)
        kept = uv[damaged].mean(1)
        # Interior of each damage region must have no retained triangle.
        self.assertFalse(((kept[:, 0]+kept[:, 1]) > 1.72).any())
        self.assertFalse(((np.abs(kept[:, 1]-.63) < .015) & (kept[:, 0] < .17)).any())
        radius = np.linalg.norm((kept-[.67, .34])*[.30, .20], axis=1)
        self.assertGreater(float(radius.min()), .13*.20*.91)

    def test_refinement_retains_material_points_and_clipping_has_exact_boundary(self):
        vertices, uv, faces = paper_mesh(.30, .20, 'fold')
        refined, _, _, columns, rows = _refine_paper(vertices, 129, 65, 3, 2)
        np.testing.assert_allclose(refined.reshape(rows, columns, 3)[::2, ::3],
                                     vertices.reshape(65, 129, 3), rtol=0, atol=1e-9)
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], np.float32)
        texcoords = points[:, :2].copy()
        triangles = np.array([[0, 1, 2], [1, 3, 2]], np.int32)
        clipped, clipped_uv, clipped_faces = _clip_paper_support(points, texcoords,
            triangles, .37-texcoords[:, 0])
        used = clipped[clipped_faces]
        self.assertLessEqual(float(used[..., 0].max()), .3700001)
        np.testing.assert_allclose(clipped[:, :2], clipped_uv, rtol=0, atol=0)
        area = np.cross(used[:, 1]-used[:, 0], used[:, 2]-used[:, 0])[:, 2].sum()/2
        self.assertAlmostEqual(float(area), .37, delta=1e-7)
        boundary = clipped[(clipped[:, 0] > .36999) & (clipped[:, 0] < .37001)]
        self.assertGreaterEqual(len(boundary), 3)

    def test_collapsed_float32_uv_slivers_are_removed_before_metric(self):
        points = np.array([[0., 0., 0.], [1.e-8, 0., 0.], [0., 1., 0.],
                           [1., 0., 0.], [1., 1., 0.]], np.float32)
        uv = points[:, :2].copy()
        uv[1] = uv[0]
        faces = np.array([[0, 1, 2], [0, 3, 2], [3, 4, 2]], np.int32)
        diagnostics = {}
        _, _, kept = _clip_paper_support(points, uv, faces, np.ones(5), diagnostics=diagnostics)
        np.testing.assert_array_equal(kept, faces[1:])
        self.assertEqual(diagnostics['degenerate_triangles_removed'], 1)
        self.assertGreater(diagnostics['minimum_resolvable_uv_edge'], 0.)

    def test_invalid_geometry_and_degenerate_camera_rejected(self):
        for kwargs in ({'geometry_seed': -1}, {'fold_count': 0}, {'edge_tear': .31},
                       {'hole_radius': float('nan')}, {'wrinkle_amplitude_m': -.1},
                       {'camera_up': (0., 0., 0.)}, {'surface_strength': 1.51},
                       {'key_size': (0., .1)}, {'key_target': (-.18, -.14, .38)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PathTraceConfig(**kwargs)


@unittest.skipUnless(importlib.util.find_spec('mitsuba'), 'Optional Mitsuba unavailable')
class IrregularCaptureTests(unittest.TestCase):
    @staticmethod
    def canonical():
        return SimpleNamespace(image=torch.full((1, 3, 80, 120), .92),
            reference=torch.zeros(1, 12, 10),
            targets={'traces': torch.ones(1, 12, 80, 120),
                     'instances': torch.ones(1, 15, 80, 120),
                     'valid': torch.ones(1, 1, 80, 120)},
            metadata={'ppm': torch.tensor([.4]),
                      'lead_id': (torch.arange(15) % 12)[None],
                      'active': torch.ones(1, 15, dtype=torch.bool)})

    def test_oblique_fragments_source_support_metadata_is_strict_finite_json(self):
        from ecg_gpu import ECGImageLayer, capture_scenario
        config, parameters = capture_scenario('clean_print', width=1024, height=768)
        parameters.update(layout='3x4+r1', paper_age=.13, edge_wear_strength=.8,
            hole_active=[1., 0., 0.], hole_center=[[-.24, .12], [0., 0.], [0., 0.]],
            hole_radius=[[.06, .05], [.03, .03], [.03, .03]], tear_raggedness=.8)
        canonical = ECGImageLayer(fs=500, config=config, parameters=parameters,
            randomize=False, seed=784).render_batch(torch.zeros(1, 12, 5000),
                stage='all', photo=False, return_instances=True)
        settings = PathTraceConfig(width=288, height=192, samples_per_pixel=1,
            surface='wrinkled', bend_radians=.55, geometry_seed=83,
            camera_origin=(.12, -.14, .38), camera_up=(.18, 1., 0.),
            missing_corner=.23, edge_tear=.12)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            capture = pathtrace_capture(canonical, config=settings, seed=241003)
        json.dumps(capture['metadata'], allow_nan=False)
        self.assertTrue(np.isfinite(capture['metadata']['max_absolute_edge_strain']))
        self.assertGreater(capture['metadata']['metric_edges_evaluated'], 0)
        self.assertIn('degenerate_triangles_removed', capture['metadata']['mesh_numerics'])

    def test_refined_damage_with_pen_and_raking_physical_light(self):
        config = PathTraceConfig(width=288, height=160, samples_per_pixel=1,
            surface='fold', bend_radians=.3, missing_corner=.2, occluder=True,
            key_origin=(-.24, -.09, .12), key_size=(.16, .09), key_radiance=20.)
        capture = pathtrace_capture(self.canonical(), config=config, seed=73)
        self.assertTrue(torch.isfinite(capture['image']).all())
        self.assertEqual(capture['metadata']['topology_grid_columns'], 257)
        self.assertEqual(capture['metadata']['topology_grid_rows'], 129)
        self.assertIn('interpolated', capture['metadata']['damage_boundary'])
        self.assertTrue((capture['lead_masks'] <= capture['paper_mask']+1e-6).all())

    def test_true_ray_removal_replay_and_material_support(self):
        torch.set_num_threads(2)
        config = PathTraceConfig(width=96, height=64, samples_per_pixel=8,
            surface='crumpled', bend_radians=1.1, fold_count=6,
            geometry_seed=41, camera_origin=(.025, -.1, .4),
            camera_up=(.1, 1., 0.))
        canonical = self.canonical()
        baseline = pathtrace_capture(canonical, config=config, seed=37)
        damage = replace(config, missing_corner=.30, edge_tear=.22, hole_radius=.12)
        capture = pathtrace_capture(canonical, config=damage, seed=37)
        replay = pathtrace_capture(canonical, config=damage, seed=37)
        self.assertGreater(float((baseline['paper_mask']-capture['paper_mask']).sum()), 20.)
        self.assertTrue((capture['paper_mask'] <= baseline['paper_mask']+1e-7).all())
        self.assertEqual(capture['lead_masks'].shape, (1, 12, 64, 96))
        self.assertEqual(capture['instance_masks'].shape, (1, 15, 64, 96))
        torch.testing.assert_close(capture['lead_masks'], capture['paper_mask'].expand(-1, 12, -1, -1))
        torch.testing.assert_close(capture['image'], replay['image'], rtol=0, atol=2e-7)
        self.assertEqual(capture['metadata']['mesh_sha256'], replay['metadata']['mesh_sha256'])
        full_removal = (baseline['paper_mask'] > .99) & (capture['paper_mask'] == 0)
        self.assertGreater(float((capture['image']-baseline['image']).abs()[full_removal.expand(-1, 3, -1, -1)].mean()), .02)
        self.assertLess(capture['metadata']['max_absolute_edge_strain'], .03)
        # A separately supplied source-support cut is also physical topology.
        canonical.targets['paper_support'] = torch.ones(1, 1, 80, 120)
        canonical.targets['paper_support'][:, :, 28:55, 30:75] = 0
        source_damage = pathtrace_capture(canonical, config=config, seed=37)
        self.assertLess(float(source_damage['paper_mask'].sum()), float(baseline['paper_mask'].sum())-20)
        self.assertIsNotNone(source_damage['metadata']['canonical_paper_support_sha256'])
        self.assertFalse(source_damage['metadata']['external_real_image_validation'])


if __name__ == '__main__':
    unittest.main()
