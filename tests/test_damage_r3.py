"""Independent material-space oracles for missing paper, not painted artifacts."""
import unittest

import torch

from ecg_gpu.gpu_document import DocumentScene


class MaterialDamageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def scene_and_parameters(self, height=80, width=120):
        scene = DocumentScene(height, width, resolution=32, shadow_samples=0)
        parameters = scene.sample_parameters(
            1, torch.ones(1), torch.ones(1),
            generator=torch.Generator().manual_seed(17))
        for key in ['tear_active', 'edge_wear_strength', 'hole_active',
                    'cut_active', 'missing_corner_active']:
            parameters[key].zero_()
        return scene, parameters

    def points(self, coordinates):
        return torch.tensor(coordinates, dtype=torch.float32)[None, None]

    def test_absent_damage_is_exact_unit_support_even_outside_source_frame(self):
        scene, p = self.scene_and_parameters()
        field = self.points([[-1.2, -1.3], [-1., -1.], [0., 0.], [1., 1.], [1.2, 1.3]])
        # Framing belongs to grid_sample's canonical support, never this mask.
        torch.testing.assert_close(scene.paper_mask(field, p), torch.ones(1, 1, 1, 5), rtol=0, atol=0)

    def test_irregular_hole_removes_center_but_retains_distant_paper(self):
        scene, p = self.scene_and_parameters()
        p['hole_active'][0, 0] = 1
        p['hole_center'][0, 0] = torch.tensor([.1, -.2])
        p['hole_radius'][0, 0] = torch.tensor([.12, .08])
        field = self.points([[.1, -.2], [.1 + .12 * 1.4, -.2], [.1, -.2 + .08 * 1.4 / scene.aspect]])
        self.assertEqual(scene.paper_mask(field, p).flatten().tolist(), [0., 1., 1.])

    def test_elliptical_hole_uses_equal_physical_units_on_both_axes(self):
        scene, p = self.scene_and_parameters(height=60, width=120)
        p['hole_active'][0, 0] = 1
        p['hole_center'].zero_()
        p['hole_radius'].fill_(.1)
        p['tear_raggedness'].zero_()
        p['hole_angle'].zero_()
        # Radius .1 material units corresponds to .1 UV in x and .2 UV in y.
        field = self.points([[.09, 0.], [.11, 0.], [0., .18], [0., .22]])
        self.assertEqual(scene.paper_mask(field, p).flatten().tolist(), [0., 1., 0., 1.])

    def test_straight_cut_is_a_capsule_with_known_kerf_and_round_tips(self):
        scene, p = self.scene_and_parameters()
        p['cut_active'][0, 0] = 1
        p['cut_center'].zero_()
        p['cut_length'].fill_(.4)
        p['cut_width'].fill_(.04)
        p['cut_angle'].zero_()
        p['tear_raggedness'].zero_()
        field = self.points([[0., 0.], [.19, 0.], [.21, 0.], [.23, 0.],
                             [0., .01 / scene.aspect], [0., .03 / scene.aspect]])
        self.assertEqual(scene.paper_mask(field, p).flatten().tolist(), [0., 0., 0., 1., 0., 1.])

    def test_all_four_missing_corners_have_explicit_orientation(self):
        scene, p = self.scene_and_parameters()
        p['missing_corner_active'].fill_(1)
        p['missing_corner_size'].fill_(.2)
        p['tear_raggedness'].zero_()
        corners = [[-.99, -.99], [.99, -.99], [.99, .99], [-.99, .99], [0., 0.]]
        for index in range(4):
            p['missing_corner_index'].fill_(index)
            expected = [1., 1., 1., 1., 1.]
            expected[index] = 0.
            self.assertEqual(scene.paper_mask(self.points(corners), p).flatten().tolist(), expected)

    def test_missing_material_follows_supplied_inverse_uv_not_sensor_axes(self):
        scene, p = self.scene_and_parameters()
        p['hole_active'][0, 0] = 1
        p['hole_center'].zero_()
        p['hole_radius'].fill_(.1)
        # The field itself can come from any camera or surface inverse map.
        sensor = self.points([[-.4, 0.], [0., 0.], [.4, 0.]])
        projected_uv = sensor.clone()
        projected_uv[..., 0] += .4
        self.assertEqual(scene.paper_mask(projected_uv, p).flatten().tolist(), [0., 1., 1.])
        self.assertEqual(scene.paper_mask(sensor, p).flatten().tolist(), [1., 0., 1.])

    def test_multiple_tears_remove_different_edges_without_emptying_center(self):
        scene, p = self.scene_and_parameters()
        p['tear_active'].fill_(1)
        p['tear_count'].fill_(4)
        p['tear_side'].zero_()
        p['tear_size'].fill_(.15)
        p['tear_position'].zero_()
        p['fold_position'].zero_()
        field = self.points([[.999, 0.], [-.999, 0.], [0., .999], [0., -.999], [0., 0.]])
        self.assertEqual(scene.paper_mask(field, p).flatten().tolist(), [0., 0., 0., 0., 1.])
        p['tear_count'].zero_()
        torch.testing.assert_close(scene.paper_mask(field, p), torch.ones(1, 1, 1, 5), rtol=0, atol=0)

    def test_hole_is_absent_from_shadow_caster_under_vertical_light(self):
        scene, p = self.scene_and_parameters(height=90, width=120)
        p['hole_active'][0, 0] = 1
        p['hole_center'].zero_()
        p['hole_radius'].fill_(.2)
        yy, xx = torch.meshgrid((torch.arange(90) + .5) * 2 / 90 - 1,
                                (torch.arange(120) + .5) * 2 / 120 - 1, indexing='ij')
        mask = scene.paper_mask(torch.stack((xx, yy), -1)[None], p)
        height = torch.full_like(mask, .2)
        shadow = DocumentScene._project_shadow(mask, height, torch.zeros(1, 1, 2))
        torch.testing.assert_close(shadow, mask, rtol=0, atol=0)
        self.assertEqual(float(shadow[0, 0, 45, 60]), 0.)
        self.assertEqual(float(shadow[0, 0, 10, 10]), 1.)

    def test_damage_fields_are_deterministic_and_binary(self):
        scene, p = self.scene_and_parameters()
        p['tear_active'].fill_(1)
        p['tear_count'].fill_(4)
        p['hole_active'].fill_(1)
        p['cut_active'].fill_(1)
        p['missing_corner_active'].fill_(1)
        p['edge_wear_strength'].fill_(1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 73), torch.linspace(-1, 1, 97), indexing='ij')
        field = torch.stack((xx, yy), -1)[None]
        first, second = scene.paper_mask(field, p), scene.paper_mask(field, p)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(((first == 0) | (first == 1)).all())
        self.assertGreater(float(first.mean()), .75)
        self.assertLess(float(first.mean()), .99)

    def test_static_edge_detail_keeps_single_tear_and_skips_advanced_damage(self):
        scene, p = self.scene_and_parameters()
        scene.damage_detail = 'edge'
        p['hole_active'].fill_(1)
        p['hole_center'].zero_()
        p['hole_radius'].fill_(.2)
        p['tear_active'].fill_(1)
        p['tear_count'].fill_(4)
        p['tear_side'].zero_()
        p['tear_position'].zero_()
        p['tear_size'].fill_(.15)
        field = self.points([[0., 0.], [.999, 0.], [-.999, 0.]])
        self.assertEqual(scene.paper_mask(field, p).flatten().tolist(), [1., 0., 1.])
        with self.assertRaisesRegex(ValueError, 'damage_detail'):
            DocumentScene(80, 120, damage_detail='unknown')

    def test_damage_kernel_can_be_traced_as_a_single_torch_graph(self):
        scene, p = self.scene_and_parameters()
        p['hole_active'][0, 0] = 1
        field = self.points([[0., 0.], [.8, .7]])
        compiled = torch.compile(scene.paper_mask, backend='eager', fullgraph=True)
        torch.testing.assert_close(compiled(field, p), scene.paper_mask(field, p), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
