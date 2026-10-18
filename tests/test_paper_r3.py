"""Material-coordinate checks for the third-round finite crease model."""
import math
import unittest

import torch

from ecg_gpu import ECGRenderer, RenderConfig


class BranchedPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def renderer(self):
        return ECGRenderer(RenderConfig(height=128, width=192, profile='photo',
                                        surface_resolution=96, shadow_samples=0))

    def isolated_crease(self, renderer):
        p = renderer.sample_parameters(1, generator=torch.Generator().manual_seed(971))
        for name, value in {'surface_mode': 4, 'surface_strength': 1., 'surface_axis': 0.,
                            'fold_angle': 0., 'fold_position': .4, 'fold_amplitude': .05,
                            'fold_sign': 1., 'fold_span': .3, 'fold_width': .03,
                            'fold_length': .7}.items():
            p[name].fill_(value)
        p['fold_position'][:, 7] = -.6
        p['fold_angle'][:, 1] = math.pi / 2
        return p

    def test_single_crease_is_off_center_and_branch_meets_displaced_junction(self):
        r = self.renderer()
        p = self.isolated_crease(r)
        surface = r.document.surface_model
        outputs = []
        for count in (0, 1, 2):
            p['crease_count'].fill_(count)
            outputs.append(surface._geometry(p, torch.zeros(1, 1, 9, 13))[0])
        # Independently locate the maximum incremental height, then compare
        # with the prescribed material-space junction. No warp or renderer
        # target is used as the geometric oracle.
        expected = ((.296, -.324), (.296, -.324 + .7 * .23))
        for index, point in enumerate(expected, 1):
            height = outputs[index][:, 2] - outputs[index - 1][:, 2]
            location = height.flatten().argmax()
            actual = surface.material[:, :, :, :].reshape(2, -1)[:, location]
            torch.testing.assert_close(actual, torch.tensor(point), rtol=0, atol=.025)
            self.assertGreater(float(height.max() - height.min()), .01)

    def test_accordion_has_unequal_material_panel_lengths(self):
        r = self.renderer()
        p = self.isolated_crease(r)
        p['surface_mode'].fill_(3)
        p['surface_frequency'].fill_(4.5)
        p['surface_phase'].fill_(.47)
        mesh, _, _ = r.document.surface_model._geometry(p, torch.zeros(1, 1, 9, 13))
        row = mesh[0, :, mesh.shape[-2] // 2]
        edges = row[:, 1:] - row[:, :-1]
        tangent = torch.atan2(edges[2], edges[0])
        curvature = tangent[1:] - tangent[:-1]
        turning = torch.where(curvature[1:] * curvature[:-1] < 0)[0]
        spacing = turning[1:] - turning[:-1]
        self.assertGreaterEqual(len(spacing), 5)
        self.assertGreaterEqual(int(spacing.max() - spacing.min()), 3)
        # Direct edge lengths are independent of the returned metric helper.
        edge_error = (edges.square().sum(0).sqrt() / (2 / row.shape[-1]) - 1).abs()
        self.assertLess(float(edge_error.mean()), .006)

    def test_noise_does_not_displace_flat_or_developable_surfaces(self):
        r = self.renderer()
        p = r.sample_parameters(8, generator=torch.Generator().manual_seed(971))
        p['surface_mode'] = torch.arange(8)
        p['surface_strength'].fill_(1.)
        a = r.document.surface_model._geometry(p, torch.zeros(8, 1, 9, 13))[0]
        b = r.document.surface_model._geometry(p, torch.ones(8, 1, 9, 13))[0]
        for index in (0, 1, 2, 3, 6, 7):
            torch.testing.assert_close(a[index], b[index], rtol=0, atol=0)
        self.assertGreater(float((a[4] - b[4]).abs().max()), 0.)

    def test_multi_seed_strain_against_independent_edge_and_diagonal_lengths(self):
        r = self.renderer()
        p = r.sample_parameters(64, generator=torch.Generator().manual_seed(971))
        p['surface_mode'] = torch.arange(64) % 8
        p['surface_strength'].fill_(1.)
        mesh, normals, _ = r.document.surface_model(p, torch.Generator().manual_seed(31))
        self.assertTrue(bool(torch.isfinite(mesh).all()))
        self.assertTrue(bool(torch.isfinite(normals).all()))
        # Direct neighboring distances independently constrain material
        # stretch and shear, including the junctions between finite ridges.
        dx, dy = 2 / mesh.shape[-1], (2 * r.document.aspect) / mesh.shape[-2]
        horizontal = (mesh[..., 1:] - mesh[..., :-1]).norm(dim=1) / dx - 1
        vertical = (mesh[..., 1:, :] - mesh[..., :-1, :]).norm(dim=1) / dy - 1
        diagonal = (mesh[..., 1:, 1:] - mesh[..., :-1, :-1]).norm(dim=1) / math.hypot(dx, dy) - 1
        for error in (horizontal, vertical, diagonal):
            self.assertLess(float(error.abs().mean((1, 2)).max()), .006)
            self.assertLess(float(torch.quantile(error.abs().flatten(), .99)), .025)


if __name__ == '__main__':
    unittest.main()
