"""Offline mechanics, shell topology, primary-ray bank and provenance checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from ecg_gpu import ECGRenderer, RenderConfig, DocumentAssets
from ecg_gpu.pathtrace import (
    PathTraceConfig,
    paper_mesh,
    pathtrace_capture,
    capture_geometry_assets,
)
from ecg_gpu.paper_mechanics import (
    SheetConfig,
    relax_sheet,
    paper_shell,
    assert_no_crossings,
    lift_cut_lips,
)


class SheetMechanicsTests(unittest.TestCase):
    def mesh(self, surface="flat"):
        return paper_mesh(0.2, 0.15, surface, 0.5, columns=9, rows=7)

    def test_free_fall_matches_discrete_analytical_gravity(self):
        vertices, uv, _ = self.mesh()
        vertices[:, 2] += 1
        c = SheetConfig(steps=5, iterations=4, damping=0.0, time_step_s=0.01)
        output, report = relax_sheet(
            vertices, uv, rows=7, columns=9, width_m=0.2, height_m=0.15, config=c
        )
        expected_drop = c.gravity_m_s2 * c.time_step_s**2 * c.steps * (c.steps + 1) / 2
        np.testing.assert_allclose(
            output[:, 2], vertices[:, 2] - expected_drop, atol=1e-7
        )
        np.testing.assert_allclose(output[:, :2], vertices[:, :2], atol=1e-8)
        self.assertLess(report["max_edge_strain"], 1e-5)

    def test_table_contact_keeps_sheet_above_table_without_uv_change(self):
        vertices, uv, faces = self.mesh()
        reference_uv = uv.copy()
        c = SheetConfig(steps=8, iterations=8)
        output, report = relax_sheet(
            vertices, uv, rows=7, columns=9, width_m=0.2, height_m=0.15, config=c
        )
        self.assertGreaterEqual(float(output[:, 2].min()), c.table_clearance_m - 1e-10)
        np.testing.assert_array_equal(uv, reference_uv)
        self.assertLess(report["max_edge_strain"], 0.001)
        assert_no_crossings(output, faces)

    def test_fold_rest_shape_memory_is_not_flattened_without_force(self):
        vertices, uv, _ = self.mesh("fold")
        c = SheetConfig(steps=2, iterations=4, gravity_m_s2=0.0, table_clearance_m=0.0)
        output, _ = relax_sheet(
            vertices, uv, rows=7, columns=9, width_m=0.2, height_m=0.15, config=c
        )
        np.testing.assert_allclose(output, vertices, atol=1e-7)

    def test_unconverged_strain_is_rejected(self):
        vertices, uv, _ = self.mesh()
        vertices[20, 0] += 0.06
        with self.assertRaises(RuntimeError):
            relax_sheet(
                vertices,
                uv,
                rows=7,
                columns=9,
                width_m=0.2,
                height_m=0.15,
                config=SheetConfig(steps=1, iterations=1, max_edge_strain=0.001),
            )

    def test_self_crossing_and_coplanar_overlap_are_rejected(self):
        a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
        for b in (
            np.array([[0.2, 0.2, -1], [0.2, 0.2, 1], [0.8, 0.2, 0]]),
            np.array([[0.2, 0.2, 0], [0.8, 0.2, 0], [0.2, 0.8, 0]]),
        ):
            with self.assertRaises(RuntimeError):
                assert_no_crossings(
                    np.concatenate([a, b]), np.array([[0, 1, 2], [3, 4, 5]])
                )
        assert_no_crossings(
            np.concatenate([a, a + [2, 0, 0]]), np.array([[0, 1, 2], [3, 4, 5]])
        )

    def test_shell_is_closed_and_matches_analytical_slab_volume(self):
        v, uv, f = self.mesh()
        thickness = 0.00006
        shell = paper_shell(v, uv, f, thickness)
        vertices = shell["side_vertices"]
        faces = np.concatenate([f, shell["back_faces"] + len(v), shell["side_faces"]])
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)
        t = vertices[faces].astype(np.float64)
        volume = np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) / 6
        self.assertAlmostEqual(volume, 0.2 * 0.15 * thickness, delta=1e-12)
        np.testing.assert_array_equal(vertices[: len(v)], v)
        self.assertEqual(len(shell["boundary_edges"]), 2 * (9 - 1) + 2 * (7 - 1))

    def test_shell_closes_removed_material_boundaries(self):
        v, uv, f = self.mesh()
        center = uv[f].mean(1)
        f = f[np.linalg.norm(center - [0.5, 0.5], axis=1) > 0.15]
        shell = paper_shell(v, uv, f, 0.00005)
        self.assertGreater(len(shell["boundary_edges"]), 28)
        self.assertEqual(len(shell["side_faces"]), 2 * len(shell["boundary_edges"]))

    def test_lifted_cut_lip_changes_shared_vertices_and_preserves_uv(self):
        v, uv, f = self.mesh()
        original = uv.copy()
        center = uv[f].mean(1)
        f = f[np.linalg.norm(center - [0.5, 0.5], axis=1) > 0.15]
        lifted = lift_cut_lips(v, uv, f, width_m=0.2, height_m=0.15, lift_m=0.0002)
        self.assertGreater(float((lifted[:, 2] - v[:, 2]).max()), 0.0001)
        np.testing.assert_array_equal(uv, original)
        np.testing.assert_array_equal(lifted[:, :2], v[:, :2])
        with self.assertRaises(ValueError):
            lift_cut_lips(
                *self.mesh()[:2],
                self.mesh()[2],
                width_m=0.2,
                height_m=0.15,
                lift_m=0.0002,
            )

    def test_invalid_mechanics_cost_and_thickness_are_rejected(self):
        for factory in (
            lambda: SheetConfig(steps=True),
            lambda: SheetConfig(damping=1.0),
            lambda: SheetConfig(time_step_s=0.0),
            lambda: PathTraceConfig(sheet=SheetConfig()),
            lambda: PathTraceConfig(paper_thickness_m=-1.0),
            lambda: PathTraceConfig(edge_fibers=1),
            lambda: PathTraceConfig(mesh_columns=2),
        ):
            with self.assertRaises(ValueError):
                factory()


@unittest.skipUnless(
    importlib.util.find_spec("mitsuba"), "Optional Mitsuba unavailable"
)
class BakedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.renderer = ECGRenderer(
            RenderConfig(width=192, height=128, profile="clean", surface_mode="off")
        )
        cls.canonical = cls.renderer(
            torch.zeros(1, 12, 1000),
            500,
            photo=False,
            as_texture=True,
            stage="all",
            return_instances=True,
            generator=torch.Generator().manual_seed(9),
        )

    def config(self, **kw):
        return PathTraceConfig(
            width=96,
            height=64,
            samples_per_pixel=1,
            label_samples_per_axis=1,
            mesh_columns=17,
            mesh_rows=13,
            camera_origin=(0.0, -0.015, 0.15),
            **kw,
        )

    def test_bank_uv_mask_matches_independent_capture_targets_and_roundtrips(self):
        capture = pathtrace_capture(
            self.canonical, config=self.config(), return_geometry=True
        )
        bank = capture_geometry_assets(capture)
        expected = (
            F.grid_sample(
                self.canonical.targets["traces"] * self.canonical.targets["valid"],
                bank.inverse_field,
                padding_mode="border",
                align_corners=False,
            )
            * bank.mask
        )
        torch.testing.assert_close(expected, capture["lead_masks"])
        self.assertTrue(torch.equal(bank.mask, capture["paper_mask"]))
        self.assertGreater(float(bank.mask.sum()), 100.0)
        torch.testing.assert_close(
            bank.normals.square().sum(1), torch.ones(1, 64, 96), atol=1e-5, rtol=0
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.npz"
            bank.to_npz(path)
            restored = DocumentAssets.from_npz(path)
            self.assertEqual(restored.provenance, bank.provenance)
            self.assertTrue(torch.equal(restored.inverse_field, bank.inverse_field))
            renderer = ECGRenderer(
                RenderConfig(width=96, height=64, surface_mode="off"),
                document_assets=restored,
            )
            self.assertEqual(
                renderer.document.asset_provenance["sampling"], "center-ray"
            )

    def test_mechanics_thickness_fibers_and_occluder_use_actual_rays(self):
        config = self.config(
            surface="fold",
            paper_thickness_m=0.00006,
            edge_fibers=8,
            sheet=SheetConfig(steps=3, iterations=16),
            occluder=True,
        )
        capture = pathtrace_capture(self.canonical, config=config, return_geometry=True)
        self.assertTrue(torch.isfinite(capture["image"]).all())
        self.assertEqual(capture["metadata"]["edge_fibers"], 8)
        self.assertLess(capture["metadata"]["mechanics"]["max_edge_strain"], 0.03)
        self.assertTrue((capture["lead_masks"] <= capture["paper_mask"] + 1e-6).all())
        self.assertTrue((capture["geometry"]["height"] >= 0).all())

    def test_geometry_export_is_explicit_and_thick_sheet_table_penetration_fails(self):
        with self.assertRaises(ValueError):
            capture_geometry_assets({"geometry": None})
        with self.assertRaises(ValueError):
            pathtrace_capture(
                self.canonical,
                config=self.config(surface="flat", paper_thickness_m=0.001),
            )


if __name__ == "__main__":
    unittest.main()
