import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ecg_gpu.digitization.cell_mesh import forward_cell_map
from ecg_gpu.digitization.complete_rectifier import fit_complete_map, render_complete


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'Optional OpenCV geometry unavailable')
class CompleteRectificationTests(unittest.TestCase):
    def fixture(self):
        row, col = np.mgrid[:12, :18]
        ids = np.column_stack((row.ravel(), col.ravel()))
        observed = ids[:, ::-1]*10.+[30, 25]
        keep = ~((ids[:, 0] >= 4) & (ids[:, 0] <= 7) & (ids[:, 1] >= 6) & (ids[:, 1] <= 11))
        return observed[keep], ids[keep]

    def test_no_grid_keeps_every_original_rgb_value(self):
        image = np.random.default_rng(13).integers(0, 256, (101, 151, 3), dtype=np.uint8)
        mapping, report = fit_complete_map(np.empty((0, 2)), np.empty((0, 2)),
                                          np.empty((0, 2)), image.shape)
        output, valid, evidence, rendered = render_complete(image, mapping)
        np.testing.assert_array_equal(output, image)
        self.assertTrue(valid.all())
        self.assertFalse(evidence.any())
        self.assertEqual(report['status'], 'original-preserved')
        self.assertFalse(rendered['physical_coordinates_available'])

    def test_affine_map_crosses_missing_grid_patch_with_correct_scale(self):
        observed, ids = self.fixture()
        mapping, report = fit_complete_map(observed, ids[:, ::-1]*5., ids, (160, 240))
        query = np.array([[95., 80.], [115., 85.], [2., 2.], [230., 155.]])
        np.testing.assert_allclose(mapping.forward(query), (query-[30, 25])/2, atol=1e-10)
        strict = forward_cell_map(observed, ids[:, ::-1]*5., ids, query[:2], None)
        self.assertTrue(np.isnan(strict).all())
        np.testing.assert_array_equal(mapping.evidence(query), [2, 2, 1, 1])
        self.assertFalse(report['grid_mask_erases_rgb'])

    def test_black_trace_inside_hole_and_header_outside_grid_survive(self):
        observed, ids = self.fixture()
        image = np.full((160, 240, 3), 255, np.uint8)
        image[78:84, 90:125] = 0
        image[4:10, 5:23] = [170, 10, 30]
        mapping, _ = fit_complete_map(observed, ids[:, ::-1]*5., ids, image.shape)
        output, _, evidence, report = render_complete(image, mapping, pixels_per_mm=2.)
        for point in (np.array([100., 80.]), np.array([10., 7.])):
            target = np.rint((point-[30, 25])/2*2-np.asarray(report['output_origin_mm'])*2).astype(int)
            np.testing.assert_array_equal(output[target[1], target[0]], image[int(point[1]), int(point[0])])
        self.assertGreater(int((evidence == 2).sum()), 0)
        self.assertGreater(int((evidence == 1).sum()), 0)

    def test_consistent_sparse_points_can_warp_without_any_complete_grid_cell(self):
        observed, ids = self.fixture()
        keep = (ids.sum(1) % 2) == 0
        mapping, _ = fit_complete_map(observed[keep], ids[keep, ::-1]*5., ids[keep], (160, 240), cells=[])
        self.assertIsNotNone(mapping)
        result = mapping.forward(np.array([[100., 80.]]))
        np.testing.assert_allclose(result, [[35., 27.5]], atol=1e-10)
        self.assertEqual(int(mapping.evidence(np.array([[100., 80.]]))[0]), 2)

    def test_independent_smooth_deformation_and_hole_roundtrip(self):
        observed, ids = self.fixture()

        def reference(points):
            x, y = points.T
            displacement = 1.3*np.sin(np.pi*x/239)*np.sin(2*np.pi*y/159)
            return (points-[30, 25])/2+np.column_stack((displacement, -.4*displacement))

        mapping, _ = fit_complete_map(observed, reference(observed), ids, (160, 240))
        query = np.random.default_rng(61).uniform([88, 65], [145, 105], (500, 2))
        inferred = mapping.forward(query)
        self.assertLess(np.quantile(np.linalg.norm(inferred-reference(query), axis=1), .95), .5)
        np.testing.assert_allclose(mapping.inverse(inferred), query, atol=1e-9)

    def test_incoherent_id_assignment_falls_back_without_erasing_input(self):
        observed, ids = self.fixture()
        shuffled = ids[np.random.default_rng(1973).permutation(len(ids))]
        mapping, report = fit_complete_map(observed, shuffled[:, ::-1]*5., shuffled, (160, 240))
        self.assertIsNone(mapping)
        self.assertEqual(report['status'], 'original-preserved')

    def test_every_target_triangle_retains_orientation_and_simple_boundary(self):
        observed, ids = self.fixture()
        canonical = ids[:, ::-1]*5.
        canonical[:, 0] += 1.2*np.sin(observed[:, 1]/17)
        mapping, report = fit_complete_map(observed, canonical, ids, (160, 240))
        self.assertIsNotNone(mapping)
        vertices = mapping.target[mapping.triangles]
        a, b = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
        areas = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        self.assertTrue((areas > 0).all() or (areas < 0).all())
        self.assertGreaterEqual(report['minimum_triangle_area_ratio'], .15)
        query = np.random.default_rng(95).uniform([1, 1], [238, 158], (4000, 2))
        np.testing.assert_allclose(mapping.inverse(mapping.forward(query)), query, atol=1e-8)

    def test_projective_pole_in_image_is_rejected(self):
        row, col = np.mgrid[:8, :10]
        ids = np.column_stack((row.ravel(), col.ravel()))
        canonical = ids[:, ::-1]*5.
        observed = canonical/(.5+.01*canonical[:, :1])
        mapping, report = fit_complete_map(observed, canonical, ids, (160, 240))
        self.assertIsNone(mapping)
        self.assertEqual(report['reason'], 'projective pole crosses the input frame')

    def test_observed_refinement_keeps_one_invertible_full_frame_mesh(self):
        observed, ids = self.fixture()
        present = {tuple(k) for k in ids}
        cells = [[r, c] for r, c in sorted(present)
                 if all(k in present for k in ((r, c+1), (r+1, c+1), (r+1, c)))]
        canonical = ids[:, ::-1]*5.
        canonical[:, 1] += .3*np.sin(observed[:, 0]/30)
        mapping, report = fit_complete_map(observed, canonical, ids, (160, 240), cells=cells)
        self.assertIn('observed vertices', report['mesh_node_layout'])
        self.assertTrue(report['boundary_self_intersections_checked'])
        query = np.random.default_rng(82).uniform([1, 1], [238, 158], (4000, 2))
        np.testing.assert_allclose(mapping.inverse(mapping.forward(query)), query, atol=1e-8)

    def test_completed_map_preserves_existing_cell_coordinates_exactly(self):
        observed, ids = self.fixture()
        present = {tuple(k) for k in ids}
        cells = [[r, c] for r, c in sorted(present)
                 if all(k in present for k in ((r, c+1), (r+1, c+1), (r+1, c)))]
        canonical = ids[:, ::-1]*5.
        canonical[:, 0] += .5*np.sin(observed[:, 1]/35)
        canonical[:, 1] += .2*np.sin(observed[:, 0]/30)
        mapping, _ = fit_complete_map(observed, canonical, ids, (160, 240), cells=cells)
        self.assertIsNotNone(mapping)
        query = np.random.default_rng(195).uniform([31, 26], [199, 134], (4000, 2))
        before = forward_cell_map(observed, canonical, ids, query, cells)
        keep = np.isfinite(before).all(1)
        np.testing.assert_allclose(mapping.forward(query[keep]), before[keep], atol=2e-10)

    def test_output_bound_keeps_original_instead_of_clipping_to_grid(self):
        observed, ids = self.fixture()
        mapping, _ = fit_complete_map(observed, ids[:, ::-1]*5., ids, (160, 240))
        image = np.random.default_rng(9).integers(0, 256, (160, 240, 3), dtype=np.uint8)
        output, valid, _, report = render_complete(image, mapping, pixels_per_mm=100., max_side=4096)
        np.testing.assert_array_equal(output, image)
        self.assertTrue(valid.all())
        self.assertEqual(report['status'], 'original-preserved')

    def test_extreme_frame_to_cell_ratio_rejects_without_unbounded_index(self):
        observed, ids = self.fixture()
        mapping, report = fit_complete_map(observed, ids[:, ::-1]*5., ids, (4000, 4000))
        self.assertIsNone(mapping)
        self.assertEqual(report['reason'], 'complete map cannot be indexed safely')

    def test_bad_coordinates_do_not_create_a_transform(self):
        observed, ids = self.fixture()
        observed[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_complete_map(observed, ids[:, ::-1]*5., ids, (160, 240))

    def test_cli_output_does_not_use_evidence_as_an_rgb_cutout(self):
        from PIL import Image
        from examples.rectify_denoising_grid import write_geometry_output
        observed, ids = self.fixture()
        image = np.full((160, 240, 3), 255, np.uint8)
        image[78:84, 90:125] = 0
        local = {'selected_indices': list(range(len(ids))), 'ids': ids.tolist(),
                 'cells': [], 'partial_status': 'rejected'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            report = write_geometry_output(image, observed, np.ones(len(ids)), local, path, pixels_per_mm=2.)
            self.assertEqual(report['status'], 'rectification-proposal')
            self.assertFalse(report['evidence_map_erases_rgb'])
            self.assertFalse(report['geometry_accuracy_verified'])
            self.assertFalse(report['sampled_map_roundtrip_checked'])
            output = np.asarray(Image.open(path/report['output_image']))
            self.assertGreater(int((output.max(-1) < 10).sum()), 100)
            np.testing.assert_array_equal(np.asarray(Image.open(path/'original-rgb.png')), image)

    def test_cli_rejection_emits_original_and_explicit_uncertainty(self):
        from PIL import Image
        from examples.rectify_denoising_grid import write_geometry_output
        image = np.random.default_rng(3).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        local = {'selected_indices': [], 'ids': [], 'cells': [], 'partial_status': 'rejected'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            report = write_geometry_output(image, np.empty((0, 2)), np.empty(0), local, path)
            self.assertEqual(report['status'], 'original-preserved')
            self.assertFalse(report['complete_geometry']['physical_coordinates_available'])
            np.testing.assert_array_equal(np.asarray(Image.open(path/report['output_image'])), image)

    def test_partial_coordinate_fallback_never_erases_visible_rgb(self):
        from PIL import Image
        from examples.rectify_denoising_grid import write_geometry_output
        row, col = np.mgrid[:8, :10]
        ids = np.column_stack((row.ravel(), col.ravel()))
        canonical = ids[:, ::-1]*5.
        observed = canonical/(.5+.01*canonical[:, :1])
        cells = [[r, c] for r in range(7) for c in range(9)]
        image = np.random.default_rng(3).integers(0, 256, (160, 240, 3), dtype=np.uint8)
        local = {'selected_indices': list(range(len(ids))), 'ids': ids.tolist(),
                 'cells': cells, 'partial_status': 'partial'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            report = write_geometry_output(image, observed, np.ones(len(ids)), local, path)
            self.assertEqual(report['status'], 'original-preserved')
            self.assertTrue(report['complete_geometry']['observed_cell_coordinates_available'])
            with np.load(path/'geometry-evidence.npz') as data:
                self.assertTrue(data['geometric_eligibility'].any())
                self.assertFalse(data['geometric_eligibility'].all())
            np.testing.assert_array_equal(np.asarray(Image.open(path/report['output_image'])), image)


if __name__ == '__main__':
    unittest.main()
