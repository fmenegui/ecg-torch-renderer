"""Output contracts for opt-in localization refinement, without NN inference."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from ecg_gpu.digitization.cell_mesh import CellMap
from ecg_gpu.digitization.complete_rectifier import fit_complete_map, render_complete
from examples import rectify_denoising_grid as cli


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'Optional OpenCV geometry unavailable')
class RefinementCLITests(unittest.TestCase):
    @staticmethod
    def fixture(noise=False):
        row, col = np.mgrid[:8, :10]
        ids = np.column_stack((row.ravel(), col.ravel()))
        points = ids[:, ::-1].astype(float)*10+[30, 25]
        if noise:
            points += np.random.default_rng(91).normal(0, .35, points.shape)
        cells = [[r, c] for r in range(7) for c in range(9)]
        local = {'selected_indices': list(range(len(ids))), 'ids': ids.tolist(),
                 'cells': cells, 'partial_status': 'partial'}
        rgb = np.full((130, 170, 3), 255, np.uint8)
        rgb[2:8, 3:20] = [170, 25, 30]
        rgb[62:65, 55:115] = 0
        return rgb, points, np.ones(len(ids)), ids, local

    def test_zero_strength_keeps_default_and_baseline_geometry(self):
        rgb, points, scores, ids, local = self.fixture()
        baseline_map, _ = fit_complete_map(points, ids[:, ::-1]*5., ids, rgb.shape, scores, local['cells'])
        expected_rgb = render_complete(rgb, baseline_map, pixels_per_mm=2.)[0]
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'default', Path(folder)/'zero'
            with mock.patch.object(cli, 'refine_grid_points', side_effect=AssertionError('refiner must stay disabled')):
                implicit = cli.write_geometry_output(rgb, points, scores, local, a, pixels_per_mm=2.)
                explicit = cli.write_geometry_output(rgb, points, scores, local, b,
                                                     pixels_per_mm=2., refinement_strength=0.)
            self.assertEqual(implicit, explicit)
            self.assertNotIn('point_refinement', implicit)
            self.assertFalse((a/'detected-grid-geometry.npz').exists())
            np.testing.assert_array_equal(np.asarray(Image.open(a/implicit['output_image'])), expected_rgb)
            with np.load(a/'observed-grid-geometry.npz') as data:
                np.testing.assert_array_equal(data['xy'], points)
            with np.load(a/'full-frame-mesh.npz') as data:
                np.testing.assert_array_equal(data['source'], baseline_map.source)
                np.testing.assert_array_equal(data['target_mm'], baseline_map.target)
                np.testing.assert_array_equal(data['triangles'], baseline_map.triangles)

    def test_refinement_archives_original_rgb_and_effective_map_coordinates(self):
        rgb, points, scores, ids, local = self.fixture(noise=True)
        original_points = points.copy()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with mock.patch.object(cli, 'refine_grid_points', wraps=cli.refine_grid_points) as spy:
                report = cli.write_geometry_output(rgb, points, scores, local, out,
                                                   pixels_per_mm=2., refinement_strength=.15)
            self.assertTrue(spy.call_args.kwargs['crease_guard'])
            self.assertTrue(report['point_refinement']['applied'])
            self.assertFalse(report['point_refinement']['fallback_to_baseline'])
            self.assertTrue(report['complete_geometry']['observed_cell_map_changed'])
            self.assertFalse(report['complete_geometry']['effective_input_cell_map_changed'])
            self.assertFalse(report['geometry_accuracy_verified'])
            np.testing.assert_array_equal(np.asarray(Image.open(out/'original-rgb.png')), rgb)
            np.testing.assert_array_equal(points, original_points)
            with np.load(out/'detected-grid-geometry.npz') as data:
                np.testing.assert_array_equal(data['xy'], original_points)
            with np.load(out/'observed-grid-geometry.npz') as data:
                effective = data['xy'].copy()
                canonical = data['canonical_mm'].copy()
            self.assertGreater(np.max(np.abs(effective-original_points)), 1e-4)
            with np.load(out/'refined-grid-candidate.npz') as data:
                np.testing.assert_array_equal(data['xy'], effective)
            with np.load(out/'full-frame-mesh.npz') as data:
                reconstructed = CellMap(data['source'], data['target_mm'], data['triangles'])
            np.testing.assert_allclose(reconstructed(effective), canonical, atol=1e-9)

    def test_out_of_frame_refined_candidate_falls_back_without_gold(self):
        rgb, points, scores, _, local = self.fixture()
        invalid = points.copy()
        invalid[0, 0] = -1.
        with tempfile.TemporaryDirectory() as folder:
            out, baseline_out = Path(folder)/'attempt', Path(folder)/'baseline'
            baseline = cli.write_geometry_output(rgb, points, scores, local, baseline_out, pixels_per_mm=2.)
            with mock.patch.object(cli, 'refine_grid_points', return_value=(invalid, {'status': 'refined'})):
                report = cli.write_geometry_output(rgb, points, scores, local, out,
                                                   pixels_per_mm=2., refinement_strength=.15)
            detail = report['point_refinement']
            self.assertFalse(detail['applied'])
            self.assertTrue(detail['fallback_to_baseline'])
            self.assertEqual(detail['fallback_count'], 1)
            self.assertIn('input image', detail['fallback_reason'])
            self.assertFalse(detail['gold_used'])
            self.assertFalse(report['complete_geometry']['observed_cell_map_changed'])
            np.testing.assert_array_equal(np.asarray(Image.open(out/'original-rgb.png')), rgb)
            np.testing.assert_array_equal(np.asarray(Image.open(out/report['output_image'])),
                                          np.asarray(Image.open(baseline_out/baseline['output_image'])))
            with np.load(out/'observed-grid-geometry.npz') as data:
                np.testing.assert_array_equal(data['xy'], points)
            with np.load(out/'refined-grid-candidate.npz') as data:
                self.assertEqual(data['xy'][0, 0], -1.)

    def test_rejected_refined_mesh_retries_the_original_observations(self):
        rgb, points, scores, _, local = self.fixture(noise=True)
        calls = []

        def controlled_fit(observed, *args, **kwargs):
            calls.append(np.asarray(observed).copy())
            if len(calls) == 1:
                return None, {'status': 'original-preserved', 'reason': 'injected incompatible frame'}
            return fit_complete_map(observed, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with mock.patch.object(cli, 'fit_complete_map', side_effect=controlled_fit):
                report = cli.write_geometry_output(rgb, points, scores, local, out,
                                                   pixels_per_mm=2., refinement_strength=.15)
            self.assertEqual(len(calls), 2)
            self.assertGreater(np.max(np.abs(calls[0]-points)), 1e-4)
            np.testing.assert_array_equal(calls[1], points)
            self.assertTrue(report['point_refinement']['fallback_to_baseline'])
            self.assertEqual(report['point_refinement']['fallback_count'], 1)
            self.assertIn('incompatible frame', report['point_refinement']['fallback_reason'])
            with np.load(out/'observed-grid-geometry.npz') as data:
                np.testing.assert_array_equal(data['xy'], points)

    def test_refiner_failure_is_an_explicit_baseline_fallback(self):
        rgb, points, scores, _, local = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with mock.patch.object(cli, 'refine_grid_points', side_effect=RuntimeError('injected sparse failure')):
                report = cli.write_geometry_output(rgb, points, scores, local, out,
                                                   pixels_per_mm=2., refinement_strength=.15)
            self.assertTrue(report['point_refinement']['fallback_to_baseline'])
            self.assertEqual(report['point_refinement']['fallback_count'], 1)
            self.assertIn('sparse failure', report['point_refinement']['fallback_reason'])
            self.assertIsNone(report['point_refinement']['candidate_data'])
            np.testing.assert_array_equal(np.asarray(Image.open(out/'original-rgb.png')), rgb)

    def test_backend_specific_candidate_fit_exception_preserves_baseline(self):
        rgb, points, scores, _, local = self.fixture(noise=True)
        calls = []

        def failing_backend(observed, *args, **kwargs):
            calls.append(np.asarray(observed).copy())
            if len(calls) == 1:
                raise ArithmeticError('injected candidate backend failure')
            return fit_complete_map(observed, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(cli, 'fit_complete_map', side_effect=failing_backend):
                report = cli.write_geometry_output(rgb, points, scores, local, Path(folder),
                                                   pixels_per_mm=2., refinement_strength=.15)
            self.assertEqual(len(calls), 2)
            np.testing.assert_array_equal(calls[1], points)
            self.assertEqual(report['point_refinement']['fallback_count'], 1)
            self.assertEqual(report['point_refinement']['candidate_fit']['status'], 'error')
            self.assertIn('ArithmeticError', report['point_refinement']['fallback_reason'])

    def test_strength_range_and_cells_conflict_fail_before_model_loading(self):
        rgb, points, scores, _, local = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'output'
            for strength in (-.1, 1.1, float('nan'), float('inf'), True):
                with self.subTest(strength=strength), self.assertRaises(ValueError):
                    cli.write_geometry_output(rgb, points, scores, local, out, refinement_strength=strength)
            with self.assertRaisesRegex(ValueError, 'full output mode'):
                cli.rectify(Path('absent.pt'), Path('absent.png'), out,
                            output_mode='cells', refinement_strength=.15)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
