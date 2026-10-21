"""Independent scoring, leakage and external-coordinate contracts."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from ecg_gpu.grid_benchmark import BenchmarkConfig, GridUNet, load_sources, load_external, _training_samples, _decode
from ecg_gpu.grid_evaluation import (match_grid_points, summarize_grid_matches,
                                     rectification_diagnostic, rectify_grid_image)
from ecg_gpu.schema_v05 import LEADS


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'Install the benchmark extra for evaluation')
class GridEvaluationTests(unittest.TestCase):
    def test_matching_maximizes_cardinality_before_distance(self):
        result = match_grid_points([[0., 0.], [.9, 0.]], [[.4, 0.], [-.6, 0.]], radius=.8)
        self.assertEqual(result['tp'], 2)
        np.testing.assert_allclose(sorted(result['errors_px']), [.5, .6])
        duplicate = match_grid_points([[0., 0.], [0., 0.]], [[0., 0.]])
        self.assertEqual((duplicate['tp'], duplicate['fp'], duplicate['fn']), (1, 1, 0))

    def test_empty_and_unmatched_scores_remain_explicit(self):
        empty = np.empty((0, 2))
        result = summarize_grid_matches([match_grid_points(empty, [[2., 3.]])])
        self.assertEqual(result['recall'], 0.)
        self.assertIsNone(result['precision'])
        self.assertIsNone(result['median_error_px'])
        self.assertIsNone(summarize_grid_matches([])['f1'])
        for points in ([[float('nan'), 0]], [[1]], [1, 2]):
            with self.assertRaises(ValueError):
                match_grid_points(points, empty)
        with self.assertRaises(ValueError):
            match_grid_points(empty, empty, radius=0)
        with self.assertRaises(ValueError):
            match_grid_points(np.zeros((65537, 2)), empty)

    def test_dense_page_scores_every_peak_and_uses_sparse_matching(self):
        heat = torch.zeros(1, 1, 256, 256)
        heat[..., 2::3, 2::3] = 1.
        yy, xx = np.meshgrid(np.arange(2, 256, 3), np.arange(2, 256, 3), indexing='ij')
        reference = np.stack((xx.ravel(), yy.ravel()), -1)
        predicted, truncated = _decode(heat, None, .5)
        self.assertFalse(truncated)
        self.assertGreater(len(predicted), 4096)
        result = match_grid_points(predicted, reference, radius=.5)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (len(reference), 0, 0))

    def test_matching_agrees_with_exhaustive_small_assignment_reference(self):
        import itertools
        rng = np.random.default_rng(7)
        for _ in range(12):
            predicted, reference = rng.normal(size=(4, 2)), rng.normal(size=(3, 2))
            best = (0, float('inf'))
            for assigned in itertools.product(range(-1, 3), repeat=4):
                used = [a for a in assigned if a >= 0]
                if len(set(used)) != len(used):
                    continue
                errors = [np.linalg.norm(predicted[i]-reference[a]) for i, a in enumerate(assigned) if a >= 0]
                if errors and max(errors) > 1.5:
                    continue
                score = (len(errors), sum(errors))
                if score[0] > best[0] or score[0] == best[0] and score[1] < best[1]:
                    best = score
            actual = match_grid_points(predicted, reference, radius=1.5)
            self.assertEqual(actual['tp'], best[0])
            self.assertAlmostEqual(float(actual['errors_px'].sum()), best[1])

    def test_rectification_matches_independent_affine_mapping_at_held_out_nodes(self):
        yy, xx = np.mgrid[:6, :7]
        canonical = np.stack((xx.ravel()*3, yy.ravel()*3), -1).astype(float)
        observed = canonical @ np.array([[1.1, .08], [.12, .9]]) + [4.2, 2.3]
        ids = np.stack((yy.ravel(), xx.ravel()), -1)
        result = rectification_diagnostic(observed, canonical, ids)
        self.assertTrue(result['oracle_correspondence'])
        self.assertGreater(result['supported_nodes'], 8)
        self.assertLess(result['p95_error_canonical_px'], 1e-12)
        image = torch.arange(20*24).reshape(1, 1, 20, 24).float().expand(1, 3, -1, -1)
        corners = np.array([[0., 0.], [23., 0.], [0., 19.], [23., 19.]])
        warped, support = rectify_grid_image(image, corners, corners, output_size=(20, 24))
        torch.testing.assert_close(warped, image, atol=4e-5, rtol=0)
        self.assertTrue(support.all())
        unsupported = rectification_diagnostic([[1, 1], [2, 2]], [[1, 1], [2, 2]], [[0, 0], [1, 0]])
        self.assertEqual(unsupported['supported_nodes'], 0)


class GridBenchmarkContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def source_file(self, directory, **override):
        values = dict(signals=np.random.default_rng(7).normal(size=(3, 12, 60)).astype('float32'), fs=100.,
            source_ids=['a', 'b', 'c'], group_ids=['p1', 'p2', 'p3'], splits=['train', 'validation', 'test'],
            lead_names=LEADS, units='mV')
        values.update(override)
        path = Path(directory)/'signals.npz'
        np.savez(path, **values)
        return path

    def test_leakage_is_checked_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.source_file(directory)
            records, fs = load_sources(path)
            self.assertEqual(fs, 100.)
            self.assertEqual(len(records), 3)
            for kwargs in ({'group_ids': ['p1', 'p1', 'p3']}, {'source_ids': ['a', 'a', 'c']},
                           {'units': 'V'}, {'lead_names': LEADS[::-1]}, {'splits': ['train']*3}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    load_sources(self.source_file(directory, **kwargs))
            with self.assertRaisesRegex(ValueError, 'waveform leakage'):
                load_sources(self.source_file(directory, signals=np.zeros((3, 12, 60))))

    def test_regimes_have_equal_source_view_budgets(self):
        samples = [dict(split='train', source_id=f's{i}', view=v, backend=b)
                   for i in range(3) for v in range(2) for b in ('online', 'offline')]
        selected = [_training_samples(samples, regime) for regime in ('online', 'offline', 'mixed')]
        self.assertEqual([len(s) for s in selected], [6]*3)
        self.assertEqual([sum(s['backend'] == 'online' for s in group) for group in selected], [6, 0, 3])
        self.assertTrue(all({(s['source_id'], s['view']) for s in group} == {(s['source_id'], s['view']) for s in selected[0]} for group in selected))

    def test_model_returns_full_resolution_and_bounded_offsets(self):
        model = GridUNet(width=4)
        image = torch.rand(2, 3, 65, 81)
        heat, delta = model(image)
        self.assertEqual(heat.shape, (2, 1, 65, 81))
        self.assertEqual(delta.shape, (2, 2, 65, 81))
        self.assertTrue((delta.abs() <= .5).all())
        (heat.mean()+delta.square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_external_coordinate_scaling_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (201, 99), 'white').save(root/'scan.png')
            row = {'id': 'scan', 'image': 'scan.png', 'sha256': hashlib.sha256((root/'scan.png').read_bytes()).hexdigest(),
                   'source_id': 's', 'group_id': 'p', 'points_xy': [[100., 49.]],
                   'annotation_region': [90, 40, 110, 60], 'annotation_method': 'independent test'}
            manifest = root/'external.json'
            manifest.write_text(json.dumps([row]))
            config = BenchmarkConfig(width=128, height=96, crop_size=64)
            sample = load_external(manifest, config)[0]
            # Image center stays at the resized image center, not edge-scaled.
            np.testing.assert_allclose(sample['xy'][0, 0], [63.5, 31.], atol=1e-5)
            for change in ({'sha256': 'wrong'}, {'id': '../bad'}, {'annotation_region': [100, 50, 90, 40]},
                           {'points_xy': [[float('nan'), 49.]]}):
                manifest.write_text(json.dumps([dict(row, **change)]))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_external(manifest, config)

    def test_configuration_validation(self):
        base = BenchmarkConfig()
        for change in ({'steps': 0}, {'views': True}, {'height': 63}, {'device': 'mps'}, {'seed': -1},
                       {'sigma': 0}, {'crop_size': 512}, {'offsets': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(base, **change)


if __name__ == '__main__':
    unittest.main()
