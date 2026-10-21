"""Independent geometry, signal, data-integrity and failure-accounting checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ecg_gpu.digitization.artifacts import _artifact_path, fetch_assets, sha256_file, verify_assets
from ecg_gpu.digitization.data import load_manifest, read_wfdb16, sequential_reference
from ecg_gpu.digitization.geometry import compose_grid_to_source, sample_source
from ecg_gpu.digitization.signals import LEADS, DigitizationResult, LeadSignal, decode_waveform_rows, split_rows


def make_record(root, *, fs=8):
    root = Path(root)
    length = 10*fs
    adc = np.stack([np.arange(length, dtype=np.int16)+i*20 for i in range(12)], axis=1)
    adc.astype('<i2').tofile(root/'record.dat')
    lines = [f'record 12 {fs} {length}']
    for col, name in enumerate(LEADS):
        checksum = int(adc[:, col].astype(np.int64).sum()) % 65536
        lines.append(f'record.dat 16 1000.0(10)/mV 16 10 {adc[0, col]} {checksum} 0 {name}')
    (root/'record.hea').write_text('\n'.join(lines)+'\n')
    Image.new('RGB', (8, 8), 'white').save(root/'image.png')
    row = {'image_id': 'case-0', 'patient_id': 'patient-1', 'split': 'test',
        'image_type': 'scan', 'image': 'image.png', 'image_sha256': sha256_file(root/'image.png'),
        'fs': fs, 'reference_header': 'record.hea',
        'reference_header_sha256': sha256_file(root/'record.hea'),
        'reference_dat_sha256': sha256_file(root/'record.dat'),
        'source': {'url': 'https://example.org/research-fixture', 'license': 'test fixture'}}
    return row, adc


class SignalContracts(unittest.TestCase):
    def test_signal_is_finite_immutable_and_owns_its_samples(self):
        a = np.array([1., 2., 3.])
        lead = LeadSignal(a, 500, 2.5)
        a[:] = 0
        np.testing.assert_array_equal(lead.values, [1, 2, 3])
        self.assertFalse(lead.values.flags.writeable)
        for samples, fs, start in (([], 500, 0), ([np.nan], 500, 0), ([1], 0, 0),
                                   ([1], True, 0), ([1], 500, -1)):
            with self.subTest(samples=samples, fs=fs, start=start), self.assertRaises(ValueError):
                LeadSignal(samples, fs, start)

    def test_parabolic_refinement_recovers_fractional_peak_and_mv(self):
        y = np.arange(20)[:, None]
        center = 8.25+np.linspace(0, .1, 13)[None]
        p = np.clip(.9-.03*(y-center)**2, 0, 1)
        p = np.repeat(p[None], 4, axis=0)
        values, observed = decode_waveform_rows(p, zero_mv=[12]*4, pixels_per_mv=2, length=13)
        np.testing.assert_allclose(values, np.repeat(((12-center)/2), 4, axis=0), atol=1e-6)
        self.assertTrue(observed.all())

    def test_gap_filling_preserves_missingness_and_rejects_absent_trace(self):
        p = np.zeros((4, 12, 10))
        p[:, 5, :] = .9
        p[:, :, 4:6] = 0
        values, observed = decode_waveform_rows(p, zero_mv=[5]*4, pixels_per_mv=1, length=10)
        np.testing.assert_allclose(values, 0, atol=1e-6)
        self.assertFalse(observed[:, 4:6].any())
        self.assertTrue(observed[:, :4].all())
        p[2] = 0
        with self.assertRaisesRegex(ValueError, 'row 2'):
            decode_waveform_rows(p, zero_mv=[5]*4, pixels_per_mv=1, length=10)

    def test_sparse_observations_and_fourier_length_are_explicit(self):
        p = np.zeros((4, 12, 9))
        p[:, 5, [2, 7]] = .9
        values, observed = decode_waveform_rows(p, zero_mv=[6]*4, pixels_per_mv=2, length=100)
        self.assertEqual(values.shape, (4, 100))
        np.testing.assert_allclose(values, .5, atol=1e-6)
        self.assertEqual(int(observed.sum()), 8)

    def test_bad_waveform_inputs_fail(self):
        p = np.full((4, 10, 8), .5)
        variants = [(p[:3], [0]*4, 1, 8), (p*np.nan, [0]*4, 1, 8),
                    (p+1, [0]*4, 1, 8), (p, [0]*3, 1, 8), (p, [0]*4, 0, 8),
                    (p, [0]*4, 1, 1)]
        for a, z, scale, length in variants:
            with self.subTest(shape=a.shape, scale=scale, length=length), self.assertRaises(ValueError):
                decode_waveform_rows(a, zero_mv=z, pixels_per_mv=scale, length=length)

    def test_sequential_lead_identity_times_and_long_ii(self):
        rows = np.zeros((4, 100))
        for row in range(3):
            for col in range(4):
                rows[row, 25*col:25*(col+1)] = row*10+col
        rows[3] = 100
        leads = split_rows(rows, 10, average_ii=False, einthoven=False)
        self.assertEqual(set(leads), set(LEADS))
        for name, expected, start in [('I', 0, 0), ('aVL', 11, 2.5),
                                      ('V3', 22, 5), ('V6', 23, 7.5), ('II', 100, 0)]:
            np.testing.assert_array_equal(leads[name].values, expected)
            self.assertEqual(leads[name].start_s, start)
        self.assertEqual(len(leads['II'].values), 100)
        self.assertEqual(len(leads['V6'].values), 25)

    def test_einthoven_only_changes_simultaneous_limb_segment(self):
        rows = np.zeros((4, 100))
        rows[0, :25] = 1
        rows[2, :25] = 2
        rows[1, :25] = 4
        rows[3] = 6
        lead = split_rows(rows, 10)
        np.testing.assert_allclose(lead['I'].values+lead['III'].values, lead['II'].values[:25])
        np.testing.assert_array_equal(lead['II'].values[25:], 6)
        np.testing.assert_array_equal(lead['aVR'].values, 0)

    def test_nondivisible_sample_count_and_result_contract(self):
        leads = split_rows(np.zeros((4, 10250)), 1025, einthoven=False)
        self.assertEqual(sum(len(leads[n].values) for n in ('I', 'aVR', 'V1', 'V4')), 10250)
        DigitizationResult(leads)
        with self.assertRaises(ValueError):
            DigitizationResult({'II': leads['II']})
        with self.assertRaises(ValueError):
            split_rows(np.zeros((4, 100)), 500)


class GeometryContracts(unittest.TestCase):
    def test_identity_preserves_edges_and_pixel_centers(self):
        y, x = torch.meshgrid(torch.arange(7), torch.arange(11), indexing='ij')
        rgb = torch.stack([x, y, x+2*y], -1).float()
        grid = torch.tensor([[[0., 0.], [10., 0.]], [[0., 6.], [10., 6.]]])
        xy = compose_grid_to_source(grid, torch.eye(3), output_size=(7, 11))
        warped, valid = sample_source(rgb, xy)
        torch.testing.assert_close(warped, rgb, atol=2e-6, rtol=0)
        self.assertTrue(valid.all())

    def test_fractional_translation_matches_analytic_ramp(self):
        y, x = torch.meshgrid(torch.arange(8), torch.arange(12), indexing='ij')
        rgb = torch.stack([x, y, 3*x+y], -1).float()
        grid = torch.tensor([[[1., 1.], [10., 1.]], [[1., 6.], [10., 6.]]])
        h = torch.tensor([[1., 0., .25], [0., 1., .5], [0., 0., 1.]])
        xy = compose_grid_to_source(grid, h, output_size=(6, 10))
        warped, valid = sample_source(rgb, xy)
        torch.testing.assert_close(warped[..., 0], xy[..., 0], atol=2e-6, rtol=0)
        torch.testing.assert_close(warped[..., 2], 3*xy[..., 0]+xy[..., 1], atol=5e-6, rtol=0)
        self.assertTrue(valid.all())

    def test_projective_inversion_happens_after_dense_interpolation(self):
        grid = torch.tensor([[[0., 0.], [10., 0.]], [[0., 6.], [10., 6.]]], dtype=torch.float64)
        h = torch.tensor([[1., 0., 0.], [0., 1., 0.], [.02, 0., 1.]], dtype=torch.float64)
        xy = compose_grid_to_source(grid, h, output_size=(3, 3))
        # Solve u=x/(1+.02*x) at the center u=5, v=3 independently.
        np.testing.assert_allclose(xy[1, 1], [5/(1-.02*5), 3/(1-.02*5)], atol=1e-12)

    def test_border_fill_does_not_count_as_observed_geometry(self):
        image = torch.ones(4, 5, 3)
        xy = torch.tensor([[[-.1, 2.], [4., 3.], [5., 3.]]])
        _, valid = sample_source(image, xy)
        self.assertEqual(valid.tolist(), [[False, True, False]])
        with self.assertRaises(ValueError):
            compose_grid_to_source(torch.zeros(2, 2, 2), torch.zeros(3, 3), output_size=(3, 3))
        with self.assertRaises(ValueError):
            sample_source(image, xy*float('nan'))

    def test_projective_horizon_invalid_shapes_and_precision(self):
        grid = torch.tensor([[[0., 0.], [1., 0.]], [[0., 1.], [1., 1.]]])
        horizon = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 0., 1.]])
        with self.assertRaisesRegex(ValueError, 'infinity'):
            compose_grid_to_source(grid, horizon, output_size=(3, 3))
        for g, h, size in ((grid[:1], torch.eye(3), (3, 3)),
                           (grid, torch.zeros(2, 2), (3, 3)),
                           (grid, torch.eye(3), (1, 3)),
                           (grid, torch.eye(3), (3., 3))):
            with self.assertRaises(ValueError):
                compose_grid_to_source(g, h, output_size=size)
        self.assertEqual(compose_grid_to_source(grid.half(), torch.eye(3), output_size=(3, 3)).dtype,
                         torch.float32)
        with self.assertRaises(ValueError):
            sample_source(torch.full((3, 4, 3), float('nan')), grid)


class DataContracts(unittest.TestCase):
    def test_wfdb_calibration_checksum_and_sequential_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, adc = make_record(root)
            full = read_wfdb16(root/'record.hea')
            np.testing.assert_allclose(full['V6'].values, (adc[:, 11]-10)/1000)
            short = sequential_reference(full)
            np.testing.assert_array_equal(short['aVR'].values, full['aVR'].values[20:40])
            self.assertEqual(short['aVR'].start_s, 2.5)
            self.assertEqual(len(short['II'].values), 80)
            corrupt = bytearray((root/'record.dat').read_bytes())
            corrupt[0] ^= 1
            (root/'record.dat').write_bytes(corrupt)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_wfdb16(root/'record.hea')

    def test_manifest_hashes_duplicates_and_patient_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, _ = make_record(root)
            path = root/'manifest.json'
            def write(rows):
                path.write_text(json.dumps({'schema_version': 1, 'cases': rows}))
            write([row])
            self.assertEqual(len(load_manifest(path)), 1)
            for variant in ([row, row], [row, dict(row, image_id='case-1', split='train')],
                            [dict(row, image_sha256='0'*64)], [dict(row, image='../image.png')],
                            [dict(row, reference_dat_sha256='0'*64)], [dict(row, source={})]):
                write(variant)
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_reject_unsupported_wfdb_units_and_nonfinite_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_record(root)
            path = root/'record.hea'
            path.write_text(path.read_text().replace('/mV', '/uV'))
            with self.assertRaises(ValueError):
                read_wfdb16(path)


class ArtifactContracts(unittest.TestCase):
    def test_artifact_paths_cannot_escape_root(self):
        for name in ('../a', '/tmp/a'):
            with self.assertRaises(ValueError):
                _artifact_path('/tmp/reference', name)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'outside').symlink_to('/tmp', target_is_directory=True)
            with self.assertRaises(ValueError):
                _artifact_path(root, 'outside/a')

    def test_existing_artifact_mismatch_and_missing_are_actionable(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = {'artifacts': [{'path': 'a.py', 'size': 1, 'sha256': '0'*64}]}
            with patch('ecg_gpu.digitization.artifacts.artifact_lock', return_value=lock):
                with self.assertRaises(FileNotFoundError):
                    verify_assets(folder)
                Path(folder, 'a.py').write_bytes(b'x')
                with self.assertRaises(ValueError):
                    fetch_assets(folder)

    def test_download_is_hash_verified_and_failure_is_atomic(self):
        import io
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = b'reviewed code'
            lock = {'artifacts': [{'path': 'a.py', 'size': len(payload),
                'sha256': hashlib.sha256(payload).hexdigest(), 'url': 'https://example.org/a'}]}
            with patch('ecg_gpu.digitization.artifacts.artifact_lock', return_value=lock), \
                 patch('urllib.request.urlopen', return_value=io.BytesIO(payload)):
                fetch_assets(root)
            self.assertEqual((root/'a.py').read_bytes(), payload)
            (root/'a.py').unlink()
            with patch('ecg_gpu.digitization.artifacts.artifact_lock', return_value=lock), \
                 patch('urllib.request.urlopen', return_value=io.BytesIO(b'wrong')):
                with self.assertRaises(ValueError):
                    fetch_assets(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_external_loader_selects_definitions_without_file_side_effects(self):
        from ecg_gpu.digitization.reference import _definitions
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'external.py'
            p.write_text("raise RuntimeError('must not execute')\ndef selected():\n    return 17\n")
            result = _definitions(p, ('selected',), {})
            self.assertEqual(result['selected'](), 17)
            with self.assertRaises(ValueError):
                _definitions(p, ('missing',), {})

    def test_checkpoint_loading_is_strict_and_sets_inference_mode(self):
        from ecg_gpu.digitization.reference import _load_weights
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'weights.pth'
            net = torch.nn.Linear(3, 2)
            torch.save({'state_dict': {'module.'+k: v for k, v in net.state_dict().items()}}, p)
            loaded = _load_weights(torch.nn.Linear(3, 2), p)
            self.assertFalse(loaded.training)
            self.assertEqual(loaded.output_type, ['infer'])
            torch.testing.assert_close(loaded.weight, net.weight)
            torch.save({'state_dict': {'unexpected': torch.ones(1)}}, p)
            with self.assertRaises(RuntimeError):
                _load_weights(net, p)


def local_assets(test):
    import os
    assets = Path(os.environ.get('ECG_DIGITIZATION_ASSETS', 'build/reference-assets'))
    if not (assets/'official_metric.py').is_file():
        test.skipTest('Locked external metric unavailable, tested in digitization integration job')
    return assets


class OfficialMetricContracts(unittest.TestCase):
    def setUp(self):
        from ecg_gpu.digitization.evaluation import OfficialMetric
        self.metric = OfficialMetric(local_assets(self))

    def test_metric_keeps_raw_bias_despite_alignment(self):
        x = np.sin(np.arange(100)*.3)
        rows = np.stack([x]*4)
        truth = split_rows(rows, 10, einthoven=False)
        pred = {n: LeadSignal(v.values+2, v.fs, v.start_s) for n, v in truth.items()}
        result = self.metric.image(truth, pred)
        self.assertAlmostEqual(result['official_snr_db'], 60)
        self.assertAlmostEqual(result['per_lead']['I']['raw_bias_mv'], 2)
        self.assertAlmostEqual(result['per_lead']['I']['raw_rmse_mv'], 2)

    def test_aggregation_uses_linear_ratios_and_exact_official_batch_score(self):
        x = np.sin(np.arange(100)*.17)
        truth = split_rows(np.stack([x]*4), 10, einthoven=False)
        predictions = [{n: LeadSignal(v.values*.8, v.fs, v.start_s) for n, v in truth.items()},
                       {n: LeadSignal(v.values*.4, v.fs, v.start_s) for n, v in truth.items()}]
        results, frames = [], []
        for i, pred in enumerate(predictions):
            results.append(self.metric.image(truth, pred, image_id=f'image{i}'))
            frames.append(self.metric._frames(truth, pred, f'image{i}'))
        pd = self.metric.pd
        official = self.metric.functions['score'](pd.concat([f[0] for f in frames]),
                                                   pd.concat([f[1] for f in frames]), 'id')
        self.assertAlmostEqual(self.metric.aggregate(results), official, places=10)
        self.assertNotAlmostEqual(self.metric.aggregate(results),
                                   np.mean([r['official_snr_db'] for r in results]), places=2)

    def test_mismatched_lengths_and_windows_are_not_silently_aligned(self):
        truth = split_rows(np.ones((4, 100)), 10, einthoven=False)
        for replacement in (LeadSignal(np.zeros(26), 10), LeadSignal(np.zeros(25), 10, 1)):
            pred = dict(truth, I=replacement)
            with self.assertRaises(ValueError):
                self.metric.image(truth, pred)

    def test_runner_accounts_for_failure_and_never_saves_zero_success(self):
        from ecg_gpu.digitization.runner import run_manifest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, _ = make_record(root)
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'cases': [row, dict(row, image_id='case-1')]}))
            truth = sequential_reference(read_wfdb16(root/'record.hea'))
            class Backend:
                count = 0
                def predict(self, image, *, fs, progress=None):
                    self.count += 1
                    if self.count == 2:
                        raise ValueError('deliberate geometry failure')
                    return DigitizationResult(truth), {}
            report = run_manifest(manifest, root/'out', Backend(), self.metric)
            self.assertEqual(report['status'], 'completed_with_failures')
            self.assertEqual(report['successes'], 1)
            self.assertEqual(report['failures'], 1)
            self.assertEqual(report['scored_cases'], 2)
            self.assertTrue(report['cases'][1]['evaluation']['failure_penalty'])
            self.assertFalse((root/'out/predictions/case-1.npz').exists())
            self.assertTrue((root/'out/predictions/case-0.npz').is_file())
            self.assertLess(report['official_snr_db'], 60)
            with self.assertRaises(FileExistsError):
                run_manifest(manifest, root/'out', Backend(), self.metric)


if __name__ == '__main__':
    unittest.main()
