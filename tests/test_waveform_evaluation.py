"""Sim-to-real waveform benchmark scoring (NumPy, no Torch needed)."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

# Load the module by path so the scorer can be tested without importing Torch.
_PATH = Path(__file__).resolve().parents[1] / 'ecg_gpu' / 'waveform_evaluation.py'
_SPEC = importlib.util.spec_from_file_location('waveform_evaluation', _PATH)
we = importlib.util.module_from_spec(_SPEC)
sys.modules['waveform_evaluation'] = we
_SPEC.loader.exec_module(we)


class SnrTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(1000) / 100
        self.reference = np.sin(2 * np.pi * 1.1 * t)

    def test_known_noise_gives_analytic_snr(self):
        noise = .1 * np.cos(2 * np.pi * 7 * np.arange(1000) / 100)
        expected = 10 * math.log10(np.square(self.reference).sum() / np.square(noise).sum())
        self.assertAlmostEqual(we.lead_snr_db(self.reference, self.reference + noise), expected, places=9)

    def test_perfect_estimate_is_capped_and_missing_samples_count_as_zero(self):
        self.assertEqual(we.lead_snr_db(self.reference, self.reference), we.MAX_SNR_DB)
        estimate = self.reference.copy()
        estimate[:500] = np.nan
        zeroed = estimate.copy()
        zeroed[:500] = 0
        self.assertAlmostEqual(we.lead_snr_db(self.reference, estimate), we.lead_snr_db(self.reference, zeroed))

    def test_offset_and_shift_options(self):
        offset = self.reference + .5
        self.assertLess(we.lead_snr_db(self.reference, offset), 10)
        self.assertEqual(we.lead_snr_db(self.reference, offset, remove_offset=True), we.MAX_SNR_DB)
        delayed = np.roll(self.reference, 7)
        self.assertGreater(we.lead_snr_db(self.reference, delayed, max_shift=10),
                           we.lead_snr_db(self.reference, delayed) + 20)

    def test_resample_and_record_mean(self):
        upsampled = we.resample(self.reference[None], 100, 500, 5000)
        self.assertEqual(upsampled.shape, (1, 5000))
        np.testing.assert_allclose(upsampled[0, ::5][:999], self.reference[:999], atol=1e-12)
        mean, per_lead = we.record_snr_db(np.stack([self.reference, np.zeros(1000)]),
                                          np.stack([self.reference + .1, np.zeros(1000)]))
        self.assertTrue(math.isnan(per_lead[1]))
        self.assertAlmostEqual(mean, per_lead[0])


class EvaluateTests(unittest.TestCase):
    def test_manifest_scoring_groups_and_bootstrap(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'refs').mkdir()
            (root / 'pred').mkdir()
            rows = []
            for index in range(8):
                signal = rng.standard_normal((12, 500))
                np.savez(root / 'refs' / f'{index}.npz', signals=signal, fs=100.)
                capture = 'scan' if index % 2 else 'phone_photo'
                rows.append({'id': f'img{index}', 'reference': f'refs/{index}.npz',
                             'patient': f'p{index // 2}', 'capture': capture})
                if index != 7:
                    noise = (.1 if capture == 'scan' else .5) * rng.standard_normal((12, 1000))
                    upsampled = we.resample(signal, 100, 200, 1000) + noise
                    np.savez(root / 'pred' / f'img{index}.npz', signals=upsampled, fs=200.)
            (root / 'manifest.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            report = we.evaluate(root / 'manifest.jsonl', root / 'pred', repeats=200)
        summary = report['summary']
        self.assertEqual(summary['all']['records'], 8)
        self.assertEqual(summary['scan']['missing_predictions'], 1)
        self.assertGreater(summary['phone_photo']['mean_snr_db'], 4)
        self.assertLess(summary['phone_photo']['mean_snr_db'], summary['scan']['mean_snr_db'] + 30)
        low, high = summary['all']['bootstrap_95']
        self.assertLessEqual(low, summary['all']['mean_snr_db'])
        self.assertGreaterEqual(high, summary['all']['mean_snr_db'])


if __name__ == '__main__':
    unittest.main()
