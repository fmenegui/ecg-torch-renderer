import csv
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ecg_gpu.digitization.selective import SignalProposal
from ecg_gpu.digitization.signals import LEADS, LeadSignal
from reproductions.pm_benchmark.catalog import (
    build,
    canonical_reference,
    group_pages,
    infer_printed_windows,
    recording_id,
)
from reproductions.pm_benchmark.contracts import join_pages, validate_training_patients


class CompletePmCatalogTests(unittest.TestCase):
    def setUp(self):
        self.full = np.random.default_rng(17).normal(0, .1, (12, 5000))

    def test_multipage_windows_are_found_from_source_not_guessed_by_columns(self):
        starts = [0] * 3 + [2500] * 3 + [0] * 3 + [2500] * 3
        short = np.stack([self.full[i, start:start + 2500] for i, start in enumerate(starts)], axis=1) + .13
        inferred, checks = infer_printed_windows(short, self.full)
        self.assertEqual(inferred, starts)
        self.assertTrue(all(c['maximum_residual_after_constant_dc_mv'] < 1e-12 for c in checks))
        self.assertTrue(all(abs(c['pm_minus_ptb_dc_mv'] - .13) < 1e-12 for c in checks))

    def test_noise_variants_retain_their_own_signal_and_rhythm(self):
        starts = [0] * 3 + [1250] * 3 + [2500] * 3 + [3750] * 3
        noisy = self.full + np.random.default_rng(8).normal(0, .03, self.full.shape)
        short = np.stack([noisy[i, start:start + 1250] for i, start in enumerate(starts)], axis=1)
        result, names = canonical_reference(short, noisy[1, :, None], starts)
        self.assertEqual(names, ['II'])
        np.testing.assert_array_equal(result[1], noisy[1])
        np.testing.assert_array_equal(result[6, 2500:3750], noisy[6, 2500:3750])
        self.assertTrue(np.isnan(result[6, :2500]).all())

    def test_ambiguous_constant_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            infer_printed_windows(np.ones((1250, 12)), np.ones((12, 5000)))

    def test_original_record_identity_under_noise_aliases(self):
        self.assertEqual(recording_id('high_freq_noise_low_LPAE_00123_hr'), 123)
        for name in ['../LPAE_00123_hr', 'LPAE_123_hr', 'unknown']:
            with self.assertRaises(ValueError):
                recording_id(name)

    def test_page_union_does_not_overwrite_conflicts_or_zero_fill(self):
        a, b = np.full((12, 10), np.nan), np.full((12, 10), np.nan)
        a[0] = 1
        b[6] = 2
        b[0, 5] = 7
        result = join_pages([SignalProposal(a, 500, 'page0', None), SignalProposal(b, 500, 'page1', None)],
                            exam_source_id='exam', fs=500, samples=10)
        self.assertTrue(np.isnan(result.values_mv[0, 5]))
        np.testing.assert_array_equal(result.values_mv[6], b[6])
        self.assertTrue(np.isnan(result.values_mv[2]).all())
        self.assertEqual(result.metadata['conflicting_samples'], 1)

    def test_other_recording_of_benchmark_patient_is_excluded_from_training(self):
        for patient in ['97', '97.0', 'ptbxl-patient-97.0']:
            with self.assertRaises(ValueError):
                validate_training_patients([{'patient_id': patient, 'recording_id': 99999}], {'patient_ids': ['ptbxl-patient-97.0']})
        self.assertTrue(validate_training_patients([{'patient_id': 98}], {'patient_ids': [97]})['passed'])

    def test_full_builder_keeps_exams_pages_variants_and_patients_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'final_data/data').mkdir(parents=True)
            originals = root / 'originals'
            originals.mkdir()
            ptb = root / 'ptb.csv'
            ptb.write_text('ecg_id,patient_id\n1,7.0\n2,7.0\n')
            rows, bank, rhythm_bank = [], {}, {}
            starts = {1: [0] * 3 + [2500] * 3 + [0] * 3 + [2500] * 3, 2: [0] * 12}
            for identity in [1, 2]:
                for suffix in ['.hea', '.dat']:
                    (originals / (f'{identity:05d}_hr' + suffix)).write_text('fixture')
                for condition, prefix in [('photos_scans', ''), ('digital_data_high_freq_noise_low', 'high_freq_noise_low_')]:
                    source = prefix + f'LPAE_{identity:05d}_hr'
                    n = 2500 if identity == 1 else 5000
                    full = self.full + (.02 if prefix else 0)
                    bank[source] = np.stack([full[i, s:s + n] for i, s in enumerate(starts[identity])], axis=1)
                    rhythm_bank[source] = np.empty((5000, 0))
                    count = 2 if identity == 1 else 4
                    for page in range(count):
                        path = f'{condition}/img_{identity}_page_{page}.jpeg'
                        image = root / 'final_data/visual_data' / path
                        image.parent.mkdir(parents=True, exist_ok=True)
                        image.write_bytes(b'independent image fixture')
                        rows.append({'Image ID': str(identity), 'ECG ID': source, 'Image relative path': path,
                            'ECG number of pages': str(count), 'Image page': str(page), 'ECG number of columns per page': '2' if identity == 1 else '1',
                            'ECG number of rows per page': '3', 'ECG number of rhythm leads': '0', 'ECG format': '3x4'})
            metadata = root / 'final_data/metadata.csv'
            with metadata.open('w') as stream:
                writer = csv.DictWriter(stream, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.savez(root / 'final_data/data/leads.npz', **bank)
            np.savez(root / 'final_data/data/rhythms.npz', **rhythm_bank)
            index = [{'name': str(p.relative_to(root)), 'size': p.stat().st_size, 'crc': zlib.crc32(p.read_bytes())}
                     for p in (root / 'final_data').rglob('*') if p.is_file()]
            (root / 'zip-index.json').write_text(json.dumps(index))
            native = {name: LeadSignal(self.full[i], 500) for i, name in enumerate(LEADS)}
            with patch('reproductions.pm_benchmark.catalog.read_wfdb16', return_value=native):
                result = build(root, ptb, originals, root / 'catalog', require_complete=False)
            self.assertEqual((result['exams'], result['page_files'], result['patients'], result['reference_variants']), (4, 12, 1, 4))
            catalog = json.loads((root / 'catalog/catalog.json').read_text())
            self.assertEqual(len(catalog['cases']), 4)
            broken = [dict(r) for r in rows if r['Image page'] != '1']
            with self.assertRaises(ValueError):
                group_pages(broken)


if __name__ == '__main__':
    unittest.main()
