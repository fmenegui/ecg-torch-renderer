import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproductions.common import sha256, write_json
from reproductions.pm_benchmark.score_clock import score
from reproductions.pm_benchmark.worker_clock import ARMS, ROOT


class ClockScoringContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.catalog, self.saved = self.folder / 'catalog', self.folder / 'saved'
        self.catalog.mkdir()
        self.saved.mkdir()
        protocol = json.loads((ROOT / 'clock-inputs.json').read_text())
        np.savez_compressed(self.catalog / 'reference.npz', values_mv=np.ones((12, 5000)), fs=500)
        for arm, value in zip(ARMS, [4., np.nan]):
            np.savez_compressed(self.saved / (arm + '.npz'), values_mv=np.full((12, 5000), value), fs=500)
        cases, images, records, exams = [], [], [], []
        for index, case in enumerate(protocol['exams']):
            cases.append(dict(case, reference='reference.npz', reference_sha256=sha256(self.catalog / 'reference.npz'),
                patient_id=str(index // 2), condition=case['exam_id'].split('-', 2)[2], source_format_scoring_only='test'))
            for page in case['pages']:
                image_hash = page['image_sha256'] or format(len(images), '064x')
                images.append({key: page[key] for key in ['image_id', 'image']} | {'image_sha256': image_hash})
                for arm in ARMS:
                    records.append({'image_id': page['image_id'], 'source_id': image_hash, 'arm': arm,
                        'prediction': arm + '.npz', 'prediction_sha256': sha256(self.saved / (arm + '.npz'))})
            for arm in ARMS:
                exams.append({'exam_id': case['exam_id'], 'arm': arm, 'prediction': arm + '.npz',
                    'prediction_sha256': sha256(self.saved / (arm + '.npz'))})
        write_json(self.catalog / 'catalog.json', {'cases': cases, 'metadata_sha256': protocol['metadata_sha256'],
                    'archive_index_sha256': protocol['zip_index_sha256']})
        write_json(self.saved / 'sealed-image-only-manifest.json', images)
        self.seal = {'reference_used': False, 'native_observer_parity_bit_exact': True, 'records': records, 'exams': exams}
        self.receipt = {'status': 'completed_inference', 'reference_inputs_received': False,
            'reference_scoring_performed': False, 'new_training': False,
            'protocol_sha256': sha256(ROOT / 'clock-inputs.json')}
        self.reseal()

    def reseal(self):
        write_json(self.saved / 'sealed-predictions.json', self.seal)
        self.receipt['prediction_seal_sha256'] = sha256(self.saved / 'sealed-predictions.json')
        write_json(self.saved / 'receipt.json', self.receipt)

    def run_score(self):
        return score(self.catalog, self.saved, catalog_sha256=sha256(self.catalog / 'catalog.json'))

    def test_lower_error_from_dropped_samples_is_explicit(self):
        report = self.run_score()
        audit = report['patient_macro_support_audit']
        self.assertEqual(report['scheduled_exams'], 12)
        self.assertEqual(report['scheduled_pages'], 22)
        self.assertEqual(audit['direct_coverage']['value'], 1.)
        self.assertEqual(audit['candidate_coverage']['value'], 0.)
        self.assertEqual(audit['dropped_partition_fraction_of_total_nrmse_reduction'], 1.)

    def test_missing_page_or_exam_is_rejected_after_valid_reseal(self):
        for key in ['records', 'exams']:
            original = self.seal[key].copy()
            self.seal[key].pop()
            self.reseal()
            with self.assertRaises(ValueError):
                self.run_score()
            self.seal[key] = original

    def test_prediction_and_image_identity_tampering_are_rejected(self):
        self.seal['records'][0]['source_id'] = 'another-image'
        self.reseal()
        with self.assertRaises(ValueError):
            self.run_score()
        self.seal['records'][0]['source_id'] = json.loads((self.saved / 'sealed-image-only-manifest.json').read_text())[0]['image_sha256']
        self.reseal()
        (self.saved / (ARMS[0] + '.npz')).write_bytes(b'changed after sealing')
        with self.assertRaises(ValueError):
            self.run_score()

    def test_valid_seal_cannot_hide_inconsistent_exam_assembly(self):
        np.savez_compressed(self.saved / 'wrong-assembly.npz', values_mv=np.full((12, 5000), 4.5), fs=500)
        self.seal['exams'][0].update(prediction='wrong-assembly.npz', prediction_sha256=sha256(self.saved / 'wrong-assembly.npz'))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'assembly'):
            self.run_score()


if __name__ == '__main__':
    unittest.main()
