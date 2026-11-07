import unittest

from reproductions.pm_benchmark.candidate_split import split_rows


class CandidateSourceSplitChecks(unittest.TestCase):
    def test_other_recordings_of_excluded_patients_are_excluded(self):
        rows = [{'ecg_id': str(i), 'patient_id': patient, 'filename_hr': f'records500/00000/{i:05d}_hr'}
                for i, patient in [(1, '10'), (2, '10.0'), (3, '20'), (4, '20.0'), (5, '30')]]
        prepared, excluded = split_rows(rows, {'patient_ids': ['ptbxl-patient-10.0']})
        self.assertEqual(excluded, [1, 2])
        self.assertEqual([r['ecg_id'] for r in prepared], [3, 4, 5])
        self.assertEqual(prepared[0]['split'], prepared[1]['split'])
        self.assertTrue(all(r['future_images_inherit_patient_split'] for r in prepared))
        self.assertEqual(split_rows(list(reversed(rows)), {'patient_ids': ['10']})[0], prepared)

    def test_duplicate_recordings_cannot_silently_enter_another_partition(self):
        row = {'ecg_id': '1', 'patient_id': '10', 'filename_hr': 'records500/00000/00001_hr'}
        with self.assertRaises(ValueError):
            split_rows([row, dict(row, patient_id='20')], {'patient_ids': []})
        with self.assertRaises(ValueError):
            split_rows([dict(row, filename_hr='records500/00000/00002_hr')], {'patient_ids': []})


if __name__ == '__main__':
    unittest.main()
