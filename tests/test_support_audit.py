import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ecg_gpu.digitization.support_audit import audit_support


class SupportAuditTests(unittest.TestCase):
    def test_dropping_bad_prediction_looks_better_without_recovering_anything(self):
        y = np.ones((12, 4))
        a = y * 4
        b = np.full_like(y, np.nan)
        r = audit_support(y, a, b)
        self.assertLess(r['candidate_nrmse_zero_missing'], r['direct_nrmse_zero_missing'])
        self.assertEqual(r['candidate_coverage'], 0)
        self.assertFalse(r['candidate_preserves_all_direct_samples'])
        self.assertFalse(r['single_scalar_superiority_claim_supported'])
        self.assertIsNone(r['common_support'])
        self.assertAlmostEqual(r['partitions']['dropped']['delta_nrmse_contribution'], -2.)

    def test_exact_additive_decomposition_with_all_four_partitions(self):
        y = np.tile([1., 2., 3., 4., np.nan], (12, 1))
        a = np.tile([2., 5., np.nan, np.nan, 88.], (12, 1))
        b = np.tile([1.5, np.nan, 3.2, np.nan, -88.], (12, 1))
        before = [v.copy() for v in (y, a, b)]
        r = audit_support(y, a, b)
        parts = r['partitions']
        self.assertEqual({v['samples'] for v in parts.values()}, {12})
        self.assertAlmostEqual(sum(v['delta_nrmse_contribution'] for v in parts.values()), r['delta_nrmse'])
        self.assertAlmostEqual(sum(v['delta_mse_mv2'] for v in parts.values()),
            r['candidate_all_printed_rmse_mv'] ** 2 - r['direct_all_printed_rmse_mv'] ** 2)
        self.assertEqual(r['common_support']['direct_rmse_mv'], 1.)
        self.assertEqual(r['common_support']['candidate_rmse_mv'], .5)
        for old, new in zip(before, (y, a, b)):
            np.testing.assert_array_equal(old, new)

    def test_perfect_pair_has_zero_contributions(self):
        y = np.ones((12, 8))
        r = audit_support(y, y, y)
        self.assertEqual(r['delta_nrmse'], 0.)
        self.assertEqual(sum(v['delta_nrmse_contribution'] for v in r['partitions'].values()), 0.)
        self.assertTrue(r['candidate_preserves_all_direct_samples'])

    def test_flat_reference_has_defined_rmse_but_no_nrmse(self):
        y = np.zeros((12, 4))
        r = audit_support(y, y, y + 1)
        self.assertIsNone(r['delta_nrmse'])
        self.assertEqual(r['candidate_all_printed_rmse_mv'], 1.)

    def test_invalid_arrays(self):
        y = np.ones((12, 4))
        for args in [(y[0], y, y), (y, y[:, :3], y), (y, y * np.inf, y), (y * np.nan, y, y)]:
            with self.assertRaises(ValueError):
                audit_support(*args)

    def test_saved_summary_always_exposes_support_loss(self):
        from ecg_gpu.digitization.external_evaluation import score_external_signals
        from examples.summarize_external_digitization import summarize
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            y = np.ones((12, 4))
            np.savez(root / 'reference.npz', values_mv=y)
            records = []
            for arm, values in [('npj_native', y * 4), ('candidate', np.full_like(y, np.nan))]:
                np.savez(root / (arm + '.npz'), values_mv=values, fs=100)
                records.append({'image_id': 'one', 'arm': arm, 'status': 'returned',
                    'prediction': arm + '.npz', 'predicted_layout': 'standard_12x1',
                    'inference_wall_seconds': 0, 'evaluation': score_external_signals(y, values, 100)})
            (root / 'manifest.json').write_text(json.dumps({'images': 1, 'patients': 1,
                'cases': [{'image_id': 'one', 'patient_id': 'patient', 'recording_id': 1,
                    'reference': 'reference.npz', 'condition': 'scan', 'reference_layout_scoring_only': '12x1'}]}))
            (root / 'report.json').write_text(json.dumps({'status': 'completed', 'cpu_max_side': 100,
                'arms': ['npj_native', 'candidate'], 'cases': records}))
            result = summarize(root / 'manifest.json', root, root / 'summary')
            audit = result['paired_support_audits']['candidate minus npj_native']
            self.assertEqual(audit['patient_macro']['candidate_coverage']['value'], 0)
            self.assertEqual(audit['patient_macro']['candidate_lost_direct_samples_in_images'], 1)
            self.assertEqual(audit['patient_macro']['dropped_partition_fraction_of_total_nrmse_reduction'], 1.)
            self.assertIsNone(audit['per_image'][0]['common_support'])


if __name__ == '__main__':
    unittest.main()
