"""Paired comparisons must retain failures and reject unmatched evidence."""
import copy
import unittest

from examples.digitization_dewarping_ablation import paired_summary


class PairedEvidence(unittest.TestCase):
    def report(self):
        return {'input_manifest_sha256': 'manifest', 'upstream_commit': 'commit',
                'device': 'cpu', 'precision': 'float32', 'torch': 'test',
                'cases': [{'name': 'fold', 'image_sha256': 'image',
                           'status': 'returned', 'evaluation': {
                               'mean_observed_lead_snr_db': 10.,
                               'printed_time_coverage': .9}}]}

    def test_failure_with_diagnostic_score_stays_failed_without_delta(self):
        a, b = self.report(), self.report()
        b['cases'][0].update(status='failed', error='layout rejected')
        result = paired_summary(a, b)['cases'][0]
        self.assertEqual(result['disabled']['status'], 'failed')
        self.assertIsNone(result['disabled_minus_enabled_snr_db'])

    def test_score_and_coverage_remain_separate_without_mutating_evidence(self):
        a, b = self.report(), self.report()
        b['cases'][0]['evaluation'].update(mean_observed_lead_snr_db=20., printed_time_coverage=.1)
        before = copy.deepcopy(b)
        result = paired_summary(a, b)['cases'][0]
        self.assertEqual(result['disabled_minus_enabled_snr_db'], 10.)
        self.assertEqual(result['disabled']['printed_time_coverage'], .1)
        self.assertEqual(b, before)

    def test_unmatched_inputs_software_and_cases_are_rejected(self):
        for field in ('input_manifest_sha256', 'upstream_commit', 'device', 'precision', 'torch'):
            a, b = self.report(), self.report()
            b[field] = 'different'
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired_summary(a, b)
        a, b = self.report(), self.report()
        b['cases'][0]['image_sha256'] = 'different'
        with self.assertRaises(ValueError):
            paired_summary(a, b)
        b['cases'] = []
        with self.assertRaises(ValueError):
            paired_summary(a, b)


if __name__ == '__main__':
    unittest.main()
