import unittest

from examples.select_rectification_profile import CANDIDATES, choose


class ProfileSelectionTests(unittest.TestCase):
    def cases(self):
        records = []
        for backend in ('torch-cached', 'offline'):
            records.append({'group_id': 'validation-patient', 'split': 'validation', 'backend': backend,
                'results': {m+'-network': {'coverage_within_0_5mm': .8, 'covered_query_fraction': .8}
                            for m in CANDIDATES}})
        return records

    def test_more_output_with_large_geometry_errors_is_not_selected(self):
        cases = self.cases()
        for row in cases:
            row['results']['anchor50-network'] = {'coverage_within_0_5mm': .95, 'covered_query_fraction': 1.}
        result = choose(cases)
        self.assertEqual(result['candidate'], 'cells')
        self.assertFalse(result['validation']['anchor50']['eligible'])

    def test_coverage_gain_with_preserved_error_budget_can_be_selected(self):
        cases = self.cases()
        for row in cases:
            row['results']['anchor50-network'] = {'coverage_within_0_5mm': .9, 'covered_query_fraction': .901}
        self.assertEqual(choose(cases)['candidate'], 'anchor50')

    def test_test_cases_are_rejected_by_profile_selector(self):
        cases = self.cases()
        cases[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            choose(cases)
