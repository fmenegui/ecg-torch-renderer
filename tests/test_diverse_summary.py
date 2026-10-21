import unittest

from examples.summarize_diverse_study import paired_delta


class PatientSummaryTests(unittest.TestCase):
    def cases(self, repeat):
        left, right = [], []
        for patient in ('a', 'b'):
            for view in range(repeat):
                a = {'tp': 10 if patient == 'a' else 0, 'fp': 0, 'fn': 0 if patient == 'a' else 10}
                b = {'tp': 0 if patient == 'a' else 10, 'fp': 0, 'fn': 10 if patient == 'a' else 0}
                left.append({'key': patient+str(view), 'group_id': patient, 'points': a})
                right.append({'key': patient+str(view), 'group_id': patient, 'points': b})
        return left, right

    def test_repeated_views_do_not_create_additional_independent_patients(self):
        a, b = self.cases(1)
        x, y = self.cases(8)
        first, repeated = paired_delta(a, b, draws=200), paired_delta(x, y, draws=200)
        self.assertEqual(first['patients'], 2)
        self.assertEqual(repeated['patients'], 2)
        self.assertEqual(first['ci95'], repeated['ci95'])
        self.assertEqual(first['delta_f1_right_minus_left'], repeated['delta_f1_right_minus_left'])

    def test_case_mismatch_is_not_silently_dropped(self):
        a, b = self.cases(2)
        with self.assertRaisesRegex(ValueError, 'matched'):
            paired_delta(a, b[:-1])


if __name__ == '__main__':
    unittest.main()
