import json
import unittest
from copy import deepcopy
from pathlib import Path

from reproductions.pm_benchmark.worker_clock import validate_inputs
from reproductions.runpod.controller import validate_plan

ROOT = Path(__file__).resolve().parents[1] / 'pm_benchmark'


class ClockWorkerContracts(unittest.TestCase):
    def test_frozen_inputs_cover_only_the_declared_pages(self):
        document = json.loads((ROOT / 'clock-inputs.json').read_text())
        cases = validate_inputs(document)
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(len(c['pages']) for c in cases), 22)

    def test_reference_fields_and_changed_clock_are_rejected(self):
        original = json.loads((ROOT / 'clock-inputs.json').read_text())
        mutants = []
        for target in ['exam', 'page', 'top']:
            document = deepcopy(original)
            if target == 'exam':
                document['exams'][0]['reference'] = 'gold.npz'
            elif target == 'page':
                document['exams'][0]['pages'][0]['true_layout'] = '6x2'
            else:
                document['references'] = []
            mutants.append(document)
        document = deepcopy(original)
        document['public_axes']['fs'] = 1000
        mutants.append(document)
        document = deepcopy(original)
        document['new_training'] = True
        mutants.append(document)
        for document in mutants:
            with self.assertRaises(ValueError):
                validate_inputs(document)

    def test_independent_cost_cap_and_no_training_are_enforced(self):
        original = json.loads((ROOT / 'clock-plan.json').read_text())
        validate_plan(original)
        for key, value in [('maximum_gpu_price_per_hour', .61), ('maximum_pod_seconds', 1801),
                           ('gpu_time_cost_cap_usd', 3), ('new_training', True),
                           ('scheduled_pages', 6600), ('volume_gb', 30)]:
            with self.assertRaises(ValueError):
                validate_plan(dict(original, **{key: value}))


if __name__ == '__main__':
    unittest.main()
