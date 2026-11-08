import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from ecg_gpu.digitization.signals import LEADS
from examples.summarize_consolidated_study import signal_summary


class ConsolidatedAnalysisTests(unittest.TestCase):
    def test_rejected_geometry_and_bad_layout_stay_in_denominator(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root/'inputs'
            original.mkdir()
            np.savez(original/'source.npz', values_mv=np.zeros((12, 10)), fs=100)
            (original/'manifest.json').write_text(json.dumps({'cases':[
                {'name':'online','metadata':{}}, {'name':'offline','metadata':{}}]}))
            patient = {'source_id':'p','group_id':'g','original':'inputs','oracle':'oracle'}
            (root/'signal-inputs.json').write_text(json.dumps([patient]))
            good = {'name':'online','status':'returned','layout_matches_expected':True,
                    'evaluation':{'printed_time_coverage':.8,'mean_observed_lead_snr_db':20.}}
            bad = dict(good, name='offline', layout_matches_expected=False)
            results = [dict(patient, arm='original', cases=[good,bad]),
                       dict(patient, arm='oracle', cases=[good,good | {'name':'offline'}]),
                       dict(patient, arm='learned', cases=[])]
            (root/'signal-results.json').write_text(json.dumps(results))
            with patch('examples.summarize_consolidated_study.printed_windows', return_value={lead:[(0,.1)] for lead in LEADS}):
                result = signal_summary(root)
            totals = result['totals']
            self.assertEqual(totals['learned']['attempted'], 2)
            self.assertEqual(totals['learned']['all_case_mean_coverage'], 0.)
            self.assertEqual(totals['original']['correct_layout'], 1)
            self.assertEqual(totals['original']['all_case_mean_coverage'], .4)
            self.assertEqual(totals['original']['snr_cases'], 1)
            self.assertEqual(result['cases'][0]['original_vs_learned']['samples'], 0)


if __name__ == '__main__':
    unittest.main()
