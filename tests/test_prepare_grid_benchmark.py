import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from ecg_gpu.schema_v05 import LEADS
from examples.prepare_grid_benchmark import prepare


class PrepareGridCohortTests(unittest.TestCase):
    def fixture(self, root, corrupt_identity=False):
        cache=root/'cache'
        cache.mkdir()
        metadata=root/'metadata.csv'
        with metadata.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['ecg_id','patient_id','strat_fold','filename_lr'])
            writer.writeheader()
            for index,fold in enumerate((1,9,10),1):
                stem=f'{index:05d}_lr'
                writer.writerow({'ecg_id':index,'patient_id':index,'strat_fold':fold,'filename_lr':'records100/00000/'+stem})
                np.zeros((1000,12),dtype='<i2').tofile(cache/(stem+'.dat'))
                identity='99999_lr' if corrupt_identity and index==1 else stem
                lines=[f'{identity} 12 100 1000']+[f'{stem}.dat 16 1000(0)/mV 16 0 0 0 0 {lead}' for lead in LEADS]
                (cache/(stem+'.hea')).write_text('\n'.join(lines)+'\n')
        return metadata,cache

    def test_verified_cache_requires_no_network(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            metadata,cache=self.fixture(root)
            with patch('examples.prepare_grid_benchmark.subprocess.run',side_effect=AssertionError('Network must not be used')):
                report=prepare(metadata,root/'cohort',train=1,validation=1,test=1,wfdb_cache=cache)
            self.assertEqual(report['counts'],{'train':1,'validation':1,'test':1})
            self.assertEqual(len(report['records']),3)
            with np.load(root/'cohort/signals.npz') as data:
                self.assertEqual(data['signals'].shape,(3,12,1000))

    def test_cache_cannot_silently_substitute_another_record(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            metadata,cache=self.fixture(root,corrupt_identity=True)
            with (patch('examples.prepare_grid_benchmark.subprocess.run',side_effect=AssertionError('Network must not be used')),
                  self.assertRaisesRegex(ValueError,'identity')):
                prepare(metadata,root/'cohort',train=1,validation=1,test=1,wfdb_cache=cache)

    def test_excluded_patient_normalizes_decimal_identity(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            metadata,cache=self.fixture(root)
            with (patch('examples.prepare_grid_benchmark.subprocess.run',side_effect=AssertionError('No network')),
                  self.assertRaisesRegex(ValueError,'Not enough')):
                prepare(metadata,root/'cohort',train=1,validation=1,test=1,wfdb_cache=cache,
                        excluded_patients=['1.0'])


if __name__=='__main__':
    unittest.main()
