"""Patient-disjoint selection and scoring for the sim-to-real npj benchmark."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location('sim2real_npj', ROOT / 'examples' / 'sim2real_npj.py')
sim = importlib.util.module_from_spec(_SPEC)
sys.modules['sim2real_npj'] = sim
_SPEC.loader.exec_module(sim)

LEADS = sim.LEADS


def write_wfdb(directory, stem, values_mv, fs):
    """Minimal interleaved WFDB16 record in the PTB-XL layout."""
    adc = np.rint(values_mv.T * 1000).astype('<i2')
    (directory / f'{stem}.dat').write_bytes(adc.tobytes())
    lines = [f'{stem} 12 {fs} {adc.shape[0]}']
    for column, name in enumerate(LEADS):
        checksum = int(adc[:, column].astype(np.int64).sum()) % 65536
        lines.append(f'{stem}.dat 16 1000.0(0)/mV 16 0 {int(adc[0, column])} {checksum} 0 {name}')
    (directory / f'{stem}.hea').write_text('\n'.join(lines) + '\n')
    return directory / f'{stem}.hea'


class SelectTests(unittest.TestCase):
    def test_split_is_patient_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / 'ptbxl.csv'
            rows = ['ecg_id,patient_id,strat_fold,filename_hr']
            for record in range(1, 401):
                rows.append(f'{record},{(record - 1) // 2}.0,{1 + (record - 1) % 10},records500/x/{record:05d}_hr')
            csv.write_text('\n'.join(rows) + '\n')
            available = set(range(1, 401))
            split = sim.select(csv, available, test_records=10, train_records=150, val_records=20)
        test_patients = {row['patient'] for row in split['test']}
        self.assertEqual(len(split['test']), 10)
        self.assertEqual(len(test_patients), 10)
        records = {r['record'] for r in split['train']} | {r['record'] for r in split['val']}
        self.assertFalse({str((r - 1) // 2) for r in records} & test_patients)

    def test_d0_inventory_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'files.json'
            names = [f'D0/D0_000022/D0_000022{suffix}' for suffix in ('.hea', '.dat', '_02.png', '_04.jpeg',
                                                                        '_05.jpeg', '_06.jpeg')]
            names += ['D0/D0_000107/D0_000107.hea', 'D0/D0_000107/D0_000107.dat', 'D0/D0_000107/D0_000107_06.jpeg',
                      'D1/x.png']
            path.write_text(json.dumps([{'name': name} for name in names]))
            self.assertEqual(sim.d0_records(path), {22})
            self.assertEqual(sim.d0_records(path, ['phone-photo']), {22, 107})


class ScoringTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(5000) / 500
        self.signals = np.stack([np.sin(2 * np.pi * (1 + i / 10) * t) * .5 for i in range(12)])

    def test_native_prediction_units_orientation_and_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.npz'
            values = self.signals[:, ::2] * 1000
            values[3, 100:400] = np.nan
            np.savez(path, values=values.T, unit='uV')
            prediction = sim.native_prediction(path, 5000)
        self.assertEqual(prediction.shape, (12, 5000))
        np.testing.assert_allclose(prediction[0, :4990:10], self.signals[0, :4990:10], atol=.02)
        self.assertTrue(np.isnan(prediction[3, 300:700]).all())
        self.assertTrue(np.isfinite(prediction[3, 1000:1100]).all())

    def test_score_end_to_end_ranks_better_predictions_higher(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            for index in range(4):
                header = write_wfdb(root, f'rec{index}', self.signals, 500)
                for kind in ('phone-photo', 'color-scan'):
                    image = root / f'rec{index}_{kind}.png'
                    image.write_bytes(b'x')
                    cases.append({'image_id': image.stem, 'patient_id': f'p{index}', 'image_type': kind,
                                  'image': image.name, 'reference_header': header.name})
            (root / 'manifest.json').write_text(json.dumps({'cases': cases}))
            reports = {}
            for arm, noise in (('good', .01), ('bad', .3)):
                predictions = root / arm
                predictions.mkdir()
                for case in cases[:-1]:
                    values = (self.signals + noise * rng.standard_normal(self.signals.shape)) * 1000
                    np.savez(predictions / f"{case['image_id']}.npz", values=values, unit='uV')
                reports[arm] = sim.score(root / 'manifest.json', predictions, root / f'{arm}.json', arm=arm,
                                         bootstrap=200)
        good, bad = reports['good']['summary'], reports['bad']['summary']
        self.assertGreater(good['all']['mean_snr_db'], bad['all']['mean_snr_db'] + 10)
        self.assertEqual(good['all']['images'], 8)
        self.assertEqual(good['all']['returned'], 7)
        low, high = good['all']['patient_bootstrap_95']
        self.assertLessEqual(low, high)
        printed = reports['good']['cases'][0]['metrics']['printed_samples']
        # 3x4 windows (11 leads x 1250 samples) plus the full lead II strip.
        self.assertEqual(printed, 11 * 1250 + 5000)



class ConsistencyTests(unittest.TestCase):
    def test_d0_suffixes_match_the_download_script(self):
        text = (ROOT / 'examples' / 'prepare_digitization_reference.py').read_text()
        for kind, suffix in sim.D0_KINDS.items():
            number, extension = suffix[1:].split('.')
            self.assertIn(f"'{kind}': ('{number}', '.{extension}')", text)


if __name__ == '__main__':
    unittest.main()
