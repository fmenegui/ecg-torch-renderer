"""Independent label identity, physical reference and grouped-export checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ecg_gpu import (ECGImageLayer, ECGSupervisionLayer, RenderConfig, SupervisionConfig,
                     export_paired_dataset, paired_samples_from_render, lead_name_detection_targets)
from ecg_gpu.gpu_renderer import _conditional_visibility
from ecg_gpu.paired_cli import generate_dataset
from ecg_gpu.schema_v05 import LEADS


class PairedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = RenderConfig(height=192, width=256, profile='clean', antialias_samples=1,
                                  surface_mode='off', shadow_samples=0)
        cls.parameters = {'layout': '3x4+r1', 'lead_present': 1., 'label_present': 1.,
                          'font_size': 8., 'label_below': False, 'quarter': 0, 'angle': 0.,
                          'occlusion_active': 0., 'pen_active': 0., 'artifact_present': 0.}
        t = torch.arange(1000) / 100
        cls.signals = torch.stack([t[None].expand(12, -1) * .02 + i * .03 for i in range(3)])

    def layer(self, **options):
        return ECGImageLayer(fs=100, config=self.config, parameters=self.parameters,
                             photo=False, randomize=False, **options)

    def render(self, signals=None, **options):
        return self.layer().render_batch(self.signals if signals is None else signals, stage='all',
            return_instances=True, return_baselines=False, return_name_instances=True,
            return_field=True, **options)

    def records(self):
        return paired_samples_from_render(self.render(), source_ids=['a', 'b', 'c'],
            group_ids=['patient-a', 'patient-b', 'patient-c'], sample_ids=['a', 'b', 'c'])

    def test_name_targets_leave_rgb_and_existing_targets_identical(self):
        layer = self.layer()
        a = layer.render_batch(self.signals, stage='all', return_instances=True)
        b = layer.render_batch(self.signals, stage='all', return_instances=True, return_name_instances=True)
        self.assertTrue(torch.equal(a.image, b.image))
        for key in a.targets:
            self.assertTrue(torch.equal(a.targets[key], b.targets[key]), key)
        self.assertEqual(b.targets['name_instances'].shape, (3, 15, 192, 256))
        grouped = torch.zeros_like(b.targets['labels'])
        grouped.scatter_reduce_(1, b.metadata['lead_id'][..., None, None].expand_as(b.targets['name_instances']),
                                b.targets['name_instances'], reduce='amax', include_self=True)
        self.assertTrue(torch.equal(grouped, b.targets['labels']))

    def test_names_repeated_strips_and_cabrera_identity_without_host_reads(self):
        layer = ECGSupervisionLayer(fs=100, config=self.config, parameters=self.parameters,
            photo=False, randomize=False, supervision=SupervisionConfig(outputs=('name_masks', 'name_boxes')))
        with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
             mock.patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('numpy')), \
             mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar read')):
            batch = layer(self.signals, parameters={'layout': '3x4-cabrera+r3'})
            targets = lead_name_detection_targets(batch)
        self.assertEqual(targets['cls'].shape, (45, 1))
        self.assertEqual(batch['name_lead_ids'][0, 12:].tolist(), [1, 6, 10])
        inverted = batch['name_amplitude_sign'] < 0
        self.assertTrue(inverted.any())
        self.assertTrue((batch['name_lead_ids'][inverted] == LEADS.index('aVR')).all())

    def test_missing_names_and_opaque_cover_are_not_detections(self):
        layer = ECGSupervisionLayer(fs=100, config=self.config, parameters=self.parameters,
            photo=False, randomize=False, supervision=SupervisionConfig(outputs=('name_masks', 'name_boxes')))
        for params in ({'label_present': 0.}, {'occlusion_active': 1., 'occlusion_x': 0.,
                       'occlusion_y': 0., 'occlusion_w': 1., 'occlusion_h': 1.}):
            batch = layer(self.signals, parameters=params)
            self.assertEqual(lead_name_detection_targets(batch)['bboxes'].shape, (0, 4))
            self.assertFalse(batch['name_masks'].any())

    def test_name_geometry_uses_the_same_photo_field_and_visibility(self):
        layer = self.layer()
        base = layer.render_batch(self.signals, stage='all', return_name_instances=True)
        photo = layer.render_batch(self.signals, stage='all', photo=True, return_name_instances=True, return_instances=True,
            return_field=True, parameters={'photo': 1., 'angle': .12, 'zoom': .87, 'perspective_x': .06})
        expected = F.grid_sample(base.targets['name_instances'], photo.inverse_field, align_corners=False)
        torch.testing.assert_close(photo.targets['name_instances'], expected, atol=2e-6, rtol=2e-6)
        visible = _conditional_visibility(photo.targets['valid'], photo.targets['frame_coverage'])
        records = paired_samples_from_render(photo, source_ids=['a', 'b', 'c'],
            group_ids=['a', 'b', 'c'], sample_ids=['a', 'b', 'c'])
        torch.testing.assert_close(records[0]['name_masks'], expected[:1] * visible[:1])

    def test_three_tasks_share_patient_split_and_immutable_signal(self):
        records = self.records()
        view = dict(records[0], sample_id='another-view-a')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'paired'
            report = export_paired_dataset([*records, view], root)
            self.assertEqual(set(report['groups'].values()), {'train', 'val', 'test'})
            pages = {row['sample_id']: row for row in report['samples']}
            self.assertEqual(pages['a']['split'], pages['another-view-a']['split'])
            for index, name in enumerate(('a', 'b', 'c')):
                page = root / 'digitization' / pages[name]['split'] / name
                with np.load(page / 'signal.npz') as saved:
                    np.testing.assert_array_equal(saved['signals'], self.signals[index].numpy())
                    self.assertEqual(float(saved['fs']), 100.)
                    self.assertEqual(saved['lead_names'].tolist(), LEADS)
                with np.load(page / 'geometry.npz') as saved:
                    self.assertEqual(saved['inverse_field'].shape, (192, 256, 2))
                self.assertEqual((page / 'image.png').read_bytes(), (root / pages[name]['detection_image']).read_bytes())
            again = export_paired_dataset(reversed(records), Path(folder) / 'reversed')
            self.assertEqual(report['groups'], again['groups'])

    def test_crop_pixels_clean_mask_binary_bmp_and_file_hashes(self):
        records = self.records()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'data'
            report = export_paired_dataset(records, root)
            crop = report['crops'][0]
            x0, y0, x1, y1 = crop['crop_box_xyxy']
            record = next(r for r in records if r['sample_id'] == crop['sample_id'])
            paths = {Path(p).name: root / p for p in crop['files']}
            expected = record['instance_masks'][0, crop['slot'], y0:y1, x0:x1]
            np.testing.assert_array_equal(np.asarray(Image.open(paths['mask.png'])), (expected * 255).round().byte().numpy())
            np.testing.assert_array_equal(np.asarray(Image.open(paths['mask.bmp'])).astype(bool), (expected > .05).numpy())
            rgb = (record['image'][0, :, y0:y1, x0:x1].permute(1, 2, 0) * 255).round().byte().numpy()
            np.testing.assert_array_equal(np.asarray(Image.open(paths['image.png'])), rgb)
            for entry in report['crops']:
                for filename, expected_hash in entry['files'].items():
                    self.assertEqual(hashlib.sha256((root / filename).read_bytes()).hexdigest(), expected_hash)

    def test_json_uses_original_time_and_cabrera_polarity(self):
        rendered = self.render(parameters={'layout': '3x4-cabrera+r1'})
        records = paired_samples_from_render(rendered, source_ids=['a', 'b', 'c'],
            group_ids=['a', 'b', 'c'], sample_ids=['a', 'b', 'c'])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'data'
            report = export_paired_dataset(records, root)
            for crop in report['crops']:
                if crop['sample_id'] != 'a':
                    continue
                signal_path = next(root / p for p in crop['files'] if p.endswith('signal.json'))
                payload = json.loads(signal_path.read_text())
                indices = np.asarray(payload['source_indices'])
                lead = payload['lead_id']
                np.testing.assert_array_equal(payload['signal_mv'], self.signals[0, lead, indices].tolist())
                np.testing.assert_allclose(payload['time_s'], indices / 100, atol=0, rtol=0)
                sign = payload['printed_amplitude_sign']
                np.testing.assert_allclose(payload['printed_signal_mv'], sign * np.asarray(payload['signal_mv']), atol=0, rtol=0)
                if payload['window_s'][0] > 0:
                    self.assertGreater(payload['time_s'][0], 0.)
                if sign < 0:
                    self.assertEqual(payload['lead_name'], 'aVR')

    def test_joint_detection_has_13_classes_and_repeated_name_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'data'
            report = export_paired_dataset(self.records(), root)
            self.assertEqual(json.loads((root / 'detection/dataset.yaml').read_text())['nc'], 13)
            classes = [int(row.split()[0]) for row in (root / report['samples'][0]['detection_labels']).read_text().splitlines()]
            self.assertEqual(classes.count(0), 13)
            self.assertEqual(classes.count(2), 2)
            self.assertEqual(set(classes), set(range(13)))

    def test_overlap_keeps_target_only_and_counts_foreign_instances(self):
        layer = self.layer()
        layer.renderer.row_table.zero_()
        layer.renderer.col_table.zero_()
        source = self.signals[:1].expand(1, 12, -1)
        rendered = layer.render_batch(source, stage='trace', return_instances=True, return_name_instances=True)
        records = paired_samples_from_render(rendered, source_ids=['a'], group_ids=['a'], sample_ids=['a'])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'data'
            report = export_paired_dataset(records, root, val_fraction=0, test_fraction=0)
            self.assertEqual(report['counts']['with_overlap'], report['counts']['crops'])
            self.assertTrue(all(c['foreign_instance_slots'] and c['crossing_pixels'] > 0 for c in report['crops']))
            self.assertTrue(all(c['slot'] not in c['foreign_instance_slots'] for c in report['crops']))

    def test_invisible_and_inactive_instances_are_explicitly_omitted(self):
        records = paired_samples_from_render(self.render(parameters={'occlusion_active': 1., 'occlusion_x': 0.,
            'occlusion_y': 0., 'occlusion_w': 1., 'occlusion_h': 1.}), source_ids=['a', 'b', 'c'],
            group_ids=['a', 'b', 'c'], sample_ids=['a', 'b', 'c'])
        with tempfile.TemporaryDirectory() as folder:
            report = export_paired_dataset(records, Path(folder) / 'data')
            self.assertEqual(report['counts']['crops'], 0)
            self.assertEqual(len(report['omitted_instances']), 45)

    def test_bad_records_and_leakage_fail_before_output_creation(self):
        records = self.records()
        bad_cases = [([records[0], dict(records[0], sample_id='same-source', group_id='other')], 'Source leakage'),
                     ([dict(records[0], sample_id='../escape')], 'safe filename'),
                     ([records[0]], 'independent groups')]
        bad = copy.copy(records[0])
        bad['image'] = bad['image'].clone()
        bad['image'][0, 0, 0, 0] = float('nan')
        bad_cases.append(([bad], 'finite'))
        bad = copy.copy(records[0])
        bad['instance_windows_s'] = bad['instance_windows_s'].clone()
        bad['instance_windows_s'][0, 0, 1] = 1000
        bad_cases.append(([bad], 'within the original'))
        for samples, message in bad_cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / 'absent'
                with self.assertRaisesRegex(ValueError, message):
                    export_paired_dataset(samples, root)
                self.assertFalse(root.exists())

    def test_nonempty_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                export_paired_dataset(self.records(), root)
            self.assertEqual((root / 'keep.txt').read_text(), 'keep')

    def test_streaming_plan_checks_missing_duplicate_and_changed_captures_atomically(self):
        records = self.records()
        plan = [{key: row[key] for key in ('source_id', 'group_id', 'sample_id', 'provenance')} for row in records]
        variants = [records[:-1], [*records, records[0]],
                    [records[0], dict(records[1], source_id='different'), records[2]]]
        for sequence in variants:
            with self.subTest(count=len(sequence)), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / 'data'
                with self.assertRaisesRegex(ValueError, 'source_plan'):
                    export_paired_dataset(iter(sequence), out, source_plan=plan)
                self.assertFalse(out.exists())
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_streaming_plan_matches_in_memory_export(self):
        records = self.records()
        plan = [{key: row[key] for key in ('source_id', 'group_id', 'sample_id', 'provenance')} for row in records]
        with tempfile.TemporaryDirectory() as folder:
            a = export_paired_dataset(records, Path(folder) / 'a')
            b = export_paired_dataset(iter(records), Path(folder) / 'b', source_plan=plan)
            self.assertEqual(a, b)

    def test_manifest_cli_reorders_declared_leads_and_keeps_original_units(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            values = np.stack([np.full(200, i * 10, dtype=np.float32) for i in range(12)])
            np.savez(root / 'source.npz', signals=values[::-1], fs=100, units='uV', lead_names=LEADS[::-1])
            (root / 'sources.json').write_text(json.dumps({'records': [
                {'input': 'source.npz', 'source_id': 'record-a', 'group_id': 'patient-a'}]}))
            report = generate_dataset(root / 'sources.json', root / 'output', val_fraction=0, test_fraction=0,
                                      width=128, height=96, layouts=('3x4',))
            self.assertEqual(report['counts']['pages'], 1)
            with np.load(root / 'output/digitization/train/record-00000-view-000/signal.npz') as saved:
                np.testing.assert_array_equal(saved['signals'], values * np.float32(.001))

    def test_export_does_not_round_float64_reference_samples(self):
        signals = self.signals[:1].double() + 1.23456789e-9
        records = paired_samples_from_render(self.render(signals), source_ids=['a'], group_ids=['a'], sample_ids=['a'])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'data'
            export_paired_dataset(records, root, val_fraction=0, test_fraction=0)
            with np.load(root / 'digitization/train/a/signal.npz') as saved:
                self.assertEqual(saved['signals'].dtype, np.dtype('float64'))
                np.testing.assert_array_equal(saved['signals'], signals[0].numpy())


if __name__ == '__main__':
    unittest.main()
