"""Independent adverse-scene target checks, separate from production helpers.

These checks establish geometric/export contracts, not photographic realism.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, capture_scenario
from ecg_gpu.dataset_export import export_yolo_dataset, samples_from_batch
from ecg_gpu.training import detection_targets


class RoundTwoQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        time = torch.arange(1001) / 100
        cls.signals = torch.stack([
            (.06 + lead * .008) * torch.sin(time * (4 + lead * .1))
            for lead in range(12)
        ])[None]

    def assert_tight_boxes(self, masks, boxes, valid, threshold=.05):
        # Brute-force coordinates independently test the production reduction.
        for channel in range(masks.shape[1]):
            coordinates = torch.nonzero(masks[0, channel] > threshold)
            if not len(coordinates):
                self.assertFalse(valid[0, channel])
                self.assertEqual(boxes[0, channel].tolist(), [-1] * 4)
                continue
            self.assertTrue(valid[0, channel])
            ys, xs = coordinates.unbind(1)
            expected = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            self.assertEqual(boxes[0, channel].tolist(), expected)

    def test_boxes_tightly_enclose_visible_traces_in_adverse_capture(self):
        source_copy = self.signals.clone()
        for scenario in ('phone_photo', 'folded_photo', 'damaged_archive'):
            with self.subTest(scenario=scenario):
                config, parameters = capture_scenario(scenario, height=96, width=128, severity=.9)
                parameters['layout'] = '3x4-cabrera+r3'
                layer = ECGSupervisionLayer(fs=100, config=config, parameters=parameters,
                    randomize=False, seed=219, supervision=SupervisionConfig(outputs=(
                        'lead_masks', 'lead_boxes', 'instance_masks', 'instance_boxes')))
                batch = layer(self.signals)
                self.assert_tight_boxes(batch['lead_masks'], batch['lead_boxes'], batch['lead_box_valid'])
                self.assert_tight_boxes(batch['instance_masks'], batch['instance_boxes'], batch['instance_box_valid'])
                self.assertTrue(torch.isfinite(batch['image']).all())
                self.assertTrue((batch['image'] >= 0).all())
                self.assertTrue((batch['image'] <= 1).all())
                for name in ('lead_masks', 'instance_masks'):
                    self.assertFalse((batch[name] * (batch['valid'] == 0)).any())
                torch.testing.assert_close(self.signals, source_copy, rtol=0, atol=0)

    def test_repeated_rhythm_instances_keep_identity_and_cabrera_polarity(self):
        config, parameters = capture_scenario('clean_print', height=120, width=160)
        parameters['layout'] = '3x4-cabrera+r3'
        layer = ECGSupervisionLayer(fs=100, config=config, parameters=parameters,
            photo=False, randomize=False, seed=12,
            supervision=SupervisionConfig(outputs=('lead_masks', 'instance_masks', 'instance_boxes')))
        batch = layer(self.signals)
        ids = batch['instance_lead_ids'][0]
        self.assertEqual(ids[12:].tolist(), [1, 6, 10])
        self.assertEqual(int((batch['instance_amplitude_sign'][0] < 0).sum()), 1)
        self.assertEqual(ids[batch['instance_amplitude_sign'][0] < 0].tolist(), [3])
        # Clean images have no warp. The lead channel is the union of every
        # occurrence with that canonical ID, including the long rhythm strip.
        for lead in range(12):
            expected = batch['instance_masks'][:, ids == lead].amax(1)
            torch.testing.assert_close(batch['lead_masks'][:, lead], expected, rtol=0, atol=0)
        self.assertEqual(int(batch['instance_box_valid'].sum()), 15)

    @staticmethod
    def analytic_batch():
        # Boxes chosen independently of the renderer, with fractional YOLO
        # answers derived by hand in a non-square image.
        return {
            'image': torch.full((1, 3, 40, 200), .5),
            'instance_boxes': torch.tensor([[[20., 10., 100., 30.], [-10., 5., 210., 45.],
                                             [0., 0., 1., 1.], [-1., -1., -1., -1.]]]),
            'instance_box_valid': torch.tensor([[True, True, True, False]]),
            'instance_active': torch.tensor([[True, True, True, True]]),
            'instance_lead_ids': torch.tensor([[1, 1, 12, 4]]),
        }

    def test_yolo_uses_normalized_xywh_and_preserves_repeated_class_instances(self):
        target = detection_targets(self.analytic_batch())
        self.assertEqual(target['cls'].flatten().tolist(), [1, 1])
        self.assertEqual(target['instance_idx'].tolist(), [0, 1])
        expected = torch.tensor([[.3, .5, .4, .5], [.5, .5625, 1., .875]])
        torch.testing.assert_close(target['bboxes'], expected, rtol=0, atol=1e-7)

    def test_offline_export_groups_augmentations_and_roundtrips_boxes_and_hashes(self):
        records = []
        for index in range(4):
            records += samples_from_batch(self.analytic_batch(), source_ids=[f'recording-{index//2}'],
                group_ids=[f'patient-{index//2}'], sample_ids=[f'augmentation-{index}'],
                provenance=[{'source_sha256': str(index//2) * 64}])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'dataset'
            manifest = export_yolo_dataset(records, out, seed=711, val_fraction=.5)
            self.assertEqual(len(manifest['samples']), 4)
            splits = {}
            for entry in manifest['samples']:
                splits.setdefault(entry['group_id'], set()).add(entry['split'])
                image_bytes = (out / entry['image']).read_bytes()
                labels_bytes = (out / entry['labels']).read_bytes()
                self.assertEqual(hashlib.sha256(image_bytes).hexdigest(), entry['image_sha256'])
                self.assertEqual(hashlib.sha256(labels_bytes).hexdigest(), entry['label_sha256'])
                values = [list(map(float, line.split())) for line in labels_bytes.decode().splitlines()]
                self.assertEqual(len(values), 2)
                torch.testing.assert_close(torch.tensor(values), torch.tensor([
                    [1., .3, .5, .4, .5], [1., .5, .5625, 1., .875]]), rtol=0, atol=1e-7)
            self.assertTrue(all(len(split) == 1 for split in splits.values()))
            self.assertEqual(set.union(*splits.values()), {'train', 'val'})
            saved = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(saved['samples'], manifest['samples'])
            reordered = export_yolo_dataset(list(reversed(records)), Path(temporary) / 'reordered',
                                           seed=711, val_fraction=.5)
            self.assertEqual(reordered['groups'], manifest['groups'])

    def test_source_hash_leakage_rejected_before_any_export_is_created(self):
        records = []
        for index in range(2):
            records += samples_from_batch(self.analytic_batch(), source_ids=[f'disguised-{index}'],
                group_ids=[f'patient-{index}'], sample_ids=[f'variant-{index}'],
                provenance=[{'source_sha256': 'a' * 64}])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'forbidden'
            with self.assertRaisesRegex(ValueError, 'Source leakage'):
                export_yolo_dataset(records, out)
            self.assertFalse(out.exists())

    def test_unmarked_paper_visibility_is_not_squared_at_subpixel_edges(self):
        config, parameters = capture_scenario('phone_photo', height=96, width=128)
        parameters.update(angle=.13, zoom=.811, artifact_present=0., pen_active=0.,
                          occlusion_active=0., tear_active=0.)
        layer = ECGSupervisionLayer(fs=100, config=config, parameters=parameters,
                                    randomize=False, seed=52)
        rendered = layer.render_batch(self.signals, photo=True, stage='all', return_scene=True)
        paper = rendered.scene['paper_mask']
        self.assertTrue(((paper > 0) & (paper < 1)).any())
        torch.testing.assert_close(rendered.targets['valid'], paper, rtol=0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
