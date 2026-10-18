"""Independent box coordinate, split-leakage and loss-gradient contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from PIL import Image

from ecg_gpu.training import detection_targets, padded_detection_targets, masked_segmentation_loss
from ecg_gpu.dataset_export import samples_from_batch, export_yolo_dataset


def fixture():
    return {'image': torch.ones(2, 3, 10, 20),
        'instance_boxes': torch.tensor([[[2, 1, 10, 5], [0, 5, 20, 10], [-1, -1, -1, -1]],
                                        [[-5, -5, 25, 15], [0, 0, 0, 2], [1, 1, 2, 2]]]),
        'instance_box_valid': torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        'instance_active': torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        'instance_lead_ids': torch.tensor([[1, 1, 11], [0, 2, 4]])}


class AdapterTests(unittest.TestCase):
    def test_known_normalized_boxes_and_repeated_instances_without_host_reads(self):
        with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host')), \
             mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar')), \
             mock.patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('numpy')):
            targets = detection_targets(fixture())
            padded = padded_detection_targets(fixture())
        torch.testing.assert_close(targets['bboxes'], torch.tensor([[.3, .3, .4, .4], [.5, .75, 1., .5], [.5, .5, 1., 1.]]))
        self.assertEqual(targets['cls'].flatten().tolist(), [1, 1, 0])
        self.assertEqual(targets['instance_idx'].tolist(), [0, 1, 0])
        self.assertEqual(targets['batch_idx'].tolist(), [0, 0, 1])
        self.assertTrue((padded['boxes'][~padded['valid']] == 0).all())

    def test_all_empty_and_nonfinite_boxes(self):
        batch = fixture()
        batch['instance_active'].zero_()
        targets = detection_targets(batch)
        self.assertEqual(targets['bboxes'].shape, (0, 4))
        self.assertEqual(targets['cls'].shape, (0, 1))
        batch = fixture()
        batch['instance_boxes'] = batch['instance_boxes'].float()
        batch['instance_boxes'][0, 0, 0] = float('nan')
        self.assertEqual(detection_targets(batch)['bboxes'].shape[0], 2)

    def test_torchvision_background_offset_and_pixel_exclusive_area(self):
        targets = detection_targets(fixture(), format='torchvision')
        self.assertEqual(targets[1]['labels'].tolist(), [1])
        self.assertEqual(targets[0]['labels'].tolist(), [2, 2])
        self.assertEqual(targets[0]['area'].tolist(), [32., 100.])
        self.assertEqual(detection_targets(fixture(), format='torchvision', label_offset=0)[1]['labels'].tolist(), [0])
        with self.assertRaises(ValueError):
            detection_targets(fixture(), label_offset=1)

    def test_visibility_requires_explicit_measurement(self):
        with self.assertRaisesRegex(ValueError, 'instance_visible_fraction'):
            detection_targets(fixture(), min_visible_fraction=.5)
        batch = fixture()
        batch['instance_visible_fraction'] = torch.tensor([[.2, .9, 0.], [.8, 0., 0.]])
        self.assertEqual(detection_targets(batch, min_visible_fraction=.5)['cls'].flatten().tolist(), [1, 0])

    def test_masked_loss_independent_channel_and_ignored_gradients(self):
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        valid = torch.tensor([[[[1., 0.], [0., 0.]]]])
        target = torch.ones_like(logits)
        loss = masked_segmentation_loss(logits, target, valid, pos_weight=torch.tensor([1., 2.]))
        loss.backward()
        self.assertAlmostEqual(loss.item(), 1.5 * .69314718)
        self.assertEqual(logits.grad[0, 0, 0, 1].item(), 0.)
        self.assertTrue((logits.grad[:, :, 0, 0] < 0).all())
        logits.grad.zero_()
        ignored = masked_segmentation_loss(logits, target, torch.zeros_like(valid))
        ignored.backward()
        self.assertEqual(ignored.item(), 0.)
        self.assertTrue((logits.grad == 0).all())

    def test_dataset_actual_png_labels_provenance_and_group_split(self):
        batch = fixture()
        records = samples_from_batch(batch, source_ids=['record-a', 'record-b'], group_ids=['patient-a', 'patient-b'], sample_ids=['a', 'b'])
        variant = dict(records[0], sample_id='a-variant')
        with tempfile.TemporaryDirectory() as folder:
            manifest = export_yolo_dataset([*records, variant], Path(folder)/'dataset')
            root = Path(folder)/'dataset'
            self.assertEqual(len(manifest['samples']), 3)
            self.assertEqual(manifest['samples'][0]['split'], manifest['samples'][2]['split'])
            self.assertNotEqual(manifest['samples'][0]['split'], manifest['samples'][1]['split'])
            for record in manifest['samples']:
                with Image.open(root/record['image']) as image:
                    self.assertEqual(image.size, (20, 10))
                for line in (root/record['labels']).read_text().splitlines():
                    values = line.split()
                    self.assertEqual(len(values), 5)
                    self.assertTrue(0 <= int(values[0]) < 12)
                    self.assertTrue(all(0 <= float(v) <= 1 for v in values[1:]))
            self.assertEqual(json.loads((root/'dataset.yaml').read_text())['nc'], 12)
            with self.assertRaises(FileExistsError):
                export_yolo_dataset(records, root)

    def test_source_and_hash_leakage_and_bad_sample_name_rejected_before_files(self):
        records = samples_from_batch(fixture(), source_ids=['a', 'b'], group_ids=['patient-a', 'patient-b'], sample_ids=['a', 'b'])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'absent'
            invalid = [records[0], dict(records[1], source_id='a')]
            with self.assertRaisesRegex(ValueError, 'Source leakage'):
                export_yolo_dataset(invalid, out)
            self.assertFalse(out.exists())
            same_hash = [dict(r, provenance={'source_sha256': 'a'*64}) for r in records]
            with self.assertRaisesRegex(ValueError, 'source_sha256'):
                export_yolo_dataset(same_hash, out)
            with self.assertRaisesRegex(ValueError, 'safe filename'):
                export_yolo_dataset([dict(records[0], sample_id='../x'), records[1]], out)
            with self.assertRaisesRegex(ValueError, 'two independent groups'):
                export_yolo_dataset(records[:1], out)
            self.assertFalse(out.exists())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_padded_and_packed_targets(self):
        batch = {key: value.cuda() for key, value in fixture().items()}
        self.assertTrue(all(value.is_cuda for value in detection_targets(batch).values()))
        self.assertTrue(all(value.is_cuda for value in padded_detection_targets(batch).values()))


if __name__ == '__main__':
    unittest.main()
