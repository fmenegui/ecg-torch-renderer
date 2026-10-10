"""One-observation segmentation, visibility, detection and network training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu.gpu_module import ECGImageLayer
    from ecg_gpu.gpu_renderer import RenderConfig
    from ecg_gpu.gpu_supervision import ECGSupervisionLayer, SupervisionConfig, mask_bounding_boxes
    from ecg_gpu.gpu_cli import render_file


@unittest.skipUnless(TORCH, 'Training extra required')
class SupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = RenderConfig(height=96, width=128, profile='clean', surface_resolution=32,
                                  shadow_samples=0, antialias_samples=1)
        cls.signals = (.12 * torch.sin(torch.arange(500) / 13))[None, None].expand(2, 12, -1)
        cls.parameters = {'layout': '3x4+r3', 'lead_present': 1., 'quarter': 0,
                          'occlusion_active': 0., 'pen_active': 0., 'artifact_present': 0.}

    def layer(self, **options):
        return ECGSupervisionLayer(fs=100, config=self.config, parameters=self.parameters,
                                   photo=False, randomize=False, seed=71, **options)

    def test_default_masks_and_rgb_use_one_render_without_baselines(self):
        layer = self.layer()
        with mock.patch.object(layer.renderer, 'forward', wraps=layer.renderer.forward) as render:
            batch = layer(self.signals)
        self.assertEqual(render.call_count, 1)
        self.assertFalse(render.call_args.kwargs['return_baselines'])
        self.assertFalse(render.call_args.kwargs['return_instances'])
        plain = ECGImageLayer(fs=100, config=self.config, parameters=self.parameters,
                              randomize=False, seed=71)
        reference = plain.render_batch(self.signals, photo=False, stage='trace')
        self.assertTrue(torch.equal(batch['image'], reference.image))
        self.assertTrue(torch.equal(batch['lead_masks'], reference.targets['traces']))
        self.assertEqual(tuple(batch['lead_masks'].shape), (2, 12, 96, 128))
        self.assertTrue(torch.equal(batch['lead_ids'][0], torch.arange(12)))

    def test_polarities_masks_boxes_and_repeated_instances(self):
        layer = self.layer(supervision=SupervisionConfig(outputs=(
            'lead_masks', 'trace_mask', 'bw', 'wb', 'lead_boxes', 'instance_masks', 'instance_boxes')))
        batch = layer(self.signals)
        self.assertTrue(torch.equal(batch['trace_mask'], batch['lead_masks'].amax(1, keepdim=True)))
        self.assertTrue(torch.equal(batch['wb'], batch['trace_mask']))
        self.assertTrue(torch.equal(batch['bw'] + batch['wb'], torch.ones_like(batch['bw'])))
        self.assertEqual(tuple(batch['instance_masks'].shape), (2, 15, 96, 128))
        self.assertEqual(tuple(batch['lead_boxes'].shape), (2, 12, 4))
        self.assertEqual(tuple(batch['instance_boxes'].shape), (2, 15, 4))
        for name in ['lead', 'instance']:
            expected, visible = mask_bounding_boxes(batch[name + '_masks'])
            self.assertTrue(torch.equal(expected, batch[name + '_boxes']))
            self.assertTrue(torch.equal(visible, batch[name + '_box_valid']))
        self.assertTrue(batch['instance_active'].all())
        self.assertEqual(tuple(batch['instance_windows_s'].shape), (2, 15, 2))
        self.assertTrue((batch['instance_windows_s'][:, 12:, 0] == 0).all())
        durations = batch['instance_windows_s'][..., 1] - batch['instance_windows_s'][..., 0]
        self.assertTrue((durations[:, 12:] > durations[:, :12].amax(-1, keepdim=True)).all())

    def test_box_pixel_edges_empty_and_threshold(self):
        masks = torch.zeros(2, 3, 8, 11)
        masks[0, 0, 2:5, 3:7] = 1
        masks[1, 2, 7, 10] = .4
        boxes, present = mask_bounding_boxes(masks, .05)
        self.assertEqual(boxes[0, 0].tolist(), [3, 2, 7, 5])
        self.assertEqual(boxes[1, 2].tolist(), [10, 7, 11, 8])
        self.assertTrue((boxes[~present] == -1).all())
        high, exists = mask_bounding_boxes(masks, .5)
        self.assertFalse(exists[1, 2])
        self.assertTrue((high[1, 2] == -1).all())

    def test_visible_targets_exclude_opaque_pixels(self):
        visible = self.layer(supervision=SupervisionConfig(outputs=('lead_masks', 'wb')))
        projected = self.layer(supervision=SupervisionConfig(outputs=('lead_masks', 'wb'), visible_only=False))
        params = {'occlusion_active': 1., 'occlusion_x': 0., 'occlusion_y': 0.,
                  'occlusion_w': 1., 'occlusion_h': 1.}
        a = visible(self.signals, parameters=params)
        b = projected(self.signals, parameters=params)
        self.assertTrue((a['valid'] == 0).all())
        self.assertTrue((a['lead_masks'] == 0).all())
        self.assertTrue(b['lead_masks'].sum() > 0)

    def test_channel_selection_binary_and_missing_slots(self):
        all_channels = self.layer()(self.signals)
        selected = self.layer(supervision=SupervisionConfig(outputs=('lead_masks', 'bw'),
                              leads=('V6', 'II'), binary=True, threshold=.3))(self.signals)
        self.assertTrue(torch.equal(selected['lead_masks'], (all_channels['lead_masks'][:, [11, 1]] > .3).float()))
        self.assertTrue(torch.equal(selected['lead_ids'][0], torch.tensor([11, 1])))
        missing = self.layer(supervision=SupervisionConfig(outputs=('lead_boxes', 'instance_boxes')))(
            self.signals, parameters={'lead_present': 0.})
        self.assertFalse(missing['lead_box_valid'].any())
        self.assertTrue((missing['lead_boxes'] == -1).all())
        self.assertFalse(missing['instance_box_valid'].any())

    def test_independent_channels_retain_crossing_pixels(self):
        layer = self.layer()
        layer.renderer.row_table.zero_()
        layer.renderer.col_table.zero_()
        layer.renderer.sign_table.fill_(1.)
        batch = layer(self.signals)
        self.assertTrue(((batch['lead_masks'] > .1).sum(1) > 1).any())

    def test_no_host_transfer_or_scalar_reads_and_gradients(self):
        layer = self.layer(supervision=SupervisionConfig(outputs=('lead_masks', 'lead_boxes', 'instance_boxes')))
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.Conv2d(8, 12, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        before = model[0].weight.detach().clone()
        source = self.signals.clone().requires_grad_()
        with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host copy')), \
             mock.patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy')), \
             mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar sync')):
            batch = layer(source)
            logits = model(batch['image'])
            pixel_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['lead_masks'], reduction='none')
            loss = (pixel_loss * batch['valid']).sum() / (batch['valid'].sum() * 12).clamp_min(1)
            loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(source.grad)
        self.assertFalse(torch.equal(before, model[0].weight))
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters()))

    def test_resume_and_configuration_guard(self):
        layer = ECGSupervisionLayer(fs=100, config=self.config, seed=84)
        layer(self.signals)
        saved = copy.deepcopy(layer.state_dict())
        expected = layer(self.signals)
        replacement = ECGSupervisionLayer(fs=100, config=self.config, seed=999)
        replacement.load_state_dict(saved)
        observed = replacement(self.signals)
        self.assertTrue(all(torch.equal(expected[key], observed[key]) for key in expected))
        changed = ECGSupervisionLayer(fs=100, config=self.config, supervision=SupervisionConfig(binary=True))
        with self.assertRaisesRegex(ValueError, 'Supervision configuration changed'):
            changed.load_state_dict(saved)

    def test_invalid_configuration(self):
        for options in [{'outputs': ('unknown',)}, {'outputs': ('bw', 'bw')}, {'leads': ('II', 'II')},
                        {'threshold': 1.}, {'threshold': float('nan')}, {'binary': 'yes'}]:
            with self.assertRaises(ValueError):
                SupervisionConfig(**options)

    def test_cli_exports_paired_arrays_pixel_boxes_and_lossless_polarities(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / 'input.npz', signals=self.signals[:1].numpy(), fs=100, units='mV')
            report = render_file(root / 'input.npz', root / 'image.png', config=self.config,
                parameters=self.parameters, device='cpu', targets_dir=root / 'targets',
                supervision=SupervisionConfig(outputs=('lead_masks', 'bw', 'wb', 'lead_boxes', 'instance_boxes')))
            with np.load(root / 'targets/targets.npz', allow_pickle=False) as source:
                self.assertEqual(source['lead_masks'].shape, (1, 12, 96, 128))
                self.assertEqual(source['instance_boxes'].shape, (1, 15, 4))
                self.assertTrue(np.array_equal(source['bw'] + source['wb'], np.ones_like(source['bw'])))
            boxes = json.loads((root / 'targets/boxes.json').read_text())
            self.assertEqual(boxes['coordinate_format'], 'xyxy_pixels_exclusive')
            a = np.asarray(Image.open(root / 'targets/bw-00000.png')).astype(int)
            b = np.asarray(Image.open(root / 'targets/wb-00000.png')).astype(int)
            self.assertTrue((np.abs(a + b - 255) <= 1).all())
            self.assertIn('lead_boxes', report['target_shapes'])

    @unittest.skipUnless(TORCH and torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_compiled_cuda_supervision_and_unet_head_under_amp(self):
        layer = self.layer(compile_raster=True, supervision=SupervisionConfig(outputs=(
            'lead_masks', 'bw', 'wb', 'lead_boxes', 'instance_masks', 'instance_boxes'))).cuda()
        source = self.signals.cuda()
        head = torch.nn.Conv2d(3, 12, 1).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            batch = layer(source)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(batch['image']), batch['lead_masks'])
        loss.backward()
        self.assertTrue(all(value.device.type == 'cuda' for value in batch.values()))
        self.assertEqual(batch['lead_masks'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(head.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
