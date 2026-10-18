"""Independent identity, visibility and pixel-edge detection oracles.

The box reference deliberately enumerates nonzero coordinates on CPU. The
production implementation uses row/column reductions entirely on-device.
"""
import importlib.util
import unittest
from unittest import mock

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGSupervisionLayer, RenderConfig, SupervisionConfig, mask_bounding_boxes
    from ecg_gpu.gpu_layouts import standard_layouts
    from ecg_gpu.schema_v05 import LEADS


def coordinate_box_oracle(masks, threshold):
    result = torch.full((*masks.shape[:2], 4), -1, dtype=torch.long)
    for sample in range(masks.shape[0]):
        for channel in range(masks.shape[1]):
            points = torch.nonzero(masks[sample, channel] > threshold)
            if len(points):
                yy, xx = zip(*points.tolist())
                result[sample, channel] = torch.tensor([min(xx), min(yy), max(xx) + 1, max(yy) + 1])
    return result


@unittest.skipUnless(TORCH, 'Training extra required')
class SupervisionIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = RenderConfig(height=97, width=131, profile='clean',
                                  surface_resolution=32, surface_mode='off',
                                  shadow_samples=0, antialias_samples=1)
        cls.parameters = {'layout': '3x4+r3', 'quarter': 0, 'lead_present': 1.,
                          'occlusion_active': 0., 'pen_active': 0., 'artifact_present': 0.,
                          'tear_active': 0., 'surface_strength': 0.}
        cls.signal = (.22 * torch.sin(torch.arange(500) / 9))[None, None].expand(1, 12, -1)

    def layer(self, *, photo=False, **config):
        return ECGSupervisionLayer(fs=100, config=self.config, photo=photo,
            parameters=self.parameters, seed=27, randomize=False,
            supervision=SupervisionConfig(outputs=(
                'lead_masks', 'lead_boxes', 'instance_masks', 'instance_boxes'), **config))

    def assert_boxes_match_coordinate_oracle(self, batch, threshold=.05):
        for kind in ('lead', 'instance'):
            boxes = coordinate_box_oracle(batch[kind + '_masks'], threshold)
            self.assertTrue(torch.equal(batch[kind + '_boxes'], boxes), kind)
            self.assertTrue(torch.equal(batch[kind + '_box_valid'], boxes[..., 0] >= 0), kind)

    def test_all_layouts_keep_semantic_union_and_unique_printed_slots(self):
        layer = self.layer()
        for spec in standard_layouts():
            with self.subTest(layout=spec.name):
                batch = layer(self.signal, parameters={'layout': spec.name})
                self.assertEqual(tuple(batch['lead_masks'].shape), (1, 12, 97, 131))
                self.assertEqual(tuple(batch['instance_masks'].shape), (1, 15, 97, 131))
                ids = batch['instance_lead_ids'][0]
                active = batch['instance_active'][0]
                expected_active = torch.zeros(15, dtype=torch.bool)
                for row in spec.rows:
                    for token in row:
                        lead = LEADS.index(token.removeprefix('-'))
                        expected_active[lead] = True
                        self.assertEqual(int(ids[lead]), lead)
                        self.assertEqual(int(batch['instance_amplitude_sign'][0, lead]),
                                         -1 if token.startswith('-') else 1)
                for index, name in enumerate(spec.rhythm):
                    expected_active[12 + index] = True
                    self.assertEqual(int(ids[12 + index]), LEADS.index(name))
                self.assertTrue(torch.equal(active, expected_active))
                self.assertFalse(batch['instance_masks'][0, ~active].any())
                # Independent grouping by the published ID, not the renderer's
                # scatter_reduce implementation. Clean geometry preserves max.
                for lead in range(12):
                    slots = batch['instance_masks'][0, ids == lead]
                    expected = slots.amax(0) if len(slots) else torch.zeros(97, 131)
                    self.assertTrue(torch.equal(batch['lead_masks'][0, lead], expected))
                self.assert_boxes_match_coordinate_oracle(batch)

    def test_repeated_rhythm_instances_are_separate_detection_objects(self):
        batch = self.layer()(self.signal, parameters={'rhythm_leads': [[1, 1, 1]]})
        # One short II and three long II instances, each with its own box.
        slots = (batch['instance_lead_ids'][0] == 1) & batch['instance_active'][0]
        self.assertEqual(int(slots.sum()), 4)
        boxes = batch['instance_boxes'][0, slots]
        self.assertEqual(len(torch.unique(boxes, dim=0)), 4)
        semantic = batch['lead_boxes'][0, 1]
        self.assertEqual(semantic.tolist(), [int(boxes[:, 0].min()), int(boxes[:, 1].min()),
                                            int(boxes[:, 2].max()), int(boxes[:, 3].max())])
        self.assertTrue((semantic[3] - semantic[1]) > (boxes[:, 3] - boxes[:, 1]).max())
        self.assertTrue((batch['instance_windows_s'][0, 12:, 0] == 0).all())
        torch.testing.assert_close(batch['instance_windows_s'][0, 12:, 1],
                                   torch.full((3,), (self.signal.shape[-1] - 1) / 100))

    def test_subset_channels_do_not_reclassify_or_drop_instance_targets(self):
        layer = self.layer(leads=('V6', 'II', 'aVR'))
        batch = layer(self.signal, parameters={'layout': '3x4-cabrera+r3'})
        self.assertEqual(batch['lead_ids'].tolist(), [[11, 1, 3]])
        self.assertEqual(tuple(batch['lead_masks'].shape), (1, 3, 97, 131))
        self.assertEqual(int(batch['instance_active'].sum()), 15)
        self.assertEqual(batch['instance_lead_ids'][0, :12].tolist(), list(range(12)))
        self.assertEqual(float(batch['instance_amplitude_sign'][0, 3]), -1.)
        self.assert_boxes_match_coordinate_oracle(batch)

    def test_random_sparse_boxes_match_coordinate_oracle_including_threshold_edges(self):
        generator = torch.Generator().manual_seed(125)
        masks = torch.rand(3, 15, 19, 23, generator=generator)
        masks[masks < .97] = 0
        masks[:, 0] = 0
        masks[0, 1, 0, 0] = .05
        masks[0, 2, -1, -1] = 1
        for threshold in (0., .05, .97, .999):
            boxes, valid = mask_bounding_boxes(masks, threshold)
            expected = coordinate_box_oracle(masks, threshold)
            self.assertTrue(torch.equal(boxes, expected))
            self.assertTrue(torch.equal(valid, expected[..., 0] >= 0))

    def test_warp_tear_pen_and_crop_use_same_visibility_for_masks_and_boxes(self):
        layer = self.layer(photo=True)
        conditions = [
            {'zoom': 1.45, 'translate_x': .22, 'translate_y': -.16,
             'angle': .17, 'perspective_x': .16, 'perspective_y': -.11},
            {'zoom': .8, 'tear_active': 1., 'tear_side': 0, 'tear_size': .8,
             'tear_position': 0., 'angle': 0.},
            {'pen_active': 1., 'pen_x': .1, 'pen_y': .38, 'pen_phase': 0.,
             'occlusion_active': 1., 'occlusion_x': .2, 'occlusion_y': .4,
             'occlusion_w': .45, 'occlusion_h': .12},
        ]
        for condition in conditions:
            with self.subTest(condition=condition):
                rendered = layer.render_batch(self.signal, stage='trace', photo=True,
                    parameters=condition, return_instances=True, return_baselines=False,
                    return_instance_boxes=False)
                original = {key: value.clone() for key, value in rendered.targets.items()}
                batch = layer.supervise(rendered)
                self.assertTrue(torch.equal(batch['lead_masks'], original['traces'] * original['valid']))
                self.assertTrue(torch.equal(batch['instance_masks'], original['instances'] * original['valid']))
                self.assertTrue(all(torch.equal(value, original[key]) for key, value in rendered.targets.items()))
                self.assertTrue((batch['lead_masks'] * (batch['valid'] == 0) == 0).all())
                self.assertTrue((batch['instance_masks'] * (batch['valid'] == 0) == 0).all())
                self.assert_boxes_match_coordinate_oracle(batch)

    def test_capture_can_hide_every_active_instance_without_relabeling_it(self):
        batch = self.layer()(self.signal, parameters={'occlusion_active': 1.,
            'occlusion_x': 0., 'occlusion_y': 0., 'occlusion_w': 1., 'occlusion_h': 1.})
        self.assertTrue(batch['instance_active'].all())
        self.assertFalse(batch['instance_box_valid'].any())
        self.assertFalse(batch['lead_box_valid'].any())
        self.assertFalse(batch['instance_masks'].any())
        self.assertTrue((batch['instance_boxes'] == -1).all())

    def test_visible_and_amodal_targets_keep_same_image_and_instance_identity(self):
        visible = self.layer()
        amodal = self.layer(visible_only=False)
        condition = {'occlusion_active': 1., 'occlusion_x': 0., 'occlusion_y': 0.,
                     'occlusion_w': 1., 'occlusion_h': .6}
        a = visible(self.signal, parameters=condition)
        b = amodal(self.signal, parameters=condition)
        self.assertTrue(torch.equal(a['image'], b['image']))
        self.assertTrue(torch.equal(a['instance_lead_ids'], b['instance_lead_ids']))
        self.assertTrue((a['instance_masks'] <= b['instance_masks']).all())
        self.assertTrue((a['instance_masks'] < b['instance_masks']).any())
        self.assert_boxes_match_coordinate_oracle(a)
        self.assert_boxes_match_coordinate_oracle(b)

    def test_nonempty_spatial_axes_are_required_for_box_extraction(self):
        for shape in ((2, 12, 0, 11), (2, 12, 7, 0)):
            with self.assertRaisesRegex(ValueError, 'nonempty image dimensions'):
                mask_bounding_boxes(torch.empty(shape))

    def test_instance_and_lead_supervision_need_no_host_reads(self):
        layer = self.layer(photo=True)
        with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host transfer')), \
             mock.patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy transfer')), \
             mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host scalar read')):
            batch = layer(self.signal)
        self.assert_boxes_match_coordinate_oracle(batch)


if __name__ == '__main__':
    unittest.main()
