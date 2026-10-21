import unittest

import numpy as np
import torch

from ecg_gpu.digitization.denoising_grid import DenoisingGridNet, restoration_loss
from ecg_gpu.offline_noise import FAMILIES, corrupt_observation


class DenoisingGridTests(unittest.TestCase):
    def test_opaque_patch_is_excluded_from_restoration(self):
        image = torch.zeros(1, 3, 32, 32)
        reference = image.clone()
        reference[:, :, 10:20, 10:20] = 1
        valid = torch.ones(1, 1, 32, 32)
        valid[:, :, 9:21, 9:21] = 0
        self.assertEqual(float(restoration_loss(image, reference, valid)['loss']), 0.)

    def test_network_backpropagates_point_and_restoration_heads(self):
        torch.set_num_threads(2)
        model = DenoisingGridNet(width=8)
        rgb = torch.rand(1, 3, 40, 56)
        output = model(rgb)
        loss = output['heat'].square().mean()+restoration_loss(output['clean'], rgb*.8, torch.ones_like(rgb[:, :1]))['loss']
        loss.backward()
        self.assertEqual(output['offsets'].shape, (1, 2, 40, 56))
        self.assertGreater(float(model.head.weight.grad[:1].abs().sum()), 0.)
        self.assertGreater(float(model.head.weight.grad[4:].abs().sum()), 0.)

    def test_noise_preserves_shapes_replays_and_updates_occlusion(self):
        rgb = np.full((64, 96, 3), 200, np.uint8)
        valid = np.ones((64, 96), np.float32)
        for family in FAMILIES:
            a, mask, metadata = corrupt_observation(rgb, valid, family=family, severity=.7, seed=12)
            b, _, _ = corrupt_observation(rgb, valid, family=family, severity=.7, seed=12)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.shape, rgb.shape)
            self.assertTrue(np.isfinite(mask).all())
            self.assertFalse(metadata['coordinates_changed'])
            if family == 'occlusion':
                self.assertGreater(int((mask == 0).sum()), 0)


if __name__ == '__main__':
    unittest.main()
