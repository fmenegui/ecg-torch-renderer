import unittest

import numpy as np


class DotterTests(unittest.TestCase):
    def test_tile_boundaries_preserve_pointwise_probabilities(self):
        import torch
        from reproductions.powerful_medical.infer import predict_tiled

        class Pointwise(torch.nn.Module):
            def forward(self, x):
                return x[:, :1] - 0.5

        rgb = np.random.default_rng(71).integers(0, 256, size=(39, 53, 3), dtype=np.uint8)
        observed = predict_tiled(Pointwise(), rgb, tile=32, stride=17)
        expected = 1 / (1 + np.exp(-(rgb[..., 0].astype(float) / 255 - 0.5)))
        np.testing.assert_allclose(observed, expected, atol=1e-7)

    def test_declared_network_trains_and_retains_odd_pixel_shape(self):
        import torch
        from reproductions.powerful_medical.dotter import Dotter
        torch.set_num_threads(2)
        torch.manual_seed(1973)
        model = Dotter(widths=(4, 8, 16))
        image = torch.rand(1, 3, 65, 79)
        target = torch.zeros(1, 1, 65, 79)
        target[:, :, 13, 23] = 1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        before = model.output.weight.detach().clone()
        output = model(image)
        self.assertEqual(output.shape, target.shape)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.output.weight))
        self.assertIn("normalization", model.architecture_choices_not_disclosed_by_authors)


if __name__ == "__main__":
    unittest.main()
