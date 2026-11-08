"""Barcodes, QR-style codes and stamps painted on the canonical page."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu import ECGRenderer, RenderConfig
    from ecg_gpu.office_marks import barcode, qr_code, stamp


@unittest.skipUnless(TORCH, 'Torch required')
class OfficeMarkTests(unittest.TestCase):
    def test_barcode_follows_its_bits_inside_the_box(self):
        bits = torch.tensor([[1, 0] * 36])
        mask = barcode((1, 40, 160), torch.ones(1), torch.tensor([8.]), torch.tensor([10.]),
                       torch.tensor([144.]), torch.tensor([20.]), bits)
        self.assertEqual(float(mask[0, :10].sum() + mask[0, 30:].sum()), 0.)
        row = mask[0, 20, 8:152]
        torch.testing.assert_close(row, torch.tensor([1., 1., 0., 0.] * 36))

    def test_qr_has_finder_patterns(self):
        bits = torch.zeros(1, 25, 25, dtype=torch.long)
        mask = qr_code((1, 50, 50), torch.ones(1), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([50.]), bits)
        # 2 px per module: finder outer ring dark, separator light, 3x3 core dark.
        self.assertEqual(float(mask[0, 0, 0]), 1.)
        self.assertEqual(float(mask[0, 2, 2]), 0.)
        self.assertEqual(float(mask[0, 6, 6]), 1.)
        self.assertEqual(float(mask[0, 0, 49]), 1.)
        self.assertEqual(float(mask[0, 49, 0]), 1.)
        self.assertEqual(float(mask[0, 49, 49]), 0.)

    def test_stamp_rings_and_inactive_records(self):
        segments = torch.ones(2, 36, dtype=torch.long)
        mask = stamp((2, 80, 80), torch.tensor([1., 0.]), torch.full((2,), 40.), torch.full((2,), 40.),
                     torch.full((2,), 30.), segments, torch.Generator().manual_seed(0))
        self.assertEqual(float(mask[1].sum()), 0.)
        self.assertGreater(float(mask[0, 40, 70]), .5)
        self.assertLess(float(mask[0, 40, 40 + 15]), .05)

    def test_renderer_paints_marks_and_keeps_traces(self):
        base = dict(height=192, width=256, profile='clean', antialias_samples=1)
        signals = torch.sin(torch.arange(800) / 9)[None, None].expand(1, 12, -1) * .3
        overrides = {'barcode_active': 1., 'qr_active': 1., 'stamp_active': 1., 'stamp_x': .5, 'stamp_y': .5}
        marked = ECGRenderer(RenderConfig(**base, office_marks=True))
        batch = marked(signals, 100, photo=False, parameters=overrides, generator=torch.Generator().manual_seed(1))
        self.assertGreater(float(batch.targets['office_marks'].sum()), 50)
        plain = ECGRenderer(RenderConfig(**base))(signals, 100, photo=False, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(batch.targets['traces'], plain.targets['traces'])
        self.assertNotIn('office_marks', ECGRenderer(RenderConfig(**base)).specification()['config'])


if __name__ == '__main__':
    unittest.main()
