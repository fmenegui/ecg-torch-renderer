import unittest
import numpy as np
import torch
from ecg_gpu.digitization.field_rectifier import GridFieldNet, chart_loss, chart_lattice

class FieldRectifierTests(unittest.TestCase):
    def test_heads_and_gradients(self):
        torch.set_num_threads(2)
        for contrast in (False, True):
            model = GridFieldNet(contrast=contrast, width=4)
            result = model(torch.rand(2, 3, 33, 49))
            self.assertEqual(result['chart'].shape, (2, 2, 33, 49))
            self.assertLessEqual(float(result['offsets'].abs().max()), .5)
            sum(x.square().mean() for x in result.values()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_chart_fractional_sampling_and_mask(self):
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(10), indexing='ij')
        chart = torch.stack(((xx+.5)/10, (yy+.5)/8))[None].requires_grad_()
        pixel = torch.tensor([[[3.25, 2.6], [6.3, 4.2]]])
        truth = (pixel+.5)/torch.tensor([10., 8.])
        value = chart_loss(chart, truth*2-1, truth, torch.tensor([[1., 0.]]))
        self.assertLess(float(value), 1e-10)
        value.backward()
        self.assertTrue(torch.isfinite(chart.grad).all())

    def test_relative_phase_permutation_and_duplicate(self):
        y, x = np.mgrid[:6, :7]
        ids = np.c_[y.ravel(), x.ravel()]
        xy = ids[:, ::-1]*20.+10
        rng = np.random.default_rng(1973)
        order = rng.permutation(len(xy))
        result = chart_lattice(xy[order], np.ones(len(xy)), (ids[order, ::-1]+[4.2, 7.3])/[96,64])
        self.assertEqual(result['status'], 'accepted')
        selected = np.array(result['selected_indices'])
        np.testing.assert_array_equal(result['ids'], ids[order[selected]])
        self.assertEqual(len(result['cells']), 30)
        duplicate = chart_lattice(np.r_[xy, xy[:1]+1], np.r_[np.ones(len(xy)), .1],
                                  np.r_[ids[:, ::-1], ids[:1, ::-1]]/[96,64])
        self.assertNotIn(len(xy), duplicate['selected_indices'])

    def test_reflected_cells_rejected(self):
        y, x = np.mgrid[:6, :7]
        ids = np.c_[y.ravel(), x.ravel()]
        result = chart_lattice(ids[:, ::-1]*[-20,20], np.ones(len(ids)), ids[:, ::-1]/[96,64])
        self.assertEqual(result['status'], 'rejected')
        self.assertGreater(result['invalid_cells'], 0)

if __name__ == '__main__':
    unittest.main()
