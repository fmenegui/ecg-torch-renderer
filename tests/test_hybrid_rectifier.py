import unittest

import numpy as np

from ecg_gpu.digitization.hybrid_rectifier import forward_supported, refine_lattice, similarity_residual
from ecg_gpu.digitization.lattice import organize_lattice


class HybridGeometryTests(unittest.TestCase):
    def test_similarity_does_not_hide_anisotropic_scale(self):
        y, x = np.mgrid[:8, :12]
        truth = np.column_stack((x.ravel(), y.ravel())).astype(float)*5
        self.assertLess(similarity_residual(truth*3+17, truth)['p95_mm'], 1e-10)
        self.assertGreater(similarity_residual(truth*[1., 1.3], truth)['p95_mm'], 1.)

    def test_no_support_across_missing_cell(self):
        y, x = np.mgrid[:3, :3]
        ids = np.column_stack((y.ravel(), x.ravel()))
        xy = ids[:, ::-1].astype(float)*10
        result = forward_supported(xy, xy, ids, np.array([[5., 5.], [15., 15.]]), [[0, 0]])
        np.testing.assert_allclose(result[0], [5, 5])
        self.assertTrue(np.isnan(result[1]).all())

    def test_refinement_preserves_affine_grid_and_reduces_jitter(self):
        y, x = np.mgrid[:10, :15]
        xy = np.column_stack((x.ravel(), y.ravel())).astype(float)*12+10
        truth = xy@np.array([[1., .08], [.15, 1.1]])
        scores = np.ones(len(truth))
        topology = organize_lattice(truth, scores)
        fit, checked = refine_lattice(truth, scores, topology)
        np.testing.assert_allclose(fit, truth[topology['selected_indices']], atol=1e-4)
        noisy = truth+np.random.default_rng(123).normal(0, .3, truth.shape)
        topology = organize_lattice(noisy, scores)
        fit, checked = refine_lattice(noisy, scores, topology, smoothness=.2)
        selected = topology['selected_indices']
        self.assertLess(np.mean((fit-truth[selected])**2), np.mean((noisy[selected]-truth[selected])**2))
        self.assertEqual(checked['status'], 'accepted')


if __name__ == '__main__':
    unittest.main()
