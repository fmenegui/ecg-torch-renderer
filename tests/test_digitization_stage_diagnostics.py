"""Independent analytical contracts for the development diagnostic drivers."""
import unittest

import numpy as np

from examples.diagnose_digitization_stages import associate, predictions_from_cells
from examples.diagnose_digitization_signals import at_reference_times, reference_field_mesh
from ecg_gpu.digitization.regions import TraceObservation


class AssociationContracts(unittest.TestCase):
    def test_unique_closest_match_keeps_exclusions_explicit(self):
        prediction = np.array([[.8, 0.], [.1, 0.], [10., 0.], [100., 100.]])
        reference = np.array([[0., 0.], [10., 0.]])
        p, r = associate(prediction, reference, tolerance=2.)
        self.assertEqual(set(p.tolist()), {1, 2})
        self.assertEqual(len(set(r.tolist())), len(r))
        self.assertEqual(len(prediction)-len(p), 2)

    def test_empty_and_outside_tolerance_never_invent_match(self):
        p, r = associate(np.empty((0, 2)), np.zeros((1, 2)))
        self.assertEqual(len(p), 0)
        p, r = associate(np.array([[2.01, 0.]]), np.zeros((1, 2)))
        self.assertEqual(len(r), 0)


class GeometryContracts(unittest.TestCase):
    def test_affine_interior_and_explicit_no_rgb_claim(self):
        ids = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        physical = ids[:, ::-1]*5.
        observed = physical*np.array([3., 2.])+np.array([10., 20.])
        true = np.array([[1., 2.], [2.5, 2.5], [4., 3.]])
        source = true*np.array([3., 2.])+np.array([10., 20.])
        queries = {'paper': {'source': source}, 'anchors': {'source': source}}
        prediction, details = predictions_from_cells(observed, ids, queries)
        np.testing.assert_allclose(prediction['paper']['coords'], true, atol=1e-12)
        self.assertFalse(prediction['paper']['native_footprint'].any())
        self.assertIn('partial coordinates', details['scope'])

    def test_missing_corner_never_bridges_unobserved_cell(self):
        ids = np.array([[0, 0], [0, 1], [1, 0]])
        prediction, _ = predictions_from_cells(ids[:, ::-1]*5., ids,
            {'paper': {'source': np.array([[2., 2.]])}})
        self.assertFalse(prediction['paper']['claimed_eligible'].any())

    def test_reference_field_affine_inverse_and_opaque_block(self):
        y, x = np.mgrid[:17, :17]
        physical = np.stack((x/4., y/8.), -1)
        valid = np.ones((17, 17))
        inverse, forward, _ = reference_field_mesh(physical, valid, step=8)
        q = np.array([[1., 1.], [3., 1.5]])
        np.testing.assert_allclose(inverse(q), q*np.array([4., 8.]), atol=1e-12)
        valid[3, 3] = 0
        _, forward, _ = reference_field_mesh(physical, valid, step=8)
        self.assertTrue(np.isnan(forward(np.array([[3., 3.]]))).all())


class SignalContracts(unittest.TestCase):
    def test_no_gap_filling_and_no_endpoint_extrapolation(self):
        observation = TraceObservation('r', 'II', np.array([0., 1., np.nan, 3., 4.]),
            np.array([True, True, False, True, True]), np.ones(5), 4., 2.)
        times = np.array([1.9, 2., 2.125, 2.5, 2.875, 3., 3.1])
        predicted = at_reference_times(observation, times)
        np.testing.assert_allclose(predicted[[1, 2, 4, 5]], [0., .5, 3.5, 4.])
        self.assertTrue(np.isnan(predicted[[0, 3, 6]]).all())


if __name__ == '__main__':
    unittest.main()
