"""Material appearance tests using independent spatial/energy expectations."""
import unittest

import torch

from ecg_gpu.gpu_material import PaperMaterial


def parameters(batch=1, **overrides):
    result = {key: torch.zeros(batch) for key in (
        'paper_age', 'stain_strength', 'coffee_strength', 'thermal_fade',
        'foxing_strength', 'fiber_strength', 'water_strength', 'thermal_darkening_strength')}
    for key, value in overrides.items():
        result[key] = torch.as_tensor(value, dtype=torch.float32).expand(batch)
    return result


class MaterialAppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_strength_has_exact_identity_for_every_factor(self):
        module = PaperMaterial(96, 128)
        output = module(parameters(2), torch.Generator().manual_seed(17))
        for name, tensor in output.items():
            channels = 1 if name == 'print_retention' else 3
            self.assertEqual(tensor.shape, (2, channels, 96, 128))
            self.assertTrue(torch.equal(tensor, torch.ones_like(tensor)))

    def test_replay_and_batch_variation(self):
        module = PaperMaterial(97, 131)
        p = parameters(2, paper_age=.8, coffee_strength=.9, stain_strength=.4,
                       foxing_strength=.7, thermal_fade=.8, fiber_strength=1.)
        first = module(p, torch.Generator().manual_seed(81))
        replay = module(p, torch.Generator().manual_seed(81))
        changed = module(p, torch.Generator().manual_seed(82))
        for name, tensor in first.items():
            self.assertTrue(torch.equal(tensor, replay[name]))
            self.assertFalse(torch.equal(tensor, changed[name]))
            self.assertFalse(torch.equal(tensor[0], tensor[1]))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertGreaterEqual(float(tensor.min()), 0.)
            self.assertLessEqual(float(tensor.max()), 1.)

    def test_oxidation_is_spatial_and_more_absorbing_in_blue(self):
        module = PaperMaterial(192, 256)
        factor = module(parameters(paper_age=.8), torch.Generator().manual_seed(22))['substrate_factor'][0]
        edge = torch.cat((factor[:, :5].flatten(1), factor[:, -5:].flatten(1),
                          factor[:, :, :5].flatten(1), factor[:, :, -5:].flatten(1)), 1)
        center = factor[:, 70:120, 90:165]
        self.assertLess(float(edge.mean()), float(center.mean()) - .04)
        self.assertTrue((factor[2] < factor[1]).all())
        self.assertTrue((factor[1] < factor[0]).all())
        self.assertGreater(float(center.std((-1, -2)).mean()), .005)

    def test_coffee_deposit_is_localized_irregular_and_not_uniform_annulus(self):
        module = PaperMaterial(240, 320)
        p = parameters(coffee_strength=1., coffee_radius=.16, stain_x=.5, stain_y=.5)
        deposit = module(p, torch.Generator().manual_seed(11))['deposit_transmission'][0, 2]
        x, y = module.uv[0]
        radius = torch.sqrt((x - .5).square() + ((y - .5) * .75).square())
        outside = radius > .29
        self.assertLess(float((1 - deposit[outside]).max()), .001)
        self.assertGreater(float((1 - deposit).max()), .25)
        # On a nominal circular annulus, pinning and variable contact radius
        # create clear angular variation. The center has only weak deposition.
        annulus = (radius > .145) & (radius < .175)
        self.assertGreater(float(deposit[annulus].std()), .07)
        center = radius < .03
        self.assertLess(float((1 - deposit[center]).mean()), .12)

    def test_stain_moves_with_material_center_without_affecting_substrate(self):
        module = PaperMaterial(160, 240)
        left = module(parameters(stain_strength=1., stain_x=.20, stain_y=.45),
                      torch.Generator().manual_seed(102))
        right = module(parameters(stain_strength=1., stain_x=.80, stain_y=.45),
                       torch.Generator().manual_seed(102))
        x = module.uv[:, :1]
        for output, target in ((left, .2), (right, .8)):
            mass = 1 - output['deposit_transmission'].mean(1, keepdim=True)
            centroid = (mass * x).sum() / mass.sum()
            self.assertAlmostEqual(float(centroid), target, delta=.035)
            self.assertTrue(torch.equal(output['substrate_factor'], torch.ones_like(output['substrate_factor'])))

    def test_thermal_fade_mottles_print_only(self):
        module = PaperMaterial(128, 256)
        output = module(parameters(thermal_fade=.9, fade_position=.4, fade_width=.08),
                        torch.Generator().manual_seed(61))
        retention = output['print_retention'][0, 0]
        self.assertLess(float(retention[:, 97:108].mean()), .4)
        self.assertGreater(float(retention[:, 220:].min()), .999)
        self.assertGreater(float(retention[:, 102].std()), .01)
        for key in ('substrate_factor', 'deposit_transmission'):
            self.assertTrue(torch.equal(output[key], torch.ones_like(output[key])))

    def test_optional_foxing_is_sparse_and_fibers_are_subtle(self):
        module = PaperMaterial(192, 256)
        output = module(parameters(foxing_strength=1.), torch.Generator().manual_seed(72))
        absorption = 1 - output['substrate_factor'][0, 2]
        self.assertGreater(float((absorption > .04).float().mean()), .001)
        self.assertLess(float((absorption > .04).float().mean()), .20)
        fibers = module(parameters(fiber_strength=1.), torch.Generator().manual_seed(72))['substrate_factor']
        self.assertGreater(float(fibers.min()), .989)
        self.assertLess(float(fibers.std()), .003)

    def test_thermal_development_darkens_background_without_fading_print(self):
        module = PaperMaterial(96, 128)
        output = module(parameters(thermal_darkening_strength=.7, fade_position=.5, fade_width=.15),
                        torch.Generator().manual_seed(19))
        background = output['substrate_factor']
        self.assertLess(float(background[..., 62:66].mean()), .15)
        self.assertGreater(float(background[..., :6].min()), .999)
        torch.testing.assert_close(background[:, :1], background[:, 1:2], rtol=0, atol=0)
        self.assertTrue(torch.equal(output['print_retention'], torch.ones_like(output['print_retention'])))

    def test_no_mutation_and_finite_strength_gradient(self):
        module = PaperMaterial(64, 96)
        age = torch.tensor([.3], requires_grad=True)
        p = parameters(paper_age=.3, coffee_strength=.5, thermal_fade=.5)
        p['paper_age'] = age
        before = {name: value.detach().clone() for name, value in p.items()}
        output = module(p, torch.Generator().manual_seed(5))
        sum(tensor.mean() for tensor in output.values()).backward()
        self.assertTrue(torch.isfinite(age.grad).all())
        self.assertLess(float(age.grad), 0.)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in p.items()))


if __name__ == '__main__':
    unittest.main()
