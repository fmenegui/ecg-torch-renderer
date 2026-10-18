"""Output parity and bounded allocations for reduced supervision workloads."""
import importlib.util
from types import SimpleNamespace
import unittest

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from ecg_gpu.gpu_supervision import ECGSupervisionLayer, SupervisionConfig
    from ecg_gpu.schema_v05 import LEADS


@unittest.skipUnless(TORCH, 'PyTorch required')
class SupervisionPerformanceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def layer(self, outputs, **options):
        # Exercise supervise independently of the substantially larger render.
        layer = ECGSupervisionLayer.__new__(ECGSupervisionLayer)
        torch.nn.Module.__init__(layer)
        layer.supervision = SupervisionConfig(outputs=outputs, **options)
        indices = [LEADS.index(lead) for lead in layer.supervision.leads]
        layer.register_buffer('_lead_indices', torch.tensor(indices), persistent=False)
        return layer

    def render(self):
        generator = torch.Generator().manual_seed(93)
        return SimpleNamespace(
            image=torch.rand(2, 3, 96, 128, generator=generator),
            targets={'traces': torch.rand(2, 12, 96, 128, generator=generator),
                     'valid': torch.rand(2, 1, 96, 128, generator=generator)})

    def test_union_only_is_bitwise_equal_to_full_supervision(self):
        rendered = self.render()
        originals = {key: value.clone() for key, value in rendered.targets.items()}
        for binary in (False, True):
            for visible_only in (False, True):
                for leads in (tuple(LEADS), ('V6', 'II')):
                    with self.subTest(binary=binary, visible_only=visible_only, leads=leads):
                        options = dict(binary=binary, visible_only=visible_only,
                                       leads=leads, threshold=.3)
                        reference = self.layer(('lead_masks', 'trace_mask', 'bw', 'wb'), **options).supervise(rendered)
                        observed = self.layer(('trace_mask', 'bw', 'wb'), **options).supervise(rendered)
                        for name in ('image', 'valid', 'trace_mask', 'bw', 'wb'):
                            self.assertTrue(torch.equal(reference[name], observed[name]), name)
        for key, original in originals.items():
            self.assertTrue(torch.equal(rendered.targets[key], original), key)

    def test_union_allocations_stay_below_a_full_lead_tensor(self):
        rendered = self.render()
        layer = self.layer(('trace_mask', 'bw', 'wb'), binary=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True) as profile:
            batch = layer.supervise(rendered)
        # This excludes the already-rendered input. A union target should never
        # allocate another [B,12,H,W] float tensor, even during thresholding.
        full_lead_bytes = rendered.targets['traces'].numel() * rendered.targets['traces'].element_size()
        largest_allocation = max(event.cpu_memory_usage for event in profile.events())
        self.assertLess(largest_allocation, full_lead_bytes)
        self.assertEqual(batch['trace_mask'].shape, (2, 1, 96, 128))

    def test_image_and_visibility_only_preserve_identity(self):
        rendered = self.render()
        batch = self.layer(()).supervise(rendered)
        self.assertEqual(set(batch), {'image', 'valid'})
        self.assertIs(batch['image'], rendered.image)
        self.assertIs(batch['valid'], rendered.targets['valid'])


if __name__ == '__main__':
    unittest.main()
