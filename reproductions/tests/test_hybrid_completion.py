from copy import deepcopy
import json
from pathlib import Path
import unittest

from reproductions.hybrid.worker_completion import scheduled_inputs, validate_inputs
from reproductions.runpod.controller import validate_plan


ROOT = Path(__file__).parents[1] / "hybrid"


class CompletionContracts(unittest.TestCase):
    def test_sealed_image_only_continuation_and_unique_identities(self):
        manifest = json.loads((ROOT / "completion-inputs.json").read_text())
        self.assertEqual(len(validate_inputs(manifest)), 7)
        for field in ["reference", "reference_waveforms", "true_layout", "printed_mask"]:
            bad = deepcopy(manifest)
            bad["specialist_images"][0][field] = "forbidden"
            with self.assertRaises(ValueError):
                validate_inputs(bad)
        bad = deepcopy(manifest)
        bad["repair_images"] = [bad["specialist_images"][0]]
        with self.assertRaises(ValueError):
            validate_inputs(bad)

    def test_model_policy_and_eligibility_cannot_be_tuned_in_the_continuation(self):
        manifest = json.loads((ROOT / "completion-inputs.json").read_text())
        for field in ["source_phase1_sha256", "policy_sha256"]:
            bad = deepcopy(manifest)
            bad[field] = "0" * 64
            with self.assertRaises(ValueError):
                validate_inputs(bad)
        bad = deepcopy(manifest)
        bad["specialist_images"].pop()
        with self.assertRaises(ValueError):
            validate_inputs(bad)

    def test_remaining_original_gpu_envelope_is_not_reset(self):
        plan = json.loads((ROOT / "completion-plan.json").read_text())
        validate_plan(plan)
        total = plan["prior_gpu_time_cost_upper_bound_usd"] + plan["maximum_pod_seconds"] * plan["maximum_gpu_price_per_hour"] / 3600
        self.assertLess(total, .45)
        for bad in [dict(plan, maximum_pod_seconds=1080), dict(plan, prior_gpu_time_cost_upper_bound_usd=.1),
                    dict(plan, prior_gpu_time_cost_upper_bound_usd=.2782),
                    dict(plan, combined_gpu_cost_cap_usd=.5)]:
            with self.assertRaises(ValueError):
                validate_plan(bad)

    def test_successful_repair_is_reused_without_another_neural_pass(self):
        manifest = json.loads((ROOT / 'completion-inputs.json').read_text())
        cases, repair = scheduled_inputs(manifest, {'reuse_repair_from_run': 41113820943})
        self.assertEqual(len(cases), 6)
        self.assertFalse(repair)
        cases, repair = scheduled_inputs(manifest, {})
        self.assertEqual(len(cases), 7)
        self.assertTrue(repair)
        with self.assertRaises(ValueError):
            scheduled_inputs(manifest, {'reuse_repair_from_run': 123})


if __name__ == "__main__":
    unittest.main()
