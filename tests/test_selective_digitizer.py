"""Independent physical-axis and failure controls for image-only selection."""

from dataclasses import replace
import unittest

import numpy as np

from ecg_gpu.digitization.selective import (
    GeometryEvidence, SelectiveDigitizer, SelectionPolicy, SignalProposal, SpecialistEvidence, select_proposal,
)


def waveforms():
    t = np.arange(1000) / 500
    return np.array([0.2 * np.sin(2 * np.pi * (1 + k / 5) * t) + 0.08 * np.cos(2 * np.pi * (8 + k) * t) for k in range(12)])


class SelectiveDigitizerTests(unittest.TestCase):
    def setUp(self):
        self.full = waveforms()
        partial = self.full.copy()
        partial[:, 300:350] = np.nan
        self.direct = SignalProposal(partial, 500, "same-source", "standard_12x1")
        self.candidate = SignalProposal(self.full, 500, "same-source", "standard_12x1")
        self.evidence = GeometryEvidence("same-source", True, True, True, True, 0.1)

    def choose(self, candidate=None, evidence=None):
        return select_proposal(self.direct, candidate or self.candidate, evidence or self.evidence)

    def test_more_support_with_exact_shared_signal_is_accepted(self):
        selected, report = self.choose()
        self.assertIs(selected, self.candidate)
        self.assertEqual(report["new_finite_samples"], 600)
        self.assertEqual(report["lost_finite_samples"], 0)
        np.testing.assert_array_equal(selected.values_mv, self.full)

    def test_rejected_result_is_the_exact_direct_object(self):
        candidate = replace(self.candidate, layout_id="limb_only")
        selected, report = self.choose(candidate)
        self.assertIs(selected, self.direct)
        self.assertEqual(report["reason"], "layout_identity_not_preserved")

    def test_a_single_lost_sample_rejects_a_larger_total_gain(self):
        values = self.full.copy()
        values[0, 10] = np.nan
        selected, report = self.choose(replace(self.candidate, values_mv=values))
        self.assertIs(selected, self.direct)
        self.assertEqual(report["lost_finite_samples"], 1)

    def test_units_and_canonical_order_are_not_guessed(self):
        with self.assertRaises(ValueError):
            replace(self.candidate, unit="uV")
        with self.assertRaises(ValueError):
            replace(self.candidate, lead_names=tuple(reversed(self.candidate.lead_names)))

    def test_a_time_origin_or_sample_rate_change_is_rejected(self):
        for proposal in [replace(self.candidate, fs=1000), replace(self.candidate, origin_s=0.002)]:
            selected, report = self.choose(proposal)
            self.assertIs(selected, self.direct)
            self.assertEqual(report["reason"], "different_physical_time_axes")

    def test_different_images_and_gold_assistance_are_rejected(self):
        for proposal, evidence in [(replace(self.candidate, source_id="other"), self.evidence), (self.candidate, replace(self.evidence, reference_used=True))]:
            self.assertIs(self.choose(proposal, evidence)[0], self.direct)

    def test_fold_crop_and_excessive_extrapolation_are_rejected(self):
        for evidence in [replace(self.evidence, boundary_simple=False), replace(self.evidence, full_frame_preserved=False), replace(self.evidence, triangle_orientation_consistent=False), replace(self.evidence, original_rgb_sampled=False), replace(self.evidence, extrapolated_fraction=0.8)]:
            self.assertIs(self.choose(evidence=evidence)[0], self.direct)

    def test_flatlines_polarity_amplitude_dc_and_timing_changes_are_rejected(self):
        variants = [np.zeros_like(self.full), -self.full, self.full * 1.3, self.full + 0.06, np.roll(self.full, 25, axis=1)]
        for values in variants:
            self.assertIs(self.choose(replace(self.candidate, values_mv=values))[0], self.direct)

    def test_swapping_nonidentical_lead_content_is_rejected(self):
        values = self.full.copy()
        values[[0, 7]] = values[[7, 0]]
        self.assertIs(self.choose(replace(self.candidate, values_mv=values))[0], self.direct)

    def test_no_material_support_gain_keeps_the_direct_result(self):
        proposal = replace(self.direct, values_mv=self.direct.values_mv.copy())
        selected, report = self.choose(proposal)
        self.assertIs(selected, self.direct)
        self.assertEqual(report["reason"], "no_material_support_gain")

    def test_completely_missing_direct_lead_is_not_silently_certified(self):
        values = self.direct.values_mv.copy()
        values[0] = np.nan
        direct = replace(self.direct, values_mv=values)
        selected, report = select_proposal(direct, self.candidate, self.evidence)
        self.assertIs(selected, direct)
        self.assertIn("shared_signal_disagreement", report["reason"])

    def test_selection_dominates_on_any_independent_printed_mask(self):
        selected, _ = self.choose()
        masks = np.random.default_rng(17).random((20, 12, 1000)) > 0.6
        for printed in masks:
            old = printed & np.isfinite(self.direct.values_mv)
            new = printed & np.isfinite(selected.values_mv)
            self.assertFalse((old & ~new).any())

    def test_result_copies_input_and_preserves_missing_samples(self):
        original = self.direct.values_mv.copy()
        proposal = replace(self.direct, values_mv=original)
        original[:] = 0
        self.assertTrue(np.isnan(proposal.values_mv[:, 300:350]).all())
        with self.assertRaises(ValueError):
            proposal.values_mv[:] = 0

    def test_invalid_numeric_contracts_are_rejected(self):
        for fs in [0, np.nan, True]:
            with self.assertRaises(ValueError):
                replace(self.direct, fs=fs)
        with self.assertRaises(ValueError):
            replace(self.direct, values_mv=np.full((12, 1000), np.inf))
        with self.assertRaises(ValueError):
            SelectionPolicy(minimum_new_samples=0)

    def test_candidate_error_preserves_input_and_exact_direct_result(self):
        original = np.full((40, 60, 3), 127, np.uint8)
        saved = original.copy()
        full = self.full

        class Backend:
            def predict(self, rgb, *, fs, samples, source_id):
                rgb[:] = 0
                return SignalProposal(full, fs, source_id, "standard_12x1")

        def failed(rgb, *, source_id):
            rgb[:] = 255
            raise RuntimeError("deliberate geometry failure")

        result, report = SelectiveDigitizer(Backend(), failed).predict(original, fs=500, samples=1000)
        np.testing.assert_array_equal(original, saved)
        np.testing.assert_array_equal(result.values_mv, full)
        self.assertEqual(report["reason"], "candidate_branch_failed")

    def test_unchanged_rgb_is_not_inferred_twice(self):
        calls = []
        full = self.full

        class Backend:
            def predict(self, rgb, *, fs, samples, source_id):
                calls.append(source_id)
                return SignalProposal(full, fs, source_id, "standard_12x1")

        def unchanged(rgb, *, source_id):
            return rgb, None

        _, report = SelectiveDigitizer(Backend(), unchanged).predict(np.zeros((10, 10, 3), np.uint8), fs=500, samples=1000)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["reason"], "unchanged_rgb")

    def test_native_specialist_requires_its_declared_layout_and_pinned_assets(self):
        evidence = SpecialistEvidence("same-source", "kaggle-second-final-six", "a" * 64)
        self.assertIs(select_proposal(self.direct, self.candidate, evidence)[0], self.direct)
        direct, candidate = replace(self.direct, layout_id="3x4+1R"), replace(self.candidate, layout_id="3x4+1R")
        mask = np.zeros_like(self.full, dtype=bool)
        for names in [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]:
            for col, name in enumerate(names):
                mask[self.direct.lead_names.index(name), col * 250:(col + 1) * 250] = True
        mask[1] = True
        direct = replace(direct, values_mv=np.where(mask, direct.values_mv, np.nan))
        candidate = replace(candidate, values_mv=np.where(mask, candidate.values_mv, np.nan))
        selected, report = select_proposal(direct, candidate, evidence)
        self.assertIs(selected, candidate)
        self.assertEqual(report["route"], "native_specialist")
        with self.assertRaises(ValueError):
            SpecialistEvidence("same-source", "kaggle-second-final-six", "missing")

    def test_unknown_and_fallback_layouts_do_not_certify_identity(self):
        for label in [None, "Unknown layout"]:
            a, b = replace(self.direct, layout_id=label), replace(self.candidate, layout_id=label)
            self.assertIs(select_proposal(a, b, self.evidence)[0], a)
        proposal = replace(self.candidate, metadata={"layout_fallback": True})
        self.assertIs(self.choose(proposal)[0], self.direct)


if __name__ == "__main__":
    unittest.main()
