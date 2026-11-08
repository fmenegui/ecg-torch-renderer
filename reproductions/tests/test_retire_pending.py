"""Owned unstarted-run retirement cannot interrupt training or live pilots."""

import unittest

from reproductions.hybrid.retire_pending import force_cancel_eligible, pending_candidates


class PendingPilotRetirementTests(unittest.TestCase):
    def run_record(self, identity, **updates):
        return {"id": identity, "name": "Frozen hybrid ECG Runpod pilot", "head_branch": "research/hybrid-digitizer-test",
                "head_sha": "old-sha", "status": "pending", **updates}

    def test_only_earlier_superseded_unstarted_runs_in_the_exact_workflow_and_branch_match(self):
        runs = [self.run_record(1), self.run_record(2, name="Full original ECG training"),
                self.run_record(3, head_branch="another-branch"), self.run_record(4, head_sha="current"),
                self.run_record(5, status="completed"), self.run_record(10)]
        self.assertEqual(pending_candidates(runs, 10, "current", "research/hybrid-digitizer-test"), [1])

    def test_an_active_earlier_pilot_defers_new_paid_work_instead_of_cancelling_it(self):
        with self.assertRaises(ValueError):
            pending_candidates([self.run_record(1, status="in_progress")], 10, "current", "research/hybrid-digitizer-test")

    def test_original_training_is_never_touched_even_when_it_is_active(self):
        runs = [self.run_record(1, name="Full original ECG training", status="in_progress")]
        self.assertEqual(pending_candidates(runs, 10, "current", "research/hybrid-digitizer-test"), [])

    def test_forced_retirement_is_impossible_after_any_job_or_for_an_active_or_other_workflow(self):
        branch = "research/hybrid-digitizer-test"
        run = self.run_record(1)
        self.assertTrue(force_cancel_eligible(run, [], branch))
        self.assertFalse(force_cancel_eligible(run, [{"name": "admit"}], branch))
        self.assertFalse(force_cancel_eligible(dict(run, status="in_progress"), [], branch))
        self.assertFalse(force_cancel_eligible(dict(run, name="Full original ECG training"), [], branch))


if __name__ == "__main__":
    unittest.main()
