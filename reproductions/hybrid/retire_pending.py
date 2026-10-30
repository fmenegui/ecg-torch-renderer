"""Retire only superseded, entirely unstarted hybrid workflow runs.

This never cancels the original training, a running pilot, or a workflow that
already has jobs. A current pilot defers instead if an older one is active.
"""

import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


def pending_candidates(runs, current_id, current_sha, branch):
    selected = []
    for run in runs:
        if (run.get("name") != "Frozen hybrid ECG Runpod pilot" or run.get("head_branch") != branch
                or int(run["id"]) >= current_id or run.get("head_sha") == current_sha):
            continue
        if run.get("status") == "in_progress":
            raise ValueError("An earlier hybrid pilot already started. Defer instead of allocating a duplicate")
        if run.get("status") in ["pending", "queued", "waiting", "requested"]:
            selected.append(int(run["id"]))
    return selected


def force_cancel_eligible(run, jobs, branch):
    return (run.get("name") == "Frozen hybrid ECG Runpod pilot" and run.get("head_branch") == branch
            and run.get("status") in ["pending", "queued", "waiting", "requested"] and not jobs)


def main():
    repo, branch = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_REF_NAME"]
    if repo != "fmenegui/ecg-torch-renderer" or not branch.startswith("research/hybrid-digitizer-"):
        raise ValueError("Retirement is restricted to this repository's hybrid research branches")
    token = os.environ["GITHUB_TOKEN"]
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com") + "/repos/" + repo

    def call(path, method="GET"):
        request = urllib.request.Request(base + path, data=b"" if method == "POST" else None, method=method,
            headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response) if response.status != 202 else {}

    runs = call("/actions/workflows/hybrid-runpod.yml/runs?per_page=100&branch=" + urllib.parse.quote(branch, safe=""))["workflow_runs"]
    candidates = pending_candidates(runs, int(os.environ["GITHUB_RUN_ID"]), os.environ["GITHUB_SHA"], branch)
    # Read all candidates before any cancellation. Nonempty jobs mean the
    # run may already have work or resources, so no cancellation is attempted.
    for identity in candidates:
        if call(f"/actions/runs/{identity}/jobs")["jobs"]:
            raise ValueError("A superseded run already has jobs. Defer for explicit inspection")
    retired = []
    forced = []
    for identity in candidates:
        call(f"/actions/runs/{identity}/cancel", method="POST")
        for _ in range(12):
            check = call(f"/actions/runs/{identity}")
            if check.get("status") == "completed" and check.get("conclusion") == "cancelled":
                retired.append(identity)
                break
            time.sleep(1)
        else:
            # GitHub's ordinary cancel can remain requested without retiring
            # an unstarted concurrency waiter. The official gh CLI exposes
            # force-cancel for this case. Re-check exact scope and zero jobs.
            check = call(f"/actions/runs/{identity}")
            jobs = call(f"/actions/runs/{identity}/jobs")["jobs"]
            if not force_cancel_eligible(check, jobs, branch):
                raise RuntimeError("Cancellation not verified and the run is no longer entirely unstarted")
            call(f"/actions/runs/{identity}/force-cancel", method="POST")
            forced.append(identity)
            for _ in range(45):
                check = call(f"/actions/runs/{identity}")
                if check.get("status") == "completed" and check.get("conclusion") == "cancelled":
                    retired.append(identity)
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Superseded unstarted workflow force-cancellation was not externally verified")
    receipt = {"schema": "unstarted-hybrid-workflow-retirement-v1", "retired_workflow_ids": retired,
               "forced_unstarted_workflow_ids": forced,
               "pod_mutations": [], "training_workflows_cancelled": [], "running_workflows_cancelled": []}
    path = Path("build/retired-pending-pilots.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
