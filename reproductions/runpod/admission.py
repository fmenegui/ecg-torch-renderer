"""Read-only admission: identical training plans never allocate twice on push."""

import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request


def first_eligible(runs, plan_sha, plan_for_commit):
    candidates = []
    for run in runs:
        if run.get("conclusion") in ["cancelled", "skipped"]:
            continue
        if plan_for_commit(run["head_sha"]) == plan_sha:
            candidates.append(int(run["id"]))
    if not candidates:
        raise ValueError("No matching workflow run found. Refusing an unaccounted allocation")
    return min(candidates)


def main():
    token = os.environ["GITHUB_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = int(os.environ["GITHUB_RUN_ID"])
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")

    def read(path):
        request = urllib.request.Request(api + "/repos/" + repository + path, headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    branch = urllib.parse.quote(os.environ["GITHUB_REF_NAME"], safe="")
    path = "/actions/workflows/reproduction-full-training.yml/runs?branch=" + branch + "&per_page=100"
    runs = read(path)["workflow_runs"]
    payload = Path("reproductions/runpod/training-plan.json").read_bytes()
    plan_sha = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    cache = {}

    def plan_for_commit(commit):
        if commit not in cache:
            cache[commit] = read("/contents/reproductions/runpod/training-plan.json?ref=" + commit)["sha"]
        return cache[commit]

    first = first_eligible(runs, plan_sha, plan_for_commit)
    admitted = first == run_id
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write("admitted=" + str(admitted).lower() + "\n")
    print(json.dumps({"schema": "runpod-plan-admission-v1", "run_id": run_id, "first_plan_run_id": first, "plan_git_blob_sha": plan_sha, "admitted": admitted, "mutations": []}))


if __name__ == "__main__":
    main()
