"""Read-only telemetry and boot logs for this conversation's training pod."""

import argparse
from datetime import datetime, timezone
import json
import os
import re
import socket
import time
import urllib.request

from reproductions.common import write_json

POD_ID = "qtobel8h5j6ykc"
POD_NAME = "ecg-repro-41070920613-1"


def redact(value, key):
    value = value.replace(key, "[redacted]") if key else value
    value = re.sub(r"(?i)(token|password|api[_-]?key)([=:\s]+)[^\s&\"']+", r"\1\2[redacted]", value)
    return re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", value)


def verify_pod(pod):
    if pod.get("id") != POD_ID or pod.get("name") != POD_NAME:
        raise ValueError("Probe only accepts the exact owned training pod")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    key = os.environ["RUNPOD_API_KEY"]

    def request_json(url, body=None):
        request = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.load(response)

    def graph(query):
        return request_json("https://api.runpod.io/graphql", {"query": query})

    result = {"schema": "owned-training-probe-v1", "read_at_utc": datetime.now(timezone.utc).isoformat(),
              "pod_id": POD_ID, "mutations": [], "optimizer_progress_verified": False,
              "limitation": "Boot logs and GPU utilization do not prove completed training epochs"}
    pod = request_json("https://api.runpod.io/v2/pods/" + POD_ID)
    verify_pod(pod)
    result["pod"] = {k: pod[k] for k in ["id", "name", "status", "gpu", "cost", "dataCenterId", "createdAt"] if k in pod}
    base = graph("query { myself { pods { id runtime { __typename } } } }")
    if base.get("errors"):
        result["schema_read_errors"] = base["errors"]
    else:
        row = next((v for v in base["data"]["myself"]["pods"] if v["id"] == POD_ID), {})
        runtime = row.get("runtime")
        if runtime:
            type_name = runtime["__typename"]
            schema = graph('query { __type(name: ' + json.dumps(type_name) + ') { fields { name type { kind name ofType { name kind ofType { name kind ofType { name kind } } } } } } }')
            fields = (schema.get("data", {}).get("__type") or {}).get("fields", [])
            result["runtime_field_names"] = [f["name"] for f in fields]
            selections = [f["name"] for f in fields if f["name"] in ["uptimeInSeconds", "cpuPercent", "memoryPercent"]]
            gpu_field = next((f for f in fields if f["name"] == "gpus"), None)
            if gpu_field:
                t = gpu_field["type"]
                while not t.get("name") and t.get("ofType"):
                    t = t["ofType"]
                gpu_schema = graph('query { __type(name: ' + json.dumps(t["name"]) + ') { fields { name } } }')
                names = [f["name"] for f in (gpu_schema.get("data", {}).get("__type") or {}).get("fields", [])]
                metrics = [n for n in names if n in ["id", "gpuUtil", "memoryUtil"]]
                if metrics:
                    selections.append("gpus { " + " ".join(metrics) + " }")
            if selections:
                query = "query { myself { pods { id runtime { " + " ".join(selections) + " } } } }"
                result["runtime_samples"] = []
                for index in range(2):
                    if index:
                        time.sleep(15)
                    sample = graph(query)
                    if sample.get("errors"):
                        result["telemetry_errors"] = sample["errors"]
                        break
                    row = next((v for v in sample["data"]["myself"]["pods"] if v["id"] == POD_ID), {})
                    result["runtime_samples"].append({"read_at_utc": datetime.now(timezone.utc).isoformat(), "runtime": row.get("runtime")})
    result["logs"] = {}
    for source in ["system", "container"]:
        request = urllib.request.Request("https://api.runpod.io/v2/pods/" + POD_ID + "/logs?source=" + source + "&tail=30",
                                         headers={"Authorization": "Bearer " + key})
        rows = []
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                for _ in range(100):
                    line = response.readline().decode(errors="replace").strip()
                    if line.startswith("data:"):
                        rows.append(redact(line[5:].strip(), key))
                    if len(rows) >= 30 or not line:
                        break
        except (OSError, socket.timeout) as error:
            result["logs"][source + "_error"] = {"type": type(error).__name__, "message": redact(str(error), key)}
        result["logs"][source] = rows
    write_json(args.out, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
