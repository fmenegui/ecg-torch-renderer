"""Read only this reproduction's resources, never credentials or other pods."""

import argparse
import json
import os
import socket
import urllib.request

from reproductions.common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    key = os.environ["RUNPOD_API_KEY"]
    results = {"schema": "runpod-owned-observation-v1", "mutations": [], "pods": [], "network_volumes": []}
    for endpoint, field, keys in [
        ("pods", "pods", ["id", "name", "status", "image", "gpu", "cost", "dataCenterId", "runtime", "createdAt"]),
        ("network-volumes", "networkVolumes", ["id", "name", "size", "dataCenter", "type"]),
    ]:
        request = urllib.request.Request("https://api.runpod.io/v2/" + endpoint, headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        rows = data[field] if isinstance(data, dict) else data
        selected = [{k: row[k] for k in keys if k in row} for row in rows if str(row.get("name", "")).startswith("ecg-repro-")]
        results["pods" if endpoint == "pods" else "network_volumes"] = selected
    query = "query { myself { pods { id name desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"
    request = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": query}).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        graph = json.load(response)
    if graph.get("errors"):
        results["graphql_error"] = graph["errors"]
    else:
        results["graphql_pods"] = [r for r in graph["data"]["myself"]["pods"] if r["name"].startswith("ecg-repro-")]
    for pod in results["pods"]:
        request = urllib.request.Request("https://api.runpod.io/v2/pods/" + pod["id"] + "/logs?source=system&tail=12", headers={"Authorization": "Bearer " + key})
        records = []
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                for _ in range(60):
                    line = response.readline().decode(errors="replace").strip()
                    if line.startswith("data:"):
                        records.append(line[5:].strip().replace(key, "[redacted]"))
                    if not line and not records:
                        continue
                    if len(records) >= 12:
                        break
        except (OSError, socket.timeout) as error:
            pod["log_read_status"] = type(error).__name__
        pod["system_log_frames"] = records
    write_json(args.out, results)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
