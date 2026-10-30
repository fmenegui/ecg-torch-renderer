"""Read only this reproduction's resources, never credentials or other pods."""

import argparse
import json
import os
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
    write_json(args.out, results)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
