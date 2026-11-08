"""Read-only catalog and prior public-data checkpoint volume discovery."""

import json
import os
from pathlib import Path
import urllib.request


def main():
    token = os.environ["RUNPOD_API_KEY"]
    out = Path("build/grid-volume-probe")
    out.mkdir(parents=True, exist_ok=True)

    def read(path):
        request = urllib.request.Request("https://api.runpod.io/v2" + path, headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    result = {"schema": "grid-checkpoint-recovery-probe-v1", "mutations": []}
    volume = read("/network-volumes/wxt8cq1hh1")
    result["existing_volume"] = {k: volume[k] for k in ["id", "name", "size", "dataCenter", "type"] if k in volume}
    result["cpu_catalog"] = read("/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=2&dataCenterIds=" + volume["dataCenter"])
    (out / "catalog.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
