"""Read-only Runpod access probe, suitable for an authorized Actions secret.

Never prints or writes the credential. Never provisions a resource.
"""

import argparse
import json
import os
import urllib.error
import urllib.request

from reproductions.common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    key = os.getenv("RUNPOD_API_KEY")
    receipt = {"schema": "runpod-control-preflight-v1", "credential_present": bool(key), "authenticated": False, "resources_created": [], "gpu_execution_started": False}
    if not key:
        receipt["status"] = "missing_credential"
    else:
        request = urllib.request.Request("https://api.runpod.io/v2/catalog/gpus", headers={"Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                catalog = json.load(response)
            receipt.update(status="authenticated_read_only", authenticated=True, catalog=catalog)
        except urllib.error.HTTPError as error:
            receipt.update(status="api_rejected", http_status=error.code)
        except (OSError, ValueError) as error:
            receipt.update(status="network_or_response_error", error_type=type(error).__name__)
    write_json(args.out, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "catalog"}))


if __name__ == "__main__":
    main()
