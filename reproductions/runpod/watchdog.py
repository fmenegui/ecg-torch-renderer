"""Independent one-pod deadline guard. Credential arrives through encrypted stdin.

The child retains the key only in memory, does not expose it to the experiment
environment or output files, and can delete only its supplied pod at deadline.
The Actions controller independently deletes and verifies this same owned pod.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id):
        raise ValueError("Invalid pod identity")
    key = sys.stdin.readline().strip()
    if not key or not 1 < args.deadline - time.time() <= 18000:
        raise ValueError("A credential and bounded future deadline are required")
    child = os.fork()
    if child:
        print(json.dumps({"watchdog_armed": True, "pod_id": args.pod_id, "deadline": args.deadline}), flush=True)
        return
    os.setsid()
    fd = os.open(os.devnull, os.O_RDWR)
    for target in [0, 1, 2]:
        os.dup2(fd, target)
    if fd > 2:
        os.close(fd)
    while time.time() < args.deadline:
        time.sleep(min(30, args.deadline - time.time()))
    for _ in range(12):
        request = urllib.request.Request("https://api.runpod.io/v2/pods/" + args.pod_id, method="DELETE", headers={"Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(request, timeout=20):
                break
        except urllib.error.HTTPError as error:
            if error.code == 404:
                break
        except OSError:
            pass
        time.sleep(5)
    os._exit(0)


if __name__ == "__main__":
    main()
