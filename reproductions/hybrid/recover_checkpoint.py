"""Copy one hash-pinned checkpoint from a borrowed volume, then remove our pod.

The borrowed volume is never created, deleted, resized or used as a workdir.
The experiment receives no account credential. A separate deadline watchdog
holds the credential in memory only and can delete only this execution's pod.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time

from reproductions.runpod.controller import Controller

VOLUME = "wxt8cq1hh1"
CHECKPOINT_SHA256 = "8115e99c3517dc0b99974b9e64744aab9eb961fd0da06f1e84a65a1e45c3dbf8"


def choose_recovery(gpus, templates, volume):
    if (volume.get("id") != VOLUME or volume.get("dataCenter") != "CA-MTL-1"
            or volume.get("size") != 50 or volume.get("name") != "ecg-diverse-offline-bank"):
        raise ValueError("Borrowed checkpoint volume identity changed")
    template = next(t for t in templates if t["id"] == "runpod-torch-v280")
    if not template["imageName"].startswith("runpod/pytorch:"):
        raise ValueError("Expected a catalog-observed official PyTorch image")
    for gpu in gpus:
        price = gpu.get("securePricePerHr")
        if gpu.get("gpuId") != "NVIDIA A40" or price is None or not 0 < price <= .60:
            continue
        if any(a.get("dataCenterId") == volume["dataCenter"] and a.get("stockStatus", "").lower()
               in ["low", "medium", "high"] for a in gpu.get("dataCenterAvailability", [])):
            return {"gpu": gpu, "template": template, "data_center": volume["dataCenter"],
                    "gpu_price_per_hour": price, "maximum_pod_seconds": 900,
                    "gpu_cost_cap_usd": .15, "borrowed_volume_mutations": []}
    raise ValueError("No in-stock co-located A40 at or below USD 0.60/hour")


class Recovery(Controller):
    def provision(self):
        if self.ledger.get("pod_id") or "volume_id" in self.ledger:
            raise ValueError("Recovery is single-use and must never own the borrowed volume")
        volume = self.rest("GET", "/network-volumes/" + VOLUME)
        selected = choose_recovery(self.call(["gpu", "list", "--include-unavailable"]),
            self.call(["template", "search", "pytorch", "--type", "official", "--limit", "20"]), volume)
        self.ledger.update(selection=selected, borrowed_volume_id=VOLUME, status="selected")
        self.save()
        print(json.dumps(selected), flush=True)
        key = self.root / "recovery-ssh-key"
        self.command(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", self.ledger["name"], "-f", str(key)])
        self.ledger["private_key"] = str(key)
        self.save()
        self.call(["ssh", "add-key", "--key-file", str(key) + ".pub"])
        self.ledger["ssh_key_added"] = True
        self.ledger["deadline"] = time.time() + 900
        self.save()
        pod = self.call(["pod", "create", "--name", self.ledger["name"], "--image", selected["template"]["imageName"],
            "--gpu-id", "NVIDIA A40", "--gpu-count", "1", "--cloud-type", "SECURE", "--data-center-ids", "CA-MTL-1",
            "--network-volume-id", VOLUME, "--volume-mount-path", "/archive", "--container-disk-in-gb", "20",
            "--ports", "22/tcp", "--ssh", "--min-cuda-version", "12.8"])
        self.ledger.update(pod_id=pod["id"], status="booting")
        actual_price = pod.get("costPerHr")
        self.ledger["created_pod_price_per_hour"] = actual_price
        self.save()
        if not isinstance(actual_price, (int, float)) or not 0 < actual_price <= .60:
            raise ValueError("Created recovery pod price is missing or exceeds USD 0.60/hour")
        self.save()
        until = min(time.time() + 480, self.ledger["deadline"] - 180)
        while time.time() < until:
            connection = self.runtime_connection(pod["id"])
            if connection:
                self.ledger["connection"] = connection
                if self.ssh("true", check=False).returncode == 0:
                    break
            time.sleep(10)
        else:
            raise TimeoutError("Checkpoint recovery did not become SSH-ready")
        self.scp(Path(__file__).parents[1] / "runpod/watchdog.py", "/tmp/ecg-watchdog.py")
        armed = self.ssh("python3 /tmp/ecg-watchdog.py --pod-id " + shlex.quote(pod["id"])
                         + " --deadline " + str(self.ledger["deadline"]), input=self.key + "\n")
        self.ledger["watchdog"] = json.loads(armed.stdout)
        self.ledger["status"] = "ready"
        self.save()

    def execute(self):
        if self.ledger.get("status") != "ready" or not self.ledger.get("watchdog", {}).get("watchdog_armed"):
            raise ValueError("An independent bounded watchdog is required")
        script = '''from pathlib import Path
import hashlib, json, shutil
expected = "''' + CHECKPOINT_SHA256 + '''"
found = []
for path in Path('/archive').rglob('joint-1973.pt'):
    if path.is_file() and not path.is_symlink() and path.stat().st_size <= 64*1024**2:
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            found.append(path)
if not found:
    raise RuntimeError('No hash-verified selected checkpoint was found')
found.sort()
shutil.copyfile(found[0], '/tmp/grid-checkpoint.pt')
print(json.dumps({'sha256': expected, 'source': str(found[0]), 'identical_hash_matches': len(found), 'borrowed_volume_write': False}))
'''
        receipt = self.ssh("python3 -", input=script, timeout=180)
        self.ledger["checkpoint_copy"] = json.loads(receipt.stdout)
        destination = self.root / "grid-checkpoint.pt"
        self.scp(destination, "/tmp/grid-checkpoint.pt", download=True)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Transferred checkpoint hash mismatch")
        self.ledger.update(checkpoint_sha256=CHECKPOINT_SHA256, results_copied_and_verified=True, status="completed")
        self.save()

    def cleanup(self, delete_volume=False):
        if delete_volume or "volume_id" in self.ledger:
            raise ValueError("Checkpoint recovery cannot delete any volume")
        # Parent cleanup recovers our uniquely named pod and SSH key only.
        super().cleanup(delete_volume=False)
        if "volume_id" in self.ledger:
            raise ValueError("An unexpected owned volume requires separate investigation")
        volume = self.rest("GET", "/network-volumes/" + VOLUME)
        if volume.get("size") != 50 or volume.get("dataCenter") != "CA-MTL-1":
            raise ValueError("Borrowed volume read-back changed")
        self.ledger["borrowed_volume_present_verified"] = True
        self.save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cli", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("action", choices=["provision", "execute", "cleanup"])
    args = p.parse_args()
    controller = Recovery(args.cli, args.out)
    try:
        getattr(controller, args.action)()
    except BaseException as error:
        controller.ledger["error"] = str(error).replace(controller.key, "[redacted]")
        controller.save()
        raise


if __name__ == "__main__":
    main()
