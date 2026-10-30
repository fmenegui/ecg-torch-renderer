"""One bounded Actions-controlled GPU pilot with owned-resource cleanup.

The control-plane credential stays in Actions and an in-memory deadline guard
on the owned pod. It is absent from experiment env, source and artifact files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request

from reproductions.common import write_json


def validate_plan(plan):
    full = plan.get("schema") == "ecg-runpod-full-training-plan-v1"
    maximum = 18000 if full else 1800
    volume = 220 if full else 30
    if not 0 < plan["maximum_pod_seconds"] <= maximum or plan["volume_gb"] != volume:
        raise ValueError("Execution exceeds its bounded resource envelope")
    if full and (plan.get("segments") != 5 or plan["maximum_gpu_price_per_hour"] > 1.00):
        raise ValueError("Full training must retain its five-segment price envelope")


def choice(plan, gpus, templates, datacenters):
    validate_plan(plan)
    template = next(t for t in templates if t["id"] == plan["template_id"])
    if not template["imageName"].startswith("runpod/pytorch:"):
        raise ValueError("Expected the observed official PyTorch image")
    for identity in plan.get("gpu_candidates", [plan["gpu_id"]]):
        for gpu in gpus:
            if gpu["gpuId"] != identity:
                continue
            if gpu.get("memoryInGb", 0) < plan.get("minimum_gpu_memory_gb", 0):
                continue
            price = gpu.get("securePricePerHr")
            if price is None or not 0 < price <= plan["maximum_gpu_price_per_hour"]:
                continue
            for availability in gpu.get("dataCenterAvailability", []):
                if availability.get("stockStatus", "").lower() not in ["low", "medium", "high"]:
                    continue
                dc = next((d for d in datacenters if d["id"] == availability["dataCenterId"]), None)
                if dc and "STANDARD" in dc.get("networkVolumeTypes", []):
                    return {"gpu": gpu, "template": template, "data_center": dc["id"], "data_center_metadata": dc, "gpu_price_per_hour": price, "read_at": datetime.now(timezone.utc).isoformat()}
    raise ValueError("No observed candidate has both acceptable price and standard network storage in the same data center")


def validate_ownership(ledger, run_id):
    if ledger.get("run_id") != run_id or ledger.get("name") != "ecg-repro-" + run_id:
        raise ValueError("Refusing to mutate a resource outside this execution")


def public_ssh_port(pod):
    for port in (pod.get("runtime") or {}).get("ports", []):
        if port.get("privatePort") == 22 and port.get("isIpPublic") and port.get("type", "tcp").lower() == "tcp":
            return {"ip": port["ip"], "port": int(port["publicPort"])}
    return None


class Controller:
    def __init__(self, cli, root, plan_path=None):
        self.cli, self.root = cli, Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = os.environ["RUNPOD_API_KEY"]
        self.plan_path = Path(plan_path) if plan_path else Path(__file__).with_name("plan.json")
        self.run_id = os.environ["GITHUB_RUN_ID"] + "-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
        self.path = self.root / "control.json"
        self.ledger = json.loads(self.path.read_text()) if self.path.exists() else {"run_id": self.run_id, "name": "ecg-repro-" + self.run_id}
        validate_ownership(self.ledger, self.run_id)

    def save(self):
        write_json(self.path, self.ledger)
        print(json.dumps({k: self.ledger[k] for k in ["name", "pod_id", "volume_id", "status", "deadline", "pod_absent_verified", "volume_absent_verified"] if k in self.ledger}), flush=True)

    def command(self, args, timeout=120, check=True, input=None):
        result = subprocess.run(args, input=input, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(result.stderr.replace(self.key, "[redacted]")[:2000])
        return result

    def call(self, args):
        result = self.command([self.cli, *args])
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def rest(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request("https://api.runpod.io/v2" + path, data=data, method=method, headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            message = error.read(3000).decode(errors="replace").replace(self.key, "[redacted]")
            raise RuntimeError(f"Runpod HTTP {error.code}: {message}") from error

    def runtime_connection(self, pod_id):
        # Observed API discrepancy: REST v2 runtime=null while GraphQL already
        # exposes a live mapped SSH port. Keep this corroborated fallback explicit.
        query = "query { myself { pods { id runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"
        request = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": query}).encode(), headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result.get("errors"):
            raise RuntimeError("Runpod GraphQL runtime query was rejected")
        pod = next((p for p in result["data"]["myself"]["pods"] if p["id"] == pod_id), {})
        return public_ssh_port(pod)

    def ssh(self, command, timeout=60, input=None, check=True):
        c = self.ledger["connection"]
        return self.command(["ssh", "-i", self.ledger["private_key"], "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", "-p", str(c["port"]), "root@" + c["ip"], command], timeout, check, input)

    def scp(self, local, remote, download=False, timeout=120):
        c = self.ledger["connection"]
        remote_path = "root@" + c["ip"] + ":" + remote
        pair = [remote_path, str(local)] if download else [str(local), remote_path]
        self.command(["scp", "-i", self.ledger["private_key"], "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", str(c["port"]), *pair], timeout=timeout)

    def provision(self):
        plan = json.loads(self.plan_path.read_text())
        validate_plan(plan)
        full = plan.get("schema") == "ecg-runpod-full-training-plan-v1"
        if full and self.ledger.get("full_training_completed"):
            self.ledger["status"] = "already_completed"
            self.save()
            return
        if full and self.ledger.get("full_training_halted"):
            self.ledger["status"] = "already_halted"
            self.save()
            return
        if self.ledger.get("pod_id") and not (full and self.ledger.get("pod_absent_verified")):
            raise ValueError("This execution already provisioned resources")
        if self.ledger.get("volume_id") and not full:
            raise ValueError("Pilot already owns a volume")
        existing_volume = None
        raw_catalog = self.rest("GET", "/catalog/datacenters")
        self.ledger["catalog_storage_support"] = [{"id": d["id"], "networkVolumeTypes": d.get("networkVolumeTypes", [])} for d in raw_catalog["dataCenters"]]
        if self.ledger.get("volume_id"):
            existing_volume = self.rest("GET", "/network-volumes/" + self.ledger["volume_id"])
            if existing_volume.get("name") != self.ledger["name"] or existing_volume.get("size") != plan["volume_gb"] or existing_volume.get("dataCenter") != self.ledger.get("selection", {}).get("data_center"):
                raise ValueError("Persisted training volume ownership or capacity changed")
            raw_catalog["dataCenters"] = [d for d in raw_catalog["dataCenters"] if d["id"] == existing_volume["dataCenter"]]
        selected = choice(plan, self.call(["gpu", "list", "--include-unavailable"]), self.call(["template", "search", "pytorch", "--type", "official", "--limit", "20"]), raw_catalog["dataCenters"])
        selected["storage_price_reference"] = {"type": "STANDARD", "usd_per_gb_month": 0.07, "volume_gb": plan["volume_gb"], "source": "https://www.runpod.io/pricing"}
        self.ledger.update(selection=selected, status="selected", plan=plan)
        if full:
            segments = self.ledger.setdefault("segments", [])
            if len(segments) >= plan["segments"]:
                raise ValueError("All bounded training segments have been used")
            segments.append({"index": len(segments) + 1, "selected_at": selected["read_at"]})
            for field in ["pod_id", "connection", "watchdog", "worker_returncode", "error", "pod_absent_verified", "results_copied_and_verified"]:
                self.ledger.pop(field, None)
        self.save()
        key = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
        key.parent.mkdir(parents=True, exist_ok=True)
        self.command(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", self.ledger["name"], "-f", str(key)])
        self.ledger["private_key"] = str(key)
        self.save()
        self.call(["ssh", "list-keys"])
        self.call(["ssh", "add-key", "--key-file", str(key) + ".pub"])
        self.ledger["ssh_key_added"] = True
        self.save()
        volume = existing_volume or self.rest("POST", "/network-volumes", {"name": self.ledger["name"], "size": plan["volume_gb"], "dataCenter": selected["data_center"], "type": "STANDARD"})
        self.ledger["volume_id"] = volume["id"]
        self.ledger["deadline"] = time.time() + plan["maximum_pod_seconds"]
        self.save()
        pod = self.call(["pod", "create", "--name", self.ledger["name"], "--image", selected["template"]["imageName"], "--gpu-id", selected["gpu"]["gpuId"], "--gpu-count", "1", "--cloud-type", "SECURE", "--data-center-ids", selected["data_center"], "--network-volume-id", volume["id"], "--volume-mount-path", "/workspace", "--container-disk-in-gb", str(plan["container_disk_gb"]), "--ports", "22/tcp", "--ssh", "--min-cuda-version", "12.8"])
        self.ledger.update(pod_id=pod["id"], status="booting")
        if full:
            self.ledger["segments"][-1].update(pod_id=pod["id"], deadline=self.ledger["deadline"])
        self.save()
        until = min(time.time() + 540, self.ledger["deadline"] - 120)
        while time.time() < until:
            conn = self.runtime_connection(pod["id"])
            if conn:
                self.ledger["connection"] = conn
                self.ledger["runtime_connection_source"] = "GraphQL live SSH mapping, REST v2 telemetry unavailable"
                if self.ssh("true", check=False).returncode == 0:
                    break
            time.sleep(10)
        else:
            raise TimeoutError("Pod did not become SSH-ready within the boot budget")
        archive = self.root / "source.tar.gz"
        self.command(["git", "archive", "--format=tar.gz", "-o", str(archive), "HEAD", "reproductions"])
        self.ssh("mkdir -p /workspace/ecg-code /workspace/outputs")
        self.scp(archive, "/workspace/source.tar.gz")
        self.ssh("tar -xzf /workspace/source.tar.gz -C /workspace/ecg-code")
        execution = self.root / "execution.json"
        write_json(execution, {"deadline": self.ledger["deadline"], "plan": plan})
        self.scp(execution, "/workspace/execution.json")
        armed = self.ssh("python3 /workspace/ecg-code/reproductions/runpod/watchdog.py --pod-id " + shlex.quote(pod["id"]) + " --deadline " + str(self.ledger["deadline"]), input=self.key + "\n")
        self.ledger["watchdog"] = json.loads(armed.stdout)
        self.ledger["status"] = "ready"
        self.save()

    def execute(self):
        if self.ledger.get("status") in ["already_completed", "already_halted"]:
            return
        if self.ledger.get("status") != "ready" or not self.ledger.get("watchdog", {}).get("watchdog_armed"):
            raise ValueError("Owned pod and independent watchdog must be ready")
        self.ledger["status"] = "running"
        self.save()
        self.ssh("rm -f /workspace/outputs/worker.exit")
        self.ssh("setsid bash -c 'bash /workspace/ecg-code/reproductions/runpod/launch.sh > /workspace/outputs/worker.log 2>&1; result=$?; echo $result > /workspace/outputs/worker.exit' </dev/null >/dev/null 2>&1 &")
        last_progress = 0
        while time.time() < self.ledger["deadline"] - 120:
            result = self.ssh("if test -f /workspace/outputs/worker.exit; then cat /workspace/outputs/worker.exit; else echo RUNNING; fi", check=False)
            if result.returncode == 0 and result.stdout.strip() != "RUNNING":
                self.ledger["worker_returncode"] = int(result.stdout.strip())
                break
            if self.ledger["plan"].get("schema") == "ecg-runpod-full-training-plan-v1" and time.time() - last_progress >= 180:
                progress = self.ssh("find /workspace/outputs/full-training -name receipt.json -maxdepth 3 -exec cat {} + 2>/dev/null", check=False)
                if progress.stdout:
                    print(progress.stdout[-10000:], flush=True)
                last_progress = time.time()
            time.sleep(20)
        self.ssh("tail -50 /workspace/outputs/worker.log", check=False)
        if self.ledger["plan"].get("schema") == "ecg-runpod-full-training-plan-v1":
            result = self.ssh("if test -f /workspace/outputs/full-training/receipt.json; then cat /workspace/outputs/full-training/receipt.json; else echo '{}'; fi", check=False)
            if result.returncode == 0:
                receipt = json.loads(result.stdout)
                self.ledger["full_training_completed"] = receipt.get("status") == "completed_full_training"
                self.ledger["full_training_halted"] = receipt.get("status") == "failed"
                self.ledger["training_progress"] = receipt
        self.collect()
        self.ledger["status"] = "completed" if self.ledger.get("worker_returncode") == 0 else "failed_or_timeout"
        self.save()
        if self.ledger["status"] != "completed":
            raise RuntimeError("GPU pilot did not complete, inspect preserved logs")

    def collect(self):
        if not self.ledger.get("connection"):
            return
        self.ssh("tar -czf /workspace/results.tar.gz -C /workspace/outputs .", timeout=120)
        expected = self.ssh("sha256sum /workspace/results.tar.gz").stdout.split()[0]
        destination = self.root / "results.tar.gz"
        self.scp(destination, "/workspace/results.tar.gz", download=True, timeout=180)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError("Results transfer hash mismatch")
        self.ledger.update(results_sha256=expected, results_copied_and_verified=True)
        self.save()

    def cleanup(self, delete_volume=False):
        # Recover an ambiguous create only by this execution's unique name.
        for resource, field in [("pod", "pod_id"), ("network-volume", "volume_id")]:
            if field not in self.ledger:
                rows = self.call([resource, "list"])
                found = [r for r in rows if r.get("name") == self.ledger["name"]]
                if len(found) == 1:
                    self.ledger[field] = found[0]["id"]
        if self.ledger.get("pod_id"):
            self.command([self.cli, "pod", "delete", self.ledger["pod_id"]], check=False)
            check = self.command([self.cli, "pod", "get", self.ledger["pod_id"]], check=False)
            self.ledger["pod_absent_verified"] = check.returncode != 0 and '"not_found"' in check.stderr
            if not self.ledger["pod_absent_verified"]:
                raise RuntimeError("Pod deletion not externally verified")
        if self.ledger.get("ssh_key_added"):
            self.call(["ssh", "remove-key", "--name", self.ledger["name"]])
            self.ledger["ssh_key_added"] = False
        if delete_volume and self.ledger.get("volume_id"):
            if self.ledger.get("status") == "running" and not self.ledger.get("results_copied_and_verified"):
                raise RuntimeError("Retaining volume because experiment outputs were not copied")
            self.command([self.cli, "network-volume", "delete", self.ledger["volume_id"]], check=False)
            check = self.command([self.cli, "network-volume", "get", self.ledger["volume_id"]], check=False)
            self.ledger["volume_absent_verified"] = check.returncode != 0 and '"not_found"' in check.stderr
        self.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plan", help="Explicit separately bounded execution plan")
    parser.add_argument("action", choices=["provision", "execute", "cleanup", "delete-volume"])
    args = parser.parse_args()
    controller = Controller(args.cli, args.out, args.plan)
    try:
        if args.action == "provision":
            controller.provision()
        elif args.action == "execute":
            controller.execute()
        else:
            controller.cleanup(delete_volume=args.action == "delete-volume")
    except BaseException as error:
        controller.ledger["error"] = str(error).replace(controller.key, "[redacted]")
        controller.save()
        raise


if __name__ == "__main__":
    main()
