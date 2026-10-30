from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from reproductions.runpod.controller import Controller, choice, pod_creation_args, public_ssh_port, validate_ownership, validate_plan
from reproductions.runpod.admission import first_eligible


class RunpodControlTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"maximum_pod_seconds": 1800, "volume_gb": 30, "gpu_id": "NVIDIA A40", "data_center": "EU-SE-1", "maximum_gpu_price_per_hour": 0.60, "template_id": "torch"}
        self.gpus = [{"gpuId": "NVIDIA A40", "securePricePerHr": 0.59, "dataCenterAvailability": [{"dataCenterId": "EU-SE-1", "stockStatus": "Medium"}]}]
        self.templates = [{"id": "torch", "imageName": "runpod/pytorch:pinned"}]
        self.datacenters = [{"id": "EU-SE-1", "networkVolumeTypes": ["STANDARD"]}]

    def test_transport_timeout_stops_descendant_side_effects(self):
        controller = Controller.__new__(Controller)
        controller.key = 'test-only-dummy-credential'
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/'late-side-effect'
            child = "from pathlib import Path; import time; time.sleep(1.3); Path(" + repr(str(marker)) + ").write_text('late-action')"
            code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(4)"
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                controller.command([sys.executable, '-c', code], timeout=1.)
            self.assertLess(time.monotonic() - started, 2.)
            time.sleep(.6)
            self.assertFalse(marker.exists(), 'A timed-out transport must not leave a descendant performing late actions')

    def test_deadline_guard_is_armed_before_a_failing_source_transfer(self):
        plan = dict(self.plan, schema='ecg-runpod-pilot-plan-v1', worker_module='reproductions.hybrid.worker',
                    storage_mode='container_disk', volume_gb=0, container_disk_gb=40)
        with tempfile.TemporaryDirectory() as directory:
            controller = Controller.__new__(Controller)
            controller.root = Path(directory)
            controller.plan_path = controller.root/'plan.json'
            controller.plan_path.write_text(json.dumps(plan))
            controller.ledger = {'name': 'ecg-repro-owned'}
            controller.key, controller.cli = 'test-only-dummy-credential', 'unused-cli'
            controller.save = lambda: None
            controller.rest = lambda method, path: {'dataCenters': self.datacenters}
            def call(args):
                if args[0] == 'gpu':
                    return self.gpus
                if args[0] == 'template':
                    return self.templates
                if args[:2] == ['pod', 'create']:
                    return {'id': 'owned-pod', 'costPerHr': .59}
                return []
            controller.call = call
            controller.runtime_connection = lambda identity: {'ip': '192.0.2.1', 'port': 22}
            def command(args, **kwargs):
                if args[:2] == ['git', 'archive']:
                    (controller.root/'source.tar.gz').write_bytes(b'independent-source-fixture')
                return subprocess.CompletedProcess(args, 0, 'fixture-commit', '')
            controller.command = command
            controller.ssh = lambda code, **kwargs: subprocess.CompletedProcess([], 0, json.dumps({'watchdog_armed': True}), '')
            def fail_transfer(*args, **kwargs):
                self.assertTrue(controller.ledger.get('watchdog', {}).get('watchdog_armed'),
                                'The pod must be protected before transferring source files')
                raise RuntimeError('Independent source transfer failure')
            controller.scp = fail_transfer
            with patch('reproductions.runpod.controller.Path.home', return_value=controller.root):
                with self.assertRaisesRegex(RuntimeError, 'Independent source transfer failure'):
                    controller.provision()

    def test_price_and_observed_location_must_both_match(self):
        self.assertEqual(choice(self.plan, self.gpus, self.templates, self.datacenters)["gpu_price_per_hour"], 0.59)
        other = deepcopy(self.gpus)
        other[0]["securePricePerHr"] = 0.61
        with self.assertRaises(ValueError):
            choice(self.plan, other, self.templates, self.datacenters)
        other = deepcopy(self.gpus)
        other[0]["dataCenterAvailability"][0]["dataCenterId"] = "another-dc"
        with self.assertRaises(ValueError):
            choice(self.plan, other, self.templates, self.datacenters)

    def test_gpu_stock_without_storage_cannot_start_training(self):
        with self.assertRaises(ValueError):
            choice(self.plan, self.gpus, self.templates, [{"id": "EU-SE-1", "networkVolumeTypes": []}])

    def test_ephemeral_inference_does_not_require_network_storage_stock(self):
        plan = dict(self.plan, schema="ecg-runpod-pilot-plan-v1", worker_module="reproductions.hybrid.worker",
                    storage_mode="container_disk", volume_gb=0, container_disk_gb=40)
        result = choice(plan, self.gpus, self.templates, [{"id": "EU-SE-1", "networkVolumeTypes": []}])
        args = pod_creation_args(plan, result, "our-pilot")
        self.assertEqual(args[args.index("--volume-in-gb") + 1], "0")
        self.assertNotIn("--network-volume-id", args)
        with self.assertRaises(ValueError):
            pod_creation_args(plan, result, "our-pilot", "borrowed-volume")

    def test_ephemeral_mode_cannot_remove_training_durability_or_expand_the_pilot(self):
        plan = dict(self.plan, schema="ecg-runpod-pilot-plan-v1", worker_module="reproductions.hybrid.worker",
                    storage_mode="container_disk", volume_gb=0, container_disk_gb=40)
        for invalid in [dict(plan, schema="ecg-runpod-full-training-plan-v1", volume_gb=220, maximum_pod_seconds=18000, segments=5),
                        dict(plan, worker_module="reproductions.runpod.worker"), dict(plan, maximum_pod_seconds=1801),
                        dict(plan, volume_gb=30), dict(plan, container_disk_gb=400), dict(plan, storage_mode="unknown")]:
            with self.assertRaises(ValueError):
                validate_plan(invalid)

    def test_training_pod_arguments_keep_the_owned_network_volume(self):
        plan = dict(self.plan, schema="ecg-runpod-full-training-plan-v1", volume_gb=220,
                    maximum_pod_seconds=18000, segments=5, container_disk_gb=40)
        selected = choice(plan, self.gpus, self.templates, self.datacenters)
        args = pod_creation_args(plan, selected, "our-training", "owned-volume")
        self.assertEqual(args[args.index("--network-volume-id") + 1], "owned-volume")
        self.assertNotIn("--volume-in-gb", args)
        with self.assertRaises(ValueError):
            pod_creation_args(plan, selected, "our-training")

    def test_budget_and_ownership_cannot_silently_expand(self):
        other = dict(self.plan, maximum_pod_seconds=3600)
        with self.assertRaises(ValueError):
            choice(other, self.gpus, self.templates, self.datacenters)
        validate_ownership({"run_id": "123-1", "name": "ecg-repro-123-1"}, "123-1")
        with self.assertRaises(ValueError):
            validate_ownership({"run_id": "another-run", "name": "someone-elses-pod"}, "123-1")

    def test_ssh_does_not_use_http_or_private_port_mappings(self):
        ports = [
            {"ip": "100.65.1.1", "isIpPublic": False, "privatePort": 22, "publicPort": 123, "type": "tcp"},
            {"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 19123, "publicPort": 456, "type": "http"},
            {"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 22, "publicPort": 789, "type": "tcp"},
        ]
        self.assertIsNone(public_ssh_port({"runtime": None}))
        self.assertEqual(public_ssh_port({"runtime": {"ports": ports}}), {"ip": "1.2.3.4", "port": 789})

    def test_full_training_envelope_needs_explicit_schema_and_caps(self):
        full = dict(self.plan, schema="ecg-runpod-full-training-plan-v1", volume_gb=220, maximum_pod_seconds=18000, segments=5)
        validate_plan(full)
        for invalid in [dict(full, segments=6), dict(full, maximum_pod_seconds=18001), dict(full, maximum_gpu_price_per_hour=1.01), dict(full, volume_gb=500), dict(full, schema="pilot")]:
            with self.assertRaises(ValueError):
                validate_plan(invalid)

    def test_source_archive_cannot_include_data_or_secret_paths(self):
        validate_plan(dict(self.plan, source_paths=["reproductions", "ecg_gpu", "examples"]))
        for paths in [[], ["reproductions", "private-data"], ["."], ["reproductions", "reproductions"]]:
            with self.assertRaises(ValueError):
                validate_plan(dict(self.plan, source_paths=paths))

    def test_a_cheaper_gpu_below_the_required_memory_is_rejected(self):
        plan = dict(self.plan, minimum_gpu_memory_gb=48)
        small = deepcopy(self.gpus)
        small[0]["memoryInGb"] = 24
        with self.assertRaises(ValueError):
            choice(plan, small, self.templates, self.datacenters)

    def test_resume_cannot_take_another_volume_or_change_its_location(self):
        full = dict(self.plan, schema="ecg-runpod-full-training-plan-v1", volume_gb=220, maximum_pod_seconds=18000, segments=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(full))
            for volume in [{"name": "another-run", "size": 220, "dataCenter": "EU-SE-1"}, {"name": "ecg-repro-123-1", "size": 220, "dataCenter": "another-dc"}]:
                controller = Controller.__new__(Controller)
                controller.plan_path = path
                controller.ledger = {"name": "ecg-repro-123-1", "volume_id": "ours", "pod_id": "previous", "pod_absent_verified": True, "selection": {"data_center": "EU-SE-1"}}
                controller.rest = lambda method, endpoint: {"dataCenters": self.datacenters} if endpoint == "/catalog/datacenters" else volume
                with self.assertRaises(ValueError):
                    controller.provision()

    def test_finished_training_does_not_allocate_another_gpu(self):
        full = dict(self.plan, schema="ecg-runpod-full-training-plan-v1", volume_gb=220, maximum_pod_seconds=18000, segments=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(full))
            controller = Controller.__new__(Controller)
            controller.plan_path = path
            controller.ledger = {"full_training_completed": True}
            controller.save = lambda: None
            controller.provision()
            controller.execute()
            self.assertEqual(controller.ledger["status"], "already_completed")

    def test_duplicate_pushes_and_doc_commits_do_not_restart_paid_training(self):
        runs = [{"id": 3, "head_sha": "docs", "conclusion": None}, {"id": 2, "head_sha": "original", "conclusion": "success"}, {"id": 1, "head_sha": "original", "conclusion": "cancelled"}]
        self.assertEqual(first_eligible(runs, "same-plan", lambda commit: "same-plan"), 2)
        self.assertEqual(first_eligible(runs, "new-plan", lambda commit: "new-plan" if commit == "docs" else "same-plan"), 3)
        with self.assertRaises(ValueError):
            first_eligible([], "same-plan", lambda commit: "same-plan")


if __name__ == "__main__":
    unittest.main()
