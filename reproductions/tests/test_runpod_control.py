from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from reproductions.runpod.controller import Controller, choice, public_ssh_port, validate_ownership, validate_plan
from reproductions.runpod.admission import first_eligible


class RunpodControlTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"maximum_pod_seconds": 1800, "volume_gb": 30, "gpu_id": "NVIDIA A40", "data_center": "EU-SE-1", "maximum_gpu_price_per_hour": 0.60, "template_id": "torch"}
        self.gpus = [{"gpuId": "NVIDIA A40", "securePricePerHr": 0.59, "dataCenterAvailability": [{"dataCenterId": "EU-SE-1", "stockStatus": "Medium"}]}]
        self.templates = [{"id": "torch", "imageName": "runpod/pytorch:pinned"}]
        self.datacenters = [{"id": "EU-SE-1", "networkVolumeTypes": ["STANDARD"]}]

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
