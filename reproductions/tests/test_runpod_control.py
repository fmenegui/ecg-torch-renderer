from copy import deepcopy
import unittest

from reproductions.runpod.controller import choice, validate_ownership


class RunpodControlTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"maximum_pod_seconds": 1800, "volume_gb": 30, "gpu_id": "NVIDIA A40", "data_center": "EU-SE-1", "maximum_gpu_price_per_hour": 0.60, "template_id": "torch"}
        self.gpus = [{"gpuId": "NVIDIA A40", "securePricePerHr": 0.59, "dataCenterAvailability": [{"dataCenterId": "EU-SE-1", "stockStatus": "Medium"}]}]
        self.templates = [{"id": "torch", "imageName": "runpod/pytorch:pinned"}]

    def test_price_and_observed_location_must_both_match(self):
        self.assertEqual(choice(self.plan, self.gpus, self.templates)["gpu_price_per_hour"], 0.59)
        other = deepcopy(self.gpus)
        other[0]["securePricePerHr"] = 0.61
        with self.assertRaises(ValueError):
            choice(self.plan, other, self.templates)
        other = deepcopy(self.gpus)
        other[0]["dataCenterAvailability"][0]["dataCenterId"] = "another-dc"
        with self.assertRaises(ValueError):
            choice(self.plan, other, self.templates)

    def test_budget_and_ownership_cannot_silently_expand(self):
        other = dict(self.plan, maximum_pod_seconds=3600)
        with self.assertRaises(ValueError):
            choice(other, self.gpus, self.templates)
        validate_ownership({"run_id": "123-1", "name": "ecg-repro-123-1"}, "123-1")
        with self.assertRaises(ValueError):
            validate_ownership({"run_id": "another-run", "name": "someone-elses-pod"}, "123-1")


if __name__ == "__main__":
    unittest.main()
