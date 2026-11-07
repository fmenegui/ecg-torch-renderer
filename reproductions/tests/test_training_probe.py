"""Credential redaction and exact resource scope for read-only observation."""

import unittest

from reproductions.runpod.training_probe import POD_ID, POD_NAME, redact, verify_pod


class TrainingProbeTests(unittest.TestCase):
    def test_logs_keep_scientific_information_without_credentials(self):
        raw = "epoch=7 loss=0.3 token=abc password: xyz api_key=secret Bearer private user-account-key"
        result = redact(raw, "user-account-key")
        for value in ["abc", "xyz", "secret", "private", "user-account-key"]:
            self.assertNotIn(value, result)
        self.assertIn("epoch=7 loss=0.3", result)

    def test_resource_identity_cannot_expand_to_a_similar_named_pod(self):
        verify_pod({"id": POD_ID, "name": POD_NAME})
        for value in [{"id": "unrelated", "name": POD_NAME}, {"id": POD_ID, "name": "another-pod"}]:
            with self.assertRaises(ValueError):
                verify_pod(value)


if __name__ == "__main__":
    unittest.main()
