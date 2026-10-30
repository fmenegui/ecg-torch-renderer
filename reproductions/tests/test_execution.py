import os
import subprocess
import sys
import tempfile
import unittest

from reproductions.common import run_bounded
from reproductions.runpod_job import preflight


class ExecutionTests(unittest.TestCase):
    def test_success_return_code_is_kept(self):
        with tempfile.TemporaryFile() as log:
            self.assertEqual(run_bounded([sys.executable, "-c", "raise SystemExit(3)"], 10, log), 3)

    def test_timeout_is_a_failure(self):
        with tempfile.TemporaryFile() as log:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded([sys.executable, "-c", "import time\ntime.sleep(30)"], 0.2, log)

    def test_preflight_never_returns_a_credential(self):
        old = os.environ.get("RUNPOD_API_KEY")
        os.environ["RUNPOD_API_KEY"] = "test-secret-not-a-real-key"
        try:
            result = preflight()
            self.assertTrue(result["api_key_present"])
            self.assertNotIn("test-secret", str(result))
        finally:
            if old is None:
                os.environ.pop("RUNPOD_API_KEY")
            else:
                os.environ["RUNPOD_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
