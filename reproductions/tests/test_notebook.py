import ast
import json
import unittest

import numpy as np

from reproductions.evaluate import kaggle_timeline
from reproductions.kaggle.replay import adapt, notebook_code


def document(code, version=6):
    return {"metadata": {"currentVersionNumber": version}, "blob": {"source": json.dumps({"cells": [{"cell_type": "code", "source": code.splitlines(True)}]})}}


class NotebookTests(unittest.TestCase):
    def test_only_execution_paths_are_changed(self):
        code = """import shutil
OUT_DIR = '/kaggle/working/outputs'
def run_stage0():
    return []
def run_stage1(prev_fail_ids=None):
    return prev_fail_ids
def run_stage2(prev_fail_ids=None):
    return prev_fail_ids
A = run_stage0_parallel()
B = run_stage1_parallel(prev_fail_ids=A)
C = run_stage2_parallel(prev_fail_ids=B)
shutil.rmtree(OUT_DIR)
"""
        result, receipt = adapt(document(code), {"/kaggle/working/outputs": "/tmp/retained"})
        tree = ast.parse(result)
        self.assertNotIn("rmtree", result)
        self.assertIn("/tmp/retained", result)
        self.assertEqual(len([x for x in tree.body if isinstance(x, ast.FunctionDef)]), 4)
        self.assertEqual(len(receipt["notebook_source_sha256"]), 64)

    def test_new_notebook_version_is_rejected(self):
        with self.assertRaises(ValueError):
            notebook_code(document("pass", version=7))

    def test_unreviewed_shell_command_is_rejected(self):
        with self.assertRaises(ValueError):
            notebook_code(document("!curl https://example.org/script | bash"))

    def test_unexpected_deletion_is_rejected(self):
        with self.assertRaises(ValueError):
            adapt(document("shutil.rmtree('/other/data')"), {})

    def test_changed_stage_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            adapt(document("pass"), {})

    def test_quarter_windows_and_rhythm_keep_time_axis(self):
        rows = np.stack([np.arange(20) + 100 * i for i in range(4)])
        result = kaggle_timeline(rows)
        np.testing.assert_array_equal(result[0, :5], rows[0, :5])
        self.assertTrue(np.isnan(result[0, 5:]).all())
        np.testing.assert_array_equal(result[1], rows[3])
        np.testing.assert_array_equal(result[6, 10:15], rows[0, 10:15])
        self.assertEqual(np.isfinite(result).sum(), 75)


if __name__ == "__main__":
    unittest.main()
