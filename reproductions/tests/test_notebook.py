import ast
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from reproductions.evaluate import kaggle_timeline
from reproductions.kaggle.replay import adapt, notebook_code


def document(code, version=6):
    return {"metadata": {"currentVersionNumber": version}, "blob": {"source": json.dumps({"cells": [{"cell_type": "code", "source": code.splitlines(True)}]})}}


class NotebookTests(unittest.TestCase):
    def test_reviewed_demo_failure_cannot_abort_unchanged_numerical_results(self):
        source = """def run_stage0(n=0, gpu_id=0):
    result = numerical_step()
    if n < 10 and gpu_id == 0:
        overlay = draw_results_stage0(rotated, keypoint)
        plt.imshow(overlay)
        plt.show()
    return result
A = run_stage0_parallel()
B = run_stage1_parallel(prev_fail_ids=A)
C = run_stage2_parallel(prev_fail_ids=B)
shutil.rmtree(OUT_DIR)
"""
        branch = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If))
        digest = hashlib.sha256(ast.dump(branch, include_attributes=False).encode()).hexdigest()
        with patch('reproductions.kaggle.replay.REVIEWED_DEMO_BLOCKS', {digest}):
            code, receipt = adapt(document(source), {})
        function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'run_stage0')
        expected = np.array([.1, .3, .2])
        namespace = {'numerical_step': lambda: expected}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'reviewed-demo-control', 'exec'), namespace)
        self.assertIs(namespace['run_stage0'](), expected)
        self.assertEqual(receipt['removed_demo_block_sha256'], [digest])

    def test_unreviewed_demo_with_numerical_work_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unreviewed numerical'):
            adapt(document('if n < 10 and gpu_id == 0:\n    signal = change_signal(signal)\n'), {})

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
        native = {x.name: ast.dump(x, include_attributes=False) for x in ast.parse(code).body if isinstance(x, ast.FunctionDef)}
        transformed = {x.name: ast.dump(x, include_attributes=False) for x in tree.body if isinstance(x, ast.FunctionDef)}
        for name, definition in native.items():
            self.assertEqual(transformed[name], definition)
        self.assertEqual(len(receipt["notebook_source_sha256"]), 64)

    def test_serial_stages_preserve_previous_native_failures(self):
        source = "A = run_stage0_parallel()\nB = run_stage1_parallel(prev_fail_ids=A)\nC = run_stage2_parallel(prev_fail_ids=B)\nshutil.rmtree(OUT_DIR)\n"
        code, _ = adapt(document(source), {})
        definitions = [x for x in ast.parse(code).body if isinstance(x, ast.FunctionDef) and x.name.startswith("_ecg_serial_")]
        namespace = {"run_stage0": lambda: ["stage0-failure"], "run_stage1": lambda prev_fail_ids: ["stage1-failure"], "run_stage2": lambda prev_fail_ids: []}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "serial-failure-control", "exec"), namespace)
        a = namespace["_ecg_serial_run_stage0"]()
        b = namespace["_ecg_serial_run_stage1"](prev_fail_ids=a)
        c = namespace["_ecg_serial_run_stage2"](prev_fail_ids=b)
        self.assertEqual(c, ["stage1-failure", "stage0-failure"])

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
