"""Exercise the published offline component contracts without external assets."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import unittest


HAS_OPTIONALS = all(importlib.util.find_spec(name) is not None
                    for name in ('cv2', 'scipy'))


@unittest.skipUnless(HAS_OPTIONALS, 'M4 offline controls require scipy and OpenCV')
class RectificationM4Contracts(unittest.TestCase):
    def test_independent_component_and_area_contracts(self):
        root = Path(__file__).resolve().parents[1]
        launcher = root/'research/rectification-methodology-m4-20261023/replay_components.py'
        for task in ('qa', 'sampling-tests', 'area-tests', 'identifiability'):
            with self.subTest(task=task):
                completed = subprocess.run(
                    [sys.executable, str(launcher), '--task', task], cwd=root,
                    env=dict(os.environ, OPENBLAS_NUM_THREADS='1',
                             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    capture_output=True, text=True, timeout=120, check=False)
                self.assertEqual(completed.returncode, 0,
                                 completed.stdout+'\n'+completed.stderr)


if __name__ == '__main__':
    unittest.main()
