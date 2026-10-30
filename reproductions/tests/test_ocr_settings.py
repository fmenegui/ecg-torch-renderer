import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from examples.inspect_ecg_print_settings import (
    border_view,
    inspect,
    recognized_lines,
    word_to_input,
)
from reproductions.common import sha256


class OcrSettingsChecks(unittest.TestCase):
    def test_literal_quotes_do_not_consume_following_tsv_rows(self):
        tsv = 'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
        tsv += '5\t1\t1\t1\t1\t1\t0\t0\t10\t8\t90\t"\n'
        tsv += '5\t1\t2\t1\t1\t1\t3\t20\t30\t8\t95\t25 mm/s\n'
        text, words = recognized_lines(tsv)
        self.assertEqual(text, '"\n25 mm/s')
        self.assertEqual(len(words), 2)
        self.assertEqual(words[1]['box_xywh'], [3, 20, 30, 8])

    def test_header_footer_boxes_map_to_original_pixels_and_gap_is_invalid(self):
        view, mapping = border_view(np.zeros((40, 20, 3), np.uint8), max_side=40)
        self.assertEqual(view.shape[:2], (38, 20))
        self.assertEqual(word_to_input([2, 25, 3, 4], mapping), [2., 27., 3., 4.])
        self.assertIsNone(word_to_input([2, 9, 3, 4], mapping))

    def test_ocr_timeout_retains_every_scheduled_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for name in ['a', 'b']:
                path = root / (name + '.png')
                Image.fromarray(np.zeros((20, 20, 3), np.uint8)).save(path)
                rows.append({'image_id': name, 'image': path.name, 'image_sha256': sha256(path)})
            inputs = root / 'images.json'
            inputs.write_text(json.dumps(rows))

            def engine(command, **kwargs):
                if '--version' in command:
                    return subprocess.CompletedProcess(command, 0, 'test-engine\n', '')
                if '--list-langs' in command:
                    return subprocess.CompletedProcess(command, 0, 'eng\n', '')
                if Path(command[1]).name == 'a.png':
                    raise subprocess.TimeoutExpired(command, 60, output=b'incomplete OCR')
                return subprocess.CompletedProcess(command, 0,
                    'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n', '')

            with patch('examples.inspect_ecg_print_settings.subprocess.run', side_effect=engine):
                result = inspect(inputs, root, root / 'outputs')
            self.assertEqual(result['scheduled_images'], 2)
            self.assertEqual(len(result['cases']), 2)
            self.assertEqual(result['cases'][0]['returncode'], 124)
            self.assertEqual(result['cases'][0]['parsed']['candidates'], [])


if __name__ == '__main__':
    unittest.main()
