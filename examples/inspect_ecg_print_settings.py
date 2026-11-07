"""Record local OCR print settings without waveform references or default units."""

import argparse
import csv
import io
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ecg_gpu.digitization.calibration import parse_print_settings
from ecg_gpu.digitization.image_io import decode_ecg_rgb
from reproductions.common import sha256, write_json


def recognized_lines(tsv):
    lines, words = {}, []
    # Tesseract emits literal, unescaped quotes in its last TSV field.
    for row in csv.DictReader(io.StringIO(tsv), delimiter='\t', quoting=csv.QUOTE_NONE):
        if row['level'] != '5' or not row['text'].strip():
            continue
        identity = tuple(row[k] for k in ['page_num', 'block_num', 'par_num', 'line_num'])
        lines.setdefault(identity, []).append(row['text'])
        words.append({'text': row['text'], 'confidence': float(row['conf']),
            'box_xywh': [int(row[k]) for k in ['left', 'top', 'width', 'height']]})
    return '\n'.join(' '.join(row) for row in lines.values()), words


def border_view(rgb, max_side=2400):
    """Fixed header/footer view with an explicit inverse map for each band."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 10:
        raise ValueError('Nontrivial uint8 RGB required')
    if type(max_side) is not int or not 32 <= max_side <= 4096:
        raise ValueError('Bounded OCR image size required')
    height, width = rgb.shape[:2]
    scale = min(1., max_side / max(height, width))
    sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
    small = np.asarray(Image.fromarray(rgb).resize((sw, sh), Image.Resampling.LANCZOS))
    header, footer = max(1, round(.2 * sh)), min(sh-1, round(.65 * sh))
    gap = 16
    view = np.concatenate([small[:header], np.full((gap, sw, 3), 255, np.uint8), small[footer:]], axis=0)
    mapping = {'input_size_wh': [width, height], 'view_size_wh': [sw, len(view)],
        'scale_x': sw / width, 'scale_y': sh / height,
        'bands': [{'view_y0': 0, 'view_y1': header, 'resized_input_y0': 0},
                  {'view_y0': header+gap, 'view_y1': len(view), 'resized_input_y0': footer}]}
    return view, mapping


def word_to_input(box, mapping):
    x, y, width, height = box
    for band in mapping['bands']:
        if band['view_y0'] <= y and y+height <= band['view_y1']:
            return [x / mapping['scale_x'],
                    (y-band['view_y0']+band['resized_input_y0']) / mapping['scale_y'],
                    width / mapping['scale_x'], height / mapping['scale_y']]
    return None


def inspect(inputs, root, out, reuse_tsv=None, use_border_view=False):
    inputs, root, out = Path(inputs), Path(root), Path(out)
    images = json.loads(inputs.read_text())
    if not images or any(set(c) != {'image_id', 'image', 'image_sha256'} for c in images):
        raise ValueError('Strict image-only input manifest required')
    if len({c['image_id'] for c in images}) != len(images):
        raise ValueError('Unique scheduled images required')
    out.mkdir(parents=True, exist_ok=False)
    version = subprocess.run(['tesseract', '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    languages = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True, check=True).stdout
    directory = re.search(r'in "([^"]+)"', languages)
    model = Path(directory[1]) / 'eng.traineddata' if directory else None
    environment = dict(os.environ, OMP_THREAD_LIMIT='1')
    prior = json.loads((Path(reuse_tsv) / 'receipt.json').read_text()) if reuse_tsv else None
    if prior and prior['input_manifest_sha256'] != sha256(inputs):
        raise ValueError('Original OCR input manifest changed')
    if prior and use_border_view:
        raise ValueError('A changed OCR view cannot reuse full-frame recognition')
    previous = {c['image_id']: c for c in prior['cases']} if prior else {}

    def read(case):
        relative = Path(case['image'])
        if relative.is_absolute() or '..' in relative.parts or sha256(root / relative) != case['image_sha256']:
            raise ValueError('Original image identity changed')
        path = out / (case['image_id'] + '.tsv')
        image_path, mapping, decoding = root / relative, None, None
        if use_border_view:
            rgb, decoding = decode_ecg_rgb(image_path)
            view, mapping = border_view(rgb)
            image_path = out / (case['image_id'] + '-ocr-view.png')
            Image.fromarray(view).save(image_path)
        if prior:
            old = previous[case['image_id']]
            mapping, decoding = old.get('ocr_view_mapping'), old.get('input_decoding')
            source = Path(reuse_tsv) / old['tsv']
            if sha256(source) != old['tsv_sha256'] or old['image_sha256'] != case['image_sha256']:
                raise ValueError('Original OCR bytes or image identity changed')
            raw, code, stderr = source.read_text(), old['returncode'], old['stderr']
        else:
            try:
                completed = subprocess.run(['tesseract', str(image_path), 'stdout', '-l', 'eng', '--psm', '11', 'tsv'],
                    capture_output=True, text=True, timeout=25 if use_border_view else 60, check=False, env=environment)
                raw, code, stderr = completed.stdout, completed.returncode, completed.stderr
            except subprocess.TimeoutExpired as error:
                raw, code, stderr = error.stdout or '', 124, 'OCR timeout, partial text is not accepted'
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8', errors='replace')
        path.write_text(raw)
        text, words = recognized_lines(raw) if code == 0 else ('', [])
        if mapping:
            for word in words:
                word['box_in_decoded_rgb'] = word_to_input(word['box_xywh'], mapping)
        parsed = dict(parse_print_settings(text), source_id=case['image_sha256'])
        return {'image_id': case['image_id'], 'image_sha256': case['image_sha256'],
            'returncode': code, 'stderr': stderr,
            'observed_text': text, 'recognized_words': words,
            'word_coordinate_frame': 'ocr_view' if mapping else 'stored_rgb',
            'ocr_view_mapping': mapping, 'input_decoding': decoding,
            'parsed': parsed, 'tsv': path.name, 'tsv_sha256': sha256(path),
            'independently_reviewed_text': False, 'physical_calibration_resolved': False,
            'remaining_required_evidence': ['grid_pixel_density', 'baseline', 'independent_time_anchor']}

    cases = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for case in pool.map(read, images):
            cases.append(case)
            write_json(out / 'progress.json', {'scheduled_images': len(images), 'completed_cases': cases})
    receipt = {'schema': 'image-only-ecg-print-settings-ocr-v1', 'ocr_engine': version,
        'ocr_model_sha256': sha256(model) if model and model.is_file() else None,
        'ocr_model': 'eng', 'page_segmentation_mode': 11, 'reference_signal_received': False,
        'input_manifest_sha256': sha256(inputs), 'scheduled_images': len(images),
        'images_with_unambiguous_speed_and_gain': sum(c['parsed']['paper_speed_mm_s'] is not None
            and c['parsed']['gain_mm_mv'] is not None for c in cases),
        'default_settings_used': False, 'waveforms_generated': False, 'cases': cases}
    receipt['raw_tsv_reused_sha256'] = sha256(Path(reuse_tsv) / 'receipt.json') if prior else None
    receipt['ocr_view_profile'] = prior.get('ocr_view_profile', 'original-encoded-full-frame') if prior else (
        'exif-header-footer-max2400' if use_border_view else 'original-encoded-full-frame')
    write_json(out / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['inputs', 'root', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reuse-tsv', type=Path)
    parser.add_argument('--border-view', action='store_true')
    args = parser.parse_args()
    receipt = inspect(args.inputs, args.root, args.out, args.reuse_tsv, args.border_view)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
