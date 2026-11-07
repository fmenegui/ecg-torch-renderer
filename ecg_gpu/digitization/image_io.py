"""Explicit EXIF-aware ECG RGB decoding with exact pixel-coordinate maps."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def exif_pixel_map(orientation, width, height):
    """Map stored pixel centers to displayed pixel centers for EXIF 1 through 8."""
    if type(orientation) is not int or orientation not in range(1, 9):
        raise ValueError('EXIF orientation must be an integer in [1,8]')
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError('Positive stored image dimensions required')
    matrices = {
        1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        2: [[-1, 0, width-1], [0, 1, 0], [0, 0, 1]],
        3: [[-1, 0, width-1], [0, -1, height-1], [0, 0, 1]],
        4: [[1, 0, 0], [0, -1, height-1], [0, 0, 1]],
        5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        6: [[0, -1, height-1], [1, 0, 0], [0, 0, 1]],
        7: [[0, -1, height-1], [-1, 0, width-1], [0, 0, 1]],
        8: [[0, 1, 0], [-1, 0, width-1], [0, 0, 1]],
    }
    return np.array(matrices[orientation], dtype=np.float64)


def decode_ecg_rgb(path):
    """Honor stored orientation once, preserving original bytes and provenance.

    Native reproduction inputs remain separate. This helper returns a new
    coordinate frame. Geometry, baseline and origin evidence must be measured
    in that frame or transformed explicitly with the recorded pixel map.
    """
    path = Path(path)
    source_id = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
        matrix = exif_pixel_map(orientation, *image.size)
        original_size = list(image.size)
        decoded = ImageOps.exif_transpose(image).convert('RGB')
        rgb = np.array(decoded, dtype=np.uint8)
        decoded_size = list(decoded.size)
    receipt = {'schema': 'exif-oriented-ecg-rgb-v1', 'source_id': source_id,
        'stored_size_wh': original_size, 'decoded_size_wh': decoded_size,
        'exif_orientation': orientation, 'orientation_applied': orientation != 1,
        'frame_id': 'exif-oriented-rgb:' + source_id,
        'stored_frame_id': 'stored-rgb:' + source_id,
        'stored_to_decoded_pixel_centers': matrix.tolist(),
        'decoded_to_stored_pixel_centers': np.linalg.inv(matrix).tolist(),
        'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'geometric_interpolation_performed': False, 'original_file_modified': False}
    return rgb, receipt
