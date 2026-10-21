"""Propose silver-reference intersections in a clean, axis-aligned red grid ROI.

Requires visual review before evaluation. Not a general detector or gold label.
Original RGB is never overwritten. Values use original-image pixel centers.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import find_peaks


def annotate(image_path, region):
    image = Image.open(image_path).convert('RGB')
    rgb = np.asarray(image).astype(float)
    red = (rgb[..., 0]-(rgb[..., 1]+rgb[..., 2])/2).clip(0)
    x0, y0, x1, y1 = region
    if not 20 <= x0 < x1 <= image.width-20 or not 20 <= y0 < y1 <= image.height-20:
        raise ValueError('ROI must leave at least 20 pixels around the image border')
    horizontal, vertical = red[y0:y1, x0:x1].mean(0), red[y0:y1, x0:x1].mean(1)
    xs = find_peaks(horizontal, distance=25, prominence=10)[0]+x0
    ys = find_peaks(vertical, distance=25, prominence=10)[0]+y0
    if min(len(xs), len(ys)) < 3:
        raise ValueError('Insufficient major red lines in ROI')
    if max(np.std(np.diff(xs))/np.mean(np.diff(xs)), np.std(np.diff(ys))/np.mean(np.diff(ys))) > .08:
        raise ValueError('Irregular spacing. Review the ROI rather than accepting proposals')
    points = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for y in ys:
        for x in xs:
            # Exclude the crossing arm and fit each line's local intensity center.
            rows = np.r_[y-14:y-4, y+5:y+15]
            columns = np.r_[x-14:x-4, x+5:x+15]
            xp = red[rows, x-4:x+5].mean(0)
            yp = red[y-4:y+5, columns].mean(1)
            xp, yp = (xp-xp.min()).clip(0), (yp-yp.min()).clip(0)
            if min(xp.sum(), yp.sum()) <= 0:
                raise ValueError('A proposed intersection lacks red-line support')
            cx, cy = float(np.dot(np.arange(x-4, x+5), xp)/xp.sum()), float(np.dot(np.arange(y-4, y+5), yp)/yp.sum())
            points.append([cx, cy])
            draw.line((cx-5, cy, cx+5, cy), fill='#00b8d4', width=1)
            draw.line((cx, cy-5, cx, cy+5), fill='#00b8d4', width=1)
    return {'points_xy': points, 'annotation_region': region,
        'annotation_method': 'Silver reference: major red profile peaks and local intensity centers. Requires visual review. Not expert gold annotations',
        'reference_uncertainty': 'Not calibrated. Pixel sampling, ink width, scan skew and resizing limit reference accuracy',
        'image_sha256': hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        'proposer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, overlay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--region', type=int, nargs=4, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite annotations')
    result, overlay = annotate(args.image, args.region)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    overlay.crop(tuple(args.region)).save(args.out.with_suffix('.png'))
