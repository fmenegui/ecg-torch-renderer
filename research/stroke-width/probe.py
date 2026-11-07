"""Perpendicular stroke width of the current raster versus segment slope.

A NumPy port of ``ecg_gpu.gpu_raster.curve_coverage`` (bounded segments, no
extrema envelope) for one straight line. Runs without Torch:

    python research/stroke-width/probe.py

Each pixel measures distance only to the curve segments inside its own column,
so steep strokes lose horizontal width. Measured with line_width=3:
slope 0 -> 3.00 px, 1 -> 2.71 px, 3 -> 1.85 px, 10 -> 1.27 px.
"""
import numpy as np

H = W = 80
SAMPLES = 4
LINE_WIDTH = 3.0
THICKNESS = LINE_WIDTH / 2 + .5
yy = np.arange(H)[:, None].astype(float)


def render(curve):
    out = np.zeros((H, W))
    for sample in range(SAMPLES):
        axis = np.arange(W) - .5 + sample / SAMPLES
        a = curve(axis)
        dy = curve(axis + 1 / SAMPLES) - a
        dx = 1 / SAMPLES
        x = .5 - sample / SAMPLES  # px[sample] = xx - axes[sample]
        py = yy - a[None, :]
        t = np.clip((x * dx + py * dy) / (dx * dx + dy ** 2), 0, 1)
        distance = np.sqrt((x - t * dx) ** 2 + (py - t * dy) ** 2)
        out = np.maximum(out, np.clip(THICKNESS - distance, 0, 1))
    return out


for slope in (0, 1, 3, 10):
    image = render(lambda x, k=slope: 40 + k * (x - 40))
    columns = slice(38, 43)
    width = image[:, columns].sum() / (5 * np.hypot(1, slope))
    print(f'slope {slope:>2}: perpendicular width {width:.2f} px (line_width {LINE_WIDTH})')
