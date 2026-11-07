"""Compare unchanged native normalization with explicit physical coordinates."""

import argparse
from pathlib import Path

import numpy as np

from ecg_gpu.digitization.physical_trace import sample_physical_trace
from reproductions.common import write_json
from reproductions.npj.time_axis_control import measure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    native = measure(args.upstream)
    fs, ppmm, speed = native['fs'], native['pixels_per_mm'], native['assumed_speed_mm_per_s']
    t = np.arange(5000) / fs
    peak = np.exp(-((t - 4.) / .02) ** 2)
    baseline = 1000 + 200 * np.arange(3)
    raw = baseline[:, None] - 100 * peak
    rows = []
    for before in native['rows']:
        missing = before['leading_missing_columns']
        observed = raw.copy()
        observed[:, :missing] = np.nan
        values, receipt = sample_physical_trace(observed, np.arange(5000),
            baseline_y_px=baseline, x_origin_px=0., pixels_per_mm_x=ppmm,
            pixels_per_mm_y=ppmm, paper_speed_mm_s=speed, gain_mm_mv=10.,
            fs=fs, samples=5000, max_gap_s=1.01 / (ppmm * speed))
        measured = int(np.nanargmax(values[0])) / fs
        assert measured == 4. and receipt['missing_samples'] == missing * 3
        rows.append({**before, 'physical_peak_s': measured,
                     'physical_peak_error_ms': (measured - 4.) * 1000,
                     'physical_output_missing_samples': receipt['missing_samples']})
    result = {'schema': 'coordinate-preserving-time-axis-comparison-v1', 'scope': 'analytical fixture only',
              'native_source': native['source'], 'native_module_sha256': native['native_module_sha256'],
              'native_source_modified': False, 'neural_network_inference': False,
              'real_photo_improvement_established': False,
              'fs': fs, 'pixels_per_mm': ppmm, 'paper_speed_mm_s': speed, 'gain_mm_mv': 10.,
              'x_origin_fixed': True, 'baselines_supplied': True, 'rows': rows}
    write_json(args.out, result)
    print(result)


if __name__ == '__main__':
    main()
