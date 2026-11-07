"""CPU evidence/eligibility microbenchmark in fresh subprocesses, not NN latency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np
from scipy.spatial import Delaunay

from audit_loader import DEFAULT_GEOMETRY, load_geometry

# Parse only the source location before importing the pinned modules.
_source_parser = argparse.ArgumentParser(add_help=False)
_source_parser.add_argument('--geometry-dir', type=Path, default=DEFAULT_GEOMETRY)
_source_args, _ = _source_parser.parse_known_args()
_geometry = load_geometry(_source_args.geometry_dir)
CompleteMap, fit_complete_map = _geometry.CompleteMap, _geometry.fit_complete_map
from bounded_evidence import BoundedEvidence


def fixture(name):
    if name == 'many_hull_vertices':
        angle = np.arange(256)*2*np.pi/256
        observed = np.column_stack((100+80*np.cos(angle), 100+80*np.sin(angle)))
        row, col = np.mgrid[:16, :16]
        ids = np.column_stack((row.ravel(), col.ravel()))
        triangles = Delaunay(observed).simplices
        mapping = CompleteMap(observed, observed, np.empty((0, 2), int), triangles,
                              {}, observed, observed, ids, [])
        bounds = ([5., 5.], [195., 195.])
    elif name == 'cells_and_hole':
        row, col = np.mgrid[:12, :18]
        ids = np.column_stack((row.ravel(), col.ravel()))
        observed = ids[:, ::-1]*10.+[30, 25]
        keep = ~((ids[:, 0] >= 4) & (ids[:, 0] <= 7)
                 & (ids[:, 1] >= 6) & (ids[:, 1] <= 11))
        observed, ids = observed[keep], ids[keep]
        mapping, details = fit_complete_map(observed, ids[:, ::-1]*5., ids, (160, 240))
        if mapping is None:
            raise RuntimeError(details)
        bounds = ([0., 0.], [239., 159.])
    else:
        raise ValueError(name)
    return mapping, bounds


def queries(mapping, bounds, count, dtype=np.float64):
    random = np.random.default_rng(1973).uniform(*bounds, size=(count, 2))
    hull = cv2.convexHull(mapping.observed.astype(np.float32)).reshape(-1, 2)
    mid = (hull+np.roll(hull, -1, axis=0))/2
    centers = mapping.source[mapping.triangles].mean(1)
    edges = mapping.source[mapping.triangles[:, [0, 1]]].mean(1)
    boundary = np.vstack((mapping.observed, hull, mid, mid+1e-7, mid-1e-7, centers, edges))
    return np.vstack((random, boundary)).astype(dtype)


def parity():
    records = []
    for name in ('cells_and_hole', 'many_hull_vertices'):
        mapping, bounds = fixture(name)
        for dtype in (np.float32, np.float64):
            points = queries(mapping, bounds, 4096, dtype)
            for budget in (17, 131072):
                wrapper = BoundedEvidence(mapping, budget)
                for selected in (points, points[::-1], points[:0]):
                    mapped = mapping.forward(selected)
                    old_e = mapping.evidence(selected)
                    old_g = mapping.eligible_geometry(selected, mapped=mapped)
                    new_e, new_g = wrapper.classify(selected, mapped=mapped)
                    np.testing.assert_array_equal(old_e, new_e)
                    np.testing.assert_array_equal(old_g, new_g)
                    assert wrapper.maximum_pairs_used <= budget
                records.append({'fixture': name, 'dtype': str(dtype), 'pair_budget': budget,
                    'queries_with_boundaries': len(points), 'hull_vertices': len(wrapper.hull),
                    'bitwise_equal': True, 'maximum_pairs_used': wrapper.maximum_pairs_used})
    return records


def worker(args):
    cv2.setNumThreads(1)
    mapping, bounds = fixture(args.case)
    points = queries(mapping, bounds, args.count)
    # A precomputed mapped argument matches render_complete's caller contract.
    mapped = mapping.forward(points)
    start = time.perf_counter()
    wrapper = BoundedEvidence(mapping, args.pair_budget) if args.variant == 'bounded_combined' else None
    setup_seconds = time.perf_counter()-start

    def memory_snapshot():
        values = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmHWM:', 'VmRSS:')):
                key, value = line.split(':', 1)
                values[key+'_mib'] = int(value.split()[0])/1024
        return values

    before_timing = memory_snapshot()

    def run():
        if wrapper is None:
            return mapping.evidence(points), mapping.eligible_geometry(points, mapped=mapped)
        return wrapper.classify(points, mapped=mapped)

    times = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        evidence, eligible = run()
        times.append(time.perf_counter()-start)
    checksum = hashlib.sha256(evidence.tobytes()+eligible.tobytes()).hexdigest()
    after_timing = memory_snapshot()
    return {'case': args.case, 'variant': args.variant, 'random_queries': args.count,
        'total_queries_including_boundaries': len(points),
        'hull_vertices': len(cv2.convexHull(mapping.observed.astype(np.float32))),
        'seconds': times, 'median_seconds': statistics.median(times),
        'setup_seconds_excluded_from_timings': setup_seconds,
        'process_peak_rss_mib': after_timing['VmHWM_mib'],
        'memory_before_timing': before_timing, 'memory_after_timing': after_timing,
        'rusage_peak_rss_mib_secondary': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'output_sha256': checksum,
        'maximum_pairs_used': wrapper.maximum_pairs_used if wrapper is not None else len(points)*len(cv2.convexHull(mapping.observed.astype(np.float32)))}


def main(args):
    checks = parity()
    records = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for case in ('cells_and_hole', 'many_hull_vertices'):
        for repetition in range(args.processes):
            # Alternate ordering across subprocess repetitions.
            variants = ('baseline', 'bounded_combined') if repetition % 2 == 0 else ('bounded_combined', 'baseline')
            for variant in variants:
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--case', case,
                    '--variant', variant, '--count', str(args.count), '--repetitions', str(args.repetitions),
                    '--pair-budget', str(args.pair_budget), '--geometry-dir', str(args.geometry_dir)]
                result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
                row = json.loads(result.stdout)
                row['fresh_process_repetition'] = repetition
                records.append(row)
    aggregate = {}
    for case in ('cells_and_hole', 'many_hull_vertices'):
        subset = [r for r in records if r['case'] == case]
        assert len({r['output_sha256'] for r in subset}) == 1
        metrics = {}
        for variant in ('baseline', 'bounded_combined'):
            arm = [r for r in subset if r['variant'] == variant]
            metrics[variant] = {'median_seconds': statistics.median([r['median_seconds'] for r in arm]),
                'median_process_peak_rss_mib': statistics.median([r['process_peak_rss_mib'] for r in arm]),
                'median_rss_before_timing_mib': statistics.median([r['memory_before_timing']['VmRSS_mib'] for r in arm]),
                'median_hwm_before_timing_mib': statistics.median([r['memory_before_timing']['VmHWM_mib'] for r in arm])}
        metrics['median_time_ratio_baseline_over_bounded'] = metrics['baseline']['median_seconds']/metrics['bounded_combined']['median_seconds']
        metrics['all_output_digests_equal'] = True
        aggregate[case] = metrics
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), 'unknown')
    report = {'scope': 'CPU evidence plus eligibility classification microbenchmark, mapped coordinates precomputed',
        'audited_commit': 'b9fd04dd6516e3469d1f36593849249c082458a8',
        'source_sha256': {name: hashlib.sha256((args.geometry_dir/name).read_bytes()).hexdigest()
            for name in ('complete_rectifier.py', 'frame_mesh.py', 'cell_mesh.py')},
        'script_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('evidence_benchmark.py', 'bounded_evidence.py', 'audit_loader.py')},
        'not_end_to_end': True, 'no_neural_inference_or_rgb_sampling_timed': True,
        'hardware': {'cpu_model': cpu_model, 'visible_logical_cpus': os.cpu_count(), 'platform': platform.platform()},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__},
        'memory_measurement': 'Linux /proc/self/status VmHWM for the exec-created mm, with VmRSS and VmHWM before timing. getrusage is secondary because its high-water can be inherited from fork/exec.',
        'threads': 'OpenCV1 and OMP/MKL/OpenBLAS1 in each fresh worker',
        'pair_budget': args.pair_budget, 'process_repetitions_per_case_and_variant': args.processes,
        'timed_repetitions_per_process': args.repetitions, 'parity': checks,
        'aggregates': aggregate, 'records': records,
        'limitations': ['Fixture maps are synthetic controls, not native detector outputs',
            'many_hull_vertices is a stress fixture with arbitrary integer IDs and no observed cells, not a physical ECG lattice',
            'Peak RSS is whole fresh worker process, including imports and fixture construction',
            'Hull precomputation setup is reported separately and excluded from timed classification',
            'Geometry must remain unchanged for the lifetime of the wrapper',
            'Bounded pair arrays do not bound memory used by original CellMaps or KDTree',
            'No inference throughput, GPU latency, or SOTA superiority is inferred']}
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir', type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case', default='cells_and_hole')
    parser.add_argument('--variant', default='baseline')
    parser.add_argument('--count', type=int, default=24576)
    parser.add_argument('--processes', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--pair-budget', type=int, default=131072)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    result = worker(args) if args.worker else main(args)
    print(json.dumps(result if args.worker else result['aggregates']))
