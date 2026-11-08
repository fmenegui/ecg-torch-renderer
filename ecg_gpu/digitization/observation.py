"""Optional periodic-band subtraction for geometry inference, never output RGB.

The sinusoid is estimated only from observed background outside a pilot-point
ROI. Plain grid lines inside the paper are not used to estimate the correction.
This is an image-processing hypothesis, not a calibrated camera noise model.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components


def pilot_box(xy, shape):
    xy = np.asarray(xy, float)
    if len(xy) < 32:
        return None
    tree = cKDTree(xy)
    spacing = float(np.median(tree.query(xy, k=2)[0][:, 1]))
    if spacing <= 0 or tree.count_neighbors(tree, 2.25*spacing) > 2000000:
        return None
    graph = tree.sparse_distance_matrix(tree, 2.25*spacing, output_type='coo_matrix')
    _, labels = connected_components(graph, directed=False)
    points = xy[labels == np.bincount(labels).argmax()]
    if len(points) < 32:
        return None
    lower, upper = np.quantile(points, [.005, .995], axis=0)
    margin = np.maximum(2*spacing, .1*(upper-lower))
    h, w = shape[:2]
    lo = np.floor(np.maximum(lower-margin, 0)).astype(int)
    hi = np.ceil(np.minimum(upper+margin+1, [w, h])).astype(int)
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def _periodic_wave(profile):
    n = len(profile)
    if n < 32:
        return None, {'applied': False}
    t = np.arange(n)
    normalized = np.linspace(-1., 1., n)
    base = np.column_stack((np.ones(n), normalized, normalized**2))
    trend = base@np.linalg.lstsq(base, profile, rcond=None)[0]
    residual = profile-trend
    denominator = float(np.sum(residual**2))
    if denominator < 1e-8:
        return None, {'applied': False}
    power = np.abs(np.fft.rfft(residual.mean(1)*np.hanning(n)))**2
    indices = np.arange(len(power))
    power[(indices < n/64) | (indices > n/4)] = 0
    k = int(power.argmax())
    concentration = float(power[max(k-1, 0):k+2].sum()/max(power.sum(), 1e-12))
    lower, upper = max(1/64, (k-.7)/n), min(.25, (k+.7)/n)
    if concentration < .4 or lower >= upper:
        return None, {'applied': False}

    def fit(frequency):
        matrix = np.column_stack((base, np.sin(2*np.pi*frequency*t), np.cos(2*np.pi*frequency*t)))
        coefficients = np.linalg.lstsq(matrix, profile, rcond=None)[0]
        error = profile-matrix@coefficients
        return float(np.sum(error**2)), matrix[:, -2:]@coefficients[-2:]

    optimized = minimize_scalar(lambda f: fit(f)[0], bounds=(lower, upper), method='bounded',
                                options={'xatol': 1e-9})
    loss, wave = fit(optimized.x)
    explained = 1-loss/denominator
    amplitude = float(np.max(np.abs(wave)))
    applied = bool(explained >= .65 and 2 <= amplitude <= 25)
    return (wave if applied else None), {'applied': applied, 'frequency_cycles_per_pixel': float(optimized.x),
        'background_variance_explained': float(explained), 'peak_concentration': concentration,
        'amplitude_rgb_levels': amplitude}


def remove_periodic_banding(rgb, pilot_points):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('uint8 HWC RGB required')
    box = pilot_box(pilot_points, rgb.shape)
    report = {'protocol': 'background-periodic-banding-v1', 'applied': False,
              'roi_xyxy': box, 'ground_truth_used': False,
              'use': 'point inference only, original RGB is the sampling source'}
    if box is None:
        return rgb.copy(), report
    image = rgb.astype(float)
    axes = []
    for axis in (0, 1):
        oriented = image if axis == 0 else image.transpose(1, 0, 2)
        first, last = (box[0], box[2]) if axis == 0 else (box[1], box[3])
        active = np.mean(np.abs(oriented-255), axis=(0, 2)) > .25
        background = active & ((np.arange(len(active)) < first) | (np.arange(len(active)) >= last))
        if int(background.sum()) < 16:
            axes.append({'axis': axis, 'applied': False, 'reason': 'insufficient observed background'})
            continue
        occupied = np.mean(np.abs(oriented-255), axis=(1, 2)) > .25
        positions = np.flatnonzero(occupied)
        if not len(positions):
            axes.append({'axis': axis, 'applied': False})
            continue
        start, stop = int(positions[0]), int(positions[-1])+1
        profile = np.median(oriented[start:stop, background], axis=1)
        wave, measured = _periodic_wave(profile)
        measured.update(axis=axis, background_samples_per_line=int(background.sum()))
        axes.append(measured)
        if wave is not None:
            region = oriented[start:stop]
            region[:, active] -= wave[:, None]
            report['applied'] = True
    report['axes'] = axes
    return image.round().clip(0, 255).astype(np.uint8), report
