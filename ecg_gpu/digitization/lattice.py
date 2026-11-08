"""Offline image-only organization of unordered major-grid intersections.

This conservative first prototype assumes a single approximately axis-aligned
visible sheet, major-grid detections and moderate local perspective. It infers
relative integer identities, never generator identities or an absolute origin.
NumPy/SciPy work belongs outside rendering and training hot paths.
"""
from collections import deque

import numpy as np


def organize_lattice(xy, scores=None, *, max_conflict_fraction=.01):
    """Infer mutual nearest axial neighbors and retain supported convex cells.

    Explicit rejection is preferable to a fabricated complete grid. Confidence
    orders traversal only, it is not a calibrated error probability. Local cell
    orientation is checked, global injectivity is not established by this code.
    """
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Finite [N,2] image pixel centers required')
    if not 0 <= max_conflict_fraction < 1:
        raise ValueError('Conflict fraction must lie in [0,1)')
    scores = np.ones(len(xy)) if scores is None else np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(xy),) or not np.isfinite(scores).all():
        raise ValueError('Finite per-point scores required')
    empty = {'status': 'rejected', 'input_points': len(xy), 'selected_indices': [],
             'ids': [], 'cells': [], 'reason': 'insufficient points',
             'oracle_correspondence': False, 'global_injectivity_checked': False}
    if len(xy) < 4:
        return empty
    if len(np.unique(xy, axis=0)) != len(xy):
        return dict(empty, reason='duplicate points')
    distance, neighbor = cKDTree(xy).query(xy, k=min(13, len(xy)))
    scale = float(np.median(distance[:, 1]))
    if scale <= 1e-8:
        return dict(empty, reason='degenerate spacing')
    delta = xy[neighbor[:, 1:]]-xy[:, None]
    length = np.linalg.norm(delta, axis=-1)
    near = (length > .55*scale) & (length < 1.3*scale)
    angle = np.arctan2(delta[..., 1], delta[..., 0])
    direction = np.mean(np.exp(4j*angle[near])) if near.any() else 0j
    if abs(direction) < .1:
        return dict(empty, reason='no dominant grid directions')
    theta = float(np.angle(direction)/4)
    axes = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    projected = delta@axes.T
    spacing = []
    for axis in range(2):
        along, across = np.abs(projected[..., axis]), np.abs(projected[..., 1-axis])
        candidates = (along > .5*scale) & (along < 2*scale) & (across < .4*along)
        nearest = np.where(candidates, along, np.inf).min(axis=1)
        finite = nearest[np.isfinite(nearest)]
        if not len(finite):
            return dict(empty, reason='missing grid axis')
        spacing.append(float(np.median(finite)))
    links = np.full((len(xy), 4), -1, dtype=int)
    for d, (axis, sign) in enumerate(((0, 1), (0, -1), (1, 1), (1, -1))):
        along = sign*projected[..., axis]
        across = np.abs(projected[..., 1-axis])
        candidates = ((along > .55*spacing[axis]) & (along < 1.5*spacing[axis])
                      & (across < .45*spacing[1-axis]))
        cost = (along/spacing[axis]-1)**2+(across/spacing[1-axis])**2
        cost = np.where(candidates, cost, np.inf)
        best = cost.argmin(axis=1)
        valid = np.isfinite(cost[np.arange(len(xy)), best])
        links[valid, d] = neighbor[:, 1:][np.arange(len(xy))[valid], best[valid]]
    opposite = (1, 0, 3, 2)
    steps = ((0, 1), (0, -1), (1, 0), (-1, 0))
    graph = [[] for _ in xy]
    for i in range(len(xy)):
        for d, j in enumerate(links[i]):
            if j >= 0 and links[j, opposite[d]] == i:
                graph[i].append((int(j), steps[d]))
    seen, components = set(), []
    for seed in np.argsort(-scores, kind='stable'):
        seed = int(seed)
        if seed in seen:
            continue
        positions, occupants = {seed: (0, 0)}, {(0, 0): seed}
        queue, conflicts, edges = deque([seed]), 0, 0
        seen.add(seed)
        while queue:
            i = queue.popleft()
            for j, step in graph[i]:
                edges += 1
                expected = tuple(positions[i][a]+step[a] for a in range(2))
                if j in positions:
                    conflicts += positions[j] != expected
                elif expected in occupants:
                    conflicts += 1
                elif j not in seen:
                    positions[j] = expected
                    occupants[expected] = j
                    seen.add(j)
                    queue.append(j)
        components.append((positions, conflicts, edges))
    positions, conflicts, edges = max(components, key=lambda c: len(c[0]))
    selected = np.array(sorted(positions), dtype=int)
    ids = np.array([positions[i] for i in selected], dtype=int)
    ids -= ids.min(axis=0)
    lookup = {tuple(identity): k for k, identity in enumerate(ids)}
    cells, inverted = [], 0
    for row, col in ids:
        corners = ((row, col), (row, col+1), (row+1, col+1), (row+1, col))
        if not all(corner in lookup for corner in corners):
            continue
        quad = xy[selected[[lookup[c] for c in corners]]]
        sides = np.roll(quad, -1, axis=0)-quad
        following = np.roll(sides, -1, axis=0)
        cross = sides[:, 0]*following[:, 1]-sides[:, 1]*following[:, 0]
        if (cross > .05*spacing[0]*spacing[1]).all():
            cells.append([int(row), int(col)])
        else:
            inverted += 1
    fraction = conflicts/max(edges, 1)
    accepted = len(cells) >= 1 and fraction <= max_conflict_fraction
    return dict(empty, status='accepted' if accepted else 'rejected',
                reason=None if accepted else 'no valid cells or excessive cycle conflicts',
                selected_indices=selected.tolist(), ids=ids.tolist(), cells=cells,
                component_points=len(selected), components=len(components),
                cycle_conflicts=int(conflicts), directed_edges=edges,
                conflict_fraction=fraction, invalid_cells=inverted,
                spacing_px=spacing, orientation_radians=theta,
                retained_point_fraction=len(selected)/len(xy))
