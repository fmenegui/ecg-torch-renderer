"""Image-only relative IDs from confidence-ordered integer graph constraints.

Integer union constraints reject conflicting cycles and duplicate occupancy.
Two-step links may join observed regions, but never create missing nodes/cells.
The local axial assumptions remain and global injectivity is not certified.
"""
import numpy as np
from scipy.spatial import cKDTree


def organize_cycle_lattice(xy, scores=None, *, max_step=2, geometry_hint=None):
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1:] != (2,) or not np.isfinite(xy).all():
        raise ValueError('Finite [N,2] image points required')
    if type(max_step) is not int or max_step not in (1, 2):
        raise ValueError('max_step must be 1 or 2')
    scores = np.ones(len(xy)) if scores is None else np.asarray(scores, float)
    if scores.shape != (len(xy),) or not np.isfinite(scores).all():
        raise ValueError('Finite per-point scores required')
    empty = {'status': 'rejected', 'input_points': len(xy), 'selected_indices': [],
             'ids': [], 'cells': [], 'reason': 'insufficient or degenerate grid',
             'oracle_correspondence': False, 'global_injectivity_checked': False,
             'organizer': 'cycle-consistent', 'max_step': max_step}
    if len(xy) < 4 or len(np.unique(xy, axis=0)) != len(xy):
        return empty
    distance, neighbors = cKDTree(xy).query(xy, k=min(33 if geometry_hint else 17, len(xy)))
    scale = float(np.median(distance[:, 1]))
    if scale < 1e-8:
        return empty
    delta = xy[neighbors[:, 1:]]-xy[:, None]
    length = np.linalg.norm(delta, axis=-1)
    near = (length > .55*scale) & (length < 1.3*scale)
    angle = np.arctan2(delta[..., 1], delta[..., 0])
    direction = np.mean(np.exp(4j*angle[near])) if near.any() else 0j
    if abs(direction) < .1 and geometry_hint is None:
        return dict(empty, reason='no dominant grid directions')
    theta = float(np.angle(direction)/4)
    if geometry_hint is not None:
        theta = float(geometry_hint['orientation_radians'])
        if not np.isfinite(theta):
            raise ValueError('Finite pilot orientation required')
    axes = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    projected = delta@axes.T
    spacing = []
    for axis in (() if geometry_hint is not None else (0, 1)):
        along, across = np.abs(projected[..., axis]), np.abs(projected[..., 1-axis])
        candidates = (along > .5*scale) & (along < 2*scale) & (across < .4*along)
        nearest = np.where(candidates, along, np.inf).min(1)
        finite = nearest[np.isfinite(nearest)]
        if not len(finite):
            return dict(empty, reason='missing grid axis')
        spacing.append(float(np.median(finite)))
    if geometry_hint is not None:
        spacing = list(map(float, geometry_hint['spacing_px']))
        if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError('Positive finite pilot spacing required')
    links, costs, increments = [], [], []
    directions = ((0, 1), (0, -1), (1, 1), (1, -1))
    for axis, sign in directions:
        along, across = sign*projected[..., axis], np.abs(projected[..., 1-axis])
        step = np.clip(np.rint(along/spacing[axis]), 1, max_step)
        cost = (along/spacing[axis]-step)**2+(across/spacing[1-axis])**2+.15*(step-1)
        if geometry_hint is not None:
            cost += .25*(1-np.clip(scores[neighbors[:, 1:]], 0, 1))
        candidates = ((along > .55*spacing[axis]) & (along < (max_step+.4)*spacing[axis])
                      & (np.abs(along/spacing[axis]-step) < .4)
                      & (across < .45*spacing[1-axis]))
        cost = np.where(candidates, cost, np.inf)
        best = cost.argmin(1)
        row = np.arange(len(xy))
        links.append(np.where(np.isfinite(cost[row, best]), neighbors[:, 1:][row, best], -1))
        costs.append(cost[row, best])
        increments.append(step[row, best].astype(int))
    links, costs, increments = map(np.asarray, (links, costs, increments))
    edges = []
    for d, (axis, sign) in enumerate(directions):
        opposite = (1, 0, 3, 2)[d]
        for i, j in enumerate(links[d]):
            if j <= i or links[opposite, j] != i or increments[d, i] != increments[opposite, j]:
                continue
            amount = sign*increments[d, i]
            step = (0, amount) if axis == 0 else (amount, 0)
            quality = costs[d, i]+costs[opposite, j]+.05*(2-np.clip(scores[i], 0, 1)-np.clip(scores[j], 0, 1))
            edges.append((float(quality), i, int(j), step))
    owner = np.arange(len(xy))
    positions = np.zeros((len(xy), 2), dtype=np.int64)
    members = {i: [i] for i in range(len(xy))}
    occupied = {i: {(0, 0): i} for i in range(len(xy))}
    cycle_rejections, collision_rejections, gap_edges = 0, 0, 0
    for _, i, j, increment in sorted(edges):
        a, b = int(owner[i]), int(owner[j])
        step = np.asarray(increment, dtype=np.int64)
        if a == b:
            cycle_rejections += int(np.any(positions[j]-positions[i] != step))
            continue
        if len(members[a]) < len(members[b]):
            a, b, i, j, step = b, a, j, i, -step
        shift = positions[i]+step-positions[j]
        moved = positions[members[b]]+shift
        keys = [tuple(k) for k in moved]
        if any(k in occupied[a] for k in keys):
            collision_rejections += 1
            continue
        positions[members[b]] = moved
        owner[members[b]] = a
        occupied[a].update(zip(keys, members[b]))
        members[a].extend(members.pop(b))
        occupied.pop(b)
        gap_edges += int(np.max(np.abs(step)) > 1)
    selected = np.array(sorted(max(members.values(), key=len)), dtype=int)
    ids = positions[selected]-positions[selected].min(0)
    lookup = {tuple(k): i for i, k in enumerate(ids)}
    cells, invalid = [], 0
    for row, col in ids:
        corners = ((row, col), (row, col+1), (row+1, col+1), (row+1, col))
        if not all(k in lookup for k in corners):
            continue
        quad = xy[selected[[lookup[k] for k in corners]]]
        edge = np.roll(quad, -1, axis=0)-quad
        after = np.roll(edge, -1, axis=0)
        cross = edge[:, 0]*after[:, 1]-edge[:, 1]*after[:, 0]
        if (cross > .05*spacing[0]*spacing[1]).all():
            cells.append([int(row), int(col)])
        else:
            invalid += 1
    return dict(empty, status='accepted' if cells else 'rejected', reason=None if cells else 'no valid cells',
                selected_indices=selected.tolist(), ids=ids.tolist(), cells=cells,
                component_points=len(selected), components=len(members), spacing_px=spacing,
                orientation_radians=theta, retained_point_fraction=len(selected)/len(xy),
                cycle_constraints_rejected=cycle_rejections, occupancy_collisions_rejected=collision_rejections,
                directed_edges=2*len(edges), cycle_conflicts=cycle_rejections,
                conflict_fraction=cycle_rejections/max(len(edges), 1), invalid_cells=invalid,
                two_step_component_joins=gap_edges, high_confidence_geometry_hint=geometry_hint is not None)
