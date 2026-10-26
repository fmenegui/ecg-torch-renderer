"""Offline relative grid IDs from competing, image-only lattice hypotheses.

The detector must supply major 5 mm intersections. Integer constraints retain
observed cells only. A bounded projective consensus can align disconnected
components without creating nodes in a gap. The grid alone cannot identify
semantic page orientation, absolute origin or the physical pitch harmonic.
"""

from __future__ import annotations

import numpy as np


def _axes_hypotheses(delta, distance, scores, neighbors):
    from scipy.ndimage import gaussian_filter1d

    length = np.linalg.norm(delta, axis=-1)
    scale = float(np.median(distance[:, 1]))
    keep = (length > 0.55 * scale) & (length < 3 * scale)
    angle = np.arctan2(delta[..., 1], delta[..., 0])
    weight = np.sqrt(
        np.clip(scores, 0, 1)[:, None] * np.clip(scores[neighbors[:, 1:]], 0, 1)
    )
    weight = weight / np.maximum(length / scale, 0.55) ** 2
    bins = np.floor(np.remainder(angle, np.pi) * 180 / np.pi).astype(int) % 180
    histogram = np.bincount(bins[keep], weights=weight[keep], minlength=180)
    histogram = gaussian_filter1d(histogram.astype(float), 2.0, mode="wrap")
    peaks = []
    for i in np.argsort(-histogram, kind="stable"):
        if all(min(abs(int(i) - p), 180 - abs(int(i) - p)) >= 18 for p in peaks):
            peaks.append(int(i))
        if len(peaks) == 5:
            break
    hypotheses = []
    pairs = []
    for i, a in enumerate(peaks):
        for b in peaks[i + 1 :]:
            separation = min(abs(a - b), 180 - abs(a - b))
            if 50 <= separation <= 90:
                pairs.append((histogram[a] * histogram[b], a, b))
    for _, a, b in sorted(pairs, reverse=True)[:4]:
        theta = (np.asarray([a, b]) + 0.5) * np.pi / 180
        axes = np.column_stack((np.cos(theta), np.sin(theta)))
        if abs(axes[1, 0]) > abs(axes[0, 0]):
            axes = axes[::-1]
        if axes[0, 0] < 0:
            axes[0] *= -1
        if np.linalg.det(axes) < 0:
            axes[1] *= -1
        hypotheses.append(axes)
    near = (length > 0.55 * scale) & (length < 1.3 * scale)
    direction = np.mean(np.exp(4j * angle[near])) if near.any() else 0j
    if abs(direction) >= 0.1:
        theta = np.angle(direction) / 4
        hypotheses.append(
            np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        )
    return hypotheses


def _edges(xy, scores, distance, neighbors, axes):
    delta = (xy[neighbors[:, 1:]] - xy[:, None]) @ np.linalg.inv(axes)
    scale = float(np.median(distance[:, 1]))
    spacing = []
    local = []
    for axis in (0, 1):
        along = np.abs(delta[..., axis])
        across = np.abs(delta[..., 1 - axis])
        usable = (along > 0.5 * scale) & (along < 3.5 * scale) & (across < 0.3 * along)
        nearest = np.where(usable, along, np.inf).min(1)
        finite = nearest[np.isfinite(nearest)]
        if len(finite) < 4:
            return [], None, None
        pitch = float(np.median(finite))
        spacing.append(pitch)
        # Nearby estimates follow slow perspective variation. The median does
        # not turn one missing immediate neighbor into a doubled local pitch.
        candidate = np.where(
            (nearest > 0.5 * pitch) & (nearest < 1.7 * pitch), nearest, np.nan
        )
        neighbor_values = candidate[neighbors[:, : min(9, neighbors.shape[1])]]
        values = np.full(len(xy), pitch)
        valid = np.isfinite(neighbor_values).any(1)
        if valid.any():
            values[valid] = np.nanmedian(neighbor_values[valid], axis=1)
        local.append(values)
    local = np.column_stack(local)
    links, costs, increments = [], [], []
    row = np.arange(len(xy))
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        along = sign * delta[..., axis] / local[:, axis, None]
        across = np.abs(delta[..., 1 - axis]) / local[:, 1 - axis, None]
        step = np.clip(np.rint(along), 1, 2).astype(int)
        valid = (
            (along > 0.6)
            & (along < 2.35)
            & (abs(along - step) < 0.35)
            & (across < 0.38)
        )
        quality = (along - step) ** 2 + across**2 + 0.12 * (step - 1)
        quality += 0.03 * (1 - np.clip(scores[neighbors[:, 1:]], 0, 1))
        quality = np.where(valid, quality, np.inf)
        best = quality.argmin(1)
        links.append(
            np.where(np.isfinite(quality[row, best]), neighbors[:, 1:][row, best], -1)
        )
        costs.append(quality[row, best])
        increments.append(step[row, best])
    links, costs, increments = map(np.asarray, (links, costs, increments))
    edges = []
    for d, (axis, sign) in enumerate(((0, 1), (0, -1), (1, 1), (1, -1))):
        opposite = (1, 0, 3, 2)[d]
        for i, j in enumerate(links[d]):
            if (
                j <= i
                or links[opposite, j] != i
                or increments[d, i] != increments[opposite, j]
            ):
                continue
            amount = int(sign * increments[d, i])
            step = (0, amount) if axis == 0 else (amount, 0)
            edges.append((float(costs[d, i] + costs[opposite, j]), i, int(j), step))
    return sorted(edges), spacing, local


def _integer_components(count, edges):
    owner = np.arange(count)
    positions = np.zeros((count, 2), dtype=np.int64)
    members = {i: [i] for i in range(count)}
    occupied = {i: {(0, 0)} for i in range(count)}
    conflicts = 0
    collisions = 0
    for _, i, j, increment in edges:
        a, b = int(owner[i]), int(owner[j])
        step = np.asarray(increment, dtype=np.int64)
        if a == b:
            conflicts += int(np.any(positions[j] - positions[i] != step))
            continue
        if len(members[a]) < len(members[b]):
            a, b, i, j, step = b, a, j, i, -step
        moved = positions[members[b]] + positions[i] + step - positions[j]
        keys = [tuple(k) for k in moved]
        if any(k in occupied[a] for k in keys):
            collisions += 1
            continue
        positions[members[b]] = moved
        owner[members[b]] = a
        occupied[a].update(keys)
        members[a].extend(members.pop(b))
        occupied.pop(b)
    components = sorted(members.values(), key=lambda m: (-len(m), min(m)))
    return positions, components, conflicts, collisions


def _cells(xy, selected, ids, spacing, axes):
    lookup = {tuple(k): i for i, k in enumerate(ids)}
    inverse = np.linalg.inv(axes)
    cells = []
    for r, c in sorted(lookup):
        keys = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
        if not all(k in lookup for k in keys):
            continue
        quad = xy[selected[[lookup[k] for k in keys]]]
        sides = np.roll(quad, -1, axis=0) - quad
        after = np.roll(sides, -1, axis=0)
        cross = sides[:, 0] * after[:, 1] - sides[:, 1] * after[:, 0]
        projected = sides @ inverse
        # No cell may span a missing row, wrap a fold or collapse to a line.
        expected = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        normalized = projected / np.asarray(spacing)
        if (cross > 0.03 * np.prod(spacing)).all() and np.max(
            abs(normalized - expected)
        ) < 0.65:
            cells.append([int(r), int(c)])
    return cells


def _align_components(xy, positions, components, spacing):
    """Align observed components only with bounded, redundant grid evidence."""
    import cv2
    from scipy.spatial import cKDTree

    selected = list(components[0])
    merged = 0
    rejected = []
    if len(selected) < 16:
        return np.array(sorted(selected)), positions, merged, len(components) - 1
    for component in components[1:]:
        if len(component) < 4:
            rejected.append(component)
            continue
        source = np.asarray(sorted(selected))
        nearest = cKDTree(xy[source]).query(xy[component])[0]
        if np.count_nonzero(nearest <= 4 * max(spacing)) < 4:
            rejected.append(component)
            continue
        cv2.setRNGSeed(20261026)
        matrix, inliers = cv2.findHomography(
            xy[source],
            positions[source, ::-1].astype(float),
            cv2.RANSAC,
            0.22,
            maxIters=1000,
            confidence=0.995,
        )
        if matrix is None or inliers is None or inliers.mean() < 0.7:
            rejected.append(component)
            continue
        homogeneous = (
            np.column_stack((xy[component], np.ones(len(component)))) @ matrix.T
        )
        if (abs(homogeneous[:, 2]) < 1e-8).any():
            rejected.append(component)
            continue
        predicted = (homogeneous[:, :2] / homogeneous[:, 2:])[:, ::-1]
        shift = np.rint(np.median(predicted - positions[component], axis=0)).astype(
            np.int64
        )
        moved = positions[component] + shift
        residual = np.linalg.norm(predicted - moved, axis=1)
        occupied = {tuple(k) for k in positions[source]}
        if np.quantile(residual, 0.9) > 0.22 or any(
            tuple(k) in occupied for k in moved
        ):
            rejected.append(component)
            continue
        positions[component] = moved
        selected.extend(component)
        merged += 1
    return np.array(sorted(selected)), positions, merged, len(rejected)


def organize_consensus_lattice(xy, scores=None):
    """Infer relative IDs without reference geometry or a trained ID predictor.

    Returns the same selected_indices/ids/cells contract as organize_lattice.
    Semantic page orientation is unresolved. Output axes follow image right
    and image down, with an assumed 5 mm major-grid pitch at the caller.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1:] != (2,) or not np.isfinite(xy).all():
        raise ValueError("Finite [N,2] image points required")
    scores = np.ones(len(xy)) if scores is None else np.asarray(scores, float)
    if scores.shape != (len(xy),) or not np.isfinite(scores).all():
        raise ValueError("Finite per-point confidence required")
    empty = {
        "status": "rejected",
        "reason": "insufficient or degenerate grid",
        "input_points": len(xy),
        "selected_indices": [],
        "ids": [],
        "cells": [],
        "organizer": "consensus",
        "oracle_correspondence": False,
        "global_injectivity_checked": False,
        "semantic_orientation_inferred": False,
        "assumed_major_grid_mm": 5.0,
        "hypotheses": [],
    }
    if len(xy) < 4 or len(np.unique(xy, axis=0)) != len(xy):
        return empty
    if len(xy) > 16384:
        raise ValueError("At most 16384 grid detections supported")
    # Geometric sorting makes tied graph traversal independent of input order.
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    points, confidence = xy[order], scores[order]
    distance, neighbors = cKDTree(points).query(points, k=min(25, len(points)))
    if np.median(distance[:, 1]) <= 1e-8:
        return empty
    delta = points[neighbors[:, 1:]] - points[:, None]
    proposals = []
    for axes in _axes_hypotheses(delta, distance, confidence, neighbors):
        edges, spacing, local = _edges(points, confidence, distance, neighbors, axes)
        if spacing is None:
            continue
        positions, components, conflicts, collisions = _integer_components(
            len(points), edges
        )
        selected, positions, merged, omitted = _align_components(
            points, positions, components, spacing
        )
        ids = positions[selected] - positions[selected].min(0)
        cells = _cells(points, selected, ids, spacing, axes)
        # Supported cell count is an image-only model selection criterion.
        # Confidence and cycle residuals only break close structural ties.
        merit = len(cells) - 2 * (conflicts + collisions)
        diagnostic = {
            "cells": len(cells),
            "selected_points": len(selected),
            "edges": len(edges),
            "cycle_rejections": conflicts,
            "occupancy_rejections": collisions,
            "components": len(components),
            "merged_components": merged,
            "omitted_components": omitted,
            "spacing_px": spacing,
            "axes": axes.tolist(),
            "merit": merit,
        }
        proposals.append(
            (
                merit,
                len(cells),
                float(confidence[selected].sum()),
                selected,
                ids,
                cells,
                diagnostic,
            )
        )
    if not proposals:
        return empty
    best = max(range(len(proposals)), key=lambda i: proposals[i][:3])
    _, _, _, selected, ids, cells, diagnostic = proposals[best]
    original = order[selected]
    reorder = np.argsort(original)
    return dict(
        empty,
        status="accepted" if cells else "rejected",
        reason=None if cells else "no supported cells",
        selected_indices=original[reorder].tolist(),
        ids=ids[reorder].tolist(),
        cells=cells,
        partial_status="partial" if len(cells) >= 16 else "rejected",
        local_filter=True,
        component_points=len(selected),
        retained_point_fraction=len(selected) / len(xy),
        components=diagnostic["components"],
        merged_components=diagnostic["merged_components"],
        spacing_px=diagnostic["spacing_px"],
        axes=diagnostic["axes"],
        orientation_radians=float(
            np.arctan2(diagnostic["axes"][0][1], diagnostic["axes"][0][0])
        ),
        cycle_conflicts=diagnostic["cycle_rejections"],
        conflict_fraction=diagnostic["cycle_rejections"] / max(diagnostic["edges"], 1),
        directed_edges=2 * diagnostic["edges"],
        selected_hypothesis=best,
        hypotheses=[p[-1] for p in proposals],
    )
