"""Public-description reconstruction of the Gridder geometry mechanism.

Patent US20240144417A1 describes centroid extraction, nearest-neighbour
directional edges, closed four-edge paths, BFS labels, missing-node regression
and per-cell colour sampling. Numerical choices below are explicit hypotheses.
Geometry alone leaves D4 orientation, absolute origin and pitch unresolved.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree, Delaunay


@dataclass
class Grid:
    points_xy: np.ndarray
    ids_rc: np.ndarray
    observed: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        self.points_xy = np.asarray(self.points_xy, dtype=float)
        ids = np.asarray(self.ids_rc)
        self.observed = np.asarray(self.observed, dtype=bool)
        if self.points_xy.ndim != 2 or self.points_xy.shape[1] != 2 or ids.shape != self.points_xy.shape or self.observed.shape != (len(ids),):
            raise ValueError("Invalid grid array shapes")
        if not np.isfinite(self.points_xy).all() or not np.isfinite(ids).all() or np.any(ids < 0) or not np.equal(ids, np.round(ids)).all():
            raise ValueError("Finite geometry and nonnegative integer relative IDs required")
        self.ids_rc = ids.astype(int)
        if len(np.unique(self.ids_rc, axis=0)) != len(ids):
            raise ValueError("Colliding grid identities")


def extract_centroids(mask, threshold=150 / 255, max_points=2520):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError("Finite single-channel Dotter output required")
    if max_points < 1:
        raise ValueError("Positive point limit required")
    weights = mask.astype(float)
    if np.issubdtype(mask.dtype, np.integer):
        weights /= 255.0
    if weights.min(initial=0) < 0 or weights.max(initial=0) > 1:
        raise ValueError("Mask must contain probabilities or uint8 intensities")
    labels, count = ndimage.label(weights > threshold)
    if not count:
        return np.empty((0, 2))
    indexes = np.arange(1, count + 1)
    areas = ndimage.sum(np.ones_like(weights), labels, indexes)
    indexes = indexes[np.argsort(-areas, kind="stable")[:max_points]]
    centers_yx = np.asarray(ndimage.center_of_mass(weights, labels, indexes))
    return centers_yx[:, ::-1]


def _direction_means(vectors):
    length = np.linalg.norm(vectors, axis=1)
    vectors = vectors[length > 1e-8]
    length = length[length > 1e-8]
    if not len(length):
        raise ValueError("No distinct grid observations")
    # Fourth harmonic estimates square-grid axes modulo quarter rotations.
    near = length <= np.quantile(length, 0.55)
    angle = np.angle(np.mean(np.exp(4j * np.arctan2(vectors[near, 1], vectors[near, 0])))) / 4
    basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    means = np.array([basis[0], basis[1], -basis[0], -basis[1]]) * np.median(length[near])
    for _ in range(20):
        assignment = np.argmin(((vectors[:, None] - means[None]) ** 2).sum(-1), axis=1)
        updated = np.array([np.median(vectors[assignment == i], axis=0) if np.any(assignment == i) else means[i] for i in range(4)])
        if np.allclose(updated, means, rtol=0, atol=1e-7):
            break
        means = updated
    return means


def organize(points_xy, neighbours=4, relative_tolerance=0.65):
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("At least four finite xy points required")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Duplicate observations cannot be assigned separate grid identities")
    k = min(neighbours + 1, len(points))
    _, nn = cKDTree(points).query(points, k=k)
    vectors = points[nn[:, 1:]] - points[:, None]
    means = _direction_means(vectors.reshape(-1, 2))
    edges = np.full((len(points), 4), -1, dtype=int)
    for i in range(len(points)):
        candidates = nn[i, 1:]
        delta = points[candidates] - points[i]
        for direction, mean in enumerate(means):
            scale = max(float(np.linalg.norm(mean)), 1e-8)
            squared = ((delta - mean) ** 2).sum(axis=1) / (scale * relative_tolerance) ** 2
            likelihood = np.exp(-0.5 * squared)
            best = int(np.argmax(likelihood))
            if squared[best] <= 1:
                edges[i, direction] = candidates[best]
    # Retain only reciprocal edges, then require membership in a unit cycle.
    for i in range(len(points)):
        for d in range(4):
            j = edges[i, d]
            if j >= 0 and edges[j, (d + 2) % 4] != i:
                edges[i, d] = -1
    supported = np.zeros_like(edges, dtype=bool)
    cycles = 0
    for start in range(len(points)):
        for first in range(4):
            node, path = start, []
            for d in [(first + j) % 4 for j in range(4)]:
                next_node = edges[node, d]
                if next_node < 0:
                    break
                path.append((node, d))
                node = next_node
            if len(path) == 4 and node == start and len({n for n, _ in path}) == 4:
                cycles += 1
                for n, d in path:
                    supported[n, d] = True
                    supported[edges[n, d], (d + 2) % 4] = True
    edges[~supported] = -1
    # The directional order is right, down, left, up in a RELATIVE chart.
    increments = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
    unseen = set(range(len(points)))
    components = []
    conflicts = 0
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        assigned = {start: np.zeros(2, dtype=int)}
        reverse = {(0, 0): start}
        queue = deque([start])
        valid = True
        while queue:
            i = queue.popleft()
            for d in range(4):
                j = edges[i, d]
                if j < 0:
                    continue
                target = assigned[i] + increments[d]
                if j in assigned:
                    if not np.array_equal(assigned[j], target):
                        valid = False
                        conflicts += 1
                    continue
                key = tuple(target)
                if key in reverse:
                    valid = False
                    conflicts += 1
                assigned[j] = target
                reverse[key] = j
                unseen.discard(j)
                queue.append(j)
        if valid and len(assigned) >= 4:
            components.append(assigned)
    if not components:
        raise ValueError("No consistent cycle-supported grid component")
    chosen = max(components, key=len)
    selection = np.array(sorted(chosen))
    ids = np.array([chosen[i] for i in selection])
    ids -= ids.min(axis=0)
    return Grid(points[selection], ids, np.ones(len(selection), dtype=bool), {
        "input_points": len(points), "retained_points": len(selection),
        "consistent_components": len(components), "unit_cycle_traversals": cycles,
        "identity_conflicts": conflicts, "direction_means_xy": means.tolist(),
        "orientation": "D4_unresolved", "origin": "relative", "physical_pitch": "not_inferred",
        "numerical_choices": {"neighbours": neighbours, "relative_tolerance": relative_tolerance, "direction_model": "global_four_cluster_gaussian_hypothesis"},
    })


def infer_missing(grid, radius=2, minimum_support=6):
    """Local affine regression inside observed hull, with separate inferred flag.

    This is one testable interpretation of the public regression description.
    No inference is performed outside observed support.
    """
    ids = grid.ids_rc
    present = {tuple(x) for x in ids}
    hull = Delaunay(ids)
    missing_ids, missing_points = [], []
    for row in range(int(ids[:, 0].min()), int(ids[:, 0].max()) + 1):
        for col in range(int(ids[:, 1].min()), int(ids[:, 1].max()) + 1):
            query = np.array([row, col])
            if (row, col) in present or hull.find_simplex(query) < 0:
                continue
            nearby = np.max(np.abs(ids - query), axis=1) <= radius
            if nearby.sum() < minimum_support:
                continue
            design = np.column_stack([ids[nearby], np.ones(nearby.sum())])
            if np.linalg.matrix_rank(design) < 3:
                continue
            fit = np.linalg.lstsq(design, grid.points_xy[nearby], rcond=None)[0]
            missing_ids.append(query)
            missing_points.append(np.r_[query, 1] @ fit)
    if not missing_ids:
        return grid
    return Grid(np.concatenate([grid.points_xy, missing_points]), np.concatenate([ids, missing_ids]), np.r_[grid.observed, np.zeros(len(missing_ids), dtype=bool)], {**grid.diagnostics, "inferred_nodes": len(missing_ids), "missing_node_model": "local_affine_inside_observed_hull_hypothesis"})


def warp_cells(image, grid, pixels_per_cell=40, include_inferred=False):
    """Inverse projective sampling per four-corner cell from ORIGINAL RGB.

    Returns a partial diagnostic raster and explicit support. It is not a
    complete-frame preservation algorithm or a physical-accuracy certificate.
    """
    import cv2
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or pixels_per_cell < 2:
        raise ValueError("RGB image and positive cell resolution required")
    coordinates = {tuple(rc): (xy, bool(observed)) for rc, xy, observed in zip(grid.ids_rc, grid.points_xy, grid.observed)}
    height = int(grid.ids_rc[:, 0].max()) * pixels_per_cell
    width = int(grid.ids_rc[:, 1].max()) * pixels_per_cell
    if not height or not width or height * width > 100_000_000:
        raise ValueError("Unsupported output bounds")
    mx, my = np.full((height, width), -1, np.float32), np.full((height, width), -1, np.float32)
    valid = np.zeros((height, width), dtype=bool)
    inferred_support = np.zeros_like(valid)
    cell_corners = np.array([[0, 0], [pixels_per_cell, 0], [pixels_per_cell, pixels_per_cell], [0, pixels_per_cell]], np.float32)
    yy, xx = np.meshgrid(np.arange(pixels_per_cell), np.arange(pixels_per_cell), indexing="ij")
    query = np.stack([xx, yy, np.ones_like(xx)], -1).reshape(-1, 3)
    for (row, col), _ in coordinates.items():
        keys = [(row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col)]
        if not all(key in coordinates for key in keys):
            continue
        observed = all(coordinates[key][1] for key in keys)
        if not observed and not include_inferred:
            continue
        source = np.array([coordinates[key][0] for key in keys], np.float32)
        if not cv2.isContourConvex(source.reshape(-1, 1, 2)) or abs(cv2.contourArea(source)) < 1e-6:
            continue
        transform = cv2.getPerspectiveTransform(cell_corners, source)
        mapped = query @ transform.T
        denominator = mapped[:, 2]
        if np.min(denominator) * np.max(denominator) <= 0:
            continue
        mapped = (mapped[:, :2] / denominator[:, None]).reshape(pixels_per_cell, pixels_per_cell, 2)
        support = (mapped[..., 0] >= 0) & (mapped[..., 0] <= image.shape[1] - 1) & (mapped[..., 1] >= 0) & (mapped[..., 1] <= image.shape[0] - 1)
        rows = slice(row * pixels_per_cell, (row + 1) * pixels_per_cell)
        cols = slice(col * pixels_per_cell, (col + 1) * pixels_per_cell)
        mx[rows, cols], my[rows, cols] = mapped[..., 0], mapped[..., 1]
        valid[rows, cols] = support
        inferred_support[rows, cols] = support & (not observed)
    rgb = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return {"rgb": rgb, "support": valid, "inferred_support": inferred_support, "map_x": mx, "map_y": my}
