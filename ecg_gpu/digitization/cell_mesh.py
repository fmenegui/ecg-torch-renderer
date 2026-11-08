"""Offline maps on explicit, complete lattice cells with shared diagonals.

Forward evaluation and inverse RGB sampling use the identical cell complex.
No Delaunay edge is allowed to replace a declared lattice edge or bridge a hole.
This does not certify global injectivity of a photographed folded surface.
"""
import numpy as np


def cell_triangles(observed, canonical, ids, cells=None):
    observed, canonical, ids = (np.asarray(x, float) for x in (observed, canonical, ids))
    if (observed.ndim != 2 or observed.shape[1:] != (2,)
            or observed.shape != canonical.shape or observed.shape != ids.shape
            or not all(np.isfinite(x).all() for x in (observed, canonical, ids))):
        raise ValueError('Matching finite [N,2] correspondences required')
    if not np.equal(ids, np.round(ids)).all() or len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError('Unique integer lattice identities required')
    lookup = {tuple(k): i for i, k in enumerate(ids.astype(np.int64))}
    selected = sorted(lookup) if cells is None else sorted({tuple(c) for c in cells})
    quads = []
    for row, col in selected:
        keys = ((row, col), (row, col+1), (row+1, col+1), (row+1, col))
        if not all(k in lookup for k in keys):
            continue
        quads.append([lookup[k] for k in keys])
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    usable = np.ones(len(quads), bool)
    for points in (observed, canonical):
        vertices = points[quads]
        sides = np.roll(vertices, -1, axis=1)-vertices
        after = np.roll(sides, -1, axis=1)
        cross = sides[..., 0]*after[..., 1]-sides[..., 1]*after[..., 0]
        tolerance = np.maximum(np.max(np.sum(sides*sides, axis=-1), axis=1)*1e-10, 1e-14) if len(quads) else np.empty(0)
        usable &= (cross > tolerance[:, None]).all(1) | (cross < -tolerance[:, None]).all(1)
    quads = quads[usable]
    return np.stack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=1).reshape(-1, 3)


class CellMap:
    """Piecewise affine map with bounded spatial buckets for point location.

    Shared edges are included. Multiple strictly interior hits are ambiguous
    and return missing support instead of silently selecting an overlapping map.
    """
    def __init__(self, source, target, triangles):
        self.source, self.target = np.asarray(source, float), np.asarray(target, float)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.buckets = {}
        if not len(self.triangles):
            return
        vertices = self.source[self.triangles]
        self.base = vertices[:, 2]
        matrix = np.stack((vertices[:, 0]-self.base, vertices[:, 1]-self.base), axis=-1)
        self.inverse = np.linalg.inv(matrix)
        lower, upper = vertices.min(1), vertices.max(1)
        self.origin = lower.min(0)
        self.scale = max(float(np.median(np.max(upper-lower, axis=1))), 1e-6)
        lo = np.floor((lower-self.origin)/self.scale-1e-8).astype(np.int64)
        hi = np.floor((upper-self.origin)/self.scale+1e-8).astype(np.int64)
        entries = np.prod(hi-lo+1, axis=1)
        if np.any(entries > 4096) or entries.sum() > 2000000:
            raise ValueError('Irregular cell geometry exceeds spatial-index bound')
        for i, (a, b) in enumerate(zip(lo, hi)):
            for x in range(int(a[0]), int(b[0])+1):
                for y in range(int(a[1]), int(b[1])+1):
                    self.buckets.setdefault((x, y), []).append(i)

    def __call__(self, query):
        query = np.asarray(query, float)
        if query.ndim != 2 or query.shape[1:] != (2,) or not np.isfinite(query).all():
            raise ValueError('Finite [N,2] query points required')
        result = np.full(query.shape, np.nan)
        if not len(self.triangles) or not len(query):
            return result
        bins = np.floor((query-self.origin)/self.scale).astype(np.int64)
        keys, inverse = np.unique(bins, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind='stable')
        offsets = np.r_[0, np.cumsum(np.bincount(inverse))]
        for group, key in enumerate(keys):
            candidates = self.buckets.get(tuple(key))
            if not candidates:
                continue
            candidates = np.asarray(candidates, dtype=int)
            indices = order[offsets[group]:offsets[group+1]]
            # Keep intermediate products bounded even for dense query images.
            chunk = max(1, 500000//len(candidates))
            for start in range(0, len(indices), chunk):
                selected = indices[start:start+chunk]
                delta = query[selected, None]-self.base[candidates]
                first = np.einsum('tij,qtj->qti', self.inverse[candidates], delta)
                weights = np.concatenate((first, 1-first.sum(-1, keepdims=True)), axis=-1)
                inside = ((weights >= -1e-8) & (weights <= 1+1e-8)).all(-1)
                ambiguous = (weights > 1e-8).all(-1).sum(-1) > 1
                keep = inside.any(-1) & ~ambiguous
                chosen = inside.argmax(-1)[keep]
                barycentric = weights[np.flatnonzero(keep), chosen]
                vertices = self.target[self.triangles[candidates[chosen]]]
                result[selected[keep]] = np.einsum('ni,nij->nj', barycentric, vertices)
        return result


def forward_cell_map(observed, canonical, ids, query, cells):
    triangles = cell_triangles(observed, canonical, ids, cells)
    return CellMap(observed, canonical, triangles)(query)


def remap_cells(rgb, observed, canonical, ids, size, *, allowed_cells=None):
    """Sample original RGB through exactly the cells used by forward evaluation."""
    import cv2
    if len(size) != 2 or any(int(v) != v or v < 1 for v in size):
        raise ValueError('Positive integer output height and width required')
    height, width = map(int, size)
    if height*width > 16777216:
        raise ValueError('Output canvas exceeds diagnostic pixel bound')
    triangles = cell_triangles(observed, canonical, ids, allowed_cells)
    mapping = CellMap(canonical, observed, triangles)
    reverse = CellMap(observed, canonical, triangles)
    sampled = np.full((height, width, 2), -1., np.float32)
    valid = np.zeros((height, width), bool)
    for start in range(0, height, 64):
        y, x = np.mgrid[start:min(start+64, height), :width]
        xy = mapping(np.column_stack((x.ravel(), y.ravel())))
        inside = (np.isfinite(xy).all(1) & (xy >= 0).all(1)
                  & (xy <= [rgb.shape[1]-1, rgb.shape[0]-1]).all(1))
        if inside.any():
            indices = np.flatnonzero(inside)
            back = reverse(xy[inside])
            query = np.column_stack((x.ravel(), y.ravel()))[inside]
            consistent = (np.isfinite(back).all(1)
                          & (np.linalg.norm(back-query, axis=1) <= 1e-5*max(height, width, 1)))
            inside[indices[~consistent]] = False
        flat = sampled[start:start+len(y)].reshape(-1, 2)
        flat[inside] = xy[inside]
        valid[start:start+len(y)].reshape(-1)[inside] = True
    warped = cv2.remap(rgb, sampled[..., 0], sampled[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped, valid
