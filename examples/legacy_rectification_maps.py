"""Frozen bd4f9e3 Delaunay maps retained only as diagnostic baselines."""
import numpy as np
from scipy.spatial import Delaunay

def legacy_grid_remap(rgb, observed, canonical, ids, size, *, allowed_cells=None):
    """Piecewise affine inverse sampling, without crossing missing grid cells.

    Coordinates denote pixel centers. Only triangles joining immediate lattice
    neighbors are supported. Outside the supported mesh stays white and false.
    This offline diagnostic is deliberately outside the torch training hot path.
    """
    import cv2
    from scipy.spatial import Delaunay
    arrays = [np.asarray(value, dtype=np.float64) for value in (observed, canonical, ids)]
    observed, canonical, ids = arrays
    if (any(a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() for a in arrays)
            or not observed.shape == canonical.shape == ids.shape or len(ids) < 3):
        raise ValueError('Matching finite [N,2] correspondences required')
    if len(np.unique(ids, axis=0)) != len(ids) or not np.equal(ids, np.round(ids)).all():
        raise ValueError('Unique integer lattice identities required')
    height, width = size
    triangulation = Delaunay(canonical)
    vertices = ids[triangulation.simplices]
    supported = (np.ptp(vertices, axis=1) <= 1).all(axis=1)
    if allowed_cells is not None:
        allowed = {tuple(cell) for cell in allowed_cells}
        supported &= np.array([tuple(cell) in allowed for cell in vertices.min(axis=1)])
    mapping = np.full((height, width, 2), -1., dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    for start in range(0, height, 64):
        y, x = np.mgrid[start:min(start+64, height), :width]
        query = np.stack((x.ravel(), y.ravel()), -1)
        simplex = triangulation.find_simplex(query)
        keep = (simplex >= 0) & supported[np.maximum(simplex, 0)]
        selected = simplex[keep]
        affine = triangulation.transform[selected]
        first = np.einsum('nij,nj->ni', affine[:, :2], query[keep]-affine[:, 2])
        weights = np.column_stack((first, 1-first.sum(axis=1)))
        sampled = np.einsum('ni,nij->nj', weights, observed[triangulation.simplices[selected]])
        inside = ((sampled >= 0).all(axis=1)
                  & (sampled <= [rgb.shape[1]-1, rgb.shape[0]-1]).all(axis=1))
        indices = np.flatnonzero(keep)[inside]
        mapping[start:start+len(y)].reshape(-1, 2)[indices] = sampled[inside]
        valid[start:start+len(y)].reshape(-1)[indices] = True
    warped = cv2.remap(rgb, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped, valid

def legacy_forward_supported(observed, canonical, ids, query, cells):
    """Map evaluation points through supported triangles without extrapolation."""
    observed, canonical, ids, query = map(np.asarray, (observed, canonical, ids, query))
    if len(observed) < 3:
        return np.full((len(query), 2), np.nan)
    mesh = Delaunay(observed)
    vertices = ids[mesh.simplices]
    allowed = {tuple(c) for c in cells}
    good = (np.ptp(vertices, axis=1) <= 1).all(1)
    good &= np.array([tuple(v) in allowed for v in vertices.min(1)])
    s = mesh.find_simplex(query)
    keep = (s >= 0) & good[np.maximum(s, 0)]
    transform = mesh.transform[s[keep]]
    a = np.einsum('nij,nj->ni', transform[:, :2], query[keep]-transform[:, 2])
    weights = np.column_stack((a, 1-a.sum(1)))
    result = np.full((len(query), 2), np.nan)
    result[keep] = np.einsum('ni,nij->nj', weights, canonical[mesh.simplices[s[keep]]])
    return result
