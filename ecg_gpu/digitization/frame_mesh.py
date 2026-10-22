"""Constrained completion of an observed cell complex over an input frame.

Observed triangles stay unchanged. Only their complement is triangulated.
Source and target share every triangle. Conflicting completions are rejected.
"""
import numpy as np
from scipy.spatial import Delaunay

from .cell_mesh import cell_triangles


def project(points, matrix):
    value = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return value[:, :2]/value[:, 2:]


def _areas(points):
    a, b = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
    return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]


def complete_frame_mesh(observed, canonical, ids, cells, shape, matrix):
    import shapely
    from shapely.geometry import Polygon, box
    height, width = shape[:2]
    corners = np.array([[0., 0.], [width-1., 0.], [width-1., height-1.], [0., height-1.]])
    outer = project(corners, matrix)
    known = cell_triangles(observed, canonical, ids, cells)
    source_points, target_points = list(observed.copy()), list(canonical.copy())
    source_index = {tuple(np.round(p, 8)): i for i, p in enumerate(source_points)}
    for source, target in zip(corners, outer):
        key = tuple(np.round(source, 8))
        if key in source_index:
            outer[np.flatnonzero((corners == source).all(1))[0]] = target_points[source_index[key]]
        else:
            source_index[key] = len(source_points)
            source_points.append(source)
            target_points.append(target)
    source_points, target_points = list(source_points), list(target_points)
    source_vertices, target_vertices = np.asarray(source_points), np.asarray(target_points)
    orientation = np.sign(_areas(outer[[[0, 1, 2]]])[0])
    if not len(known):
        triangles = Delaunay(source_vertices).simplices
        area_ratio = _areas(target_vertices[triangles])/_areas(source_vertices[triangles])
        if not (area_ratio*orientation > 1e-10).all():
            return None, 'unconstrained point completion folds or collapses'
        return (source_vertices, target_vertices, triangles), None
    shapes = shapely.polygons(observed[known])
    supported = shapely.union_all(shapes)
    if float(shapely.area(shapes).sum()-supported.area) > 1e-7*(height-1)*(width-1):
        return None, 'observed cells overlap in the source image'
    frame = box(0, 0, width-1, height-1)
    unknown = frame.difference(supported)
    pieces = list(unknown.geoms) if hasattr(unknown, 'geoms') else [unknown]
    triangles = known.tolist()

    def coordinate_index(point):
        key = tuple(np.round(point, 8))
        if key not in source_index:
            return None
        return source_index[key]

    for piece in pieces:
        if piece.is_empty or piece.area < 1e-10:
            continue
        if piece.geom_type != 'Polygon':
            continue
        rings = [piece.exterior]+list(piece.interiors)
        target_rings = []
        for ring in rings:
            indices = [coordinate_index(point) for point in np.asarray(ring.coords)[:-1]]
            if None in indices:
                return None, 'cell-boundary intersections lack unique correspondence'
            target_rings.append(np.asarray(target_points)[indices])
        target_piece = Polygon(target_rings[0], target_rings[1:])
        candidates = []
        if target_piece.is_valid and not target_piece.is_empty:
            target_lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(target_points)}
            complete = True
            for triangle in shapely.constrained_delaunay_triangles(target_piece).geoms:
                indices = [target_lookup.get(tuple(np.round(p, 8))) for p in np.asarray(triangle.exterior.coords)[:3]]
                if None in indices:
                    complete = False
                    break
                candidates.append(indices)
            if not complete:
                candidates = []
        options = [candidates] if candidates else []
        source_candidates = []
        for triangle in shapely.constrained_delaunay_triangles(piece).geoms:
            indices = [coordinate_index(p) for p in np.asarray(triangle.exterior.coords)[:3]]
            if None in indices:
                return None, 'unexpected unconstrained interpolation vertex'
            source_candidates.append(indices)
        options.append(source_candidates)
        accepted = None
        for candidate in options:
            candidate = np.asarray(candidate, int).reshape(-1, 3)
            source_area = _areas(source_vertices[candidate])
            target_area = _areas(target_vertices[candidate])
            good = (np.abs(source_area) > 1e-10) & (target_area/source_area*orientation > 1e-10)
            if good.all() and abs(np.abs(source_area).sum()/2-piece.area) <= max(1e-7, piece.area*1e-7):
                accepted = candidate
                break
        if accepted is None:
            return None, 'no orientation-preserving completion of a grid gap'
        triangles.extend(accepted.tolist())
    triangles = np.asarray(triangles, int).reshape(-1, 3)
    source_area = np.abs(_areas(source_vertices[triangles])).sum()/2
    if abs(source_area-frame.area) > frame.area*1e-7:
        return None, 'completed mesh does not cover the entire input frame'
    target_area = np.abs(_areas(target_vertices[triangles])).sum()/2
    if abs(target_area-Polygon(outer).area) > max(1e-7, target_area*1e-7):
        return None, 'completed mesh has incompatible target coverage'
    used = np.unique(triangles)
    reindex = np.full(len(source_vertices), -1, int)
    reindex[used] = np.arange(len(used))
    return (source_vertices[used], target_vertices[used], reindex[triangles]), None
