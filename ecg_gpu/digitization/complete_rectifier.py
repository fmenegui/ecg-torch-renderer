"""Offline full-frame geometric maps with evidence separate from visible RGB.

An observed-cell mask is evidence, not permission to erase input pixels.
The fitted mesh covers the entire input rectangle. Its boundary stays simple
and every triangle must preserve the orientation of a finite global map.
Unknown geometry stays explicitly inferred. A rejected fit preserves the input.
"""
import numpy as np
from scipy.spatial import cKDTree

from .cell_mesh import CellMap, cell_triangles


def _project(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


class CompleteMap:
    """One shared mesh for full-image inverse sampling and forward auditing."""
    def __init__(self, source, target, ids, triangles, report, observed, canonical, grid_ids, cells):
        self.source, self.target, self.ids, self.triangles = source, target, ids, triangles
        self.report = report
        self.forward = CellMap(source, target, triangles)
        self.inverse = CellMap(target, source, triangles)
        strict_triangles = cell_triangles(observed, canonical, grid_ids, cells)
        self.observed_cells = CellMap(observed, observed, strict_triangles)
        self.cell_coordinates = CellMap(observed, canonical, strict_triangles)
        self.observed = observed
        vertices = np.unique(strict_triangles) if len(strict_triangles) else np.arange(len(observed))
        point_error = np.linalg.norm(self.forward(observed[vertices])-canonical[vertices], axis=1)
        self.nearby_points = observed[vertices[point_error <= .2]]
        self.tree = cKDTree(self.nearby_points) if len(self.nearby_points) >= 4 else None
        if self.tree is not None:
            distances = self.tree.query(self.nearby_points, k=min(5, len(self.nearby_points)))[0]
            self.local_spacing = np.median(distances[:, 1:], axis=1)

    def eligible_geometry(self, points, *, radius=.75, mapped=None):
        """Conservative coordinate eligibility. It never masks the visible image.

        Observed cells must agree with their original local map. Inferred holes
        must lie inside the observed hull and close to consistent grid evidence.
        This is a heuristic, with geometric error measured separately.
        """
        local = self.cell_coordinates(points)
        observed = np.isfinite(local).all(1)
        prediction = self.forward(points) if mapped is None else mapped
        agreement = np.linalg.norm(local-prediction, axis=1) <= .05
        result = observed & agreement
        if self.tree is not None:
            distance, nearest = self.tree.query(points)
            nearby = distance <= radius*self.local_spacing[nearest]
            result |= ~observed & nearby & (self.evidence(points) == 2)
        return result & np.isfinite(prediction).all(1)

    def evidence(self, points):
        """1 extrapolated, 2 interpolated, 3 observed cell. None certifies accuracy."""
        import cv2
        hull = cv2.convexHull(self.observed.astype(np.float32)).reshape(-1, 2)
        edges = np.roll(hull, -1, axis=0)-hull
        delta = points[:, None]-hull
        cross = edges[None, :, 0]*delta[..., 1]-edges[None, :, 1]*delta[..., 0]
        inside = (cross >= -1e-5).all(1) | (cross <= 1e-5).all(1)
        result = np.where(inside, 2, 1).astype(np.uint8)
        result[np.isfinite(self.observed_cells(points)).all(1)] = 3
        return result


def fit_complete_map(observed, canonical, ids, shape, scores=None, cells=None, *, global_prior='projective'):
    """Fit image-only geometry. No gold page mask, trace, or field is accepted."""
    import cv2
    observed, canonical = np.asarray(observed, float), np.asarray(canonical, float)
    ids = np.asarray(ids).reshape(-1, 2)
    if (observed.ndim != 2 or observed.shape[1:] != (2,) or canonical.shape != observed.shape
            or ids.shape != observed.shape or not np.isfinite(observed).all()
            or not np.isfinite(canonical).all()):
        raise ValueError('Finite corresponding [N,2] grid coordinates required')
    if len(shape) < 2 or min(shape[:2]) < 2:
        raise ValueError('Nondegenerate image required')
    if global_prior not in ('affine', 'projective'):
        raise ValueError('Global prior must be affine or projective')
    scores = np.ones(len(observed)) if scores is None else np.asarray(scores, float)
    if scores.shape != (len(observed),) or not np.isfinite(scores).all():
        raise ValueError('Finite per-point confidence required')
    if not ((observed >= 0).all() and (observed <= [shape[1]-1, shape[0]-1]).all()):
        raise ValueError('Observed points must lie in the input image')
    rejection = {'status': 'original-preserved', 'reason': 'insufficient 2D grid evidence',
                 'physical_coordinates_available': False, 'rgb_source': 'original input',
                 'grid_mask_erases_rgb': False, 'geometry_is_inferred': True}
    if len(observed) < 16 or min(len(np.unique(ids[:, k])) for k in (0, 1)) < 4:
        return None, rejection
    matrix, inliers = cv2.findHomography(observed, canonical, cv2.RANSAC, 1.5, maxIters=2000, confidence=.995)
    if matrix is None or inliers is None:
        return None, dict(rejection, reason='no robust global geometric fit')
    inliers = inliers.ravel().astype(bool)
    supported = np.zeros(len(ids), bool)
    if cells is not None:
        supported[np.unique(cell_triangles(observed, canonical, ids, cells))] = True
    minimum_fraction = .35 if supported.sum() >= 16 else .6
    if inliers.mean() < minimum_fraction or inliers.sum() < 16:
        return None, dict(rejection, reason='inconsistent grid identities', global_inlier_fraction=float(inliers.mean()))
    height, width = shape[:2]
    corners = np.array([[0., 0.], [width-1., 0.], [width-1., height-1.], [0., height-1.]])
    denominator = np.column_stack((corners, np.ones(4))) @ matrix[2]
    if np.min(np.abs(denominator)) < 1e-6 or not ((denominator > 0).all() or (denominator < 0).all()):
        return None, dict(rejection, reason='projective pole crosses the input frame')
    if global_prior == 'affine':
        # The background is not coplanar with the paper. Extending a projective
        # fit from a small detected patch can massively stretch unseen regions.
        # A finite affine background prior leaves perspective to the local field.
        affine = np.linalg.lstsq(np.column_stack((observed[inliers], np.ones(inliers.sum()))),
                                  canonical[inliers], rcond=None)[0].T
        matrix = np.vstack((affine, [0., 0., 1.]))
        if abs(np.linalg.det(matrix)) < 1e-9:
            return None, dict(rejection, reason='degenerate full-frame affine prior')
    from .frame_mesh import complete_frame_mesh
    mesh, reason = complete_frame_mesh(observed, canonical, ids, cells, shape, matrix)
    if mesh is None:
        return None, dict(rejection, reason=reason)
    nodes, target, triangles = mesh
    scale = np.ptp(target, axis=0).max()
    if scale > 3000:
        return None, dict(rejection, reason='unbounded projected image extent')
    source_triangles, target_triangles = nodes[triangles], target[triangles]

    def signed_area(vertices):
        a, b = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
        return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]

    reverse = signed_area(source_triangles) < 0
    triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    source_triangles, target_triangles = nodes[triangles], target[triangles]
    ratio = signed_area(target_triangles)/signed_area(source_triangles)
    if not ((ratio > 1e-10).all() or (ratio < -1e-10).all()):
        return None, dict(rejection, reason='completed map changes orientation')
    baseline = _project(nodes, matrix)[triangles]
    relative_ratio = ratio/(signed_area(baseline)/signed_area(source_triangles))
    report = dict(rejection, status='full-frame-rectification', reason=None,
        physical_coordinates_available=True, global_inlier_fraction=float(inliers.mean()),
        method='constrained completion preserving original observed-cell triangles',
        global_prior=global_prior, mesh_triangles=len(triangles),
        minimum_triangle_area_ratio=float(relative_ratio.min()),
        boundary='full source image boundary, with finite projective target',
        boundary_self_intersections_checked=True, all_mesh_triangles_orientation_checked=True,
        mesh_node_layout='observed vertices and full-frame boundary',
        observed_cell_map_changed=False,
        output_boundary_crops_to_detected_grid=False, evidence_is_accuracy_certificate=False)
    try:
        mapping = CompleteMap(nodes, target, np.empty((0, 2), int), triangles, report,
                              observed, canonical, ids, cells)
    except (ValueError, np.linalg.LinAlgError) as error:
        return None, dict(rejection, reason='complete map cannot be indexed safely', detail=str(error))
    return mapping, report


def render_complete(rgb, mapping, *, pixels_per_mm=8., max_side=4096, return_eligibility=False):
    """Return original RGB samples and a separate evidence image, never a grid cutout."""
    import cv2
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('uint8 HWC RGB required')
    if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError('Positive finite output density required')
    if mapping is None:
        result = (rgb.copy(), np.ones(rgb.shape[:2], bool), np.zeros(rgb.shape[:2], np.uint8), {
            'status': 'original-preserved', 'resampled': False, 'source_coverage': 1.,
            'physical_coordinates_available': False})
        return result+(np.zeros(rgb.shape[:2], bool),) if return_eligibility else result
    lower = np.floor(mapping.target.min(0)*pixels_per_mm)-1
    upper = np.ceil(mapping.target.max(0)*pixels_per_mm)+1
    width, height = (upper-lower+1).astype(int)
    if max(width, height) > max_side or height*width > 16777216:
        result = (rgb.copy(), np.ones(rgb.shape[:2], bool), np.zeros(rgb.shape[:2], np.uint8), {
            'status': 'original-preserved', 'reason': 'full projected frame exceeds output bound',
            'resampled': False, 'source_coverage': 1., 'physical_coordinates_available': False})
        return result+(np.zeros(rgb.shape[:2], bool),) if return_eligibility else result
    sampled = np.full((height, width, 2), -1., np.float32)
    evidence = np.zeros((height, width), np.uint8)
    valid = np.zeros((height, width), bool)
    eligible = np.zeros((height, width), bool) if return_eligibility else None
    for start in range(0, height, 48):
        yy, xx = np.mgrid[start:min(start+48, height), :width]
        physical = (np.column_stack((xx.ravel(), yy.ravel()))+lower)/pixels_per_mm
        source = mapping.inverse(physical)
        inside = (np.isfinite(source).all(1) & (source >= -1e-6).all(1)
                  & (source <= [rgb.shape[1]-1+1e-6, rgb.shape[0]-1+1e-6]).all(1))
        points = source[inside].clip([0, 0], [rgb.shape[1]-1, rgb.shape[0]-1])
        sampled[start:start+len(yy)].reshape(-1, 2)[inside] = points
        valid[start:start+len(yy)].reshape(-1)[inside] = True
        evidence[start:start+len(yy)].reshape(-1)[inside] = mapping.evidence(points)
        if return_eligibility:
            eligible[start:start+len(yy)].reshape(-1)[inside] = mapping.eligible_geometry(points, mapped=physical[inside])
    warped = cv2.remap(rgb, sampled[..., 0], sampled[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    metadata = dict(mapping.report, output_origin_mm=(lower/pixels_per_mm).tolist(),
        pixels_per_mm=float(pixels_per_mm), resampled=True,
        sampling_valid_fraction=float(valid.mean()),
        observed_cell_fraction=float((evidence == 3).mean()),
        interpolated_fraction=float((evidence == 2).mean()),
        extrapolated_fraction=float((evidence == 1).mean()))
    if return_eligibility:
        metadata['geometric_eligibility_fraction'] = float(eligible.mean())
        metadata['eligibility_is_accuracy_certificate'] = False
        return warped, valid, evidence, metadata, eligible
    return warped, valid, evidence, metadata
