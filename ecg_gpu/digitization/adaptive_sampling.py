"""Bounded, original-RGB sampling of a fully checked piecewise affine mesh.

Resolution is chosen from the estimated pixel/mm Jacobian and an explicit
canvas budget. A valid mesh is not proof of correct lattice identities or ECG
calibration. When the budget forces downsampling, the receipt says so.
"""

import math

import numpy as np


def _areas(vertices):
    a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def validate_mesh(source, target_mm, triangles, image_shape):
    import shapely
    from shapely.geometry import box
    source, target = np.asarray(source, float), np.asarray(target_mm, float)
    raw_triangles = np.asarray(triangles)
    if (source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape
            or len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Finite corresponding [N,2] mesh vertices required")
    if raw_triangles.ndim != 2 or raw_triangles.shape[1:] != (3,) or not len(raw_triangles):
        raise ValueError("Explicit nonempty triangle connectivity required")
    if not np.equal(raw_triangles, np.round(raw_triangles)).all():
        raise ValueError("Integer triangle indices required")
    tri = np.asarray(raw_triangles, np.int64)
    if tri.min() < 0 or tri.max() >= len(source):
        raise ValueError("Triangle index outside the vertex bank")
    height, width = image_shape[:2]
    if min(height, width) < 2 or not ((source >= -1e-7).all() and (source <= [width - 1 + 1e-7, height - 1 + 1e-7]).all()):
        raise ValueError("Mesh does not match the original image frame")
    sa, ta = _areas(source[tri]), _areas(target[tri])
    if (np.abs(sa) <= 1e-10).any() or (np.abs(ta) <= 1e-10).any() or not (ta / sa > 0).all():
        raise ValueError("Fold, reflection or degenerate mesh triangle")
    source_shapes, target_shapes = shapely.polygons(source[tri]), shapely.polygons(target[tri])
    source_union, target_union = shapely.union_all(source_shapes), shapely.union_all(target_shapes)
    frame = box(0, 0, width - 1, height - 1)
    tolerance = max(1e-7, frame.area * 1e-7)
    if source_union.symmetric_difference(frame).area > tolerance or abs(np.abs(sa).sum() / 2 - frame.area) > tolerance:
        raise ValueError("Source mesh has missing or overlapping frame support")
    if target_union.geom_type != "Polygon" or not target_union.is_valid or len(target_union.interiors):
        raise ValueError("Target mesh has disconnected, holed or invalid support")
    if abs(np.abs(ta).sum() / 2 - target_union.area) > max(1e-7, target_union.area * 1e-7):
        raise ValueError("Target mesh overlaps itself")
    return source, target, tri, {"source_frame_covered": True, "target_boundary_simple": True,
                                "triangle_orientation_consistent": True, "global_triangle_overlap_checked": True,
                                "lattice_identity_accuracy_verified": False}


def choose_density(source, target, triangles, *, max_side=3000, minimum_ppmm=4.0):
    if type(max_side) is not int or not 16 <= max_side <= 4096:
        raise ValueError("The supported output side budget is 16 to 4096 pixels")
    if not math.isfinite(minimum_ppmm) or minimum_ppmm <= 0:
        raise ValueError("A positive finite density is required")
    sv, tv = source[triangles], target[triangles]
    jacobian = np.linalg.solve(tv[:, 1:] - tv[:, :1], sv[:, 1:] - sv[:, :1])
    singular = np.linalg.svd(jacobian, compute_uv=False)[:, 0]
    requested = max(float(singular.max()), minimum_ppmm)
    extent = np.ptp(target, axis=0)
    # Covers floor/ceil and the one-pixel guard on both boundaries.
    effective = min(requested, (max_side - 4) / float(extent.max()))
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError("Degenerate physical extent")
    return effective, {"schema": "adaptive-original-rgb-density-v1", "requested_pixels_per_mm": requested,
                       "effective_pixels_per_mm": effective, "max_side": max_side,
                       "estimated_source_density_quantiles": np.quantile(singular, [0.5, 0.95, 1]).tolist(),
                       "budget_requires_downsampling": effective < requested - 1e-8,
                       "original_rgb_sampled": True, "reference_used": False,
                       "antialias_or_optical_recovery_claimed": False}


def render_mesh(rgb, source, target_mm, triangles, *, max_side=3000, pixels_per_mm=None):
    import cv2
    from matplotlib.tri import Triangulation
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Original uint8 RGB required")
    source, target, tri, checks = validate_mesh(source, target_mm, triangles, image.shape)
    density, report = choose_density(source, target, tri, max_side=max_side)
    if pixels_per_mm is not None:
        if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
            raise ValueError("Positive finite explicit sampling density required")
        density = float(pixels_per_mm)
        report.update(effective_pixels_per_mm=density, explicit_density_override=True,
                      budget_requires_downsampling=density < report["requested_pixels_per_mm"] - 1e-8)
    lower, upper = np.floor(target.min(0) * density) - 1, np.ceil(target.max(0) * density) + 1
    width, height = (upper - lower + 1).astype(int)
    if max(width, height) > max_side or height * width > 16777216:
        raise ValueError("Proposed canvas exceeds the declared pixel budget")
    # Preserve the supplied triangles. Replacing their connectivity by a fresh
    # Delaunay triangulation can silently change a curved paper reconstruction.
    oriented = tri.copy()
    reverse = _areas(target[oriented]) < 0
    oriented[reverse] = oriented[reverse][:, [0, 2, 1]]
    finder = Triangulation(target[:, 0], target[:, 1], triangles=oriented).get_trifinder()
    tv, sv = target[oriented], source[oriented]
    inverse = np.linalg.inv(np.stack((tv[:, 0] - tv[:, 2], tv[:, 1] - tv[:, 2]), axis=-1))
    output = np.full((height, width, 3), 255, np.uint8)
    valid = np.zeros((height, width), bool)
    for start in range(0, height, 64):
        yy, xx = np.mgrid[start:min(height, start + 64), :width]
        q = (np.column_stack((xx.ravel(), yy.ravel())) + lower) / density
        hit = np.asarray(finder(q[:, 0], q[:, 1]))
        keep = hit >= 0
        mapping = np.full((len(q), 2), -1, np.float32)
        indices = hit[keep]
        uv = np.einsum("nij,nj->ni", inverse[indices], q[keep] - tv[indices, 2])
        p = sv[indices, 2] + uv[:, :1] * (sv[indices, 0] - sv[indices, 2]) + uv[:, 1:] * (sv[indices, 1] - sv[indices, 2])
        inside = ((p >= -1e-6) & (p <= [image.shape[1] - 1 + 1e-6, image.shape[0] - 1 + 1e-6])).all(1)
        locations = np.flatnonzero(keep)[inside]
        mapping[locations] = p[inside].clip([0, 0], [image.shape[1] - 1, image.shape[0] - 1])
        maps = mapping.reshape((*yy.shape, 2))
        output[start:start + len(yy)] = cv2.remap(image, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        valid[start:start + len(yy)].reshape(-1)[locations] = True
    report.update(checks, output_shape=list(output.shape), output_origin_mm=(lower / density).tolist(),
                  sampled_output_fraction=float(valid.mean()), sampler="fixed-triangle affine map and OpenCV bilinear RGB sampling")
    return output, valid, report
