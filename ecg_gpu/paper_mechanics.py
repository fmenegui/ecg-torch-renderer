"""Offline compliant sheet relaxation and finite-thickness shell construction.

This CPU asset-preparation module is never imported by the training hot path.
Distance-based bending preserves a prescribed crease rest shape. It is not a
calibrated constitutive model for paper, plasticity or a general cloth solver.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SheetConfig:
    steps: int = 12
    iterations: int = 24
    time_step_s: float = 1 / 120
    stretch_compliance: float = 1e-9
    bend_compliance: float = 1e-5
    damping: float = 0.08
    gravity_m_s2: float = 9.81
    table_clearance_m: float = 0.00012
    max_edge_strain: float = 0.03

    def __post_init__(self):
        for key, high in (("steps", 240), ("iterations", 200)):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= high:
                raise ValueError(f"{key} must be an integer in [1,{high}]")
        for key, value in asdict(self).items():
            if key in ("steps", "iterations"):
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Nonnegative finite {key} required")
        if (
            self.time_step_s <= 0
            or self.damping >= 1
            or not 0 < self.max_edge_strain <= 0.1
        ):
            raise ValueError(
                "Use positive time step, damping <1 and strain tolerance in (0,.1]"
            )


def _constraint_groups(rows, columns, uv, width_m, height_m, rest):
    grid = np.arange(rows * columns).reshape(rows, columns)
    material = uv * [width_m, height_m]
    groups = []
    # Disjoint endpoint coloring makes vectorized updates equivalent to a
    # sequential solve within each group, without duplicate-index scatter.
    for dy, dx, bending in (
        (0, 1, False),
        (1, 0, False),
        (1, 1, False),
        (1, -1, False),
        (0, 2, True),
        (2, 0, True),
    ):
        yy, xx = np.mgrid[:rows, :columns]
        ok = (yy + dy < rows) & (xx + dx >= 0) & (xx + dx < columns)
        period = 2 * max(abs(dx), abs(dy))
        coordinate = xx if abs(dx) else yy
        for color in range(period):
            select = ok & (coordinate % period == color)
            a, b = grid[select], grid[(yy + dy)[select], (xx + dx)[select]]
            if not len(a):
                continue
            delta = (rest[b] - rest[a]) if bending else (material[b] - material[a])
            length = np.linalg.norm(delta, axis=1)
            groups.append((a, b, length, bending))
    return groups


def relax_sheet(vertices, uv, *, rows, columns, width_m, height_m, config=None):
    """XPBD distance constraints, gravity and unilateral table contact.

    Inverse masses are one in solver units. Compliance is illustrative, not a
    measured Young modulus. The rest shape supplies finite crease memory.
    Unresolved strain raises. Self-contact is verified by ``assert_no_crossings``
    and rejected rather than repaired. UV and topology are never reparameterized.
    """
    c = config or SheetConfig()
    if not isinstance(c, SheetConfig):
        raise TypeError("config must be SheetConfig")
    x, uv = (
        np.array(vertices, dtype=np.float64, copy=True),
        np.asarray(uv, dtype=np.float64),
    )
    if x.shape != (rows * columns, 3) or uv.shape != (rows * columns, 2):
        raise ValueError("Regular grid vertices and UV required")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(uv).all()
        or min(width_m, height_m) <= 0
    ):
        raise ValueError("Finite mesh and positive material dimensions required")
    groups = _constraint_groups(rows, columns, uv, width_m, height_m, x)
    velocity = np.zeros_like(x)
    for _ in range(c.steps):
        previous = x.copy()
        velocity[:, 2] -= c.gravity_m_s2 * c.time_step_s
        x += velocity * c.time_step_s
        multipliers = [np.zeros(len(a)) for a, _, _, _ in groups]
        for _ in range(c.iterations):
            for index, (a, b, rest_length, bending) in enumerate(groups):
                delta = x[b] - x[a]
                length = np.linalg.norm(delta, axis=1)
                alpha = (
                    c.bend_compliance if bending else c.stretch_compliance
                ) / c.time_step_s**2
                update = (-(length - rest_length) - alpha * multipliers[index]) / (
                    2 + alpha
                )
                correction = (
                    update[:, None] * delta / np.maximum(length[:, None], 1e-15)
                )
                x[a] -= correction
                x[b] += correction
                multipliers[index] += update
            x[:, 2] = np.maximum(x[:, 2], c.table_clearance_m)
        velocity = (x - previous) / c.time_step_s * (1 - c.damping)
    strain = max(
        float(np.max(np.abs(np.linalg.norm(x[b] - x[a], axis=1) / length - 1)))
        for a, b, length, bending in groups
        if not bending
    )
    if strain > c.max_edge_strain:
        raise RuntimeError(f"Sheet solve did not converge: edge strain {strain:.6g}")
    return x.astype(np.float32), {
        "model": "xpbd-distance-rest-shape-v1",
        "config": asdict(c),
        "max_edge_strain": strain,
        "minimum_table_distance_m": float(x[:, 2].min()),
        "self_contact": "intersections rejected, no collision-response solver",
    }


def _triangles_intersect(a, b, epsilon=1e-10):
    """Separating-axis test, including the coplanar in-plane axes."""
    ea, eb = np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b
    na, nb = np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])
    axes = [na, nb]
    axes.extend(np.cross(x, y) for x in ea for y in eb)
    # In-plane edge normals are required for coplanar triangles. They are also
    # valid separating axes in 3D, so adding them cannot create false overlap.
    axes.extend(np.cross(na, x) for x in ea)
    axes.extend(np.cross(nb, x) for x in eb)
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-15:
            continue
        pa, pb = a @ (axis / norm), b @ (axis / norm)
        if pa.max() < pb.min() - epsilon or pb.max() < pa.min() - epsilon:
            return False
    return True


def assert_no_crossings(vertices, faces):
    """Reject intersecting nonadjacent faces using an AABB sweep broad phase.

    Faces sharing vertices are topological neighbors and are excluded. This
    check does not prove positive clearance or continuous collision safety.
    """
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    triangles = vertices[faces]
    low, high = triangles.min(1), triangles.max(1)
    order = np.argsort(low[:, 0], kind="stable")
    active = np.empty(0, dtype=np.int64)
    for index in order:
        active = active[high[active, 0] >= low[index, 0] - 1e-10]
        candidates = active[
            (high[active, 1:] >= low[index, 1:] - 1e-10).all(1)
            & (high[index, 1:] >= low[active, 1:] - 1e-10).all(1)
        ]
        if len(candidates):
            shared = (faces[candidates, :, None] == faces[index, None, :]).any((1, 2))
            for j in candidates[~shared]:
                if _triangles_intersect(triangles[index], triangles[j]):
                    raise RuntimeError(
                        f"Unresolved sheet self-intersection between faces {j} and {index}"
                    )
        active = np.append(active, index)


def paper_shell(vertices, uv, faces, thickness_m):
    """Offset backside and close every boundary, including cut/hole edges.

    The printed front is retained exactly. Side/back faces have separate shape
    IDs in the ray tracer and cannot acquire ECG labels. Thickness uses averaged
    face normals, so highly curved offsets still need intersection validation.
    """
    if not math.isfinite(thickness_m) or not 0 < thickness_m <= 0.001:
        raise ValueError("Paper thickness must be in (0,.001] meters")
    vertices, uv, faces = np.asarray(vertices), np.asarray(uv), np.asarray(faces)
    normal = np.zeros_like(vertices, dtype=np.float64)
    triangle = vertices[faces]
    face_normal = np.cross(
        triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    )
    for corner in range(3):
        np.add.at(normal, faces[:, corner], face_normal)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    # Clipping can leave unused grid vertices. They never form a shell face.
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 1e-15)
    backside = vertices - thickness_m * normal
    oriented = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(
        np.sort(oriented, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if (counts > 2).any():
        raise ValueError("Nonmanifold paper mesh")
    boundary = oriented[counts[inverse] == 1]
    n = len(vertices)
    side_faces = np.concatenate(
        [
            np.stack([boundary[:, 1], boundary[:, 0], boundary[:, 0] + n], 1),
            np.stack([boundary[:, 1], boundary[:, 0] + n, boundary[:, 1] + n], 1),
        ]
    )
    return {
        "back_vertices": backside.astype(np.float32),
        "back_faces": faces[:, ::-1].copy(),
        "side_vertices": np.concatenate([vertices, backside]).astype(np.float32),
        "side_uv": np.concatenate([uv, uv]),
        "side_faces": side_faces.astype(np.int32),
        "boundary_edges": boundary,
    }


def lift_cut_lips(vertices, uv, faces, *, width_m, height_m, lift_m):
    """Prescribed local lift along interior cut/hole boundaries, preserving UV.

    Outer sheet edges remain unchanged. This models an opened damaged edge,
    not crack propagation. Invalid resulting strain is rejected by the caller.
    """
    if not math.isfinite(lift_m) or not 0 <= lift_m <= 0.003:
        raise ValueError("Cut lip lift must be in [0,.003] meters")
    result = np.array(vertices, copy=True)
    if lift_m == 0:
        return result
    shell = paper_shell(vertices, uv, faces, 0.000001)
    edges = shell["boundary_edges"]
    center = uv[edges].mean(1)
    interior = ((center > 1e-6) & (center < 1 - 1e-6)).all(1)
    edges = edges[interior]
    if not len(edges):
        raise ValueError("Cut lip lift requires an interior damage boundary")
    material = np.asarray(uv, dtype=np.float64) * [width_m, height_m]
    distance = np.full(len(vertices), np.inf)
    for a, b in edges:
        delta = material[b] - material[a]
        t = np.clip(((material - material[a]) @ delta) / (delta @ delta), 0, 1)
        distance = np.minimum(
            distance,
            np.linalg.norm(material - material[a] - t[:, None] * delta, axis=1),
        )
    # A finite millimeter-scale bend tapers smoothly into intact material.
    span = max(0.004, lift_m * 8)
    result[:, 2] += lift_m * np.exp(-((distance / span) ** 2))
    return result
