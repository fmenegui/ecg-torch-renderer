"""Offline fixed area quadrature, composed directly to the original image.

No detector, geometry estimator, intermediate warp or training-path integration.
NumPy/OpenCV work here is deliberately outside the torch rendering hot path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import cv2
import numpy as np


Inverse = Callable[[np.ndarray], np.ndarray]
Trust = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class OutputGrid:
    """Physical pixel centres and their rectangular integration cells.

    ``shape_hw`` is (height, width), while origin and spacing are (x, y).
    Pixel (row, col) has centre origin + (col*dx, row*dy), with cell
    [centre_x-dx/2, centre_x+dx/2] x [centre_y-dy/2, centre_y+dy/2].
    """

    shape_hw: tuple[int, int]
    origin_mm: tuple[float, float]
    spacing_mm: tuple[float, float]

    def __post_init__(self):
        if (len(self.shape_hw) != 2 or any(isinstance(v, bool) or
                not isinstance(v, (int, np.integer)) or v <= 0 for v in self.shape_hw)):
            raise ValueError("shape_hw must contain two positive integers")
        if (len(self.origin_mm) != 2 or len(self.spacing_mm) != 2 or
                not np.isfinite(self.origin_mm).all() or
                not np.isfinite(self.spacing_mm).all() or min(self.spacing_mm) <= 0):
            raise ValueError("origin and positive spacing must be finite (x,y) pairs")


@dataclass
class SamplingStats:
    """Deterministic working-set counters, separate from process peak RSS.

    ``max_accounted_array_bytes`` sums the major arrays simultaneously owned
    inside one quadrature iteration. It is an audit counter, NOT an allocator
    or process-memory bound. OpenCV and user callbacks can allocate internally.
    Coordinate plane count never depends on subgrid**2.
    """

    tiles: int = 0
    inverse_calls: int = 0
    inverse_points: int = 0
    max_query_points: int = 0
    max_coordinate_bytes: int = 0
    max_accounted_array_bytes: int = 0
    source_float_bytes: int = 0


@dataclass
class SampleTile:
    rows: slice
    cols: slice
    image: np.ndarray
    finite_geometry_fraction: np.ndarray
    source_cell_fraction: np.ndarray
    full_reconstruction_support_fraction: np.ndarray
    trusted_geometry_fraction: np.ndarray | None


@dataclass
class SampledImage:
    image: np.ndarray
    finite_geometry_fraction: np.ndarray
    source_cell_fraction: np.ndarray
    full_reconstruction_support_fraction: np.ndarray
    trusted_geometry_fraction: np.ndarray | None
    stats: SamplingStats


def _tiles(shape_hw, budget):
    height, width = shape_hw
    # OpenCV remap also requires each destination dimension < SHRT_MAX.
    cols = min(width, budget, 32766)
    rows = min(height, max(1, budget // cols), 32766)
    for top in range(0, height, rows):
        for left in range(0, width, cols):
            yield slice(top, min(top+rows, height)), slice(left, min(left+cols, width))


def _validate_source(source):
    if (not isinstance(source, np.ndarray) or source.dtype != np.uint8 or
            source.ndim != 3 or source.shape[2] not in (3, 4) or
            min(source.shape[:2]) < 1 or max(source.shape[:2]) >= 32767):
        raise ValueError("source must be nonempty uint8 RGB/RGBA, dimensions <32767")


def _source(source, fill_rgb):
    _validate_source(source)
    if (len(fill_rgb) != 3 or not np.isfinite(fill_rgb).all() or
            any(float(x) < 0 or float(x) > 255 for x in fill_rgb)):
        raise ValueError("fill_rgb must be three finite encoded values in [0,255]")
    floating = source.astype(np.float32)
    if source.shape[2] == 4:
        # Preserve alpha through integration. Hidden RGB has zero weight.
        floating[..., :3] *= floating[..., 3:4] / np.float32(255)
        fill = (0., 0., 0., 0.)
    else:
        fill = tuple(float(x) for x in fill_rgb)
    return floating, fill


def iter_area_tiles(
    source_u8: np.ndarray,
    inverse: Inverse,
    grid: OutputGrid,
    *,
    subgrid: int = 2,
    reconstruction: str = "bilinear",
    max_tile_pixels: int = 16384,
    fill_rgb: tuple[float, float, float] = (255., 255., 255.),
    trust: Trust | None = None,
    stats: SamplingStats | None = None,
) -> Iterator[SampleTile]:
    """Yield bounded tiles with RGB retention independent of geometry trust.

    ``inverse`` accepts Nx2 physical (x,y) centres and returns Nx2 original
    source pixel-centre coordinates, with pixel (row,col) at (col,row).
    Non-finite results are invalid. ``trust`` accepts the same physical points
    and must return N Boolean values. Trusted invalid points are excluded.
    Trust is evidence only and NEVER masks or changes image values.

    Both callbacks must be pure per-point maps, independent of batch size.
    Fixed midpoint quadrature averages ENCODED colour, not linear light.
    Float32 coordinates and OpenCV's interpolation table are intentional for
    numerical compatibility with the M3 candidate. RGB outside the original
    source reconstruction uses explicit fill. RGBA always uses transparent
    fill and premultiplied integration, returning straight uint8 RGBA.

    Coordinate working storage is O(max_tile_pixels), independent of n*n or
    output area. A float32 source copy is retained for the iterator lifetime.
    Consumers must release tiles if bounded TOTAL memory is required. A map
    callback may have its own unbounded cache, outside this interface's bound.
    """
    if (isinstance(subgrid, bool) or not isinstance(subgrid, (int, np.integer))
            or subgrid not in (2, 4, 8)):
        raise ValueError("subgrid must be 2, 4 or 8")
    if reconstruction not in ("bilinear", "boxhold"):
        raise ValueError("reconstruction must be bilinear or boxhold")
    if (isinstance(max_tile_pixels, bool) or not isinstance(max_tile_pixels, (int, np.integer))
            or max_tile_pixels < 1):
        raise ValueError("max_tile_pixels must be a positive integer")
    if not isinstance(grid, OutputGrid):
        raise TypeError("grid must be OutputGrid")
    source, fill = _source(source_u8, fill_rgb)
    stats = stats if stats is not None else SamplingStats()
    stats.source_float_bytes = source.nbytes
    src_h, src_w, channels = source.shape
    interpolation = cv2.INTER_LINEAR if reconstruction == "bilinear" else cv2.INTER_NEAREST
    offsets = (np.arange(subgrid, dtype=np.float64)+.5)/subgrid-.5
    origin = np.asarray(grid.origin_mm, dtype=np.float64)
    spacing = np.asarray(grid.spacing_mm, dtype=np.float64)
    denominator = subgrid*subgrid
    for rows, cols in _tiles(grid.shape_hw, int(max_tile_pixels)):
        height, width = rows.stop-rows.start, cols.stop-cols.start
        yy, xx = np.mgrid[rows, cols]
        # Do not increment coordinates from the previous tile: centres use
        # absolute integer indices, giving tile-size-independent arithmetic.
        centres = np.stack((xx, yy), -1).astype(np.float64)*spacing+origin
        accum = np.zeros((height, width, channels), dtype=np.float32)
        counts = np.zeros((4 if trust is not None else 3, height, width), dtype=np.uint16)
        for oy in offsets:
            for ox in offsets:
                physical = (centres + np.array([ox, oy])*spacing).reshape(-1, 2)
                mapped = np.asarray(inverse(physical), dtype=np.float64)
                if mapped.shape != physical.shape:
                    raise ValueError("inverse must return an Nx2 array")
                finite = np.isfinite(mapped).all(1)
                # Do not cast unbounded values to float32 or hand them to
                # OpenCV. A Boolean mask forces fill, never a valid sentinel.
                can_contribute = (finite & (mapped[:, 0] >= -1) & (mapped[:, 0] <= src_w)
                                  & (mapped[:, 1] >= -1) & (mapped[:, 1] <= src_h))
                coordinates = np.zeros_like(mapped, dtype=np.float32)
                coordinates[can_contribute] = mapped[can_contribute]
                inside_cell = (finite & (mapped[:, 0] >= -.5) & (mapped[:, 0] < src_w-.5)
                               & (mapped[:, 1] >= -.5) & (mapped[:, 1] < src_h-.5))
                if reconstruction == "bilinear":
                    # Support of nonzero interpolation weights, using the
                    # float32 coordinates actually sent to the sampler.
                    support = (can_contribute & (coordinates[:, 0] >= 0)
                               & (coordinates[:, 0] <= src_w-1)
                               & (coordinates[:, 1] >= 0) & (coordinates[:, 1] <= src_h-1))
                else:
                    indices = np.rint(coordinates)
                    support = (can_contribute & (indices[:, 0] >= 0)
                               & (indices[:, 0] < src_w) & (indices[:, 1] >= 0)
                               & (indices[:, 1] < src_h))
                coords = coordinates.reshape(height, width, 2)
                sample = cv2.remap(source, coords[..., 0], coords[..., 1], interpolation,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
                sample.reshape(-1, channels)[~can_contribute] = fill
                accum += sample
                counts[0] += finite.reshape(height, width)
                counts[1] += inside_cell.reshape(height, width)
                counts[2] += support.reshape(height, width)
                if trust is not None:
                    trusted = np.asarray(trust(physical))
                    if trusted.shape != (len(physical),) or trusted.dtype != np.bool_:
                        raise ValueError("trust must return N Boolean values")
                    counts[3] += (trusted & finite).reshape(height, width)
                stats.inverse_calls += 1
                stats.inverse_points += len(physical)
                stats.max_query_points = max(stats.max_query_points, len(physical))
                coordinate_bytes = centres.nbytes+physical.nbytes+mapped.nbytes+coordinates.nbytes
                stats.max_coordinate_bytes = max(stats.max_coordinate_bytes, coordinate_bytes)
                accounted = (coordinate_bytes+accum.nbytes+counts.nbytes+sample.nbytes+
                             finite.nbytes+can_contribute.nbytes+inside_cell.nbytes+support.nbytes+
                             xx.nbytes+yy.nbytes)
                if reconstruction == "boxhold":
                    accounted += indices.nbytes
                stats.max_accounted_array_bytes = max(stats.max_accounted_array_bytes, accounted)
        accum /= np.float32(denominator)
        if channels == 4:
            alpha = accum[..., 3:4]
            colour = np.zeros_like(accum[..., :3])
            np.divide(accum[..., :3]*np.float32(255), alpha, out=colour, where=alpha > 0)
            accum[..., :3] = colour
        image = np.clip(np.rint(accum), 0, 255).astype(np.uint8)
        if channels == 4:
            image[..., :3][image[..., 3] == 0] = 0
        fractions = counts.astype(np.float32)/np.float32(denominator)
        stats.tiles += 1
        yield SampleTile(rows, cols, image, fractions[0], fractions[1], fractions[2],
                         fractions[3] if trust is not None else None)


def sample_image(source_u8: np.ndarray, inverse: Inverse, grid: OutputGrid, **kwargs) -> SampledImage:
    """Collect the iterator, adding O(output area) image and evidence storage."""
    _validate_source(source_u8)
    if not isinstance(grid, OutputGrid):
        raise TypeError("grid must be OutputGrid")
    stats = kwargs.pop("stats", None)
    stats = stats if stats is not None else SamplingStats()
    image = np.empty((*grid.shape_hw, source_u8.shape[-1]), dtype=np.uint8)
    finite = np.empty(grid.shape_hw, dtype=np.float32)
    cells = np.empty_like(finite)
    support = np.empty_like(finite)
    trusted = np.empty_like(finite) if kwargs.get("trust") is not None else None
    for tile in iter_area_tiles(source_u8, inverse, grid, stats=stats, **kwargs):
        index = (tile.rows, tile.cols)
        image[index] = tile.image
        finite[index] = tile.finite_geometry_fraction
        cells[index] = tile.source_cell_fraction
        support[index] = tile.full_reconstruction_support_fraction
        if trusted is not None:
            trusted[index] = tile.trusted_geometry_fraction
    return SampledImage(image, finite, cells, support, trusted, stats)
