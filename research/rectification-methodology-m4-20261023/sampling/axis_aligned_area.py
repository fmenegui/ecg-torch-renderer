"""Exact rectangle integration of the SOURCE-CELL constant reconstruction.

This is a separate, failure-directed positive control. It is not a general
rectifier, not the M3 bilinear model and not a certificate for unknown ink.
Only diagonal affine inverse maps are accepted, never approximated from shear.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from streaming_sampler import OutputGrid, _tiles, _validate_source


@dataclass
class AreaTile:
    rows: slice
    cols: slice
    encoded_mean: np.ndarray
    image: np.ndarray
    source_cell_area_fraction: np.ndarray


class AxisAlignedAreaSampler:
    """Continuous summed-area table for uint8 RGB or premultiplied RGBA.

    Build storage is O(source area), float64 channels. Query coordinate
    storage is O(tile pixels) independent of source footprint size. There is
    no subgrid and no repeated resampling. Encoded colour arithmetic only.
    """

    def __init__(self, source_u8, inverse_matrix, inverse_offset, *, fill_rgb=(255., 255., 255.)):
        _validate_source(source_u8)
        matrix = np.asarray(inverse_matrix, dtype=np.float64)
        offset = np.asarray(inverse_offset, dtype=np.float64)
        if (matrix.shape != (2, 2) or offset.shape != (2,) or not np.isfinite(matrix).all()
                or not np.isfinite(offset).all()):
            raise ValueError("finite affine 2x2 matrix and 2-vector required")
        if matrix[0, 1] != 0 or matrix[1, 0] != 0:
            raise ValueError("exact axis-area route rejects every shear or rotation")
        if matrix[0, 0] == 0 or matrix[1, 1] == 0:
            raise ValueError("singular affine inverse")
        if (len(fill_rgb) != 3 or not np.isfinite(fill_rgb).all()
                or min(fill_rgb) < 0 or max(fill_rgb) > 255):
            raise ValueError("fill_rgb must contain finite encoded values in [0,255]")
        self.scale = np.diag(matrix).copy()
        self.offset = offset.copy()
        self.height, self.width, self.channels = source_u8.shape
        source = source_u8.astype(np.float64)
        if self.channels == 4:
            # Keep RGB*alpha and alpha*255 as integer-valued float64 until
            # after integration. Their full sums remain below 2**53 for the
            # accepted source dimensions, avoiding fractional cumsum drift.
            source[..., :3] *= source[..., 3:4]
            source[..., 3] *= 255.
            self.fill = np.zeros(4, dtype=np.float64)
        else:
            self.fill = np.asarray(fill_rgb, dtype=np.float64)
        # Integrate residual to the fill, so outside source automatically
        # contributes its prescribed colour without renormalising a crop.
        source -= self.fill
        self.integral = np.zeros((self.height+1, self.width+1, self.channels), dtype=np.float64)
        np.cumsum(source, axis=0, out=self.integral[1:, 1:])
        np.cumsum(self.integral[1:, 1:], axis=1, out=self.integral[1:, 1:])
        # Conservative arithmetic guard for prefix interpolation/subtraction.
        # It is not an optical error estimate. Large tables and tiny boxes
        # need a different accumulator or direct local integration instead.
        magnitudes = np.array([np.max(np.abs(self.integral[..., channel]))
                               for channel in range(self.channels)])
        self._roundoff_mass = 64*np.finfo(np.float64).eps*magnitudes
        if self.channels == 4:
            self._roundoff_mass /= 255.

    def _prefix(self, x, y):
        # Source centres x=j correspond to cell edges j+/-0.5.
        x = np.clip(x+.5, 0., self.width)
        y = np.clip(y+.5, 0., self.height)
        ix = np.minimum(np.floor(x).astype(np.intp), self.width-1)
        iy = np.minimum(np.floor(y).astype(np.intp), self.height-1)
        fx, fy = (x-ix)[..., None], (y-iy)[..., None]
        top_left = self.integral[iy, ix]
        top_right = self.integral[iy, ix+1]
        bottom_left = self.integral[iy+1, ix]
        bottom_right = self.integral[iy+1, ix+1]
        return (top_left*(1-fx)*(1-fy)+top_right*fx*(1-fy)
                +bottom_left*(1-fx)*fy+bottom_right*fx*fy)

    def iter_tiles(self, grid: OutputGrid, *, max_tile_pixels=16384):
        if not isinstance(grid, OutputGrid):
            raise TypeError("grid must be OutputGrid")
        if (isinstance(max_tile_pixels, bool) or not isinstance(max_tile_pixels, (int, np.integer))
                or max_tile_pixels < 1):
            raise ValueError("max_tile_pixels must be a positive integer")
        spacing = np.asarray(grid.spacing_mm, dtype=np.float64)
        halfwidth = np.abs(self.scale)*spacing/2
        if not np.isfinite(halfwidth).all() or np.any(halfwidth <= 0):
            raise ValueError("non-finite or unresolved footprint")
        for rows, cols in _tiles(grid.shape_hw, int(max_tile_pixels)):
            yy, xx = np.mgrid[rows, cols]
            centres = np.stack((xx, yy), -1)*spacing+grid.origin_mm
            mapped = centres*self.scale+self.offset
            lower, upper = mapped-halfwidth, mapped+halfwidth
            area = (upper[..., 0]-lower[..., 0])*(upper[..., 1]-lower[..., 1])
            if (not np.isfinite(lower).all() or not np.isfinite(upper).all() or
                    not np.isfinite(area).all() or np.any(area <= 0)):
                raise ValueError("non-finite or numerically unresolved footprint area")
            roundoff = self._roundoff_mass/area[..., None]
            if np.any(roundoff >= .25):
                raise ValueError("footprint too small for prefix-sum numerical precision")
            total = (self._prefix(upper[..., 0], upper[..., 1])
                     -self._prefix(lower[..., 0], upper[..., 1])
                     -self._prefix(upper[..., 0], lower[..., 1])
                     +self._prefix(lower[..., 0], lower[..., 1]))
            mean = self.fill+total/area[..., None]
            if self.channels == 4:
                mean /= 255.
            overlap = np.maximum(0., np.minimum(upper, [self.width-.5, self.height-.5])
                                 -np.maximum(lower, [-.5, -.5]))
            fraction = np.prod(overlap, axis=-1)/area
            if self.channels == 4:
                # A zero-alpha cell can acquire tiny positive roundoff from
                # prefix cancellation. Never amplify that into hidden RGB.
                # Only values within the arithmetic uncertainty are zeroed.
                numerical_zero = np.abs(mean[..., 3]) <= roundoff[..., 3]
                mean[numerical_zero] = 0
                colour = np.zeros_like(mean[..., :3])
                np.divide(mean[..., :3]*255., mean[..., 3:4], out=colour,
                          where=mean[..., 3:4] > 0)
                mean[..., :3] = colour
            image = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
            if self.channels == 4:
                image[..., :3][image[..., 3] == 0] = 0
            yield AreaTile(rows, cols, mean, image, fraction)

    def sample(self, grid: OutputGrid, *, max_tile_pixels=16384):
        image = np.empty((*grid.shape_hw, self.channels), dtype=np.uint8)
        mean = np.empty((*grid.shape_hw, self.channels), dtype=np.float64)
        fraction = np.empty(grid.shape_hw, dtype=np.float64)
        for tile in self.iter_tiles(grid, max_tile_pixels=max_tile_pixels):
            image[tile.rows, tile.cols] = tile.image
            mean[tile.rows, tile.cols] = tile.encoded_mean
            fraction[tile.rows, tile.cols] = tile.source_cell_area_fraction
        return image, mean, fraction
