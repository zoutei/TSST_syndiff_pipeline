"""Canonical SCC mapping grid: science rectangle + bottom convolution pad rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAPGRID_VERSION = 2
DEFAULT_CONV_PAD_NATIVE = 8
DEFAULT_X_LEFT_DEAD = 44
DEFAULT_X_RIGHT_DEAD = 44
DEFAULT_Y_EDGE_STRIP = 30


class MappingGridError(ValueError):
    """Raised when MappingGrid invariants or artifact metadata are invalid."""


@dataclass(frozen=True)
class MappingGrid:
    """Science-aligned grid with optional bottom pad rows below ``ffi_y=0``."""

    ffi_xmin: int
    ffi_ymin: int
    ffi_xmax: int
    ffi_ymax: int
    oversampling: int = 1
    conv_pad_native: int = DEFAULT_CONV_PAD_NATIVE

    def __post_init__(self) -> None:
        if self.ffi_xmax <= self.ffi_xmin:
            raise MappingGridError(
                f"invalid x bounds: xmin={self.ffi_xmin}, xmax={self.ffi_xmax}"
            )
        if self.ffi_ymax <= self.ffi_ymin:
            raise MappingGridError(
                f"invalid y bounds: ymin={self.ffi_ymin}, ymax={self.ffi_ymax}"
            )
        if self.oversampling < 1:
            raise MappingGridError(f"oversampling must be >= 1, got {self.oversampling}")
        if self.conv_pad_native < 0:
            raise MappingGridError(
                f"conv_pad_native must be >= 0, got {self.conv_pad_native}"
            )
        if self.ffi_ymin > 0:
            raise MappingGridError(
                f"ffi_ymin must be <= 0 (bottom pad rows), got {self.ffi_ymin}"
            )

    @property
    def width_native(self) -> int:
        return int(self.ffi_xmax - self.ffi_xmin)

    @property
    def height_native(self) -> int:
        return int(self.ffi_ymax - self.ffi_ymin)

    @property
    def width_os(self) -> int:
        return int(self.width_native * self.oversampling)

    @property
    def height_os(self) -> int:
        return int(self.height_native * self.oversampling)

    def array_shape_native(self) -> tuple[int, int]:
        return (self.height_native, self.width_native)

    def array_shape_os(self) -> tuple[int, int]:
        return (self.height_os, self.width_os)

    @classmethod
    def from_ffi_shape(
        cls,
        nx: int,
        ny: int,
        *,
        x_left_dead: int = DEFAULT_X_LEFT_DEAD,
        x_right_dead: int = DEFAULT_X_RIGHT_DEAD,
        y_edge_strip: int = DEFAULT_Y_EDGE_STRIP,
        conv_pad_native: int = DEFAULT_CONV_PAD_NATIVE,
        oversampling: int = 1,
    ) -> MappingGrid:
        """Build grid from full FFI dimensions and dead-strip defaults."""
        if nx <= 0 or ny <= 0:
            raise MappingGridError(f"FFI shape must be positive, got ({nx}, {ny})")
        ffi_xmin = int(x_left_dead)
        ffi_xmax = int(nx - x_right_dead)
        ffi_ymax = int(ny - y_edge_strip)
        ffi_ymin = -int(conv_pad_native)
        return cls(
            ffi_xmin=ffi_xmin,
            ffi_ymin=ffi_ymin,
            ffi_xmax=ffi_xmax,
            ffi_ymax=ffi_ymax,
            oversampling=int(oversampling),
            conv_pad_native=int(conv_pad_native),
        )

    @classmethod
    def from_mapping_dict(cls, doc: dict[str, Any]) -> MappingGrid:
        """Parse ``mapping_grid`` block from sidecar or diff job JSON."""
        if "mapping_grid" in doc:
            block = doc["mapping_grid"]
        else:
            block = doc
        return cls(
            ffi_xmin=int(block["ffi_xmin"]),
            ffi_ymin=int(block["ffi_ymin"]),
            ffi_xmax=int(block["ffi_xmax"]),
            ffi_ymax=int(block["ffi_ymax"]),
            oversampling=int(block.get("oversampling_factor", block.get("oversampling", 1))),
            conv_pad_native=int(block.get("conv_pad_native", DEFAULT_CONV_PAD_NATIVE)),
        )

    @classmethod
    def from_sidecar(cls, doc: dict[str, Any]) -> MappingGrid:
        """Load from ``field_mode_assembly.json`` schema v3."""
        version = int(doc.get("schema_version", 0))
        if version < 3:
            raise MappingGridError(
                f"field_mode_assembly schema_version must be >= 3, got {version}"
            )
        if "mapping_grid" not in doc:
            raise MappingGridError("field_mode_assembly v3 requires mapping_grid block")
        return cls.from_mapping_dict(doc)

    @classmethod
    def from_fits_header(cls, hdr: Any, *, require_mapgrid: bool = True) -> MappingGrid:
        """Load from master mapping FITS HDU header (MAPGRID=2)."""
        if require_mapgrid:
            mapgrid = hdr.get("MAPGRID")
            if mapgrid is None:
                raise MappingGridError("FITS header missing MAPGRID keyword (v1 artifact)")
            if int(mapgrid) < MAPGRID_VERSION:
                raise MappingGridError(
                    f"FITS MAPGRID must be >= {MAPGRID_VERSION}, got {mapgrid}"
                )
        for key in ("XMIN", "YMIN", "XMAX", "YMAX"):
            if key not in hdr:
                raise MappingGridError(f"FITS header missing {key}")
        return cls(
            ffi_xmin=int(hdr["XMIN"]),
            ffi_ymin=int(hdr["YMIN"]),
            ffi_xmax=int(hdr["XMAX"]),
            ffi_ymax=int(hdr["YMAX"]),
            oversampling=int(hdr.get("OVERSAMP", hdr.get("OVERSAMPING", 1))),
            conv_pad_native=int(hdr.get("CONVPAD", DEFAULT_CONV_PAD_NATIVE)),
        )

    def to_mapping_dict(self) -> dict[str, Any]:
        return {
            "ffi_xmin": self.ffi_xmin,
            "ffi_ymin": self.ffi_ymin,
            "ffi_xmax": self.ffi_xmax,
            "ffi_ymax": self.ffi_ymax,
            "oversampling_factor": self.oversampling,
            "conv_pad_native": self.conv_pad_native,
        }

    def to_fits_header_updates(self) -> dict[str, int]:
        return {
            "XMIN": self.ffi_xmin,
            "YMIN": self.ffi_ymin,
            "XMAX": self.ffi_xmax,
            "YMAX": self.ffi_ymax,
            "MAPGRID": MAPGRID_VERSION,
            "CONVPAD": self.conv_pad_native,
            "OVERSAMP": self.oversampling,
        }

    def _grid_width(self, *, oversampled: bool | None = None) -> int:
        use_os = self.oversampling > 1 if oversampled is None else oversampled
        return self.width_os if use_os else self.width_native

    def ffi_to_local(self, ffi_x: int | float, ffi_y: int | float) -> tuple[int, int]:
        lx = int(round(ffi_x)) - self.ffi_xmin
        ly = int(round(ffi_y)) - self.ffi_ymin
        if not self.contains_local(lx, ly):
            raise MappingGridError(
                f"FFI ({ffi_x}, {ffi_y}) -> local ({lx}, {ly}) out of grid bounds"
            )
        return lx, ly

    def local_to_ffi(self, lx: int, ly: int) -> tuple[int, int]:
        if not self.contains_local(lx, ly):
            raise MappingGridError(f"local ({lx}, {ly}) out of grid bounds")
        return self.ffi_xmin + int(lx), self.ffi_ymin + int(ly)

    def local_to_flat(self, lx: int, ly: int, *, oversampled: bool = False) -> int:
        if not self.contains_local(lx, ly, oversampled=oversampled):
            raise MappingGridError(f"local ({lx}, {ly}) out of grid bounds")
        width = self._grid_width(oversampled=oversampled)
        return int(ly) * width + int(lx)

    def flat_to_local(self, flat_id: int, *, oversampled: bool = False) -> tuple[int, int]:
        flat = int(flat_id)
        if flat < 0:
            raise MappingGridError(f"flat_id must be >= 0, got {flat}")
        width = self._grid_width(oversampled=oversampled)
        height = self.height_os if oversampled else self.height_native
        ly, lx = divmod(flat, width)
        if ly >= height:
            raise MappingGridError(
                f"flat_id {flat_id} out of grid (height={height}, width={width})"
            )
        return lx, ly

    def flat_to_ffi(self, flat_id: int, *, oversampled: bool = False) -> tuple[int, int]:
        lx, ly = self.flat_to_local(flat_id, oversampled=oversampled)
        return self.local_to_ffi(lx, ly)

    def contains_local(
        self, lx: int, ly: int, *, oversampled: bool = False
    ) -> bool:
        width = self._grid_width(oversampled=oversampled)
        height = self.height_os if oversampled else self.height_native
        return 0 <= lx < width and 0 <= ly < height

    def contains_flat(self, flat_id: int, *, oversampled: bool = False) -> bool:
        try:
            self.flat_to_local(flat_id, oversampled=oversampled)
        except MappingGridError:
            return False
        return True

    def contains_ffi(self, ffi_x: int | float, ffi_y: int | float) -> bool:
        x = int(round(ffi_x))
        y = int(round(ffi_y))
        return (
            self.ffi_xmin <= x < self.ffi_xmax
            and self.ffi_ymin <= y < self.ffi_ymax
        )

    def contains_science_ffi(self, ffi_x: int | float, ffi_y: int | float) -> bool:
        x = int(round(ffi_x))
        y = int(round(ffi_y))
        return (
            self.ffi_xmin <= x < self.ffi_xmax
            and 0 <= y < self.ffi_ymax
        )

    def science_ffi_bounds(self) -> dict[str, Any]:
        """Diff science arrays: no bottom pad rows."""
        return {
            "x_min": self.ffi_xmin,
            "x_max": self.ffi_xmax,
            "y_min": 0,
            "y_max": self.ffi_ymax,
            "shape": (self.ffi_ymax - 0, self.width_native),
        }

    def science_bounds_1based(self) -> dict[str, int]:
        """1-based inclusive FFI row/col limits for masking catalogs."""
        return {
            "col_lo": int(self.ffi_xmin) + 1,
            "col_hi": int(self.ffi_xmax),
            "row_lo": 1,
            "row_hi": int(self.ffi_ymax),
        }

    def template_ffi_bounds(self) -> dict[str, Any]:
        """Full template grid including bottom pad rows."""
        return {
            "x_min": self.ffi_xmin,
            "x_max": self.ffi_xmax,
            "y_min": self.ffi_ymin,
            "y_max": self.ffi_ymax,
            "shape": self.array_shape_native(),
        }

    def ffi_coords_for_wcs(
        self,
        lx: int | float,
        ly: int | float,
    ) -> tuple[float, float]:
        """Convert local grid indices to original TESS FFI pixels for WCS calls."""
        ffi_x, ffi_y = self.local_to_ffi(int(round(lx)), int(round(ly)))
        return float(ffi_x), float(ffi_y)


def load_mapping_grid_from_master(path: str | Path) -> MappingGrid:
    """Load MappingGrid from master_pixels2skycells FITS (MAPGRID>=2 + shape check)."""
    from astropy.io import fits

    with fits.open(str(Path(path))) as hdul:
        hdu_idx = 1 if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None else 0
        hdu = hdul[hdu_idx]
        grid = MappingGrid.from_fits_header(hdu.header)
        data = getattr(hdu, "data", None)
        if data is not None:
            expected = grid.array_shape_os()
            if tuple(data.shape) != tuple(expected):
                raise MappingGridError(
                    f"master FITS shape {tuple(data.shape)} != MappingGrid {expected} "
                    f"(rebuild mapping with MAPGRID>={MAPGRID_VERSION})"
                )
        return grid


def compute_rkernel(scale_px: float) -> int:
    """Hotpants kernel half-width in pixels."""
    return int(2.5 * float(scale_px))


def compute_conv_pad_native(
    rkernel: int,
    *,
    template_conv_pad_spare_px: int = 4,
) -> int:
    """Bottom pad row count stored in CONVPAD."""
    return int(rkernel) + int(template_conv_pad_spare_px)


def create_coords_for_grid(
    grid: MappingGrid,
    oversampling_factor: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build TESS FFI pixel coordinates and local flat IDs for mapping.

    Returns
    -------
    tpix_coord_input
        ``(N, 2)`` array with columns ``[ty, tx]`` in **original FFI** pixels.
    flat_ids
        Local 0-based flat indices into the grid (OS-aware when ``F > 1``).
    """
    f = int(oversampling_factor if oversampling_factor is not None else grid.oversampling)
    if f < 1:
        raise MappingGridError(f"oversampling_factor must be >= 1, got {f}")
    width = grid.width_native * f
    height = grid.height_native * f
    gy, gx = np.mgrid[:height, :width]
    ty = grid.ffi_ymin + (gy.astype(np.float64) + 0.5) / f - 0.5
    tx = grid.ffi_xmin + (gx.astype(np.float64) + 0.5) / f - 0.5
    tpix = np.column_stack([ty.ravel(), tx.ravel()])
    flat_ids = np.arange(height * width, dtype=np.int64)
    return tpix, flat_ids
