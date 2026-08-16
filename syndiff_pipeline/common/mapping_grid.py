"""Canonical SCC mapping grid for the paired-padding geometry contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MAPGRID_VERSION = 3
MAPGRID_V3_VERSION = MAPGRID_VERSION
DEFAULT_CONV_PAD_NATIVE = 8
DEFAULT_X_LEFT_DEAD = 44
DEFAULT_X_RIGHT_DEAD = 44
DEFAULT_Y_EDGE_STRIP = 30


class MappingGridError(ValueError):
    """Raised when MappingGrid invariants or artifact metadata are invalid."""


@dataclass(frozen=True)
class MappingGrid:
    """A template-support grid and its science rectangle.

    ``ffi_*`` is the full template-support rectangle T. The explicit
    ``science_*_ffi`` fields identify the observational rectangle S.
    The template rectangle is the only rectangle represented by mapping/WCS;
    fabricated science padding is an array/mask concern owned by later stages.
    """

    ffi_xmin: int
    ffi_ymin: int
    ffi_xmax: int
    ffi_ymax: int
    oversampling: int = 1
    conv_pad_native: int = DEFAULT_CONV_PAD_NATIVE
    mapgrid_version: int = MAPGRID_VERSION
    science_xmin_ffi: int | None = None
    science_ymin_ffi: int | None = None
    science_xmax_ffi: int | None = None
    science_ymax_ffi: int | None = None

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
        if self.mapgrid_version != MAPGRID_VERSION:
            raise MappingGridError(
                f"unsupported MAPGRID version {self.mapgrid_version}; "
                f"this pipeline requires MAPGRID={MAPGRID_VERSION}"
            )
        sx0 = self.ffi_xmin if self.science_xmin_ffi is None else int(self.science_xmin_ffi)
        sy0 = 0 if self.science_ymin_ffi is None else int(self.science_ymin_ffi)
        sx1 = self.ffi_xmax if self.science_xmax_ffi is None else int(self.science_xmax_ffi)
        sy1 = self.ffi_ymax if self.science_ymax_ffi is None else int(self.science_ymax_ffi)
        if not (self.ffi_xmin <= sx0 < sx1 <= self.ffi_xmax and self.ffi_ymin <= sy0 < sy1 <= self.ffi_ymax):
            raise MappingGridError("science bounds must be contained by template bounds")
        object.__setattr__(self, "science_xmin_ffi", sx0)
        object.__setattr__(self, "science_ymin_ffi", sy0)
        object.__setattr__(self, "science_xmax_ffi", sx1)
        object.__setattr__(self, "science_ymax_ffi", sy1)

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

    # Explicit names keep the physical geometry contract readable at call
    # sites. The ``ffi_*`` fields are the serialized template-support bounds.
    @property
    def science_xmin(self) -> int:
        return int(self.science_xmin_ffi)

    @property
    def science_xmax(self) -> int:
        return int(self.science_xmax_ffi)

    @property
    def science_ymin(self) -> int:
        return int(self.science_ymin_ffi)

    @property
    def science_ymax(self) -> int:
        return int(self.science_ymax_ffi)

    @property
    def template_xmin(self) -> int:
        return self.ffi_xmin

    @property
    def template_xmax(self) -> int:
        return self.ffi_xmax

    @property
    def template_ymin(self) -> int:
        return self.ffi_ymin

    @property
    def template_ymax(self) -> int:
        return self.ffi_ymax

    @property
    def pad_left(self) -> int:
        return self.science_xmin - self.template_xmin

    @property
    def pad_right(self) -> int:
        return self.template_xmax - self.science_xmax

    @property
    def pad_bottom(self) -> int:
        return self.science_ymin - self.template_ymin

    @property
    def pad_top(self) -> int:
        return self.template_ymax - self.science_ymax

    @property
    def pad_left_native(self) -> int:
        return self.pad_left

    @property
    def pad_right_native(self) -> int:
        return self.pad_right

    @property
    def pad_bottom_native(self) -> int:
        return self.pad_bottom

    @property
    def pad_top_native(self) -> int:
        return self.pad_top

    def geometry_recipe(self) -> dict[str, Any]:
        """Return the complete, unambiguous geometry declaration.

        MAPGRID=3 records every datum needed to reproduce the paired
        template-support/science-padding contract; these fields are part of
        the geometry fingerprint and therefore cannot be omitted by a
        producer without changing the artifact identity.
        """
        recipe: dict[str, Any] = {
            "mapgrid_version": self.mapgrid_version,
            "coordinate_frame": "full_ffi",
            "template_bounds_ffi": {
                "x_min": self.template_xmin,
                "x_max": self.template_xmax,
                "y_min": self.template_ymin,
                "y_max": self.template_ymax,
            },
            "science_bounds_ffi": {
                "x_min": self.science_xmin,
                "x_max": self.science_xmax,
                "y_min": self.science_ymin,
                "y_max": self.science_ymax,
            },
            "conv_pad_native": self.conv_pad_native,
            "oversampling_factor": self.oversampling,
            "pad_native": {
                "left": self.pad_left,
                "right": self.pad_right,
                "bottom": self.pad_bottom,
                "top": self.pad_top,
            },
        }
        if self.mapgrid_version == MAPGRID_VERSION:
            native_slice = self.science_slice_native()
            os_slice = self.science_slice_os()
            recipe.update(
                {
                    # T is the mapped/template-support rectangle; S is the
                    # observational science rectangle selected from T.
                    "physical_template_bounds_ffi": dict(recipe["template_bounds_ffi"]),
                    "template_support_bounds_ffi": dict(recipe["template_bounds_ffi"]),
                    "support_policy": "bounded_support_pad",
                    "science_pad_policy": "neutral_invalid",
                    "science_slice_native": [
                        [native_slice[0].start, native_slice[0].stop],
                        [native_slice[1].start, native_slice[1].stop],
                    ],
                    "science_slice_os": [
                        [os_slice[0].start, os_slice[0].stop],
                        [os_slice[1].start, os_slice[1].stop],
                    ],
                    "effective_support_pad_native": self.conv_pad_native,
                }
            )
            recipe["pad_kind"] = {k: "physical" for k in ("left", "right", "bottom", "top")}
            recipe["pixel_convention"] = "half_open_ffi_integer_pixels"
        return recipe

    @property
    def geometry_fingerprint(self) -> str:
        """Stable content fingerprint for propagation through artifacts."""
        payload = json.dumps(
            self.geometry_recipe(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

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
        mapgrid_version: int = MAPGRID_VERSION,
    ) -> MappingGrid:
        """Build grid from full FFI dimensions and dead-strip defaults."""
        if nx <= 0 or ny <= 0:
            raise MappingGridError(f"FFI shape must be positive, got ({nx}, {ny})")
        sxmin = int(x_left_dead)
        sxmax = int(nx - x_right_dead)
        symin = 0
        symax = int(ny - y_edge_strip)
        p = int(conv_pad_native)
        if int(mapgrid_version) != MAPGRID_VERSION:
            raise MappingGridError(
                f"from_ffi_shape requires MAPGRID={MAPGRID_VERSION}, got {mapgrid_version}"
            )
        ffi_xmin, ffi_xmax = sxmin - p, sxmax + p
        ffi_ymin, ffi_ymax = symin - p, symax + p
        return cls(
            ffi_xmin=ffi_xmin,
            ffi_ymin=ffi_ymin,
            ffi_xmax=ffi_xmax,
            ffi_ymax=ffi_ymax,
            oversampling=int(oversampling),
            conv_pad_native=int(conv_pad_native),
            mapgrid_version=int(mapgrid_version),
            science_xmin_ffi=sxmin,
            science_ymin_ffi=symin,
            science_xmax_ffi=sxmax,
            science_ymax_ffi=symax,
        )

    @classmethod
    def from_science_bounds(
        cls,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        *,
        pad: int = DEFAULT_CONV_PAD_NATIVE,
        oversampling: int = 1,
        mapgrid_version: int = MAPGRID_V3_VERSION,
    ) -> "MappingGrid":
        """Construct paired-padding geometry from half-open science bounds."""
        if mapgrid_version != MAPGRID_V3_VERSION:
            raise MappingGridError("from_science_bounds is only valid for MAPGRID=3")
        p = int(pad)
        if p < 0:
            raise MappingGridError("pad must be >= 0")
        return cls(
            ffi_xmin=int(x_min) - p, ffi_ymin=int(y_min) - p,
            ffi_xmax=int(x_max) + p, ffi_ymax=int(y_max) + p,
            oversampling=int(oversampling), conv_pad_native=p,
            mapgrid_version=MAPGRID_V3_VERSION,
            science_xmin_ffi=int(x_min), science_ymin_ffi=int(y_min),
            science_xmax_ffi=int(x_max), science_ymax_ffi=int(y_max),
        )

    @classmethod
    def from_mapping_dict(cls, doc: dict[str, Any]) -> MappingGrid:
        """Parse ``mapping_grid`` block from sidecar or diff job JSON."""
        if "mapping_grid" in doc:
            block = doc["mapping_grid"]
        else:
            block = doc
        if "mapgrid_version" not in block:
            raise MappingGridError("mapping_grid requires explicit mapgrid_version=3")
        version = int(block["mapgrid_version"])
        if version != MAPGRID_VERSION:
            raise MappingGridError(
                f"mapping_grid requires MAPGRID={MAPGRID_VERSION}, got {version}"
            )
        sb = block.get("science_bounds_ffi", {})
        required = ("x_min", "y_min", "x_max", "y_max")
        if not all(k in sb for k in required) and not all(k in block for k in ("science_xmin", "science_ymin", "science_xmax", "science_ymax")):
            raise MappingGridError("MAPGRID=3 requires explicit science_bounds_ffi")
        grid = cls(
            ffi_xmin=int(block["ffi_xmin"]), ffi_ymin=int(block["ffi_ymin"]),
            ffi_xmax=int(block["ffi_xmax"]), ffi_ymax=int(block["ffi_ymax"]),
            oversampling=int(block.get("oversampling_factor", block.get("oversampling", 1))),
            conv_pad_native=int(block.get("conv_pad_native", DEFAULT_CONV_PAD_NATIVE)),
            mapgrid_version=version,
            science_xmin_ffi=int(block.get("science_xmin", sb.get("x_min"))),
            science_ymin_ffi=int(block.get("science_ymin", sb.get("y_min"))),
            science_xmax_ffi=int(block.get("science_xmax", sb.get("x_max"))),
            science_ymax_ffi=int(block.get("science_ymax", sb.get("y_max"))),
        )
        expected = grid.to_mapping_dict()
        for key in (
            "science_xmin", "science_xmax", "science_ymin", "science_ymax",
            "template_xmin", "template_xmax", "template_ymin", "template_ymax",
        ):
            if key in block and int(block[key]) != int(expected[key]):
                raise MappingGridError(
                    f"mapping_grid {key}={block[key]!r} disagrees with serialized geometry "
                    f"({expected[key]!r})"
                )
        if "coordinate_frame" in block and str(block["coordinate_frame"]) != "full_ffi":
            raise MappingGridError(
                f"mapping_grid coordinate_frame must be full_ffi, got {block['coordinate_frame']!r}"
            )
        if "geometry_fingerprint" in block and str(block["geometry_fingerprint"]) != grid.geometry_fingerprint:
            raise MappingGridError("mapping_grid geometry_fingerprint does not match serialized bounds")
        return grid

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
        grid = cls.from_mapping_dict(doc)
        return grid

    @classmethod
    def from_fits_header(cls, hdr: Any, *, require_mapgrid: bool = True) -> MappingGrid:
        """Load from a MAPGRID=3 master mapping FITS HDU header."""
        if require_mapgrid:
            mapgrid = hdr.get("MAPGRID")
            if mapgrid is None:
                raise MappingGridError("FITS header missing MAPGRID keyword (v1 artifact)")
            if int(mapgrid) != MAPGRID_VERSION:
                raise MappingGridError(
                    f"FITS MAPGRID must equal {MAPGRID_VERSION}, got {mapgrid}"
                )
        for key in ("XMIN", "YMIN", "XMAX", "YMAX"):
            if key not in hdr:
                raise MappingGridError(f"FITS header missing {key}")
        if "MAPGRID" not in hdr:
            raise MappingGridError("FITS header missing MAPGRID keyword")
        version = int(hdr["MAPGRID"])
        if not all(k in hdr for k in ("SCIXMIN", "SCIYMIN", "SCIXMAX", "SCIYMAX", "PADL", "PADR", "PADB", "PADT")):
            raise MappingGridError("MAPGRID=3 FITS header lacks explicit science/pad geometry")
        grid = cls(
            ffi_xmin=int(hdr["XMIN"]),
            ffi_ymin=int(hdr["YMIN"]),
            ffi_xmax=int(hdr["XMAX"]),
            ffi_ymax=int(hdr["YMAX"]),
            oversampling=int(hdr.get("OVERSAMP", hdr.get("OVERSAMPING", 1))),
            conv_pad_native=int(hdr.get("CONVPAD", DEFAULT_CONV_PAD_NATIVE)),
            mapgrid_version=version,
            science_xmin_ffi=int(hdr.get("SCIXMIN", hdr["XMIN"])),
            science_ymin_ffi=int(hdr.get("SCIYMIN", 0)),
            science_xmax_ffi=int(hdr.get("SCIXMAX", hdr["XMAX"])),
            science_ymax_ffi=int(hdr.get("SCIYMAX", hdr["YMAX"])),
        )
        if "COORDFRM" in hdr and str(hdr["COORDFRM"]).strip() != "full_ffi":
            raise MappingGridError("FITS COORDFRM must be full_ffi")
        if "SCIXMIN" in hdr and int(hdr["SCIXMIN"]) != grid.science_xmin:
            raise MappingGridError("FITS SCIXMIN disagrees with MappingGrid science bounds")
        if "SCIXMAX" in hdr and int(hdr["SCIXMAX"]) != grid.science_xmax:
            raise MappingGridError("FITS SCIXMAX disagrees with MappingGrid science bounds")
        if "SCIYMIN" in hdr and int(hdr["SCIYMIN"]) != grid.science_ymin:
            raise MappingGridError("FITS SCIYMIN disagrees with MappingGrid science bounds")
        if "SCIYMAX" in hdr and int(hdr["SCIYMAX"]) != grid.science_ymax:
            raise MappingGridError("FITS SCIYMAX disagrees with MappingGrid science bounds")
        if "GEOMFP" in hdr and str(hdr["GEOMFP"]).strip() != grid.geometry_fingerprint:
            raise MappingGridError("FITS GEOMFP does not match MappingGrid geometry")
        return grid

    def to_mapping_dict(self) -> dict[str, Any]:
        payload = {
            "ffi_xmin": self.ffi_xmin,
            "ffi_ymin": self.ffi_ymin,
            "ffi_xmax": self.ffi_xmax,
            "ffi_ymax": self.ffi_ymax,
            "oversampling_factor": self.oversampling,
            "conv_pad_native": self.conv_pad_native,
            "mapgrid_version": self.mapgrid_version,
        }
        payload.update(
            {
                "coordinate_frame": "full_ffi",
                "science_xmin": self.science_xmin,
                "science_xmax": self.science_xmax,
                "science_ymin": self.science_ymin,
                "science_ymax": self.science_ymax,
                "template_xmin": self.template_xmin,
                "template_xmax": self.template_xmax,
                "template_ymin": self.template_ymin,
                "template_ymax": self.template_ymax,
                "pad_left": self.pad_left,
                "pad_right": self.pad_right,
                "pad_bottom": self.pad_bottom,
                "pad_top": self.pad_top,
                "geometry_fingerprint": self.geometry_fingerprint,
            }
        )
        if self.mapgrid_version == MAPGRID_VERSION:
            payload["science_bounds_ffi"] = {
                "x_min": self.science_xmin, "x_max": self.science_xmax,
                "y_min": self.science_ymin, "y_max": self.science_ymax,
            }
            payload["physical_template_bounds_ffi"] = {
                "x_min": self.template_xmin, "x_max": self.template_xmax,
                "y_min": self.template_ymin, "y_max": self.template_ymax,
            }
            payload["pad_kind"] = {k: "physical" for k in ("left", "right", "bottom", "top")}
            payload["pixel_convention"] = "half_open_ffi_integer_pixels"
        return payload

    def to_fits_header_updates(self) -> dict[str, Any]:
        return {
            "XMIN": self.ffi_xmin,
            "YMIN": self.ffi_ymin,
            "XMAX": self.ffi_xmax,
            "YMAX": self.ffi_ymax,
            "MAPGRID": self.mapgrid_version,
            "CONVPAD": self.conv_pad_native,
            "OVERSAMP": self.oversampling,
            "COORDFRM": "full_ffi",
            "SCIXMIN": self.science_xmin,
            "SCIXMAX": self.science_xmax,
            "SCIYMIN": self.science_ymin,
            "SCIYMAX": self.science_ymax,
            "GEOMFP": self.geometry_fingerprint,
            "PADL": self.pad_left,
            "PADR": self.pad_right,
            "PADB": self.pad_bottom,
            "PADT": self.pad_top,
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

    # Named forms are the public geometry contract; the short names above are
    # retained for existing baseline callers.
    def ffi_to_template_local(self, ffi_x: int | float, ffi_y: int | float) -> tuple[int, int]:
        return self.ffi_to_local(ffi_x, ffi_y)

    def template_local_to_ffi(self, lx: int, ly: int) -> tuple[int, int]:
        return self.local_to_ffi(lx, ly)

    def ffi_to_science_local(self, ffi_x: int | float, ffi_y: int | float) -> tuple[int, int]:
        x, y = int(round(ffi_x)), int(round(ffi_y))
        if not self.contains_science_ffi(x, y):
            raise MappingGridError(f"FFI ({ffi_x}, {ffi_y}) outside science bounds")
        return x - self.science_xmin, y - self.science_ymin

    def science_to_ffi_local(self, lx: int, ly: int) -> tuple[int, int]:
        x, y = self.science_xmin + int(lx), self.science_ymin + int(ly)
        if not self.contains_science_ffi(x, y):
            raise MappingGridError(f"science-local ({lx}, {ly}) out of bounds")
        return x, y

    def ffi_to_template_os(self, ffi_x: int | float, ffi_y: int | float, F: int | None = None) -> tuple[int, int]:
        factor = self.oversampling if F is None else int(F)
        if factor < 1 or not self.contains_ffi(ffi_x, ffi_y):
            raise MappingGridError(f"FFI ({ffi_x}, {ffi_y}) outside template bounds")
        lx, ly = self.ffi_to_local(ffi_x, ffi_y)
        return lx * factor, ly * factor

    def science_slice_os(self) -> tuple[slice, slice]:
        f = self.oversampling
        return (
            slice((self.science_ymin - self.template_ymin) * f, (self.science_ymax - self.template_ymin) * f),
            slice((self.science_xmin - self.template_xmin) * f, (self.science_xmax - self.template_xmin) * f),
        )

    def template_shape_os(self) -> tuple[int, int]:
        return self.array_shape_os()

    def science_shape_os(self) -> tuple[int, int]:
        return ((self.science_ymax - self.science_ymin) * self.oversampling,
                (self.science_xmax - self.science_xmin) * self.oversampling)

    def science_slice_native(self) -> tuple[slice, slice]:
        """Slice selecting science S from a native template-support plane."""
        return (
            slice(self.science_ymin - self.template_ymin, self.science_ymax - self.template_ymin),
            slice(self.science_xmin - self.template_xmin, self.science_xmax - self.template_xmin),
        )

    @property
    def template_bounds_ffi(self) -> dict[str, Any]:
        return self.template_ffi_bounds()

    @property
    def science_bounds_ffi(self) -> dict[str, Any]:
        return self.science_ffi_bounds()

    def science_local_to_template_local(self, lx: int, ly: int) -> tuple[int, int]:
        """Translate S-local indices to T-local indices (native pixels)."""
        x, y = self.science_to_ffi_local(lx, ly)
        return self.ffi_to_template_local(x, y)

    def template_local_to_science_local(self, lx: int, ly: int) -> tuple[int, int]:
        """Translate T-local indices to S-local indices, rejecting padding."""
        x, y = self.template_local_to_ffi(lx, ly)
        return self.ffi_to_science_local(x, y)

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
            self.science_xmin <= x < self.science_xmax
            and self.science_ymin <= y < self.science_ymax
        )

    def science_ffi_bounds(self) -> dict[str, Any]:
        """Diff science arrays: no bottom pad rows."""
        return {
            "x_min": self.science_xmin,
            "x_max": self.science_xmax,
            "y_min": self.science_ymin,
            "y_max": self.science_ymax,
            "shape": (self.science_ymax - self.science_ymin,
                      self.science_xmax - self.science_xmin),
        }

    def science_bounds_1based(self) -> dict[str, int]:
        """1-based inclusive FFI row/col limits for masking catalogs."""
        return {
            "col_lo": int(self.science_xmin) + 1,
            "col_hi": int(self.science_xmax),
            "row_lo": int(self.science_ymin) + 1,
            "row_hi": int(self.science_ymax),
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
    """Load MappingGrid from master FITS (MAPGRID=3 + shape check)."""
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
                    f"(rebuild mapping with MAPGRID={MAPGRID_VERSION})"
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
