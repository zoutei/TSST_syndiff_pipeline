"""Coordinate contract checks for MappingGrid v2/v3 pipeline stages."""

from __future__ import annotations

from typing import Any

import numpy as np

from syndiff_pipeline.common.mapping_grid import (
    MappingGrid,
    MappingGridError,
    compute_rkernel,
)

__all__ = [
    "CoordinatePreflightError",
    "assert_wcs_uses_ffi_coords",
    "validate_coordinate_contract",
    "validate_conv_pad_for_diff",
]


class CoordinatePreflightError(MappingGridError):
    """Raised when coordinate or padding contracts are violated."""


def assert_wcs_uses_ffi_coords(
    tpix: np.ndarray,
    grid: MappingGrid,
    *,
    oversampling_factor: int = 1,
) -> None:
    """
    Verify ``tpix`` rows are original FFI pixels, not mistaken local indices.

    Checks all four physical edges (including the baseline bottom pad) and
    rejects coordinates that look like local array indices.
    """
    if tpix.ndim != 2 or tpix.shape[1] != 2:
        raise CoordinatePreflightError(
            f"tpix must be (N, 2) [ty, tx], got shape {tpix.shape}"
        )
    f = max(1, int(oversampling_factor))
    width = grid.width_native * f
    height = grid.height_native * f
    if len(tpix) != width * height:
        raise CoordinatePreflightError(
            f"tpix length {len(tpix)} != expected grid size {width * height}"
        )

    # Every edge is sampled at both corners and the midpoint.  Expected
    # values are derived from the serialized physical bounds, never from a
    # local-index assumption or a hard-coded crop origin.
    xs = (0, width // 2, width - 1)
    ys = (0, height // 2, height - 1)
    for ly in ys:
        for lx in xs:
            idx = ly * width + lx
            expected_y = grid.ffi_ymin + (ly + 0.5) / f - 0.5
            expected_x = grid.ffi_xmin + (lx + 0.5) / f - 0.5
            got_y, got_x = map(float, tpix[idx])
            if not (np.isclose(got_y, expected_y) and np.isclose(got_x, expected_x)):
                raise CoordinatePreflightError(
                    f"tpix edge sample ({lx},{ly})={got_y, got_x} != "
                    f"physical FFI expectation {expected_y, expected_x}"
                )

    # A local-index implementation typically starts at (0, 0), whereas this
    # baseline starts at the physical x crop origin and bottom pad y origin.
    if np.isclose(float(tpix[0, 0]), 0.0) and np.isclose(float(tpix[0, 1]), 0.0):
        raise CoordinatePreflightError(
            "tpix appears to use local indices instead of FFI coordinates"
        )


def validate_coordinate_contract(
    grid: MappingGrid,
    crop_bounds: dict[str, Any],
    template_hdr: dict[str, Any] | None = None,
) -> None:
    """Cross-check science crop_bounds and optional template FITS headers."""
    science = grid.science_ffi_bounds()
    for key in ("x_min", "x_max", "y_min", "y_max", "shape"):
        if key not in crop_bounds:
            raise CoordinatePreflightError(f"crop_bounds missing {key!r}")
        crop_val = crop_bounds[key]
        science_val = science[key]
        if isinstance(crop_val, (list, tuple)) or isinstance(science_val, (list, tuple)):
            crop_val, science_val = tuple(crop_val), tuple(science_val)
        if crop_val != science_val:
            raise CoordinatePreflightError(
                f"crop_bounds[{key!r}]={crop_bounds[key]!r} != "
                f"grid.science_ffi_bounds()[{key!r}]={science[key]!r}"
            )

    if template_hdr is None:
        return
    if "MAPGRID" not in template_hdr:
        raise CoordinatePreflightError("template header missing MAPGRID=3")
    if int(template_hdr["MAPGRID"]) != 3 or int(template_hdr["MAPGRID"]) != grid.mapgrid_version:
        raise CoordinatePreflightError(
            f"template MAPGRID={template_hdr['MAPGRID']} != grid MAPGRID={grid.mapgrid_version}"
        )
    if "COORDFRM" in template_hdr and str(template_hdr["COORDFRM"]).strip() != "full_ffi":
        raise CoordinatePreflightError(
            f"template coordinate frame must be full_ffi, got {template_hdr['COORDFRM']!r}"
        )
    for key, attr in (
        ("XMIN", "ffi_xmin"),
        ("YMIN", "ffi_ymin"),
        ("XMAX", "ffi_xmax"),
        ("YMAX", "ffi_ymax"),
    ):
        if key in template_hdr and int(template_hdr[key]) != getattr(grid, attr):
            raise CoordinatePreflightError(
                f"template header {key}={template_hdr[key]} != grid.{attr}"
            )
    if "CONVPAD" in template_hdr and int(template_hdr["CONVPAD"]) != grid.conv_pad_native:
        raise CoordinatePreflightError(
            f"template CONVPAD={template_hdr['CONVPAD']} != grid.conv_pad_native"
        )
    for key, value in (("PADL", grid.pad_left), ("PADR", grid.pad_right),
                       ("PADB", grid.pad_bottom), ("PADT", grid.pad_top)):
        if key not in template_hdr or int(template_hdr[key]) != value:
            raise CoordinatePreflightError(f"template {key} does not match MAPGRID=3 geometry")
    expected_fp = grid.geometry_fingerprint
    if "GEOMFP" in template_hdr and str(template_hdr["GEOMFP"]).strip() != expected_fp:
        raise CoordinatePreflightError(
            f"template GEOMFP={template_hdr['GEOMFP']!r} != grid geometry fingerprint {expected_fp}"
        )


def validate_conv_pad_for_diff(
    grid: MappingGrid,
    *,
    scale_px: float,
) -> None:
    """
    Fail if diff kernel half-width exceeds mapping-time CONVPAD.

    ``spare`` margin is mapping-only; diff checks ``rkernel_diff <= conv_pad_native``.
    """
    rkernel_diff = compute_rkernel(scale_px)
    if rkernel_diff > grid.conv_pad_native:
        raise CoordinatePreflightError(
            f"diff rkernel={rkernel_diff} exceeds mapping CONVPAD={grid.conv_pad_native}; "
            "rebuild mapping with larger template_conv_pad_spare_px"
        )
