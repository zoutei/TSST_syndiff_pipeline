"""Coordinate contract checks for MappingGrid v2 pipeline stages."""

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

    Checks pad rows (``ffi_y < 0``) and science bottom row ``(ffi_xmin, 0)``.
    """
    if tpix.ndim != 2 or tpix.shape[1] != 2:
        raise CoordinatePreflightError(
            f"tpix must be (N, 2) [ty, tx], got shape {tpix.shape}"
        )
    f = max(1, int(oversampling_factor))
    width = grid.width_native * f
    pad_ly = grid.conv_pad_native * f - 1
    if pad_ly < 0:
        return
    pad_idx = pad_ly * width
    if pad_idx >= len(tpix):
        raise CoordinatePreflightError("tpix too short for pad-row spot check")
    ty_pad, tx_pad = float(tpix[pad_idx, 0]), float(tpix[pad_idx, 1])
    if ty_pad >= 0:
        raise CoordinatePreflightError(
            f"pad-row tpix must have ffi_y < 0, got ty={ty_pad} at index {pad_idx}"
        )

    science_bottom_ly = grid.conv_pad_native * f
    science_idx = science_bottom_ly * width
    if science_idx >= len(tpix):
        raise CoordinatePreflightError("tpix too short for science-bottom spot check")
    ty_sci, tx_sci = float(tpix[science_idx, 0]), float(tpix[science_idx, 1])
    if int(round(ty_sci)) != 0:
        raise CoordinatePreflightError(
            f"science bottom row must have ffi_y=0, got ty={ty_sci}"
        )
    if int(round(tx_sci)) != grid.ffi_xmin:
        raise CoordinatePreflightError(
            f"science bottom-left must have ffi_x={grid.ffi_xmin}, got tx={tx_sci}"
        )

    # Local-index mistake would put pad row at ty ~= conv_pad_native - 1 (non-negative).
    mistaken_ty = float(grid.conv_pad_native - 1)
    mistaken_tx = 0.0
    if np.isclose(ty_pad, mistaken_ty) and np.isclose(tx_pad, mistaken_tx):
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
