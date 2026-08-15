"""Science/template array pairing for diff stages (pad + trim contract)."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.common.mapping_grid import MappingGrid, MappingGridError

__all__ = [
    "pad_mask_bottom",
    "prepare_science_template_pairing",
    "trim_padded_products",
    "zero_pad_science_bottom",
]


def zero_pad_science_bottom(
    sci_native: np.ndarray,
    pad_rows: int,
) -> np.ndarray:
    """Zero-pad science array at the bottom by ``pad_rows`` native rows."""
    pad = int(pad_rows)
    if pad < 0:
        raise MappingGridError(f"pad_rows must be >= 0, got {pad}")
    if pad == 0:
        return np.asarray(sci_native)
    sci = np.asarray(sci_native)
    if sci.ndim != 2:
        raise MappingGridError(f"science array must be 2-D, got shape {sci.shape}")
    out = np.zeros((sci.shape[0] + pad, sci.shape[1]), dtype=sci.dtype)
    out[pad:, :] = sci
    return out


def pad_mask_bottom(mask_native: np.ndarray, pad_rows: int) -> np.ndarray:
    """Bottom-pad a boolean Hotpants/phot mask, marking new rows as bad.

    ``zero_pad_science_bottom`` is correct for flux/error arrays, where the
    padded rows are legitimately zero-valued science. A mask's padded rows
    are fabricated pad geometry with no real observation behind them --
    filling them with the mask's own "good" value (0/False, per
    ``difference_imaging.masking.bits``) would hand Hotpants' substamp and
    kernel-fit selection a strip of data that looks like real, valid,
    flat-zero sky. Pad with ``True`` (masked/excluded) instead -- same
    "edge, not real sky" role as the ``EDGE`` static-mask bit.
    """
    pad = int(pad_rows)
    if pad < 0:
        raise MappingGridError(f"pad_rows must be >= 0, got {pad}")
    if pad == 0:
        return np.asarray(mask_native, dtype=bool)
    mask = np.asarray(mask_native, dtype=bool)
    if mask.ndim != 2:
        raise MappingGridError(f"mask array must be 2-D, got shape {mask.shape}")
    out = np.ones((mask.shape[0] + pad, mask.shape[1]), dtype=bool)
    out[pad:, :] = mask
    return out


def trim_padded_products(arr: np.ndarray, pad_rows: int) -> np.ndarray:
    """Remove bottom pad rows from a native diff-stage product."""
    pad = int(pad_rows)
    if pad <= 0:
        return np.asarray(arr)
    data = np.asarray(arr)
    if data.shape[0] <= pad:
        raise MappingGridError(
            f"cannot trim {pad} rows from array with height {data.shape[0]}"
        )
    return data[pad:, :]


def prepare_science_template_pairing(
    sci_native: np.ndarray,
    tmpl_native: np.ndarray,
    grid: MappingGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad science to match template height for kernel_fit / hotpants / convolve.

    Template is expected at ``grid.template_ffi_bounds()`` shape; science at
    ``grid.science_ffi_bounds()`` shape.
    """
    pad = grid.conv_pad_native
    sci = np.asarray(sci_native)
    tmpl = np.asarray(tmpl_native)
    expected_sci = grid.science_ffi_bounds()["shape"]
    expected_tmpl = grid.template_ffi_bounds()["shape"]
    if tuple(sci.shape) != tuple(expected_sci):
        raise MappingGridError(
            f"science shape {sci.shape} != expected {expected_sci}"
        )
    if tuple(tmpl.shape) != tuple(expected_tmpl):
        raise MappingGridError(
            f"template shape {tmpl.shape} != expected {expected_tmpl}"
        )
    sci_padded = zero_pad_science_bottom(sci, pad)
    if sci_padded.shape != tmpl.shape:
        raise MappingGridError(
            f"padded science {sci_padded.shape} != template {tmpl.shape}"
        )
    return sci_padded, tmpl
