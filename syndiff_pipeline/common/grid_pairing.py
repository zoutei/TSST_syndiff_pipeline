"""Science/template array pairing for diff stages (pad + trim contract)."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.common.mapping_grid import MappingGrid, MappingGridError

__all__ = [
    "pad_mask_bottom",
    "prepare_science_template_pairing",
    "trim_padded_products",
    "zero_pad_science_bottom",
    "pad_science_array",
    "pad_mask_to_template",
]


def zero_pad_science_bottom(
    sci_native: np.ndarray,
    pad_rows: int,
) -> np.ndarray:
    """Reject the removed bottom-only padding API."""
    raise MappingGridError(
        "bottom-only science padding was removed; use pad_science_array with MAPGRID=3"
    )


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
    raise MappingGridError(
        "bottom-only mask padding was removed; use pad_mask_to_template with MAPGRID=3"
    )


def trim_padded_products(
    arr: np.ndarray,
    pad_rows: int | None = None,
    *,
    grid: MappingGrid | None = None,
) -> np.ndarray:
    """Trim a T product to S using the canonical MAPGRID=3 grid slice."""
    if grid is not None:
        data = np.asarray(arr)
        expected = tuple(grid.template_ffi_bounds()["shape"])
        if data.ndim < 2 or tuple(data.shape[-2:]) != expected:
            raise MappingGridError(
                f"product shape {data.shape} does not end in template shape {expected}"
            )
        ys, xs = grid.science_slice_native()
        return data[..., ys, xs]
    raise MappingGridError("trim_padded_products requires MappingGrid MAPGRID=3")


def _science_shape(grid: MappingGrid) -> tuple[int, int]:
    """Return the actual S array shape for MAPGRID=3."""
    return (
        int(grid.science_ymax - grid.science_ymin),
        int(grid.science_xmax - grid.science_xmin),
    )


def pad_science_array(
    science: np.ndarray,
    grid: MappingGrid,
    *,
    neutral_value: float = 0.0,
) -> np.ndarray:
    """Place an S array into the exact T rectangle, padding all four edges.

    The fabricated pixels are deliberately neutral-valued; callers must pair
    this with :func:`pad_mask_to_template` so they cannot be selected as real
    sky by Hotpants or kernel fitting.
    """
    if int(grid.mapgrid_version) != 3:
        raise MappingGridError("science padding requires MAPGRID=3")
    data = np.asarray(science)
    expected = _science_shape(grid)
    if data.ndim != 2 or tuple(data.shape) != expected:
        raise MappingGridError(f"science shape {data.shape} != expected {expected}")
    out = np.full(grid.template_ffi_bounds()["shape"], neutral_value, dtype=data.dtype)
    ys, xs = grid.science_slice_native()
    out[ys, xs] = data
    return out


def pad_mask_to_template(mask: np.ndarray, grid: MappingGrid) -> np.ndarray:
    """Place an S boolean mask into T, excluding every fabricated edge pixel."""
    if int(grid.mapgrid_version) != 3:
        raise MappingGridError("mask padding requires MAPGRID=3")
    data = np.asarray(mask, dtype=bool)
    expected = _science_shape(grid)
    if data.ndim != 2 or tuple(data.shape) != expected:
        raise MappingGridError(f"mask shape {data.shape} != expected {expected}")
    out = np.ones(grid.template_ffi_bounds()["shape"], dtype=bool)
    ys, xs = grid.science_slice_native()
    out[ys, xs] = data
    return out


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
    sci = np.asarray(sci_native)
    tmpl = np.asarray(tmpl_native)
    expected_sci = _science_shape(grid)
    expected_tmpl = grid.template_ffi_bounds()["shape"]
    if tuple(sci.shape) != tuple(expected_sci):
        raise MappingGridError(
            f"science shape {sci.shape} != expected {expected_sci}"
        )
    if tuple(tmpl.shape) != tuple(expected_tmpl):
        raise MappingGridError(
            f"template shape {tmpl.shape} != expected {expected_tmpl}"
        )
    sci_padded = pad_science_array(sci, grid)
    if sci_padded.shape != tmpl.shape:
        raise MappingGridError(
            f"padded science {sci_padded.shape} != template {tmpl.shape}"
        )
    return sci_padded, tmpl
