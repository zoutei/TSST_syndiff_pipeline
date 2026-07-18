"""Template FITS coverage in FFI pixel coordinates."""

from __future__ import annotations

import logging

import numpy as np
from astropy.io import fits

from syndiff_pipeline.common.wcs_grouping import open_fits_memmap

log = logging.getLogger(__name__)


def _oversampling_from_header(hdr: fits.Header) -> int:
    """Return template oversampling factor from ``OVERSAMP`` (default 1)."""
    if "OVERSAMP" not in hdr:
        return 1
    try:
        factor = int(hdr["OVERSAMP"])
    except (TypeError, ValueError):
        return 1
    return max(1, factor)


def template_coverage_ffi_bounds(tmpl_path: str) -> dict:
    """
    Return FFI-coordinate bounds covered by a syndiff template FITS.

    Uses ``XMIN``/``XMAX``/``YMIN``/``YMAX`` header keywords when present;
    otherwise assumes full-chip origin ``(0, 0)`` with array shape divided by
    ``OVERSAMP`` when that keyword is set.

    Coverage bounds and ``shape`` are always in **base (native) FFI pixels**.
    ``oversampling_factor`` is the template array oversampling relative to that
    native grid (1 = native).
    """
    with open_fits_memmap(tmpl_path) as hdul:
        primary = hdul[0]
        if primary.data is not None:
            data = primary.data
            hdr = primary.header
        else:
            data = hdul[1].data
            # Extension headers usually carry XMIN/OVERSAMP; primary-only
            # keywords (some hand-written / star mini templates) still count.
            hdr = hdul[1].header.copy()
            for key in ("OVERSAMP", "XMIN", "XMAX", "YMIN", "YMAX"):
                if key not in hdr and key in primary.header:
                    hdr[key] = primary.header[key]
        ny, nx = data.shape
        os_factor = _oversampling_from_header(hdr)

    if all(k in hdr for k in ("XMIN", "XMAX", "YMIN", "YMAX")):
        x_min = int(hdr["XMIN"])
        x_max = int(hdr["XMAX"])
        y_min = int(hdr["YMIN"])
        y_max = int(hdr["YMAX"])
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "shape": (y_max - y_min, x_max - x_min),
            "oversampling_factor": os_factor,
        }

    if os_factor > 1:
        if ny % os_factor != 0 or nx % os_factor != 0:
            raise ValueError(
                f"Template {tmpl_path} shape {(ny, nx)} is not divisible by "
                f"OVERSAMP={os_factor}"
            )
        native_ny, native_nx = ny // os_factor, nx // os_factor
    else:
        native_ny, native_nx = ny, nx

    return {
        "x_min": 0,
        "x_max": native_nx,
        "y_min": 0,
        "y_max": native_ny,
        "shape": (native_ny, native_nx),
        "oversampling_factor": os_factor,
    }


def crop_bounds_subset_of_coverage(crop_bounds: dict, coverage: dict) -> bool:
    """True when *crop_bounds* lies inside template *coverage* (FFI coords)."""
    return (
        crop_bounds["x_min"] >= coverage["x_min"]
        and crop_bounds["y_min"] >= coverage["y_min"]
        and crop_bounds["x_max"] <= coverage["x_max"]
        and crop_bounds["y_max"] <= coverage["y_max"]
    )


def template_crop_slices(tmpl_path: str, crop_bounds: dict) -> tuple[slice, slice]:
    """
    Return ``(y_slice, x_slice)`` into a template image for *crop_bounds* (FFI coords).

    When the template is oversampled (``OVERSAMP`` > 1), slices are scaled into
    the high-resolution array while *crop_bounds* remain native FFI coordinates.

    Raises :exc:`ValueError` when the crop extends outside template coverage.
    """
    coverage = template_coverage_ffi_bounds(tmpl_path)
    if not crop_bounds_subset_of_coverage(crop_bounds, coverage):
        raise ValueError(
            f"Diff crop {crop_bounds} extends outside template coverage {coverage} "
            f"for {tmpl_path}"
        )
    os_factor = int(coverage.get("oversampling_factor") or 1)
    ox = coverage["x_min"]
    oy = coverage["y_min"]
    x0 = (crop_bounds["x_min"] - ox) * os_factor
    x1 = (crop_bounds["x_max"] - ox) * os_factor
    y0 = (crop_bounds["y_min"] - oy) * os_factor
    y1 = (crop_bounds["y_max"] - oy) * os_factor
    return slice(y0, y1), slice(x0, x1)


def block_sum_oversampled_to_native(
    arr: np.ndarray,
    oversampling_factor: int,
) -> np.ndarray:
    """Sum ``F×F`` HR blocks into native TESS pixels (COUNT-style reduce)."""
    factor = max(1, int(oversampling_factor))
    if factor <= 1:
        return np.asarray(arr)
    data = np.asarray(arr)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {data.shape}")
    ny, nx = data.shape
    if ny % factor != 0 or nx % factor != 0:
        raise ValueError(
            f"Array shape {data.shape} not divisible by oversampling_factor={factor}"
        )
    return data.reshape(ny // factor, factor, nx // factor, factor).sum(axis=(1, 3))


def load_template_count_cropped(tmpl_path: str, crop_bounds: dict) -> np.ndarray | None:
    """
    Load the syndiff template ``COUNT`` extension cropped to *crop_bounds*.

    Returns a **native**-resolution COUNT plane (HR COUNT is block-summed when
    ``OVERSAMP`` > 1) so it matches science/ref shapes for PS1 coverage masking.

    Returns ``None`` when the FITS has no ``COUNT`` extension (legacy templates).
    """
    coverage = template_coverage_ffi_bounds(tmpl_path)
    y_slice, x_slice = template_crop_slices(tmpl_path, crop_bounds)
    with open_fits_memmap(tmpl_path) as hdul:
        try:
            count_hdu = hdul["COUNT"]
        except KeyError:
            log.warning(
                "Template %s has no COUNT extension; skipping PS1 coverage mask",
                tmpl_path,
            )
            return None
        count = np.asarray(count_hdu.data[y_slice, x_slice], dtype=np.int32)
    os_factor = int(coverage.get("oversampling_factor") or 1)
    if os_factor > 1:
        count = block_sum_oversampled_to_native(count, os_factor).astype(np.int32)
    return count
