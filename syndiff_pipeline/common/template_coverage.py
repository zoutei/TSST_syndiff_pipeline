"""Template FITS coverage in FFI pixel coordinates."""

from __future__ import annotations

import logging

import numpy as np
from astropy.io import fits

from syndiff_pipeline.common.wcs_grouping import open_fits_memmap

log = logging.getLogger(__name__)


def template_coverage_ffi_bounds(tmpl_path: str) -> dict:
    """
    Return FFI-coordinate bounds covered by a syndiff template FITS.

    Uses ``XMIN``/``XMAX``/``YMIN``/``YMAX`` header keywords when present;
    otherwise assumes full-chip origin ``(0, 0)`` with array shape.
    """
    with open_fits_memmap(tmpl_path) as hdul:
        if hdul[0].data is not None:
            data = hdul[0].data
            hdr = hdul[0].header
        else:
            data = hdul[1].data
            hdr = hdul[1].header
        ny, nx = data.shape

    if all(k in hdr for k in ("XMIN", "XMAX", "YMIN", "YMAX")):
        return {
            "x_min": int(hdr["XMIN"]),
            "x_max": int(hdr["XMAX"]),
            "y_min": int(hdr["YMIN"]),
            "y_max": int(hdr["YMAX"]),
            "shape": (int(hdr["YMAX"]) - int(hdr["YMIN"]), int(hdr["XMAX"]) - int(hdr["XMIN"])),
        }
    return {
        "x_min": 0,
        "x_max": nx,
        "y_min": 0,
        "y_max": ny,
        "shape": (ny, nx),
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

    Raises :exc:`ValueError` when the crop extends outside template coverage.
    """
    coverage = template_coverage_ffi_bounds(tmpl_path)
    if not crop_bounds_subset_of_coverage(crop_bounds, coverage):
        raise ValueError(
            f"Diff crop {crop_bounds} extends outside template coverage {coverage} "
            f"for {tmpl_path}"
        )
    ox = coverage["x_min"]
    oy = coverage["y_min"]
    x0, x1 = crop_bounds["x_min"] - ox, crop_bounds["x_max"] - ox
    y0, y1 = crop_bounds["y_min"] - oy, crop_bounds["y_max"] - oy
    return slice(y0, y1), slice(x0, x1)


def load_template_count_cropped(tmpl_path: str, crop_bounds: dict) -> np.ndarray | None:
    """
    Load the syndiff template ``COUNT`` extension cropped to *crop_bounds*.

    Returns ``None`` when the FITS has no ``COUNT`` extension (legacy templates).
    """
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
        return np.asarray(count_hdu.data[y_slice, x_slice], dtype=np.int32)
