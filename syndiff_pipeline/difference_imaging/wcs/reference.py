"""Build tesswcs reference WCS cropped to hp_d bounds."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from astropy.io import fits
from astropy.wcs import WCS


def reference_wcs_from_tesswcs(
    sector: int,
    camera: int,
    ccd: int,
    crop_bounds: dict[str, Any],
) -> WCS:
    from tesswcs import WCS as TessWCS

    hdr = TessWCS.from_sector(int(sector), int(camera), int(ccd)).to_header()
    x_min = int(crop_bounds["x_min"])
    y_min = int(crop_bounds["y_min"])
    ny, nx = crop_bounds["shape"]

    hdr = deepcopy(hdr)
    if "CRPIX1" in hdr:
        hdr["CRPIX1"] = float(hdr["CRPIX1"]) - x_min
    if "CRPIX2" in hdr:
        hdr["CRPIX2"] = float(hdr["CRPIX2"]) - y_min
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = int(nx)
    hdr["NAXIS2"] = int(ny)
    hdr.set("XMIN", x_min, "Crop xmin in full FFI pixels")
    hdr.set("XMAX", int(crop_bounds["x_max"]), "Crop xmax (exclusive)")
    hdr.set("YMIN", y_min, "Crop ymin in full FFI pixels")
    hdr.set("YMAX", int(crop_bounds["y_max"]), "Crop ymax (exclusive)")

    wcs = WCS(hdr)
    wcs.array_shape = (int(ny), int(nx))
    return wcs
