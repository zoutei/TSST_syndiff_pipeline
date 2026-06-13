"""
DS9 region files marking the science target and additional forced-photometry sources.

Coordinates use DS9 ``image`` space in the **cropped ROI** (crop-local astropy
pixel indices + 1). This matches pipeline ``x``/``y`` and cropped diff/science FITS.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.stages import photometry
from syndiff_pipeline.difference_imaging.support.paths import TARGETS_DS9_REGION_BASENAME

log = logging.getLogger(__name__)

_DEFAULT_CIRCLE_RADIUS_PX = 3.0


def primary_target_region_label(
    target_name: str,
    sector: int,
    camera: int,
    ccd: int,
) -> str:
    """Region text label: ``{name}_s{sector}_c{camera}_k{ccd}``."""
    safe = re.sub(r"[^\w.-]+", "_", str(target_name).strip())
    return f"{safe}_s{int(sector):04d}_c{int(camera)}_k{int(ccd)}"


def crop_local_to_ds9_xy(x_crop: float, y_crop: float) -> tuple[float, float]:
    """Convert crop-local 0-based (x, y) to DS9 1-based ROI image coords."""
    return float(x_crop) + 1.0, float(y_crop) + 1.0


def _ref_manifest_row_index(
    wcs_table: pd.DataFrame, ref_ffi_path: str
) -> Optional[int]:
    path_col = "path" if "path" in wcs_table.columns else "filename"
    try:
        ref_r = Path(ref_ffi_path).resolve()
    except Exception:
        ref_r = Path(os.path.expanduser(ref_ffi_path))
    ref_abs = os.path.abspath(os.path.expanduser(str(ref_ffi_path)))
    for i in range(len(wcs_table)):
        p = wcs_table.iloc[i].get(path_col)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        ps = str(p).strip()
        if not ps:
            continue
        try:
            if Path(ps).resolve() == ref_r:
                return i
        except Exception:
            if os.path.abspath(os.path.expanduser(ps)) == ref_abs:
                return i
    return None


def _representative_crop_xy(
    target_xy: np.ndarray, ref_idx: Optional[int]
) -> tuple[float, float]:
    arr = np.asarray(target_xy, dtype=np.float64)
    if ref_idx is not None and 0 <= ref_idx < len(arr):
        x, y = float(arr[ref_idx, 0]), float(arr[ref_idx, 1])
        if np.isfinite(x) and np.isfinite(y):
            return x, y
    mx = float(np.nanmedian(arr[:, 0]))
    my = float(np.nanmedian(arr[:, 1]))
    if not (np.isfinite(mx) and np.isfinite(my)):
        raise ValueError("no finite crop-local (x, y) for target")
    return mx, my


def iter_target_ds9_circles(
    *,
    target_ra: float,
    target_dec: float,
    additional_forced_targets: Iterable[dict],
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    ref_ffi_path: str,
    primary_label: str,
) -> list[tuple[str, float, float, str]]:
    """
    Return ``(label, x_ds9, y_ds9, color)`` for primary (green) and extras (blue).

    Positions are crop-local ROI coordinates (+1 for DS9), using the reference
    FFI row when available.
    """
    ref_idx = _ref_manifest_row_index(wcs_table, ref_ffi_path)
    primary_xy = photometry.per_frame_target_crop_xy(
        wcs_table, float(target_ra), float(target_dec), crop_bounds
    )
    pcx, pcy = _representative_crop_xy(primary_xy, ref_idx)
    px, py = crop_local_to_ds9_xy(pcx, pcy)
    circles: list[tuple[str, float, float, str]] = [
        (primary_label, px, py, "green"),
    ]

    for pt in additional_forced_targets:
        name = str(pt.get("name", "extra"))
        extra_xy = photometry.resolve_forced_target_xy(
            pt, primary_xy, wcs_table, crop_bounds
        )
        cx, cy = _representative_crop_xy(extra_xy, ref_idx)
        x_ds9, y_ds9 = crop_local_to_ds9_xy(cx, cy)
        circles.append((name, x_ds9, y_ds9, "blue"))

    return circles


def format_targets_ds9_regions(
    circles: Iterable[tuple[str, float, float, str]],
    *,
    crop_bounds: Optional[dict] = None,
    circle_radius_px: float = _DEFAULT_CIRCLE_RADIUS_PX,
) -> str:
    """Format DS9 region file text (``image`` coordinates, 1-based crop-local)."""
    lines = [
        "# Region file format: DS9 version 4.1",
    ]
    if crop_bounds is not None:
        xm = int(crop_bounds.get("x_min", 0))
        ym = int(crop_bounds.get("y_min", 0))
        shape = crop_bounds.get("shape")
        shape_s = f"{shape[1]}x{shape[0]}" if shape is not None else "?"
        lines.append(
            f"# crop-local image coords (1-based); FFI ROI origin x_min={xm} "
            f"y_min={ym} size={shape_s}"
        )
    lines.extend(
        [
            "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" "
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
            "image",
        ]
    )
    r = float(circle_radius_px)
    for label, x, y, color in circles:
        safe = str(label).replace(")", "").replace("(", "")
        lines.append(
            f"circle({x:.4f},{y:.4f},{r:.2f}) # color={color} text={{{safe}}}"
        )
    return "\n".join(lines) + "\n"


def write_targets_ds9_regions(
    output_dir: str,
    *,
    target_ra: float,
    target_dec: float,
    target_name: str,
    sector: int,
    camera: int,
    ccd: int,
    additional_forced_targets: Iterable[dict],
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    ref_ffi_path: str,
    circle_radius_px: float = _DEFAULT_CIRCLE_RADIUS_PX,
    out_basename: str = TARGETS_DS9_REGION_BASENAME,
) -> str:
    """
    Write ``{output_dir}/{out_basename}`` with primary (green) and extra (blue) circles.

    Returns the absolute path of the region file.
    """
    primary_label = primary_target_region_label(target_name, sector, camera, ccd)
    circles = iter_target_ds9_circles(
        target_ra=target_ra,
        target_dec=target_dec,
        additional_forced_targets=additional_forced_targets,
        wcs_table=wcs_table,
        crop_bounds=crop_bounds,
        ref_ffi_path=ref_ffi_path,
        primary_label=primary_label,
    )
    content = format_targets_ds9_regions(
        circles, crop_bounds=crop_bounds, circle_radius_px=circle_radius_px
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / out_basename
    out_path.write_text(content, encoding="utf-8")
    log.info("Wrote DS9 target regions %s (%d sources)", out_path, len(circles))
    return str(out_path)
