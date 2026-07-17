"""Helpers to resolve an FFI id → per-epoch mask FITS."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    tess_product_id_from_ffi_path,
)
from syndiff_pipeline.masking.catalog import MaskCatalog

log = logging.getLogger(__name__)

_DIGITS_ONLY = re.compile(r"^(\d{10,})$")
_TESS_DIGITS = re.compile(r"^(?:tess)?(\d{10,})$", re.IGNORECASE)


def normalize_ffi_product_id(ffi_id: str | int) -> str:
    """
    Normalize an FFI identifier to ``tess<digits>``.

    Accepts a product id (``tess2020019142923``), bare digits, or any path /
    basename that starts with the SPOC product id.
    """
    if isinstance(ffi_id, (int, np.integer)):
        return f"tess{int(ffi_id)}"
    s = str(ffi_id).strip()
    pid = tess_product_id_from_ffi_path(s)
    if pid:
        return pid
    m = _TESS_DIGITS.match(s)
    if m:
        return f"tess{m.group(1)}"
    raise ValueError(f"Cannot parse FFI product id from {ffi_id!r}")


def _manifest_product_id_series(wcs_table: pd.DataFrame) -> pd.Series:
    if "product_id" in wcs_table.columns:
        return wcs_table["product_id"].astype(str)
    col = "filename" if "filename" in wcs_table.columns else "path"
    if col not in wcs_table.columns:
        raise ValueError("wcs_table needs product_id, filename, or path")
    return wcs_table[col].map(lambda x: tess_product_id_from_ffi_path(str(x)) or "")


def lookup_ffi_row(wcs_table: pd.DataFrame, ffi_id: str | int) -> pd.Series:
    """Return the manifest row for *ffi_id* (raises if missing)."""
    pid = normalize_ffi_product_id(ffi_id)
    pids = _manifest_product_id_series(wcs_table)
    hit = wcs_table.loc[pids == pid]
    if hit.empty:
        # also try bare digits match against product ids
        digits = pid.replace("tess", "")
        hit = wcs_table.loc[pids.str.replace("tess", "", regex=False) == digits]
    if hit.empty:
        raise KeyError(f"FFI id {ffi_id!r} ({pid}) not found in manifest")
    return hit.iloc[0]


def btjd_for_ffi_id(wcs_table: pd.DataFrame, ffi_id: str | int) -> float:
    """Look up BTJD for an FFI id from the frame manifest."""
    row = lookup_ffi_row(wcs_table, ffi_id)
    if "btjd" not in row.index or not np.isfinite(float(row["btjd"])):
        raise ValueError(f"No finite btjd for FFI {ffi_id!r}")
    return float(row["btjd"])


def mask_array_for_ffi_id(
    catalog: MaskCatalog,
    ffi_id: str | int,
    *,
    wcs_table: pd.DataFrame | None = None,
    which: str = "full",
    as_bool: bool = False,
) -> np.ndarray:
    """
    Return the mask array for one FFI.

    *ffi_id* may be a product id, bare digits, path, **or** an integer cadence
    when *wcs_table* is omitted (then *ffi_id* is treated as cadence).
    """
    if wcs_table is None:
        # cadence or btjd passthrough
        return catalog.mask_at(ffi_id, which=which, as_bool=as_bool)  # type: ignore[arg-type]
    btjd = btjd_for_ffi_id(wcs_table, ffi_id)
    return catalog.mask_at(btjd, which=which, as_bool=as_bool)


def write_mask_fits_for_ffi(
    catalog: MaskCatalog,
    ffi_id: str | int,
    out_path: str | Path,
    *,
    wcs_table: pd.DataFrame | None = None,
    which: str = "full",
    overwrite: bool = True,
) -> Path:
    """
    Write an int16 mask FITS for one FFI id.

    Header keywords: ``FFI_ID``, ``BTJD`` (when known), ``MASKWHCH``, ``CADENCE``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()
    btjd = None
    cadence = None
    pid = None
    if wcs_table is not None:
        pid = normalize_ffi_product_id(ffi_id)
        btjd = btjd_for_ffi_id(wcs_table, ffi_id)
        cadence = catalog.resolve_cadence(btjd)
        mask = catalog.mask_at(btjd, which=which)
        hdr["FFI_ID"] = (pid, "TESS FFI product id")
        hdr["BTJD"] = (float(btjd), "Barycentric TESS JD")
    else:
        mask = catalog.mask_at(ffi_id, which=which)  # type: ignore[arg-type]
        if isinstance(ffi_id, (int, np.integer)) or (
            isinstance(ffi_id, str) and _DIGITS_ONLY.match(str(ffi_id))
        ):
            cadence = int(ffi_id)
            hdr["FFI_ID"] = (str(ffi_id), "cadence or raw id")
        else:
            try:
                pid = normalize_ffi_product_id(ffi_id)
                hdr["FFI_ID"] = (pid, "TESS FFI product id")
            except ValueError:
                hdr["FFI_ID"] = (str(ffi_id), "raw id")

    if cadence is not None:
        hdr["CADENCE"] = (int(cadence), "0-based asteroid cadence index")
    hdr["MASKWHCH"] = (which, "MaskCatalog.mask_at which=")
    hdr["BITPIX"] = 16

    hdu = fits.PrimaryHDU(np.asarray(mask, dtype=np.int16), header=hdr)
    hdu.writeto(out_path, overwrite=overwrite)
    log.info("Wrote mask FITS → %s (ffi=%s cadence=%s)", out_path, pid or ffi_id, cadence)
    return out_path.resolve()


def select_begin_mid_end_ffi_ids(
    wcs_table: pd.DataFrame,
) -> tuple[str, str, str]:
    """
    Pick three FFI product ids at the beginning, middle, and end of the sector.

    Rows with non-finite ``btjd`` are dropped; remaining rows are sorted by BTJD.
    """
    df = wcs_table.copy()
    if "btjd" not in df.columns:
        raise ValueError("wcs_table must have a btjd column")
    df = df.loc[np.isfinite(pd.to_numeric(df["btjd"], errors="coerce"))].copy()
    if df.empty:
        raise ValueError("No rows with finite btjd in wcs_table")
    df["_pid"] = _manifest_product_id_series(df)
    df = df.loc[df["_pid"].astype(bool)].sort_values("btjd").reset_index(drop=True)
    n = len(df)
    idxs = (0, n // 2, n - 1)
    return tuple(str(df.iloc[i]["_pid"]) for i in idxs)  # type: ignore[return-value]


def select_asteroid_active_ffi_ids(
    catalog: MaskCatalog,
    wcs_table: pd.DataFrame,
) -> tuple[str, str, str]:
    """
    Pick three FFI ids at early / mid / late cadences that have bit-128 pixels.

    Useful when sector begin/mid/end fall outside the asteroid window for the crop.
    """
    if not catalog.has_temporal() or catalog.asteroid_times is None:
        raise ValueError("catalog has no asteroid intervals/times")
    iv = catalog.asteroid_intervals
    assert iv is not None
    active: set[int] = set()
    for lo, hi in zip(iv["cadence_lo"].to_numpy(int), iv["cadence_hi"].to_numpy(int)):
        active.update(range(int(lo), int(hi) + 1))
    if not active:
        raise ValueError("No active asteroid cadences in crop")
    cads = sorted(active)
    picks = (cads[0], cads[len(cads) // 2], cads[-1])
    times = catalog.asteroid_times.set_index("cadence")["btjd"]
    pids: list[str] = []
    df = wcs_table.loc[
        np.isfinite(pd.to_numeric(wcs_table["btjd"], errors="coerce"))
    ].copy()
    df["_pid"] = _manifest_product_id_series(df)
    btjd = pd.to_numeric(df["btjd"], errors="coerce").to_numpy(float)
    for c in picks:
        t = float(times.loc[c])
        j = int(np.argmin(np.abs(btjd - t)))
        pids.append(str(df.iloc[j]["_pid"]))
    return pids[0], pids[1], pids[2]


def write_sector_sample_mask_fits(
    catalog: MaskCatalog,
    wcs_table: pd.DataFrame,
    out_dir: str | Path,
    *,
    which: str = "full",
    prefix: str = "mask",
) -> list[Path]:
    """
    Write three mask FITS files for begin / mid / end FFIs of the sector.

    Filenames: ``{prefix}_{label}_{product_id}.fits`` with
    ``label`` in ``begin``, ``mid``, ``end``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    begin, mid, end = select_begin_mid_end_ffi_ids(wcs_table)
    paths: list[Path] = []
    for label, pid in (("begin", begin), ("mid", mid), ("end", end)):
        dest = out_dir / f"{prefix}_{label}_{pid}.fits"
        paths.append(
            write_mask_fits_for_ffi(
                catalog, pid, dest, wcs_table=wcs_table, which=which
            )
        )
    return paths


def load_catalog_for_event(
    ws_root: str | Path,
    *,
    crop_bounds: dict | None = None,
    data_root: str | Path | None = None,
    sector: int | None = None,
    camera: int | None = None,
    ccd: int | None = None,
) -> MaskCatalog:
    """Load MaskCatalog from an event workspace (+ optional SCC asteroids)."""
    from syndiff_pipeline.masking.asteroids import load_asteroid_products
    from syndiff_pipeline.masking.settings import default_asteroid_intervals_dir

    cat = MaskCatalog.from_workspace(ws_root, crop_bounds=crop_bounds)
    if cat.has_temporal() or data_root is None or sector is None:
        return cat
    root = default_asteroid_intervals_dir(data_root, sector, camera or 1, ccd or 1)
    iv, tm = load_asteroid_products(root)
    if iv is None:
        return cat
    return MaskCatalog.from_arrays(
        cat.static,
        tns_table=cat.tns_table,
        asteroid_intervals_ffi=iv,
        asteroid_times=tm,
        crop_bounds=crop_bounds,
    )
