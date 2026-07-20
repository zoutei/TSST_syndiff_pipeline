"""
Per-SCC ``ffi_list`` inventory: one row per FFI with full HDU1 header blob.

Opening a TESS FFI (``.fits.fz`` / ``.fits.gz``, tens of MB) just to read its
WCS header is CPU-bound on decompression. The ``ffi_list`` is SCC-shared
(``data_root``-scoped) so mapping, bind, and field remap never re-pay that cost
once the list is complete.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from filelock import FileLock

from syndiff_pipeline.common.download import manifest_basename_from_local
from syndiff_pipeline.common.scc_paths import (
    FFI_LIST_CSV_BASENAME,
    scc_ffi_list_csv,
    scc_ffi_list_parquet,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# WCS header keywords needed to build an astropy WCS. Single source of truth --
# syndiff_pipeline.common.wcs_grouping imports these rather than redefining them.
WCS_KEYS = [
    "NAXIS", "NAXIS1", "NAXIS2",
    "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
]
SIP_KEY_PREFIXES = ("A_", "B_", "AP_", "BP_")

_MIN_KEYS = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD2_2")

_SLIM_CSV_COLUMNS = (
    "filename",
    "wcs_ok",
    "DATE-OBS",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "NAXIS1",
    "NAXIS2",
)


def ffi_list_parquet_path(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Path to the shared per-SCC ``ffi_list.parquet``."""
    return scc_ffi_list_parquet(data_root, sector, camera, ccd)


def ffi_list_csv_path(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Path to the slim CSV twin of the shared FFI list."""
    return scc_ffi_list_csv(data_root, sector, camera, ccd)


def _wcs_header_complete(header: fits.Header) -> bool:
    for key in _MIN_KEYS:
        if key not in header:
            return False
    return True


def _header_cards_bytes(hdr: fits.Header) -> bytes:
    raw = hdr.tostring()
    if isinstance(raw, bytes):
        return raw
    return raw.encode("latin1")


def extract_ffi_header_record(path: str | Path, *, open_fits: Callable) -> dict:
    """
    Open one FFI and snapshot HDU1 header into an ``ffi_list`` row.

    Always returns a row (``wcs_ok=False`` on open/extract failure).
    """
    row = {
        "filename": manifest_basename_from_local(path),
        "wcs_ok": False,
        "date_obs": None,
        "header_cards": b"",
        "schema_version": SCHEMA_VERSION,
    }
    try:
        with open_fits(path) as hdul:
            hdr = hdul[1].header
            row["header_cards"] = _header_cards_bytes(hdr)
            row["date_obs"] = hdr.get("DATE-OBS")
            row["wcs_ok"] = _wcs_header_complete(hdr)
    except Exception as exc:
        log.warning("ffi_list: could not extract header from %s: %s", path, exc)
    return row


def load_ffi_list(ffi_list_path: str | Path) -> pd.DataFrame:
    """Load ``ffi_list.parquet``, or an empty frame indexed by logical filename."""
    ffi_list_path = Path(ffi_list_path)
    if not ffi_list_path.is_file():
        return pd.DataFrame(index=pd.Index([], name="filename", dtype="object"))
    df = pd.read_parquet(ffi_list_path)
    if "filename" in df.columns:
        df = df.set_index("filename")
    df.index = df.index.astype(str)
    return df


def logical_keys_from_paths(paths: Iterable[str | Path]) -> set[str]:
    return {manifest_basename_from_local(p) for p in paths}


def ffi_list_is_complete(
    local_paths: Sequence[str | Path],
    ffi_list_df: pd.DataFrame,
) -> bool:
    """True when every local logical FFI key has a row (any ``wcs_ok``)."""
    if ffi_list_df.empty and local_paths:
        return False
    required = logical_keys_from_paths(local_paths)
    if not required:
        return True
    have = set(ffi_list_df.index.astype(str))
    return required.issubset(have)


def _slim_csv_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for filename, series in df.iterrows():
        row = {
            "filename": str(filename),
            "wcs_ok": bool(series.get("wcs_ok", False)),
            "DATE-OBS": series.get("date_obs"),
            "CRVAL1": None,
            "CRVAL2": None,
            "CRPIX1": None,
            "CRPIX2": None,
            "NAXIS1": None,
            "NAXIS2": None,
        }
        cards = series.get("header_cards")
        if cards is not None and not (isinstance(cards, float) and pd.isna(cards)):
            try:
                hdr = header_from_cached_row(series)
                row["DATE-OBS"] = row["DATE-OBS"] or hdr.get("DATE-OBS")
                for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "NAXIS1", "NAXIS2"):
                    if key in hdr:
                        row[key] = hdr[key]
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows, columns=list(_SLIM_CSV_COLUMNS))


def _write_ffi_list_artifacts(ffi_list_path: Path, combined: pd.DataFrame) -> None:
    ffi_list_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ffi_list_path.with_suffix(".parquet.tmp")
    out = combined.reset_index()
    out.to_parquet(tmp, index=False)
    os.replace(tmp, ffi_list_path)
    csv_path = ffi_list_path.with_name(FFI_LIST_CSV_BASENAME)
    slim = _slim_csv_rows(combined)
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    slim.to_csv(csv_tmp, index=False)
    os.replace(csv_tmp, csv_path)


def upsert_ffi_list_rows(ffi_list_path: str | Path, rows: Sequence[dict]) -> None:
    """Locked merge-by-filename upsert with atomic parquet + slim CSV write."""
    if not rows:
        return
    ffi_list_path = Path(ffi_list_path)
    lock_path = str(ffi_list_path) + ".lock"
    with FileLock(lock_path):
        existing = load_ffi_list(ffi_list_path)
        new_df = pd.DataFrame(rows).set_index("filename")
        combined = pd.concat([existing, new_df]) if not existing.empty else new_df
        combined = combined[~combined.index.duplicated(keep="last")]
        _write_ffi_list_artifacts(ffi_list_path, combined)


def ensure_scc_ffi_list(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    paths: Iterable[str | Path],
    *,
    open_fits: Callable,
) -> pd.DataFrame:
    """
    Return ``ffi_list`` covering every logical key in ``paths``, backfilling misses.
    """
    path_list = [Path(p) for p in paths]
    ffi_list_path = ffi_list_parquet_path(data_root, sector, camera, ccd)
    existing = load_ffi_list(ffi_list_path)
    have = set(existing.index.astype(str)) if not existing.empty else set()
    missing = [
        p for p in path_list
        if manifest_basename_from_local(p) not in have
    ]
    if missing:
        new_rows = [extract_ffi_header_record(p, open_fits=open_fits) for p in missing]
        upsert_ffi_list_rows(ffi_list_path, new_rows)
        existing = load_ffi_list(ffi_list_path)
        log.info(
            "ffi_list: extracted %d/%d missing entries; list now has %d rows (%s)",
            len(new_rows),
            len(missing),
            len(existing),
            ffi_list_path,
        )
    return existing


def rebuild_scc_ffi_list(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    paths: Iterable[str | Path],
    *,
    open_fits: Callable,
) -> pd.DataFrame:
    """Force re-extract every local FFI into ``ffi_list`` (cold backfill)."""
    path_list = [Path(p) for p in paths]
    ffi_list_path = ffi_list_parquet_path(data_root, sector, camera, ccd)
    rows = [extract_ffi_header_record(p, open_fits=open_fits) for p in path_list]
    upsert_ffi_list_rows(ffi_list_path, rows)
    df = load_ffi_list(ffi_list_path)
    log.info("ffi_list: rebuilt %d rows at %s", len(df), ffi_list_path)
    return df


def median_crval_from_cache(
    ffi_list_df: pd.DataFrame,
    paths: Sequence[str | Path],
) -> tuple[float, float]:
    """Median CRVAL across ``wcs_ok`` rows for the given local paths."""
    import numpy as np

    rvals: list[float] = []
    dvals: list[float] = []
    for p in paths:
        key = manifest_basename_from_local(p)
        if key not in ffi_list_df.index:
            continue
        row = ffi_list_df.loc[key]
        if not bool(row.get("wcs_ok", False)):
            continue
        try:
            hdr = header_from_cached_row(row)
            rvals.append(float(hdr["CRVAL1"]))
            dvals.append(float(hdr["CRVAL2"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not rvals:
        raise RuntimeError("No usable WCS headers for SCC median-CRVAL anchor")
    return float(np.median(rvals)), float(np.median(dvals))


def header_from_cached_row(row: "pd.Series") -> fits.Header:
    """Deserialize HDU1 header from one ``ffi_list`` row (no file I/O)."""
    cards = row.get("header_cards")
    if cards is not None and not (isinstance(cards, float) and pd.isna(cards)):
        if isinstance(cards, memoryview):
            cards = bytes(cards)
        if isinstance(cards, str):
            cards = cards.encode("latin1")
        return fits.Header.fromstring(cards)
    return fits.Header()


def wcs_from_cached_row(row: "pd.Series") -> WCS:
    """Reconstruct an astropy ``WCS`` from one ``ffi_list`` row."""
    hdr = header_from_cached_row(row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(hdr)
