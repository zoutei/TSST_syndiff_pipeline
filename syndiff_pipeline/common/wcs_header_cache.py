"""
wcs_header_cache.py
====================
Shared, SCC-scoped cache of per-FFI WCS header keywords.

Opening a TESS FFI (``.fits.fz`` / ``.fits.gz``, tens of MB) just to read its
WCS header is CPU-bound on decompression, not I/O latency (an OS-page-cache-warm
re-open of the same file is no faster than a cold one) -- roughly
130-160 ms/file regardless of how few bytes are actually needed. The ~80
WCS-relevant keywords (CRVAL/CRPIX/CD/CTYPE/CUNIT + SIP distortion terms)
are the only thing any consumer actually needs, and reconstructing a
``WCS`` purely from those cached keywords is numerically identical to
building it from the file's full header (validated on real TESS FFIs:
``world_to_pixel_values`` discrepancy ~1e-9 px) at ~9 ms/frame with zero
file I/O once cached.

The cache is SCC-shared (``data_root``-scoped, not per-event), matching the
existing storage convention used by ``skycell_pixel_mapping/``,
``catalogs/``, and ``skycell_remaps/`` (see ``docs/markdown/storage_layout.md``)
-- so multiple targets on the same sector/camera/ccd, and repeated passes
within one event's own ``wcs_grouping`` run, never re-pay the decompression
cost for the same physical file.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from filelock import FileLock

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


from syndiff_pipeline.common.scc_paths import scc_wcs_cache_csv, scc_wcs_cache_parquet


def wcs_cache_path(data_root: str | Path, sector: int, camera: int, ccd: int) -> Path:
    """Path to the shared per-SCC WCS-header-keyword cache parquet."""
    return scc_wcs_cache_parquet(data_root, sector, camera, ccd)


def wcs_cache_csv_path(data_root: str | Path, sector: int, camera: int, ccd: int) -> Path:
    """Path to the CSV twin of the shared WCS cache."""
    return scc_wcs_cache_csv(data_root, sector, camera, ccd)


def _is_wcs_key(key: str) -> bool:
    if not key:
        return False
    return key in WCS_KEYS or key.startswith(SIP_KEY_PREFIXES)


def extract_wcs_keywords(path: str | Path, *, open_fits: Callable) -> Optional[dict]:
    """
    Open one FFI (the expensive, decompression-bound step -- only ever called
    on a genuine cache miss) and pull the WCS-relevant header keywords plus
    a few bookkeeping columns.

    ``open_fits`` is the context-manager-returning callable to use (pass
    ``wcs_grouping.open_fits_memmap`` so gz/memmap resolution isn't
    duplicated here). Returns ``None`` if the file can't be opened or is
    missing the minimum WCS keys.
    """
    try:
        with open_fits(path) as hdul:
            hdr = hdul[1].header
            row = {"filename": Path(str(path)).name}
            for key in hdr.keys():
                if _is_wcs_key(key):
                    row[key] = hdr[key]
            row["DATE-OBS"] = hdr.get("DATE-OBS", None)
            row["NAXIS1"] = hdr.get("NAXIS1", None)
            row["NAXIS2"] = hdr.get("NAXIS2", None)
    except Exception as exc:
        log.warning("wcs_header_cache: could not extract WCS keywords from %s: %s", path, exc)
        return None
    if any(k not in row for k in _MIN_KEYS):
        return None
    return row


def load_wcs_cache(cache_path: str | Path) -> pd.DataFrame:
    """Load the cache parquet, or an empty frame (indexed by filename) if absent."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return pd.DataFrame(index=pd.Index([], name="filename"))
    return pd.read_parquet(cache_path).set_index("filename")


def load_or_build_wcs_cache(
    paths: Iterable[str | Path],
    cache_path: str | Path,
    *,
    open_fits: Callable,
) -> pd.DataFrame:
    """
    Return a WCS-keyword table covering every filename in ``paths``,
    extracting (and persisting) only the entries missing from ``cache_path``.

    Concurrent-safe: the read-modify-append-write cycle is guarded by
    ``FileLock(str(cache_path) + ".lock")`` (the same discipline
    ``skycell_remap.py`` already uses for ``remap_index.parquet``), so
    multiple events' ``wcs_grouping`` runs on the same SCC can safely
    populate the cache together.
    """
    path_list = [Path(p) for p in paths]
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(cache_path) + ".lock"):
        existing = load_wcs_cache(cache_path)
        have = set(existing.index)
        missing = [p for p in path_list if p.name not in have]

        if missing:
            new_rows = [extract_wcs_keywords(p, open_fits=open_fits) for p in missing]
            new_rows = [r for r in new_rows if r is not None]
            if new_rows:
                new_df = pd.DataFrame(new_rows).set_index("filename")
                combined = pd.concat([existing, new_df]) if not existing.empty else new_df
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.reset_index().to_parquet(cache_path, index=False)
                csv_path = cache_path.with_name("wcs_cache.csv")
                combined.reset_index().to_csv(csv_path, index=False)
                existing = combined
            log.info(
                "wcs_header_cache: extracted %d/%d missing entries; cache now has %d rows (%s)",
                len(new_rows), len(missing), len(existing), cache_path,
            )

    return existing


def header_from_cached_row(row: "pd.Series") -> fits.Header:
    """Reconstruct a minimal ``fits.Header`` from one cached row (no file I/O)."""
    hdr = fits.Header()
    for key, value in row.items():
        if key == "DATE-OBS" or pd.isna(value):
            continue
        hdr[key] = value
    return hdr


def wcs_from_cached_row(row: "pd.Series") -> WCS:
    """
    Reconstruct an astropy ``WCS`` from one cached row, no file I/O.

    Numerically identical to building the WCS from the file's full header
    (validated on real TESS FFIs: ``world_to_pixel_values`` discrepancy
    ~1e-9 px).
    """
    hdr = header_from_cached_row(row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(hdr)
