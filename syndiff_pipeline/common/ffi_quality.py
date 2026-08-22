"""
Per-FFI quality-flag (``DQUALITY``) extraction from the cached ``ffi_list``
header blob.

No ``ffi_list.parquet`` schema migration: ``DQUALITY`` (and any other header
keyword a caller needs) is parsed lazily from the already-stored
``header_cards`` blob (see ``common/wcs_header_cache.py``), so old parquet
files keep working unchanged.
"""

from __future__ import annotations

import pandas as pd

from syndiff_pipeline.common.wcs_header_cache import header_from_cached_row

DQUALITY_KEY = "DQUALITY"


def dquality_for_stem(ffi_list_df: pd.DataFrame, logical_filename: str) -> int:
    """``DQUALITY`` bitmask for one ``ffi_list`` row; ``0`` when absent/unreadable."""
    if ffi_list_df is None or logical_filename not in ffi_list_df.index:
        return 0
    row = ffi_list_df.loc[logical_filename]
    try:
        hdr = header_from_cached_row(row)
    except Exception:
        return 0
    value = hdr.get(DQUALITY_KEY)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def dquality_by_filename(ffi_list_df: pd.DataFrame) -> dict[str, int]:
    """``{logical_filename: DQUALITY}`` for every row in *ffi_list_df*."""
    if ffi_list_df is None or len(ffi_list_df) == 0:
        return {}
    out: dict[str, int] = {}
    for filename in ffi_list_df.index:
        out[str(filename)] = dquality_for_stem(ffi_list_df, filename)
    return out


def quality_ok_mask(dquality: int, bitmask: int) -> bool:
    """``True`` when *dquality* trips none of the disqualifying *bitmask* bits."""
    return (int(dquality) & int(bitmask)) == 0
