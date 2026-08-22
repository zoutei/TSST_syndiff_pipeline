"""Unit tests for DQUALITY extraction from the cached ffi_list header blob."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.ffi_quality import (
    dquality_by_filename,
    dquality_for_stem,
    quality_ok_mask,
)
from syndiff_pipeline.common.wcs_header_cache import _header_cards_bytes


def _row(dquality: int | None) -> dict:
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    if dquality is not None:
        hdr["DQUALITY"] = dquality
    return {"header_cards": _header_cards_bytes(hdr)}


def _make_ffi_list_df(rows: dict[str, int | None]) -> pd.DataFrame:
    records = []
    for filename, dq in rows.items():
        rec = _row(dq)
        rec["filename"] = filename
        records.append(rec)
    return pd.DataFrame(records).set_index("filename")


def test_dquality_for_stem_reads_header_value():
    df = _make_ffi_list_df({"a.fits": 64, "b.fits": None})
    assert dquality_for_stem(df, "a.fits") == 64
    assert dquality_for_stem(df, "b.fits") == 0


def test_dquality_for_stem_missing_row_returns_zero():
    df = _make_ffi_list_df({"a.fits": 1})
    assert dquality_for_stem(df, "not_present.fits") == 0
    assert dquality_for_stem(None, "a.fits") == 0


def test_dquality_by_filename_covers_every_row():
    df = _make_ffi_list_df({"a.fits": 2, "b.fits": 8})
    out = dquality_by_filename(df)
    assert out == {"a.fits": 2, "b.fits": 8}
    assert dquality_by_filename(None) == {}
    assert dquality_by_filename(df.iloc[0:0]) == {}


def test_quality_ok_mask_bit_logic():
    bitmask = 1 | 2 | 4  # attitude tweak | safe mode | coarse point
    assert quality_ok_mask(0, bitmask) is True
    assert quality_ok_mask(8, bitmask) is True  # unrelated bit set
    assert quality_ok_mask(2, bitmask) is False
    assert quality_ok_mask(1 | 8, bitmask) is False
