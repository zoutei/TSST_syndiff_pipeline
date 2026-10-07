"""Tests for the Gaia->PS1 best-neighbour magnitude cache (no network; fetch injected)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing import gaia_projection_catalog as gpc

BIG = 4_295_000_000_000_000_001  # > 2**53: breaks if routed through float


def _fetch_factory(calls, matched):
    def fetch(ids):
        calls.append(list(ids))
        rows = [i for i in ids if i in matched]
        return pd.DataFrame({
            "source_id": np.array(rows, dtype="int64"),
            "ps1_obj_id": np.array([r + 1 for r in rows], dtype="int64"),
            "angular_distance": 0.1, "number_of_neighbours": 1, "number_of_mates": 0,
            "r_mean_psf_mag": 15.0, "obj_info_flag": 3, "quality_flag": 1,
        })
    return fetch


def test_chunking_cache_unmatched_and_int64(tmp_path):
    ids = [BIG + k for k in range(1203)]
    matched = set(ids[::2])
    calls = []
    f = _fetch_factory(calls, matched)
    out = gpc.ensure_gaia_ps1_mags(tmp_path, "skycell.2528.001", ids, fetch=f)
    assert [len(c) for c in calls] == [500, 500, 203]
    assert len(out) == 1203
    assert out["source_id"].astype("int64").tolist() == sorted(ids)
    assert int(out["ps1_match"].sum()) == len(matched)
    row = out[out["source_id"] == ids[1]].iloc[0]
    assert not row["ps1_match"] and pd.isna(row["r_mean_psf_mag"]) and pd.isna(row["ps1_obj_id"])
    assert out[out["source_id"] == ids[0]].iloc[0]["ps1_obj_id"] == ids[0] + 1

    path = gpc.ps1_mags_path(tmp_path, 2528)
    assert path.name == "proj_2528.parquet" and "gaia_ps1_best_neighbour/v1" in str(path)
    meta = json.loads(path.with_name(path.name + ".meta.json").read_text())
    assert meta["n_rows"] == 1203 and meta["service_url"] == gpc.GAIA_TAP_URL
    assert "{ids}" in meta["query_template"] and "timestamp_utc" in meta

    # Fully cached request, including non-matches: no new fetch.
    n = len(calls)
    again = gpc.ensure_gaia_ps1_mags(tmp_path, 2528, ids[:10], fetch=f)
    assert len(calls) == n and len(again) == 10

    # Partly new IDs: only the new ones are fetched.
    new = [BIG + 5000, BIG + 5001]
    out2 = gpc.ensure_gaia_ps1_mags(tmp_path, 2528, ids[:3] + new, fetch=f)
    assert calls[-1] == new and len(out2) == 5
    meta = json.loads(path.with_name(path.name + ".meta.json").read_text())
    assert meta["n_rows"] == 1205


def test_query_template_columns():
    q = gpc.PS1_MAGS_QUERY_TEMPLATE
    for frag in ("original_ext_source_id AS ps1_obj_id", "b.angular_distance", "b.number_of_mates",
                 "p.g_mean_psf_mag_error", "p.y_mean_psf_mag", "p.obj_info_flag", "p.quality_flag",
                 "WHERE b.source_id IN ({ids})"):
        assert frag in q
