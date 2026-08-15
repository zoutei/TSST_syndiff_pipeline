"""Tests for centroids LC bucket builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from syndiff_pipeline.difference_imaging.stages.centroids_lc_buckets import (
    LaneContext,
    _slim_frame_table,
    parse_lane_path,
    source_id_bucket,
)


def test_parse_lane_path(tmp_path):
    lane = tmp_path / "data" / "s0020" / "c3" / "k3" / "diff_linear"
    lane.mkdir(parents=True)
    data_root, sector, camera, ccd, name = parse_lane_path(lane)
    assert data_root == tmp_path / "data"
    assert (sector, camera, ccd, name) == (20, 3, 3, "diff_linear")


def test_slim_frame_table():
    merged = pd.DataFrame(
        {
            "source_id": [101, 102],
            "flux_fit": [1.0, 2.0],
            "flux_err": [0.1, 0.2],
            "x_fit": [10.0, 11.0],
            "y_fit": [20.0, 21.0],
            "x_err": [0.01, 0.02],
            "y_err": [0.03, 0.04],
            "flags": [0, 0],
            "qfit": [0.05, 0.06],
        }
    )
    out = _slim_frame_table(merged, stem="tess_test", btjd=100.0)
    assert list(out.columns) == [
        "source_id",
        "btjd",
        "ffi_stem",
        "flux_fit",
        "flux_err",
        "x_fit",
        "y_fit",
        "x_err",
        "y_err",
        "flags",
        "qfit",
    ]
    assert len(out) == 2
    assert out.iloc[0]["ffi_stem"] == "tess_test"


def test_source_id_bucket_spreads_gaia_ids():
    import numpy as np

    # Gaia DR3 source_id values are multiples of 64; low-bit mod is degenerate.
    sid = np.array([1674962276786878848, 1675030824464170880], dtype=np.uint64)
    buckets = {source_id_bucket(int(x), 64) for x in sid}
    assert len(buckets) == 2


def test_lane_context_output_dir():
    ctx = LaneContext(
        lane_root=Path("/lane"),
        data_root=Path("/data"),
        sector=20,
        camera=3,
        ccd=3,
        diff_lane_name="diff_linear",
        centroids_label="centroids_r1",
        hp_d_label="hp_d",
        n_buckets=64,
    )
    assert ctx.output_dir == Path("/lane/centroids_r1_lc")
