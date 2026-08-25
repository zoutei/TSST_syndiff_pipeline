"""Regression coverage for SCC Gaia-catalog selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from syndiff_pipeline.common.scc_paths import scc_diff_dir
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.execute import _load_gaia_catalog


def test_explicit_gaia_catalog_beats_stale_lane_pipeline_cache(tmp_path: Path):
    """A fresh source must not be shadowed by an old crop-local catalog."""
    data_root = tmp_path / "data"
    lane = scc_diff_dir(data_root, 52, 2, 1, store_name="linear")
    lane.mkdir(parents=True)
    stale = lane / "gaia_catalog_pipeline.csv"
    fresh = tmp_path / "gaia_catalog_s0052_2_1.csv"
    pd.DataFrame({"source_id": [1], "ra": [1.0]}).to_csv(stale, index=False)
    pd.DataFrame({"source_id": [2], "ra": [2.0]}).to_csv(fresh, index=False)

    cfg = SynDiffConfig(
        data_root=str(data_root),
        sector=52,
        camera=2,
        ccd=1,
        output_store_name="linear",
        gaia_catalog=str(fresh),
    )

    loaded = _load_gaia_catalog(cfg)
    assert loaded is not None
    assert loaded["source_id"].tolist() == [2]
