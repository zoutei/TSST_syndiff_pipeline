"""Shared helpers for tests of the v2 (seam-fix) combined/convolved stores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing import gaia_projection_catalog as gpc


def write_projection_catalog(data_root, projection, n: int = 3) -> Path:
    """Write a tiny per-projection Gaia catalogue (parquet + meta) for a
    synthetic projection id, bypassing the bundled PS1 table."""
    path = gpc.projection_catalog_path(data_root, projection)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "source_id": pd.array(list(range(1, n + 1)), dtype="Int64"),
        "ra": np.linspace(10.0, 10.1, n),
        "dec": np.linspace(20.0, 20.1, n),
        "phot_g_mean_mag": 12.0,
        "phot_bp_mean_mag": 12.5,
        "phot_rp_mean_mag": 11.5,
    })
    df.to_parquet(path, index=False)
    meta = path.with_name(path.name + ".meta.json")
    meta.write_text(json.dumps({
        "scheme": gpc.GAIA_PROJECTION_SCHEME,
        "projection": gpc.projection_id(projection),
        "content_sha256": gpc.content_sha256(df),
    }))
    return path
