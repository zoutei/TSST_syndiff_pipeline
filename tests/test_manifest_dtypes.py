"""Manifest column dtype handling."""

from __future__ import annotations

import pandas as pd

from syndiff_pipeline.difference_imaging.support.manifest import (
    apply_hotpants_workspace_results,
)


def test_apply_hotpants_workspace_results_coerces_float_error_column():
    df = pd.DataFrame(
        {
            "path": ["/data/tess2020057105921-s0022-3-3-0174-s_ffic.fits"],
            "hotpants_ks_d_ok": [pd.NA],
            "hotpants_ks_d_error": [float("nan")],
            "diff_ks_d_path": [float("nan")],
        }
    )
    results = [
        {
            "success": True,
            "error_msg": "",
            "path": "/event/ws/ks_d/tess2020057105921_ks_d.fits",
        }
    ]
    out = apply_hotpants_workspace_results(
        df,
        ["/data/tess2020057105921-s0022-3-3-0174-s_ffic.fits"],
        results,
        "ks_d",
    )
    assert out.loc[0, "hotpants_ks_d_ok"] is True
    assert out.loc[0, "hotpants_ks_d_error"] == ""
    assert "tess2020057105921_ks_d.fits" in str(out.loc[0, "diff_ks_d_path"])
