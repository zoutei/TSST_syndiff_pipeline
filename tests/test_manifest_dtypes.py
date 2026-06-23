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
            "hotpants_kd_d_ok": [pd.NA],
            "hotpants_kd_d_error": [float("nan")],
            "diff_kd_d_path": [float("nan")],
        }
    )
    results = [
        {
            "success": True,
            "error_msg": "",
            "path": "/event/ws/kd_d/tess2020057105921_kd_d.fits",
        }
    ]
    out = apply_hotpants_workspace_results(
        df,
        ["/data/tess2020057105921-s0022-3-3-0174-s_ffic.fits"],
        results,
        "kd_d",
    )
    assert out.loc[0, "hotpants_kd_d_ok"] is True
    assert out.loc[0, "hotpants_kd_d_error"] == ""
    assert "tess2020057105921_kd_d.fits" in str(out.loc[0, "diff_kd_d_path"])
