"""Tests for single-kernel diff pipeline helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    kernel_fit_params_to_hotpants,
    parse_kernel_fit,
)
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.support.min_background import (
    angle_score_series,
    pick_best_angle_ffi,
)


def test_pick_best_angle_ffi():
    manifest = pd.DataFrame(
        {
            "path": ["/a/f1.fits", "/a/f2.fits"],
            "Earth_Camera_Angle": [10.0, 50.0],
            "Moon_Camera_Angle": [20.0, 30.0],
            "wcs_ok": [True, True],
        }
    )
    path, score = pick_best_angle_ffi(manifest, weighting_factor=0.5)
    assert path.endswith("f2.fits")
    scores = angle_score_series(manifest, 0.5)
    assert score == pytest.approx(float(scores.iloc[1]))


def test_parse_kernel_fit_and_hotpants_bridge():
    stage = {
        "kind": "kernel_fit",
        "output": "kernel_fit",
        "weighting_factor": 0.5,
        "phot_box_size": 4,
        "hp_bgo": 3,
    }
    kf = parse_kernel_fit(stage, 0)
    hp = kernel_fit_params_to_hotpants(kf)
    assert hp.hp_bgo == 3
    assert hp.write_stamps is False


def test_validate_single_kernel_pipeline():
    cfg = SynDiffConfig(
        output_dir="/tmp/test_event",
        pipeline=[
            {"kind": "shared_mask"},
            {
                "kind": "kernel_fit",
                "output": "kernel_fit",
                "phot_box_size": 4,
            },
            {
                "kind": "convolved_templates",
                "inputs": {"kernel_fit": "kernel_fit"},
                "output": "tmpl_conv",
            },
            {
                "kind": "kernel_subtract",
                "inputs": {"convolved": "tmpl_conv"},
                "output": {"diffs": "ks_d", "phot_bkg": "ks_b"},
            },
            {
                "kind": "background",
                "inputs": {"bkg_in": "ks_b"},
                "output": "ks_b_s",
                "recombine_inputs": False,
                "steps": {
                    "spatial": {"enabled": False},
                    "temporal": {"enabled": True, "tile_size": 256},
                    "strap": {"enabled": False},
                },
            },
            {
                "kind": "subtract",
                "inputs": {"expression": "ks_d + ks_b - ks_b_s"},
                "output": "ks_d_s",
            },
            {
                "kind": "forced_photometry",
                "inputs": {"diffs": "ks_d_s"},
                "output": "lc_prf",
                "methods": [
                    {"name": "prf", "type": "psf", "psf_type": "prf"},
                ],
            },
        ],
    )
    validate_pipeline(cfg)
