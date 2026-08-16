"""WCS-only orchestration must not require the template handoff."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.execute import run_config_pipeline
from syndiff_pipeline.difference_imaging.stages.astrometry import (
    pipeline_needs_template_handoff,
)


def _wcs_only_config(tmp_path: Path) -> SynDiffConfig:
    return SynDiffConfig(
        output_dir=str(tmp_path / "event"),
        data_root=str(tmp_path / "data"),
        sector=20,
        camera=3,
        ccd=3,
        pipeline=[
            {"external_workspaces": ["centroids_r1", "hp_d"]},
            {
                "kind": "temporal_wcs",
                "inputs": {"centroids": "centroids_r1", "diffs": "hp_d"},
                "output": "wcs",
            },
        ],
    )


def test_wcs_only_pipeline_reaches_temporal_stage_without_handoff(tmp_path):
    cfg = _wcs_only_config(tmp_path)
    assert pipeline_needs_template_handoff(cfg.pipeline) is False

    with (
        patch(
            "syndiff_pipeline.difference_imaging.orchestration.execute._load_template_handoff",
            side_effect=AssertionError("template handoff must not be loaded"),
        ),
        patch(
            "syndiff_pipeline.difference_imaging.orchestration.execute._load_gaia_catalog",
            return_value=pd.DataFrame(),
        ),
        patch(
            "syndiff_pipeline.difference_imaging.stages.temporal_wcs.run_temporal_wcs_all_frames",
            return_value=(3, 3),
        ) as run_temporal,
    ):
        run_config_pipeline(cfg)

    run_temporal.assert_called_once()
    lane = Path(cfg.data_root) / "s0020" / "c3" / "k3" / "diff"
    assert (lane / "wcs" / "temporal_wcs_artifact.json").is_file()


def test_mixed_pipeline_still_requires_template_handoff():
    pipeline = [
        {"external_workspaces": ["centroids_r1", "hp_d"]},
        {
            "kind": "temporal_wcs",
            "inputs": {"centroids": "centroids_r1", "diffs": "hp_d"},
            "output": "wcs",
        },
        {"kind": "kernel_fit", "inputs": {"diffs": "hp_d"}, "output": "kernel"},
    ]
    assert pipeline_needs_template_handoff(pipeline) is True
