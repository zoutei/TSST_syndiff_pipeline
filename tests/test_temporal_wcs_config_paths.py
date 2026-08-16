"""Configuration and SCC-path contracts for the temporal-WCS stage."""

from pathlib import Path

import pytest

from syndiff_pipeline.common.scc_paths import (
    scc_per_ffi_wcs_dir,
    scc_temporal_wcs_dir,
    scc_wcs_debug_dir,
    scc_wcs_dir,
)
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    parse_temporal_wcs,
    validate_stage_for_kind,
)
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline


def _stage(**overrides):
    stage = {
        "kind": "temporal_wcs",
        "inputs": {"centroids": "centroids_r1", "diffs": "hp_d"},
        "output": "temporal_wcs",
    }
    stage.update(overrides)
    return stage


def test_temporal_wcs_defaults_are_production_cheb5_bspline():
    p = parse_temporal_wcs(_stage(), 0)
    assert p.version == "temporal_cheb5_bspline_v1"
    assert p.spatial_basis == "chebyshev"
    assert p.cheb_degree == 5
    assert p.temporal_basis == "bspline"
    assert p.spline_degree == 3


@pytest.mark.parametrize(
    "key,value",
    [("version", "other"), ("spatial_basis", "monomial"), ("cheb_degree", 4), ("temporal_basis", "linear"), ("spline_degree", 2)],
)
def test_temporal_wcs_rejects_nonproduction_model(key, value):
    with pytest.raises(ValueError):
        parse_temporal_wcs(_stage(**{key: value}), 0)


def test_temporal_wcs_requires_centroids_and_diffs():
    with pytest.raises(ValueError, match="inputs.centroids"):
        validate_pipeline(
            SynDiffConfig(
                pipeline=[_stage(inputs={"diffs": "hp_d"})],
                pipeline_external_workspace_labels=["hp_d"],
            )
        )


def test_temporal_wcs_validates_workspace_labels():
    cfg = SynDiffConfig(
        pipeline=[
            {"kind": "shared_mask"},
            {"kind": "temporal_wcs", "inputs": {"centroids": "missing", "diffs": "hp_d"}, "output": "tv"},
        ]
    )
    with pytest.raises(ValueError, match="missing"):
        validate_pipeline(cfg)


def test_wcs_artifacts_and_debug_are_scc_rooted_and_disjoint(tmp_path: Path):
    root = tmp_path / "data"
    wcs = scc_wcs_dir(root, 20, 3, 3)
    assert wcs == root / "s0020" / "c3" / "k3" / "wcs"
    assert scc_per_ffi_wcs_dir(root, 20, 3, 3) == wcs / "per_ffi_cheb5"
    temporal = scc_temporal_wcs_dir(root, 20, 3, 3)
    debug = scc_wcs_debug_dir(root, 20, 3, 3)
    assert temporal == wcs / "temporal_cheb5_bspline_v1"
    assert debug == root / "s0020" / "c3" / "k3" / "debug_plots" / "wcs_temporal_cheb5_bspline_v1"
    assert wcs not in debug.parents and debug not in wcs.parents


def test_wcs_version_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError):
        scc_wcs_dir(tmp_path, 20, 3, 3, version="../debug_plots")
