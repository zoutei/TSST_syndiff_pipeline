from pathlib import Path

import pytest

from syndiff_pipeline.common.scc_paths import (
    scc_debug_plots_dir,
    scc_mapping_dir,
    scc_mapping_master_pixels2skycells,
    scc_remap_dir,
    scc_templates_dir,
)
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    MappingStageParams,
    RemapStageParams,
    parse_stage_params,
)


def test_tvwcs_lane_paths_are_disjoint_from_linear(tmp_path):
    linear = scc_mapping_dir(tmp_path, 20, 3, 3, oversampling_factor=1)
    tv = scc_mapping_dir(tmp_path, 20, 3, 3, oversampling_factor=4, store_name="tvwcs")
    assert linear == Path(tmp_path) / "s0020/c3/k3/mapping/oversampling_1"
    assert tv == Path(tmp_path) / "s0020/c3/k3/mapping_tvwcs/oversampling_4"
    assert tv != linear
    assert scc_remap_dir(tmp_path, 20, 3, 3, oversampling_factor=4, store_name="tvwcs").parts[-2:] == ("remap_tvwcs", "oversampling_4")
    assert scc_templates_dir(tmp_path, 20, 3, 3, oversampling_factor=4, store_name="tvwcs").parts[-2:] == ("templates_tvwcs", "oversampling_4")
    assert scc_mapping_master_pixels2skycells(tmp_path, 20, 3, 3, oversampling_factor=4, store_name="tvwcs").parent == tv


def test_debug_plot_categories_are_separate(tmp_path):
    root = scc_debug_plots_dir(tmp_path, 20, 3, 3)
    assert scc_debug_plots_dir(tmp_path, 20, 3, 3, "mapping_tvwcs_os4") == root / "mapping_tvwcs_os4"
    assert scc_debug_plots_dir(tmp_path, 20, 3, 3, "remap_tvwcs_os4") != root / "mapping_tvwcs_os4"
    with pytest.raises(ValueError):
        scc_debug_plots_dir(tmp_path, 20, 3, 3, "../mapping")


def test_new_wcs_and_drift_names_are_valid_with_legacy_aliases():
    stages = parse_stage_params({
        "mapping": {"store_name": "tvwcs", "oversampling_factor": 4, "wcs_source": "temporal_wcs"},
        "remap": {"store_name": "tvwcs", "drift_source": "per_skycell_temporal_wcs"},
    })
    assert stages.mapping.store_name == "tvwcs"
    assert stages.mapping.wcs_source == "temporal_wcs"
    assert stages.mapping.temporal_wcs_version == "temporal_cheb5_bspline_v1"
    assert stages.remap.drift_source == "per_skycell_temporal_wcs"
    assert RemapStageParams(drift_source="point").drift_source == "point"


def test_mapping_rejects_unknown_wcs_source():
    with pytest.raises(ValueError, match="wcs_source"):
        MappingStageParams(wcs_source="header")
