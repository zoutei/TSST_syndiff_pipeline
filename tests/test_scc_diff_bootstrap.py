"""Tests for SCC diff bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.common.mapping_grid import MappingGrid, MappingGridError
from syndiff_pipeline.common.scc_paths import (
    scc_diff_bookkeeping_dir,
    scc_diff_dir,
    scc_remap_dir,
    scc_templates_dir,
)
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    bootstrap_scc_diff,
)
from syndiff_pipeline.template_creation.processing.field_remap import (
    GROUP_ID_PER_FRAME_NPY,
)


@pytest.fixture
def bootstrap_tree(tmp_path):
    data_root = tmp_path / "data"
    sector, camera, ccd = 20, 1, 1
    grid = MappingGrid.from_ffi_shape(2048, 2048)
    os_factor = 1

    tmpl = scc_templates_dir(data_root, sector, camera, ccd, oversampling_factor=os_factor)
    tmpl.mkdir(parents=True)
    sidecar = {
        "schema_version": 3,
        "mapping_grid": grid.to_mapping_dict(),
        "base_tess_shape": list(grid.array_shape_native()),
        "oversampling_factor": os_factor,
    }
    (tmpl / "field_mode_assembly.json").write_text(json.dumps(sidecar))

    remap = scc_remap_dir(data_root, sector, camera, ccd, oversampling_factor=os_factor)
    remap.mkdir(parents=True)
    np.save(remap / GROUP_ID_PER_FRAME_NPY, np.array([0, 1, 0], dtype=np.int32))

    ffi_dir = data_root / "s0020" / "c1" / "k1" / "ffi"
    ffi_dir.mkdir(parents=True)
    names = [
        "tess2020019142923-s0020-1-1-0165-s_ffic.fits",
        "tess2020019142924-s0020-1-1-0165-s_ffic.fits",
        "tess2020019142925-s0020-1-1-0165-s_ffic.fits",
    ]
    for name in names:
        (ffi_dir / name).write_bytes(b"")

    return data_root, sector, camera, ccd, grid


def test_bootstrap_writes_bookkeeping(bootstrap_tree):
    data_root, sector, camera, ccd, grid = bootstrap_tree
    result = bootstrap_scc_diff(
        data_root=data_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
        template_store_name=None,
        output_store_name="smoke",
        remap_store_name=None,
    )
    assert result.crop_bounds == grid.science_ffi_bounds()
    assert result.frames_csv_path.is_file()
    assert result.diff_job_path.is_file()
    assert len(result.frames_df) == 3
    assert result.diff_store_root == scc_diff_dir(
        data_root, sector, camera, ccd, store_name="smoke"
    )
    assert result.bookkeeping_dir == scc_diff_bookkeeping_dir(
        data_root, sector, camera, ccd
    )


def test_bootstrap_rejects_v2_sidecar(bootstrap_tree):
    data_root, sector, camera, ccd, _ = bootstrap_tree
    tmpl = scc_templates_dir(data_root, sector, camera, ccd, oversampling_factor=1)
    (tmpl / "field_mode_assembly.json").write_text(
        json.dumps({"schema_version": 2, "roi_bounds": [0, 0, 10, 10]})
    )
    with pytest.raises(MappingGridError, match="schema_version"):
        bootstrap_scc_diff(
            data_root=data_root,
            sector=sector,
            camera=camera,
            ccd=ccd,
            template_store_name=None,
            output_store_name=None,
            remap_store_name=None,
        )
