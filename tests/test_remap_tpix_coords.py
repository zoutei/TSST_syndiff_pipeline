"""Remap tpix + mapping verify gates require MAPGRID>=2 (no legacy fallback)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from syndiff_pipeline.common.mapping_grid import MappingGrid, MappingGridError
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    RemapStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
)
from syndiff_pipeline.template_creation.orchestration.verify import verify_mapping
from syndiff_pipeline.template_creation.processing.field_remap import _build_remap_tpix


@pytest.fixture
def mapgrid_master_fits(tmp_path):
    grid = MappingGrid.from_ffi_shape(2048, 2048)
    path = tmp_path / "master_pixels2skycells.fits.fz"
    data = np.zeros(grid.array_shape_native(), dtype=np.int32)
    hdu = fits.PrimaryHDU(data=data)
    for key, val in grid.to_fits_header_updates().items():
        hdu.header[key] = val
    hdu.writeto(path, overwrite=True)
    return path, grid


def _write_mapgrid_master(path, grid: MappingGrid | None = None, *, mapgrid: int | None = 2):
    grid = grid or MappingGrid.from_ffi_shape(256, 256, conv_pad_native=2)
    data = np.zeros(grid.array_shape_native(), dtype=np.int16)
    hdu = fits.PrimaryHDU(data=data)
    for key, val in grid.to_fits_header_updates().items():
        hdu.header[key] = val
    if mapgrid is None:
        del hdu.header["MAPGRID"]
    else:
        hdu.header["MAPGRID"] = int(mapgrid)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return grid


def _resolved_for_mapping(tmp_path, *, sector=22, camera=3, ccd=3) -> ResolvedTargetConfig:
    target = Target(sector, camera, ccd, 228.0, 52.0, "2020dgc")
    mapping_root = tmp_path / "mapping"
    return ResolvedTargetConfig(
        target=target,
        data_root=str(tmp_path / "data"),
        ffi_dir=str(tmp_path / "data" / "tess_ffi"),
        event_dir=str(tmp_path / "events" / target.label()),
        skycell_wcs_csv=str(tmp_path / "skycell_wcs.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            remap=RemapStageParams(),
            downsample=DownsampleStageParams(single_offset=True),
        ),
        mapping_root=str(mapping_root),
        zarr_dir=str(tmp_path / "data" / "ps1_skycells_zarr"),
        template_output_base=str(tmp_path / "shifted_downsampled"),
    )


def test_build_remap_tpix_uses_ffi_coords(mapgrid_master_fits):
    master_path, grid = mapgrid_master_fits
    tpix = _build_remap_tpix(
        master_path=master_path,
        base_tess_shape=grid.array_shape_native(),
        oversampling_factor=1,
    )
    pad_idx = (grid.conv_pad_native - 1) * grid.width_native
    assert float(tpix[pad_idx, 0]) < 0
    science_idx = grid.conv_pad_native * grid.width_native
    assert int(round(float(tpix[science_idx, 1]))) == grid.ffi_xmin
    assert int(round(float(tpix[science_idx, 0]))) == 0


def test_build_remap_tpix_rejects_legacy_master(tmp_path):
    path = tmp_path / "legacy.fits"
    data = np.zeros((100, 80), dtype=np.int32)
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
    with pytest.raises(MappingGridError):
        _build_remap_tpix(
            master_path=path,
            base_tess_shape=(100, 80),
            oversampling_factor=1,
        )


def test_build_remap_tpix_rejects_missing_master_path():
    with pytest.raises(MappingGridError, match="MAPGRID"):
        _build_remap_tpix(
            master_path=None,
            base_tess_shape=(100, 80),
            oversampling_factor=1,
        )


def test_exact_regmap_requires_tpix_coord_input():
    """Hybrid Exact must not fall back to create_tess_pixel_coordinates."""
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        exact_regmap_for_tess_ids,
    )

    with pytest.raises(ValueError, match="tpix_coord_input is required"):
        exact_regmap_for_tess_ids(
            tess_wcs=None,  # type: ignore[arg-type]
            skycell_row={"NAME": "skycell.0001.0001"},
            tess_ids=np.array([0], dtype=np.int32),
            data_shape=(10, 10),
            tpix_coord_input=None,  # type: ignore[arg-type]
        )


def test_verify_mapping_requires_mapgrid_v3(tmp_path):
    resolved = _resolved_for_mapping(tmp_path)
    scc = (
        tmp_path
        / "mapping"
        / "sector_0022"
        / "camera_3"
        / "ccd_3"
    )
    csv_path = scc / "tess_s0022_3_3_master_skycells_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("NAME,projection\nskycell.0001.0001,0001\n", encoding="utf-8")

    missing = verify_mapping(resolved)
    assert not missing.ok
    assert "missing" in missing.message.lower() or "FITS" in missing.message

    master = scc / "tess_s0022_3_3_master_pixels2skycells.fits"
    _write_mapgrid_master(master, mapgrid=None)
    legacy = verify_mapping(resolved)
    assert not legacy.ok
    assert "MAPGRID" in legacy.message

    grid = _write_mapgrid_master(master, mapgrid=3)
    ok = verify_mapping(resolved)
    assert ok.ok
    assert "MAPGRID=3" in ok.message
    assert str(grid.array_shape_os()) in ok.message or "shape=" in ok.message
