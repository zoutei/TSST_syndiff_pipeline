"""Mapping verify MAPGRID=2 gate (padded SCC v2 §16)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    ResolvedTargetConfig,
)
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


def _tiny_grid() -> MappingGrid:
    """Compact grid so fixtures stay small and fast."""
    return MappingGrid(
        ffi_xmin=0,
        ffi_ymin=-2,
        ffi_xmax=8,
        ffi_ymax=6,
        oversampling=1,
        conv_pad_native=2,
    )


def _resolved(tmp: Path, *, mapping_root: Path | None = None) -> ResolvedTargetConfig:
    target = Target(
        sector=22,
        camera=3,
        ccd=3,
        target_ra=228.0,
        target_dec=52.0,
        target_name="mapgrid_verify",
    )
    root = mapping_root if mapping_root is not None else tmp / "mapping"
    return ResolvedTargetConfig(
        target=target,
        data_root=str(tmp / "data"),
        ffi_dir=str(tmp / "data" / "ffi"),
        event_dir=str(tmp / "events" / target.label()),
        skycell_wcs_csv=str(tmp / "skycell_wcs.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            remap=RemapStageParams(),
            downsample=DownsampleStageParams(single_offset=True),
        ),
        mapping_root=str(root),
        zarr_dir=str(tmp / "data" / "ps1_skycells_zarr"),
        template_output_base=str(tmp / "templates"),
    )


def _write_csv(mapping_leaf: Path, *, sector: int = 22, camera: int = 3, ccd: int = 3) -> Path:
    mapping_leaf.mkdir(parents=True, exist_ok=True)
    csv_path = (
        mapping_leaf
        / f"tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list.csv"
    )
    csv_path.write_text("NAME,projection\nskycell.0001.0001,0001\n", encoding="utf-8")
    return csv_path


def _write_master(
    mapping_leaf: Path,
    grid: MappingGrid,
    *,
    sector: int = 22,
    camera: int = 3,
    ccd: int = 3,
    mapgrid: int | None = 2,
    data_shape: tuple[int, int] | None = None,
) -> Path:
    mapping_leaf.mkdir(parents=True, exist_ok=True)
    path = (
        mapping_leaf
        / f"tess_s{sector:04d}_{camera}_{ccd}_master_pixels2skycells.fits"
    )
    hdr = fits.Header()
    for key, val in grid.to_fits_header_updates().items():
        hdr[key] = val
    if mapgrid is None:
        del hdr["MAPGRID"]
    else:
        hdr["MAPGRID"] = int(mapgrid)
    shape = data_shape if data_shape is not None else grid.array_shape_os()
    pri = fits.PrimaryHDU()
    img = fits.ImageHDU(data=np.zeros(shape, dtype=np.int16), header=hdr)
    fits.HDUList([pri, img]).writeto(path, overwrite=True)
    return path


def test_verify_mapping_mapgrid_v2_ok(tmp_path: Path) -> None:
    leaf = tmp_path / "mapping"
    grid = _tiny_grid()
    _write_csv(leaf)
    _write_master(leaf, grid)
    result = verify_mapping(_resolved(tmp_path, mapping_root=leaf))
    assert result.ok
    assert "MAPGRID v2" in result.message


def test_verify_mapping_rejects_mapgrid_v1(tmp_path: Path) -> None:
    leaf = tmp_path / "mapping"
    grid = _tiny_grid()
    _write_csv(leaf)
    _write_master(leaf, grid, mapgrid=1)
    result = verify_mapping(_resolved(tmp_path, mapping_root=leaf))
    assert not result.ok
    assert "MAPGRID" in result.message


def test_verify_mapping_rejects_missing_mapgrid_keyword(tmp_path: Path) -> None:
    leaf = tmp_path / "mapping"
    grid = _tiny_grid()
    _write_csv(leaf)
    _write_master(leaf, grid, mapgrid=None)
    result = verify_mapping(_resolved(tmp_path, mapping_root=leaf))
    assert not result.ok
    assert "MAPGRID" in result.message


def test_verify_mapping_rejects_shape_mismatch(tmp_path: Path) -> None:
    leaf = tmp_path / "mapping"
    grid = _tiny_grid()
    _write_csv(leaf)
    _write_master(leaf, grid, data_shape=(3, 3))
    result = verify_mapping(_resolved(tmp_path, mapping_root=leaf))
    assert not result.ok
    assert "shape" in result.message.lower()


def test_verify_mapping_missing_master_fits(tmp_path: Path) -> None:
    leaf = tmp_path / "mapping"
    _write_csv(leaf)
    result = verify_mapping(_resolved(tmp_path, mapping_root=leaf))
    assert not result.ok
    assert "pixels2skycells" in result.message.lower() or "missing" in result.message.lower()


def test_verify_mapping_missing_csv(tmp_path: Path) -> None:
    result = verify_mapping(_resolved(tmp_path))
    assert not result.ok
    assert "CSV" in result.message
