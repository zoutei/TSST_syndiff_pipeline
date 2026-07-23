"""Unit tests for verify_scc_diff (SCC-primary diff bookkeeping gate)."""

from __future__ import annotations

import json
from pathlib import Path

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.common.scc_paths import scc_diff_bookkeeping_dir, scc_templates_dir
from syndiff_pipeline.difference_imaging.orchestration.diff_verify import verify_scc_diff
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    DIFF_JOB_BASENAME,
    FIELD_MODE_ASSEMBLY_BASENAME,
    FRAMES_CSV_BASENAME,
)


SECTOR, CAMERA, CCD = 20, 1, 1


def _bookkeeping_dir(
    data_root: Path,
    *,
    oversampling_factor: int = 1,
    template_store_name: str | None = None,
) -> Path:
    return scc_diff_bookkeeping_dir(
        data_root,
        SECTOR,
        CAMERA,
        CCD,
        oversampling_factor=oversampling_factor,
        template_store_name=template_store_name,
    )


def _legacy_bookkeeping_dir(data_root: Path) -> Path:
    return data_root / f"s{SECTOR:04d}" / f"c{CAMERA}" / f"k{CCD}" / "bookkeeping" / "diff"


def _write_ok_tree(
    data_root: Path,
    *,
    job_schema: int = 2,
    sidecar_schema: int = 3,
    include_mapping_grid: bool = True,
    template_store_name: str | None = None,
    oversampling_factor: int = 1,
    legacy_path: bool = False,
) -> MappingGrid:
    grid = MappingGrid.from_ffi_shape(2048, 2048)
    bk = (
        _legacy_bookkeeping_dir(data_root)
        if legacy_path
        else _bookkeeping_dir(
            data_root,
            oversampling_factor=oversampling_factor,
            template_store_name=template_store_name,
        )
    )
    bk.mkdir(parents=True)
    job = {
        "schema_version": job_schema,
        "sector": SECTOR,
        "camera": CAMERA,
        "ccd": CCD,
        "geometry_mode": "field",
    }
    if include_mapping_grid:
        job["mapping_grid"] = grid.to_mapping_dict()
    (bk / DIFF_JOB_BASENAME).write_text(json.dumps(job), encoding="utf-8")
    (bk / FRAMES_CSV_BASENAME).write_text(
        "product_id,path,group_id\ntess1,/tmp/a.fits,0\n",
        encoding="utf-8",
    )
    tmpl = scc_templates_dir(
        data_root,
        SECTOR,
        CAMERA,
        CCD,
        oversampling_factor=oversampling_factor,
        store_name=template_store_name,
    )
    tmpl.mkdir(parents=True)
    sidecar = {
        "schema_version": sidecar_schema,
        "mapping_grid": grid.to_mapping_dict(),
    }
    (tmpl / FIELD_MODE_ASSEMBLY_BASENAME).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return grid


def test_verify_scc_diff_success(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ok_tree(data_root)
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD)
    assert errors == []


def test_verify_scc_diff_success_named_template_lane(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ok_tree(data_root, template_store_name="smoke")
    errors = verify_scc_diff(
        data_root,
        SECTOR,
        CAMERA,
        CCD,
        template_store_name="smoke",
    )
    assert errors == []


def test_verify_scc_diff_missing_bookkeeping(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    # Sidecar only — no diff_job / frames.csv
    grid = MappingGrid.from_ffi_shape(2048, 2048)
    tmpl = scc_templates_dir(data_root, SECTOR, CAMERA, CCD, oversampling_factor=1)
    tmpl.mkdir(parents=True)
    (tmpl / FIELD_MODE_ASSEMBLY_BASENAME).write_text(
        json.dumps({"schema_version": 3, "mapping_grid": grid.to_mapping_dict()}),
        encoding="utf-8",
    )
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD)
    assert any("diff_job.json" in e for e in errors)
    assert any("frames.csv" in e for e in errors)


def test_verify_scc_diff_missing_sidecar(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    bk = _bookkeeping_dir(data_root)
    bk.mkdir(parents=True)
    grid = MappingGrid.from_ffi_shape(2048, 2048)
    (bk / DIFF_JOB_BASENAME).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mapping_grid": grid.to_mapping_dict(),
            }
        ),
        encoding="utf-8",
    )
    (bk / FRAMES_CSV_BASENAME).write_text("product_id\ntess1\n", encoding="utf-8")
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD)
    assert len(errors) == 1
    assert "field_mode_assembly.json" in errors[0]


def test_verify_scc_diff_job_schema_too_old(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ok_tree(data_root, job_schema=1)
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD)
    assert any("schema_version < 2" in e for e in errors)


def test_verify_scc_diff_job_missing_mapping_grid(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ok_tree(data_root, include_mapping_grid=False)
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD)
    assert any("missing mapping_grid" in e for e in errors)


def test_verify_scc_diff_success_os4_lane(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ok_tree(data_root, oversampling_factor=4)
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD, oversampling_factor=4)
    assert errors == []


def test_verify_scc_diff_os4_does_not_see_os1_tree(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ok_tree(data_root, oversampling_factor=1)
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD, oversampling_factor=4)
    assert any("diff_job.json" in e for e in errors)
    assert any("frames.csv" in e for e in errors)


def test_verify_scc_diff_rejects_legacy_flat_bookkeeping_path(tmp_path: Path) -> None:
    """Pre-lane ``bookkeeping/diff/`` (no oversampling leaf) is not read."""
    data_root = tmp_path / "data"
    _write_ok_tree(data_root, legacy_path=True)
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD)
    assert any("diff_job.json" in e for e in errors)
    assert any("frames.csv" in e for e in errors)


def test_verify_scc_diff_sidecar_schema_too_old(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ok_tree(data_root, sidecar_schema=2)
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD)
    assert any("requires schema v3" in e for e in errors)


def test_verify_scc_diff_invalid_job_json(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ok_tree(data_root)
    bk = _bookkeeping_dir(data_root)
    (bk / DIFF_JOB_BASENAME).write_text("{not-json", encoding="utf-8")
    errors = verify_scc_diff(data_root, SECTOR, CAMERA, CCD)
    assert any("invalid JSON" in e for e in errors)
