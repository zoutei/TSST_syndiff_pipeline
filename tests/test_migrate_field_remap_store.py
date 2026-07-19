"""Tests for legacy field remap store migration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from syndiff_pipeline.common.scc_paths import scc_remap_dir, scc_templates_dir
from syndiff_pipeline.template_creation.processing.field_remap import REMAP_MANIFEST_NAME
from syndiff_pipeline.template_creation.processing.migrate_field_remap_store import (
    migrate_scc_remap_artifacts,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import ShiftSchedule


def _legacy_templates_store(
    data_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
) -> Path:
    store = scc_templates_dir(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    store.mkdir(parents=True)
    (store / "contribs").mkdir()
    (store / "template_manifest.json").write_text("{}")
    (store / "field_mode_assembly.json").write_text(
        json.dumps({"schema_version": 1, "store_root": str(store)})
    )

    schedule = ShiftSchedule(
        skycell_names=np.array(["skycell.1.1"]),
        sx_float=np.zeros((1, 1), dtype=np.float32),
        sy_float=np.zeros((1, 1), dtype=np.float32),
        sx_int=np.zeros((1, 1), dtype=np.int16),
        sy_int=np.zeros((1, 1), dtype=np.int16),
        frame_valid=np.ones(1, dtype=bool),
        meta={"reference_ffi": "/data/ref.fits"},
    )
    schedule.save(store / "shift_schedule.npz")
    (store / "shift_schedule.json").write_text('{"schema_version": 1}')
    pd.DataFrame(
        {
            "group_id": [0],
            "skycell": ["skycell.1.1"],
            "sx_int": [0],
            "sy_int": [0],
            "qx": [0.0],
            "qy": [0.0],
            "cache_key": ["x"],
        }
    ).to_parquet(store / "template_group_shifts.parquet", index=False)
    (store / "template_groups.json").write_text('{"groups": []}')

    cache = store / "exact_cache"
    cache.mkdir()
    (cache / "exact.npz").write_bytes(b"cache-data")
    return store


def test_migrate_creates_dest_layout(tmp_path: Path):
    data_root = tmp_path / "data"
    sector, camera, ccd = 23, 1, 3
    source = _legacy_templates_store(data_root, sector, camera, ccd)
    dest = scc_remap_dir(data_root, sector, camera, ccd, oversampling_factor=1)

    result = migrate_scc_remap_artifacts(data_root, sector, camera, ccd)

    assert dest.is_dir()
    assert (dest / "shift_schedule.npz").is_file()
    assert (dest / "shift_schedule.json").is_file()
    assert (dest / "template_group_shifts.parquet").is_file()
    assert (dest / "template_groups.json").is_file()
    assert (dest / "exact_cache" / "exact.npz").is_file()
    assert result["manifest_written"] is True
    manifest = json.loads((dest / REMAP_MANIFEST_NAME).read_text())
    assert manifest["geometry_mode"] == "field"
    assert manifest["migrated"] is True
    assert manifest["oversampling_factor"] == 1
    assert "written_at" in manifest


def test_migrate_leaves_contribs_in_templates(tmp_path: Path):
    data_root = tmp_path / "data"
    sector, camera, ccd = 23, 1, 3
    source = _legacy_templates_store(data_root, sector, camera, ccd)
    contrib = source / "contribs" / "skycell.1.1_0_0.npz"
    contrib.write_bytes(b"contrib")

    migrate_scc_remap_artifacts(data_root, sector, camera, ccd)

    assert contrib.is_file()
    assert (source / "template_manifest.json").is_file()
    assert (source / "field_mode_assembly.json").is_file()
    assert (source / "shift_schedule.npz").is_file()
    assert not (
        scc_remap_dir(data_root, sector, camera, ccd, oversampling_factor=1) / "contribs"
    ).exists()


def test_migrate_idempotent_second_run(tmp_path: Path):
    data_root = tmp_path / "data"
    sector, camera, ccd = 23, 1, 3
    _legacy_templates_store(data_root, sector, camera, ccd)

    first = migrate_scc_remap_artifacts(data_root, sector, camera, ccd)
    second = migrate_scc_remap_artifacts(data_root, sector, camera, ccd)

    assert first["copied"]
    assert not second["copied"]
    assert second["manifest_written"] is False
    assert set(second["skipped"]) >= {
        "shift_schedule.npz",
        "shift_schedule.json",
        "template_group_shifts.parquet",
        "template_groups.json",
        "exact_cache/",
    }
