"""Unit tests for field remap store layout and path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from unittest.mock import MagicMock

from syndiff_pipeline.template_creation.processing.field_remap import (
    EXACT_CACHE_L4A_DIRNAME,
    REMAP_MANIFEST_NAME,
    exact_cache_dir_for_read_root,
    exact_cache_l4a_dir_for_read_root,
    remap_root,
    resolve_remap_read_root,
    run_field_remap_scc,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import (
    ShiftSchedule,
    assign_groups_from_schedule,
)


def _minimal_schedule(n_frames: int = 2, n_skycells: int = 1) -> ShiftSchedule:
    return ShiftSchedule(
        skycell_names=np.array([f"skycell.{i}.{i}" for i in range(n_skycells)]),
        sx_float=np.zeros((n_frames, n_skycells), dtype=np.float32),
        sy_float=np.zeros((n_frames, n_skycells), dtype=np.float32),
        sx_int=np.zeros((n_frames, n_skycells), dtype=np.int16),
        sy_int=np.zeros((n_frames, n_skycells), dtype=np.int16),
        frame_valid=np.ones(n_frames, dtype=bool),
        meta={"reference_ffi": "/data/ref.fits"},
    )


def test_exact_cache_l4a_dir_for_read_root(tmp_path: Path):
    read_root = tmp_path / "remap" / "oversampling_1"
    read_root.mkdir(parents=True)
    l4a_dir = exact_cache_l4a_dir_for_read_root(read_root)
    assert l4a_dir == read_root / EXACT_CACHE_L4A_DIRNAME
    assert exact_cache_dir_for_read_root(read_root) == l4a_dir


def test_resolve_remap_read_root_prefers_remap_manifest(tmp_path: Path):
    remap = tmp_path / "remap" / "oversampling_1"
    templates = tmp_path / "templates" / "oversampling_1"
    remap.mkdir(parents=True)
    templates.mkdir(parents=True)
    (remap / REMAP_MANIFEST_NAME).write_text("{}")
    (templates / "shift_schedule.npz").write_bytes(b"legacy")

    read_root, legacy = resolve_remap_read_root(remap, templates)
    assert read_root == remap
    assert legacy is False


def test_resolve_remap_read_root_legacy_fallback(tmp_path: Path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    remap = tmp_path / "remap" / "oversampling_1"
    templates = tmp_path / "templates" / "oversampling_1"
    remap.mkdir(parents=True)
    templates.mkdir(parents=True)
    sched = _minimal_schedule()
    sched.save(templates / "shift_schedule.npz")

    read_root, legacy = resolve_remap_read_root(remap, templates)
    assert read_root == templates
    assert legacy is True
    assert "legacy colocated templates store" in caplog.text


def test_run_field_remap_scc_writes_layout(tmp_path: Path, monkeypatch):
    """Remap stage writes manifest, schedule, groups under remap/."""
    data_root = tmp_path / "data"
    sector, camera, ccd = 1, 1, 1
    store = remap_root(data_root, sector, camera, ccd, oversampling_factor=1)
    event_dir = tmp_path / "event"
    event_dir.mkdir()
    (event_dir / "frames.csv").write_text(
        "path\n/data/f0.fits\n/data/f1.fits\n/data/f2.fits\n"
    )

    schedule = _minimal_schedule(n_frames=3, n_skycells=2)
    schedule.meta["frame_filenames"] = ["f0.fits", "f1.fits", "f2.fits"]
    assignment = assign_groups_from_schedule(
        schedule,
        grouping_quantum_ps1_px=1.0,
        cache_quantum_ps1_px=1.0,
        keying="absolute",
    )

    def fake_ensure(**kwargs):
        store_root = kwargs["store_root"]
        schedule.save(store_root / "shift_schedule.npz")
        return schedule

    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._ensure_shift_schedule",
        fake_ensure,
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap.assign_groups_from_schedule",
        lambda *a, **k: assignment,
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._master_skycell_id_map",
        lambda _path: (np.zeros((2, 2), dtype=np.int32), {}),
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._master_pixels2skycells_path",
        lambda *a, **k: tmp_path / "master.fits",
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_abutting.abutting_undirected_pairs",
        lambda master: np.zeros((0, 2), dtype=np.int32),
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._load_frame_wcs",
        lambda _df, _i: MagicMock(name="tess_wcs"),
    )

    result = run_field_remap_scc(
        sector=sector,
        camera=camera,
        ccd=ccd,
        data_root=data_root,
        event_dir=event_dir,
        mapping_root=tmp_path / "mapping",
        base_tess_shape=(10, 12),
        oversampling_factor=1,
        scc_only=False,
        store_root=store,
    )

    assert result["output_dir"] == str(store)
    assert (store / REMAP_MANIFEST_NAME).is_file()
    assert (store / "shift_schedule.npz").is_file()
    assert (store / "template_group_shifts.parquet").is_file()
    assert (store / "template_groups.json").is_file()
    manifest = json.loads((store / REMAP_MANIFEST_NAME).read_text())
    assert manifest["geometry_mode"] == "field"
    assert manifest["cache_quantum_ps1_px"] == 1.0
    assert manifest["keying"] == "absolute"
    assert manifest["n_groups"] == len(assignment.groups)
    assert manifest["reference_ffi"] == "/data/ref.fits"
    assert manifest["exact_cache_l4a"] == EXACT_CACHE_L4A_DIRNAME
    assert manifest["exact_cache_l4b"] == "exact_cache_l4b"
    assert (store / EXACT_CACHE_L4A_DIRNAME).is_dir()
    assert (store / "exact_cache_l4b").is_dir()


def test_remap_root_matches_scc_paths(tmp_path: Path):
    path = remap_root(tmp_path, 15, 2, 3, oversampling_factor=1)
    assert path == tmp_path / "s0015" / "c2" / "k3" / "remap" / "oversampling_1"
