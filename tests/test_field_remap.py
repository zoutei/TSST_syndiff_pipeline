"""Unit tests for field remap store layout and path resolution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.template_creation.processing.field_remap import (
    EXACT_CACHE_L4A_DIRNAME,
    REMAP_MANIFEST_NAME,
    _l4a_exact_skycell_batch,
    _reset_remap_worker,
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


def test_l4a_exact_skycell_batch_loads_regmap_once(monkeypatch, tmp_path: Path):
    import syndiff_pipeline.template_creation.processing.field_remap as fr

    calls: list[str] = []

    def fake_read(skycell: str) -> np.ndarray:
        calls.append(skycell)
        return np.zeros((4, 4), dtype=np.int32)

    def fake_one_shift(
        skycell: str,
        epoch_id: int,
        sx_i: int,
        sy_i: int,
        rep_frame_index: int,
        assignment_map: np.ndarray,
    ) -> str:
        assert assignment_map.shape == (4, 4)
        return f"{skycell}:e{epoch_id}:{sx_i},{sy_i}:r{rep_frame_index}"

    monkeypatch.setattr(fr, "_read_regmap_assignment", fake_read)
    monkeypatch.setattr(fr, "_l4a_exact_one_shift", fake_one_shift)
    _reset_remap_worker()

    statuses = _l4a_exact_skycell_batch(
        "skycell.1.1",
        [(0, 1, 0, 3), (1, 0, -1, 5), (2, 2, 1, 7)],
    )
    assert calls == ["skycell.1.1"]
    assert statuses == [
        "skycell.1.1:e0:1,0:r3",
        "skycell.1.1:e1:0,-1:r5",
        "skycell.1.1:e2:2,1:r7",
    ]


def test_reset_joblib_executor_args_allows_second_pool_compare():
    """L4a then L4b must not crash on joblib initargs == (smoke_4)."""
    import joblib.executor as joblib_executor

    from syndiff_pipeline.template_creation.processing.field_remap import (
        _build_remap_worker_payload,
        _reset_joblib_executor_args,
    )

    schedule = _minimal_schedule(n_frames=2, n_skycells=2)
    common = dict(
        schedule=schedule,
        mapping_root=Path("/m"),
        sector=1,
        camera=1,
        ccd=1,
        base_tess_shape=(4, 4),
        oversampling_factor=1,
        intra_skycell_R=1,
        exact_l4a_dir=Path("/l4a"),
        exact_l4b_dir=Path("/l4b"),
        rebuild_remap_cache=False,
        rebuild_inter_skycell_cache=False,
        scratch_regmaps={},
        wcs_mode="event",
        frames_df=pd.DataFrame({"path": ["a.fits", "b.fits"]}),
    )
    l4a_payload = _build_remap_worker_payload(**common)
    # Old L4b path: master ndarray in initargs → joblib `_executor_args ==` raises.
    l4b_old = dict(l4a_payload)
    l4b_old["master"] = np.zeros((4, 4), dtype=np.int16)
    joblib_executor._executor_args = {"initargs": (l4a_payload,)}
    with pytest.raises(ValueError, match="ambiguous"):
        _ = joblib_executor._executor_args == {"initargs": (l4b_old,)}

    # Fixed path: master_path only (no ndarray) → compare is False, no raise.
    l4b_payload = _build_remap_worker_payload(
        **common, master_path="/fake/master.fits", idx_to_name={1: "skycell.0.0"}
    )
    assert l4b_payload["master"] is None
    assert l4b_payload["master_path"] == "/fake/master.fits"
    assert ({"initargs": (l4a_payload,)} == {"initargs": (l4b_payload,)}) is False

    # Belt-and-suspenders: clear cached args before L4b Parallel (production call site).
    joblib_executor._executor_args = {"initargs": (l4a_payload,)}
    _reset_joblib_executor_args()
    assert joblib_executor._executor_args is None


def test_init_remap_worker_loads_master_from_path(monkeypatch, tmp_path: Path):
    import syndiff_pipeline.template_creation.processing.field_remap as fr

    master = np.arange(6, dtype=np.int32).reshape(2, 3)
    master_path = tmp_path / "master.fits"

    def fake_map(path: Path):
        assert Path(path) == master_path
        return master, {"skycell.0.0": 1}

    monkeypatch.setattr(fr, "_master_skycell_id_map", fake_map)
    monkeypatch.setattr(
        fr,
        "create_tess_pixel_coordinates",
        lambda shape, os: (np.zeros((2, 2)), None),
        raising=False,
    )
    # pancakes.create_tess_pixel_coordinates is imported inside _init_remap_worker
    import syndiff_pipeline.template_creation.processing.pancakes as pancakes

    monkeypatch.setattr(
        pancakes,
        "create_tess_pixel_coordinates",
        lambda shape, os: (np.zeros((2, 2)), None),
    )

    fr._reset_remap_worker()
    payload = {
        "base_tess_shape": (2, 2),
        "oversampling_factor": 1,
        "master": None,
        "master_path": str(master_path),
        "scratch_regmaps": {},
        "wcs_mode": "event",
        "frames_df": None,
        "ffi_list_df": None,
        "frame_filenames": None,
        "skycell_rows": {},
        "mapping_root": str(tmp_path),
        "sector": 1,
        "camera": 1,
        "ccd": 1,
        "schedule": _minimal_schedule(),
        "intra_skycell_R": 1,
        "exact_l4a_dir": str(tmp_path),
        "exact_l4b_dir": str(tmp_path),
        "rebuild_remap_cache": False,
        "rebuild_inter_skycell_cache": False,
        "idx_to_name": {},
    }
    fr._init_remap_worker(payload)
    np.testing.assert_array_equal(fr._REMAP_WORKER["master"], master)
