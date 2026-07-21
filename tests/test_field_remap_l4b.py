"""Integration tests for remap inter-skycell (L4b) rim Exact cache writer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.template_creation.processing.field_abutting import (
    l4b_rim_cache_basename,
    l4b_rim_path,
)
from syndiff_pipeline.template_creation.processing.field_remap import (
    EXACT_CACHE_L4B_DIRNAME,
    REMAP_MANIFEST_NAME,
    _group_l4b_epochs_by_pair,
    _l4b_rim_pair_batch,
    remap_root,
    run_field_remap_scc,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import (
    ShiftSchedule,
    assign_groups_from_schedule,
    build_pair_epochs,
)


def _synthetic_schedule() -> ShiftSchedule:
    """Two abutting skycells with three unique pair-states across three frames."""
    names = np.array(["skycell.1.1", "skycell.1.2"])
    n_frames, n_cells = 3, 2
    sx_int = np.array([[0, 1], [0, 1], [1, 1]], dtype=np.int16)
    sy_int = np.array([[0, 0], [0, 1], [0, 0]], dtype=np.int16)
    return ShiftSchedule(
        skycell_names=names,
        sx_float=sx_int.astype(np.float32),
        sy_float=sy_int.astype(np.float32),
        sx_int=sx_int,
        sy_int=sy_int,
        frame_valid=np.ones(n_frames, dtype=bool),
        meta={"reference_ffi": "/data/ref.fits", "frame_filenames": ["f0.fits", "f1.fits", "f2.fits"]},
    )


def _synthetic_master() -> tuple[np.ndarray, dict[str, int]]:
    master = np.array([[10, 20]], dtype=np.int32)
    name_to_id = {"skycell.1.1": 10, "skycell.1.2": 20}
    return master, name_to_id


def _expected_pair_epoch_count(schedule: ShiftSchedule, master: np.ndarray) -> int:
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
        build_col_of_name,
        pair_column_indices,
    )

    _, name_to_id = _synthetic_master()
    idx_to_name = {int(v): str(k) for k, v in name_to_id.items()}
    pair_ids = abutting_undirected_pairs(master)
    pair_idx = pair_column_indices(
        pair_ids,
        name_to_id=name_to_id,
        col_of_name=build_col_of_name(schedule.skycell_names),
        idx_to_name=idx_to_name,
    )
    assignment = assign_groups_from_schedule(
        schedule,
        grouping_quantum_ps1_px=1.0,
        cache_quantum_ps1_px=1.0,
        keying="absolute",
    )
    pair_epochs, _ = build_pair_epochs(
        schedule,
        assignment.group_id_per_frame,
        pair_ids=pair_ids,
        pair_idx=pair_idx,
    )
    return int(len(pair_epochs))


@pytest.fixture
def remap_l4b_env(tmp_path: Path, monkeypatch):
    """Minimal remap store with mocked WCS/catalog/Exact for inter-skycell run."""
    data_root = tmp_path / "data"
    sector, camera, ccd = 1, 1, 1
    store = remap_root(data_root, sector, camera, ccd, oversampling_factor=1)
    event_dir = tmp_path / "event"
    event_dir.mkdir()
    frames_csv = event_dir / "frames.csv"
    frames_csv.write_text("path\n/data/fake0.fits\n/data/fake1.fits\n/data/fake2.fits\n")

    schedule = _synthetic_schedule()
    master, name_to_id = _synthetic_master()
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
        lambda _path: (master, name_to_id),
    )
    # Real MAPGRID master so _build_remap_tpix hard-fails are not tripped by a
    # fake path; master pixel contents are still mocked via _master_skycell_id_map.
    from astropy.io import fits

    from syndiff_pipeline.common.mapping_grid import MappingGrid

    grid = MappingGrid.from_ffi_shape(256, 256, conv_pad_native=2)
    master_path = tmp_path / "master_pixels2skycells.fits"
    hdu = fits.PrimaryHDU(data=np.zeros(grid.array_shape_native(), dtype=np.int32))
    for key, val in grid.to_fits_header_updates().items():
        hdu.header[key] = val
    hdu.writeto(master_path, overwrite=True)
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._master_pixels2skycells_path",
        lambda *a, **k: master_path,
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._load_frame_wcs",
        lambda _df, _i: MagicMock(name="tess_wcs"),
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._skycell_catalog_row",
        lambda mapping_root, sector, camera, ccd, skycell, **k: pd.Series(
            {"NAME": str(skycell), "RA": 0.0, "DEC": 0.0}
        ),
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_hybrid_exact.candidate_tess_ids_for_l4a",
        lambda *a, **k: (np.array([], dtype=np.int32), np.zeros((4, 6), dtype=bool)),
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._read_regmap_assignment",
        lambda skycell: np.zeros((4, 6), dtype=np.int32),
    )

    rim_a = np.array([101, 102], dtype=np.int32)
    rim_b = np.array([201, 202], dtype=np.int32)

    def fake_shared(master_arr, id_a, id_b):
        return rim_a.copy(), rim_b.copy()

    def fake_exact(_wcs, _row, tids, **kwargs):
        n = max(1, len(np.asarray(tids)))
        return np.arange(n, dtype=np.int32) + 1000

    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_hybrid_exact.shared_abutting_border_tess_ids",
        fake_shared,
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_hybrid_exact.exact_regmap_for_tess_ids",
        fake_exact,
    )

    return {
        "data_root": data_root,
        "event_dir": event_dir,
        "store": store,
        "sector": sector,
        "camera": camera,
        "ccd": ccd,
        "schedule": schedule,
        "master": master,
        "expected_n": _expected_pair_epoch_count(schedule, master),
    }


def test_inter_skycell_writes_expected_npzs(remap_l4b_env):
    env = remap_l4b_env
    result = run_field_remap_scc(
        sector=env["sector"],
        camera=env["camera"],
        ccd=env["ccd"],
        data_root=env["data_root"],
        event_dir=env["event_dir"],
        mapping_root=Path("/fake/mapping"),
        base_tess_shape=(4, 6),
        scc_only=False,
        store_root=env["store"],
    )

    l4b_dir = env["store"] / EXACT_CACHE_L4B_DIRNAME
    npz_files = sorted(l4b_dir.rglob("*_rim.npz"))
    assert result["n_inter_skycell_pair_states"] == env["expected_n"]
    assert result["n_inter_skycell_written"] == env["expected_n"]
    assert len(npz_files) == env["expected_n"]

    manifest = json.loads((env["store"] / REMAP_MANIFEST_NAME).read_text())
    assert manifest["n_inter_skycell_pair_states"] == env["expected_n"]
    assert manifest["n_pair_epochs"] == env["expected_n"]
    assert manifest["schema_version"] == 3
    assert "l4b_policy" not in manifest

    with np.load(npz_files[0]) as z:
        assert "exact_tid_lo" in z
        assert "exact_tid_hi" in z
        assert "rep_frame_index" in z
        assert "pair_epoch_id" in z
        assert int(z["id_lo"]) <= int(z["id_hi"])


def test_inter_skycell_basename_undirected(remap_l4b_env):
    env = remap_l4b_env
    run_field_remap_scc(
        sector=env["sector"],
        camera=env["camera"],
        ccd=env["ccd"],
        data_root=env["data_root"],
        event_dir=env["event_dir"],
        mapping_root=Path("/fake/mapping"),
        base_tess_shape=(4, 6),
        scc_only=False,
        store_root=env["store"],
    )
    l4b_dir = env["store"] / EXACT_CACHE_L4B_DIRNAME
    paths = {p.relative_to(l4b_dir).as_posix() for p in l4b_dir.rglob("*.npz")}
    # Undirected pair folder; epoch filenames include e{N}_
    assert any(p.startswith("pair_10__20/") for p in paths)
    canonical = l4b_rim_cache_basename(10, 20, 0, 0, 1, 0)
    swapped = l4b_rim_cache_basename(20, 10, 1, 0, 0, 0)
    assert canonical == swapped


def test_inter_skycell_skips_existing_unless_rebuild(remap_l4b_env):
    env = remap_l4b_env
    kwargs = dict(
        sector=env["sector"],
        camera=env["camera"],
        ccd=env["ccd"],
        data_root=env["data_root"],
        event_dir=env["event_dir"],
        mapping_root=Path("/fake/mapping"),
        base_tess_shape=(4, 6),
        scc_only=False,
        store_root=env["store"],
    )
    first = run_field_remap_scc(**kwargs)
    second = run_field_remap_scc(**kwargs)
    assert first["n_inter_skycell_written"] == env["expected_n"]
    assert second["n_inter_skycell_written"] == 0

    third = run_field_remap_scc(**kwargs, rebuild_inter_skycell_cache=True)
    assert third["n_inter_skycell_written"] == env["expected_n"]


def test_group_l4b_epochs_by_pair():
    pair_epochs = pd.DataFrame(
        {
            "id_lo": [10, 10, 30, 10],
            "id_hi": [20, 20, 40, 20],
            "pair_epoch_id": [0, 1, 0, 2],
            "sx_lo": [0, 1, 0, 2],
            "sy_lo": [0, 0, 1, 0],
            "sx_hi": [1, 1, 0, 1],
            "sy_hi": [0, 1, 0, 0],
            "rep_frame_index": [0, 1, 2, 3],
        }
    )
    batches = _group_l4b_epochs_by_pair(pair_epochs)
    assert [pair for pair, _ in batches] == [(10, 20), (30, 40)]
    assert len(batches[0][1]) == 3
    assert len(batches[1][1]) == 1
    assert sum(len(epochs) for _, epochs in batches) == len(pair_epochs)
    assert batches[0][1][0] == (0, 0, 0, 1, 0, 0)
    assert batches[1][1][0] == (0, 0, 1, 0, 0, 2)


def test_l4b_pair_batch_calls_border_ids_once(monkeypatch):
    import syndiff_pipeline.template_creation.processing.field_remap as fr
    import syndiff_pipeline.template_creation.processing.field_hybrid_exact as hy

    calls: list[tuple[int, int]] = []
    one_epoch_calls: list[tuple[int, int, int]] = []

    def fake_shared(_master, id_a, id_b):
        calls.append((int(id_a), int(id_b)))
        return (
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
        )

    def fake_one_epoch(
        id_lo,
        id_hi,
        pair_epoch_id,
        *_args,
        **_kwargs,
    ):
        one_epoch_calls.append((int(id_lo), int(id_hi), int(pair_epoch_id)))
        return "skip"

    monkeypatch.setattr(hy, "shared_abutting_border_tess_ids", fake_shared)
    monkeypatch.setattr(fr, "_l4b_rim_one_epoch", fake_one_epoch)
    monkeypatch.setattr(
        fr,
        "_worker_ps1_info",
        lambda skycell: pd.Series({"NAME": skycell}),
    )

    fr._reset_remap_worker()
    fr._REMAP_WORKER.update(
        {
            "exact_l4b_dir": "/tmp/l4b_unused",
            "rebuild_inter_skycell_cache": False,
            "idx_to_name": {10: "skycell.1.1", 20: "skycell.1.2"},
            "master": np.array([[10, 20]], dtype=np.int32),
            "skycell_rows": {},
        }
    )

    epochs = [
        (0, 0, 0, 1, 0, 0),
        (1, 1, 0, 1, 1, 1),
        (2, 2, 0, 1, 0, 2),
    ]
    statuses = _l4b_rim_pair_batch(10, 20, epochs)
    assert statuses == ["skip", "skip", "skip"]
    assert calls == [(10, 20)]
    assert one_epoch_calls == [(10, 20, 0), (10, 20, 1), (10, 20, 2)]


def test_l4b_reused_l4a_worker_loads_master_once(monkeypatch):
    """Simulate loky reuse: L4a init leaves idx_to_name but no master ndarray."""
    import syndiff_pipeline.template_creation.processing.field_remap as fr
    import syndiff_pipeline.template_creation.processing.field_hybrid_exact as hy

    master = np.array([[10, 20]], dtype=np.int32)
    map_calls = 0

    def fake_map(_path):
        nonlocal map_calls
        map_calls += 1
        return master, {"skycell.1.1": 10, "skycell.1.2": 20}

    def fake_shared(_master, id_a, id_b):
        return (
            np.array([1], dtype=np.int32),
            np.array([2], dtype=np.int32),
        )

    def fake_one_epoch(*_args, **_kwargs):
        return "write"

    monkeypatch.setattr(fr, "_master_skycell_id_map", fake_map)
    monkeypatch.setattr(hy, "shared_abutting_border_tess_ids", fake_shared)
    monkeypatch.setattr(fr, "_l4b_rim_one_epoch", fake_one_epoch)
    monkeypatch.setattr(
        fr,
        "_worker_ps1_info",
        lambda skycell: pd.Series({"NAME": skycell}),
    )

    fr._reset_remap_worker()
    # L4a-style worker state: idx_to_name from payload, master not loaded yet.
    fr._REMAP_WORKER.update(
        {
            "exact_l4b_dir": "/tmp/l4b_unused",
            "rebuild_inter_skycell_cache": False,
            "master_path": "/fake/master.fits",
            "idx_to_name": {10: "skycell.1.1", 20: "skycell.1.2"},
            "master": None,
            "skycell_rows": {},
        }
    )

    epochs = [(0, 0, 0, 1, 0, 0)]
    statuses = _l4b_rim_pair_batch(10, 20, epochs)
    assert statuses == ["write"]
    assert map_calls == 1
    assert fr._REMAP_WORKER["master"] is not None
