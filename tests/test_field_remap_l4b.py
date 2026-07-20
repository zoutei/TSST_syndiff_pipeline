"""Integration tests for remap L4b F2 rim Exact cache writer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.template_creation.processing.field_abutting import (
    l4b_rim_cache_basename,
    unique_pair_states,
)
from syndiff_pipeline.template_creation.processing.field_remap import (
    EXACT_CACHE_L4B_DIRNAME,
    REMAP_MANIFEST_NAME,
    remap_root,
    run_field_remap_scc,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import (
    ShiftSchedule,
    assign_groups_from_schedule,
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


def _expected_pair_state_count(schedule: ShiftSchedule, master: np.ndarray) -> int:
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
    return len(
        unique_pair_states(
            schedule.sx_int,
            schedule.sy_int,
            pair_idx,
            schedule.frame_valid,
            pair_ids=pair_ids,
        )
    )


@pytest.fixture
def remap_l4b_env(tmp_path: Path, monkeypatch):
    """Minimal remap store with mocked WCS/catalog/Exact for L4b-only run."""
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
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_remap._master_pixels2skycells_path",
        lambda *a, **k: Path("/fake/master.fits"),
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
        "expected_n": _expected_pair_state_count(schedule, master),
    }


def test_l4b_pair_state_writes_expected_npzs(remap_l4b_env):
    env = remap_l4b_env
    result = run_field_remap_scc(
        sector=env["sector"],
        camera=env["camera"],
        ccd=env["ccd"],
        data_root=env["data_root"],
        event_dir=env["event_dir"],
        mapping_root=Path("/fake/mapping"),
        base_tess_shape=(4, 6),
        apply_hybrid_exact=False,
        l4b_policy="pair_state",
        scc_only=False,
        store_root=env["store"],
    )

    l4b_dir = env["store"] / EXACT_CACHE_L4B_DIRNAME
    npz_files = sorted(l4b_dir.glob("pair_*_rim.npz"))
    assert result["n_l4b_pair_states"] == env["expected_n"]
    assert result["n_l4b_written"] == env["expected_n"]
    assert len(npz_files) == env["expected_n"]

    manifest = json.loads((env["store"] / REMAP_MANIFEST_NAME).read_text())
    assert manifest["l4b_policy"] == "pair_state"
    assert manifest["n_l4b_pair_states"] == env["expected_n"]
    assert manifest["schema_version"] == 2

    with np.load(npz_files[0]) as z:
        assert "exact_tid_lo" in z
        assert "exact_tid_hi" in z
        assert "rep_frame_index" in z
        assert str(z["l4b_policy"]) == "pair_state"
        assert int(z["id_lo"]) <= int(z["id_hi"])


def test_l4b_basename_undirected(remap_l4b_env):
    env = remap_l4b_env
    run_field_remap_scc(
        sector=env["sector"],
        camera=env["camera"],
        ccd=env["ccd"],
        data_root=env["data_root"],
        event_dir=env["event_dir"],
        mapping_root=Path("/fake/mapping"),
        base_tess_shape=(4, 6),
        apply_hybrid_exact=False,
        l4b_policy="pair_state",
        scc_only=False,
        store_root=env["store"],
    )
    l4b_dir = env["store"] / EXACT_CACHE_L4B_DIRNAME
    names = {p.name for p in l4b_dir.glob("*.npz")}
    # Same pair-state keyed from either endpoint order must collide to one file.
    canonical = l4b_rim_cache_basename(10, 20, 0, 0, 1, 0)
    swapped = l4b_rim_cache_basename(20, 10, 1, 0, 0, 0)
    assert canonical == swapped
    assert canonical in names


def test_l4b_skips_existing_unless_rebuild(remap_l4b_env):
    env = remap_l4b_env
    kwargs = dict(
        sector=env["sector"],
        camera=env["camera"],
        ccd=env["ccd"],
        data_root=env["data_root"],
        event_dir=env["event_dir"],
        mapping_root=Path("/fake/mapping"),
        base_tess_shape=(4, 6),
        apply_hybrid_exact=False,
        l4b_policy="pair_state",
        scc_only=False,
        store_root=env["store"],
    )
    first = run_field_remap_scc(**kwargs)
    second = run_field_remap_scc(**kwargs)
    assert first["n_l4b_written"] == env["expected_n"]
    assert second["n_l4b_written"] == 0

    third = run_field_remap_scc(**kwargs, rebuild_l4b_cache=True)
    assert third["n_l4b_written"] == env["expected_n"]
