"""Architecture A L5 compose: L4a + L4b group-scoped downsample."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing.field_abutting import (
    l4b_rim_cache_basename,
)
from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
    compose_group_hybrid_assignment,
    hybrid_assignment_from_exact_cache,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    assemble_group_from_contribs,
    contrib_basename,
    contrib_path,
    parse_contrib_basename,
    write_contrib,
)
from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
    build_l4a_hybrid_assignment,
)


def _tiny_master() -> tuple[np.ndarray, dict[str, int]]:
    master = np.zeros((4, 6), dtype=np.int32)
    master[:, :3] = 10
    master[:, 3:] = 20
    name_to_id = {"skycell.1.1": 10, "skycell.1.2": 20}
    return master, name_to_id


def _frozen_a(master: np.ndarray, skycell_id: int = 10) -> np.ndarray:
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        shared_abutting_border_tess_ids,
    )

    t_x = master.shape[1]
    border_ids, _ = shared_abutting_border_tess_ids(master, skycell_id, 20)
    # PS1 map wide enough to host rim pixels (3 cols for A territory).
    frozen = np.full((master.shape[0], 3), -1, dtype=np.int32)
    for y in range(frozen.shape[0]):
        frozen[y, :2] = 100 + y * 10 + np.arange(2)
    if border_ids.size:
        frozen[:, 2] = border_ids[: frozen.shape[0]]
    return frozen


def _write_l4a_cache(path: Path, frozen: np.ndarray, sx: int, sy: int, marker: int) -> None:
    linear, mask = build_l4a_hybrid_assignment(frozen, sx, sy, exact_tid=None, hybrid_R=1)
    exact = linear.copy()
    if mask.any():
        exact[mask] = int(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, exact_tid=exact.astype(np.int32))


def _write_l4b_cache(
    path: Path,
    *,
    id_lo: int,
    id_hi: int,
    sx_lo: int,
    sy_lo: int,
    sx_hi: int,
    sy_hi: int,
    marker_lo: int,
    marker_hi: int,
    frozen_shape: tuple[int, int],
    master: np.ndarray,
) -> None:
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        shared_abutting_border_tess_ids,
    )

    exact_lo = np.full(frozen_shape, -1, dtype=np.int32)
    exact_hi = np.full(frozen_shape, -1, dtype=np.int32)
    ids_a, ids_b = shared_abutting_border_tess_ids(master, id_lo, id_hi)
    if ids_a.size:
        exact_lo[:, -1] = int(marker_lo)
    if ids_b.size:
        exact_hi[:, 0] = int(marker_hi)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        exact_tid_lo=exact_lo,
        exact_tid_hi=exact_hi,
        id_lo=np.int32(id_lo),
        id_hi=np.int32(id_hi),
        sx_lo=np.int16(sx_lo),
        sy_lo=np.int16(sy_lo),
        sx_hi=np.int16(sx_hi),
        sy_hi=np.int16(sy_hi),
        rep_frame_index=np.int32(0),
        l4b_policy=np.array("pair_state"),
    )


def test_l4b_rim_overwrites_l4a_overlap():
    master, name_to_id = _tiny_master()
    frozen = _frozen_a(master)
    sx_a, sy_a = 0, 0
    l4a_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4a"
    l4b_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4b"
    l4a_name = contrib_basename("skycell.1.1", sx_a, sy_a).replace(".npz", "_exact.npz")
    _write_l4a_cache(l4a_dir / l4a_name, frozen, sx_a, sy_a, marker=9001)

    rim_name = l4b_rim_cache_basename(10, 20, sx_a, sy_a, 0, 1)
    _write_l4b_cache(
        l4b_dir / rim_name,
        id_lo=10,
        id_hi=20,
        sx_lo=sx_a,
        sy_lo=sy_a,
        sx_hi=0,
        sy_hi=1,
        marker_lo=7777,
        marker_hi=8888,
        frozen_shape=frozen.shape,
        master=master,
    )

    hybrid, meta = compose_group_hybrid_assignment(
        frozen,
        skycell="skycell.1.1",
        skycell_id=10,
        sx_int=sx_a,
        sy_int=sy_a,
        master=master,
        group_shifts={"skycell.1.1": (sx_a, sy_a), "skycell.1.2": (0, 1)},
        name_to_id=name_to_id,
        l4a_cache_path=l4a_dir / l4a_name,
        l4b_cache_dir=l4b_dir,
        require_inter_skycell_cache=True,
    )
    assert meta["n_inter_skycell_patches"] == 1
    l4a_only, _ = hybrid_assignment_from_exact_cache(
        frozen, sx_a, sy_a, l4a_dir / l4a_name, hybrid_R=1
    )
    assert not np.array_equal(hybrid, l4a_only)
    rim_col = frozen.shape[1] - 1
    assert int(hybrid[0, rim_col]) == 7777
    assert int(l4a_only[0, rim_col]) == 9001


def test_group_qualified_contribs_differ_for_shared_type1_key():
    master, name_to_id = _tiny_master()
    frozen = _frozen_a(master)
    sx_a, sy_a = 0, 0
    store = Path(tempfile.mkdtemp())
    l4a_dir = store / "exact_cache_l4a"
    l4b_dir = store / "exact_cache_l4b"
    l4a_name = contrib_basename("skycell.1.1", sx_a, sy_a).replace(".npz", "_exact.npz")
    _write_l4a_cache(l4a_dir / l4a_name, frozen, sx_a, sy_a, marker=100)

    group0_shifts = {"skycell.1.1": (sx_a, sy_a), "skycell.1.2": (0, 1)}
    group1_shifts = {"skycell.1.1": (sx_a, sy_a), "skycell.1.2": (2, 0)}

    rim0 = l4b_rim_cache_basename(10, 20, sx_a, sy_a, 0, 1)
    rim1 = l4b_rim_cache_basename(10, 20, sx_a, sy_a, 2, 0)
    _write_l4b_cache(
        l4b_dir / rim0,
        id_lo=10,
        id_hi=20,
        sx_lo=sx_a,
        sy_lo=sy_a,
        sx_hi=0,
        sy_hi=1,
        marker_lo=501,
        marker_hi=502,
        frozen_shape=frozen.shape,
        master=master,
    )
    _write_l4b_cache(
        l4b_dir / rim1,
        id_lo=10,
        id_hi=20,
        sx_lo=sx_a,
        sy_lo=sy_a,
        sx_hi=2,
        sy_hi=0,
        marker_lo=601,
        marker_hi=602,
        frozen_shape=frozen.shape,
        master=master,
    )

    h0, _ = compose_group_hybrid_assignment(
        frozen,
        skycell="skycell.1.1",
        skycell_id=10,
        sx_int=sx_a,
        sy_int=sy_a,
        master=master,
        group_shifts=group0_shifts,
        name_to_id=name_to_id,
        l4a_cache_path=l4a_dir / l4a_name,
        l4b_cache_dir=l4b_dir,
    )
    h1, _ = compose_group_hybrid_assignment(
        frozen,
        skycell="skycell.1.1",
        skycell_id=10,
        sx_int=sx_a,
        sy_int=sy_a,
        master=master,
        group_shifts=group1_shifts,
        name_to_id=name_to_id,
        l4a_cache_path=l4a_dir / l4a_name,
        l4b_cache_dir=l4b_dir,
    )
    rim_col = frozen.shape[1] - 1
    assert int(h0[0, rim_col]) == 501
    assert int(h1[0, rim_col]) == 601
    assert not np.array_equal(h0, h1)

    p0 = contrib_path(store, "skycell.1.1", sx_a, sy_a, group_id=0)
    p1 = contrib_path(store, "skycell.1.1", sx_a, sy_a, group_id=1)
    assert p0.name.endswith("_gid0.npz")
    assert p1.name.endswith("_gid1.npz")
    assert p0 != p1
    assert parse_contrib_basename(p0.name) == ("skycell.1.1", sx_a, sy_a, 0)


def test_missing_l4b_cache_raises_when_required():
    master, name_to_id = _tiny_master()
    frozen = _frozen_a(master)
    l4a_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4a"
    l4b_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4b"
    l4a_name = contrib_basename("skycell.1.1", 0, 0).replace(".npz", "_exact.npz")
    _write_l4a_cache(l4a_dir / l4a_name, frozen, 0, 0, marker=1)

    with pytest.raises(FileNotFoundError, match="inter-skycell rim cache missing"):
        compose_group_hybrid_assignment(
            frozen,
            skycell="skycell.1.1",
            skycell_id=10,
            sx_int=0,
            sy_int=0,
            master=master,
            group_shifts={"skycell.1.1": (0, 0), "skycell.1.2": (0, 1)},
            name_to_id=name_to_id,
            l4a_cache_path=l4a_dir / l4a_name,
            l4b_cache_dir=l4b_dir,
            require_inter_skycell_cache=True,
        )


def test_l4a_only_legacy_contrib_basename():
    name = contrib_basename("skycell.2588.036", -2, 5)
    assert name == "skycell.2588.036_sx-2_sy+5.npz"
    assert parse_contrib_basename(name) == ("skycell.2588.036", -2, 5)

    gid_name = contrib_basename("skycell.2588.036", -2, 5, group_id=3)
    assert gid_name == "skycell.2588.036_sx-2_sy+5_gid3.npz"
    assert parse_contrib_basename(gid_name) == ("skycell.2588.036", -2, 5, 3)


def test_assemble_loads_group_qualified_contrib():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "contribs").mkdir()
        (root / "field_mode_assembly.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "l4b_policy": "pair_state",
                    "group_scoped_contribs": True,
                }
            )
        )
        write_contrib(
            root,
            "skycell.1.1",
            1,
            0,
            group_id=2,
            indices=np.array([5], dtype=np.int64),
            flux_sum=np.array([42.0]),
            count=np.array([1.0]),
        )
        out = assemble_group_from_contribs(
            root,
            [("skycell.1.1", 1, 0)],
            shape=(2, 4),
            group_id=2,
        )
        assert float(out["flux_sum"].ravel()[5]) == 42.0


def test_composite_key_differs_when_neighbour_shift_differs():
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
    )
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _composite_key_for_group,
        _neighbours_by_skycell_id,
    )

    master, name_to_id = _tiny_master()
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    neighbours = _neighbours_by_skycell_id(abutting_undirected_pairs(master))
    k0 = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=0,
        sx_int=1,
        sy_int=0,
        group_shifts={"skycell.1.1": (1, 0), "skycell.1.2": (0, 1)},
        neighbour_ids=neighbours[10],
        id_to_name=id_to_name,
        epoch_index=None,
    )
    k1 = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=1,
        sx_int=1,
        sy_int=0,
        group_shifts={"skycell.1.1": (1, 0), "skycell.1.2": (2, 0)},
        neighbour_ids=neighbours[10],
        id_to_name=id_to_name,
        epoch_index=None,
    )
    assert k0 != k1


def test_composite_key_zero_shift_differs_on_neighbour_l4b():
    """(0,0) own-shift must still separate groups with different neighbour geometry."""
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
    )
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _composite_key_for_group,
        _neighbours_by_skycell_id,
    )

    master, name_to_id = _tiny_master()
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    neighbours = _neighbours_by_skycell_id(abutting_undirected_pairs(master))
    k0 = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=0,
        sx_int=0,
        sy_int=0,
        group_shifts={"skycell.1.1": (0, 0), "skycell.1.2": (0, 1)},
        neighbour_ids=neighbours[10],
        id_to_name=id_to_name,
        epoch_index=None,
    )
    k1 = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=1,
        sx_int=0,
        sy_int=0,
        group_shifts={"skycell.1.1": (0, 0), "skycell.1.2": (2, 0)},
        neighbour_ids=neighbours[10],
        id_to_name=id_to_name,
        epoch_index=None,
    )
    assert k0 != k1
    assert k0[0] == 0 and k0[1] == 0
    assert k0 != ("zero",)


def test_composite_key_zero_shift_epoch_uses_roll0_sentinel():
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
    )
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _composite_key_for_group,
        _neighbours_by_skycell_id,
    )

    master, name_to_id = _tiny_master()
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    neighbours = _neighbours_by_skycell_id(abutting_undirected_pairs(master))
    # Minimal epoch index: only L4b pair epochs for (0,0) vs neighbour shifts.
    epoch_index = {
        "l4a": {},
        "l4b": {
            (10, 20, 0, 0, 0, 0, 1): 11,
            (10, 20, 1, 0, 0, 2, 0): 22,
        },
    }
    k0 = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=0,
        sx_int=0,
        sy_int=0,
        group_shifts={"skycell.1.1": (0, 0), "skycell.1.2": (0, 1)},
        neighbour_ids=neighbours[10],
        id_to_name=id_to_name,
        epoch_index=epoch_index,
    )
    k1 = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=1,
        sx_int=0,
        sy_int=0,
        group_shifts={"skycell.1.1": (0, 0), "skycell.1.2": (2, 0)},
        neighbour_ids=neighbours[10],
        id_to_name=id_to_name,
        epoch_index=epoch_index,
    )
    assert k0[0] == "roll0"
    assert k1[0] == "roll0"
    assert k0 != k1


def test_any_nonempty_contrib_finds_late_key(tmp_path: Path):
    """All-skip resume must not false-fail when only late sorted keys are nonempty."""
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _any_nonempty_contrib,
    )

    store = tmp_path / "templates"
    (store / "contribs").mkdir(parents=True)
    # First 8 keys empty; key 20 nonempty (beyond the old first-8 sample).
    key_list = []
    for i in range(24):
        gid = i
        skycell = f"skycell.1.{i % 3 + 1}"
        # Normalize to real-looking names used by contrib_path
        skycell = "skycell.1.1" if i < 20 else "skycell.9.9"
        key_list.append((gid, skycell, 0, 0))
        write_contrib(
            store,
            skycell,
            0,
            0,
            group_id=gid,
            indices=(
                np.array([1], dtype=np.int64)
                if i == 20
                else np.array([], dtype=np.int64)
            ),
            flux_sum=(
                np.array([1.0]) if i == 20 else np.array([], dtype=np.float64)
            ),
            count=(
                np.array([1.0]) if i == 20 else np.array([], dtype=np.float64)
            ),
        )
    assert _any_nonempty_contrib(store, key_list)
    # Only empties → False
    empty_keys = key_list[:8]
    assert not _any_nonempty_contrib(store, empty_keys)


def test_composite_key_merges_identical_neighbour_geometry():
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
    )
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _build_skycell_composite_index,
        _composite_key_for_group,
        _neighbours_by_skycell_id,
    )

    master, name_to_id = _tiny_master()
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    neighbours = _neighbours_by_skycell_id(abutting_undirected_pairs(master))
    shifts = {"skycell.1.1": (1, 0), "skycell.1.2": (0, 1)}
    k0 = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=0,
        sx_int=1,
        sy_int=0,
        group_shifts=shifts,
        neighbour_ids=neighbours[10],
        id_to_name=id_to_name,
        epoch_index=None,
    )
    k1 = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=1,
        sx_int=1,
        sy_int=0,
        group_shifts=shifts,
        neighbour_ids=neighbours[10],
        id_to_name=id_to_name,
        epoch_index=None,
    )
    assert k0 == k1

    key_list = [
        (0, "skycell.1.1", 1, 0),
        (1, "skycell.1.1", 1, 0),
        (2, "skycell.1.1", 1, 0),
    ]
    group_shifts_by_gid = {0: shifts, 1: shifts, 2: shifts}
    index = _build_skycell_composite_index(
        key_list=key_list,
        group_shifts_by_gid=group_shifts_by_gid,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        neighbours_by_id=neighbours,
        epoch_index=None,
    )
    assert len(index["skycell.1.1"]) == 1
    assert len(next(iter(index["skycell.1.1"].values()))) == 3


def test_composite_key_index_skips_skycells_missing_from_master_map():
    """Remap's shift schedule can reference skycells outside the current
    master id map (buffer-region skycells, or a convolved store built under a
    since-rebuilt mapping); these must be skipped, not raise."""
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
    )
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _build_skycell_composite_index,
        _neighbours_by_skycell_id,
    )

    master, name_to_id = _tiny_master()
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    neighbours = _neighbours_by_skycell_id(abutting_undirected_pairs(master))
    shifts = {"skycell.1.1": (1, 0)}

    key_list = [
        (0, "skycell.1.1", 1, 0),
        (0, "skycell.stale.99", 0, 0),  # not in name_to_id
    ]
    index = _build_skycell_composite_index(
        key_list=key_list,
        group_shifts_by_gid={0: shifts},
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        neighbours_by_id=neighbours,
        epoch_index=None,
    )
    assert list(index.keys()) == ["skycell.1.1"]


def test_l5_skycell_batch_loads_regmap_and_zarr_once(monkeypatch, tmp_path: Path):
    import syndiff_pipeline.template_creation.processing.field_downsample as fd
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
    )

    master, name_to_id = _tiny_master()
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    neighbours = fd._neighbours_by_skycell_id(abutting_undirected_pairs(master))
    store = tmp_path / "templates"
    (store / "contribs").mkdir(parents=True)

    regmap_calls: list[str] = []
    zarr_calls: list[str] = []

    def fake_read(skycell: str) -> np.ndarray:
        regmap_calls.append(skycell)
        return np.arange(12, dtype=np.int32).reshape(4, 3)

    def fake_zarr(_zstore, skycell: str):
        zarr_calls.append(skycell)
        data = np.ones((4, 3), dtype=np.float32)
        mask = np.zeros((4, 3), dtype=np.int32)
        return data, mask

    monkeypatch.setattr(fd, "_read_regmap_assignment_l5", fake_read)
    monkeypatch.setattr(fd, "_load_zarr_skycell", fake_zarr)
    monkeypatch.setattr(
        fd,
        "_bin_skycell_contrib",
        lambda **kwargs: (
            np.array([1], dtype=np.int64),
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
        ),
    )

    shifts = {"skycell.1.1": (1, 0), "skycell.1.2": (0, 1)}
    buckets = {
        (1, 0, ((20, 0, 1),)): [(0, 1, 0), (1, 1, 0), (2, 1, 0)],
    }
    fd._reset_l5_worker()
    fd._init_l5_worker(
        {
            "store": str(store),
            "rebuild_field_store": True,
            "mapping_root": str(tmp_path),
            "sector": 1,
            "camera": 1,
            "ccd": 1,
            "oversampling_factor": 1,
            "scratch_regmaps": {},
            "zarr_path": str(tmp_path / "dummy.zarr"),
            "zstore": object(),  # skip real zarr open; _load_zarr_skycell mocked
            "name_to_id": name_to_id,
            "id_to_name": id_to_name,
            "master_map": master,
            "pair_ids": abutting_undirected_pairs(master),
            "neighbours_by_id": neighbours,
            "group_shifts_by_gid": {0: shifts, 1: shifts, 2: shifts},
            "epoch_index": None,
            "exact_cache_l4a_dir": str(tmp_path / "l4a"),
            "exact_cache_l4b_dir": str(tmp_path / "l4b"),
            "base_tess_shape": (4, 6),
            "roi_bounds": (0, 0, 6, 4),
            "ignore_mask": 0,
            "intra_skycell_R": 1,
        }
    )
    # Bypass compose (would need caches); force zero-key path via patched buckets
    # that still go through compose — monkeypatch compose to identity.
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_hybrid_exact.compose_group_hybrid_assignment",
        lambda frozen, **kwargs: (frozen, {"n_l4b_patches": 0, "cache_hit": True}),
    )
    result = fd._l5_skycell_batch("skycell.1.1", buckets)
    fd._reset_l5_worker()

    assert regmap_calls == ["skycell.1.1"]
    assert zarr_calls == ["skycell.1.1"]
    assert result["n_writes"] == 3
    assert result["n_compose"] == 1
    for gid in (0, 1, 2):
        assert contrib_path(store, "skycell.1.1", 1, 0, group_id=gid).is_file()


def test_write_contrib_atomic_no_lock(tmp_path: Path):
    """Concurrent-safe atomic replace leaves a readable NPZ without store lock."""
    from syndiff_pipeline.template_creation.processing.field_templates import (
        load_contrib,
    )

    store = tmp_path / "store"
    (store / "contribs").mkdir(parents=True)
    out = write_contrib(
        store,
        "skycell.1.1",
        0,
        0,
        group_id=7,
        indices=np.array([3, 4], dtype=np.int64),
        flux_sum=np.array([1.5, 2.5]),
        count=np.array([1.0, 1.0]),
    )
    assert out.is_file()
    assert not list(out.parent.glob(".*.tmp.npz"))
    data = load_contrib(out)
    assert list(data["indices"]) == [3, 4]


def test_bin_skycell_contrib_uses_mapping_grid_contains_flat():
    from syndiff_pipeline.common.mapping_grid import MappingGrid
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _bin_skycell_contrib,
    )

    # Local grid 2x3 (ffi x=[0,3), y=[0,2)); flats 0..5 in-grid, 6+ out.
    grid = MappingGrid(
        ffi_xmin=0, ffi_ymin=0, ffi_xmax=3, ffi_ymax=2, oversampling=1, conv_pad_native=0
    )
    assignment = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    # Inject an out-of-grid flat that must be dropped.
    assignment = assignment.copy()
    assignment[1, 2] = 99
    ps1 = np.ones((2, 3), dtype=np.float32)
    mask = np.zeros((2, 3), dtype=np.int32)
    binned = _bin_skycell_contrib(
        assignment=assignment,
        ps1_data=ps1,
        ps1_mask=mask,
        sx_int=0,
        sy_int=0,
        base_tess_shape=grid.array_shape_native(),
        roi_bounds=(0, 0, 3, 2),
        ignore_mask=0,
        mapping_grid=grid,
    )
    assert binned is not None
    idxs = set(int(i) for i in binned[0])
    assert idxs == {0, 1, 2, 3, 4}
    assert 99 not in idxs
    assert 5 not in idxs


def test_compose_skips_intra_when_apply_intra_false():
    master, name_to_id = _tiny_master()
    frozen = _frozen_a(master)
    l4a_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4a"
    l4b_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4b"
    l4a_name = contrib_basename("skycell.1.1", 1, 0).replace(".npz", "_exact.npz")
    _write_l4a_cache(l4a_dir / l4a_name, frozen, 1, 0, marker=9001)

    hybrid, meta = compose_group_hybrid_assignment(
        frozen,
        skycell="skycell.1.1",
        skycell_id=10,
        sx_int=1,
        sy_int=0,
        master=master,
        group_shifts={"skycell.1.1": (1, 0), "skycell.1.2": (0, 1)},
        name_to_id=name_to_id,
        l4a_cache_path=l4a_dir / l4a_name,
        l4b_cache_dir=l4b_dir,
        apply_intra_skycell=False,
        apply_inter_skycell=False,
        require_intra_skycell_cache=False,
        require_inter_skycell_cache=False,
    )
    assert meta["apply_intra_skycell"] is False
    assert meta.get("intra_skycell_roll_only") is True
    assert meta["n_inter_skycell_patches"] == 0
    from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
        roll_assignment,
    )

    linear = roll_assignment(frozen, 1, 0, convention="assignment")
    assert np.array_equal(hybrid, linear)


def test_compose_skips_inter_when_apply_inter_false():
    master, name_to_id = _tiny_master()
    frozen = _frozen_a(master)
    sx_a, sy_a = 0, 0
    l4a_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4a"
    l4b_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4b"
    l4a_name = contrib_basename("skycell.1.1", sx_a, sy_a).replace(".npz", "_exact.npz")
    _write_l4a_cache(l4a_dir / l4a_name, frozen, sx_a, sy_a, marker=9001)

    rim_name = l4b_rim_cache_basename(10, 20, sx_a, sy_a, 0, 1)
    _write_l4b_cache(
        l4b_dir / rim_name,
        id_lo=10,
        id_hi=20,
        sx_lo=sx_a,
        sy_lo=sy_a,
        sx_hi=0,
        sy_hi=1,
        marker_lo=7777,
        marker_hi=8888,
        frozen_shape=frozen.shape,
        master=master,
    )

    hybrid, meta = compose_group_hybrid_assignment(
        frozen,
        skycell="skycell.1.1",
        skycell_id=10,
        sx_int=sx_a,
        sy_int=sy_a,
        master=master,
        group_shifts={"skycell.1.1": (sx_a, sy_a), "skycell.1.2": (0, 1)},
        name_to_id=name_to_id,
        l4a_cache_path=l4a_dir / l4a_name,
        l4b_cache_dir=l4b_dir,
        apply_inter_skycell=False,
        require_inter_skycell_cache=False,
    )
    assert meta["n_inter_skycell_patches"] == 0
    l4a_only, _ = hybrid_assignment_from_exact_cache(
        frozen, sx_a, sy_a, l4a_dir / l4a_name, hybrid_R=1
    )
    assert np.array_equal(hybrid, l4a_only)


def test_compose_tolerates_missing_rim_cache_when_not_required():
    """A rim cache that failed to write upstream (e.g. remap's known int16
    overflow on extreme drift) must not abort downsample when
    apply_inter_skycell=True but require_inter_skycell_cache=False — it
    should skip that one patch and fall back to the L4a-only assignment."""
    master, name_to_id = _tiny_master()
    frozen = _frozen_a(master)
    sx_a, sy_a = 0, 0
    l4a_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4a"
    l4b_dir = Path(tempfile.mkdtemp()) / "exact_cache_l4b"  # no rim cache written
    l4a_name = contrib_basename("skycell.1.1", sx_a, sy_a).replace(".npz", "_exact.npz")
    _write_l4a_cache(l4a_dir / l4a_name, frozen, sx_a, sy_a, marker=9001)

    hybrid, meta = compose_group_hybrid_assignment(
        frozen,
        skycell="skycell.1.1",
        skycell_id=10,
        sx_int=sx_a,
        sy_int=sy_a,
        master=master,
        group_shifts={"skycell.1.1": (sx_a, sy_a), "skycell.1.2": (0, 1)},
        name_to_id=name_to_id,
        l4a_cache_path=l4a_dir / l4a_name,
        l4b_cache_dir=l4b_dir,
        apply_inter_skycell=True,
        require_inter_skycell_cache=False,
    )
    assert meta["n_inter_skycell_patches"] == 0
    assert meta["n_inter_skycell_missing"] == 1
    l4a_only, _ = hybrid_assignment_from_exact_cache(
        frozen, sx_a, sy_a, l4a_dir / l4a_name, hybrid_R=1
    )
    assert np.array_equal(hybrid, l4a_only)


def test_composite_key_omits_inter_when_apply_inter_false():
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _composite_key_for_group,
    )

    epoch_index = {
        "l4a": {("skycell.1.1", 0, 1, 0): 42},
        "l4b": {(10, 20, 0, 1, 0, 0, 1): 7},
    }
    with_inter = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=0,
        sx_int=1,
        sy_int=0,
        group_shifts={"skycell.1.1": (1, 0), "skycell.1.2": (0, 1)},
        neighbour_ids=[20],
        id_to_name={10: "skycell.1.1", 20: "skycell.1.2"},
        epoch_index=epoch_index,
        apply_intra_skycell=True,
        apply_inter_skycell=True,
    )
    without_inter = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=0,
        sx_int=1,
        sy_int=0,
        group_shifts={"skycell.1.1": (1, 0), "skycell.1.2": (0, 1)},
        neighbour_ids=[20],
        id_to_name={10: "skycell.1.1", 20: "skycell.1.2"},
        epoch_index=epoch_index,
        apply_intra_skycell=True,
        apply_inter_skycell=False,
    )
    assert with_inter == (42, ((20, 7),))
    assert without_inter == (42,)

    roll_only = _composite_key_for_group(
        skycell="skycell.1.1",
        skycell_id=10,
        group_id=0,
        sx_int=1,
        sy_int=0,
        group_shifts={"skycell.1.1": (1, 0), "skycell.1.2": (0, 1)},
        neighbour_ids=[20],
        id_to_name={10: "skycell.1.1", 20: "skycell.1.2"},
        epoch_index=epoch_index,
        apply_intra_skycell=False,
        apply_inter_skycell=False,
    )
    assert roll_only == ("roll0",)
