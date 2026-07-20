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
        require_l4b_cache=True,
    )
    assert meta["n_l4b_patches"] == 1
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

    with pytest.raises(FileNotFoundError, match="L4b rim cache missing"):
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
            require_l4b_cache=True,
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
