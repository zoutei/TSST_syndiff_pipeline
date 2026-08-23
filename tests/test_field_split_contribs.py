"""H.1: interior/seam-delta contrib split must reconstruct the plain
group-qualified contrib exactly, with real intra+inter-skycell blending
active (not mocked) -- see spicy-squishing-ritchie.md Part H."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from syndiff_pipeline.template_creation.processing.field_abutting import (
    abutting_undirected_pairs,
    l4b_rim_cache_basename,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    assemble_group_from_contribs,
    assemble_group_from_split_contribs,
    contrib_basename,
    interior_contrib_path,
    seam_delta_contrib_path,
)
from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
    build_l4a_hybrid_assignment,
)


# --- fixture helpers, mirrored from test_field_downsample_compose.py's own
# (untested-across-files) helpers, kept local to avoid cross-test-file imports.


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

    border_ids, _ = shared_abutting_border_tess_ids(master, skycell_id, 20)
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


def _synthetic_ps1(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    data = (np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) + 1.0) * 3.5
    mask = np.zeros(shape, dtype=np.int32)
    return data, mask


def test_split_contribs_reconstruct_plain_contrib_with_real_blending(
    tmp_path: Path, monkeypatch
) -> None:
    import syndiff_pipeline.template_creation.processing.field_downsample as fd
    from syndiff_pipeline.common.mapping_grid import MappingGrid

    master, name_to_id = _tiny_master()
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    neighbours = fd._neighbours_by_skycell_id(abutting_undirected_pairs(master))
    frozen = _frozen_a(master)
    ps1_data, ps1_mask = _synthetic_ps1(frozen.shape)

    store = tmp_path / "templates"
    (store / "contribs").mkdir(parents=True)
    l4a_dir = tmp_path / "l4a"
    l4b_dir = tmp_path / "l4b"

    sx_a, sy_a = 0, 0
    l4a_name = contrib_basename("skycell.1.1", sx_a, sy_a).replace(".npz", "_exact.npz")
    # Markers must be valid in-grid TESS flat ids (native grid here is
    # 4x3=12 pixels) so the patched border pixel survives
    # MappingGrid.contains_flat filtering and shows up in the binned flux --
    # not an out-of-range sentinel that gets dropped either way.
    _write_l4a_cache(l4a_dir / l4a_name, frozen, sx_a, sy_a, marker=4)

    # Two groups: skycell.1.1's own shift (0,0) is identical in both -- only
    # its neighbour's (skycell.1.2) shift differs -- so the *interior*
    # contrib is shareable, but the inter-skycell rim patch genuinely
    # differs, exactly the scenario H.1's split targets.
    group0_shifts = {"skycell.1.1": (sx_a, sy_a), "skycell.1.2": (0, 1)}
    group1_shifts = {"skycell.1.1": (sx_a, sy_a), "skycell.1.2": (2, 0)}
    rim0 = l4b_rim_cache_basename(10, 20, sx_a, sy_a, 0, 1)
    rim1 = l4b_rim_cache_basename(10, 20, sx_a, sy_a, 2, 0)
    _write_l4b_cache(
        l4b_dir / rim0,
        id_lo=10, id_hi=20, sx_lo=sx_a, sy_lo=sy_a, sx_hi=0, sy_hi=1,
        marker_lo=5, marker_hi=6, frozen_shape=frozen.shape, master=master,
    )
    _write_l4b_cache(
        l4b_dir / rim1,
        id_lo=10, id_hi=20, sx_lo=sx_a, sy_lo=sy_a, sx_hi=2, sy_hi=0,
        marker_lo=9, marker_hi=10, frozen_shape=frozen.shape, master=master,
    )

    def fake_read(skycell: str) -> np.ndarray:
        assert skycell == "skycell.1.1"
        return frozen

    def fake_zarr(_zstore, skycell: str):
        assert skycell == "skycell.1.1"
        return ps1_data, ps1_mask

    monkeypatch.setattr(fd, "_read_regmap_assignment_l5", fake_read)
    monkeypatch.setattr(fd, "_load_zarr_skycell", fake_zarr)

    # Two composite keys (content unused inside _l5_skycell_batch beyond
    # being a dict key) -- one gid each, so each triggers its own compose
    # call with its own group_shifts, matching how two distinct L4b
    # pair-states would really fan out.
    buckets = {("k0",): [(0, sx_a, sy_a)], ("k1",): [(1, sx_a, sy_a)]}

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
            "zstore": object(),
            "name_to_id": name_to_id,
            "id_to_name": id_to_name,
            "master_map": master,
            "pair_ids": abutting_undirected_pairs(master),
            "neighbours_by_id": neighbours,
            "group_shifts_by_gid": {0: group0_shifts, 1: group1_shifts},
            "epoch_index": None,
            "exact_cache_l4a_dir": str(l4a_dir),
            "exact_cache_l4b_dir": str(l4b_dir),
            "base_tess_shape": frozen.shape,
            "roi_bounds": (0, 0, frozen.shape[1], frozen.shape[0]),
            "ignore_mask": 0,
            "intra_skycell_R": 1,
            "write_split_contribs": True,
            "mapping_grid": MappingGrid(
                ffi_xmin=0,
                ffi_ymin=0,
                ffi_xmax=frozen.shape[1],
                ffi_ymax=frozen.shape[0],
                oversampling=1,
                conv_pad_native=0,
            ),
        }
    )
    result = fd._l5_skycell_batch("skycell.1.1", buckets)
    fd._reset_l5_worker()

    assert result["n_writes"] == 2
    assert result["n_compose"] == 2

    # Interior contrib written once, shared by both groups (group-independent).
    assert interior_contrib_path(store, "skycell.1.1", sx_a, sy_a).is_file()
    # Seam-delta contrib written per group, and the two groups' deltas differ
    # (real blending exercised, not a degenerate no-op).
    delta0 = seam_delta_contrib_path(store, "skycell.1.1", sx_a, sy_a, 0)
    delta1 = seam_delta_contrib_path(store, "skycell.1.1", sx_a, sy_a, 1)
    assert delta0.is_file() and delta1.is_file()

    shape = frozen.shape
    for gid in (0, 1):
        plain = assemble_group_from_contribs(
            store, [("skycell.1.1", sx_a, sy_a)], shape=shape, group_id=gid
        )
        split = assemble_group_from_split_contribs(
            store, [("skycell.1.1", sx_a, sy_a)], shape=shape, group_id=gid
        )
        np.testing.assert_array_equal(split["flux_sum"], plain["flux_sum"])
        np.testing.assert_array_equal(split["count"], plain["count"])
        np.testing.assert_array_equal(split["mask_count"], plain["mask_count"])

    plain0 = assemble_group_from_contribs(
        store, [("skycell.1.1", sx_a, sy_a)], shape=shape, group_id=0
    )
    plain1 = assemble_group_from_contribs(
        store, [("skycell.1.1", sx_a, sy_a)], shape=shape, group_id=1
    )
    assert not np.array_equal(plain0["flux_sum"], plain1["flux_sum"])
