"""H.1+H.2 integration: build_group_convolved_template (crop-aware,
disk-cached per-skycell patches) must match convolving the densely
assembled+cropped group template directly, on a real (synthetic) H.1
split-contrib store with genuine intra+inter-skycell blending."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from syndiff_pipeline.template_creation.processing.field_abutting import (
    abutting_undirected_pairs,
    l4b_rim_cache_basename,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    assemble_group_from_split_contribs,
    contrib_basename,
)
from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
    build_l4a_hybrid_assignment,
)

from syndiff_pipeline.difference_imaging.stages.convolved_templates_patch_cache import (
    build_group_convolved_template,
    precompute_basis_valid_maps,
)


def _master(ny: int, nx: int, split_col: int) -> tuple[np.ndarray, dict[str, int]]:
    master = np.zeros((ny, nx), dtype=np.int32)
    master[:, :split_col] = 10
    master[:, split_col:] = 20
    return master, {"skycell.1.1": 10, "skycell.1.2": 20}


def _frozen(master: np.ndarray, skycell_id: int, width: int) -> np.ndarray:
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        shared_abutting_border_tess_ids,
    )

    ny = master.shape[0]
    other_id = 20 if skycell_id == 10 else 10
    border_ids, _ = shared_abutting_border_tess_ids(master, skycell_id, other_id)
    frozen = np.full((ny, width), -1, dtype=np.int32)
    for y in range(ny):
        frozen[y, : width - 1] = 1000 + y * 100 + np.arange(width - 1)
    if border_ids.size:
        frozen[:, -1] = border_ids[:ny]
    return frozen


def _write_l4a_cache(path: Path, frozen: np.ndarray, sx: int, sy: int, marker: int) -> None:
    linear, mask = build_l4a_hybrid_assignment(frozen, sx, sy, exact_tid=None, hybrid_R=1)
    exact = linear.copy()
    if mask.any():
        exact[mask] = int(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, exact_tid=exact.astype(np.int32))


def _write_l4b_cache(
    path: Path, *, id_lo, id_hi, sx_lo, sy_lo, sx_hi, sy_hi, marker_lo, marker_hi,
    frozen_shape, master,
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
        path, exact_tid_lo=exact_lo, exact_tid_hi=exact_hi,
        id_lo=np.int32(id_lo), id_hi=np.int32(id_hi),
        sx_lo=np.int16(sx_lo), sy_lo=np.int16(sy_lo),
        sx_hi=np.int16(sx_hi), sy_hi=np.int16(sy_hi),
        rep_frame_index=np.int32(0), l4b_policy=np.array("pair_state"),
    )


def test_patch_cache_matches_dense_assemble_and_convolve(tmp_path: Path, monkeypatch):
    import syndiff_pipeline.template_creation.processing.field_downsample as fd
    from syndiff_pipeline.common.mapping_grid import MappingGrid

    ny, nx = 40, 30
    split_col = 15
    master, name_to_id = _master(ny, nx, split_col)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    neighbours = fd._neighbours_by_skycell_id(abutting_undirected_pairs(master))

    frozen_a = _frozen(master, 10, split_col)
    frozen_b = _frozen(master, 20, nx - split_col)

    store = tmp_path / "templates"
    (store / "contribs").mkdir(parents=True)
    l4a_dir = tmp_path / "l4a"
    l4b_dir = tmp_path / "l4b"

    sx_a, sy_a = 0, 0
    sx_b, sy_b = 0, 0
    l4a_name_a = contrib_basename("skycell.1.1", sx_a, sy_a).replace(".npz", "_exact.npz")
    l4a_name_b = contrib_basename("skycell.1.2", sx_b, sy_b).replace(".npz", "_exact.npz")
    _write_l4a_cache(l4a_dir / l4a_name_a, frozen_a, sx_a, sy_a, marker=5)
    _write_l4a_cache(l4a_dir / l4a_name_b, frozen_b, sx_b, sy_b, marker=6)

    group_shifts = {"skycell.1.1": (sx_a, sy_a), "skycell.1.2": (sx_b, sy_b)}
    rim = l4b_rim_cache_basename(10, 20, sx_a, sy_a, sx_b, sy_b)
    _write_l4b_cache(
        l4b_dir / rim, id_lo=10, id_hi=20, sx_lo=sx_a, sy_lo=sy_a, sx_hi=sx_b, sy_hi=sy_b,
        marker_lo=7, marker_hi=8, frozen_shape=frozen_a.shape, master=master,
    )
    # skycell.1.2's own l4b rim cache uses frozen_b's own shape for its side.
    rim_b_shape_cache = l4b_dir / rim
    # (single shared file covers both sides via lo/hi arrays sized to each
    # skycell's own frozen shape -- write once more with skycell.1.2's shape
    # for the hi side by reusing the same helper against frozen_b.)
    exact_lo = np.full(frozen_a.shape, -1, dtype=np.int32)
    exact_hi = np.full(frozen_b.shape, -1, dtype=np.int32)
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        shared_abutting_border_tess_ids,
    )
    ids_a, ids_b = shared_abutting_border_tess_ids(master, 10, 20)
    if ids_a.size:
        exact_lo[:, -1] = 7
    if ids_b.size:
        exact_hi[:, 0] = 8
    np.savez_compressed(
        rim_b_shape_cache, exact_tid_lo=exact_lo, exact_tid_hi=exact_hi,
        id_lo=np.int32(10), id_hi=np.int32(20),
        sx_lo=np.int16(sx_a), sy_lo=np.int16(sy_a),
        sx_hi=np.int16(sx_b), sy_hi=np.int16(sy_b),
        rep_frame_index=np.int32(0), l4b_policy=np.array("pair_state"),
    )

    ps1_a = (np.arange(frozen_a.size, dtype=np.float64).reshape(frozen_a.shape) + 1.0) * 2.1
    ps1_b = (np.arange(frozen_b.size, dtype=np.float64).reshape(frozen_b.shape) + 1.0) * 3.3

    def fake_read(skycell: str) -> np.ndarray:
        return frozen_a if skycell == "skycell.1.1" else frozen_b

    def fake_zarr(_zstore, skycell: str):
        if skycell == "skycell.1.1":
            return ps1_a, np.zeros(frozen_a.shape, dtype=np.int32)
        return ps1_b, np.zeros(frozen_b.shape, dtype=np.int32)

    monkeypatch.setattr(fd, "_read_regmap_assignment_l5", fake_read)
    monkeypatch.setattr(fd, "_load_zarr_skycell", fake_zarr)

    grid = MappingGrid(
        ffi_xmin=0, ffi_ymin=0, ffi_xmax=nx, ffi_ymax=ny, oversampling=1, conv_pad_native=0
    )
    for skycell, frozen in (("skycell.1.1", frozen_a), ("skycell.1.2", frozen_b)):
        skycell_id = name_to_id[skycell]
        l4a_name = l4a_name_a if skycell == "skycell.1.1" else l4a_name_b
        buckets = {("k0",): [(0, 0, 0)]}
        fd._reset_l5_worker()
        fd._init_l5_worker(
            {
                "store": str(store),
                "rebuild_field_store": True,
                "mapping_root": str(tmp_path),
                "sector": 1, "camera": 1, "ccd": 1, "oversampling_factor": 1,
                "scratch_regmaps": {}, "zarr_path": str(tmp_path / "dummy.zarr"),
                "zstore": object(),
                "name_to_id": name_to_id, "id_to_name": id_to_name,
                "master_map": master,
                "pair_ids": abutting_undirected_pairs(master),
                "neighbours_by_id": neighbours,
                "group_shifts_by_gid": {0: group_shifts},
                "epoch_index": None,
                "exact_cache_l4a_dir": str(l4a_dir),
                "exact_cache_l4b_dir": str(l4b_dir),
                "base_tess_shape": (ny, nx),
                "roi_bounds": (0, 0, nx, ny),
                "ignore_mask": 0,
                "intra_skycell_R": 1,
                "write_split_contribs": True,
                "mapping_grid": grid,
            }
        )
        fd._l5_skycell_batch(skycell, buckets)
        fd._reset_l5_worker()

    shifts = [("skycell.1.1", sx_a, sy_a), ("skycell.1.2", sx_b, sy_b)]

    # A genuine sub-crop, not the whole grid, to exercise the offset math.
    x0, x1, y0, y1 = 4, 26, 6, 34
    dense = assemble_group_from_split_contribs(
        store, shifts, shape=(ny, nx), crop=(x0, x1, y0, y1), group_id=0
    )["flux_sum"]

    sigma_gauss = [0.6]
    deg_fixe = [1]
    ker_order = 1
    from hotpants.pure.kernel import calculate_kernel_basis

    basis_arr = np.asarray(calculate_kernel_basis((9, 9), sigma_gauss, deg_fixe), dtype=np.float64)
    n_comp_ker = basis_arr.shape[0]
    hw_kernel = basis_arr.shape[1] // 2
    n_spatial = (ker_order + 1) * (ker_order + 2) // 2
    n_needed = 1 + 1 + (n_comp_ker - 1) * n_spatial
    rng = np.random.default_rng(7)
    kernel_sol = rng.normal(size=n_needed)

    # Reference: convolve the densely-assembled, already-cropped array
    # directly via this module's own (independently verified) whole-image
    # convolution primitives.
    from syndiff_pipeline.difference_imaging.stages.convolved_templates_patch_cache import (
        recombine_basis_maps_full,
    )

    dense_maps = precompute_basis_valid_maps(dense, basis_arr)
    conv_dense = recombine_basis_maps_full(
        dense_maps, kernel_sol,
        ny_hr=dense.shape[0], nx_hr=dense.shape[1], hw_kernel=hw_kernel,
        kc_step=6, n_comp_ker=n_comp_ker, ker_order=ker_order, oversample=1,
    )

    conv_patched = build_group_convolved_template(
        store, shifts, group_id=0,
        base_tess_shape=(ny, nx), crop_hr=(x0, x1, y0, y1),
        basis_funcs=basis_arr, kernel_solution=kernel_sol,
        hw_kernel=hw_kernel, kc_step=6, n_comp_ker=n_comp_ker, ker_order=ker_order,
        oversample=1,
    )

    np.testing.assert_allclose(conv_patched, conv_dense, rtol=0, atol=1e-8)
    # Sanity: interior is genuinely nonzero (real signal, not a degenerate
    # all-zero comparison).
    assert np.abs(conv_dense[hw_kernel:-hw_kernel, hw_kernel:-hw_kernel]).max() > 0
