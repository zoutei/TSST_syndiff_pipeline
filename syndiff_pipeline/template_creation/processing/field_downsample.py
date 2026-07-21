"""Field-mode downsample (L5): bin sparse contribs from remap artifacts.

Reads shift schedule, group artifacts, and optional ``exact_cache_l4a/`` from
``remap/oversampling_{N}/`` (or legacy colocated ``templates/`` during
migration) and writes ``contribs/`` under ``templates/oversampling_{N}/``.
"""

from __future__ import annotations

import errno
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.wcs_grouping import _frames_csv_path
from syndiff_pipeline.template_creation.processing.field_remap import (
    GID_EPOCH_INDEX_NPZ,
    GROUP_ID_PER_FRAME_NPY,
    REMAP_MANIFEST_NAME,
    _find_regmap,
    _mapping_scc_dir,
    _master_pixels2skycells_path,
    _master_skycell_id_map,
    exact_cache_dir_for_read_root,
    exact_cache_l4b_dir_for_read_root,
    load_gid_epoch_index,
    load_remap_shifts_df,
    remap_root,
    resolve_l4a_epoch_id,
    resolve_remap_read_root,
)
from syndiff_pipeline.template_creation.processing.field_abutting import (
    l4a_exact_path,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    FieldManifest,
    FITS_DIRNAME,
    MANIFEST_NAME,
    MATERIALIZED_FITS_SIDECAR,
    assemble_group_from_contribs,
    build_field_fits_header,
    contrib_basename,
    contrib_path,
    field_fits_path,
    templates_root,
    write_contrib,
    write_field_group_fits,
    write_template_manifest,
    _roi_bounds_to_assemble_crop,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import (
    ShiftSchedule,
    assign_groups_from_schedule,
    write_group_artifacts,
)

log = logging.getLogger(__name__)

# Re-export mapping helpers for existing tests/callers.
__all__ = [
    "_find_regmap",
    "_mapping_scc_dir",
    "_master_pixels2skycells_path",
    "_bin_skycell_contrib",
    "_build_skycell_composite_index",
    "_composite_key_for_group",
    "_init_l5_worker",
    "_l5_skycell_batch",
    "_neighbours_by_skycell_id",
    "_reset_l5_worker",
    "assemble_field_group_count",
    "assemble_field_group_flux",
    "materialize_field_fits_for_store",
    "run_field_downsample_scc",
]


def _load_remap_manifest(read_root: Path) -> dict[str, Any]:
    path = read_root / REMAP_MANIFEST_NAME
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def _update_frames_group_ids(event_dir: Path, group_id_per_frame: np.ndarray) -> None:
    frames_path = Path(_frames_csv_path(event_dir))
    frames = pd.read_csv(frames_path)
    n = min(len(frames), len(group_id_per_frame))
    if "group_id" not in frames.columns:
        frames["group_id"] = -1
    col = frames.columns.get_loc("group_id")
    if n:
        frames.iloc[:n, col] = np.asarray(group_id_per_frame[:n], dtype=np.int64)
    frames.to_csv(frames_path, index=False)


def _bin_skycell_contrib(
    *,
    assignment: np.ndarray,
    ps1_data: np.ndarray,
    ps1_mask: np.ndarray,
    sx_int: int,
    sy_int: int,
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    ignore_mask: int,
    mapping_grid=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Sparse bin one skycell contribution.

    Field mode must pass ``mapping_grid``; pixels outside the grid (via
    ``MappingGrid.contains_flat``) are dropped before ``argsort``.
    ``roi_bounds`` is retained for call-site compatibility / materialize only.
    """
    from syndiff_pipeline.template_creation.processing.downsample import (
        _aggregate_sorted_groups,
    )

    if assignment.shape != ps1_data.shape:
        raise ValueError(
            f"regmap assignment shape {assignment.shape} != PS1 data shape {ps1_data.shape}"
        )
    if mapping_grid is None:
        raise ValueError(
            "_bin_skycell_contrib requires mapping_grid (MappingGrid.contains_flat); "
            "legacy ROI crop filtering was removed"
        )
    t_y, t_x = base_tess_shape
    del roi_bounds  # grid APIs own membership; bounds kept on signature for callers

    if int(sx_int) == 0 and int(sy_int) == 0:
        ps1_shifted = ps1_data
        mask_shifted = ps1_mask
    else:
        ps1_shifted = np.roll(ps1_data, (int(sy_int), int(sx_int)), axis=(0, 1))
        mask_shifted = np.roll(ps1_mask, (int(sy_int), int(sx_int)), axis=(0, 1))

    pind_full = np.asarray(assignment).ravel()
    ps1_full = np.asarray(ps1_shifted).ravel()
    mask_full = np.asarray(mask_shifted).ravel()

    shape_os = mapping_grid.array_shape_os()
    shape_native = mapping_grid.array_shape_native()
    if (int(t_y), int(t_x)) == shape_os:
        oversampled = True
    elif (int(t_y), int(t_x)) == shape_native:
        oversampled = False
    else:
        oversampled = int(getattr(mapping_grid, "oversampling", 1)) > 1

    keep = np.isfinite(pind_full) & (pind_full >= 0)
    # Vectorized MappingGrid.contains_flat: local flat ids are [0, H*W).
    width = mapping_grid.width_os if oversampled else mapping_grid.width_native
    height = mapping_grid.height_os if oversampled else mapping_grid.height_native
    keep &= pind_full < (int(height) * int(width))
    if not np.any(keep):
        return None

    flat_keep = np.flatnonzero(keep)
    pind = pind_full[flat_keep]
    ps1_rav = ps1_full[flat_keep]
    mask_rav = mask_full[flat_keep]

    sort_ind = np.argsort(pind)
    pind_sorted = pind[sort_ind]
    change_pts = np.where(np.diff(pind_sorted) > 0)[0] + 1
    group_starts = np.concatenate([[0], change_pts]).astype(np.intp)
    tess_pixels = pind_sorted[group_starts].astype(np.int64)
    if len(tess_pixels) == 0:
        return None

    sums, counts, mask_counts = _aggregate_sorted_groups(
        ps1_rav[sort_ind], mask_rav[sort_ind], group_starts, ignore_mask
    )

    return (
        tess_pixels.astype(np.int64),
        sums.astype(np.float64),
        counts.astype(np.float64),
        mask_counts.astype(np.float64),
    )


def _load_zarr_skycell(zstore, skycell: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(zstore[f"{skycell}_data"][:], dtype=np.float32)
    try:
        mask = np.asarray(zstore[f"{skycell}_mask"][:])
    except Exception:
        mask = np.zeros(data.shape, dtype=np.int32)
    return data, mask


def _neighbours_by_skycell_id(
    pair_ids: np.ndarray,
) -> dict[int, list[int]]:
    """Map skycell id → sorted neighbour ids from undirected abutting pairs."""
    from collections import defaultdict

    out: dict[int, list[int]] = defaultdict(list)
    for id_lo, id_hi in np.asarray(pair_ids):
        lo, hi = int(id_lo), int(id_hi)
        out[lo].append(hi)
        out[hi].append(lo)
    return {k: sorted(set(v)) for k, v in out.items()}


def _composite_key_for_group(
    *,
    skycell: str,
    skycell_id: int,
    group_id: int,
    sx_int: int,
    sy_int: int,
    group_shifts: dict[str, tuple[int, int]],
    neighbour_ids: list[int],
    id_to_name: dict[int, str],
    epoch_index: dict[str, Any] | None,
    apply_intra_skycell: bool = True,
    apply_inter_skycell: bool = True,
) -> tuple[Any, ...]:
    """Geometry identity for one (skycell, group): intra epoch + neighbour inter epochs.

    Groups that share this key produce identical hybrid assignment maps (and
    therefore identical binned contribs) for this skycell.

    Own-shift ``(0, 0)`` has no intra Exact epoch (remap skips zeros); the intra
    slot is the sentinel ``"roll0"``. Neighbour inter pair-epochs / shifts are
    still part of the key so groups that differ only on the rim do not merge
    when ``apply_inter_skycell`` is True.

    When ``apply_inter_skycell`` is False, neighbour terms are omitted. When
    ``apply_intra_skycell`` is False, the intra slot is ``"roll0"`` (roll-only).
    """
    from syndiff_pipeline.template_creation.processing.field_remap import (
        resolve_l4a_epoch_id,
        resolve_l4b_pair_epoch_id,
    )

    sx_i, sy_i = int(sx_int), int(sy_int)
    is_zero = sx_i == 0 and sy_i == 0

    if epoch_index is not None:
        if not apply_intra_skycell or is_zero:
            intra_part: Any = "roll0"
        else:
            intra_part = int(
                resolve_l4a_epoch_id(
                    epoch_index,
                    skycell=str(skycell),
                    group_id=int(group_id),
                    sx_int=sx_i,
                    sy_int=sy_i,
                )
            )
        if not apply_inter_skycell:
            return (intra_part,)
        nb_parts: list[tuple[int, int]] = []
        for neighbour_id in neighbour_ids:
            nb_name = id_to_name.get(int(neighbour_id))
            if nb_name is None or nb_name not in group_shifts:
                continue
            sx_nb, sy_nb = group_shifts[nb_name]
            lo = min(int(skycell_id), int(neighbour_id))
            hi = max(int(skycell_id), int(neighbour_id))
            if int(skycell_id) == lo:
                sx_lo, sy_lo, sx_hi, sy_hi = sx_i, sy_i, int(sx_nb), int(sy_nb)
            else:
                sx_lo, sy_lo, sx_hi, sy_hi = int(sx_nb), int(sy_nb), sx_i, sy_i
            pair_epoch = resolve_l4b_pair_epoch_id(
                epoch_index,
                id_lo=lo,
                id_hi=hi,
                group_id=int(group_id),
                sx_lo=sx_lo,
                sy_lo=sy_lo,
                sx_hi=sx_hi,
                sy_hi=sy_hi,
            )
            nb_parts.append((int(neighbour_id), int(pair_epoch)))
        return (intra_part, tuple(sorted(nb_parts)))

    # Legacy (no epoch index): own shift + neighbour shifts identify geometry.
    if not apply_inter_skycell:
        return (sx_i, sy_i)
    nb_parts_legacy: list[tuple[int, int, int]] = []
    for neighbour_id in neighbour_ids:
        nb_name = id_to_name.get(int(neighbour_id))
        if nb_name is None or nb_name not in group_shifts:
            continue
        sx_nb, sy_nb = group_shifts[nb_name]
        nb_parts_legacy.append((int(neighbour_id), int(sx_nb), int(sy_nb)))
    return (sx_i, sy_i, tuple(sorted(nb_parts_legacy)))


def _contrib_has_indices(path: Path) -> bool:
    """True if a contrib NPZ exists and has at least one sparse index."""
    from syndiff_pipeline.template_creation.processing.field_templates import (
        load_contrib,
    )

    if not path.is_file():
        return False
    data = load_contrib(path)
    return len(np.asarray(data["indices"])) > 0


def _any_nonempty_contrib(
    store: Path,
    key_list: list[tuple[int, str, int, int]],
) -> bool:
    """Scan contrib keys until one nonempty file is found.

    Uses a stride across the full list first (avoids early-skycell bias on
    sorted ``key_list``), then fills remaining indices. Stops at the first hit.
    """
    n = len(key_list)
    if n == 0:
        return False
    step = max(1, n // 128)
    order = list(range(0, n, step))
    seen = set(order)
    order.extend(i for i in range(n) if i not in seen)
    for i in order:
        gid_i, skycell, sx_i, sy_i = key_list[i]
        p = contrib_path(store, skycell, sx_i, sy_i, group_id=int(gid_i))
        if _contrib_has_indices(p):
            return True
    return False

def _build_skycell_composite_index(
    *,
    key_list: list[tuple[int, str, int, int]],
    group_shifts_by_gid: dict[int, dict[str, tuple[int, int]]],
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
    neighbours_by_id: dict[int, list[int]],
    epoch_index: dict[str, Any] | None,
    apply_intra_skycell: bool = True,
    apply_inter_skycell: bool = True,
) -> dict[str, dict[tuple[Any, ...], list[tuple[int, int, int]]]]:
    """Per skycell: composite_key → list of (group_id, sx, sy)."""
    from collections import defaultdict

    by_skycell: dict[str, dict[tuple[Any, ...], list[tuple[int, int, int]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for gid, skycell, sx_i, sy_i in key_list:
        skycell_id = name_to_id.get(str(skycell))
        if skycell_id is None:
            raise KeyError(f"skycell {skycell!r} missing from master id map")
        ckey = _composite_key_for_group(
            skycell=str(skycell),
            skycell_id=int(skycell_id),
            group_id=int(gid),
            sx_int=int(sx_i),
            sy_int=int(sy_i),
            group_shifts=group_shifts_by_gid[int(gid)],
            neighbour_ids=neighbours_by_id.get(int(skycell_id), []),
            id_to_name=id_to_name,
            epoch_index=epoch_index,
            apply_intra_skycell=bool(apply_intra_skycell),
            apply_inter_skycell=bool(apply_inter_skycell),
        )
        by_skycell[str(skycell)][ckey].append((int(gid), int(sx_i), int(sy_i)))
    return {sc: dict(buckets) for sc, buckets in by_skycell.items()}


# Per-process caches for field L5 skycell-batch workers (loky reuses processes).
_L5_WORKER: dict[str, Any] = {}


def _init_l5_worker(payload: dict[str, Any]) -> None:
    global _L5_WORKER
    _L5_WORKER = dict(payload)
    # Preserve an injected zstore (tests); otherwise open lazily in the batch.


def _ensure_l5_worker(payload: dict[str, Any]) -> None:
    if not _L5_WORKER:
        _init_l5_worker(payload)


def _reset_l5_worker() -> None:
    global _L5_WORKER
    _L5_WORKER = {}


def _read_regmap_assignment_l5(skycell: str) -> np.ndarray:
    scratch = _L5_WORKER.get("scratch_regmaps") or {}
    regmap_path = scratch.get(skycell)
    if regmap_path is None:
        regmap_path = str(
            _find_regmap(
                Path(_L5_WORKER["mapping_root"]),
                int(_L5_WORKER["sector"]),
                int(_L5_WORKER["camera"]),
                int(_L5_WORKER["ccd"]),
                skycell,
                oversampling_factor=int(_L5_WORKER["oversampling_factor"]),
            )
        )
    with fits.open(regmap_path) as hdul:
        if "TESS_PIXEL_MAP" in hdul:
            return np.asarray(hdul["TESS_PIXEL_MAP"].data)
        return np.asarray(hdul[1].data)


def _l5_skycell_batch(
    skycell: str,
    buckets: dict[tuple[Any, ...], list[tuple[int, int, int]]],
) -> dict[str, Any]:
    """Load regmap+zarr once; compose+bin once per composite key; fan-out writes."""
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        compose_group_hybrid_assignment,
        shared_abutting_border_tess_ids,
    )

    if not _L5_WORKER:
        raise RuntimeError("L5 worker not initialized; call _init_l5_worker first")
    store = Path(_L5_WORKER["store"])
    rebuild = bool(_L5_WORKER["rebuild_field_store"])
    name_to_id = _L5_WORKER["name_to_id"]
    id_to_name = _L5_WORKER["id_to_name"]
    master_map = _L5_WORKER["master_map"]
    pair_ids = _L5_WORKER["pair_ids"]
    neighbours_by_id = _L5_WORKER["neighbours_by_id"]
    group_shifts_by_gid = _L5_WORKER["group_shifts_by_gid"]
    epoch_index = _L5_WORKER["epoch_index"]
    exact_cache_l4a_dir = Path(_L5_WORKER["exact_cache_l4a_dir"])
    exact_cache_l4b_dir = Path(_L5_WORKER["exact_cache_l4b_dir"])
    base_tess_shape = tuple(_L5_WORKER["base_tess_shape"])
    roi_bounds = tuple(_L5_WORKER["roi_bounds"])
    ignore_mask = int(_L5_WORKER["ignore_mask"])
    intra_skycell_R = int(_L5_WORKER["intra_skycell_R"])
    apply_intra_skycell = bool(_L5_WORKER.get("apply_intra_skycell", True))
    apply_inter_skycell = bool(_L5_WORKER.get("apply_inter_skycell", True))

    skycell_id = int(name_to_id[str(skycell)])
    neighbour_ids = list(neighbours_by_id.get(skycell_id, []))
    border_ids_by_neighbour = {
        int(nb): shared_abutting_border_tess_ids(master_map, skycell_id, int(nb))[0]
        for nb in neighbour_ids
    }

    n_writes = 0
    n_skips = 0
    n_compose = 0
    n_nonempty = 0
    n_regmap_opens = 0
    n_zarr_loads = 0

    # Skip-only short circuit: if every fan-out target exists, avoid IO.
    pending: dict[tuple[Any, ...], list[tuple[int, int, int]]] = {}
    for ckey, gid_list in buckets.items():
        need: list[tuple[int, int, int]] = []
        for gid, sx_i, sy_i in gid_list:
            out = contrib_path(store, skycell, sx_i, sy_i, group_id=int(gid))
            if out.is_file() and not rebuild:
                n_skips += 1
                continue
            if out.is_file() and rebuild:
                out.unlink()
            need.append((int(gid), int(sx_i), int(sy_i)))
        if need:
            pending[ckey] = need
        else:
            # All skipped — still count the composite key as "done" for progress.
            pass

    if not pending:
        # All keys skipped: probe existing contribs so resume does not report
        # nonempty=0 when early sorted keys happen to be empty.
        for ckey, gid_list in buckets.items():
            gid, sx_i, sy_i = gid_list[0]
            p = contrib_path(store, skycell, sx_i, sy_i, group_id=int(gid))
            if _contrib_has_indices(p):
                n_nonempty += len(gid_list)
                break
        return {
            "skycell": skycell,
            "n_composite_keys": len(buckets),
            "n_compose": 0,
            "n_writes": n_writes,
            "n_skips": n_skips,
            "n_nonempty": n_nonempty,
            "n_regmap_opens": 0,
            "n_zarr_loads": 0,
        }

    assignment_map = _read_regmap_assignment_l5(skycell)
    n_regmap_opens = 1
    zstore = _L5_WORKER.get("zstore")
    if zstore is None:
        import zarr

        zstore = zarr.open(str(_L5_WORKER["zarr_path"]), mode="r")
        _L5_WORKER["zstore"] = zstore
    ps1_data, ps1_mask = _load_zarr_skycell(zstore, skycell)
    n_zarr_loads = 1

    for ckey, gid_list in pending.items():
        # Representative group for compose (any member shares the composite key).
        rep_gid, sx_i, sy_i = gid_list[0]
        group_shifts = group_shifts_by_gid[int(rep_gid)]
        is_zero = int(sx_i) == 0 and int(sy_i) == 0

        # (0,0): no intra Exact cache (remap skips zero epochs). Still compose so
        # inter-skycell rim patches apply when neighbours differ.
        require_intra = bool(apply_intra_skycell) and not is_zero
        if is_zero or not apply_intra_skycell:
            l4a_cache_path = exact_cache_l4a_dir / "_unused_roll0_exact.npz"
        else:
            cache_name = contrib_basename(skycell, sx_i, sy_i).replace(
                ".npz", "_exact.npz"
            )
            l4a_cache_path = exact_cache_l4a_dir / cache_name
            if epoch_index is not None:
                epoch_id = resolve_l4a_epoch_id(
                    epoch_index,
                    skycell=str(skycell),
                    group_id=int(rep_gid),
                    sx_int=int(sx_i),
                    sy_int=int(sy_i),
                )
                l4a_cache_path = l4a_exact_path(
                    exact_cache_l4a_dir, str(skycell), epoch_id, int(sx_i), int(sy_i)
                )
        hybrid_map, _meta = compose_group_hybrid_assignment(
            assignment_map,
            skycell=str(skycell),
            skycell_id=skycell_id,
            sx_int=int(sx_i),
            sy_int=int(sy_i),
            master=master_map,
            group_shifts=group_shifts,
            name_to_id=name_to_id,
            l4a_cache_path=l4a_cache_path,
            l4b_cache_dir=exact_cache_l4b_dir,
            group_id=int(rep_gid),
            epoch_index=epoch_index,
            hybrid_R=intra_skycell_R,
            apply_intra_skycell=bool(apply_intra_skycell),
            apply_inter_skycell=bool(apply_inter_skycell),
            require_intra_skycell_cache=require_intra,
            require_inter_skycell_cache=bool(apply_inter_skycell),
            pair_ids=pair_ids,
            id_to_name=id_to_name,
            neighbour_ids=neighbour_ids,
            border_ids_by_neighbour=border_ids_by_neighbour,
        )
        n_compose += 1
        binned = _bin_skycell_contrib(
            assignment=hybrid_map,
            ps1_data=ps1_data,
            ps1_mask=ps1_mask,
            sx_int=0,
            sy_int=0,
            base_tess_shape=base_tess_shape,
            roi_bounds=roi_bounds,
            ignore_mask=ignore_mask,
            mapping_grid=_L5_WORKER.get("mapping_grid"),
        )

        if binned is None:
            idx = np.array([], dtype=np.int64)
            sums = np.array([], dtype=np.float64)
            counts = np.array([], dtype=np.float64)
            mcounts = np.array([], dtype=np.float64)
        else:
            idx, sums, counts, mcounts = binned
            if len(idx) > 0:
                n_nonempty += len(gid_list)

        for gid, sx_g, sy_g in gid_list:
            write_contrib(
                store,
                skycell,
                sx_g,
                sy_g,
                indices=idx,
                flux_sum=sums,
                count=counts,
                mask_count=mcounts,
                group_id=int(gid),
            )
            n_writes += 1

    return {
        "skycell": skycell,
        "n_composite_keys": len(buckets),
        "n_compose": n_compose,
        "n_writes": n_writes,
        "n_skips": n_skips,
        "n_nonempty": n_nonempty,
        "n_regmap_opens": n_regmap_opens,
        "n_zarr_loads": n_zarr_loads,
    }


def run_field_downsample_scc(
    *,
    sector: int,
    camera: int,
    ccd: int,
    data_root: str | Path,
    event_dir: str | Path,
    mapping_root: str | Path,
    convolved_dir: str | Path,
    roi_bounds: tuple[int, int, int, int],
    base_tess_shape: tuple[int, int],
    oversampling_factor: int = 1,
    ignore_mask_bits: list[int] | None = None,
    grouping_quantum_ps1_px: float = 1.0,
    cache_quantum_ps1_px: float = 1.0,
    keying: str = "absolute",
    materialize_fits: bool = False,
    n_jobs: int = 1,
    update_frames_csv: bool = True,
    store_root: str | Path | None = None,
    remap_store_root: str | Path | None = None,
    rebuild_field_store: bool = False,
    stage_regmaps_to_scratch: bool | None = None,
    scc_only: bool = False,
    ffi_dir: str | Path | None = None,
    ref_ffi_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    apply_intra_skycell: bool = True,
    apply_inter_skycell: bool = True,
    mapping_grid=None,
) -> dict[str, Any]:
    """
    Bin sparse contribs into the SCC templates store (L5 only).

    Requires remap artifacts under ``remap/oversampling_{N}/``. Reads
    intra-skycell and/or inter-skycell exact caches (controlled by
    ``apply_intra_skycell`` / ``apply_inter_skycell``) and always writes
    group-qualified contribs (``_gid{N}``).

    Skycell-major dispatch with composite-key fan-out: regmap/zarr load once
    per skycell; compose+bin once per distinct geometry; write one NPZ per
    ``group_id`` (identical content when groups share geometry).

    Parameters ``ffi_dir`` and ``ref_ffi_path`` are accepted for dispatch
    compatibility but ignored (remap must run separately).
    """
    import os as _os
    import time as _time
    import zarr
    from joblib import delayed

    from syndiff_pipeline.common.joblib_progress import parallel_map_with_optional_tqdm
    from syndiff_pipeline.template_creation.processing import field_downsample_progress
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
    )

    del ffi_dir, ref_ffi_path  # remap stage owns schedule build inputs
    if mapping_grid is None:
        from syndiff_pipeline.common.mapping_grid import MappingGridError

        raise MappingGridError(
            "field downsample requires MappingGrid (MAPGRID>=2 / field_mode_assembly "
            "schema_version>=3); rebuild mapping then re-run downsample"
        )
    progress_file = Path(progress_path) if progress_path is not None else None
    t_run0 = _time.perf_counter()
    if progress_file is not None:
        field_downsample_progress.init_field_setup_progress(progress_file)

    event_dir = Path(event_dir)
    data_root = Path(data_root)
    mapping_root = Path(mapping_root)
    store = Path(store_root) if store_root is not None else templates_root(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    remap_store = (
        Path(remap_store_root)
        if remap_store_root is not None
        else remap_root(data_root, sector, camera, ccd, oversampling_factor=oversampling_factor)
    )
    store.mkdir(parents=True, exist_ok=True)
    (store / "contribs").mkdir(exist_ok=True)

    remap_read, _legacy = resolve_remap_read_root(remap_store, store)
    remap_manifest = _load_remap_manifest(remap_read)
    intra_skycell_R = 1
    if remap_manifest:
        intra_skycell_R = int(
            remap_manifest.get(
                "intra_skycell_R",
                remap_manifest.get("hybrid_R", intra_skycell_R),
            )
        )
        cache_quantum_ps1_px = float(
            remap_manifest.get("cache_quantum_ps1_px", cache_quantum_ps1_px)
        )
        keying = str(remap_manifest.get("keying", keying))
    group_scoped_contribs = True

    schedule_path = remap_read / "shift_schedule.npz"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"shift schedule missing under remap read root {remap_read}")
    t_m = _time.perf_counter()
    schedule = ShiftSchedule.load(schedule_path)
    log.info("Loaded shift schedule from %s in %.1fs", schedule_path, _time.perf_counter() - t_m)
    t_m = _time.perf_counter()
    shifts_df = load_remap_shifts_df(remap_read)
    log.info(
        "Loaded template_group_shifts (%d rows, %d groups) in %.1fs",
        len(shifts_df),
        int(shifts_df["group_id"].nunique()) if not shifts_df.empty else 0,
        _time.perf_counter() - t_m,
    )
    t_m = _time.perf_counter()
    assignment = assign_groups_from_schedule(
        schedule,
        grouping_quantum_ps1_px=grouping_quantum_ps1_px,
        cache_quantum_ps1_px=cache_quantum_ps1_px,
        keying=keying,
    )
    log.info(
        "Assigned groups from schedule (%d frames) in %.1fs",
        len(assignment.group_id_per_frame),
        _time.perf_counter() - t_m,
    )
    gid_npy = remap_read / GROUP_ID_PER_FRAME_NPY
    if gid_npy.is_file():
        loaded_gids = np.load(gid_npy)
        if loaded_gids.shape != assignment.group_id_per_frame.shape:
            raise RuntimeError(
                f"group_id_per_frame.npy shape {loaded_gids.shape} != "
                f"schedule-derived {assignment.group_id_per_frame.shape}; rebuild remap"
            )
        if not np.array_equal(loaded_gids, assignment.group_id_per_frame):
            raise RuntimeError(
                "group_id_per_frame.npy disagrees with assign_groups_from_schedule; "
                "rebuild remap or downsample with matching code"
            )
    epoch_index = None
    epoch_index_path = remap_read / GID_EPOCH_INDEX_NPZ
    if epoch_index_path.is_file():
        t_m = _time.perf_counter()
        epoch_index = load_gid_epoch_index(
            epoch_index_path,
            include_inter=bool(apply_inter_skycell),
        )
        log.info(
            "Loaded gid_epoch_index (%d intra, %d inter keys; include_inter=%s) in %.1fs",
            len(epoch_index.get("l4a", {})),
            len(epoch_index.get("l4b", {})),
            bool(apply_inter_skycell),
            _time.perf_counter() - t_m,
        )
    if not scc_only:
        write_group_artifacts(
            assignment,
            event_dir,
            geometry_mode="field",
            grouping_quantum_ps1_px=grouping_quantum_ps1_px,
            cache_quantum_ps1_px=cache_quantum_ps1_px,
        )
    if update_frames_csv and not scc_only:
        _update_frames_group_ids(event_dir, assignment.group_id_per_frame)

    exact_cache_l4a_dir = exact_cache_dir_for_read_root(remap_read)
    exact_cache_l4b_dir = exact_cache_l4b_dir_for_read_root(remap_read)

    ignore_mask = 0
    for bit in ignore_mask_bits or [12]:
        ignore_mask |= 1 << int(bit)

    zarr_path = Path(convolved_dir)
    if zarr_path.suffix != ".zarr" or not zarr_path.name.endswith(".zarr"):
        zarr_path = zarr_path / f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"
    if not zarr_path.exists():
        alt = list(Path(convolved_dir).glob(f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}*.zarr"))
        if not alt:
            from syndiff_pipeline.common.scc_paths import scc_convolved_zarr

            zarr_path = scc_convolved_zarr(data_root, sector, camera, ccd)
            if not zarr_path.exists():
                raise FileNotFoundError(f"convolved zarr not found: {zarr_path}")
        else:
            zarr_path = alt[0]
    zarr.open(str(zarr_path), mode="r")

    master_path = _master_pixels2skycells_path(
        mapping_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
    )
    t_m = _time.perf_counter()
    master_map, name_to_id = _master_skycell_id_map(master_path)
    log.info(
        "Opened zarr %s and master map %s (%d skycells) in %.1fs",
        zarr_path,
        master_path,
        len(name_to_id),
        _time.perf_counter() - t_m,
    )

    keys = {
        (str(r.skycell), int(r.sx_int), int(r.sy_int))
        for r in shifts_df.itertuples(index=False)
    }
    group_shifts_by_gid: dict[int, dict[str, tuple[int, int]]] = {}
    for gid in sorted(shifts_df["group_id"].unique()):
        rows = shifts_df.loc[shifts_df["group_id"] == int(gid)]
        group_shifts_by_gid[int(gid)] = {
            str(r.skycell): (int(r.sx_int), int(r.sy_int))
            for r in rows.itertuples(index=False)
        }

    from syndiff_pipeline.template_creation.processing.downsample import (
        resolve_downsample_scratch_dir,
        resolve_stage_regmaps_to_scratch,
        stage_regmap_files_to_scratch,
    )

    scratch_regmaps: dict[str, str] = {}
    do_stage = resolve_stage_regmaps_to_scratch(stage_regmaps_to_scratch)
    if not do_stage:
        log.info("Regmap scratch staging disabled; reading regmaps from NFS")
    elif do_stage:
        sky_reg: list[tuple[str, str]] = []
        for sc in sorted({k[0] for k in keys}):
            try:
                sky_reg.append(
                    (
                        sc,
                        str(
                            _find_regmap(
                                mapping_root,
                                sector,
                                camera,
                                ccd,
                                sc,
                                oversampling_factor=oversampling_factor,
                            )
                        ),
                    )
                )
            except FileNotFoundError:
                continue
        if sky_reg:
            log.info(
                "Staging %d skycell regmaps to Condor scratch (may take minutes)...",
                len(sky_reg),
            )
            try:
                local_paths, scratch_dir, n_staged, elapsed = stage_regmap_files_to_scratch(
                    [p for _, p in sky_reg],
                    sector=sector,
                    camera=camera,
                    ccd=ccd,
                    oversampling_factor=oversampling_factor,
                )
                scratch_regmaps = {sc: lp for (sc, _), lp in zip(sky_reg, local_paths)}
                log.info(
                    "Staged %d/%d ROI regmaps to scratch %s in %.1fs",
                    n_staged,
                    len(sky_reg),
                    scratch_dir,
                    elapsed,
                )
            except OSError as exc:
                if getattr(exc, "errno", None) != errno.ENOSPC:
                    raise
                os_suffix = f"_os{oversampling_factor}" if oversampling_factor > 1 else ""
                scratch_dir = (
                    resolve_downsample_scratch_dir()
                    / f"syndiff_downsample_regmaps_{sector:04d}_{camera}_{ccd}{os_suffix}"
                )
                if scratch_dir.is_dir():
                    shutil.rmtree(scratch_dir, ignore_errors=True)
                scratch_regmaps = {}
                log.warning(
                    "Scratch staging hit ENOSPC (%s); continuing with NFS regmap paths",
                    exc,
                )

    key_list = sorted(
        (
            int(r.group_id),
            str(r.skycell),
            int(r.sx_int),
            int(r.sy_int),
        )
        for r in shifts_df.itertuples(index=False)
        if (str(r.skycell), int(r.sx_int), int(r.sy_int)) in keys
    )

    pair_ids = abutting_undirected_pairs(master_map)
    id_to_name = {int(nid): str(name) for name, nid in name_to_id.items()}
    neighbours_by_id = _neighbours_by_skycell_id(pair_ids)
    t_m = _time.perf_counter()
    composite_index = _build_skycell_composite_index(
        key_list=key_list,
        group_shifts_by_gid=group_shifts_by_gid,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        neighbours_by_id=neighbours_by_id,
        epoch_index=epoch_index,
        apply_intra_skycell=bool(apply_intra_skycell),
        apply_inter_skycell=bool(apply_inter_skycell),
    )
    skycell_batches = sorted(composite_index.items())
    n_composite_keys = sum(len(buckets) for _, buckets in skycell_batches)
    log.info(
        "Built composite-key index: %d skycells, %d composite keys, %d contrib keys "
        "(apply_intra=%s apply_inter=%s) in %.1fs",
        len(skycell_batches),
        n_composite_keys,
        len(key_list),
        bool(apply_intra_skycell),
        bool(apply_inter_skycell),
        _time.perf_counter() - t_m,
    )

    worker_payload = {
        "store": str(store),
        "rebuild_field_store": bool(rebuild_field_store),
        "mapping_root": str(mapping_root),
        "sector": int(sector),
        "camera": int(camera),
        "ccd": int(ccd),
        "oversampling_factor": int(oversampling_factor),
        "scratch_regmaps": dict(scratch_regmaps),
        "zarr_path": str(zarr_path),
        "name_to_id": dict(name_to_id),
        "id_to_name": dict(id_to_name),
        "master_map": master_map,
        "pair_ids": pair_ids,
        "neighbours_by_id": neighbours_by_id,
        "group_shifts_by_gid": group_shifts_by_gid,
        "epoch_index": epoch_index,
        "exact_cache_l4a_dir": str(exact_cache_l4a_dir),
        "exact_cache_l4b_dir": str(exact_cache_l4b_dir),
        "base_tess_shape": tuple(base_tess_shape),
        "roi_bounds": tuple(roi_bounds),
        "ignore_mask": int(ignore_mask),
        "intra_skycell_R": int(intra_skycell_R),
        "apply_intra_skycell": bool(apply_intra_skycell),
        "apply_inter_skycell": bool(apply_inter_skycell),
        "mapping_grid": mapping_grid,
    }

    if progress_file is not None:
        field_downsample_progress.init_field_progress(
            progress_file,
            n_skycells=len(skycell_batches),
            n_composite_keys=n_composite_keys,
            n_contrib_keys=len(key_list),
            oversampling_factor=int(oversampling_factor),
        )

    n_jobs_eff = max(1, min(int(n_jobs), len(skycell_batches) or 1))
    hybrid_cap = int(_os.environ.get("SYNDIFF_HYBRID_MAX_JOBS", "24"))
    avail = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (
        _os.cpu_count() or hybrid_cap
    )
    n_jobs_eff = min(n_jobs_eff, max(1, hybrid_cap), max(1, avail))
    log.info(
        "Starting field L5 workers: n_jobs_eff=%d skycell_batches=%d setup_elapsed=%.1fs",
        n_jobs_eff,
        len(skycell_batches),
        _time.perf_counter() - t_run0,
    )
    def _on_batch_result(result: dict[str, Any]) -> None:
        if progress_file is None:
            return
        field_downsample_progress.mark_skycell_batch_done(
            progress_file,
            n_composite_keys=int(result.get("n_composite_keys", 0)),
            n_writes=int(result.get("n_writes", 0)),
            n_skips=int(result.get("n_skips", 0)),
        )

    _reset_l5_worker()
    t_bin0 = _time.perf_counter()
    if n_jobs_eff == 1 or len(skycell_batches) <= 1:
        _init_l5_worker(worker_payload)
        batch_results = []
        for skycell, buckets in skycell_batches:
            result = _l5_skycell_batch(skycell, buckets)
            _on_batch_result(result)
            batch_results.append(result)
    else:
        batch_results = parallel_map_with_optional_tqdm(
            (
                delayed(_l5_skycell_batch)(skycell, buckets)
                for skycell, buckets in skycell_batches
            ),
            n_tasks=len(skycell_batches),
            desc="field L5 skycells",
            n_jobs_eff=n_jobs_eff,
            initializer=_init_l5_worker,
            initargs=(worker_payload,),
            on_result=_on_batch_result,
            prefer="processes",
        )
    t_bin = _time.perf_counter() - t_bin0
    _reset_l5_worker()

    n_written = sum(int(r.get("n_writes", 0)) for r in batch_results)
    n_skipped = sum(int(r.get("n_skips", 0)) for r in batch_results)
    n_compose = sum(int(r.get("n_compose", 0)) for r in batch_results)
    nonempty = sum(int(r.get("n_nonempty", 0)) for r in batch_results)
    n_regmap_opens = sum(int(r.get("n_regmap_opens", 0)) for r in batch_results)
    n_zarr_loads = sum(int(r.get("n_zarr_loads", 0)) for r in batch_results)

    if nonempty == 0 and len(key_list) > 0 and (n_written + n_skipped) > 0:
        if n_written > 0:
            # This run wrote only empty contribs.
            raise RuntimeError(
                f"field store has {len(key_list)} contrib keys but all written "
                f"contribs are empty (regmap/ROI mismatch?)"
            )
        # All-skip resume: stride across the full key list (not just the first
        # 8 sorted rows) until a nonempty contrib is found.
        if not _any_nonempty_contrib(store, key_list):
            raise RuntimeError(
                f"field store has {len(key_list)} contrib keys but all "
                f"contribs are empty (regmap/ROI mismatch?)"
            )
        nonempty = 1

    perf_meta = {
        "n_skycells": len(skycell_batches),
        "n_composite_keys": int(n_composite_keys),
        "n_contrib_keys": len(key_list),
        "n_compose": int(n_compose),
        "n_regmap_opens": int(n_regmap_opens),
        "n_zarr_loads": int(n_zarr_loads),
        "n_jobs_eff": int(n_jobs_eff),
        "bin_elapsed_s": round(float(t_bin), 3),
        "total_elapsed_s": round(float(_time.perf_counter() - t_run0), 3),
        "composite_reuse_ratio": (
            round(len(key_list) / n_composite_keys, 3) if n_composite_keys else None
        ),
    }
    if progress_file is not None:
        field_downsample_progress.set_perf_metadata(progress_file, **perf_meta)

    write_template_manifest(
        store,
        FieldManifest(
            geometry_mode="field",
            scope="scc",
            assembly="sparse_sum",
            materialize_fits=bool(materialize_fits),
            sector=int(sector),
            camera=int(camera),
            ccd=int(ccd),
            contribs_dir="contribs",
            groups=list(assignment.groups),
        ),
    )
    sidecar = {
        "schema_version": 3,
        "store_root": str(store),
        "remap_root": str(remap_store),
        "output_store_name": None,
        "remap_store_name": None,
        "zarr_path": str(zarr_path),
        "base_tess_shape": list(base_tess_shape),
        "oversampling_factor": int(oversampling_factor),
        "ignore_mask": int(ignore_mask),
        "intra_skycell_R": int(intra_skycell_R),
        "apply_intra_skycell": bool(apply_intra_skycell),
        "apply_inter_skycell": bool(apply_inter_skycell),
        "group_scoped_contribs": True,
        "materialize_fits": bool(materialize_fits),
        "architecture_note": (
            "Group-qualified contribs (_gid{N}); skycell-major composite-key fan-out; "
            f"apply_intra_skycell={bool(apply_intra_skycell)}, "
            f"apply_inter_skycell={bool(apply_inter_skycell)}"
        ),
        "flux_note": (
            "Field contribs are in convolved/PS1 flux units; Hotpants may need "
            "a per-event flux scale vs linear ADU templates (~1e3–1e4)."
        ),
        "perf": perf_meta,
        "mapping_grid": mapping_grid.to_mapping_dict(),
        "geometry_mode": "field",
    }
    # Infer lane names from path leaves for provenance / A/B bookkeeping.
    from syndiff_pipeline.common.scc_paths import REMAP_SUBDIR, TEMPLATES_SUBDIR

    def _lane_from_path(path: Path, base: str) -> str | None:
        leaf = path.name
        if leaf.startswith("oversampling_"):
            leaf = path.parent.name
        prefix = f"{base}_"
        if leaf == base:
            return None
        if leaf.startswith(prefix):
            return leaf[len(prefix) :]
        return None

    sidecar["output_store_name"] = _lane_from_path(store, TEMPLATES_SUBDIR)
    sidecar["remap_store_name"] = _lane_from_path(Path(remap_store), REMAP_SUBDIR)
    fits_provenance = {
        "intra_skycell_R": int(intra_skycell_R),
        "apply_intra_skycell": bool(apply_intra_skycell),
        "apply_inter_skycell": bool(apply_inter_skycell),
        "group_scoped_contribs": True,
        "n_intra_skycell_keys": remap_manifest.get("n_intra_skycell_keys")
        or remap_manifest.get("n_exact_keys"),
        "n_inter_skycell_pair_states": remap_manifest.get("n_inter_skycell_pair_states")
        or remap_manifest.get("n_l4b_pair_states", 0),
        "exact_cache_l4a_dir": str(exact_cache_l4a_dir),
        "exact_cache_l4b_dir": str(exact_cache_l4b_dir),
        "remap_root": str(remap_store),
    }
    materialized_fits: dict[str, Any] | None = None
    if materialize_fits:
        materialized_fits = materialize_field_fits_for_store(
            store,
            shifts_df,
            sector=int(sector),
            camera=int(camera),
            ccd=int(ccd),
            base_tess_shape=base_tess_shape,
            roi_bounds=roi_bounds,
            oversampling_factor=int(oversampling_factor),
            group_scoped_contribs=True,
            provenance=fits_provenance,
            mapping_grid=mapping_grid,
        )
        sidecar["materialized_fits"] = materialized_fits
    (store / "field_mode_assembly.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    if progress_file is not None:
        field_downsample_progress.set_progress_phase(progress_file, "complete")

    if not scc_only:
        event_dir.mkdir(parents=True, exist_ok=True)
        serialized_keys = [
            [int(gid), str(s), int(x), int(y)] for gid, s, x, y in key_list
        ]
        (event_dir / "field_contrib_keys.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "store_root": str(store),
                    "remap_root": str(remap_store),
                    "n_contrib_keys": len(key_list),
                    "n_composite_keys": int(n_composite_keys),
                    "keys": serialized_keys,
                }
            )
            + "\n"
        )

    assembly_path = store / "field_mode_assembly.json"
    manifest_path = store / MANIFEST_NAME
    artifacts = [str(assembly_path), str(manifest_path)]
    return {
        "output_dir": str(store),
        "remap_root": str(remap_store),
        "n_groups": len(assignment.groups),
        "n_contrib_keys": len(key_list),
        "n_composite_keys": int(n_composite_keys),
        "n_contribs_written": n_written,
        "n_contribs_skipped": n_skipped,
        "n_compose": int(n_compose),
        "geometry_mode": "field",
        "intra_skycell_R": int(intra_skycell_R),
        "group_scoped_contribs": True,
        "rebuild_field_store": bool(rebuild_field_store),
        "materialize_fits": bool(materialize_fits),
        "materialized_fits": materialized_fits,
        "perf": perf_meta,
        "artifacts": artifacts,
        "expected_count": len(artifacts),
        "produced_count": int(assembly_path.is_file()) + int(manifest_path.is_file()),
    }


def _group_shifts_present(
    store_root: str | Path,
    shifts_df: pd.DataFrame,
    group_id: int,
    *,
    present_only: bool,
    group_scoped_contribs: bool | None = None,
) -> list[tuple[str, int, int]]:
    rows = shifts_df.loc[shifts_df["group_id"] == int(group_id)]
    if rows.empty:
        raise KeyError(f"group_id={group_id} not in template_group_shifts")
    shifts = [
        (str(r.skycell), int(r.sx_int), int(r.sy_int))
        for r in rows.itertuples(index=False)
    ]
    if group_scoped_contribs is None:
        group_scoped_contribs = _store_uses_group_scoped_contribs(store_root)
    if present_only:
        shifts = [
            (s, x, y)
            for (s, x, y) in shifts
            if contrib_path(
                store_root,
                s,
                x,
                y,
                group_id=int(group_id) if group_scoped_contribs else None,
            ).is_file()
        ]
        if not shifts:
            raise FileNotFoundError(
                f"No materialized contribs for group_id={group_id} under {store_root}"
            )
    return shifts


def _store_uses_group_scoped_contribs(store_root: str | Path) -> bool:
    """Field mode always uses group-qualified contrib keys."""
    sidecar_path = Path(store_root) / "field_mode_assembly.json"
    if sidecar_path.is_file():
        try:
            payload = json.loads(sidecar_path.read_text())
            if "group_scoped_contribs" in payload:
                return bool(payload["group_scoped_contribs"])
            # Legacy sidecars without the flag may have unqualified contribs.
            if payload.get("l4b_policy") == "none":
                return False
        except Exception:
            pass
    return True


def assemble_field_group_flux(
    store_root: str | Path,
    shifts_df: pd.DataFrame,
    group_id: int,
    *,
    shape: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    present_only: bool | None = None,
    group_scoped_contribs: bool | None = None,
) -> np.ndarray:
    if present_only is None:
        present_only = crop is not None
    if group_scoped_contribs is None:
        group_scoped_contribs = _store_uses_group_scoped_contribs(store_root)
    shifts = _group_shifts_present(
        store_root,
        shifts_df,
        group_id,
        present_only=present_only,
        group_scoped_contribs=group_scoped_contribs,
    )
    out = assemble_group_from_contribs(
        store_root,
        shifts,
        shape=shape,
        crop=crop,
        group_id=int(group_id) if group_scoped_contribs else None,
    )
    flux = out["flux_sum"]
    count = out["count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, flux / count, 0.0)
    return mean.astype(np.float64)


def assemble_field_group_count(
    store_root: str | Path,
    shifts_df: pd.DataFrame,
    group_id: int,
    *,
    shape: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    present_only: bool | None = None,
    group_scoped_contribs: bool | None = None,
) -> np.ndarray:
    if present_only is None:
        present_only = crop is not None
    if group_scoped_contribs is None:
        group_scoped_contribs = _store_uses_group_scoped_contribs(store_root)
    shifts = _group_shifts_present(
        store_root,
        shifts_df,
        group_id,
        present_only=present_only,
        group_scoped_contribs=group_scoped_contribs,
    )
    out = assemble_group_from_contribs(
        store_root,
        shifts,
        shape=shape,
        crop=crop,
        group_id=int(group_id) if group_scoped_contribs else None,
    )
    return np.asarray(out["count"], dtype=np.float64)


def materialize_field_fits_for_store(
    store_root: str | Path,
    shifts_df: pd.DataFrame,
    *,
    sector: int,
    camera: int,
    ccd: int,
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int] | None = None,
    oversampling_factor: int = 1,
    group_scoped_contribs: bool | None = None,
    provenance: dict[str, Any] | None = None,
    mapping_grid=None,
) -> dict[str, Any]:
    """Write per-group field template FITS from hybrid-binned sparse contribs.

    Uses the same assemble helpers as the lazy template loader. Never re-bins
    from frozen/roll-only maps at FITS time.
    """
    root = Path(store_root)
    if group_scoped_contribs is None:
        group_scoped_contribs = _store_uses_group_scoped_contribs(root)
    crop = _roi_bounds_to_assemble_crop(roi_bounds)
    group_ids = sorted(int(g) for g in shifts_df["group_id"].unique())
    if not group_ids:
        raise RuntimeError(f"no group_id values in template_group_shifts under {root}")

    fits_dir = root / FITS_DIRNAME
    fits_dir.mkdir(parents=True, exist_ok=True)
    prov = dict(provenance or {})
    written: list[dict[str, Any]] = []

    for gid in group_ids:
        flux = assemble_field_group_flux(
            root,
            shifts_df,
            gid,
            shape=base_tess_shape,
            crop=crop,
            present_only=False,
            group_scoped_contribs=group_scoped_contribs,
        )
        count = assemble_field_group_count(
            root,
            shifts_df,
            gid,
            shape=base_tess_shape,
            crop=crop,
            present_only=False,
            group_scoped_contribs=group_scoped_contribs,
        )
        out_path = field_fits_path(
            root,
            sector,
            camera,
            ccd,
            gid,
            oversampling_factor=oversampling_factor,
        )
        header = build_field_fits_header(
            sector=sector,
            camera=camera,
            ccd=ccd,
            group_id=gid,
            oversampling_factor=oversampling_factor,
            roi_bounds=roi_bounds,
            provenance=prov,
            mapping_grid=mapping_grid,
        )
        written_path = write_field_group_fits(out_path, flux, count, header=header)
        written.append(
            {
                "group_id": int(gid),
                "path": str(Path(written_path).relative_to(root)),
                "shape": [int(flux.shape[0]), int(flux.shape[1])],
            }
        )
        log.info("Materialized field FITS group_id=%d -> %s", gid, written_path)

    payload = {
        "schema_version": 1,
        "fits_dir": FITS_DIRNAME,
        "n_groups": len(written),
        "groups": written,
        "provenance": prov,
    }
    (root / MATERIALIZED_FITS_SIDECAR).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
