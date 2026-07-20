"""Exact L4a remaps for field-mode hybrid assignments.

Wraps production ``process_skycell_pixel_mapping`` so callers can Exact-recompute
only the TESS footprints that cover the R=1 seam/rim band, then patch via
``build_l4a_hybrid_assignment``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
    abutting_rim_ps1_mask,
    apply_hybrid_patch,
    build_l4a_hybrid_assignment,
)

log = logging.getLogger(__name__)


def exact_regmap_for_tess_ids(
    tess_wcs: WCS,
    skycell_row: pd.Series | dict[str, Any],
    tess_ids: np.ndarray,
    *,
    data_shape: tuple[int, int],
    oversampling_factor: int = 1,
) -> np.ndarray:
    """Exact PS1→TESS assignment for a subset of global TESS flat ids."""
    from syndiff_pipeline.template_creation.processing.pancakes import (
        create_tess_pixel_coordinates,
        get_ps1_wcs_information,
        process_skycell_pixel_mapping,
    )

    if not isinstance(skycell_row, pd.Series):
        skycell_row = pd.Series(skycell_row)
    tpix, _ = create_tess_pixel_coordinates(data_shape, oversampling_factor)
    _, ps1_wcs, ps1_shape = get_ps1_wcs_information(skycell_row)
    tids = np.asarray(tess_ids, dtype=np.int32)
    tids = tids[(tids >= 0) & (tids < len(tpix))]
    if tids.size == 0:
        return np.full(ps1_shape, -1, dtype=np.int32)
    return process_skycell_pixel_mapping(
        tess_wcs,
        tpix,
        ps1_wcs,
        ps1_shape,
        tids,
        oversampling_factor=oversampling_factor,
    )


def candidate_tess_ids_for_l4a(
    frozen_tid: np.ndarray,
    sx_int: int,
    sy_int: int,
    *,
    hybrid_R: int = 1,
    extra_tess_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """TESS ids whose Exact footprints should cover the L4a recompute mask."""
    linear, mask = build_l4a_hybrid_assignment(
        frozen_tid, sx_int, sy_int, exact_tid=None, hybrid_R=hybrid_R
    )
    tids = np.unique(linear[mask])
    tids = tids[tids >= 0].astype(np.int32)
    if extra_tess_ids is not None and len(extra_tess_ids):
        extra = np.asarray(extra_tess_ids, dtype=np.int32)
        extra = extra[extra >= 0]
        tids = np.unique(np.concatenate([tids, extra]))
    return tids, mask


def build_hybrid_assignment_with_exact(
    frozen_tid: np.ndarray,
    sx_int: int,
    sy_int: int,
    tess_wcs: WCS,
    skycell_row: pd.Series | dict[str, Any],
    *,
    data_shape: tuple[int, int],
    hybrid_R: int = 1,
    oversampling_factor: int = 1,
    extra_tess_ids: np.ndarray | None = None,
    exact_cache_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Roll + Exact-patch L4a hybrid assignment for one ``(sx, sy)``.

    Returns ``(hybrid_tid, meta)``. Caches Exact under ``exact_cache_path`` when set.
    """
    tids, mask = candidate_tess_ids_for_l4a(
        frozen_tid,
        sx_int,
        sy_int,
        hybrid_R=hybrid_R,
        extra_tess_ids=extra_tess_ids,
    )
    meta: dict[str, Any] = {
        "n_mask": int(mask.sum()),
        "n_tess_ids": int(tids.size),
        "cache_hit": False,
    }
    if tids.size == 0 or int(mask.sum()) == 0:
        hybrid, _ = build_l4a_hybrid_assignment(
            frozen_tid, sx_int, sy_int, exact_tid=None, hybrid_R=hybrid_R
        )
        meta["skipped_exact"] = True
        return hybrid, meta

    exact: Optional[np.ndarray] = None
    cache_path = Path(exact_cache_path) if exact_cache_path else None
    if cache_path is not None and cache_path.is_file():
        # A partially-written / zero-size cache NPZ (e.g. a run killed mid-write)
        # must be treated as a cache MISS and recomputed — not allowed to raise
        # and silently degrade this key to a data-roll fallback (which would then
        # be written as a roll-quality contrib and cached as "done").
        try:
            with np.load(cache_path) as z:
                exact = np.asarray(z["exact_tid"], dtype=np.int32)
            meta["cache_hit"] = True
        except Exception as exc:
            log.warning(
                "corrupt exact cache %s (%s); recomputing", cache_path.name, exc
            )
            try:
                cache_path.unlink()
            except OSError:
                pass
            exact = None
    if exact is None:
        exact = exact_regmap_for_tess_ids(
            tess_wcs,
            skycell_row,
            tids,
            data_shape=data_shape,
            oversampling_factor=oversampling_factor,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, exact_tid=exact.astype(np.int32))

    hybrid, _ = build_l4a_hybrid_assignment(
        frozen_tid, sx_int, sy_int, exact_tid=exact, hybrid_R=hybrid_R
    )
    meta["skipped_exact"] = False
    return hybrid, meta


def hybrid_assignment_from_exact_cache(
    frozen_tid: np.ndarray,
    sx_int: int,
    sy_int: int,
    exact_cache_path: str | Path,
    *,
    hybrid_R: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build hybrid assignment from a pre-built Exact cache (L5 downsample only)."""
    cache_path = Path(exact_cache_path)
    meta: dict[str, Any] = {"cache_hit": False, "cache_path": str(cache_path)}
    if not cache_path.is_file():
        raise FileNotFoundError(f"exact cache missing: {cache_path}")
    try:
        with np.load(cache_path) as z:
            exact = np.asarray(z["exact_tid"], dtype=np.int32)
        meta["cache_hit"] = True
    except Exception as exc:
        raise RuntimeError(f"corrupt exact cache {cache_path.name}: {exc}") from exc
    hybrid, _ = build_l4a_hybrid_assignment(
        frozen_tid, sx_int, sy_int, exact_tid=exact, hybrid_R=hybrid_R
    )
    return hybrid, meta


def abutting_border_tess_ids(
    master: np.ndarray,
    skycell_id: int,
) -> np.ndarray:
    """TESS flat ids on the exclusive abutting border of ``skycell_id`` in master."""
    t_y, t_x = master.shape
    owned = master == int(skycell_id)
    if not owned.any():
        return np.array([], dtype=np.int32)
    # 4-neighbour where ownership changes
    border = np.zeros_like(owned)
    border[:, :-1] |= owned[:, :-1] & ~owned[:, 1:]
    border[:, 1:] |= owned[:, 1:] & ~owned[:, :-1]
    border[:-1, :] |= owned[:-1, :] & ~owned[1:, :]
    border[1:, :] |= owned[1:, :] & ~owned[:-1, :]
    ys, xs = np.nonzero(border & owned)
    return (ys.astype(np.int64) * t_x + xs.astype(np.int64)).astype(np.int32)


def shared_abutting_border_tess_ids(
    master: np.ndarray,
    skycell_id_a: int,
    skycell_id_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exclusive A|B abutting TESS flat ids (A-owned and B-owned sides).

    Used for full F2 pair-state L4b (not yet wired in production remap/downsample).
    """
    t_x = master.shape[1]
    a = master == int(skycell_id_a)
    b = master == int(skycell_id_b)
    if not a.any() or not b.any():
        empty = np.array([], dtype=np.int32)
        return empty, empty
    touch = np.zeros_like(a)
    touch[:, :-1] |= a[:, :-1] & b[:, 1:]
    touch[:, 1:] |= a[:, 1:] & b[:, :-1]
    touch[:-1, :] |= a[:-1, :] & b[1:, :]
    touch[1:, :] |= a[1:, :] & b[:-1, :]
    # Expand one pixel onto each owned side along 4-neigh
    a_border = np.zeros_like(a)
    b_border = np.zeros_like(b)
    a_border[:, :-1] |= a[:, :-1] & touch[:, 1:]
    a_border[:, 1:] |= a[:, 1:] & touch[:, :-1]
    a_border[:-1, :] |= a[:-1, :] & touch[1:, :]
    a_border[1:, :] |= a[1:, :] & touch[:-1, :]
    b_border[:, :-1] |= b[:, :-1] & touch[:, 1:]
    b_border[:, 1:] |= b[:, 1:] & touch[:, :-1]
    b_border[:-1, :] |= b[:-1, :] & touch[1:, :]
    b_border[1:, :] |= b[1:, :] & touch[:-1, :]
    # Also include touch pixels themselves assigned to A or B
    a_border |= touch & a
    b_border |= touch & b
    ya, xa = np.nonzero(a_border)
    yb, xb = np.nonzero(b_border)
    ids_a = (ya.astype(np.int64) * t_x + xa.astype(np.int64)).astype(np.int32)
    ids_b = (yb.astype(np.int64) * t_x + xb.astype(np.int64)).astype(np.int32)
    return ids_a, ids_b


def patch_l4b_rim_from_cache(
    hybrid_tid: np.ndarray,
    *,
    linear_tid: np.ndarray,
    skycell_id: int,
    neighbour_id: int,
    master: np.ndarray,
    exact_cache_path: str | Path,
) -> np.ndarray:
    """Patch one neighbour's L4b rim onto ``hybrid_tid``; L4b wins on overlap."""
    cache_path = Path(exact_cache_path)
    with np.load(cache_path) as z:
        id_lo = int(z["id_lo"])
        id_hi = int(z["id_hi"])
        exact_lo = np.asarray(z["exact_tid_lo"], dtype=np.int32)
        exact_hi = np.asarray(z["exact_tid_hi"], dtype=np.int32)

    ids_self, _ids_nb = shared_abutting_border_tess_ids(
        master, int(skycell_id), int(neighbour_id)
    )
    if ids_self.size == 0:
        return hybrid_tid

    if int(skycell_id) == id_lo:
        exact_tid = exact_lo
    elif int(skycell_id) == id_hi:
        exact_tid = exact_hi
    else:
        raise ValueError(
            f"skycell_id {skycell_id} not in L4b cache pair ({id_lo}, {id_hi})"
        )

    if exact_tid.size == 0:
        return hybrid_tid

    # Rim location is defined on the rolled linear map (border tess ids), not
    # post-L4a exact tess ids which may differ inside the seam band.
    rim_mask = abutting_rim_ps1_mask(linear_tid, ids_self)
    return apply_hybrid_patch(hybrid_tid, exact_tid, rim_mask)


def compose_group_hybrid_assignment(
    frozen_tid: np.ndarray,
    *,
    skycell: str,
    skycell_id: int,
    sx_int: int,
    sy_int: int,
    master: np.ndarray,
    group_shifts: Mapping[str, tuple[int, int]],
    name_to_id: Mapping[str, int],
    l4a_cache_path: str | Path,
    l4b_cache_dir: str | Path,
    hybrid_R: int = 1,
    apply_l4b: bool = True,
    require_l4a_cache: bool = True,
    require_l4b_cache: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compose L4a hybrid then L4b rim patches for one group context."""
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
        l4b_rim_cache_basename,
    )

    cache_path = Path(l4a_cache_path)
    if require_l4a_cache and not cache_path.is_file():
        raise FileNotFoundError(f"L4a exact cache missing: {cache_path}")
    if cache_path.is_file():
        hybrid, meta = hybrid_assignment_from_exact_cache(
            frozen_tid,
            sx_int,
            sy_int,
            cache_path,
            hybrid_R=int(hybrid_R),
        )
    else:
        hybrid, _ = build_l4a_hybrid_assignment(
            frozen_tid, sx_int, sy_int, exact_tid=None, hybrid_R=int(hybrid_R)
        )
        meta = {"cache_hit": False, "cache_path": str(cache_path), "l4a_only_roll": True}

    linear_tid, _ = build_l4a_hybrid_assignment(
        frozen_tid, sx_int, sy_int, exact_tid=None, hybrid_R=int(hybrid_R)
    )

    meta = dict(meta)
    meta["n_l4b_patches"] = 0

    if not apply_l4b:
        return hybrid, meta

    l4b_dir = Path(l4b_cache_dir)
    pair_ids = abutting_undirected_pairs(master)

    for id_lo, id_hi in pair_ids:
        if int(skycell_id) not in (int(id_lo), int(id_hi)):
            continue
        neighbour_id = int(id_hi) if int(skycell_id) == int(id_lo) else int(id_lo)
        nb_name = None
        for name, nid in name_to_id.items():
            if int(nid) == neighbour_id:
                nb_name = str(name)
                break
        if nb_name is None or nb_name not in group_shifts:
            continue
        sx_nb, sy_nb = group_shifts[nb_name]
        rim_name = l4b_rim_cache_basename(
            int(skycell_id),
            neighbour_id,
            int(sx_int),
            int(sy_int),
            int(sx_nb),
            int(sy_nb),
        )
        rim_path = l4b_dir / rim_name
        ids_self, _ = shared_abutting_border_tess_ids(
            master, int(skycell_id), neighbour_id
        )
        if ids_self.size == 0:
            continue
        if not rim_path.is_file():
            if require_l4b_cache:
                raise FileNotFoundError(f"L4b rim cache missing: {rim_path}")
            continue
        hybrid = patch_l4b_rim_from_cache(
            hybrid,
            linear_tid=linear_tid,
            skycell_id=int(skycell_id),
            neighbour_id=neighbour_id,
            master=master,
            exact_cache_path=rim_path,
        )
        meta["n_l4b_patches"] = int(meta["n_l4b_patches"]) + 1

    return hybrid, meta
