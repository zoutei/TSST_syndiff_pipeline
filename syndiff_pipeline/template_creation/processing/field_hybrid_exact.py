"""Exact L4a remaps for field-mode hybrid assignments.

Wraps production ``process_skycell_pixel_mapping`` so callers can Exact-recompute
only the TESS footprints that cover the R=1 seam/rim band, then patch via
``build_l4a_hybrid_assignment``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
    tpix_coord_input: np.ndarray,
    oversampling_factor: int = 1,
    ps1_wcs: WCS | None = None,
    ps1_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Exact PS1→TESS assignment for a subset of global TESS flat ids.

    ``tpix_coord_input`` must be FFI-pixel coordinates from
    :func:`~syndiff_pipeline.common.mapping_grid.create_coords_for_grid`
    (MappingGrid). The legacy ``create_tess_pixel_coordinates(data_shape)``
    fallback is banned on v2 field/remap paths.
    """
    from syndiff_pipeline.template_creation.processing.pancakes import (
        get_ps1_wcs_information,
        process_skycell_pixel_mapping,
    )

    if not isinstance(skycell_row, pd.Series):
        skycell_row = pd.Series(skycell_row)
    if tpix_coord_input is None:
        raise ValueError(
            "tpix_coord_input is required (pass MappingGrid coords via "
            "create_coords_for_grid); create_tess_pixel_coordinates(data_shape) "
            "fallback is banned on v2 field/remap paths"
        )
    tpix = np.asarray(tpix_coord_input)
    if tpix.ndim != 2 or tpix.shape[1] != 2:
        raise ValueError(
            f"tpix_coord_input must be (N, 2) [ty, tx] FFI coords; got shape {tpix.shape}"
        )
    if ps1_wcs is None or ps1_shape is None:
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
    tpix_coord_input: np.ndarray,
    hybrid_R: int = 1,
    oversampling_factor: int = 1,
    extra_tess_ids: np.ndarray | None = None,
    exact_cache_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Roll + Exact-patch L4a hybrid assignment for one ``(sx, sy)``.

    ``tpix_coord_input`` must be MappingGrid FFI coords (see
    :func:`exact_regmap_for_tess_ids`). Returns ``(hybrid_tid, meta)``.
    Caches Exact under ``exact_cache_path`` when set.
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
            tpix_coord_input=tpix_coord_input,
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
    ids_self: np.ndarray | None = None,
) -> np.ndarray:
    """Patch one neighbour's L4b rim onto ``hybrid_tid``; L4b wins on overlap.

    When ``ids_self`` is provided (hoisted border ids), skips recomputing
    :func:`shared_abutting_border_tess_ids`.
    """
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        load_l4b_rim_side,
    )
    from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
        apply_sparse_patch_inplace,
    )

    cache_path = Path(exact_cache_path)

    if ids_self is None:
        ids_self, _ids_nb = shared_abutting_border_tess_ids(
            master, int(skycell_id), int(neighbour_id)
        )
    else:
        ids_self = np.asarray(ids_self, dtype=np.int32)
    if ids_self.size == 0:
        return hybrid_tid

    # Reads only this skycell's side, and accepts both cache layouts.
    idx, val = load_l4b_rim_side(cache_path, skycell_id=int(skycell_id))
    if idx.size == 0:
        return hybrid_tid

    # Rim location is defined on the rolled linear map (border tess ids), not
    # post-L4a exact tess ids which may differ inside the seam band.
    rim_mask = abutting_rim_ps1_mask(linear_tid, ids_self)
    out = np.asarray(hybrid_tid).copy()
    apply_sparse_patch_inplace(out.ravel(), rim_mask.ravel(), idx, val)
    return out


def _patch_l4b_rim_sparse(
    hybrid_tid: np.ndarray,
    *,
    rim_mask: np.ndarray,
    skycell_id: int,
    exact_cache_path: str | Path,
    loader: Any | None = None,
) -> np.ndarray:
    """In-place sparse rim patch using a precomputed (already rolled) rim mask.

    Equivalent to :func:`patch_l4b_rim_from_cache` but skips both the per-key
    ``abutting_rim_ps1_mask`` rebuild and the full-array copy.
    """
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        load_l4b_rim_side,
    )
    from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
        apply_sparse_patch_inplace,
    )

    load = loader if loader is not None else load_l4b_rim_side
    idx, val = load(Path(exact_cache_path), skycell_id=int(skycell_id))
    if len(idx) == 0:
        return hybrid_tid
    apply_sparse_patch_inplace(hybrid_tid.ravel(), rim_mask.ravel(), idx, val)
    return hybrid_tid


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
    group_id: int | None = None,
    epoch_index: Mapping[str, Any] | None = None,
    hybrid_R: int = 1,
    apply_intra_skycell: bool = True,
    apply_inter_skycell: bool = True,
    require_intra_skycell_cache: bool = True,
    require_inter_skycell_cache: bool = True,
    pair_ids: Sequence[tuple[int, int]] | np.ndarray | None = None,
    id_to_name: Mapping[int, str] | None = None,
    neighbour_ids: Sequence[int] | None = None,
    border_ids_by_neighbour: Mapping[int, np.ndarray] | None = None,
    seam_mask_base: np.ndarray | None = None,
    rim_mask_base_by_neighbour: Mapping[int, np.ndarray] | None = None,
    rim_cache_loader: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compose intra-skycell hybrid then inter-skycell rim patches for one group.

    When ``epoch_index`` and ``group_id`` are provided (schema v3), inter-skycell
    rim paths are resolved via pair-epoch lookup. Otherwise falls back to the
    legacy flat ``l4b_rim_cache_basename`` under ``l4b_cache_dir``.

    Optional hoisted metadata (``pair_ids``, ``id_to_name``, ``neighbour_ids``,
    ``border_ids_by_neighbour``) avoids recomputing abutting geometry per call.

    Three further hoists, all per-skycell rather than per-key, and all producing
    bit-identical output (see
    :func:`~...hybrid_regmaps.seam_roll_is_exact_for_shift` for the one precondition):

    ``seam_mask_base``
        Dilated recompute mask built on the *unrolled* frozen map; rolled here
        instead of recomputing ``needs_recompute_mask`` per key. Callers must
        only pass this when ``seam_roll_is_exact_for_shift`` holds.
    ``rim_mask_base_by_neighbour``
        Per-neighbour rim masks built on the unrolled frozen map.
        ``abutting_rim_ps1_mask`` is elementwise, so rolling commutes exactly.
    ``rim_cache_loader``
        ``loader(path, want_lo) -> (idx, val, id_lo, id_hi)``; lets a worker
        memoize sparse rim payloads across keys. Defaults to
        :func:`~...field_abutting.load_l4b_rim_side`.

    When ``apply_intra_skycell`` is False, returns the rolled linear assignment
    without applying the intra-skycell exact patch. When ``apply_inter_skycell``
    is False, skips inter-skycell rim patches.
    """
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
        l4b_rim_cache_basename,
        l4b_rim_path,
    )
    from syndiff_pipeline.template_creation.processing.field_remap import (
        resolve_l4b_pair_epoch_id,
    )
    from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
        needs_recompute_mask,
        roll_assignment,
        roll_mask,
    )

    intra_cache_path = Path(l4a_cache_path)
    if require_intra_skycell_cache and not intra_cache_path.is_file():
        raise FileNotFoundError(f"intra-skycell exact cache missing: {intra_cache_path}")

    # One assignment roll for both intra hybrid and inter rim masking.
    linear_tid = roll_assignment(
        frozen_tid, sx_int, sy_int, convention="assignment"
    )
    # The rolled array is freshly allocated, so it can be patched in place --
    # but only when no later step still needs the *unpatched* map. The fallback
    # rim path derives its mask from ``linear_tid``, so in that case keep a copy.
    needs_unpatched_linear = bool(apply_inter_skycell) and (
        rim_mask_base_by_neighbour is None
    )
    if apply_intra_skycell and intra_cache_path.is_file():
        try:
            with np.load(intra_cache_path) as z:
                exact = np.asarray(z["exact_tid"], dtype=np.int32)
            meta = {"cache_hit": True, "cache_path": str(intra_cache_path)}
        except Exception as exc:
            raise RuntimeError(
                f"corrupt exact cache {intra_cache_path.name}: {exc}"
            ) from exc
        if seam_mask_base is not None:
            mask = roll_mask(seam_mask_base, sx_int, sy_int)
        else:
            mask = needs_recompute_mask(linear_tid, R=int(hybrid_R))
        if needs_unpatched_linear:
            hybrid = apply_hybrid_patch(linear_tid, exact, mask)
        else:
            # Equivalent to apply_hybrid_patch, without the full-array copy.
            hybrid = linear_tid
            replace = mask & (exact >= 0)
            hybrid[replace] = exact[replace]
    else:
        hybrid = linear_tid
        meta = {
            "cache_hit": False,
            "cache_path": str(intra_cache_path),
            "intra_skycell_roll_only": True,
        }

    meta = dict(meta)
    meta["apply_intra_skycell"] = bool(apply_intra_skycell)
    meta["apply_inter_skycell"] = bool(apply_inter_skycell)
    meta["n_inter_skycell_patches"] = 0
    meta["n_inter_skycell_missing"] = 0
    meta["pair_epoch_ids"] = []
    if group_id is not None:
        meta["group_id"] = int(group_id)

    if not apply_inter_skycell:
        return hybrid, meta

    inter_cache_dir = Path(l4b_cache_dir)
    if pair_ids is None:
        pair_ids = abutting_undirected_pairs(master)
    if id_to_name is None:
        id_to_name = {int(nid): str(name) for name, nid in name_to_id.items()}
    use_epochs = epoch_index is not None and group_id is not None

    if neighbour_ids is not None:
        neighbours = [int(n) for n in neighbour_ids]
    else:
        neighbours = []
        for id_lo, id_hi in pair_ids:
            if int(skycell_id) not in (int(id_lo), int(id_hi)):
                continue
            neighbours.append(
                int(id_hi) if int(skycell_id) == int(id_lo) else int(id_lo)
            )

    for neighbour_id in neighbours:
        nb_name = id_to_name.get(int(neighbour_id))
        if nb_name is None or nb_name not in group_shifts:
            continue
        sx_nb, sy_nb = group_shifts[nb_name]
        lo = min(int(skycell_id), int(neighbour_id))
        hi = max(int(skycell_id), int(neighbour_id))
        if use_epochs:
            if int(skycell_id) == lo:
                sx_lo, sy_lo, sx_hi, sy_hi = (
                    int(sx_int),
                    int(sy_int),
                    int(sx_nb),
                    int(sy_nb),
                )
            else:
                sx_lo, sy_lo, sx_hi, sy_hi = (
                    int(sx_nb),
                    int(sy_nb),
                    int(sx_int),
                    int(sy_int),
                )
            try:
                pair_epoch_id = resolve_l4b_pair_epoch_id(
                    epoch_index,
                    id_lo=lo,
                    id_hi=hi,
                    group_id=int(group_id),
                    sx_lo=sx_lo,
                    sy_lo=sy_lo,
                    sx_hi=sx_hi,
                    sy_hi=sy_hi,
                )
            except KeyError:
                if require_inter_skycell_cache:
                    raise
                meta["n_inter_skycell_missing"] = int(meta["n_inter_skycell_missing"]) + 1
                continue
            rim_path = l4b_rim_path(
                inter_cache_dir, lo, hi, pair_epoch_id, sx_lo, sy_lo, sx_hi, sy_hi
            )
            meta["pair_epoch_ids"].append(int(pair_epoch_id))
        else:
            rim_name = l4b_rim_cache_basename(
                int(skycell_id),
                neighbour_id,
                int(sx_int),
                int(sy_int),
                int(sx_nb),
                int(sy_nb),
            )
            rim_path = inter_cache_dir / rim_name
        if border_ids_by_neighbour is not None and int(neighbour_id) in border_ids_by_neighbour:
            ids_self = border_ids_by_neighbour[int(neighbour_id)]
        else:
            ids_self, _ = shared_abutting_border_tess_ids(
                master, int(skycell_id), neighbour_id
            )
        if ids_self.size == 0:
            continue
        if not rim_path.is_file():
            if require_inter_skycell_cache:
                raise FileNotFoundError(f"inter-skycell rim cache missing: {rim_path}")
            meta["n_inter_skycell_missing"] = int(meta["n_inter_skycell_missing"]) + 1
            continue
        if rim_mask_base_by_neighbour is not None:
            hybrid = _patch_l4b_rim_sparse(
                hybrid,
                rim_mask=roll_mask(
                    rim_mask_base_by_neighbour[int(neighbour_id)], sx_int, sy_int
                ),
                skycell_id=int(skycell_id),
                exact_cache_path=rim_path,
                loader=rim_cache_loader,
            )
        else:
            hybrid = patch_l4b_rim_from_cache(
                hybrid,
                linear_tid=linear_tid,
                skycell_id=int(skycell_id),
                neighbour_id=neighbour_id,
                master=master,
                exact_cache_path=rim_path,
                ids_self=ids_self,
            )
        meta["n_inter_skycell_patches"] = int(meta["n_inter_skycell_patches"]) + 1

    return hybrid, meta
