# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Blend grouping (stars fit simultaneously, not in isolation) + stamp extraction.

Grouping is spatial-only, built from warmstart-distorted positions at a single
reference frame (middle of the fit window). Single-linkage on <max_sep_px
separation, padded to a fixed max_group_size so the flux solve can batch over
groups. Each group gets one shared SxS data stamp per frame, cut around a
single integer pixel center derived from primary members' reference positions.
Companions are attached by mag-dependent L∞ distance after initial grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .epsf_model import STAMP_PHYSICAL

# FrameImage (→ data → astropy) and scipy are prep-only. Keep them lazy so
# ``run_fit --from-bundle`` can import GroupSet / bucket_groups_by_size lean.

MAX_GROUP_SIZE_CAP = 8


@dataclass
class GroupSet:
    n_groups: int
    max_group_size: int
    members: np.ndarray  # (n_groups, max_group_size) int, star index into the input table; -1 = pad
    valid: np.ndarray  # (n_groups, max_group_size) bool
    kept_star_mask: np.ndarray  # (n_stars_in,) bool, True if the star survived grouping
    dropped_oversized: int  # count of stars dropped/pruned for group size > max_group_size


def companion_attach_radius_px(
    tess_mag: float | np.ndarray,
    stamp_physical: int = STAMP_PHYSICAL,
) -> float | np.ndarray:
    """Mag-dependent L∞ attach radius (px) for modeling a neighbor in a stamp.

    Threshold is the stamp diagonal scale plus a mag-dependent pad::

        r = h * sqrt(2) + delta(mag),  h = stamp_physical // 2

    ``delta``: mag < 9 → +6; 9–11 → +5; 11–12 → +4; 12–13 → +3.
    """
    h = int(stamp_physical) // 2
    base = float(h) * float(np.sqrt(2.0))
    mag = np.asarray(tess_mag, dtype=float)
    # Brighter → larger pad (need core + wings farther out).
    delta = np.where(
        mag < 9.0, 6.0,
        np.where(mag < 11.0, 5.0, np.where(mag < 12.0, 4.0, 3.0)),
    )
    r = base + delta
    if np.ndim(tess_mag) == 0:
        return float(np.asarray(r))
    return r.astype(np.float64)


def build_groups(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_sep_px: float = 7.0,
    max_group_size: int = 4,
) -> GroupSet:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(x)
    xy = np.column_stack([x, y])
    if n == 0:
        return GroupSet(0, max_group_size, np.zeros((0, max_group_size), dtype=int),
                         np.zeros((0, max_group_size), dtype=bool), np.zeros(0, dtype=bool), 0)

    tree = cKDTree(xy)
    pairs = tree.query_pairs(max_sep_px, output_type="ndarray")
    if len(pairs):
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    else:
        adj = coo_matrix((n, n))
    _, labels = connected_components(adj, directed=False)

    kept_star_mask = np.zeros(n, dtype=bool)
    kept_groups: list[np.ndarray] = []
    dropped_oversized = 0
    for g in np.unique(labels):
        members = np.where(labels == g)[0]
        if len(members) > max_group_size:
            dropped_oversized += len(members)
            continue
        kept_groups.append(members)
        kept_star_mask[members] = True

    n_groups = len(kept_groups)
    members_arr = np.full((n_groups, max_group_size), -1, dtype=int)
    valid_arr = np.zeros((n_groups, max_group_size), dtype=bool)
    for gi, members in enumerate(kept_groups):
        members_arr[gi, : len(members)] = members
        valid_arr[gi, : len(members)] = True

    return GroupSet(n_groups, max_group_size, members_arr, valid_arr, kept_star_mask, dropped_oversized)


def _stamp_centers_from_members(
    member_lists: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integer stamp centers from unweighted means of per-group member positions."""
    n_groups = len(member_lists)
    cx = np.zeros(n_groups, dtype=np.int64)
    cy = np.zeros(n_groups, dtype=np.int64)
    for gi, members in enumerate(member_lists):
        if len(members) == 0:
            continue
        cx[gi] = int(round(float(np.mean(x[members]))))
        cy[gi] = int(round(float(np.mean(y[members]))))
    return cx, cy


def _prune_members_to_cap(
    members: np.ndarray,
    primary: np.ndarray,
    mags: np.ndarray,
    cap: int,
) -> tuple[np.ndarray, int]:
    """Keep all primaries that fit, then brightest companions, up to ``cap``.

    Returns ``(kept_members, n_pruned)``.
    """
    members = np.asarray(members, dtype=int)
    if len(members) <= cap:
        return members, 0
    prim_set = set(int(i) for i in np.asarray(primary, dtype=int))
    prims = [i for i in members if int(i) in prim_set]
    comps = [i for i in members if int(i) not in prim_set]
    # Prefer keeping primaries; if still over cap, keep brightest primaries.
    prims_sorted = sorted(prims, key=lambda i: float(mags[i]))
    if len(prims_sorted) >= cap:
        kept = np.array(prims_sorted[:cap], dtype=int)
        return kept, int(len(members) - len(kept))
    n_comp = cap - len(prims_sorted)
    comps_sorted = sorted(comps, key=lambda i: float(mags[i]))  # bright first
    kept = np.array(prims_sorted + comps_sorted[:n_comp], dtype=int)
    return kept, int(len(members) - len(kept))


def augment_groups_with_stamp_neighbors(
    groups: GroupSet,
    x: np.ndarray,
    y: np.ndarray,
    neighbor_indices: np.ndarray,
    primary_members_by_group: list[np.ndarray],
    *,
    neighbor_mags: np.ndarray | None = None,
    stamp_physical: int = STAMP_PHYSICAL,
    companion_radius_px: float | None = None,
    max_group_size: int = 4,
    max_group_size_cap: int = MAX_GROUP_SIZE_CAP,
    mags: np.ndarray | None = None,
) -> tuple[GroupSet, dict[str, int], np.ndarray, np.ndarray]:
    """Add Gaia neighbors whose PSF should be modelled in each group's stamp.

    Stamp centers are derived from ``primary_members_by_group`` only (not
    recentroided onto companions). A neighbor is attached if its L∞ distance
    to that center is ≤ ``r_attach``:

    - if ``companion_radius_px`` is set: flat radius for all neighbors (debug);
    - else: mag-dependent ``companion_attach_radius_px(mag, stamp_physical)``
      using ``neighbor_mags`` (required in that case).

    When a group exceeds ``max_group_size_cap``, faintest companions are pruned
    first (primaries kept preferentially) instead of dropping the whole group.

    Returns ``(groups, stats, stamp_center_x, stamp_center_y)`` for surviving
    groups (primary-only centers).
    """
    n_groups = groups.n_groups
    if n_groups == 0:
        empty = np.zeros(0, dtype=np.int64)
        return groups, {"companions_added": 0, "groups_grown": 0, "dropped_oversized": 0}, empty, empty

    cx, cy = _stamp_centers_from_members(primary_members_by_group, x, y)
    neighbor_indices = np.asarray(neighbor_indices, dtype=int)
    if companion_radius_px is None:
        if neighbor_mags is None:
            raise ValueError("neighbor_mags is required when companion_radius_px is None")
        neighbor_mags_arr = np.asarray(neighbor_mags, dtype=float)
        if neighbor_mags_arr.shape[0] == len(x):
            ni_mag = neighbor_mags_arr[neighbor_indices]
        elif neighbor_mags_arr.shape[0] == len(neighbor_indices):
            ni_mag = neighbor_mags_arr
        else:
            raise ValueError("neighbor_mags must align with x or neighbor_indices")
        ni_radius = np.asarray(companion_attach_radius_px(ni_mag, stamp_physical), dtype=float)
    else:
        ni_radius = np.full(len(neighbor_indices), float(companion_radius_px))

    if mags is None:
        if neighbor_mags is not None and np.asarray(neighbor_mags).shape[0] == len(x):
            mags = np.asarray(neighbor_mags, dtype=float)
        else:
            mags = np.zeros(len(x), dtype=float)
    mags = np.asarray(mags, dtype=float)

    from scipy.spatial import cKDTree

    # Vectorized neighbor attach: the attach radius only takes a handful of
    # distinct values (companion_attach_radius_px has 4 mag tiers; a flat
    # override has 1), so bucket candidates by radius and run one KDTree
    # query_ball_point per bucket against *all* group centers at once,
    # instead of a Python double loop over (candidates x groups). L-infinity
    # (Chebyshev, p=inf) matches the original max(|dx|,|dy|) <= r test.
    centers = np.column_stack([cx, cy]).astype(float)
    added_by_group: list[set[int]] = [set() for _ in range(n_groups)]
    if n_groups > 0 and len(neighbor_indices) > 0:
        for r_val in np.unique(ni_radius):
            tier_ni = neighbor_indices[ni_radius == r_val]
            tier_xy = np.column_stack([x[tier_ni], y[tier_ni]])
            tier_tree = cKDTree(tier_xy)
            hits = tier_tree.query_ball_point(centers, r=float(r_val), p=np.inf)
            for gi, local_idx in enumerate(hits):
                if local_idx:
                    added_by_group[gi].update(int(v) for v in tier_ni[local_idx])

    new_member_lists: list[np.ndarray] = []
    companions_added = 0
    groups_grown = 0

    for gi in range(n_groups):
        existing = set(int(i) for i in primary_members_by_group[gi])
        added = added_by_group[gi] - existing
        if added:
            companions_added += len(added)
            groups_grown += 1
            existing |= added
        new_member_lists.append(np.array(sorted(existing), dtype=int))

    max_observed = max((len(m) for m in new_member_lists), default=0)
    new_max = max(max_group_size, min(max_observed, max_group_size_cap))

    kept_groups: list[np.ndarray] = []
    kept_primary_lists: list[np.ndarray] = []
    dropped_oversized = 0
    for members, prim in zip(new_member_lists, primary_members_by_group):
        kept, n_pruned = _prune_members_to_cap(members, prim, mags, new_max)
        dropped_oversized += n_pruned
        if len(kept) == 0:
            continue
        kept_groups.append(np.sort(kept))
        prim_set = set(int(i) for i in prim)
        kept_primary_lists.append(np.array([i for i in kept if int(i) in prim_set], dtype=int))

    n_kept = len(kept_groups)
    stamp_cx, stamp_cy = _stamp_centers_from_members(kept_primary_lists, x, y)
    members_arr = np.full((n_kept, new_max), -1, dtype=int)
    valid_arr = np.zeros((n_kept, new_max), dtype=bool)
    kept_star_mask = np.zeros(len(x), dtype=bool)
    for gi, members in enumerate(kept_groups):
        members_arr[gi, : len(members)] = members
        valid_arr[gi, : len(members)] = True
        kept_star_mask[members] = True

    new_groups = GroupSet(
        n_kept, new_max, members_arr, valid_arr, kept_star_mask, dropped_oversized,
    )
    stats: dict[str, int] = {
        "companions_added": companions_added,
        "groups_grown": groups_grown,
        "dropped_oversized": dropped_oversized,
    }
    return new_groups, stats, stamp_cx, stamp_cy


def filter_groups_by_member_radius(
    groups: GroupSet,
    x: np.ndarray,
    y: np.ndarray,
    stamp_center_x: np.ndarray,
    stamp_center_y: np.ndarray,
    *,
    max_member_radius_px: float = 5.0,
) -> tuple[GroupSet, np.ndarray, np.ndarray, dict[str, int]]:
    """Drop/trim members farther than ``max_member_radius_px`` (Euclidean) from center.

    Pass ``max_member_radius_px <= 0`` to skip (identity). Prefer mag-dependent
    attach instead of this post-filter.
    """
    if max_member_radius_px is None or float(max_member_radius_px) <= 0.0:
        stats = {"dropped_groups": 0, "trimmed_members": 0, "n_groups_kept": groups.n_groups}
        return groups, np.asarray(stamp_center_x, dtype=np.int64), np.asarray(stamp_center_y, dtype=np.int64), stats

    kept_members: list[np.ndarray] = []
    kept_cx: list[int] = []
    kept_cy: list[int] = []
    dropped_groups = 0
    trimmed_members = 0
    for gi in range(groups.n_groups):
        idx = groups.members[gi][groups.valid[gi]]
        if len(idx) == 0:
            dropped_groups += 1
            continue
        dx = x[idx] - float(stamp_center_x[gi])
        dy = y[idx] - float(stamp_center_y[gi])
        dist = np.sqrt(dx**2 + dy**2)
        keep = dist <= max_member_radius_px
        trimmed_members += int((~keep).sum())
        kept = idx[keep]
        if len(kept) == 0:
            dropped_groups += 1
            continue
        kept_members.append(kept)
        kept_cx.append(int(stamp_center_x[gi]))
        kept_cy.append(int(stamp_center_y[gi]))

    n_groups = len(kept_members)
    max_k = groups.max_group_size
    members_arr = np.full((n_groups, max_k), -1, dtype=int)
    valid_arr = np.zeros((n_groups, max_k), dtype=bool)
    kept_star_mask = np.zeros(len(x), dtype=bool)
    for gi, members in enumerate(kept_members):
        members_arr[gi, : len(members)] = members
        valid_arr[gi, : len(members)] = True
        kept_star_mask[members] = True
    new_groups = GroupSet(
        n_groups, max_k, members_arr, valid_arr, kept_star_mask, groups.dropped_oversized,
    )
    stats = {
        "dropped_groups": dropped_groups,
        "trimmed_members": trimmed_members,
        "n_groups_kept": n_groups,
    }
    return new_groups, np.asarray(kept_cx, dtype=np.int64), np.asarray(kept_cy, dtype=np.int64), stats


def bucket_groups_by_size(
    groups: GroupSet,
    stamp_center_x: np.ndarray,
    stamp_center_y: np.ndarray,
    *,
    tiers: tuple[int, ...] = (1, 2, 4, 8),
) -> list[tuple[GroupSet, np.ndarray, np.ndarray, np.ndarray]]:
    """Split one ``GroupSet`` into per-tier sub-``GroupSet``s by true member count.

    ``flux_solve.solve_group_fluxes`` pays ``K^3`` cost per group at the
    *global* ``groups.max_group_size``, for every group, including isolated
    1-star groups, whenever any single group anywhere in the set is crowded.
    Splitting into fixed-K tiers (each group assigned to the smallest tier
    that fits its true member count) lets the caller run the loss/solve once
    per tier at that tier's own, usually much smaller, K -- most groups in a
    real field are small, so most of the fleet's compute moves to small K.

    ``tiers`` must be ascending and its max must cover ``groups.max_group_size``
    (every group already fits there by construction -- this only fails if
    called with a ``tiers`` tuple that doesn't match the group set it's given).

    Returns one ``(bucket_groups, bucket_stamp_center_x, bucket_stamp_center_y,
    orig_group_idx)`` per *non-empty* tier, in ascending K order.
    ``orig_group_idx`` (shape ``(bucket_groups.n_groups,)``) is each bucket-local
    group's index into the *original* ``groups``/``stamp_center_x/y`` -- use it
    to slice any other already-built per-group array (e.g. a ``StampBatch`` from
    ``extract_stamps``, or a TNS/asteroid ``mask_active``) without needing to
    recompute it per bucket.
    """
    tiers = tuple(sorted(tiers))
    if groups.n_groups == 0:
        return []
    if tiers[-1] < groups.max_group_size:
        raise ValueError(
            f"tiers={tiers} does not cover groups.max_group_size={groups.max_group_size}; "
            f"every group must fit in some tier"
        )
    sizes = groups.valid.sum(axis=1)  # (n_groups,) true member count per group
    n_stars = groups.kept_star_mask.shape[0]

    buckets: list[tuple[GroupSet, np.ndarray, np.ndarray, np.ndarray]] = []
    for k in tiers:
        lower_tiers = [t for t in tiers if t < k]
        lo = lower_tiers[-1] if lower_tiers else 0
        # Each group belongs to exactly the smallest tier >= its size; since
        # tiers are sorted, that's (prev_tier, k] with prev_tier the tier
        # immediately below (or 0 for the first tier) -- lo itself is
        # included in the *previous* tier's range, not this one, so this
        # must be a strict ">", not ">=".
        in_tier = (sizes > lo) & (sizes <= k)
        idx = np.where(in_tier)[0]
        if len(idx) == 0:
            continue
        n_bucket = len(idx)
        members_arr = np.full((n_bucket, k), -1, dtype=int)
        valid_arr = np.zeros((n_bucket, k), dtype=bool)
        kept_star_mask = np.zeros(n_stars, dtype=bool)
        for out_i, gi in enumerate(idx):
            n_valid = int(sizes[gi])
            mem = groups.members[gi][groups.valid[gi]]
            members_arr[out_i, :n_valid] = mem
            valid_arr[out_i, :n_valid] = True
            kept_star_mask[mem] = True
        bucket_groups = GroupSet(
            n_groups=n_bucket, max_group_size=k, members=members_arr, valid=valid_arr,
            kept_star_mask=kept_star_mask, dropped_oversized=0,
        )
        buckets.append((
            bucket_groups,
            np.asarray(stamp_center_x)[idx],
            np.asarray(stamp_center_y)[idx],
            idx,
        ))
    return buckets


def stamp_fits_in_array(
    stamp_center_x: np.ndarray,
    stamp_center_y: np.ndarray,
    *,
    array_origin: tuple[int, int],
    array_shape: tuple[int, int],
    stamp: int = STAMP_PHYSICAL,
) -> np.ndarray:
    """Boolean per group: S×S stamp around center fits inside the loaded array."""
    half = stamp // 2
    ox, oy = array_origin
    ny, nx = array_shape
    cx = np.asarray(stamp_center_x, dtype=np.int64)
    cy = np.asarray(stamp_center_y, dtype=np.int64)
    x0 = cx - ox - half
    y0 = cy - oy - half
    x1 = x0 + stamp
    y1 = y0 + stamp
    return (x0 >= 0) & (y0 >= 0) & (x1 <= nx) & (y1 <= ny)


def drop_groups_off_array(
    groups: GroupSet,
    stamp_center_x: np.ndarray,
    stamp_center_y: np.ndarray,
    *,
    array_origin: tuple[int, int],
    array_shape: tuple[int, int],
    stamp: int = STAMP_PHYSICAL,
    n_stars: int | None = None,
) -> tuple[GroupSet, np.ndarray, np.ndarray, dict[str, int]]:
    """Drop groups whose fixed stamp does not fit in the loaded region crop."""
    keep = stamp_fits_in_array(
        stamp_center_x, stamp_center_y,
        array_origin=array_origin, array_shape=array_shape, stamp=stamp,
    )
    n_drop = int((~keep).sum())
    if n_drop == 0:
        stats = {"dropped_groups": 0, "n_groups_kept": groups.n_groups}
        return (
            groups,
            np.asarray(stamp_center_x, dtype=np.int64),
            np.asarray(stamp_center_y, dtype=np.int64),
            stats,
        )

    keep_idx = np.where(keep)[0]
    n_kept = len(keep_idx)
    max_k = groups.max_group_size
    members_arr = np.full((n_kept, max_k), -1, dtype=int)
    valid_arr = np.zeros((n_kept, max_k), dtype=bool)
    n_star = int(n_stars) if n_stars is not None else int(groups.kept_star_mask.shape[0])
    kept_star_mask = np.zeros(n_star, dtype=bool)
    for out_i, gi in enumerate(keep_idx):
        members_arr[out_i] = groups.members[gi]
        valid_arr[out_i] = groups.valid[gi]
        mem = groups.members[gi][groups.valid[gi]]
        kept_star_mask[mem] = True
    new_groups = GroupSet(
        n_kept, max_k, members_arr, valid_arr, kept_star_mask, groups.dropped_oversized,
    )
    stats = {"dropped_groups": n_drop, "n_groups_kept": n_kept}
    return (
        new_groups,
        np.asarray(stamp_center_x, dtype=np.int64)[keep_idx],
        np.asarray(stamp_center_y, dtype=np.int64)[keep_idx],
        stats,
    )


@dataclass
class StampBatch:
    data: np.ndarray  # (n_groups, n_frames, S, S) electrons/s
    noise: np.ndarray  # (n_groups, n_frames, S, S) electrons/s
    weight: np.ndarray  # (n_groups, n_frames, S, S) 0/1 coverage (ones; hp_d MASK unused)
    stamp_center_x: np.ndarray  # (n_groups,) int, region-local pixel column
    stamp_center_y: np.ndarray  # (n_groups,) int, region-local pixel row
    exposure_days: np.ndarray  # (n_frames,)


def group_stamp_centers(groups: GroupSet, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One integer (round) pixel center per group, from the unweighted member mean."""
    member_lists = [groups.members[gi][groups.valid[gi]] for gi in range(groups.n_groups)]
    return _stamp_centers_from_members(member_lists, x, y)


def extract_stamps(
    groups: GroupSet,
    frames: list,  # list[FrameImage]; type deferred to avoid importing data.py
    x: np.ndarray,
    y: np.ndarray,
    *,
    array_origin: tuple[int, int] = (0, 0),
    stamp: int = STAMP_PHYSICAL,
    stamp_center_x: np.ndarray | None = None,
    stamp_center_y: np.ndarray | None = None,
) -> StampBatch:
    """Cut one fixed-center stamp per group out of every frame's region crop.

    ``x, y`` (reference-frame warmstart positions) and the returned
    ``stamp_center_x/y`` are in the same crop-local coordinate system as the
    WCS model (unshifted); ``array_origin``
    is that system's offset from index (0,0) of ``frames[i].cal`` (e.g. the
    lower-left corner of a margin-expanded load region), so array indices are
    ``center - array_origin - half``.

    When ``stamp_center_x/y`` are provided (e.g. from primary members only),
    those fixed centers are used instead of recomputing from all group members.
    """
    if stamp_center_x is None or stamp_center_y is None:
        cx, cy = group_stamp_centers(groups, x, y)
    else:
        cx, cy = stamp_center_x, stamp_center_y
    half = stamp // 2
    ox, oy = array_origin
    ny, nx = frames[0].cal.shape
    n_groups = groups.n_groups
    n_frames = len(frames)
    data = np.zeros((n_groups, n_frames, stamp, stamp), dtype=np.float32)
    noise = np.zeros((n_groups, n_frames, stamp, stamp), dtype=np.float32)
    weight = np.zeros((n_groups, n_frames, stamp, stamp), dtype=np.float32)

    for gi in range(n_groups):
        x0, y0 = int(cx[gi]) - ox - half, int(cy[gi]) - oy - half
        x1, y1 = x0 + stamp, y0 + stamp
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            continue  # stamp runs off the loaded array; leave weight=0 (fully masked)
        for fi, frame in enumerate(frames):
            data[gi, fi] = frame.cal[y0:y1, x0:x1]
            noise[gi, fi] = frame.noise[y0:y1, x0:x1]
            weight[gi, fi] = (~frame.bad[y0:y1, x0:x1]).astype(np.float32)

    exposure_days = np.array([f.exposure_days for f in frames], dtype=np.float64)
    return StampBatch(data, noise, weight, cx, cy, exposure_days)
