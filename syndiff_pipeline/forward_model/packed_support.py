# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Irregular packed-stamp ingest, P/K tiering, and hp_d pixel gather.

The other agent supplies per-stamp member star indices + scored pixel xy lists.
This module pads them into JAX-batchable tiers and gathers ``hp_d`` into 1D
``(n_groups, n_frames, P)`` arrays for the packed optimization path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .groups import GroupSet, MAX_GROUP_SIZE_CAP

DEFAULT_K_TIERS: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_P_TIERS: tuple[int, ...] = (64, 128, 256, 512)


def ensure_tiers_cover(n_max: int, tiers: tuple[int, ...]) -> tuple[int, ...]:
    """Return sorted tiers, doubling the last until ``n_max`` fits.

    Large irregular merges (``max_group_size`` up to 8) can produce P≫512
    supports. Hard-failing on that is worse than growing the P ladder once;
    the extra tier only pads the oversized stamps.
    """
    if int(n_max) <= 0:
        return tuple(sorted(int(t) for t in tiers))
    out = sorted(int(t) for t in tiers)
    if not out:
        raise ValueError("tiers must be non-empty")
    while out[-1] < int(n_max):
        nxt = out[-1] * 2
        if nxt <= out[-1]:
            raise ValueError(f"cannot grow tiers beyond {out[-1]} to cover {n_max}")
        out.append(nxt)
    return tuple(out)


@dataclass
class IrregularStamp:
    """One joint-fit stamp from the membership / pixel-selection agent."""

    member_star_idx: np.ndarray  # (K_g,) int into global star table
    pix_x: np.ndarray  # (P_g,) float detector/crop pixel centers
    pix_y: np.ndarray  # (P_g,) float
    stamp_center_x: float | None = None  # optional primary/window center
    stamp_center_y: float | None = None


@dataclass
class PackedStampBatch:
    """Padded packed supports for one (K_tier, P_tier) bucket."""

    data: np.ndarray  # (n_groups, n_frames, P) electrons/s
    noise: np.ndarray  # (n_groups, n_frames, P)
    weight: np.ndarray  # (n_groups, n_frames, P) pix_valid coverage
    pix_x: np.ndarray  # (n_groups, P)
    pix_y: np.ndarray  # (n_groups, P)
    pix_valid: np.ndarray  # (n_groups, P) float 0/1
    members: np.ndarray  # (n_groups, K) int
    valid: np.ndarray  # (n_groups, K) bool
    k_tier: int
    p_tier: int
    orig_stamp_idx: np.ndarray  # (n_groups,) index into the input stamp list


def _smallest_tier(n: int, tiers: tuple[int, ...]) -> int:
    for t in tiers:
        if int(n) <= int(t):
            return int(t)
    raise ValueError(f"value {n} exceeds max tier {tiers[-1]} in {tiers}")


def validate_stamp_peak_in_support(
    stamp: IrregularStamp,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    *,
    core_margin_px: float = 2.0,
) -> bool:
    """True if every member's reference xy lies in the axis-aligned pix bbox with margin."""
    members = np.asarray(stamp.member_star_idx, dtype=int)
    if members.size == 0 or np.asarray(stamp.pix_x).size == 0:
        return False
    px = np.asarray(stamp.pix_x, dtype=float)
    py = np.asarray(stamp.pix_y, dtype=float)
    x0, x1 = float(px.min()) - core_margin_px, float(px.max()) + core_margin_px
    y0, y1 = float(py.min()) - core_margin_px, float(py.max()) + core_margin_px
    for si in members:
        xi = float(x_ref[int(si)])
        yi = float(y_ref[int(si)])
        if not (x0 <= xi <= x1 and y0 <= yi <= y1):
            return False
    return True


def pack_irregular_stamps(
    stamps: list[IrregularStamp],
    *,
    k_tiers: tuple[int, ...] = DEFAULT_K_TIERS,
    p_tiers: tuple[int, ...] = DEFAULT_P_TIERS,
    n_stars: int | None = None,
    drop_invalid_peak: bool = False,
    x_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    core_margin_px: float = 2.0,
) -> list[tuple[GroupSet, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pad stamps into (K_tier, P_tier) buckets.

    Returns a list of
    ``(groups, pix_x, pix_y, pix_valid, orig_stamp_idx)`` per nonempty bucket,
    sorted by (K_tier, P_tier).
    """
    k_tiers = tuple(sorted(int(t) for t in k_tiers))
    p_tiers = tuple(sorted(int(t) for t in p_tiers))
    if max(k_tiers) > MAX_GROUP_SIZE_CAP:
        raise ValueError(f"k_tiers max {max(k_tiers)} > MAX_GROUP_SIZE_CAP={MAX_GROUP_SIZE_CAP}")

    kept: list[tuple[int, IrregularStamp]] = []
    for si, st in enumerate(stamps):
        if drop_invalid_peak:
            if x_ref is None or y_ref is None:
                raise ValueError("x_ref/y_ref required when drop_invalid_peak=True")
            if not validate_stamp_peak_in_support(st, x_ref, y_ref, core_margin_px=core_margin_px):
                continue
        kept.append((si, st))

    max_p = 0
    max_k = 0
    for _, st in kept:
        kg = int(np.asarray(st.member_star_idx).size)
        pg = int(np.asarray(st.pix_x).size)
        if kg > 0 and pg > 0:
            max_k = max(max_k, kg)
            max_p = max(max_p, pg)
    # Grow K/P ladders so oversized stamps pack instead of hard-crashing export.
    k_tiers = ensure_tiers_cover(max_k, k_tiers)
    if max(k_tiers) > MAX_GROUP_SIZE_CAP:
        raise ValueError(
            f"stamp K={max_k} needs tier {max(k_tiers)} > MAX_GROUP_SIZE_CAP={MAX_GROUP_SIZE_CAP}"
        )
    p_tiers = ensure_tiers_cover(max_p, p_tiers)

    buckets: dict[tuple[int, int], list[tuple[int, IrregularStamp]]] = {}
    for si, st in kept:
        kg = int(np.asarray(st.member_star_idx).size)
        pg = int(np.asarray(st.pix_x).size)
        if kg == 0 or pg == 0:
            continue
        kt = _smallest_tier(kg, k_tiers)
        pt = _smallest_tier(pg, p_tiers)
        buckets.setdefault((kt, pt), []).append((si, st))

    if n_stars is None:
        n_stars = 0
        for _, st in kept:
            if st.member_star_idx.size:
                n_stars = max(n_stars, int(np.max(st.member_star_idx)) + 1)

    out: list[tuple[GroupSet, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for (kt, pt) in sorted(buckets.keys()):
        items = buckets[(kt, pt)]
        n_g = len(items)
        members = np.full((n_g, kt), -1, dtype=np.int32)
        valid = np.zeros((n_g, kt), dtype=bool)
        pix_x = np.zeros((n_g, pt), dtype=np.float32)
        pix_y = np.zeros((n_g, pt), dtype=np.float32)
        pix_valid = np.zeros((n_g, pt), dtype=np.float32)
        orig = np.zeros((n_g,), dtype=np.int32)
        kept_star = np.zeros((n_stars,), dtype=bool)

        for gi, (si, st) in enumerate(items):
            orig[gi] = int(si)
            m = np.asarray(st.member_star_idx, dtype=np.int32).ravel()
            px = np.asarray(st.pix_x, dtype=np.float32).ravel()
            py = np.asarray(st.pix_y, dtype=np.float32).ravel()
            if px.shape != py.shape:
                raise ValueError(f"stamp {si}: pix_x/pix_y length mismatch")
            km, pm = int(m.size), int(px.size)
            members[gi, :km] = m
            valid[gi, :km] = True
            pix_x[gi, :pm] = px
            pix_y[gi, :pm] = py
            pix_valid[gi, :pm] = 1.0
            for idx in m:
                if 0 <= int(idx) < n_stars:
                    kept_star[int(idx)] = True

        groups = GroupSet(n_g, kt, members, valid, kept_star, 0)
        out.append((groups, pix_x, pix_y, pix_valid, orig))
    return out


def gather_packed_pixels(
    frames: list,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    pix_valid: np.ndarray,
    *,
    array_origin: tuple[int, int] = (0, 0),
    scratch_dir: Path | None = None,
    scratch_name: str | None = None,
    frame_chunk: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather ``hp_d`` cal/noise into ``(n_groups, n_frames, P)``.

    Pixel centers are rounded to nearest integer array indices after subtracting
    ``array_origin``. Out-of-bounds or invalid slots get data=0, noise=1, weight=0.
    """
    n_groups, p_tier = pix_x.shape
    n_frames = len(frames)
    ox, oy = int(array_origin[0]), int(array_origin[1])
    ny, nx = frames[0].cal.shape
    shape = (n_groups, n_frames, p_tier)
    if scratch_dir is None:
        data = np.zeros(shape, dtype=np.float32)
        noise = np.ones(shape, dtype=np.float32)
        weight = np.zeros(shape, dtype=np.float32)
    else:
        # Keep the full-orbit tensor on durable disk.  NPY memmaps preserve
        # the ordinary ndarray interface required by PackedStampBatch while
        # preventing all tiers from becoming resident simultaneously.
        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        name = scratch_name or f"g{n_groups}_p{p_tier}"
        data = np.lib.format.open_memmap(scratch_dir / f"{name}_data.npy", mode="w+", dtype=np.float32, shape=shape)
        noise = np.lib.format.open_memmap(scratch_dir / f"{name}_noise.npy", mode="w+", dtype=np.float32, shape=shape)
        # Write the v2 on-disk representation directly.  Fresh NPY files are
        # zero-filled by the filesystem, so padded slots are already inactive
        # without faulting the full tensor into the page cache.
        weight = np.lib.format.open_memmap(scratch_dir / f"{name}_weight.npy", mode="w+", dtype=np.uint8, shape=shape)

    ix = np.rint(pix_x - ox).astype(np.int64)
    iy = np.rint(pix_y - oy).astype(np.int64)
    in_b = (
        (pix_valid > 0.5)
        & (ix >= 0) & (ix < nx)
        & (iy >= 0) & (iy < ny)
    )

    frame_chunk = max(1, int(frame_chunk))
    for f0 in range(0, n_frames, frame_chunk):
        f1 = min(n_frames, f0 + frame_chunk)
        # Vectorize over groups/pixels for each frame.  The temporary arrays
        # are only (groups x P) for one frame, so this removes the extremely
        # slow Python group/pixel loop without changing the bounded-memory
        # design.
        safe_ix = np.clip(ix, 0, nx - 1)
        safe_iy = np.clip(iy, 0, ny - 1)
        for fi in range(f0, f1):
            frame = frames[fi]
            vals = frame.cal[safe_iy, safe_ix]
            nvals = frame.noise[safe_iy, safe_ix]
            valid_now = in_b & (~frame.bad[safe_iy, safe_ix])
            data[:, fi, :] = np.where(valid_now, vals, 0.0)
            noise[:, fi, :] = np.where(valid_now, nvals, 1.0)
            weight[:, fi, :] = (valid_now & (pix_valid > 0.5)).astype(weight.dtype)
        if progress is not None:
            progress(f1, n_frames)

    if scratch_dir is not None:
        data.flush()
        noise.flush()
        weight.flush()

    return data, noise, weight


def build_packed_stamp_batches(
    stamps: list[IrregularStamp],
    frames: list,
    *,
    k_tiers: tuple[int, ...] = DEFAULT_K_TIERS,
    p_tiers: tuple[int, ...] = DEFAULT_P_TIERS,
    array_origin: tuple[int, int] = (0, 0),
    n_stars: int | None = None,
    x_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    drop_invalid_peak: bool = False,
    core_margin_px: float = 2.0,
    scratch_dir: Path | None = None,
    frame_chunk: int = 32,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> list[PackedStampBatch]:
    """Full ingest: pack stamps into tiers and gather frame pixels."""
    packed = pack_irregular_stamps(
        stamps,
        k_tiers=k_tiers,
        p_tiers=p_tiers,
        n_stars=n_stars,
        drop_invalid_peak=drop_invalid_peak,
        x_ref=x_ref,
        y_ref=y_ref,
        core_margin_px=core_margin_px,
    )
    batches: list[PackedStampBatch] = []
    for bucket_i, (groups, pix_x, pix_y, pix_valid, orig) in enumerate(packed):
        data, noise, weight = gather_packed_pixels(
            frames, pix_x, pix_y, pix_valid, array_origin=array_origin,
            scratch_dir=scratch_dir,
            scratch_name=f"tier{bucket_i}_k{groups.max_group_size}_p{pix_x.shape[1]}",
            frame_chunk=frame_chunk,
            progress=(
                (lambda done, total, bi=bucket_i, ng=groups.n_groups, pt=pix_x.shape[1]:
                 progress(bi, done, total, int(ng * pt)))
                if progress is not None else None
            ),
        )
        # coverage already in weight; ensure pad stays zero
        weight = weight * pix_valid[:, None, :]
        batches.append(
            PackedStampBatch(
                data=data,
                noise=noise,
                weight=weight,
                pix_x=pix_x,
                pix_y=pix_y,
                pix_valid=pix_valid,
                members=groups.members,
                valid=groups.valid,
                k_tier=int(groups.max_group_size),
                p_tier=int(pix_x.shape[1]),
                orig_stamp_idx=orig,
            )
        )
    return batches


def build_packed_stamp_batches_single_pass(
    stamps: list[IrregularStamp],
    frames,
    *,
    k_tiers: tuple[int, ...] = DEFAULT_K_TIERS,
    p_tiers: tuple[int, ...] = DEFAULT_P_TIERS,
    array_origin: tuple[int, int] = (0, 0),
    n_stars: int | None = None,
    x_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    drop_invalid_peak: bool = False,
    core_margin_px: float = 2.0,
    scratch_dir: Path | None = None,
    frame_chunk: int = 32,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> list[PackedStampBatch]:
    """Pack and gather all tiers while opening each frame exactly once.

    The legacy implementation traverses the frame sequence once per tier.  For
    FITS-backed streams that causes the same ``hp_d`` file to be reopened ten
    times.  This implementation allocates the tier outputs first, then loads
    each frame once and scatters its pixels into every tier.  Outputs retain
    the same tier-native shapes and may be disk-backed through ``scratch_dir``.
    """
    packed = pack_irregular_stamps(
        stamps, k_tiers=k_tiers, p_tiers=p_tiers, n_stars=n_stars,
        drop_invalid_peak=drop_invalid_peak, x_ref=x_ref, y_ref=y_ref,
        core_margin_px=core_margin_px,
    )
    if not packed:
        return []
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("cannot gather packed stamps from zero frames")
    first = frames[0]
    ny, nx = first.cal.shape
    ox, oy = int(array_origin[0]), int(array_origin[1])
    stores = []
    for bucket_i, (groups, pix_x, pix_y, pix_valid, orig) in enumerate(packed):
        shape = (groups.n_groups, n_frames, pix_x.shape[1])
        if scratch_dir is None:
            data = np.zeros(shape, dtype=np.float32)
            noise = np.ones(shape, dtype=np.float32)
            weight = np.zeros(shape, dtype=np.uint8)
        else:
            scratch_dir = Path(scratch_dir)
            scratch_dir.mkdir(parents=True, exist_ok=True)
            name = f"single_tier{bucket_i}_k{groups.max_group_size}_p{pix_x.shape[1]}"
            data = np.lib.format.open_memmap(scratch_dir / f"{name}_data.npy", mode="w+", dtype=np.float32, shape=shape)
            noise = np.lib.format.open_memmap(scratch_dir / f"{name}_noise.npy", mode="w+", dtype=np.float32, shape=shape)
            weight = np.lib.format.open_memmap(scratch_dir / f"{name}_weight.npy", mode="w+", dtype=np.uint8, shape=shape)
        ix = np.rint(pix_x - ox).astype(np.int64)
        iy = np.rint(pix_y - oy).astype(np.int64)
        in_b = ((pix_valid > 0.5) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny))
        safe_ix = np.clip(ix, 0, nx - 1)
        safe_iy = np.clip(iy, 0, ny - 1)
        stores.append((groups, pix_x, pix_y, pix_valid, orig, data, noise, weight, safe_ix, safe_iy, in_b))

    # Concatenate tier coordinates so each frame performs one NumPy gather for
    # cal/noise/mask instead of one gather per tier.  Slices map the result back
    # to each tier's native (group, P) layout.
    offsets = [0]
    for s in stores:
        offsets.append(offsets[-1] + int(s[0].n_groups * s[1].shape[1]))
    all_ix = np.concatenate([s[8].reshape(-1) for s in stores])
    all_iy = np.concatenate([s[9].reshape(-1) for s in stores])
    all_in = np.concatenate([s[10].reshape(-1) for s in stores])

    for fi in range(n_frames):
        # Reuse the shape-probe frame for fi=0; every subsequent frame is
        # opened exactly once by the streaming sequence.
        frame = first if fi == 0 else frames[fi]
        all_bad = frame.bad[all_iy, all_ix]
        valid_all = all_in & (~all_bad)
        vals_all = np.where(valid_all, frame.cal[all_iy, all_ix], 0.0)
        noise_all = np.where(valid_all, frame.noise[all_iy, all_ix], 1.0)
        for si, (groups, pix_x, pix_y, pix_valid, orig, data, noise, weight, safe_ix, safe_iy, in_b) in enumerate(stores):
            lo, hi = offsets[si], offsets[si + 1]
            shape = (groups.n_groups, pix_x.shape[1])
            data[:, fi, :] = vals_all[lo:hi].reshape(shape)
            noise[:, fi, :] = noise_all[lo:hi].reshape(shape)
            weight[:, fi, :] = (valid_all[lo:hi].reshape(shape) & (pix_valid > 0.5)).astype(np.uint8)
        if progress is not None and (fi + 1 == n_frames or (fi + 1) % 256 == 0):
            total_cells = sum(int(s[0].n_groups * s[1].shape[1]) for s in stores)
            progress(-1, fi + 1, n_frames, total_cells)

    batches = []
    for groups, pix_x, pix_y, pix_valid, orig, data, noise, weight, *_ in stores:
        if scratch_dir is not None:
            data.flush(); noise.flush(); weight.flush()
        weight = weight * pix_valid[:, None, :]
        batches.append(PackedStampBatch(
            data=data, noise=noise, weight=weight, pix_x=pix_x, pix_y=pix_y,
            pix_valid=pix_valid, members=groups.members, valid=groups.valid,
            k_tier=int(groups.max_group_size), p_tier=int(pix_x.shape[1]),
            orig_stamp_idx=orig,
        ))
    return batches


def irregular_stamps_from_assignments(assignments) -> list[IrregularStamp]:
    """Convert ``irregular_stamps.SegmentAssignment`` rows to ``IrregularStamp``.

    Requires each assignment to have non-empty ``pix_x`` / ``pix_y`` and
    ``member_indices``. Pixel coordinates stay in the same detector/crop frame
    used by the segmenter (full-FFI or region-local).
    """
    out: list[IrregularStamp] = []
    for a in assignments:
        mem = np.asarray(getattr(a, "member_indices"), dtype=np.int32).ravel()
        px = getattr(a, "pix_x", None)
        py = getattr(a, "pix_y", None)
        if px is None or py is None:
            raise ValueError(
                "SegmentAssignment missing pix_x/pix_y; run build_epsf_support_stamps "
                "(or mask_to_pix_xy) before packing"
            )
        px = np.asarray(px, dtype=np.float32).ravel()
        py = np.asarray(py, dtype=np.float32).ravel()
        if mem.size == 0 or px.size == 0:
            continue
        if px.shape != py.shape:
            raise ValueError("assignment pix_x/pix_y length mismatch")
        cx = getattr(a, "stamp_center_x", None)
        cy = getattr(a, "stamp_center_y", None)
        out.append(
            IrregularStamp(
                member_star_idx=mem,
                pix_x=px,
                pix_y=py,
                stamp_center_x=float(cx) if cx is not None else None,
                stamp_center_y=float(cy) if cy is not None else None,
            )
        )
    return out


def irregular_stamps_from_mask_dir(mask_dir) -> list[IrregularStamp]:
    """Load ``mask_primary_*_seg*.npz`` exports from the segmentation notebook."""
    from pathlib import Path

    mask_dir = Path(mask_dir)
    paths = sorted(mask_dir.glob("mask_primary_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no mask_primary_*.npz under {mask_dir}")
    stamps: list[IrregularStamp] = []
    for p in paths:
        z = np.load(p)
        mem = np.asarray(z["member_indices"], dtype=np.int32).ravel()
        px = np.asarray(z["pix_x"], dtype=np.float32).ravel()
        py = np.asarray(z["pix_y"], dtype=np.float32).ravel()
        cx = float(z["stamp_center_x"]) if "stamp_center_x" in z.files else None
        cy = float(z["stamp_center_y"]) if "stamp_center_y" in z.files else None
        if mem.size == 0 or px.size == 0:
            continue
        stamps.append(
            IrregularStamp(
                member_star_idx=mem, pix_x=px, pix_y=py,
                stamp_center_x=cx, stamp_center_y=cy,
            )
        )
    return stamps


def stamp_centers_from_irregular(
    stamps: list[IrregularStamp],
    x_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integer stamp centers: stored primary center, else mean of member xy."""
    n = len(stamps)
    cx = np.zeros(n, dtype=np.float32)
    cy = np.zeros(n, dtype=np.float32)
    for i, st in enumerate(stamps):
        if st.stamp_center_x is not None and st.stamp_center_y is not None:
            cx[i] = float(st.stamp_center_x)
            cy[i] = float(st.stamp_center_y)
            continue
        mem = np.asarray(st.member_star_idx, dtype=int)
        if x_ref is None or y_ref is None or mem.size == 0:
            # Fall back to pixel-support centroid.
            px = np.asarray(st.pix_x, dtype=float)
            py = np.asarray(st.pix_y, dtype=float)
            cx[i] = float(np.round(px.mean())) if px.size else 0.0
            cy[i] = float(np.round(py.mean())) if py.size else 0.0
        else:
            cx[i] = float(np.round(np.mean(x_ref[mem])))
            cy[i] = float(np.round(np.mean(y_ref[mem])))
    return cx, cy


def concat_packed_batches(
    batches: list[PackedStampBatch],
    *,
    stamp_center_x: np.ndarray,
    stamp_center_y: np.ndarray,
    n_stars: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int,
]:
    """Flatten tier batches into one FitBundle-shaped packed block.

    Pads every group to ``max(K_tier)`` and ``max(P_tier)`` across batches.
    ``stamp_center_*`` must be indexed by each batch's ``orig_stamp_idx``.

    Returns
    ``(data, noise, weight, pix_x, pix_y, pix_valid, members, valid,
      cx, cy, kept_star_mask, max_group_size)``.
    """
    if not batches:
        raise ValueError("no packed batches to concatenate")
    k_max = max(int(b.k_tier) for b in batches)
    p_max = max(int(b.p_tier) for b in batches)
    n_frames = int(batches[0].data.shape[1])
    parts = []
    for b in batches:
        n_g = int(b.data.shape[0])
        data = np.zeros((n_g, n_frames, p_max), dtype=np.float32)
        noise = np.ones((n_g, n_frames, p_max), dtype=np.float32)
        weight = np.zeros((n_g, n_frames, p_max), dtype=np.float32)
        pix_x = np.zeros((n_g, p_max), dtype=np.float32)
        pix_y = np.zeros((n_g, p_max), dtype=np.float32)
        pix_valid = np.zeros((n_g, p_max), dtype=np.float32)
        members = np.full((n_g, k_max), -1, dtype=np.int32)
        valid = np.zeros((n_g, k_max), dtype=bool)
        pt, kt = int(b.p_tier), int(b.k_tier)
        data[:, :, :pt] = b.data
        noise[:, :, :pt] = b.noise
        weight[:, :, :pt] = b.weight
        pix_x[:, :pt] = b.pix_x
        pix_y[:, :pt] = b.pix_y
        pix_valid[:, :pt] = b.pix_valid
        members[:, :kt] = b.members
        valid[:, :kt] = b.valid
        cx = np.asarray(stamp_center_x, dtype=np.float32)[b.orig_stamp_idx]
        cy = np.asarray(stamp_center_y, dtype=np.float32)[b.orig_stamp_idx]
        parts.append((data, noise, weight, pix_x, pix_y, pix_valid, members, valid, cx, cy))

    data = np.concatenate([p[0] for p in parts], axis=0)
    noise = np.concatenate([p[1] for p in parts], axis=0)
    weight = np.concatenate([p[2] for p in parts], axis=0)
    pix_x = np.concatenate([p[3] for p in parts], axis=0)
    pix_y = np.concatenate([p[4] for p in parts], axis=0)
    pix_valid = np.concatenate([p[5] for p in parts], axis=0)
    members = np.concatenate([p[6] for p in parts], axis=0)
    valid = np.concatenate([p[7] for p in parts], axis=0)
    cx = np.concatenate([p[8] for p in parts], axis=0)
    cy = np.concatenate([p[9] for p in parts], axis=0)
    kept = np.zeros((int(n_stars),), dtype=bool)
    for gi in range(members.shape[0]):
        for si in members[gi][valid[gi]]:
            if 0 <= int(si) < n_stars:
                kept[int(si)] = True
    return data, noise, weight, pix_x, pix_y, pix_valid, members, valid, cx, cy, kept, k_max


def packed_tiers_from_batches(batches: list[PackedStampBatch]):
    """Build v2 ``PackedTier`` rows directly, without global-P arrays.

    Group ordering is exactly the ordering used by :func:`concat_packed_batches`:
    batches in input order and rows within each batch.  Thus these tiers can
    replace the dense arrays on a bundle whose group metadata came from that
    concat operation, while avoiding a second dense rebucketing pass.
    """
    if not batches:
        raise ValueError("no packed batches")
    from .fit_bundle import PackedTier

    n_frames = int(batches[0].data.shape[1])
    offset = 0
    tiers = []
    for i, batch in enumerate(batches):
        ng, pt, kt = int(batch.data.shape[0]), int(batch.p_tier), int(batch.k_tier)
        if batch.data.shape != (ng, n_frames, pt):
            raise ValueError(f"batch {i}: data shape inconsistent with p_tier")
        if batch.noise.shape != batch.data.shape or batch.weight.shape != batch.data.shape:
            raise ValueError(f"batch {i}: data/noise/weight shape mismatch")
        if batch.pix_x.shape != (ng, pt) or batch.pix_y.shape != (ng, pt):
            raise ValueError(f"batch {i}: pix coordinate shape mismatch")
        if batch.pix_valid.shape != (ng, pt):
            raise ValueError(f"batch {i}: pix_valid shape mismatch")
        weight = np.asarray(batch.weight)
        if np.any((weight != 0) & (weight != 1)):
            raise ValueError(f"batch {i}: weight must be binary for uint8 v2 storage")
        tiers.append(PackedTier(
            data=np.ascontiguousarray(batch.data, dtype=np.float32),
            noise=np.ascontiguousarray(batch.noise, dtype=np.float32),
            weight_u8=np.ascontiguousarray(weight, dtype=np.uint8),
            pix_x=np.ascontiguousarray(batch.pix_x, dtype=np.float32),
            pix_y=np.ascontiguousarray(batch.pix_y, dtype=np.float32),
            pix_valid=np.ascontiguousarray(batch.pix_valid, dtype=np.float32),
            group_idx=np.arange(offset, offset + ng, dtype=np.int32),
            k_tier=kt, p_tier=pt,
        ))
        offset += ng
    return tiers


def bucket_packed_by_kp(
    groups: GroupSet,
    stamp_center_x: np.ndarray,
    stamp_center_y: np.ndarray,
    pix_valid: np.ndarray,
    *,
    k_tiers: tuple[int, ...] = DEFAULT_K_TIERS,
    p_tiers: tuple[int, ...] = DEFAULT_P_TIERS,
) -> list[tuple[GroupSet, np.ndarray, np.ndarray, np.ndarray, int, int]]:
    """Split packed groups into nonempty ``(K_tier, P_tier)`` buckets.

    Returns ``(bucket_groups, cx, cy, orig_idx, k_tier, p_tier)`` per bucket.
    ``bucket_groups`` is padded to ``k_tier``; callers should also slice
    ``pix_*`` / ``data`` to ``p_tier`` via ``orig_idx``.
    """
    k_tiers = tuple(sorted(int(t) for t in k_tiers))
    p_tiers = tuple(sorted(int(t) for t in p_tiers))
    if groups.n_groups == 0:
        return []
    sizes_k = np.asarray(groups.valid.sum(axis=1), dtype=int)
    sizes_p = np.asarray((np.asarray(pix_valid) > 0.5).sum(axis=1), dtype=int)
    k_tiers = ensure_tiers_cover(int(sizes_k.max()) if sizes_k.size else 0, k_tiers)
    p_tiers = ensure_tiers_cover(int(sizes_p.max()) if sizes_p.size else 0, p_tiers)
    n_stars = int(groups.kept_star_mask.shape[0])
    buckets: dict[tuple[int, int], list[int]] = {}
    for gi in range(groups.n_groups):
        kt = _smallest_tier(int(sizes_k[gi]), k_tiers)
        pt = _smallest_tier(max(int(sizes_p[gi]), 1), p_tiers)
        buckets.setdefault((kt, pt), []).append(gi)

    out: list[tuple[GroupSet, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for (kt, pt) in sorted(buckets.keys()):
        idx = np.asarray(buckets[(kt, pt)], dtype=np.int32)
        n_g = int(idx.size)
        members = np.full((n_g, kt), -1, dtype=np.int32)
        valid = np.zeros((n_g, kt), dtype=bool)
        kept = np.zeros((n_stars,), dtype=bool)
        for bi, gi in enumerate(idx):
            m = groups.members[gi][groups.valid[gi]]
            km = min(int(m.size), kt)
            members[bi, :km] = m[:km]
            valid[bi, :km] = True
            for si in m[:km]:
                if 0 <= int(si) < n_stars:
                    kept[int(si)] = True
        bg = GroupSet(n_g, kt, members, valid, kept, 0)
        out.append((
            bg,
            np.asarray(stamp_center_x)[idx],
            np.asarray(stamp_center_y)[idx],
            idx,
            int(kt),
            int(pt),
        ))
    return out
