# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""ePSF-sized stamp assembly for packed / irregular fit supports.

Preferred workflow (``build_epsf_support_stamps``):

1. Select Gaia primaries in a mag window isolated by ``min_sep_px`` from all
   stars with ``tess_mag < bright_mag_max``.
2. Detect on ``|hp_d|/NOISE`` (σ=5), then **erode** the binary mask
   (default 1 px) so 1–2 px bridges split into separate labels; no dilation.
3. **One stamp per eroded segment** that contains ≥1 primary: members = every
   Gaia star with ``tess_mag < bright_mag_max`` on that segment (including any
   other co-segment primaries). If ``K > max_group_size`` (default 4), reject
   the stamp. Primaries on background (no segment) each get a solo fallback.
4. Size odd squares by growing from ``S_min``: assign in-reach eroded-segment
   pixels to the nearest member and take the smallest enclosing square
   (no blind square pad — that overshoots into neighboring islands).
5. If a sized square covers pixels of a **foreign** eroded label, merge those
   islands into one stamp (reject if ``K > max_group_size``).
6. Scored pixels = union of those full squares (no SNR prune of corners).
7. **Exclusive pixels:** if two stamps share scored pixels, each contested
   pixel is kept only in the stamp whose brightest member is closest
   (Euclidean); losers lose that pixel.

Legacy full-field ``detect_merged_segments`` / ``assign_segment_members`` remain
for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass
class MergedSegmentation:
    """Merged label map on a region crop (0 = background)."""

    label_map: np.ndarray  # (ny, nx) int
    n_labels: int
    n_sigma: float
    pad_px: int
    npixels: int
    erode_px: int = 0


@dataclass
class SegmentAssignment:
    """One primary + its support mask + joint-fit Gaia members."""

    primary_index: int  # index into the full Gaia-aligned x/y/mag arrays
    segment_label: int
    mask: np.ndarray  # (ny, nx) bool, crop-local
    member_indices: np.ndarray  # indices into the same Gaia arrays (includes primary)
    n_pixels: int
    stamp_center_x: int = 0  # integer detector/crop center of the primary window
    stamp_center_y: int = 0
    pix_x: np.ndarray | None = None  # (P,) float pixel centers (detector / +region)
    pix_y: np.ndarray | None = None
    member_stamp_sizes: np.ndarray | None = None  # (K,) odd S per member
    bbox: tuple[int, int, int, int] | None = None  # crop-local (y0, y1, x0, x1) half-open


DEFAULT_STAMP_PHYSICAL = 13
DEFAULT_STAMP_MIN = 5


def detect_merged_segments(
    cal: np.ndarray,
    noise: np.ndarray,
    *,
    n_sigma: float = 3.0,
    npixels: int = 5,
    pad_px: int = 1,
    erode_px: int = 0,
    connectivity: int = 8,
) -> MergedSegmentation:
    """photutils detect on ``|cal|/noise``, then optional dilate and/or erode.

    ``cal`` / ``noise`` are region-crop arrays (ny, nx). Detection uses the
    absolute SNR map so signed difference-image residuals are treated as flux.
    Stamp assembly uses ``pad_px=0`` and ``erode_px>=1`` so thin bridges split
    before labeling; ``pad_px>0`` remains for legacy merge diagnostics.
    """
    from photutils.segmentation import detect_sources

    cal = np.asarray(cal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if cal.shape != noise.shape:
        raise ValueError(f"cal/noise shape mismatch: {cal.shape} vs {noise.shape}")

    snr = np.abs(cal) / (noise + 1e-12)
    try:
        segm = detect_sources(
            snr, threshold=float(n_sigma), n_pixels=int(npixels), connectivity=int(connectivity),
        )
    except TypeError:
        segm = detect_sources(
            snr, threshold=float(n_sigma), npixels=int(npixels), connectivity=int(connectivity),
        )
    if segm is None:
        label_map = np.zeros(cal.shape, dtype=np.int32)
        return MergedSegmentation(
            label_map, 0, float(n_sigma), int(pad_px), int(npixels), erode_px=int(erode_px),
        )

    binary = segm.data > 0
    structure = ndi.generate_binary_structure(2, 2 if connectivity >= 8 else 1)
    if pad_px > 0:
        binary = ndi.binary_dilation(binary, structure=structure, iterations=int(pad_px))
    if erode_px > 0:
        binary = ndi.binary_erosion(binary, structure=structure, iterations=int(erode_px))
    label_map, n_labels = ndi.label(binary, structure=structure)
    return MergedSegmentation(
        label_map=np.asarray(label_map, dtype=np.int32),
        n_labels=int(n_labels),
        n_sigma=float(n_sigma),
        pad_px=int(pad_px),
        npixels=int(npixels),
        erode_px=int(erode_px),
    )


def select_isolated_primaries(
    x: np.ndarray,
    y: np.ndarray,
    mag: np.ndarray,
    *,
    mag_lo: float,
    mag_hi: float,
    bright_mag_max: float = 13.0,
    min_sep_px: float = 6.0,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    edge_margin_px: float = 0.0,
) -> np.ndarray:
    """Indices of mag-window stars isolated by ``min_sep_px`` from all bright neighbors.

    Isolation neighbors are all stars with ``mag < bright_mag_max`` (and finite
    xy). Optional region bounds keep primaries inset by ``edge_margin_px``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag = np.asarray(mag, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(mag)
    in_region = np.ones(len(x), dtype=bool)
    if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
        m = float(edge_margin_px)
        in_region = (
            (x >= float(x_min) + m)
            & (x < float(x_max) - m)
            & (y >= float(y_min) + m)
            & (y < float(y_max) - m)
        )
    cand = finite & in_region & (mag >= float(mag_lo)) & (mag <= float(mag_hi))
    neigh = finite & in_region & (mag < float(bright_mag_max))
    if int(cand.sum()) == 0:
        return np.zeros(0, dtype=int)
    xy_n = np.column_stack([x[neigh], y[neigh]])
    if xy_n.shape[0] < 2:
        return np.flatnonzero(cand)

    tree = cKDTree(xy_n)
    dist, _ = tree.query(np.column_stack([x[cand], y[cand]]), k=2)
    nn = dist[:, 1] if dist.ndim == 2 else np.full(int(cand.sum()), np.inf)
    keep_local = nn >= float(min_sep_px)
    return np.flatnonzero(cand)[keep_local]


def _pixel_at(x: float, y: float, shape: tuple[int, int], *, x0: float, y0: float) -> tuple[int, int] | None:
    """Crop-local float xy → integer pixel in label_map coords, or None if OOB."""
    ix = int(np.floor(float(x) - float(x0)))
    iy = int(np.floor(float(y) - float(y0)))
    ny, nx = shape
    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
        return None
    return iy, ix


def stamp_window_slices(
    cx: int,
    cy: int,
    *,
    stamp_physical: int = DEFAULT_STAMP_PHYSICAL,
    ny: int,
    nx: int,
    region_x_min: int = 0,
    region_y_min: int = 0,
) -> tuple[slice, slice, int, int, int, int]:
    """Return crop-array slices for an odd S×S window centered on (cx, cy).

    ``cx, cy`` are crop-local detector coordinates (same frame as warmstart xy).
    Returns ``(yslice, xslice, x0_full, x1_full, y0_full, y1_full)`` where the
    full-frame half-open bounds are clipped to the region crop.
    """
    s = int(stamp_physical)
    if s % 2 == 0 or s < 1:
        raise ValueError(f"stamp_physical must be odd and positive, got {s}")
    h = s // 2
    x0 = int(cx) - h
    x1 = int(cx) + h + 1
    y0 = int(cy) - h
    y1 = int(cy) + h + 1
    x0c = max(x0, int(region_x_min))
    x1c = min(x1, int(region_x_min) + int(nx))
    y0c = max(y0, int(region_y_min))
    y1c = min(y1, int(region_y_min) + int(ny))
    xs = slice(x0c - int(region_x_min), x1c - int(region_x_min))
    ys = slice(y0c - int(region_y_min), y1c - int(region_y_min))
    return ys, xs, x0c, x1c, y0c, y1c


def paint_stamp_window(
    mask: np.ndarray,
    cx: int,
    cy: int,
    *,
    stamp_physical: int = DEFAULT_STAMP_PHYSICAL,
    region_x_min: int = 0,
    region_y_min: int = 0,
) -> None:
    """OR an S×S window centered on (cx, cy) into ``mask`` (crop-local bool)."""
    ny, nx = mask.shape
    ys, xs, *_ = stamp_window_slices(
        cx, cy, stamp_physical=stamp_physical, ny=ny, nx=nx,
        region_x_min=region_x_min, region_y_min=region_y_min,
    )
    if ys.start < ys.stop and xs.start < xs.stop:
        mask[ys, xs] = True


def chebyshev_sep(x0: float, y0: float, x1: float, y1: float) -> float:
    return float(max(abs(float(x0) - float(x1)), abs(float(y0) - float(y1))))


def contaminates_primary_stamp(
    primary_x: float,
    primary_y: float,
    companion_x: float,
    companion_y: float,
    *,
    stamp_physical: int = DEFAULT_STAMP_PHYSICAL,
) -> bool:
    """True if companion peak is in / overlaps the primary S×S (L∞ ≤ S-1)."""
    s = int(stamp_physical)
    return chebyshev_sep(primary_x, primary_y, companion_x, companion_y) <= float(s - 1)


def mask_to_pix_xy(
    mask: np.ndarray,
    *,
    region_x_min: float = 0.0,
    region_y_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean crop mask → float pixel-center coordinates (crop-local)."""
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    pix_x = xs.astype(np.float64) + float(region_x_min)
    pix_y = ys.astype(np.float64) + float(region_y_min)
    return pix_x, pix_y


def prune_mask_by_snr(
    mask: np.ndarray,
    cal: np.ndarray,
    noise: np.ndarray,
    *,
    n_sigma: float = 5.0,
) -> np.ndarray:
    """Keep mask pixels with ``|cal| >= n_sigma * noise`` (legacy optional path)."""
    keep = np.asarray(mask, dtype=bool) & (
        np.abs(np.asarray(cal, dtype=np.float64))
        >= float(n_sigma) * (np.asarray(noise, dtype=np.float64) + 1e-12)
    )
    return keep


def outer_ring_all_outside_label(
    label_map: np.ndarray,
    cx: int,
    cy: int,
    stamp_physical: int,
    star_label: int,
    *,
    region_x_min: int = 0,
    region_y_min: int = 0,
) -> bool:
    """True if every in-bounds outer-ring pixel is not ``star_label``.

    Outer ring = pixels in the S×S window excluding the inner (S-2)×(S-2).
    If there are no in-bounds outer pixels (degenerate), returns False (do not shrink).
    """
    s = int(stamp_physical)
    if s < 3 or star_label <= 0:
        return False
    label_map = np.asarray(label_map)
    ny, nx = label_map.shape
    h = s // 2
    x0 = int(cx) - h - int(region_x_min)
    y0 = int(cy) - h - int(region_y_min)
    # Full S×S in array coords (may be partially OOB).
    n_in = 0
    all_outside = True
    for dy in range(s):
        for dx in range(s):
            on_outer = dy == 0 or dy == s - 1 or dx == 0 or dx == s - 1
            if not on_outer:
                continue
            ix = x0 + dx
            iy = y0 + dy
            if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
                continue
            n_in += 1
            if int(label_map[iy, ix]) == int(star_label):
                all_outside = False
                break
        if not all_outside:
            break
    if n_in == 0:
        return False
    return all_outside


def shrink_square_size(
    label_map: np.ndarray,
    cx: int,
    cy: int,
    star_label: int,
    *,
    s_max: int = DEFAULT_STAMP_PHYSICAL,
    s_min: int = DEFAULT_STAMP_MIN,
    region_x_min: int = 0,
    region_y_min: int = 0,
) -> int:
    """Odd square size in ``[s_min, s_max]`` by shrinking empty outer rings.

    Legacy helper; prefer ``grow_enclosing_square_size``. If ``star_label <= 0``
    (peak on background), returns ``s_max``.
    """
    s_max = int(s_max)
    s_min = int(s_min)
    if s_max % 2 == 0 or s_min % 2 == 0:
        raise ValueError("s_max and s_min must be odd")
    if star_label <= 0:
        return s_max
    s = s_max
    while s > s_min:
        if outer_ring_all_outside_label(
            label_map, cx, cy, s, star_label,
            region_x_min=region_x_min, region_y_min=region_y_min,
        ):
            s -= 2
        else:
            break
    return int(s)


def grow_enclosing_square_size(
    pixel_iy: np.ndarray,
    pixel_ix: np.ndarray,
    cx: int,
    cy: int,
    *,
    s_max: int = DEFAULT_STAMP_PHYSICAL,
    s_min: int = DEFAULT_STAMP_MIN,
    region_x_min: int = 0,
    region_y_min: int = 0,
    enclose_pad_px: int = 0,
) -> int:
    """Smallest odd ``S∈[s_min,s_max]`` whose square covers the given pixels.

    ``pixel_iy/ix`` are crop-array indices. ``cx,cy`` are detector-frame integer
    centers (same frame as warmstart ``round(x), round(y)``). Grows from
    ``s_min`` upward. After the tight enclose size, optionally add
    ``2 * enclose_pad_px`` (one pixel border on each side) then clamp to
    ``s_max``. Empty pixel list → ``s_min`` (then pad/clamp).
    """
    s_max = int(s_max)
    s_min = int(s_min)
    pad = int(enclose_pad_px)
    if s_max % 2 == 0 or s_min % 2 == 0:
        raise ValueError("s_max and s_min must be odd")
    if len(pixel_iy) == 0:
        s = s_min
    else:
        cx_arr = int(cx) - int(region_x_min)
        cy_arr = int(cy) - int(region_y_min)
        d = np.maximum(
            np.abs(np.asarray(pixel_ix, dtype=int) - cx_arr),
            np.abs(np.asarray(pixel_iy, dtype=int) - cy_arr),
        )
        d_max = int(d.max())
        s_need = 2 * d_max + 1
        if s_need % 2 == 0:
            s_need += 1
        s = max(s_min, min(s_max, s_need))
    if pad > 0:
        s = min(s_max, s + 2 * pad)
        if s % 2 == 0:
            s = min(s_max, s + 1)
    return int(s)


def square_stamp_pixels(
    cx: float,
    cy: float,
    *,
    stamp_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the complete integer-pixel footprint of an odd square stamp.

    Coordinates are in the detector frame (not crop-array coordinates), which
    makes the result suitable for comparing irregular supports and compact
    faint-anchor squares without a lossy mask conversion.
    """
    s = int(stamp_size)
    if s < 1 or s % 2 == 0:
        raise ValueError(f"stamp_size must be an odd positive integer, got {stamp_size}")
    x0 = int(round(float(cx))) - s // 2
    y0 = int(round(float(cy))) - s // 2
    yy, xx = np.mgrid[y0:y0 + s, x0:x0 + s]
    return xx.reshape(-1).astype(np.float32), yy.reshape(-1).astype(np.float32)


def select_isolated_faint_anchor_stamps(
    x: np.ndarray,
    y: np.ndarray,
    tess_mag: np.ndarray,
    *,
    faint_mag_range: tuple[float, float] = (11.0, 13.0),
    catalog_mag_max: float = 13.0,
    isolation_radius_px: float = 5.0,
    stamp_size: int = 7,
    claimed_pixels: set[tuple[int, int]] | None = None,
    candidate_indices: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[list["SegmentAssignment"], dict[str, int]]:
    """Choose strict, single-star faint WCS anchors.

    Each survivor has an exact square footprint and is rejected when either
    (a) *any* catalog star through ``catalog_mag_max`` lies within the Euclidean
    isolation radius, or (b) one of its pixels was already claimed by an
    irregular bright support or an earlier (brighter) faint anchor.  The
    pixel set is updated in-place when supplied so callers can share it with
    their bright ePSF supports.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag = np.asarray(tess_mag, dtype=float)
    lo, hi = (float(v) for v in faint_mag_range)
    if not (lo < hi):
        raise ValueError(f"invalid faint_mag_range={faint_mag_range}")
    if int(stamp_size) > 7:
        raise ValueError("faint WCS anchor stamps must be compact (stamp_size <= 7)")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(mag)
    catalog = np.flatnonzero(finite & (mag <= float(catalog_mag_max)))
    candidates = np.flatnonzero(finite & (mag > lo) & (mag <= hi))
    if candidate_indices is not None:
        candidates = np.intersect1d(candidates, np.asarray(candidate_indices, dtype=int))
    # Deterministic first-claim policy: brightest (smallest magnitude) first.
    candidates = candidates[np.argsort(mag[candidates], kind="stable")]
    claimed = claimed_pixels if claimed_pixels is not None else set()
    stats = {
        "n_candidates": int(candidates.size),
        "rejected_catalog_proximity": 0,
        "rejected_pixel_overlap": 0,
        "rejected_out_of_bounds": 0,
        "retained": 0,
    }
    accepted: list[SegmentAssignment] = []
    r2 = float(isolation_radius_px) ** 2
    for idx in candidates:
        other = catalog[catalog != int(idx)]
        if other.size:
            d2 = (x[other] - x[idx]) ** 2 + (y[other] - y[idx]) ** 2
            if np.any(d2 < r2):
                stats["rejected_catalog_proximity"] += 1
                continue
        px, py = square_stamp_pixels(x[idx], y[idx], stamp_size=int(stamp_size))
        pixels = {(int(round(px_i)), int(round(py_i))) for px_i, py_i in zip(px, py)}
        if bounds is not None:
            x0, y0, x1, y1 = (float(v) for v in bounds)
            if any(not (x0 <= xx < x1 and y0 <= yy < y1) for xx, yy in pixels):
                stats["rejected_out_of_bounds"] += 1
                continue
        if pixels & claimed:
            stats["rejected_pixel_overlap"] += 1
            continue
        claimed.update(pixels)
        cx, cy = int(round(x[idx])), int(round(y[idx]))
        accepted.append(SegmentAssignment(
            primary_index=int(idx),
            segment_label=0,
            mask=np.zeros((0, 0), dtype=bool),
            member_indices=np.asarray([idx], dtype=np.int32),
            n_pixels=int(px.size),
            stamp_center_x=cx,
            stamp_center_y=cy,
            pix_x=px,
            pix_y=py,
            member_stamp_sizes=np.asarray([int(stamp_size)], dtype=np.int32),
        ))
        stats["retained"] += 1
    return accepted, stats


def cap_faint_anchors_by_mag(
    faint_assignments: list["SegmentAssignment"],
    tess_mag: np.ndarray,
    *,
    n_bright: int,
    max_stars: int,
) -> tuple[list["SegmentAssignment"], dict]:
    """Trim ``faint_assignments`` so ``n_bright + len(result) <= max_stars``.

    ``select_isolated_faint_anchor_stamps`` processes candidates
    brightest-first (stable sort by ``tess_mag``) and appends winners in that
    order, so ``faint_assignments`` arrives already brightest-to-faintest --
    truncating from the end keeps the brightest and drops the faintest,
    without needing to re-sort. Bright irregular-group primaries/companions
    (``n_bright``) are never touched here: they keep full contamination
    treatment regardless of the cap.

    ``max_stars <= 0`` disables the cap (returns the input unchanged).
    Returns ``(kept_assignments, stats)`` where ``stats`` has ``capped``
    (bool), ``dropped`` (int), and ``effective_mag_hi`` (float, the faintest
    kept mag, or the input's own lower bound if everything was dropped).
    """
    n_faint_before = len(faint_assignments)
    if int(max_stars) <= 0 or (n_bright + n_faint_before) <= int(max_stars):
        return faint_assignments, {"capped": False, "dropped": 0, "effective_mag_hi": None}

    mag = np.asarray(tess_mag, dtype=float)
    keep_faint = max(0, int(max_stars) - int(n_bright))
    dropped = n_faint_before - keep_faint
    kept = faint_assignments[:keep_faint]
    kept_mags = [float(mag[a.primary_index]) for a in kept]
    stats = {
        "capped": True,
        "dropped": int(dropped),
        "effective_mag_hi": max(kept_mags) if kept_mags else None,
    }
    return kept, stats


def member_stamp_sizes_for_segment(
    label_map: np.ndarray,
    member_cx: np.ndarray,
    member_cy: np.ndarray,
    star_label: int | np.ndarray | list[int],
    *,
    s_max: int = DEFAULT_STAMP_PHYSICAL,
    s_min: int = DEFAULT_STAMP_MIN,
    region_x_min: int = 0,
    region_y_min: int = 0,
    enclose_pad_px: int = 0,
) -> np.ndarray:
    """Per-member odd sizes that jointly enclose in-reach segment pixels.

    ``star_label`` may be one label or several (merged islands). Segment pixels
    within Chebyshev ``(s_max-1)/2`` of any member are assigned to the nearest
    member; each member grows the smallest square covering its assigned pixels,
    then applies optional ``enclose_pad_px`` (default 0). Members with no
    assigned pixels get ``s_min`` (then pad). If no positive labels, every
    member gets ``s_max``.
    """
    k = len(member_cx)
    s_max = int(s_max)
    s_min = int(s_min)
    sizes = np.full(k, s_min, dtype=int)
    if k == 0:
        return sizes
    labels = np.atleast_1d(np.asarray(star_label, dtype=int)).ravel()
    labels = labels[labels > 0]
    if len(labels) == 0:
        sizes[:] = s_max
        return sizes

    label_map = np.asarray(label_map)
    half = s_max // 2
    ys, xs = np.where(np.isin(label_map, labels))
    if len(ys) == 0:
        sizes[:] = s_max
        return sizes

    cx_arr = np.asarray(member_cx, dtype=int) - int(region_x_min)
    cy_arr = np.asarray(member_cy, dtype=int) - int(region_y_min)
    # Filter to pixels reachable by some member at S_max.
    d_all = np.maximum(
        np.abs(xs[None, :] - cx_arr[:, None]),
        np.abs(ys[None, :] - cy_arr[:, None]),
    )  # (K, P)
    reach = d_all.min(axis=0) <= half
    if not np.any(reach):
        sizes[:] = s_max
        return sizes
    ys = ys[reach]
    xs = xs[reach]
    d_all = d_all[:, reach]
    owner = np.argmin(d_all, axis=0)

    for mi in range(k):
        sel = owner == mi
        sizes[mi] = grow_enclosing_square_size(
            ys[sel], xs[sel],
            int(member_cx[mi]), int(member_cy[mi]),
            s_max=s_max, s_min=s_min,
            region_x_min=int(region_x_min), region_y_min=int(region_y_min),
            enclose_pad_px=int(enclose_pad_px),
        )
    return sizes


def foreign_labels_in_mask(
    mask: np.ndarray,
    label_map: np.ndarray,
    home_labels: set[int] | list[int],
) -> np.ndarray:
    """Positive labels present in ``mask`` that are not in ``home_labels``."""
    home = {int(v) for v in home_labels if int(v) > 0}
    labs = np.asarray(label_map)[np.asarray(mask, dtype=bool)]
    if labs.size == 0:
        return np.zeros(0, dtype=np.int32)
    labs = labs[labs > 0]
    if home:
        labs = labs[~np.isin(labs, list(home))]
    return np.unique(labs).astype(np.int32)


def build_epsf_support_stamps(
    primary_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mag: np.ndarray,
    *,
    cal: np.ndarray | None = None,
    noise: np.ndarray | None = None,
    label_map: np.ndarray | None = None,
    stamp_physical: int = DEFAULT_STAMP_PHYSICAL,
    stamp_min: int = DEFAULT_STAMP_MIN,
    bright_mag_max: float = 13.0,
    max_group_size: int = 4,
    region_x_min: int = 0,
    region_y_min: int = 0,
    shape: tuple[int, int] | None = None,
    prune_snr: bool = False,
    n_sigma: float = 5.0,
    npixels: int = 5,
    erode_px: int = 1,
    enclose_pad_px: int = 0,
    resolve_overlaps: bool = True,
) -> list[SegmentAssignment]:
    """Build stamps: one per eroded segment that hosts ≥1 isolated primary.

    All Gaia ``tess_mag < bright_mag_max`` peaks on that segment become members
    (co-segment primaries are merged into the same stamp). Groups with
    ``K > max_group_size`` are rejected. After sizing, if a stamp square covers
    foreign eroded-label pixels, those islands are merged into one stamp
    (again rejecting if ``K`` overflows). The reported ``primary_index`` is the
    brightest isolated primary in the group. Primaries on background get a
    solo ``S_max`` fallback stamp each.

    When ``resolve_overlaps`` is True (default), contested scored pixels are
    kept only on the stamp whose brightest member is closest.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag = np.asarray(mag, dtype=float)
    primaries = np.asarray(primary_indices, dtype=int)
    s_max = int(stamp_physical)
    s_min = int(stamp_min)
    max_k = int(max_group_size)
    pad = int(enclose_pad_px)

    if label_map is None:
        if cal is None or noise is None:
            raise ValueError("cal and noise are required when label_map is not given")
        seg = detect_merged_segments(
            cal, noise,
            n_sigma=float(n_sigma), npixels=int(npixels),
            pad_px=0, erode_px=int(erode_px),
        )
        label_map = seg.label_map
    label_map = np.asarray(label_map)
    if shape is None:
        shape = label_map.shape
    ny, nx = int(shape[0]), int(shape[1])
    primary_set = {int(p) for p in primaries}

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(mag)
    bright = finite & (mag < float(bright_mag_max))
    bright_idx = np.flatnonzero(bright)

    # Peak label for every bright star (0 if OOB / background).
    star_label = np.zeros(len(x), dtype=np.int32)
    for i in bright_idx:
        pix = _pixel_at(x[i], y[i], label_map.shape, x0=region_x_min, y0=region_y_min)
        if pix is not None:
            star_label[i] = int(label_map[pix])

    # Group isolated primaries by eroded segment (lab=0 → one solo group each).
    groups: dict[int, list[int]] = {}
    solo_bg: list[int] = []
    for pi in primaries:
        pi = int(pi)
        lab = int(star_label[pi])
        if lab <= 0:
            solo_bg.append(pi)
        else:
            groups.setdefault(lab, []).append(pi)

    def _members_for_labels(lab_set: set[int]) -> list[int]:
        return sorted(
            (int(i) for i in bright_idx if int(star_label[i]) in lab_set),
            key=lambda i: float(mag[i]),
        )

    def _primaries_for_labels(lab_set: set[int]) -> list[int]:
        out_p: list[int] = []
        for lab in lab_set:
            out_p.extend(groups.get(int(lab), []))
        # Also any primary whose peak sits on these labels (already in groups).
        return sorted(set(out_p), key=lambda i: float(mag[i]))

    def _emit(lab_set: set[int], members: np.ndarray, pi: int) -> SegmentAssignment:
        cx = int(round(float(x[pi])))
        cy = int(round(float(y[pi])))
        member_cx = np.asarray([int(round(float(x[mi]))) for mi in members], dtype=int)
        member_cy = np.asarray([int(round(float(y[mi]))) for mi in members], dtype=int)
        labels_arr = np.asarray(sorted(lab_set), dtype=int)
        sizes = member_stamp_sizes_for_segment(
            label_map, member_cx, member_cy, labels_arr,
            s_max=s_max, s_min=s_min,
            region_x_min=int(region_x_min), region_y_min=int(region_y_min),
            enclose_pad_px=pad,
        )
        mask = np.zeros((ny, nx), dtype=bool)
        for mi, sk in zip(members, sizes):
            paint_stamp_window(
                mask, int(round(float(x[mi]))), int(round(float(y[mi]))),
                stamp_physical=int(sk),
                region_x_min=int(region_x_min), region_y_min=int(region_y_min),
            )
        if prune_snr:
            if cal is None or noise is None:
                raise ValueError("cal and noise required when prune_snr=True")
            mask = prune_mask_by_snr(mask, cal, noise, n_sigma=n_sigma)
            if not mask.any():
                paint_stamp_window(
                    mask, cx, cy, stamp_physical=int(sizes[0]),
                    region_x_min=int(region_x_min), region_y_min=int(region_y_min),
                )
        if int(mask.sum()) == 0:
            ys = xs = np.array([], dtype=int)
            pix_x = np.array([], dtype=np.float64)
            pix_y = np.array([], dtype=np.float64)
            bbox = None
            n_pix = 0
        else:
            ys, xs = np.where(mask)
            pix_x = xs.astype(np.float64) + float(region_x_min)
            pix_y = ys.astype(np.float64) + float(region_y_min)
            bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            n_pix = int(len(xs))
        # Report primary's own label when possible.
        home_lab = int(star_label[pi]) if int(star_label[pi]) > 0 else (
            int(min(lab_set)) if lab_set else 0
        )
        return SegmentAssignment(
            primary_index=pi,
            segment_label=home_lab,
            mask=mask,
            member_indices=members,
            n_pixels=n_pix,
            stamp_center_x=cx,
            stamp_center_y=cy,
            pix_x=pix_x,
            pix_y=pix_y,
            member_stamp_sizes=np.asarray(sizes, dtype=int),
            bbox=bbox,
        )

    # Clusters start as one label each (only labels that host a primary).
    clusters: list[set[int]] = [{int(lab)} for lab in sorted(groups.keys())]

    def _try_build(lab_set: set[int]) -> SegmentAssignment | None:
        on_seg = _members_for_labels(lab_set)
        if len(on_seg) == 0 or len(on_seg) > max_k:
            return None
        pri_list = _primaries_for_labels(lab_set)
        if not pri_list:
            # Foreign-only island pulled in without a primary — still need a
            # designated index; use brightest member (should not happen for
            # top-level clusters).
            pi = int(on_seg[0])
        else:
            pi = int(pri_list[0])
        rest = [i for i in on_seg if i != pi]
        members = np.asarray([pi] + rest, dtype=int)
        return _emit(set(lab_set), members, pi)

    # Iterate: build → if square hits foreign labels with Gaia, merge → rebuild.
    max_iters = 16
    for _ in range(max_iters):
        rebuilt: list[set[int]] = []
        changed = False
        for lab_set in clusters:
            a = _try_build(lab_set)
            if a is None:
                # Overcrowded or empty — drop (reject).
                changed = True
                continue
            foreign = foreign_labels_in_mask(a.mask, label_map, lab_set)
            if len(foreign) == 0:
                rebuilt.append(set(lab_set))
                continue
            # Only merge foreign labels that host bright Gaia (membership relevant).
            add: set[int] = set()
            for flab in foreign.tolist():
                flab = int(flab)
                if any(int(star_label[i]) == flab for i in bright_idx):
                    add.add(flab)
            if not add:
                rebuilt.append(set(lab_set))
                continue
            new_set = set(lab_set) | add
            # Absorb any other cluster that shares labels with new_set.
            for other in clusters:
                if other is lab_set:
                    continue
                if other & new_set:
                    new_set |= other
            rebuilt.append(new_set)
            changed = True
        # Deduplicate clusters (union overlapping sets).
        uniq: list[set[int]] = []
        for s in rebuilt:
            merged_into = False
            for u in uniq:
                if u & s:
                    u |= s
                    merged_into = True
                    break
            if not merged_into:
                uniq.append(set(s))
        # Fix transitive overlaps among uniq.
        stable = False
        while not stable:
            stable = True
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    if uniq[i] & uniq[j]:
                        uniq[i] |= uniq[j]
                        uniq.pop(j)
                        stable = False
                        changed = True
                        break
                if not stable:
                    break
        clusters = uniq
        if not changed:
            break

    out: list[SegmentAssignment] = []
    for lab_set in clusters:
        a = _try_build(lab_set)
        if a is not None:
            # Final foreign check: if still hitting Gaia-bearing foreign labels
            # after max iters, reject rather than emit a contaminated stamp.
            foreign = foreign_labels_in_mask(a.mask, label_map, lab_set)
            add = [
                int(fl) for fl in foreign
                if any(int(star_label[i]) == int(fl) for i in bright_idx)
            ]
            if add:
                continue
            out.append(a)

    for pi in solo_bg:
        if int(pi) in primary_set and any(
            int(pi) in set(map(int, a.member_indices)) for a in out
        ):
            continue
        out.append(_emit(set(), np.asarray([pi], dtype=int), int(pi)))

    if resolve_overlaps:
        out, _ov = resolve_overlapping_stamp_pixels(
            out, x, y, mag,
            region_x_min=int(region_x_min), region_y_min=int(region_y_min),
        )
    return out


def _assignment_brightest_index(
    a: SegmentAssignment,
    mag: np.ndarray | None,
) -> int:
    """Index of the brightest member (smallest mag), else ``primary_index``."""
    if mag is None or a.member_indices is None or len(a.member_indices) == 0:
        return int(a.primary_index)
    mi = np.asarray(a.member_indices, dtype=int)
    m = np.asarray(mag, dtype=float)
    finite = np.isfinite(m[mi])
    if not np.any(finite):
        return int(a.primary_index)
    return int(mi[finite][np.argmin(m[mi[finite]])])


def _refresh_assignment_from_mask(
    a: SegmentAssignment,
    mask: np.ndarray,
    *,
    region_x_min: float = 0.0,
    region_y_min: float = 0.0,
) -> SegmentAssignment | None:
    """Rebuild pix/bbox/n_pixels from ``mask``; return None if empty."""
    mask_b = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask_b)
    if len(xs) == 0:
        return None
    bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    pix_x = xs.astype(np.float64) + float(region_x_min)
    pix_y = ys.astype(np.float64) + float(region_y_min)
    return SegmentAssignment(
        primary_index=int(a.primary_index),
        segment_label=int(a.segment_label),
        mask=mask_b,
        member_indices=np.asarray(a.member_indices, dtype=int),
        n_pixels=int(len(xs)),
        stamp_center_x=int(a.stamp_center_x),
        stamp_center_y=int(a.stamp_center_y),
        pix_x=pix_x,
        pix_y=pix_y,
        member_stamp_sizes=(
            None if a.member_stamp_sizes is None
            else np.asarray(a.member_stamp_sizes, dtype=int)
        ),
        bbox=bbox,
    )


def resolve_overlapping_stamp_pixels(
    assignments: list[SegmentAssignment],
    x: np.ndarray,
    y: np.ndarray,
    mag: np.ndarray | None = None,
    *,
    region_x_min: float = 0.0,
    region_y_min: float = 0.0,
) -> tuple[list[SegmentAssignment], dict]:
    """Make stamp supports pixel-exclusive by nearest brightest-member ownership.

    For every pixel claimed by ≥2 stamps, keep it only in the stamp whose
    brightest Gaia member (smallest ``mag``; falls back to ``primary_index``)
    is closest in Euclidean detector coordinates. Update masks / pix lists /
    bboxes; drop stamps that become empty.

    Returns ``(assignments_out, stats)`` with ``n_pairs``, ``n_contested_px``,
    ``n_pixels_removed``, ``n_dropped``, plus arrays for diagnostics:
    ``contested_iy/ix``, ``winner_pre_idx`` (index into the *input* list),
    ``kept_pre_indices`` (input indices that survived), and ``involved_pre_idx``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag_arr = None if mag is None else np.asarray(mag, dtype=float)
    stats: dict = {
        "n_pairs": 0,
        "n_contested_px": 0,
        "n_pixels_removed": 0,
        "n_dropped": 0,
        "contested_iy": np.zeros(0, dtype=np.int32),
        "contested_ix": np.zeros(0, dtype=np.int32),
        "winner_pre_idx": np.zeros(0, dtype=np.int32),
        "kept_pre_indices": np.arange(len(assignments), dtype=np.int32),
        "involved_pre_idx": np.zeros(0, dtype=np.int32),
    }
    if len(assignments) < 2:
        return list(assignments), stats

    pairs = find_overlapping_stamp_pairs(assignments)
    stats["n_pairs"] = int(len(pairs))
    if not pairs:
        return list(assignments), stats

    anchors: list[tuple[float, float]] = []
    for a in assignments:
        bi = _assignment_brightest_index(a, mag_arr)
        anchors.append((float(x[bi]), float(y[bi])))

    # Contested pixels: (iy, ix) -> set of claiming stamp indices.
    contested: dict[tuple[int, int], set[int]] = {}
    for i, j, _nov in pairs:
        ai = assignments[i]
        aj = assignments[j]
        bi = ai.bbox
        bj = aj.bbox
        if bi is None or bj is None:
            continue
        y0 = max(bi[0], bj[0])
        y1 = min(bi[1], bj[1])
        x0 = max(bi[2], bj[2])
        x1 = min(bi[3], bj[3])
        if y1 <= y0 or x1 <= x0:
            continue
        mi = np.asarray(ai.mask, dtype=bool)
        mj = np.asarray(aj.mask, dtype=bool)
        both = mi[y0:y1, x0:x1] & mj[y0:y1, x0:x1]
        if not both.any():
            continue
        ys, xs = np.where(both)
        for dy, dx in zip(ys.tolist(), xs.tolist()):
            key = (y0 + int(dy), x0 + int(dx))
            contested.setdefault(key, set()).update((int(i), int(j)))

    stats["n_contested_px"] = int(len(contested))
    involved = sorted({c for claimants in contested.values() for c in claimants})
    if not involved and pairs:
        involved = sorted({i for i, j, _ in pairs} | {j for i, j, _ in pairs})
    stats["involved_pre_idx"] = np.asarray(involved, dtype=np.int32)
    if not contested:
        return list(assignments), stats

    # Winner per contested pixel.
    winners: dict[tuple[int, int], int] = {}
    rx = float(region_x_min)
    ry = float(region_y_min)
    for (iy, ix), claimants in contested.items():
        px = float(ix) + rx
        py = float(iy) + ry
        best_i: int | None = None
        best_d = np.inf
        for ci in sorted(claimants):
            ax, ay = anchors[ci]
            d = (ax - px) * (ax - px) + (ay - py) * (ay - py)
            if d < best_d:
                best_d = d
                best_i = int(ci)
        assert best_i is not None
        winners[(iy, ix)] = best_i

    c_iy = np.asarray([k[0] for k in contested.keys()], dtype=np.int32)
    c_ix = np.asarray([k[1] for k in contested.keys()], dtype=np.int32)
    c_win = np.asarray([winners[k] for k in contested.keys()], dtype=np.int32)
    stats["contested_iy"] = c_iy
    stats["contested_ix"] = c_ix
    stats["winner_pre_idx"] = c_win

    # Apply removals.
    masks = [np.asarray(a.mask, dtype=bool).copy() for a in assignments]
    n_removed = 0
    for (iy, ix), claimants in contested.items():
        win = winners[(iy, ix)]
        for ci in claimants:
            if ci == win:
                continue
            if masks[ci][iy, ix]:
                masks[ci][iy, ix] = False
                n_removed += 1
    stats["n_pixels_removed"] = int(n_removed)

    out: list[SegmentAssignment] = []
    kept: list[int] = []
    n_dropped = 0
    for i, (a, mask) in enumerate(zip(assignments, masks)):
        refreshed = _refresh_assignment_from_mask(
            a, mask, region_x_min=rx, region_y_min=ry,
        )
        if refreshed is None:
            n_dropped += 1
            continue
        kept.append(int(i))
        out.append(refreshed)
    stats["n_dropped"] = int(n_dropped)
    stats["kept_pre_indices"] = np.asarray(kept, dtype=np.int32)
    return out, stats


def overlap_connected_components(
    pairs: list[tuple[int, int, int]],
) -> list[list[int]]:
    """Connected components of stamp indices linked by overlap pairs."""
    if not pairs:
        return []
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, _ in pairs:
        union(int(i), int(j))
    groups: dict[int, list[int]] = {}
    for node in list(parent.keys()):
        groups.setdefault(find(node), []).append(node)
    return [sorted(v) for v in groups.values()]


def find_overlapping_stamp_pairs(
    assignments: list[SegmentAssignment],
) -> list[tuple[int, int, int]]:
    """Return ``(i, j, n_overlap_px)`` for pairs of stamps whose masks intersect.

    Indices ``i < j`` refer to positions in ``assignments``. Uses cached stamp
    bboxes + KDTree neighbor queries, then ANDs only the overlapping crop —
    never a full-image pairwise mask loop.
    """
    n = len(assignments)
    if n < 2:
        return []

    def _bbox(a: SegmentAssignment) -> tuple[int, int, int, int] | None:
        if a.bbox is not None:
            return a.bbox
        if a.n_pixels <= 0:
            return None
        ys, xs = np.where(np.asarray(a.mask, dtype=bool))
        if len(xs) == 0:
            return None
        return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)

    centers = np.empty((n, 2), dtype=float)
    radii = np.zeros(n, dtype=float)
    boxes: list[tuple[int, int, int, int] | None] = [None] * n
    for i, a in enumerate(assignments):
        bi = _bbox(a)
        boxes[i] = bi
        if bi is None:
            centers[i] = (np.nan, np.nan)
            continue
        y0, y1, x0, x1 = bi
        centers[i] = (0.5 * (y0 + y1 - 1), 0.5 * (x0 + x1 - 1))
        radii[i] = 0.5 * float(np.hypot(y1 - y0, x1 - x0)) + 0.5

    valid = np.isfinite(centers[:, 0])
    if int(valid.sum()) < 2:
        return []
    idx = np.flatnonzero(valid)
    tree = cKDTree(centers[idx])
    r_max = float(radii[idx].max())
    pairs: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a_local, neigh in enumerate(tree.query_ball_tree(tree, r=2.0 * r_max)):
        i = int(idx[a_local])
        bi = boxes[i]
        if bi is None:
            continue
        y0i, y1i, x0i, x1i = bi
        ri = float(radii[i])
        mi = np.asarray(assignments[i].mask, dtype=bool)
        for b_local in neigh:
            if b_local <= a_local:
                continue
            j = int(idx[b_local])
            if float(np.hypot(*(centers[i] - centers[j]))) > ri + float(radii[j]):
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            bj = boxes[j]
            if bj is None:
                continue
            y0j, y1j, x0j, x1j = bj
            if y1i <= y0j or y1j <= y0i or x1i <= x0j or x1j <= x0i:
                continue
            y0, y1 = max(y0i, y0j), min(y1i, y1j)
            x0, x1 = max(x0i, x0j), min(x1i, x1j)
            mj = np.asarray(assignments[j].mask, dtype=bool)
            n_ov = int(np.count_nonzero(mi[y0:y1, x0:x1] & mj[y0:y1, x0:x1]))
            if n_ov > 0:
                pairs.append((key[0], key[1], n_ov))
    return pairs


def assign_segment_members(
    label_map: np.ndarray,
    primary_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mag: np.ndarray,
    *,
    bright_mag_max: float = 13.0,
    region_x_min: float = 0.0,
    region_y_min: float = 0.0,
) -> list[SegmentAssignment]:
    """Map each primary to its merged segment and Gaia members inside the mask.

    Legacy diagnostic path. ``x,y`` are crop-local detector coordinates.
    """
    label_map = np.asarray(label_map)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag = np.asarray(mag, dtype=float)
    bright = np.isfinite(x) & np.isfinite(y) & np.isfinite(mag) & (mag < float(bright_mag_max))
    bright_idx = np.flatnonzero(bright)

    star_label = np.zeros(len(x), dtype=np.int32)
    for i in bright_idx:
        pix = _pixel_at(x[i], y[i], label_map.shape, x0=region_x_min, y0=region_y_min)
        if pix is None:
            continue
        star_label[i] = int(label_map[pix])

    out: list[SegmentAssignment] = []
    for pi in np.asarray(primary_indices, dtype=int):
        lab = int(star_label[pi])
        if lab <= 0:
            continue
        mask = label_map == lab
        members = bright_idx[star_label[bright_idx] == lab]
        if int(pi) not in set(int(m) for m in members):
            members = np.concatenate([np.asarray([pi], dtype=int), members])
        ys, xs = np.where(mask)
        bbox = (
            (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            if len(xs) else None
        )
        out.append(
            SegmentAssignment(
                primary_index=int(pi),
                segment_label=lab,
                mask=mask,
                member_indices=np.asarray(members, dtype=int),
                n_pixels=int(len(xs)),
                bbox=bbox,
            )
        )
    return out


def mask_to_ds9_polygons(
    mask: np.ndarray,
    *,
    region_x_min: float = 0.0,
    region_y_min: float = 0.0,
    outer_only: bool = True,
    max_vertices: int = 4000,
    bbox: tuple[int, int, int, int] | None = None,
) -> list[str]:
    """Convert a boolean mask to DS9 ``polygon(...)`` strings (image 1-based).

    Pass ``bbox=(y0,y1,x0,x1)`` (half-open, crop-local) to contour only the
    stamp support — required for full-FFI masks that are mostly False.
    """
    from skimage.measure import find_contours

    mask_b = np.asarray(mask, dtype=bool)
    if bbox is not None:
        y0, y1, x0, x1 = (int(v) for v in bbox)
        y0 = max(0, y0); x0 = max(0, x0)
        y1 = min(mask_b.shape[0], y1); x1 = min(mask_b.shape[1], x1)
        if y1 <= y0 or x1 <= x0 or not mask_b[y0:y1, x0:x1].any():
            return []
        sub = mask_b[y0:y1, x0:x1]
        contours = find_contours(sub.astype(np.float64), 0.5)
        # Shift contour coords back into full-mask (crop) pixel space.
        y_off, x_off = float(y0), float(x0)
    else:
        if not mask_b.any():
            return []
        contours = find_contours(mask_b.astype(np.float64), 0.5)
        y_off = x_off = 0.0
    if not contours:
        return []
    if outer_only:
        contours = [max(contours, key=len)]

    polys: list[str] = []
    for cont in contours:
        if len(cont) > max_vertices:
            step = int(np.ceil(len(cont) / max_vertices))
            cont = cont[::step]
        xs = cont[:, 1] + x_off - float(region_x_min) + 1.0
        ys = cont[:, 0] + y_off - float(region_y_min) + 1.0
        if len(xs) < 3:
            continue
        pts = " ".join(f"{xf:.3f},{yf:.3f}" for xf, yf in zip(xs, ys))
        polys.append(f"polygon({pts})")
    return polys


def write_fit_regions_reg(
    path: Path | str,
    assignments: list[SegmentAssignment],
    *,
    x: np.ndarray,
    y: np.ndarray,
    mag: np.ndarray,
    region_x_min: float,
    region_y_min: float,
    radius_px: float = 2.5,
    polygon_color: str = "yellow",
    primary_color: str = "red",
    member_color: str = "cyan",
) -> Path:
    """DS9 region file: support polygons + labeled circles for fit stars."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mag = np.asarray(mag, dtype=float)

    lines = [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "image",
    ]
    seen_seg: set[int] = set()
    for a in assignments:
        if int(a.segment_label) not in seen_seg:
            seen_seg.add(int(a.segment_label))
            for poly in mask_to_ds9_polygons(a.mask, bbox=a.bbox):
                lines.append(f"{poly} # color={polygon_color} width=2")

    role: dict[int, str] = {}
    for a in assignments:
        for mi in a.member_indices:
            mi = int(mi)
            if mi == int(a.primary_index):
                role[mi] = "primary"
            else:
                role.setdefault(mi, "member")
    for mi, r in role.items():
        xi = float(x[mi]) - float(region_x_min) + 1.0
        yi = float(y[mi]) - float(region_y_min) + 1.0
        color = primary_color if r == "primary" else member_color
        lab = f"{float(mag[mi]):.2f}"
        lines.append(
            f"circle({xi:.3f},{yi:.3f},{radius_px}) # color={color} width=2 text={{{lab}}}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def hp_d_crop_origin(hp_d_path: Path | str) -> tuple[int, int]:
    """Return ``(XMIN, YMIN)`` from an ``hp_d`` CAL header (full-FFI → crop)."""
    with fits.open(Path(hp_d_path), memmap=True) as hdul:
        hdr = hdul[1].header
        return int(hdr.get("XMIN", 0)), int(hdr.get("YMIN", 0))


def load_tess_wcs_from_cache(
    wcs_cache_path: Path | str,
    stem: str,
):
    """Build an astropy SIP ``WCS`` for one FFI from SCC ``wcs_cache.csv/.parquet``.

    ``stem`` is the hp_d / frame stem (e.g. ``tess2019362015923-s0020-3-3``);
    cache filenames look like ``{stem}-0165-s_ffic.fits.gz``. Pixel coords from
    this WCS are **full-FFI** (buffer columns included).
    """
    from astropy.wcs import WCS

    path = Path(wcs_cache_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "filename" not in df.columns:
        raise ValueError(f"{path} missing 'filename' column")
    stem = str(stem)
    hit = df[df["filename"].astype(str).str.startswith(stem)]
    if hit.empty:
        hit = df[df["filename"].astype(str).str.contains(stem, regex=False)]
    if hit.empty:
        raise KeyError(f"no wcs_cache row for stem={stem!r} in {path}")
    row = hit.iloc[0]
    hdr = fits.Header()
    for key, val in row.items():
        if key in ("filename", "DATE-OBS"):
            continue
        if pd.isna(val):
            continue
        try:
            hdr[str(key)] = val.item() if hasattr(val, "item") else val
        except Exception:
            hdr[str(key)] = val
    return WCS(hdr, naxis=2)


def ra_dec_to_hp_d_xy(
    wcs,
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    crop_x_min: int = 0,
    crop_y_min: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """RA/Dec → 0-based **hp_d crop-local** pixels via full-FFI SIP WCS.

    ``wcs`` must be the uncropped TESS FFI WCS (e.g. from ``wcs_cache.csv``).
    Subtract ``(crop_x_min, crop_y_min)`` (= hp_d ``XMIN/YMIN``, typically
    ``(44, 0)``) so coordinates land on the science/difference array.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    x_full, y_full = wcs.world_to_pixel_values(ra, dec)
    x = np.asarray(x_full, dtype=np.float64) - float(crop_x_min)
    y = np.asarray(y_full, dtype=np.float64) - float(crop_y_min)
    return x, y


def save_label_map(
    path: Path | str,
    label_map: np.ndarray,
    *,
    n_sigma: float,
    erode_px: int,
    npixels: int = 5,
    pad_px: int = 0,
    stem: str = "",
    frame_index: int | None = None,
) -> Path:
    """Persist eroded segmentation for reuse across notebook re-runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        label_map=np.asarray(label_map, dtype=np.int32),
        n_sigma=float(n_sigma),
        erode_px=int(erode_px),
        npixels=int(npixels),
        pad_px=int(pad_px),
        stem=str(stem),
        frame_index=-1 if frame_index is None else int(frame_index),
    )
    return path


def load_label_map(path: Path | str) -> tuple[np.ndarray, dict]:
    """Load ``save_label_map`` output → ``(label_map, meta_dict)``."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        label_map = np.asarray(z["label_map"], dtype=np.int32)
        meta = {
            "n_sigma": float(z["n_sigma"]),
            "erode_px": int(z["erode_px"]),
            "npixels": int(z["npixels"]),
            "pad_px": int(z["pad_px"]),
            "stem": str(z["stem"]),
            "frame_index": int(z["frame_index"]),
        }
    return label_map, meta


def write_stamp_mask_npz(
    path: Path | str,
    assignment: SegmentAssignment,
) -> Path:
    """Write one stamp mask as a bbox-cropped bool array (not a full-FFI canvas)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = assignment
    if a.bbox is not None:
        y0, y1, x0, x1 = a.bbox
        sub = np.asarray(a.mask[y0:y1, x0:x1], dtype=bool)
        bbox = np.asarray(a.bbox, dtype=np.int32)
    else:
        ys, xs = np.where(np.asarray(a.mask, dtype=bool))
        if len(xs) == 0:
            sub = np.zeros((0, 0), dtype=bool)
            bbox = np.asarray([0, 0, 0, 0], dtype=np.int32)
        else:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            sub = np.asarray(a.mask[y0:y1, x0:x1], dtype=bool)
            bbox = np.asarray([y0, y1, x0, x1], dtype=np.int32)
    np.savez_compressed(
        path,
        mask=sub,
        bbox=bbox,
        y0=int(bbox[0]),
        x0=int(bbox[2]),
        primary_index=int(a.primary_index),
        segment_label=int(a.segment_label),
        member_indices=np.asarray(a.member_indices, dtype=np.int32),
        member_stamp_sizes=(
            np.asarray(a.member_stamp_sizes, dtype=np.int32)
            if a.member_stamp_sizes is not None
            else np.zeros(0, dtype=np.int32)
        ),
        pix_x=np.asarray(a.pix_x if a.pix_x is not None else [], dtype=np.float64),
        pix_y=np.asarray(a.pix_y if a.pix_y is not None else [], dtype=np.float64),
        stamp_center_x=int(a.stamp_center_x),
        stamp_center_y=int(a.stamp_center_y),
        n_pixels=int(a.n_pixels),
    )
    return path


def write_hp_d_ref_fits(
    path: Path | str,
    cal: np.ndarray,
    noise: np.ndarray,
    *,
    stem: str,
    btjd: float,
    region_x_min: int,
    region_y_min: int,
    region_x_max: int,
    region_y_max: int,
    sector: int | None = None,
    camera: int | None = None,
    ccd: int | None = None,
    frame_index: int | None = None,
) -> Path:
    """Write region-crop hp_d (ext1) + NOISE (ext2) for DS9 with the region file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["STEM"] = str(stem)
    hdr["BTJD"] = float(btjd)
    hdr["XMIN"] = int(region_x_min)
    hdr["YMIN"] = int(region_y_min)
    hdr["XMAX"] = int(region_x_max)
    hdr["YMAX"] = int(region_y_max)
    if sector is not None:
        hdr["SECTOR"] = int(sector)
    if camera is not None:
        hdr["CAMERA"] = int(camera)
    if ccd is not None:
        hdr["CCD"] = int(ccd)
    if frame_index is not None:
        hdr["FRAMEIDX"] = int(frame_index)
    hdr["BUNIT"] = "e-/s"
    hdr["EXTNAME"] = "DIFF"
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=np.asarray(cal, dtype=np.float32), header=hdr, name="DIFF")
    hdr2 = fits.Header()
    hdr2["EXTNAME"] = "NOISE"
    hdr2["BUNIT"] = "e-/s"
    hdu2 = fits.ImageHDU(data=np.asarray(noise, dtype=np.float32), header=hdr2, name="NOISE")
    fits.HDUList([hdu0, hdu1, hdu2]).writeto(path, overwrite=True)
    return path


def assignments_to_summary(
    assignments: list[SegmentAssignment],
    *,
    source_id: np.ndarray | None = None,
    mag: np.ndarray,
) -> pd.DataFrame:
    """One row per primary assignment for CSV export."""
    mag = np.asarray(mag, dtype=float)
    rows = []
    for a in assignments:
        mem_mags = mag[a.member_indices]
        sid = None if source_id is None else int(source_id[a.primary_index])
        sizes = a.member_stamp_sizes
        if sizes is None:
            sizes_str = ""
            primary_s = ""
        else:
            sizes = np.asarray(sizes, dtype=int)
            sizes_str = ",".join(str(int(v)) for v in sizes)
            primary_s = int(sizes[0]) if len(sizes) else ""
        rows.append(
            {
                "primary_index": a.primary_index,
                "source_id": sid,
                "segment_label": a.segment_label,
                "primary_tess_mag": float(mag[a.primary_index]),
                "primary_stamp_size": primary_s,
                "member_stamp_sizes": sizes_str,
                "n_pixels": a.n_pixels,
                "n_members": int(len(a.member_indices)),
                "member_indices": ",".join(str(int(i)) for i in a.member_indices),
                "member_tess_mags": ",".join(f"{float(m):.3f}" for m in mem_mags),
            }
        )
    return pd.DataFrame(rows)
