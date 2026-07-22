"""Hybrid L4a/L4b regmap helpers (roll + Exact patch on seam/rim masks).

These operate on PS1-shaped TESS-pixel-id assignment maps (``tid`` arrays where
``tid < 0`` means unassigned). They do not call PanCAKES; callers supply Exact
patches for the flagged pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

__all__ = [
    "tess_ownership_boundary",
    "footprint_edge",
    "needs_recompute_mask",
    "apply_hybrid_patch",
    "apply_sparse_patch_inplace",
    "abutting_rim_ps1_mask",
    "roll_assignment",
    "roll_mask",
    "invalid_border_margin",
    "seam_roll_is_exact_for_shift",
    "stencil_roll_is_exact",
    "build_l4a_hybrid_assignment",
]


def tess_ownership_boundary(tid: np.ndarray) -> np.ndarray:
    """True where a PS1 pixel's TESS id differs from its +x or +y neighbour."""
    tid = np.asarray(tid)
    valid = tid >= 0
    bx = np.zeros(tid.shape, dtype=bool)
    by = np.zeros(tid.shape, dtype=bool)
    bx[:, :-1] = valid[:, :-1] & valid[:, 1:] & (tid[:, :-1] != tid[:, 1:])
    by[:-1, :] = valid[:-1, :] & valid[1:, :] & (tid[:-1, :] != tid[1:, :])
    return bx | by


def footprint_edge(tid: np.ndarray) -> np.ndarray:
    """True at valid <-> invalid transitions of the assignment footprint."""
    tid = np.asarray(tid)
    valid = tid >= 0
    ex = np.zeros(tid.shape, dtype=bool)
    ey = np.zeros(tid.shape, dtype=bool)
    ex[:, :-1] = valid[:, :-1] != valid[:, 1:]
    ey[:-1, :] = valid[:-1, :] != valid[1:, :]
    return ex | ey


def needs_recompute_mask(linear_tid: np.ndarray, R: int = 1) -> np.ndarray:
    """GT-free Type I mask: dilate(ownership_boundary ∪ footprint_edge, R)."""
    seed = tess_ownership_boundary(linear_tid) | footprint_edge(linear_tid)
    if int(R) <= 0:
        return seed
    struct = np.ones((2 * int(R) + 1, 2 * int(R) + 1), dtype=bool)
    return binary_dilation(seed, structure=struct)


def apply_hybrid_patch(
    linear_tid: np.ndarray,
    exact_tid: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Replace ``linear_tid`` with ``exact_tid`` on ``mask`` (Exact must be defined)."""
    linear_tid = np.asarray(linear_tid)
    exact_tid = np.asarray(exact_tid)
    mask = np.asarray(mask, dtype=bool)
    if linear_tid.shape != exact_tid.shape:
        raise ValueError(
            f"shape mismatch linear={linear_tid.shape} exact={exact_tid.shape}"
        )
    out = linear_tid.copy()
    replace = mask & (exact_tid >= 0)
    out[replace] = exact_tid[replace]
    return out


def abutting_rim_ps1_mask(
    tid: np.ndarray,
    border_tess_ids: np.ndarray | set[int] | list[int],
) -> np.ndarray:
    """Type II: PS1 pixels whose TESS id lies on the shared abutting border set."""
    tid = np.asarray(tid)
    if not np.issubdtype(tid.dtype, np.integer):
        # Regmap FITS files store TESS_PIXEL_MAP as whole-number floats (e.g.
        # float32); fancy-indexing with lut[tid] below requires an integer
        # array regardless of the on-disk dtype.
        tid = tid.astype(np.int64)
    if not isinstance(border_tess_ids, np.ndarray):
        border_tess_ids = np.fromiter(border_tess_ids, dtype=np.int64)
    else:
        border_tess_ids = np.asarray(border_tess_ids, dtype=np.int64)
    if border_tess_ids.size == 0:
        return np.zeros(tid.shape, dtype=bool)
    # Clip negative sentinel ids out of the lookup table domain.
    border_tess_ids = border_tess_ids[border_tess_ids >= 0]
    if border_tess_ids.size == 0:
        return np.zeros(tid.shape, dtype=bool)
    max_id = int(max(int(tid.max(initial=-1)), int(border_tess_ids.max())))
    lut = np.zeros(max_id + 1, dtype=bool)
    lut[border_tess_ids] = True
    valid = tid >= 0
    return valid & lut[np.clip(tid, 0, max_id)]


def roll_assignment(
    frozen_tid: np.ndarray,
    sx_int: int,
    sy_int: int,
    *,
    convention: str = "assignment",
) -> np.ndarray:
    """
    Integer-roll a frozen assignment map.

    ``convention='assignment'`` uses ``np.roll(..., (-sx, -sy))`` matching the
    notebook assignment-roll (opposite of production PS1 *data* roll).
    ``convention='data'`` uses ``(+sx, +sy)`` like rolling PS1 flux before binning.
    """
    frozen_tid = np.asarray(frozen_tid)
    sx_i, sy_i = int(sx_int), int(sy_int)
    if convention == "assignment":
        return np.roll(np.roll(frozen_tid, -sx_i, axis=1), -sy_i, axis=0)
    if convention == "data":
        return np.roll(np.roll(frozen_tid, sx_i, axis=1), sy_i, axis=0)
    raise ValueError(f"unknown convention {convention!r}")


def roll_mask(mask: np.ndarray, sx_int: int, sy_int: int) -> np.ndarray:
    """Roll a PS1-shaped mask with the ``convention='assignment'`` sign.

    Mirrors :func:`roll_assignment` so a mask precomputed on the *unrolled*
    frozen map lands on the same pixels as the rolled assignment.
    """
    return np.roll(
        np.roll(np.asarray(mask), -int(sx_int), axis=1), -int(sy_int), axis=0
    )


def invalid_border_margin(frozen_tid: np.ndarray, *, R: int = 1) -> int:
    """Minimum distance from any assigned pixel to the nearest array edge."""
    tid = np.asarray(frozen_tid)
    valid = tid >= 0
    if not valid.any():
        return int(max(tid.shape))
    rows, cols = np.nonzero(valid)
    h, width = tid.shape
    return int(
        min(
            int(rows.min()),
            int(cols.min()),
            h - 1 - int(rows.max()),
            width - 1 - int(cols.max()),
        )
    )


def seam_roll_is_exact_for_shift(
    frozen_tid: np.ndarray,
    sx_int: int,
    sy_int: int,
    *,
    R: int = 1,
) -> bool:
    """True when intra-skycell seam masks for this shift may be precomputed and rolled."""
    need = max(abs(int(sx_int)), abs(int(sy_int))) + int(R) + 1
    return need <= invalid_border_margin(frozen_tid, R=int(R))


def stencil_roll_is_exact(
    frozen_tid: np.ndarray,
    max_abs_shift: int,
    *,
    R: int = 1,
) -> bool:
    """Backward-compatible skycell-wide seam-roll guard (uses ``max_abs_shift``)."""
    need = int(max_abs_shift) + int(R) + 1
    return need <= invalid_border_margin(frozen_tid, R=int(R))


def apply_sparse_patch_inplace(
    out_flat: np.ndarray,
    mask_flat: np.ndarray,
    idx: np.ndarray,
    val: np.ndarray,
) -> int:
    """Scatter ``val`` into ``out_flat`` at ``idx`` where ``mask_flat`` is set.

    Sparse equivalent of ``apply_hybrid_patch(out, exact, mask)``: ``idx``/``val``
    already carry only ``exact >= 0`` pixels, which is exactly the
    ``mask & (exact >= 0)`` selection the dense form applies. Mutates in place to
    avoid a full-array copy per patch.
    """
    idx = np.asarray(idx)
    if idx.size == 0:
        return 0
    sel = mask_flat[idx]
    if not sel.any():
        return 0
    hit = idx[sel]
    out_flat[hit] = np.asarray(val)[sel]
    return int(hit.size)


def build_l4a_hybrid_assignment(
    frozen_tid: np.ndarray,
    sx_int: int,
    sy_int: int,
    exact_tid: np.ndarray | None = None,
    *,
    hybrid_R: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """L4a: roll frozen map, optionally Exact-patch the R=1 seam/rim mask.

    Returns ``(hybrid_tid, needs_recompute_mask)``. When ``exact_tid`` is None,
    returns the linear rolled assignment and the mask that *would* be Exact'd
    (callers can fill Exact later).
    """
    linear = roll_assignment(frozen_tid, sx_int, sy_int, convention="assignment")
    mask = needs_recompute_mask(linear, R=int(hybrid_R))
    if exact_tid is None:
        return linear, mask
    return apply_hybrid_patch(linear, exact_tid, mask), mask
