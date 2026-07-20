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
    "abutting_rim_ps1_mask",
    "roll_assignment",
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
