"""Abutting skycell pairs and Type II (L4b) pair-state enumeration.

Ports notebook logic from ``dev/distortion_aware_template/`` for production L4b:
undirected master-map neighbours and unique ``(sx_A, sy_A, sx_B, sy_B)`` keys
per border (cardinality ``n_type2_pair_states_sum``), not frame-to-frame
transitions (``count_l4b_events``).
"""

from __future__ import annotations

import logging
import re
from typing import Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "abutting_undirected_pairs",
    "build_col_of_name",
    "build_name_to_id",
    "count_unique_pair_states_sum",
    "l4b_rim_cache_basename",
    "pair_column_indices",
    "parse_l4b_rim_cache_basename",
    "unique_pair_states",
]

_L4B_RIM_CACHE_RE = re.compile(
    r"^pair_(?P<id_lo>\d+)__(?P<id_hi>\d+)_"
    r"sx(?P<sx_lo>[+-]?\d+)_sy(?P<sy_lo>[+-]?\d+)_"
    r"sx(?P<sx_hi>[+-]?\d+)_sy(?P<sy_hi>[+-]?\d+)_rim\.npz$"
)


def abutting_undirected_pairs(master: np.ndarray) -> np.ndarray:
    """Return ``(n_pairs, 2)`` int32 undirected master skycell ids ``(min_id, max_id)``.

    A pair is recorded when horizontally or vertically adjacent pixels belong to
    different non-negative skycell ids (4-neighbour topology on the master map).
    """
    master = np.asarray(master)
    chunks: list[np.ndarray] = []
    for s0, s1 in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])]:
        a = master[s0].ravel()
        b = master[s1].ravel()
        m = (a != b) & (a >= 0) & (b >= 0)
        aa = a[m]
        bb = b[m]
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        chunks.append(np.stack([lo, hi], axis=1))
    if not chunks:
        return np.zeros((0, 2), dtype=np.int32)
    stacked = np.concatenate(chunks, axis=0)
    return np.unique(stacked, axis=0).astype(np.int32)


def build_name_to_id(master_names: Sequence[str]) -> dict[str, int]:
    """Map stripped master-table skycell name -> dense master id index."""
    return {str(name).strip(): int(i) for i, name in enumerate(master_names)}


def build_col_of_name(skycell_names: Sequence[str]) -> dict[str, int]:
    """Map skycell name -> column index in shift-schedule arrays."""
    return {str(name).strip(): int(i) for i, name in enumerate(skycell_names)}


def pair_column_indices(
    pair_ids: np.ndarray,
    *,
    name_to_id: Mapping[str, int],
    col_of_name: Mapping[str, int],
    idx_to_name: Mapping[int, str],
) -> np.ndarray:
    """Map undirected ``(id_lo, id_hi)`` pairs to shift-schedule column pairs.

    Skips pairs whose endpoints are absent from the schedule column map.
    """
    rows: list[tuple[int, int]] = []
    for id_lo, id_hi in np.asarray(pair_ids, dtype=np.int32):
        na = idx_to_name.get(int(id_lo))
        nb = idx_to_name.get(int(id_hi))
        if na is None or nb is None:
            continue
        ca = col_of_name.get(na)
        cb = col_of_name.get(nb)
        if ca is None or cb is None:
            continue
        rows.append((int(ca), int(cb)))
    if not rows:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def _pack_shift_pair(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Pack ``(sx, sy)`` int shifts into one int64 per element (notebook ``pack2``)."""
    sx_ = np.asarray(sx, dtype=np.int64)
    sy_ = np.asarray(sy, dtype=np.int64)
    return (sx_ << 16) | (sy_ & 0xFFFF)


def _unpack_shift_pair(packed: int) -> tuple[int, int]:
    sx = int(packed) >> 16
    sy = int(packed) & 0xFFFF
    if sy >= 0x8000:
        sy -= 0x10000
    return sx, sy


def unique_pair_states(
    sx_int: np.ndarray,
    sy_int: np.ndarray,
    pair_idx: np.ndarray,
    frame_valid: np.ndarray,
    *,
    pair_ids: np.ndarray | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
    """Enumerate unique pair-states over valid frames for each abutting border.

    Parameters
    ----------
    sx_int, sy_int
        Integer PS1 shifts shaped ``(n_frames, n_skycells)``.
    pair_idx
        ``(n_borders, 2)`` column indices ``(col_a, col_b)`` into the shift arrays.
        Column 0 is the lower master id side when ``pair_ids`` is supplied.
    frame_valid
        Boolean mask of frames participating in the enumeration.
    pair_ids
        Optional ``(n_borders, 2)`` undirected master ids ``(id_lo, id_hi)`` aligned
        with ``pair_idx``. When omitted, column indices are used as stand-in ids.

    Returns
    -------
    list of ``(id_a, id_b, sx_a, sy_a, sx_b, sy_b)``
        One entry per unique 4-tuple on each border (sum equals
        ``n_type2_pair_states_sum`` in the trigger notebook).
    """
    sx_int = np.asarray(sx_int)
    sy_int = np.asarray(sy_int)
    pair_idx = np.asarray(pair_idx, dtype=np.int32)
    valid = np.asarray(frame_valid, dtype=bool)

    if pair_idx.ndim != 2 or pair_idx.shape[1] != 2:
        raise ValueError(f"pair_idx must be (n_borders, 2), got {pair_idx.shape}")
    if pair_ids is not None:
        pair_ids = np.asarray(pair_ids, dtype=np.int32)
        if pair_ids.shape != pair_idx.shape:
            raise ValueError(
                f"pair_ids shape {pair_ids.shape} != pair_idx shape {pair_idx.shape}"
            )

    n_borders = pair_idx.shape[0]
    col_a = pair_idx[:, 0]
    col_b = pair_idx[:, 1]

    sx_a = sx_int[:, col_a]
    sy_a = sy_int[:, col_a]
    sx_b = sx_int[:, col_b]
    sy_b = sy_int[:, col_b]

    pa = _pack_shift_pair(sx_a, sy_a)
    pb = _pack_shift_pair(sx_b, sy_b)
    pair_state = (pa << 32) | (pb & 0xFFFFFFFF)

    out: list[tuple[int, int, int, int, int, int]] = []
    for p in range(n_borders):
        if pair_ids is not None:
            id_a, id_b = int(pair_ids[p, 0]), int(pair_ids[p, 1])
        else:
            id_a, id_b = int(col_a[p]), int(col_b[p])

        unique_packed = np.unique(pair_state[valid, p])
        for packed in unique_packed:
            pa_u = int(packed) >> 32
            pb_u = int(packed) & 0xFFFFFFFF
            sx_a_u, sy_a_u = _unpack_shift_pair(pa_u)
            sx_b_u, sy_b_u = _unpack_shift_pair(pb_u)
            out.append((id_a, id_b, sx_a_u, sy_a_u, sx_b_u, sy_b_u))

    return out


def count_unique_pair_states_sum(
    sx_int: np.ndarray,
    sy_int: np.ndarray,
    pair_idx: np.ndarray,
    frame_valid: np.ndarray,
    *,
    pair_ids: np.ndarray | None = None,
) -> int:
    """Return ``n_type2_pair_states_sum``: unique 4-tuples summed over borders."""
    sx_int = np.asarray(sx_int)
    sy_int = np.asarray(sy_int)
    pair_idx = np.asarray(pair_idx, dtype=np.int32)
    valid = np.asarray(frame_valid, dtype=bool)

    if pair_idx.size == 0:
        return 0

    col_a = pair_idx[:, 0]
    col_b = pair_idx[:, 1]
    pa = _pack_shift_pair(sx_int[:, col_a], sy_int[:, col_a])
    pb = _pack_shift_pair(sx_int[:, col_b], sy_int[:, col_b])
    pair_state = (pa << 32) | (pb & 0xFFFFFFFF)

    n_states = np.array(
        [len(np.unique(pair_state[valid, p])) for p in range(pair_state.shape[1])],
        dtype=np.int64,
    )
    return int(n_states.sum())


def l4b_rim_cache_basename(
    id_a: int,
    id_b: int,
    sx_a: int,
    sy_a: int,
    sx_b: int,
    sy_b: int,
) -> str:
    """Basename for one L4b F2 rim Exact cache entry (undirected pair-state key)."""
    id_lo, id_hi = (int(id_a), int(id_b)) if int(id_a) <= int(id_b) else (int(id_b), int(id_a))
    if id_a == id_lo:
        sx_lo, sy_lo, sx_hi, sy_hi = int(sx_a), int(sy_a), int(sx_b), int(sy_b)
    else:
        sx_lo, sy_lo, sx_hi, sy_hi = int(sx_b), int(sy_b), int(sx_a), int(sy_a)
    return (
        f"pair_{id_lo}__{id_hi}_"
        f"sx{sx_lo:+d}_sy{sy_lo:+d}_"
        f"sx{sx_hi:+d}_sy{sy_hi:+d}_rim.npz"
    )


def parse_l4b_rim_cache_basename(
    name: str,
) -> tuple[int, int, int, int, int, int] | None:
    """Parse :func:`l4b_rim_cache_basename`; return ``(id_lo, id_hi, sx_lo, sy_lo, sx_hi, sy_hi)``."""
    m = _L4B_RIM_CACHE_RE.match(str(name))
    if not m:
        return None
    return (
        int(m.group("id_lo")),
        int(m.group("id_hi")),
        int(m.group("sx_lo")),
        int(m.group("sy_lo")),
        int(m.group("sx_hi")),
        int(m.group("sy_hi")),
    )
