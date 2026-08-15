"""Abutting skycell pairs and Type II (L4b) pair-state enumeration.

Ports notebook logic from ``dev/distortion_aware_template/`` for production L4b:
undirected master-map neighbours and unique ``(sx_A, sy_A, sx_B, sy_B)`` keys
per border (cardinality ``n_type2_pair_states_sum``), not frame-to-frame
transitions (``count_l4b_events``).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "L4B_RIM_FORMAT_VERSION",
    "abutting_undirected_pairs",
    "build_col_of_name",
    "build_name_to_id",
    "count_unique_pair_states_sum",
    "l4a_exact_path",
    "l4b_rim_cache_basename",
    "l4b_rim_is_sparse",
    "l4b_rim_path",
    "load_l4b_rim_side",
    "pair_column_indices",
    "pair_subdir_name",
    "parse_l4b_rim_cache_basename",
    "sparsify_l4b_rim_payload",
    "unique_pair_states",
    "write_l4b_rim_cache",
]

# v1 = dense ``exact_tid_lo``/``exact_tid_hi`` (two PS1-shaped int32 arrays).
# v2 = sparse: only valid (tid >= 0) rim pixels are stored, as ``didx_*`` (the
# *differences* between successive flat indices, int32) plus ``val_*``.
#
# Storing raw indices would be a regression on disk: the dense arrays are almost
# entirely the -1 sentinel and so compress ~1500x, and a plain (idx, val) pair
# measured 5x *larger* than the dense file it replaced. The indices are sorted
# and the rim is contiguous, so their deltas are mostly 1 and compress far
# better. Measured on a real rim cache: dense 409 KB / 519 ms per read, raw
# sparse 1890 KB / 17 ms, delta sparse 74 KB / 10 ms -- 5.5x smaller *and* ~50x
# faster. Readers accept both layouts.
L4B_RIM_FORMAT_VERSION = 2

_L4B_RIM_CACHE_RE = re.compile(
    r"^pair_(?P<id_lo>\d+)__(?P<id_hi>\d+)_"
    r"sx(?P<sx_lo>[+-]?\d+)_sy(?P<sy_lo>[+-]?\d+)_"
    r"sx(?P<sx_hi>[+-]?\d+)_sy(?P<sy_hi>[+-]?\d+)_rim\.npz$"
)

_L4B_EPOCH_RIM_RE = re.compile(
    r"^e(?P<epoch_id>\d+)_"
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
) -> tuple[np.ndarray, np.ndarray]:
    """Map undirected ``(id_lo, id_hi)`` pairs to shift-schedule column pairs.

    Returns ``(pair_ids_kept, pair_idx)``, row-aligned: pairs whose endpoints
    are absent from the schedule column map are dropped from *both* arrays.
    Callers must use the returned ``pair_ids_kept`` (not the input
    ``pair_ids``) alongside ``pair_idx`` — zipping the original, unfiltered
    ``pair_ids`` with this ``pair_idx`` silently misaligns every pair after
    the first drop (or raises ``IndexError`` once ``pair_idx`` runs out).
    """
    kept_ids: list[tuple[int, int]] = []
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
        kept_ids.append((int(id_lo), int(id_hi)))
        rows.append((int(ca), int(cb)))
    if not rows:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)
    return np.asarray(kept_ids, dtype=np.int32), np.asarray(rows, dtype=np.int32)


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
    """Legacy flat basename for one L4b F2 rim Exact cache entry (undirected pair-state key).

    Prefer :func:`l4b_rim_path` for schema-v3 epoch caches under ``pair_*/`` subfolders.
    """
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


def pair_subdir_name(id_lo: int, id_hi: int) -> str:
    """Directory name under ``exact_cache_l4b/`` for an undirected abutting pair."""
    lo, hi = (int(id_lo), int(id_hi)) if int(id_lo) <= int(id_hi) else (int(id_hi), int(id_lo))
    return f"pair_{lo}__{hi}"


def l4a_exact_path(
    l4a_root: Path | str,
    skycell: str,
    epoch_id: int,
    sx_int: int,
    sy_int: int,
) -> Path:
    """Schema-v3 L4a Exact path: ``{l4a_root}/{skycell}/e{epoch}_sx±_sy±_exact.npz``."""
    name = str(skycell)
    fname = f"e{int(epoch_id)}_sx{int(sx_int):+d}_sy{int(sy_int):+d}_exact.npz"
    return Path(l4a_root) / name / fname


def l4b_rim_path(
    l4b_root: Path | str,
    id_lo: int,
    id_hi: int,
    pair_epoch_id: int,
    sx_lo: int,
    sy_lo: int,
    sx_hi: int,
    sy_hi: int,
) -> Path:
    """Schema-v3 L4b rim path under ``pair_{lo}__{hi}/``."""
    lo, hi = (int(id_lo), int(id_hi)) if int(id_lo) <= int(id_hi) else (int(id_hi), int(id_lo))
    if (int(id_lo), int(id_hi)) != (lo, hi):
        sx_lo, sy_lo, sx_hi, sy_hi = int(sx_hi), int(sy_hi), int(sx_lo), int(sy_lo)
    fname = (
        f"e{int(pair_epoch_id)}_"
        f"sx{int(sx_lo):+d}_sy{int(sy_lo):+d}_"
        f"sx{int(sx_hi):+d}_sy{int(sy_hi):+d}_rim.npz"
    )
    return Path(l4b_root) / pair_subdir_name(lo, hi) / fname


def sparsify_l4b_rim_payload(
    exact_tid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one dense rim side to ``(idx, val)`` over valid (``tid >= 0``) pixels.

    An empty/absent side (``exact_tid.size == 0``) yields two empty arrays.
    """
    flat = np.asarray(exact_tid, dtype=np.int32).ravel()
    if flat.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
    idx = np.flatnonzero(flat >= 0)
    return idx.astype(np.int64, copy=False), flat[idx].astype(np.int32, copy=False)


def _delta_encode_indices(idx: np.ndarray) -> np.ndarray:
    """Gap-encode ascending flat indices as int32 successive differences."""
    arr = np.asarray(idx, dtype=np.int64)
    if arr.size == 0:
        return np.array([], dtype=np.int32)
    return np.diff(arr, prepend=np.int64(0)).astype(np.int32)


def _delta_decode_indices(didx: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_delta_encode_indices`."""
    arr = np.asarray(didx)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    return np.cumsum(arr.astype(np.int64))


def write_l4b_rim_cache(
    path: Path | str,
    *,
    exact_tid_lo: np.ndarray,
    exact_tid_hi: np.ndarray,
    id_lo: int,
    id_hi: int,
    sx_lo: int,
    sy_lo: int,
    sx_hi: int,
    sy_hi: int,
    pair_epoch_id: int,
    rep_frame_index: int,
    ps1_shape: tuple[int, int] | None = None,
) -> Path:
    """Write one sparse (v2) L4b rim NPZ via temp file + atomic replace.

    The dense sides are stored as flat ``(idx, val)`` pairs. Writing through a
    temp file matters for resume safety: the legacy in-place
    ``np.savez_compressed`` could leave a truncated NPZ at the final path that a
    later ``is_file()`` skip check would wrongly treat as complete.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lo = np.asarray(exact_tid_lo, dtype=np.int32)
    hi = np.asarray(exact_tid_hi, dtype=np.int32)
    if ps1_shape is None:
        for side in (lo, hi):
            if side.ndim == 2:
                ps1_shape = (int(side.shape[0]), int(side.shape[1]))
                break
    idx_lo, val_lo = sparsify_l4b_rim_payload(lo)
    idx_hi, val_hi = sparsify_l4b_rim_payload(hi)

    payload = {
        "format_version": np.int32(L4B_RIM_FORMAT_VERSION),
        "didx_lo": _delta_encode_indices(idx_lo),
        "val_lo": val_lo,
        "didx_hi": _delta_encode_indices(idx_hi),
        "val_hi": val_hi,
        "id_lo": np.int32(id_lo),
        "id_hi": np.int32(id_hi),
        # int16 overflows real shift magnitudes (observed as large as ~35000,
        # well past int16's +-32767 range), raising and dropping the whole
        # rim-cache entry. shift_schedule.py's schema for these same fields
        # already declares int32, and field_remap.py's reader already
        # upcasts to int32 on load -- int16 here was simply too narrow.
        "sx_lo": np.int32(sx_lo),
        "sy_lo": np.int32(sy_lo),
        "sx_hi": np.int32(sx_hi),
        "sy_hi": np.int32(sy_hi),
        "pair_epoch_id": np.int32(pair_epoch_id),
        "rep_frame_index": np.int32(rep_frame_index),
    }
    if ps1_shape is not None:
        payload["ps1_shape"] = np.asarray(ps1_shape, dtype=np.int64)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".tmp.npz", dir=str(out.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(out)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return out


def l4b_rim_is_sparse(npz: Mapping[str, np.ndarray]) -> bool:
    """True when an opened L4b rim NPZ uses the v2 sparse layout."""
    files = getattr(npz, "files", None)
    keys = set(files) if files is not None else set(npz)
    return "didx_lo" in keys or "didx_hi" in keys


def load_l4b_rim_side(
    path: Path | str,
    *,
    skycell_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the side of an L4b rim cache owned by ``skycell_id`` as ``(idx, val)``.

    Reads only that side. Handles both the v2 sparse layout and the legacy v1
    dense layout (``exact_tid_lo``/``exact_tid_hi``) -- NPZ members are separate
    zip entries, so the unused side is never decompressed either way. Halving the
    decompression is worth ~2x on its own; a v2 cache is ~45x cheaper again.
    """
    with np.load(path) as z:
        id_lo = int(z["id_lo"])
        id_hi = int(z["id_hi"])
        if int(skycell_id) == id_lo:
            suffix = "lo"
        elif int(skycell_id) == id_hi:
            suffix = "hi"
        else:
            raise ValueError(
                f"skycell_id {skycell_id} not in L4b cache pair ({id_lo}, {id_hi})"
            )
        if l4b_rim_is_sparse(z):
            return (
                _delta_decode_indices(z[f"didx_{suffix}"]),
                np.asarray(z[f"val_{suffix}"], dtype=np.int32),
            )
        dense = np.asarray(z[f"exact_tid_{suffix}"], dtype=np.int32)
    return sparsify_l4b_rim_payload(dense)


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
