"""Unit tests for abutting-pair discovery and L4b pair-state enumeration."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.template_creation.processing.field_abutting import (
    abutting_undirected_pairs,
    build_col_of_name,
    build_name_to_id,
    count_unique_pair_states_sum,
    l4a_exact_path,
    l4b_rim_cache_basename,
    l4b_rim_path,
    load_l4b_rim_side,
    pair_column_indices,
    pair_subdir_name,
    parse_l4b_rim_cache_basename,
    unique_pair_states,
    write_l4b_rim_cache,
)


def test_abutting_undirected_pairs_synthetic_grid():
    """2x2 checkerboard of ids 0,1,2,3 has known horizontal+vertical contacts."""
    master = np.array(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=np.int32,
    )
    pairs = abutting_undirected_pairs(master)
    expected = {(0, 1), (0, 2), (1, 3), (2, 3)}
    got = {tuple(row) for row in pairs}
    assert got == expected
    assert pairs.dtype == np.int32
    # undirected canonical form: min_id first
    for lo, hi in pairs:
        assert lo < hi


def test_abutting_undirected_pairs_ignores_invalid():
    master = np.full((3, 3), -1, dtype=np.int32)
    master[1, 1] = 5
    master[1, 2] = 7
    pairs = abutting_undirected_pairs(master)
    assert len(pairs) == 1
    assert tuple(pairs[0]) == (5, 7)


def test_undirected_key_symmetry():
    master = np.zeros((4, 6), dtype=np.int32)
    master[:, :3] = 10
    master[:, 3:] = 20
    pairs = abutting_undirected_pairs(master)
    assert len(pairs) == 1
    id_a, id_b = int(pairs[0, 0]), int(pairs[0, 1])
    assert id_a < id_b
    name_a = l4b_rim_cache_basename(id_a, id_b, 1, 0, 2, -1)
    name_b = l4b_rim_cache_basename(id_b, id_a, 2, -1, 1, 0)
    assert name_a == name_b


def test_unique_pair_states_count_matches_notebook_metric():
    """Cardinality must equal sum of per-border unique packed states."""
    n_frames, n_cells = 6, 2
    pair_idx = np.array([[0, 1]], dtype=np.int32)
    pair_ids = np.array([[10, 20]], dtype=np.int32)
    frame_valid = np.ones(n_frames, dtype=bool)

    sx_int = np.zeros((n_frames, n_cells), dtype=np.int32)
    sy_int = np.zeros((n_frames, n_cells), dtype=np.int32)
    # Three distinct pair-states on the single border.
    sx_int[0, 0], sy_int[0, 0] = 0, 0
    sx_int[0, 1], sy_int[0, 1] = 1, 0
    sx_int[1, 0], sy_int[1, 0] = 0, 0
    sx_int[1, 1], sy_int[1, 1] = 1, 1
    sx_int[2, 0], sy_int[2, 0] = 2, 0
    sx_int[2, 1], sy_int[2, 1] = 0, 0
    # Repeat states on later frames
    sx_int[3:, 0] = sx_int[2, 0]
    sy_int[3:, 0] = sy_int[2, 0]
    sx_int[3:, 1] = sx_int[2, 1]
    sy_int[3:, 1] = sy_int[2, 1]

    states = unique_pair_states(
        sx_int, sy_int, pair_idx, frame_valid, pair_ids=pair_ids
    )
    assert len(states) == 3
    assert count_unique_pair_states_sum(
        sx_int, sy_int, pair_idx, frame_valid, pair_ids=pair_ids
    ) == 3

    expected = {
        (10, 20, 0, 0, 1, 0),
        (10, 20, 0, 0, 1, 1),
        (10, 20, 2, 0, 0, 0),
    }
    assert set(states) == expected


def test_unique_pair_states_respects_frame_valid_mask():
    n_frames = 4
    pair_idx = np.array([[0, 1]], dtype=np.int32)
    frame_valid = np.array([True, False, True, True])

    sx_int = np.array(
        [
            [0, 0],
            [9, 9],
            [1, 0],
            [1, 0],
        ],
        dtype=np.int32,
    )
    sy_int = np.zeros((n_frames, 2), dtype=np.int32)

    n_sum = count_unique_pair_states_sum(sx_int, sy_int, pair_idx, frame_valid)
    # valid frames 0,2,3 -> states (0,0)|(0,0) and (1,0)|(0,0)
    assert n_sum == 2


def test_unique_pair_states_multiple_borders_sums():
    """Sum over borders matches notebook ``n_type2_pair_states_sum`` pattern."""
    n_frames = 3
    n_cells = 3
    pair_idx = np.array([[0, 1], [1, 2]], dtype=np.int32)
    frame_valid = np.ones(n_frames, dtype=bool)

    sx_int = np.zeros((n_frames, n_cells), dtype=np.int32)
    sy_int = np.zeros((n_frames, n_cells), dtype=np.int32)
    # border 0-1: 2 states across frames
    sx_int[:, 1] = [0, 1, 1]
    # border 1-2: 3 states across frames
    sx_int[:, 2] = [0, 0, 2]

    n_sum = count_unique_pair_states_sum(sx_int, sy_int, pair_idx, frame_valid)
    assert n_sum == 5
    assert len(unique_pair_states(sx_int, sy_int, pair_idx, frame_valid)) == 5


def test_l4b_rim_cache_basename_roundtrip():
    name = l4b_rim_cache_basename(42, 99, -1, 2, 3, -4)
    parsed = parse_l4b_rim_cache_basename(name)
    assert parsed == (42, 99, -1, 2, 3, -4)
    assert name == "pair_42__99_sx-1_sy+2_sx+3_sy-4_rim.npz"


def test_pair_column_indices_and_name_helpers():
    master_names = ["skycell.0.0", "skycell.1.0", "skycell.2.0"]
    name_to_id = build_name_to_id(master_names)
    col_of_name = build_col_of_name(["skycell.1.0", "skycell.0.0"])
    idx_to_name = {v: k for k, v in name_to_id.items()}

    pair_ids = np.array([[0, 1]], dtype=np.int32)
    kept_ids, cols = pair_column_indices(
        pair_ids,
        name_to_id=name_to_id,
        col_of_name=col_of_name,
        idx_to_name=idx_to_name,
    )
    assert cols.shape == (1, 2)
    assert kept_ids.shape == (1, 2)
    assert tuple(kept_ids[0]) == (0, 1)
    # schedule column for skycell.0.0 is 1, for skycell.1.0 is 0
    assert tuple(cols[0]) == (1, 0)


def test_pair_column_indices_drops_pairs_missing_from_schedule():
    """Endpoints absent from the schedule are dropped from both outputs, row-aligned."""
    master_names = ["skycell.0.0", "skycell.1.0", "skycell.2.0"]
    name_to_id = build_name_to_id(master_names)
    idx_to_name = {v: k for k, v in name_to_id.items()}
    # Only skycell.0.0 and skycell.1.0 made it into the shift schedule;
    # skycell.2.0 (id 2) did not (e.g. filtered out upstream).
    col_of_name = build_col_of_name(["skycell.1.0", "skycell.0.0"])

    pair_ids = np.array([[0, 1], [1, 2]], dtype=np.int32)
    kept_ids, cols = pair_column_indices(
        pair_ids,
        name_to_id=name_to_id,
        col_of_name=col_of_name,
        idx_to_name=idx_to_name,
    )
    # Only the (0, 1) pair survives; (1, 2) is dropped from both arrays.
    assert kept_ids.shape == (1, 2)
    assert cols.shape == (1, 2)
    assert tuple(kept_ids[0]) == (0, 1)
    assert tuple(cols[0]) == (1, 0)


def test_write_l4b_rim_cache_survives_shifts_beyond_int16_range(tmp_path):
    """Regression: shift magnitudes beyond int16's +-32767 range used to
    raise ("Python integer -34109 out of bounds for int16") and silently
    drop the whole rim-cache entry -- observed in production on a real
    remap backfill. sx/sy fields must be wide enough (int32, matching
    shift_schedule.py's schema for the same fields) to hold real shifts.
    """
    path = tmp_path / "rim.npz"
    exact_tid_lo = np.array([[1, -1], [-1, 2]], dtype=np.int32)
    exact_tid_hi = np.array([[-1, 3], [4, -1]], dtype=np.int32)
    write_l4b_rim_cache(
        path,
        exact_tid_lo=exact_tid_lo,
        exact_tid_hi=exact_tid_hi,
        id_lo=977,
        id_hi=979,
        sx_lo=10402,
        sy_lo=-34109,
        sx_hi=10415,
        sy_hi=-35399,
        pair_epoch_id=63,
        rep_frame_index=0,
    )
    with np.load(path) as z:
        assert int(z["sx_lo"]) == 10402
        assert int(z["sy_lo"]) == -34109
        assert int(z["sx_hi"]) == 10415
        assert int(z["sy_hi"]) == -35399
    idx_lo, val_lo = load_l4b_rim_side(path, skycell_id=977)
    assert list(val_lo) == [1, 2]
    idx_hi, val_hi = load_l4b_rim_side(path, skycell_id=979)
    assert list(val_hi) == [3, 4]


def test_l4a_l4b_epoch_paths():
    from pathlib import Path

    root = Path("/tmp/exact")
    p = l4a_exact_path(root, "skycell.1.2", 3, 5, -1)
    assert p.as_posix().endswith("skycell.1.2/e3_sx+5_sy-1_exact.npz")
    q = l4b_rim_path(root, 20, 10, 2, 1, 0, 0, 0)
    assert q.as_posix().endswith("pair_10__20/e2_sx+0_sy+0_sx+1_sy+0_rim.npz")
    assert pair_subdir_name(20, 10) == "pair_10__20"
