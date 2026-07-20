"""Unit tests for skycell shift debug plot helpers."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.template_creation.processing.shift_schedule_plots import (
    _orbit_segments,
    _pick_grid_skycells,
)


def test_orbit_segments_split() -> None:
    assert _orbit_segments(10, []) == [(0, 10)]
    assert _orbit_segments(10, [[0, 4], [4, 10]]) == [(0, 4), (4, 10)]


def test_pick_grid_bottom_left_is_low_xy() -> None:
    # 3×3 centroids on a square; names are row-major from low-y up.
    xs = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    ys = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    names = np.array([f"c{i}" for i in range(9)])
    picks = _pick_grid_skycells(xs, ys, names)
    bl = next(p for p in picks if p["plot_row"] == 2 and p["plot_col"] == 0)
    tr = next(p for p in picks if p["plot_row"] == 0 and p["plot_col"] == 2)
    assert bl["col_idx"] == 0  # (0,0)
    assert tr["col_idx"] == 8  # (2,2)
    center = next(p for p in picks if p["is_center"])
    assert center["col_idx"] == 4
