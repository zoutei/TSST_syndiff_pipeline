"""Deterministic producer ownership tests for cross-projection padding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing.cross_projection_padding import (
    compose_ordered_reprojected_patches,
    ordered_cross_projection_placements,
)


def _metadata():
    return {
        "projection": "skycell.1234",
        "rows": {0: [("skycell.1234.001", 0), ("skycell.1234.002", 1)]},
    }


def _table():
    return pd.DataFrame([
        {
            "projection": "1234", "y": 0, "x": 0, "NAME": "skycell.1234.001",
            "pad_skycell_left": "skycell.9000.010/skycell.8000.011",
            "pad_skycell_bottom": "skycell.7000.012",
        },
        {
            "projection": "1234", "y": 0, "x": 1, "NAME": "skycell.1234.002",
            "pad_skycell_right": "skycell.6000.013",
        },
    ])


def test_normalized_placements_keep_recipient_and_slash_order():
    placements = ordered_cross_projection_placements(_metadata(), _table(), 0, [0, 1])
    assert [(p.source_skycell, p.recipient_skycell, p.location, p.priority) for p in placements] == [
        ("skycell.7000.012", "skycell.1234.001", "bottom", 0),
        ("skycell.9000.010", "skycell.1234.001", "left", 1),
        ("skycell.8000.011", "skycell.1234.001", "left", 2),
        ("skycell.6000.013", "skycell.1234.002", "right", 3),
    ]


def test_overlapping_reprojected_patches_use_priority_not_input_completion_order():
    target = np.zeros((4, 4), dtype=np.float64)
    # Deliberately provide later-priority work first, as a thread pool might.
    operations = [
        (2, target, (1, 3, 1, 3), np.full((2, 2), 20.0), np.ones((2, 2))),
        (1, target, (1, 3, 1, 3), np.full((2, 2), 10.0), np.ones((2, 2))),
    ]
    compose_ordered_reprojected_patches(operations)
    np.testing.assert_array_equal(target[1:3, 1:3], 20.0)
