"""Field-mode star mini-template shift builder (WS3)."""

import numpy as np
import pandas as pd

from syndiff_pipeline.star.mini_downsample import build_field_star_shifts


def _group_shifts():
    # 3 groups over skycells A, B (+ an out-of-ROI cell C); groups 0 and 2 share
    # the same (A, B) integer shifts and must collapse to one mini template.
    return pd.DataFrame(
        [
            dict(group_id=0, skycell="A", sx_int=1, sy_int=0),
            dict(group_id=0, skycell="B", sx_int=2, sy_int=-1),
            dict(group_id=0, skycell="C", sx_int=9, sy_int=9),
            dict(group_id=1, skycell="A", sx_int=1, sy_int=1),
            dict(group_id=1, skycell="B", sx_int=2, sy_int=-1),
            dict(group_id=2, skycell="A", sx_int=1, sy_int=0),
            dict(group_id=2, skycell="B", sx_int=2, sy_int=-1),
        ]
    )


def test_signature_dedup_and_group_map():
    offsets, shifts_dict, group_to_index = build_field_star_shifts(
        _group_shifts(), [0, 1, 2], ["A", "B"]
    )
    # groups 0 and 2 collapse -> 2 distinct mini templates
    assert offsets.shape == (2, 2)
    assert group_to_index == {0: 0, 1: 1, 2: 0}
    assert set(shifts_dict) == {(0.0, 0.0), (1.0, 0.0)}


def test_shifts_restricted_to_involved_skycells():
    _, shifts_dict, _ = build_field_star_shifts(_group_shifts(), [0], ["A", "B"])
    df = shifts_dict[(0.0, 0.0)]
    assert list(df["NAME"]) == ["A", "B"]  # out-of-ROI 'C' dropped
    assert list(df["shift_x"]) == [1, 2]
    assert list(df["shift_y"]) == [0, -1]


def test_missing_skycell_defaults_zero():
    # group 5 has no row for B -> B defaults to (0, 0)
    gs = pd.DataFrame([dict(group_id=5, skycell="A", sx_int=3, sy_int=4)])
    _, shifts_dict, group_to_index = build_field_star_shifts(gs, [5], ["A", "B"])
    df = shifts_dict[(0.0, 0.0)]
    assert dict(zip(df["NAME"], zip(df["shift_x"], df["shift_y"]))) == {
        "A": (3, 4),
        "B": (0, 0),
    }
    assert group_to_index == {5: 0}


def test_empty_groups():
    offsets, shifts_dict, group_to_index = build_field_star_shifts(
        _group_shifts(), [], ["A", "B"]
    )
    assert offsets.shape == (0, 2)
    assert shifts_dict == {}
    assert group_to_index == {}
