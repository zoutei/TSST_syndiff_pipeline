"""
Ragged-row master-array sizing and placement (s0020/3/3 stale-band regression).

When the TESS chip footprint clips a PS1 projection diagonally, the selected
rows are ragged: ``starting_x`` (projection-wide minimum x) comes from one row
while other rows extend to a larger maximum x. Placement anchors every cell at
slot ``x - starting_x``, so the master row array must span the global x-range
— sizing it by the longest row's *cell count* undersizes it and the high-x
cells overflow. Historically overflowing cells were warned-and-skipped,
silently leaving stale or missing convolved data in the output Zarr (observed
as bright cross-projection bands in s0020/3/3 templates; 15 cells of
projections 2559/2589/2627 affected).

Also covers the companion ragged-row fixes in cross_projection_padding:
``create_master_array_wcs`` anchoring and global-slot ``actual_index``.
"""

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.template_creation.processing.ps1_process import (
    CELL_OVERLAP,
    PAD_SIZE,
    assemble_row_from_bundles,
    create_master_array_config,
    extract_projection_metadata,
)
from syndiff_pipeline.template_creation.processing.cross_projection_padding import (
    _parse_row_padding_requirements_df,
    create_master_array_wcs,
)

W, H = 600, 500  # synthetic cell dims (must exceed CELL_OVERLAP=480)


def _cell_row(projection, y, x, width=W, height=H):
    return {
        "NAME": f"skycell.{projection}.0{y}{x}",
        "projection": projection,
        "y": y,
        "x": x,
        "NAXIS1": width,
        "NAXIS2": height,
        "CRVAL1": 100.0,
        "CRVAL2": 30.0,
        "CRPIX1": 240.0 - x * (width - CELL_OVERLAP),
        "CRPIX2": 240.0 - y * (height - CELL_OVERLAP),
        "CD1_1": -7e-5,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 7e-5,
    }


def _ragged_df():
    """Mimics s0020/3/3 projection 2559: starting_x from one row, max x from another."""
    rows = []
    for x in (3, 4, 5):  # row y=3 sets starting_x=3
        rows.append(_cell_row(2559, 3, x))
    for x in (7, 8, 9):  # row y=9 extends to x=9 (slot index 6 > 3 cells)
        rows.append(_cell_row(2559, 9, x))
    return pd.DataFrame(rows)


def test_config_spans_global_x_range():
    metadata = extract_projection_metadata(_ragged_df(), "2559")
    assert metadata["starting_x"] == 3
    assert metadata["max_cells_per_row"] == 3  # the old, undersized quantity
    assert metadata["slot_count"] == 7  # x=3..9

    config = create_master_array_config(metadata)
    stride = config.cell_width - CELL_OVERLAP
    for x in (3, 4, 5, 7, 8, 9):
        cell_index = x - config.starting_x
        target_x_end = PAD_SIZE + cell_index * stride + W
        assert target_x_end <= config.width, f"cell at x={x} would overflow"


def test_assemble_places_ragged_row_cells():
    metadata = extract_projection_metadata(_ragged_df(), "2559")
    config = create_master_array_config(metadata)
    target = np.empty((config.height, config.width), dtype=np.float32)

    bundles = [
        {
            "skycell_id": f"skycell.2559.09{x}",
            "combined_image": np.full((H, W), float(x), dtype=np.float32),
            "combined_mask": np.zeros((H, W), dtype=np.uint16),
            "x_coord": x,
        }
        for x in (7, 8, 9)
    ]
    positions, masks = assemble_row_from_bundles(target, bundles, config)
    assert set(positions) == {"skycell.2559.097", "skycell.2559.098", "skycell.2559.099"}
    # the x=9 cell (slot 6) is the one the old sizing dropped
    x_start, x_end, y_start, y_end = positions["skycell.2559.099"]
    assert x_end <= config.width
    stride = config.cell_width - CELL_OVERLAP
    assert x_start == PAD_SIZE + 6 * stride


def test_assemble_raises_instead_of_silently_skipping():
    metadata = extract_projection_metadata(_ragged_df(), "2559")
    config = create_master_array_config(metadata)
    # Undersized array reproduces the pre-fix overflow condition.
    target = np.empty((config.height, config.width - 2 * W), dtype=np.float32)
    bundles = [
        {
            "skycell_id": "skycell.2559.099",
            "combined_image": np.zeros((H, W), dtype=np.float32),
            "combined_mask": np.zeros((H, W), dtype=np.uint16),
            "x_coord": 9,
        }
    ]
    with pytest.raises(RuntimeError, match="out of bounds for master array"):
        assemble_row_from_bundles(target, bundles, config)


def test_master_wcs_anchored_at_row_first_slot():
    df = _ragged_df()
    metadata = extract_projection_metadata(df, "2559")
    config = create_master_array_config(metadata)
    stride = config.cell_width - CELL_OVERLAP

    # Row y=3 starts at starting_x: anchor must stay PAD_SIZE (legacy behavior).
    wcs3 = create_master_array_wcs(metadata, config, 3)
    first3 = df[(df["y"] == 3) & (df["x"] == 3)].iloc[0]
    assert wcs3.wcs.crpix[0] == pytest.approx(first3["CRPIX1"] + PAD_SIZE)

    # Row y=9 starts at x=7 (slot 4): anchor must include the slot offset.
    wcs9 = create_master_array_wcs(metadata, config, 9)
    first9 = df[(df["y"] == 9) & (df["x"] == 7)].iloc[0]
    assert wcs9.wcs.crpix[0] == pytest.approx(first9["CRPIX1"] + PAD_SIZE + 4 * stride)

    # Both rows' WCS must agree with cell placement: the array pixel where a
    # cell's CRPIX lands must map back to the shared CRVAL.
    for wcs, first, slot in ((wcs3, first3, 0), (wcs9, first9, 4)):
        ax = PAD_SIZE + slot * stride + (first["CRPIX1"] - 1)
        ay = PAD_SIZE + (first["CRPIX2"] - 1)
        world = wcs.pixel_to_world(ax, ay)
        assert world.ra.deg == pytest.approx(first["CRVAL1"], abs=1e-8)
        assert world.dec.deg == pytest.approx(first["CRVAL2"], abs=1e-8)


def test_padding_actual_index_is_global_slot():
    df = _ragged_df()
    # Attach a cross-projection pad requirement to the ragged top row's first
    # cell (y=9, x=7): its index within the row is 0, but its global slot is 4.
    df["pad_skycell_top"] = ""
    df.loc[(df["y"] == 9) & (df["x"] == 7), "pad_skycell_top"] = "skycell.2560.007"

    metadata = extract_projection_metadata(df, "2559")
    meta = {
        "projection": "2559",
        "rows": metadata["rows"],
        "dataframe": metadata["dataframe"],
        "starting_x": metadata["starting_x"],
    }
    all_row_ids = sorted(metadata["rows"].keys())
    reqs = _parse_row_padding_requirements_df(meta, df, 9, all_row_ids)
    assert "skycell.2560.007" in reqs
    assert reqs["skycell.2560.007"].actual_index == 4

    # Metadata without starting_x (identify_all_padding_sources variant) must
    # derive the same global slot from the projection dataframe.
    meta_no_sx = {k: v for k, v in meta.items() if k != "starting_x"}
    reqs2 = _parse_row_padding_requirements_df(meta_no_sx, df, 9, all_row_ids)
    assert reqs2["skycell.2560.007"].actual_index == 4
