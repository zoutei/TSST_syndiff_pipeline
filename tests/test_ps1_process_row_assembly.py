"""Regression test for the ``assemble_row_from_bundles`` global-vs-row-local
x-offset bug.

Background
----------
``extract_projection_metadata`` computes ``starting_x`` as the
*projection-global* minimum x-grid-coordinate across every row, while
``create_master_array_config`` sizes the row-assembly master array purely
from ``max_cells_per_row`` (the largest cell *count* in any single row).
PS1's skycell tessellation isn't a uniform rectangle: a row near a
projection/declination edge can have fewer cells than ``max_cells_per_row``
while still starting at an x-coordinate well to the right of the global
minimum.

``assemble_row_from_bundles`` used to compute each cell's placement offset
as ``x_coord - config.starting_x`` (the global minimum). For a row whose own
cells start to the right of the global minimum, this inflated the offset
enough to overflow the master array -- even though the row has fewer cells
than the array was sized for -- and the cell was silently dropped with a
"[Assembler] Cell ... out of bounds for master array" warning. It also meant
the "first cell in this row" branch (``cell_index == 0``, which skips the
overlap trim) essentially only fired for the row containing the global
minimum, so other rows' leftmost cell was incorrectly trimmed by
``EFFECTIVE_OVERLAP`` pixels even when nothing overflowed.

This test builds a two-row projection where row 1 starts well to the right
of row 0 (which holds the global minimum) and has fewer cells than row 0,
and asserts every cell in both rows is placed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing import ps1_process as pp


def _cell_image(size: int, fill: float) -> np.ndarray:
    return np.full((size, size), fill, dtype=np.float32)


def _bundle(skycell_id: str, x_coord: int, image: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "skycell_id": skycell_id,
        "x_coord": x_coord,
        "combined_image": image,
        "combined_mask": mask,
    }


def _projection_df() -> pd.DataFrame:
    """Row 0 has 3 cells starting at the global minimum (x=0); row 1 has
    only 2 cells but starts well to the right (x=5), i.e. fewer cells than
    ``max_cells_per_row`` while not containing the global minimum."""
    rows = []
    for x in (0, 1, 2):
        rows.append(
            {"projection": "skycell.9999", "y": 0, "NAME": f"skycell.9999.0{x}", "x": x, "NAXIS1": 520, "NAXIS2": 520}
        )
    for x in (5, 6):
        rows.append(
            {"projection": "skycell.9999", "y": 1, "NAME": f"skycell.9999.1{x}", "x": x, "NAXIS1": 520, "NAXIS2": 520}
        )
    return pd.DataFrame(rows)


def test_row_not_starting_at_global_minimum_places_all_cells():
    df = _projection_df()
    metadata = pp.extract_projection_metadata(df, "skycell.9999")
    assert metadata["starting_x"] == 0
    assert metadata["max_cells_per_row"] == 3

    config = pp.create_master_array_config(metadata)
    state = pp.initialize_processing_state(config)

    row1_bundles = [
        _bundle("skycell.9999.15", 5, _cell_image(520, 1.0), np.zeros((520, 520), dtype=np.uint16)),
        _bundle("skycell.9999.16", 6, _cell_image(520, 2.0), np.zeros((520, 520), dtype=np.uint16)),
    ]

    positions, masks = pp.assemble_row_from_bundles(state.current_array, row1_bundles, config)

    assert set(positions.keys()) == {"skycell.9999.15", "skycell.9999.16"}
    assert set(masks.keys()) == {"skycell.9999.15", "skycell.9999.16"}


def test_row_starting_at_global_minimum_still_places_all_cells():
    """Sanity check: the row containing the global minimum is unaffected."""
    df = _projection_df()
    metadata = pp.extract_projection_metadata(df, "skycell.9999")
    config = pp.create_master_array_config(metadata)
    state = pp.initialize_processing_state(config)

    row0_bundles = [
        _bundle("skycell.9999.00", 0, _cell_image(520, 1.0), np.zeros((520, 520), dtype=np.uint16)),
        _bundle("skycell.9999.01", 1, _cell_image(520, 2.0), np.zeros((520, 520), dtype=np.uint16)),
        _bundle("skycell.9999.02", 2, _cell_image(520, 3.0), np.zeros((520, 520), dtype=np.uint16)),
    ]

    positions, masks = pp.assemble_row_from_bundles(state.current_array, row0_bundles, config)

    assert set(positions.keys()) == {"skycell.9999.00", "skycell.9999.01", "skycell.9999.02"}
    assert set(masks.keys()) == {"skycell.9999.00", "skycell.9999.01", "skycell.9999.02"}


def test_row_offset_placement_is_self_consistent_and_in_bounds():
    """The row-local first cell should use the untrimmed (cell_index == 0)
    branch, and every placed cell's bounds must fit inside the master array
    -- the two symptoms of the original bug."""
    df = _projection_df()
    metadata = pp.extract_projection_metadata(df, "skycell.9999")
    config = pp.create_master_array_config(metadata)
    state = pp.initialize_processing_state(config)

    row1_bundles = [
        _bundle("skycell.9999.15", 5, _cell_image(520, 1.0), np.zeros((520, 520), dtype=np.uint16)),
        _bundle("skycell.9999.16", 6, _cell_image(520, 2.0), np.zeros((520, 520), dtype=np.uint16)),
    ]

    positions, _masks = pp.assemble_row_from_bundles(state.current_array, row1_bundles, config)

    for cell_name, (x_start, x_end, y_start, y_end) in positions.items():
        assert 0 <= x_start < x_end <= state.current_array.shape[1], cell_name
        assert 0 <= y_start < y_end <= state.current_array.shape[0], cell_name

    # The row's own first cell (x_coord=5) should land at PAD_SIZE (the
    # untrimmed cell_index==0 placement), not further right as if it were
    # offset from the global minimum (x_coord=0).
    x_start_first, _, _, _ = positions["skycell.9999.15"]
    assert x_start_first == pp.PAD_SIZE
