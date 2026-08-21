from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.template_creation.processing.pancakes import (
    master_skycells_csv_paths,
    save_master_mapping,
    save_updated_skycell_csv,
)


def test_mapgrid3_master_map_and_skycell_csv_publish_geometry(tmp_path):
    grid = MappingGrid.from_ffi_shape(
        32, 28, x_left_dead=4, x_right_dead=4, y_edge_strip=3,
        conv_pad_native=2, oversampling=1, mapgrid_version=3,
    )
    selected = pd.DataFrame({"NAME": ["skycell.1234.000"], "RA": [10.0], "DEC": [20.0]})
    mapping = np.zeros(grid.array_shape_native(), dtype=np.int32)
    save_master_mapping(
        mapping, selected, "ffi.fits", fits.Header({"SECTOR": 1}), grid.array_shape_native(),
        str(tmp_path), 1, 1, 1, oversampling_factor=1, mapping_grid=grid,
    )
    csv_partial, csv_final = master_skycells_csv_paths(str(tmp_path), 1, 1, 1, 1)
    # The stage publishes the CSV separately after worker completion; the
    # partial file is enough to verify the mapping-to-PS1 handoff contract.
    table = pd.read_csv(csv_partial)
    assert int(table.loc[0, "MAPGRID"]) == 3
    assert table.loc[0, "GEOMFP"] == grid.geometry_fingerprint
    assert table.loc[0, "COORDFRM"] == "full_ffi"
    assert table.loc[0, "SCIENCE_PAD_POLICY"] == "neutral_invalid"
    assert table.loc[0, "TEMPORAL_EXTRAP_POLICY"] == "bounded_support_pad"

    master = next(tmp_path.rglob("*master_pixels2skycells.fits.fz"))
    with fits.open(master) as hdul:
        hdr = hdul[1].header
        assert hdr["MAPGRID"] == 3
        assert hdr["GEOMFP"] == grid.geometry_fingerprint
        assert tuple(hdul[1].data.shape) == grid.array_shape_native()


def test_padding_update_preserves_mapping_grid_metadata(tmp_path):
    grid = MappingGrid.from_ffi_shape(
        32, 28, x_left_dead=4, x_right_dead=4, y_edge_strip=3,
        conv_pad_native=2, oversampling=1, mapgrid_version=3,
    )
    selected = pd.DataFrame({"NAME": ["skycell.1234.000"]})
    save_updated_skycell_csv(
        selected,
        str(tmp_path),
        1,
        1,
        1,
        mapping_grid=grid,
    )
    partial, _ = master_skycells_csv_paths(str(tmp_path), 1, 1, 1, 1)
    table = pd.read_csv(partial)
    assert int(table.loc[0, "MAPGRID"]) == 3
    assert table.loc[0, "GEOMFP"] == grid.geometry_fingerprint
    assert table.loc[0, "COORDFRM"] == "full_ffi"
