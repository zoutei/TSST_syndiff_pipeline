"""TNS paint bit 64 and crop convert."""

import numpy as np
import pandas as pd

from syndiff_pipeline.masking import bits
from syndiff_pipeline.masking.settings import DEFAULT_TNS_PUBLIC_ZIP_URL
from syndiff_pipeline.masking.tns import build_transient_fixed, paint_tns_bit


def test_default_tns_url():
    assert "tns_public_objects" in DEFAULT_TNS_PUBLIC_ZIP_URL


def test_paint_tns_bit_crop_convert():
    mask = np.zeros((40, 40), dtype=np.int16)
    crop = {"x_min": 100, "y_min": 200, "x_max": 140, "y_max": 240}
    # full-FFI 0-based at (110, 210) → crop (10, 10)
    table = pd.DataFrame(
        {
            "source_id": ["SN 2020ut"],
            "x": [110.0],
            "y": [210.0],
            "radius_px": [3],
        }
    )
    out = paint_tns_bit(mask, table, crop)
    assert out[10, 10] & bits.TNS
    assert (out & bits.TNS).sum() > 1


def test_build_transient_fixed_from_tesspoint():
    seeds = pd.DataFrame(
        {
            "source_id": ["AT 2020a"],
            "ra": [10.0],
            "dec": [20.0],
            "mag_tns": [15.0],
            "x_tesspoint_1based": [100.0],
            "y_tesspoint_1based": [200.0],
        }
    )
    # science array check: col 100, row 200 are on-science
    table = build_transient_fixed(seeds, 20, 3, 3)
    assert len(table) == 1
    assert abs(table.iloc[0]["x"] - 99.0) < 1e-6
    assert table.iloc[0]["radius_px"] > 0
