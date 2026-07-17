"""MaskCatalog.mask_at cadence / btjd contract."""

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.masking import bits
from syndiff_pipeline.difference_imaging.masking.asteroids import convert_intervals_to_crop_local
from syndiff_pipeline.difference_imaging.masking.catalog import MaskCatalog


def test_mask_at_static_full_temporal():
    static = np.zeros((10, 10), dtype=np.int16)
    static[2, 3] = bits.BRIGHT_CAT
    times = pd.DataFrame({"cadence": [0, 1, 2], "btjd": [100.0, 100.02, 100.04]})
    # FFI 1-based row/col → crop at (0,0)
    iv = pd.DataFrame(
        {
            "target_id": ["a"],
            "row": [5],
            "col": [6],
            "cadence_lo": [1],
            "cadence_hi": [1],
        }
    )
    crop = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
    cat = MaskCatalog.from_arrays(
        static,
        asteroid_intervals_ffi=iv,
        asteroid_times=times,
        crop_bounds=crop,
    )
    # crop convert once: row5,col6 → y=4,x=5
    assert len(cat.asteroid_intervals) == 1
    assert int(cat.asteroid_intervals.iloc[0]["y"]) == 4
    assert int(cat.asteroid_intervals.iloc[0]["x"]) == 5

    s = cat.mask_at(which="static")
    assert s[2, 3] == bits.BRIGHT_CAT
    assert s[4, 5] == 0

    t = cat.mask_at(1, which="temporal")
    assert t[4, 5] == bits.ASTEROID
    assert t[2, 3] == 0

    f = cat.mask_at(1, which="full")
    assert f[2, 3] == bits.BRIGHT_CAT
    assert f[4, 5] & bits.ASTEROID

    # btjd → cadence 1
    f2 = cat.mask_at(100.02, which="full")
    assert f2[4, 5] & bits.ASTEROID

    out = np.zeros((10, 10), dtype=bool)
    cat.mask_at(1, which="full", out=out, as_bool=True)
    assert out[4, 5] and out[2, 3]


def test_convert_intervals_oob_dropped():
    crop = {"x_min": 100, "y_min": 100, "x_max": 110, "y_max": 110}
    iv = pd.DataFrame(
        {
            "target_id": ["a", "b"],
            "row": [105, 5],  # second OOB
            "col": [105, 5],
            "cadence_lo": [0, 0],
            "cadence_hi": [0, 0],
        }
    )
    out = convert_intervals_to_crop_local(iv, crop, (10, 10))
    assert len(out) == 1
    assert int(out.iloc[0]["y"]) == 4
    assert int(out.iloc[0]["x"]) == 4
