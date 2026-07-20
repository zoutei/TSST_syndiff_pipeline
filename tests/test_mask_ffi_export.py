"""FFI id → mask FITS helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.difference_imaging.masking.catalog import MaskCatalog
from syndiff_pipeline.difference_imaging.masking.ffi_mask import (
    normalize_ffi_product_id,
    select_begin_mid_end_ffi_ids,
    write_mask_fits_for_ffi,
    write_sector_sample_mask_fits,
)


def test_normalize_ffi_product_id():
    assert normalize_ffi_product_id("tess2020019142923") == "tess2020019142923"
    assert normalize_ffi_product_id(2020019142923) == "tess2020019142923"
    assert (
        normalize_ffi_product_id("tess2020019142923-s0020-3-3-0165-s_ffic.fits")
        == "tess2020019142923"
    )


def test_write_mask_fits_for_ffi_by_cadence(tmp_path):
    static = np.zeros((8, 8), dtype=np.int16)
    static[1, 1] = 1
    times = pd.DataFrame({"cadence": [0, 1, 2], "btjd": [10.0, 10.02, 10.04]})
    iv = pd.DataFrame(
        {
            "target_id": ["a"],
            "row": [3],
            "col": [3],
            "cadence_lo": [1],
            "cadence_hi": [1],
        }
    )
    crop = {"x_min": 0, "y_min": 0, "x_max": 8, "y_max": 8}
    cat = MaskCatalog.from_arrays(
        static, asteroid_intervals_ffi=iv, asteroid_times=times, crop_bounds=crop
    )
    path = write_mask_fits_for_ffi(cat, 1, tmp_path / "m.fits")
    data = fits.getdata(path)
    assert data[1, 1] == 1
    assert data[2, 2] & 128  # row3-1=2, col3-1=2


def test_select_begin_mid_end_and_write(tmp_path):
    static = np.zeros((6, 6), dtype=np.int16)
    cat = MaskCatalog(static=static)
    wcs = pd.DataFrame(
        {
            "filename": [
                "tess1000000000001-s_ffic.fits",
                "tess1000000000002-s_ffic.fits",
                "tess1000000000003-s_ffic.fits",
                "tess1000000000004-s_ffic.fits",
                "tess1000000000005-s_ffic.fits",
            ],
            "btjd": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    begin, mid, end = select_begin_mid_end_ffi_ids(wcs)
    assert begin == "tess1000000000001"
    assert mid == "tess1000000000003"
    assert end == "tess1000000000005"
    paths = write_sector_sample_mask_fits(cat, wcs, tmp_path / "out")
    assert len(paths) == 3
    assert all(p.is_file() for p in paths)
    assert "begin" in paths[0].name and "end" in paths[2].name
