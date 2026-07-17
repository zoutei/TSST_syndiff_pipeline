"""Consumer helpers: Hotpants mask (ignore bit 32) + strap source bits 1|32."""

import numpy as np
import pandas as pd

from syndiff_pipeline.masking import bits
from syndiff_pipeline.masking.bits import strap_source_mask
from syndiff_pipeline.difference_imaging.stages.hotpants import _resolve_hotpants_mask_array
from syndiff_pipeline.masking.catalog import MaskCatalog


def test_hotpants_helper_static_ndarray():
    m = np.zeros((4, 4), dtype=np.int16)
    m[1, 1] = 1
    out = _resolve_hotpants_mask_array(m, None, None)
    assert out.dtype == bool
    assert out[1, 1]


def test_hotpants_helper_ignores_faint_cat_static():
    m = np.zeros((4, 4), dtype=np.int16)
    m[0, 0] = bits.FAINT_CAT
    m[1, 1] = bits.BRIGHT_CAT
    out = _resolve_hotpants_mask_array(m, None, None)
    assert not out[0, 0]
    assert out[1, 1]


def test_hotpants_helper_catalog_full():
    static = np.zeros((5, 5), dtype=np.int16)
    static[0, 0] = bits.BRIGHT_CAT
    static[0, 1] = bits.FAINT_CAT
    cat = MaskCatalog(static=static)
    out = _resolve_hotpants_mask_array(static, cat, None)
    assert out[0, 0]
    assert not out[0, 1]
    assert not out[1, 1]


def test_strap_uses_bright_and_faint():
    m = np.array([1, 32, 2, 4], dtype=np.int16)
    s = strap_source_mask(m)
    assert s.tolist() == [True, True, False, False]
    assert bits.STRAP_SOURCE_BITS == (1 | 32)


def test_resume_loads_scc_asteroids(tmp_path):
    """Hotpants/background resume should attach SCC asteroid intervals from disk."""
    from syndiff_pipeline.difference_imaging.orchestration.execute import (
        _ensure_mask_catalog_loaded,
    )
    from astropy.io import fits

    ws = tmp_path / "ws"
    ws.mkdir()
    static = np.zeros((20, 20), dtype=np.int16)
    static[0, 0] = bits.BRIGHT_CAT
    fits.writeto(ws / "shared_mask.fits.gz", static, overwrite=True)

    data_root = tmp_path / "data"
    scc = data_root / "catalogs" / "sector_0020" / "camera_1" / "ccd_1" / "asteroids"
    scc.mkdir(parents=True)
    iv = pd.DataFrame(
        {
            "target_id": ["1"],
            "row": [10],
            "col": [10],
            "cadence_lo": [0],
            "cadence_hi": [0],
        }
    )
    tm = pd.DataFrame({"cadence": [0], "btjd": [100.0]})
    iv.to_parquet(scc / "pixel_intervals.parquet", index=False)
    tm.to_parquet(scc / "asteroid_ffi_times.parquet", index=False)

    crop = {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20}
    cat = _ensure_mask_catalog_loaded(
        str(ws),
        None,
        None,
        crop_bounds=crop,
        data_root=str(data_root),
        sector=20,
        camera=1,
        ccd=1,
    )
    assert cat.has_temporal()
    m = cat.mask_at(0, which="full")
    assert m[9, 9] & bits.ASTEROID  # 1-based row/col 10 → y=x=9


def test_spatial_accepts_per_frame_mask_cube():
    from syndiff_pipeline.difference_imaging.stages.background.spatial import (
        spatial_step,
    )

    flux = np.ones((3, 8, 8), dtype=np.float64)
    mask = np.zeros((3, 8, 8), dtype=np.int16)
    mask[1, 4, 4] = bits.ASTEROID
    out = spatial_step(flux, mask, box_size=4, n_jobs=1)
    assert out.shape == (3, 8, 8)
