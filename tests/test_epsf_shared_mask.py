"""ePSF static-mask policy: ignore all star bits 1|2|32; reject 4/8/16/64/128."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages.epsf import (
    EPSF_SHARED_MASK_BITS,
    _load_static_mask_2d,
    epsf_reject_mask_at,
)
from syndiff_pipeline.masking.bits import EPSF_IGNORE_BITS
from syndiff_pipeline.masking.catalog import MaskCatalog


def test_epsf_ignore_bits_are_all_stars():
    assert EPSF_SHARED_MASK_BITS == EPSF_IGNORE_BITS == (1 | 2 | 32)


def test_load_static_mask_epsf_policy(tmp_path):
    shape = (8, 8)
    raw = np.zeros(shape, dtype=np.int16)
    raw[0, 0] = 1  # bright cat — keep
    raw[0, 1] = 2  # sat cross alone — keep
    raw[0, 2] = 3  # 1|2 — keep
    raw[0, 3] = 4  # strap — reject
    raw[0, 4] = 8  # edge — reject
    raw[0, 5] = 16  # PS1 — reject
    raw[0, 6] = 32  # faint star square — keep
    raw[0, 7] = 64  # TNS — reject
    raw[1, 0] = 128  # asteroid — reject
    raw[1, 1] = 33  # 1|32 — keep
    path = tmp_path / "shared_mask.fits"
    fits.writeto(path, raw, overwrite=True)

    mask = _load_static_mask_2d(str(path), shape)
    assert mask is not None
    assert bool(mask[0, 0]) is False
    assert bool(mask[0, 1]) is False
    assert bool(mask[0, 2]) is False
    assert bool(mask[0, 3]) is True
    assert bool(mask[0, 4]) is True
    assert bool(mask[0, 5]) is True
    assert bool(mask[0, 6]) is False
    assert bool(mask[0, 7]) is True
    assert bool(mask[1, 0]) is True
    assert bool(mask[1, 1]) is False


def test_epsf_reject_mask_at_includes_asteroid_temporal():
    import pandas as pd

    static = np.zeros((5, 5), dtype=np.int16)
    static[2, 2] = 32  # faint star — ignored for ePSF
    intervals = pd.DataFrame(
        {
            "y": [1],
            "x": [1],
            "cadence_lo": [0],
            "cadence_hi": [0],
        }
    )
    times = pd.DataFrame({"cadence": [0], "btjd": [100.0]})
    cat = MaskCatalog(
        static=static,
        asteroid_intervals=intervals,
        asteroid_times=times,
        crop_bounds={"x_min": 0, "y_min": 0, "shape": (5, 5)},
    )

    reject = epsf_reject_mask_at(cat, 100.0)
    assert bool(reject[2, 2]) is False  # faint star kept
    assert bool(reject[1, 1]) is True  # asteroid rejected
