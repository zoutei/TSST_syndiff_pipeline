"""ePSF shared_mask: ignore bits 1|2; reject 4/8/16/32/64/128; bit2 alone → keep."""

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
    _load_shared_mask_2d,
)
from syndiff_pipeline.masking.bits import EPSF_IGNORE_BITS


def test_epsf_ignore_bits_are_bright_and_cross():
    assert EPSF_SHARED_MASK_BITS == EPSF_IGNORE_BITS == (1 | 2)


def test_load_shared_mask_epsf_policy(tmp_path):
    shape = (8, 8)
    raw = np.zeros(shape, dtype=np.int16)
    raw[0, 0] = 1  # bright cat — keep
    raw[0, 1] = 2  # sat cross alone — keep
    raw[0, 2] = 3  # 1|2 — keep
    raw[0, 3] = 4  # strap — reject
    raw[0, 4] = 8  # edge — reject
    raw[0, 5] = 16  # PS1 — reject
    raw[0, 6] = 32  # faint — reject
    raw[0, 7] = 64  # TNS — reject
    raw[1, 0] = 128  # asteroid — reject
    path = tmp_path / "shared_mask.fits"
    fits.writeto(path, raw, overwrite=True)

    mask = _load_shared_mask_2d(str(path), shape)
    assert mask is not None
    assert bool(mask[0, 0]) is False
    assert bool(mask[0, 1]) is False
    assert bool(mask[0, 2]) is False
    assert bool(mask[0, 3]) is True
    assert bool(mask[0, 4]) is True
    assert bool(mask[0, 5]) is True
    assert bool(mask[0, 6]) is True
    assert bool(mask[0, 7]) is True
    assert bool(mask[1, 0]) is True
