"""ePSF shared_mask loading uses only bright-star (2) and strap (4) bits."""

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


def test_epsf_shared_mask_bits_are_bright_and_strap():
    assert EPSF_SHARED_MASK_BITS == (2 | 4)


def test_load_shared_mask_ignores_catalog_bit(tmp_path):
    """Bit value 1 (catalog Gaia) must not exclude ePSF stars."""
    shape = (8, 8)
    raw = np.zeros(shape, dtype=np.int16)
    raw[0, 0] = 1  # catalog only
    raw[0, 1] = 2  # bright cross
    raw[0, 2] = 4  # strap
    raw[0, 3] = 3  # catalog | bright
    raw[0, 4] = 5  # catalog | strap
    raw[0, 5] = 8  # edge dead zone — ignored for ePSF
    path = tmp_path / "shared_mask.fits"
    fits.writeto(path, raw, overwrite=True)

    mask = _load_shared_mask_2d(str(path), shape)
    assert mask is not None
    assert mask.dtype == bool
    assert bool(mask[0, 0]) is False  # catalog-only ignored
    assert bool(mask[0, 1]) is True
    assert bool(mask[0, 2]) is True
    assert bool(mask[0, 3]) is True  # has bit value 2
    assert bool(mask[0, 4]) is True  # has bit value 4
    assert bool(mask[0, 5]) is False  # edge bit ignored
    assert bool(mask[1, 1]) is False
