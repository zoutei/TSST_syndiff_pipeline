"""Bit layout and consumer predicates."""

from syndiff_pipeline.masking import bits
import numpy as np


def test_bit_constants():
    assert bits.BRIGHT_CAT == 1
    assert bits.SAT_CROSS == 2
    assert bits.STRAP == 4
    assert bits.EDGE == 8
    assert bits.PS1 == 16
    assert bits.FAINT_CAT == 32
    assert bits.TNS == 64
    assert bits.ASTEROID == 128
    assert bits.EPSF_IGNORE_BITS == (1 | 2)
    assert bits.STRAP_SOURCE_BITS == (1 | 32)


def test_epsf_reject_mask():
    m = np.array([[1, 2, 3, 4, 8, 16, 32, 64, 128, 0]], dtype=np.int16)
    r = bits.epsf_reject_mask(m)
    assert r.tolist() == [[False, False, False, True, True, True, True, True, True, False]]


def test_full_mask_bool():
    m = np.array([[0, 1, 32]], dtype=np.int16)
    assert bits.full_mask_bool(m).tolist() == [[False, True, True]]


def test_strap_source_bits():
    m = np.array([1, 32, 33, 2, 4], dtype=np.int16)
    s = bits.strap_source_mask(m)
    assert s.tolist() == [True, True, True, False, False]
