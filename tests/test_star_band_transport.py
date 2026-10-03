"""Per-band templates: exact band split, moment identity, linear kernel distillation."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from syndiff_pipeline.star.band_transport import Snapshot, sparse_gaussian
from syndiff_pipeline.template_creation.processing import perband as PB
from syndiff_pipeline.template_creation.processing.band_utils import combine_rizy_bands


def _header(boffset, bsoften, exptime):
    from astropy.io import fits

    h = fits.Header()
    h["BOFFSET"], h["BSOFTEN"], h["EXPTIME"] = boffset, bsoften, exptime
    return h.tostring()


@pytest.fixture
def raw_bands():
    rng = np.random.default_rng(1)
    bands = {b: rng.normal(5.0, 3.0, (40, 50)).astype(np.float32) for b in PB.BANDS}
    bands["z"][3, 4] = np.nan
    headers = {
        b: _header(100.0 + k, 50.0 + 3 * k, 30.0 + k) for k, b in enumerate(PB.BANDS)
    }
    return bands, headers


def test_weighted_bands_sum_bitwise_equals_combiner(raw_bands):
    bands, headers = raw_bands
    combined, _ = combine_rizy_bands(bands, headers_data=headers)
    wb = PB.weighted_band_images(bands, headers)
    s = PB.sum_bands(wb)
    both_nan = np.isnan(s) & np.isnan(combined)
    assert np.array_equal(np.isnan(s), np.isnan(combined))
    assert np.array_equal(s[~both_nan], combined[~both_nan])


def test_split_like_combined_reproduces_zeroing_exactly(raw_bands):
    bands, headers = raw_bands
    wb = PB.weighted_band_images(bands, headers)
    before = PB.sum_bands(wb)
    after = before.copy()
    after[10:20, 5:30] = 0  # a removed segment
    after[0:3, :] = 0  # outside the segments
    split = PB.split_like_combined(wb, after)
    s = PB.sum_bands(split)
    fin = np.isfinite(after)
    assert np.array_equal(np.isfinite(s), fin)
    assert np.array_equal(s[fin] == 0, after[fin] == 0)
    r = PB.split_residual(split, after)
    assert r["max_rel_to_peak"] < 1e-6 and r["n_nan_mismatch"] == 0
    for b in PB.BANDS:
        assert np.all(split[b][10:20, 5:30] == 0)


def test_sparse_blur_matches_full_filter_at_edges():
    rng = np.random.default_rng(16)
    for dtype in (np.float32, np.float64):
        a = np.zeros((173, 189), dtype=dtype)
        a[1:8, :5] = rng.normal(size=(7, 5))
        a[80:86, 91:97] = rng.normal(size=(6, 6))
        a[-1, -1] = 3
        for cval in (0.0, np.nan):
            expected = gaussian_filter(a, 3.0, radius=17, mode="constant", cval=cval)
            np.testing.assert_array_equal(
                sparse_gaussian(a, 3.0, 17, cval=cval), expected
            )


def test_unknown_operator_generation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported template operator"):
        Snapshot({"operator_version": "unverified_next_generation"})
