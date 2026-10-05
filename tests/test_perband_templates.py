"""Per-band templates: exact band split, moment identity, linear kernel distillation."""
import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import chromatic_kernels as CK
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
    headers = {b: _header(100.0 + k, 50.0 + 3 * k, 30.0 + k) for k, b in enumerate(PB.BANDS)}
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
    after[10:20, 5:30] = 0          # a removed segment
    after[0:3, :] = 0               # outside the segments
    split = PB.split_like_combined(wb, after)
    s = PB.sum_bands(split)
    fin = np.isfinite(after)
    assert np.array_equal(np.isfinite(s), fin)
    assert np.array_equal(s[fin] == 0, after[fin] == 0)
    r = PB.split_residual(split, after)
    assert r["max_rel_to_peak"] < 1e-6 and r["n_nan_mismatch"] == 0
    for b in PB.BANDS:
        assert np.all(split[b][10:20, 5:30] == 0)


def test_moment_identity_equals_band_convolution():
    """sum_b K_b (*) (w_b F_b) == sum_k Kt_k (*) M_k when K_b = sum_k d_b^k Kt_k (k <= 3 exact)."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(0)
    wb = {b: rng.random((32, 32)) for b in PB.BANDS}
    d = np.array([(PB.PS1_LAMBDA_NM[b] - 800.0) / 100.0 for b in PB.BANDS])
    V = np.vander(d, 4, increasing=True)
    Kt = rng.normal(size=(4, 7, 7))
    K = np.einsum("bk,kij->bij", V, Kt)
    direct = sum(fftconvolve(wb[b], K[n], mode="same") for n, b in enumerate(PB.BANDS))
    M = PB.moment_images(wb, order=3)
    assert np.allclose(M[0], sum(wb.values()))
    via = sum(fftconvolve(M[k], Kt[k], mode="same") for k in range(4))
    assert np.abs(via - direct).max() < 1e-12 * np.abs(direct).max()


def _gauss_grid(g=63, os=4, sx=1.0, sy=1.1, x0=0.0):
    ax = (np.arange(g) - (g - 1) // 2) / os
    X, Y = np.meshgrid(ax, ax)
    E = np.exp(-0.5 * (((X - x0) / sx) ** 2 + (Y / sy) ** 2))
    return E / E.sum()


def test_fourier_kernel_is_linear_and_unit_sum():
    S = np.array([[0.24, 0.0], [0.0, 0.26]])
    P0 = _gauss_grid()
    P1 = _gauss_grid(x0=0.05) - P0          # zero-sum chromatic derivative
    eps = 3e-5
    K0 = CK.fourier_kernel(P0, S, eps)
    K1 = CK.fourier_kernel(P1, S, eps)
    for db in (-6.5, -1.4, 2.8, 6.4):
        Kb = CK.fourier_kernel(P0 + db * P1, S, eps)
        assert np.abs(Kb - (K0 + db * K1)).max() < 1e-12
        assert abs(Kb.sum() - 1.0) < 1e-6


def test_hat_weights_partition_of_unity():
    nodes = np.linspace(0, 2048, 6)
    c = np.linspace(-50, 2100, 997)
    H = CK.hat_weights(c, nodes)
    assert np.allclose(H.sum(0), 1.0)
    assert np.all(H >= 0)


def test_convolve_node_blended_conserves_flux():
    rng = np.random.default_rng(3)
    T = np.zeros((120, 140))
    T[30:90, 30:110] = rng.random((60, 80))
    nr, nc, N = 3, 4, 9
    K = rng.random((nr, nc, N, N))
    K /= K.sum(axis=(2, 3), keepdims=True)
    hx = CK.hat_weights(np.arange(140.0), np.linspace(0, 139, nc))
    hy = CK.hat_weights(np.arange(120.0), np.linspace(0, 119, nr))
    out = CK.convolve_node_blended(T, K, hx, hy, workers=1)
    assert abs(out.sum() - T.sum()) < 1e-9 * T.sum()
    # per-band sum == sum of per-band convolutions
    Tb = {"r": 0.3 * T, "i": 0.7 * T}
    Kb = {"r": K, "i": K}
    tot = CK.convolve_bands(Tb, Kb, hx, hy, workers=1)
    assert np.allclose(tot, out)
