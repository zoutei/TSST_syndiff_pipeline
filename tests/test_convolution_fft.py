"""FFT Gaussian blur == the dask-image gaussian_filter path (convolution_utils.apply_gaussian_convolution)."""
import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import convolution_utils as CU


def _img(shape, dtype, seed=0, nan_frac=0.0, edge_nan=False):
    rng = np.random.default_rng(seed)
    x = np.zeros(shape)
    m = rng.random(shape) < 0.05
    x[m] = rng.exponential(5.0, m.sum())
    x += rng.normal(0, 0.01, shape)
    if nan_frac:
        x[rng.random(shape) < nan_frac] = np.nan
    if edge_nan:
        x[: shape[0] // 5, :] = np.nan
    return x.astype(dtype)


def _compare(img, sigma, radius, cval, rtol):
    a = CU.apply_gaussian_convolution(img, sigma=sigma, radius=radius, cval=cval, method="dask")
    b = CU.apply_gaussian_convolution(img, sigma=sigma, radius=radius, cval=cval, method="fft")
    assert a.dtype == b.dtype == (img.dtype if np.issubdtype(img.dtype, np.floating) else np.float64)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN pattern differs"
    fin = np.isfinite(a)
    if fin.any():
        pk = np.abs(a[fin]).max()
        assert np.abs(a[fin].astype(float) - b[fin].astype(float)).max() <= rtol * pk


@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-12), (np.float32, 1e-6)])
@pytest.mark.parametrize("cval", [np.nan, 0.0])
def test_fft_matches_dask_clean(dtype, rtol, cval):
    _compare(_img((300, 283), dtype), 4.0, 47, cval, rtol)


@pytest.mark.parametrize("cval", [np.nan, 0.0])
def test_fft_matches_dask_with_interior_nans(cval):
    _compare(_img((257, 311), np.float64, nan_frac=2e-4, edge_nan=True), 4.0, 47, cval, 1e-12)


@pytest.mark.parametrize("cval", [np.nan, 0.0])
def test_fft_matches_scipy_image_smaller_than_kernel(cval):
    """dask-image cannot overlap-chunk an image smaller than the kernel; compare with scipy.ndimage directly."""
    from scipy.ndimage import gaussian_filter
    img = _img((60, 41), np.float64)
    a = gaussian_filter(img, 4.0, mode="constant", cval=cval, truncate=47 / 4.0)
    b = CU.apply_gaussian_convolution(img, sigma=4.0, radius=47, cval=cval, method="fft")
    assert np.array_equal(np.isnan(a), np.isnan(b))
    fin = np.isfinite(a)
    if fin.any():
        assert np.abs(a[fin] - b[fin]).max() <= 1e-12 * np.abs(a[fin]).max()


def test_fft_production_ratio_sigma40_radius470():
    """Production kernel (sigma 40, radius 470) on a canvas just larger than the support."""
    _compare(_img((1100, 1050), np.float32, seed=3), 40.0, 470, np.nan, 1e-6)


def test_kernel_matches_scipy_halfwidth():
    k = CU.gaussian_kernel_1d(40.0, 470)
    assert k.size == 2 * 470 + 1 and abs(k.sum() - 1) < 1e-15


def test_env_selects_method(monkeypatch):
    monkeypatch.setenv("SYNDIFF_BLUR_METHOD", "bogus")
    with pytest.raises(ValueError):
        CU.apply_gaussian_convolution(np.zeros((10, 10)), sigma=1.0, radius=3)


def test_fft_exact_zero_outside_support():
    img = np.zeros((200, 220)); img[20:25, 30:33] = 7.0
    out = CU.apply_gaussian_convolution(img, sigma=2.0, radius=8, cval=0.0, method="fft")
    assert (out[40:, :] == 0).all() and (out[:, 50:] == 0).all()
    assert out[22, 31] > 0


def test_default_method_is_dask(monkeypatch):
    """Production default is the dask path (decision D2, 2026-10-07); FFT is opt-in."""
    monkeypatch.delenv("SYNDIFF_BLUR_METHOD", raising=False)
    assert CU._default_method() == "dask"
    with CU.blur_method("fft"):
        assert CU._default_method() == "fft"
    monkeypatch.setenv("SYNDIFF_BLUR_METHOD", "fft")
    assert CU._default_method() == "fft"
