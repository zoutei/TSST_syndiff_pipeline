"""
Simple convolution utilities for TESS PSF application.

The truncated Gaussian pre-blur has two implementations of the same operator,
``scipy.ndimage.gaussian_filter(mode="constant", cval=cval, truncate=radius / sigma)``: the chunked dask-image path
(``method="dask"``, the default, used for every production product) and a real FFT (``method="fft"``, opt-in via
``method=``, ``blur_method("fft")`` or env ``SYNDIFF_BLUR_METHOD=fft``; added 2026-10-05). The FFT is not bit-identical to
dask and the blur method is not part of the convolved-store fingerprint, so do not mix the two within one store.
Both compute:

* kernel: the separable, normalised 1-D Gaussian ``exp(-x^2 / 2 sigma^2)`` on ``|x| <= int(radius + 0.5)``
  (scipy's ``lw = int(truncate * sigma + 0.5)``), applied along both axes;
* boundary: pixels outside the image take the constant ``cval``. With ``cval = 0`` that is a zero-padded linear
  convolution. With ``cval = NaN`` every output pixel whose kernel support crosses the image edge is NaN, i.e. a band
  ``lw`` pixels wide along every edge;
* NaN inputs propagate: every output pixel whose (square, separable) kernel support contains an input NaN is NaN;
* exact zeros: an output pixel whose kernel support holds only zeros is exactly 0, as in direct convolution.

Measured 2026-10-05 on a 7271 x 7238 canvas (sigma 40, radius 470): dask 211 CPU-s, FFT 4.3 CPU-s on one thread,
max |difference| / peak 1.6e-15 on float64 input. For float32 input scipy rounds the intermediate (after the first
axis) to float32, so the two differ by at most a few float32 ulp.
"""

import contextlib
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

BLUR_METHODS = ("fft", "dask")
_METHOD_OVERRIDE: list[str] = []


@contextlib.contextmanager
def blur_method(method: str):
    """Force the default method inside this block (e.g. ``"fft"`` to opt in). An explicit ``method=`` argument
    still wins."""
    if method not in BLUR_METHODS:
        raise ValueError(f"method={method!r}; expected one of {BLUR_METHODS}")
    _METHOD_OVERRIDE.append(method)
    try:
        yield
    finally:
        _METHOD_OVERRIDE.pop()


def _default_method() -> str:
    if _METHOD_OVERRIDE:
        return _METHOD_OVERRIDE[-1]
    m = os.environ.get("SYNDIFF_BLUR_METHOD", "dask").strip().lower()
    if m not in BLUR_METHODS:
        raise ValueError(f"SYNDIFF_BLUR_METHOD={m!r}; expected one of {BLUR_METHODS}")
    return m


def _default_workers() -> int:
    try:
        return max(1, int(os.environ.get("SYNDIFF_FFT_WORKERS", "1")))
    except ValueError:
        return 1


def gaussian_kernel_1d(sigma: float, radius: float) -> np.ndarray:
    """scipy.ndimage's normalised 1-D Gaussian for ``truncate = radius / sigma`` (half-width ``int(radius + 0.5)``)."""
    lw = int(float(radius) / float(sigma) * float(sigma) + 0.5)
    x = np.arange(-lw, lw + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def _box_dilate(mask: np.ndarray, lw: int, *, border: bool) -> np.ndarray:
    """``out[i, j]`` = any ``mask`` within ``|di|, |dj| <= lw`` (outside the image counts as ``border``)."""
    from scipy.ndimage import maximum_filter1d

    m = mask.astype(np.uint8)
    for axis in (0, 1):
        m = maximum_filter1d(m, size=2 * lw + 1, axis=axis, mode="constant", cval=1 if border else 0)
    return m.astype(bool)


def _gaussian_fft(image: np.ndarray, sigma: float, radius: float, cval: float, workers: int) -> np.ndarray:
    import scipy.fft as sfft

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {img.shape}")
    out_dtype = img.dtype if np.issubdtype(img.dtype, np.floating) else np.float64
    k1 = gaussian_kernel_1d(sigma, radius)
    lw = (k1.size - 1) // 2
    h, w = img.shape
    x = np.array(img, dtype=np.float64, copy=True)
    nan = np.isnan(x)
    has_nan = bool(nan.any())
    if has_nan:
        x[nan] = 0.0
    # Direct convolution is exactly 0 wherever the kernel support holds only zeros; the FFT leaves ~1e-16 * peak
    # round-off there. Restore the exact zeros (templates are mostly zeroed background).
    dead = ~_box_dilate(x != 0.0, lw, border=False)
    shape = (sfft.next_fast_len(h + 2 * lw, real=True), sfft.next_fast_len(w + 2 * lw, real=True))
    spec = sfft.rfft2(x, s=shape, workers=workers)
    del x
    ky = sfft.fft(k1, n=shape[0], workers=workers)
    kx = sfft.rfft(k1, n=shape[1], workers=workers)
    spec *= ky[:, None]
    spec *= kx[None, :]
    full = sfft.irfft2(spec, s=shape, workers=workers)
    del spec
    out = np.array(full[lw:lw + h, lw:lw + w], dtype=out_dtype)
    del full
    out[dead] = 0.0
    nan_border = bool(np.isnan(cval))
    if has_nan or nan_border:
        out[_box_dilate(nan, lw, border=nan_border)] = np.nan
    return out


def apply_gaussian_convolution(
    image: np.ndarray,
    sigma: float = 60.0,
    radius: int = 470,
    *,
    cval: float = np.nan,
    method: str | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Apply Gaussian convolution to simulate TESS PSF.

    Args:
        image: Input image array
        sigma: Gaussian sigma parameter
        radius: Kernel truncation radius in pixels (truncate = radius / sigma)
        cval: Constant fill value for ``mode="constant"`` boundary padding.
            Production PS1 mosaics use the default ``np.nan`` for masked gaps;
            isolated star cutouts should pass ``0.0`` so tight cutouts do not
            lose flux to NaN contamination at the edges.
        method: ``"dask"`` (default, or env ``SYNDIFF_BLUR_METHOD``; the chunked dask-image ``gaussian_filter``
            that built every production product) or ``"fft"`` (opt-in, same operator, ~50x less CPU, not bit-identical).
        workers: FFT threads (default env ``SYNDIFF_FFT_WORKERS``, else 1). The dask path ignores it.

    Returns:
        Convolved image array (same dtype as a floating-point input)
    """
    method = _default_method() if method is None else str(method).lower()
    if method == "fft" and not (cval == 0.0 or np.isnan(cval)):
        method = "dask"   # only the 0 / NaN boundaries are implemented in the FFT path
    if method == "fft":
        convolved = _gaussian_fft(image, float(sigma), float(radius), float(cval),
                                  _default_workers() if workers is None else max(1, int(workers)))
    elif method == "dask":
        import dask.array as da
        from dask_image.ndfilters import gaussian_filter as dask_gaussian_filter

        truncate = radius / sigma
        dimage = da.from_array(image, chunks=(1024, 1024))
        convolved = dask_gaussian_filter(
            dimage,
            sigma=sigma,
            mode="constant",
            cval=cval,
            truncate=truncate,
        ).compute()
    else:
        raise ValueError(f"method={method!r}; expected one of {BLUR_METHODS}")
    logger.debug(f"Applied Gaussian convolution (sigma={sigma}, method={method}): {np.shape(image)}")
    return convolved
