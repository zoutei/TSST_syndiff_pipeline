"""Per-band template kernels distilled from an ePSF, and the node-blended band convolution.

Given, at every ePSF node n, one ePSF grid per PS1 band (``E[b, n]``, pixel-integrated,
oversampled by ``os`` = 4, centre sample at the grid middle), the kernel ``K[b, n]`` is the
OS-subcell kernel such that

    blocksum( G (*) box (*) K[b, n] )  ~=  E[b, n] sampled at native stride,

where G is the template's own Gaussian pre-blur (covariance ``sigma_G`` in TESS px^2) and box
the native pixel. It is solved by Wiener-regularised Fourier division (the Phase A
``distill_fourier`` method, 2026-09-24):

    K^ = E^ tri^ H* / (|H|^2 + eps),   H = G^ box^,  tri^ = sinc^2(f/os).

With ``normalize="dc"`` the kernel is scaled by the constant ``1 + eps`` instead of by its own
sum. That keeps the map E -> K LINEAR, so kernels distilled from ``P_0 + d_b P_1`` equal
``K(P_0) + d_b K(P_1)`` and the per-band convolution equals the moment-image convolution
exactly. For a unit-sum E the two normalisations agree.

The band model image is then

    M = sum_b sum_n (w_n T_b) (*) K[b, n],

with w_n the bilinear node weights at the SOURCE subcell (flat extrapolation outside the node
grid, exactly like the forward fitter's ``bilinear_cell``). Flux is conserved because
sum_n w_n = 1 and every kernel sums to 1.
"""
from __future__ import annotations

import time
from typing import Mapping, Optional

import numpy as np


def fourier_kernel(
    E: np.ndarray,
    sigma_G: np.ndarray,
    eps: float,
    *,
    n_kernel: int = 63,
    with_tri: bool = True,
    os: int = 4,
    npad: int = 256,
    normalize: str = "dc",
) -> np.ndarray:
    """One node's OS-subcell kernel from its ePSF grid ``E`` (odd size, centre = middle)."""
    if E.shape[0] != E.shape[1] or E.shape[0] % 2 != 1:
        raise ValueError(f"E must be square with odd size, got {E.shape}")
    if n_kernel % 2 != 1:
        raise ValueError("n_kernel must be odd")
    ce = (E.shape[0] - 1) // 2
    Ep = np.zeros((npad, npad))
    Ep[:E.shape[0], :E.shape[1]] = E
    Ep = np.roll(Ep, (-ce, -ce), axis=(0, 1))
    Ef = np.fft.fft2(Ep)
    f = np.fft.fftfreq(npad, d=1.0 / os)
    FX, FY = np.meshgrid(f, f)
    S = np.asarray(sigma_G, dtype=np.float64)
    G = np.exp(-2 * np.pi ** 2 * (S[0, 0] * FX ** 2 + 2 * S[0, 1] * FX * FY + S[1, 1] * FY ** 2))
    H = G * np.sinc(FX) * np.sinc(FY)
    tri = (np.sinc(FX / os) * np.sinc(FY / os)) ** 2 if with_tri else 1.0
    Kf = Ef * tri * np.conj(H) / (np.abs(H) ** 2 + eps)
    k = np.real(np.fft.ifft2(Kf))
    k = np.roll(k, ((n_kernel - 1) // 2, (n_kernel - 1) // 2), axis=(0, 1))[:n_kernel, :n_kernel]
    if normalize == "dc":
        return k * (1.0 + eps)
    if normalize == "sum":
        return k / k.sum()
    if normalize == "none":
        return k
    raise ValueError(f"unknown normalize={normalize!r}")


def band_kernels(
    E_bands: np.ndarray,
    sigma_G: np.ndarray,
    eps: np.ndarray,
    with_tri: np.ndarray,
    *,
    n_kernel: int = 63,
    os: int = 4,
    normalize: str = "dc",
) -> np.ndarray:
    """Kernels for every band and node. ``E_bands``: (B, nr, nc, g, g); returns (B, nr, nc, N, N).

    ``eps``/``with_tri`` are per node (nr, nc) and shared by all bands, so the map stays linear.
    """
    B, nr, nc = E_bands.shape[:3]
    K = np.zeros((B, nr, nc, n_kernel, n_kernel))
    for b in range(B):
        for i in range(nr):
            for j in range(nc):
                K[b, i, j] = fourier_kernel(E_bands[b, i, j], sigma_G[i, j], float(eps[i, j]),
                                            n_kernel=n_kernel, with_tri=bool(with_tri[i, j]),
                                            os=os, normalize=normalize)
    return K


# ------------------------------------------------------------------ geometry
def subcell_centres_sci(grid, axis: str) -> np.ndarray:
    """Science-local native coordinate of every OS subcell centre along ``axis``.

    MappingGrid convention: ffi = ffi_min + (I + 0.5)/F - 0.5; science-local = ffi - science_min.
    """
    F = int(grid.oversampling)
    if axis == "x":
        n, fmin, smin = grid.width_os, grid.ffi_xmin, grid.science_xmin_ffi
    elif axis == "y":
        n, fmin, smin = grid.height_os, grid.ffi_ymin, grid.science_ymin_ffi
    else:
        raise ValueError(axis)
    I = np.arange(n, dtype=np.float64)
    return (fmin - smin) + (I + 0.5) / F - 0.5


def hat_weights(coord: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """(n_nodes, len(coord)) piecewise-linear node weights with flat extrapolation.

    Replicates the forward fitter's ``bilinear_cell`` (searchsorted side='right', cell index
    clipped to [0, n-2], weight clipped to [0, 1]).
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64)
    n = nodes.size
    j0 = np.clip(np.searchsorted(nodes, coord, side="right") - 1, 0, n - 2)
    w = np.clip((coord - nodes[j0]) / (nodes[j0 + 1] - nodes[j0]), 0.0, 1.0)
    H = np.zeros((n, coord.size), dtype=np.float64)
    idx = np.arange(coord.size)
    H[j0, idx] += 1.0 - w
    H[j0 + 1, idx] += w
    return H


# ------------------------------------------------------------------ convolution
def convolve_node_blended(
    T: np.ndarray,
    K: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    *,
    workers: int = 4,
) -> np.ndarray:
    """sum_n (w_n T) (*) K_n, w_{ij}(X, Y) = hy_i(Y) hx_j(X); float64, same shape as T.

    ``hx``: (nc, W) and ``hy``: (nr, H) node weights at the source subcell centres.
    True convolution (``fftconvolve`` semantics, odd kernel, centre = zero offset); light that
    would land outside T is dropped.
    """
    import scipy.fft
    from scipy.signal import fftconvolve

    T = np.asarray(T)
    nr, nc, N, _ = K.shape
    m = (N - 1) // 2
    H, W = T.shape
    if hx.shape != (nc, W) or hy.shape != (nr, H):
        raise ValueError(f"weights {hx.shape}/{hy.shape} do not match K {K.shape} and T {T.shape}")
    out = np.zeros((H, W), dtype=np.float64)
    with scipy.fft.set_workers(workers):
        for i in range(nr):
            rows = np.nonzero(hy[i])[0]
            if rows.size == 0:
                continue
            r0, r1 = rows[0], rows[-1] + 1
            for j in range(nc):
                cols = np.nonzero(hx[j])[0]
                if cols.size == 0:
                    continue
                c0, c1 = cols[0], cols[-1] + 1
                A = np.nan_to_num(T[r0:r1, c0:c1].astype(np.float64)) * hy[i, r0:r1, None] * hx[j, None, c0:c1]
                if not A.any():
                    continue
                C = fftconvolve(A, K[i, j], mode="full")
                orow0, ocol0 = r0 - m, c0 - m
                pr0, pc0 = max(0, -orow0), max(0, -ocol0)
                pr1 = C.shape[0] - max(0, orow0 + C.shape[0] - H)
                pc1 = C.shape[1] - max(0, ocol0 + C.shape[1] - W)
                out[orow0 + pr0:orow0 + pr1, ocol0 + pc0:ocol0 + pc1] += C[pr0:pr1, pc0:pc1]
    return out


def convolve_bands(
    T_bands: Mapping[str, np.ndarray],
    K_bands: Mapping[str, np.ndarray],
    hx: np.ndarray,
    hy: np.ndarray,
    *,
    workers: int = 4,
    return_per_band: bool = False,
    verbose: bool = False,
):
    """M = sum_b sum_n (w_n T_b) (*) K_b[n]. Returns M (and ``{band: M_b}`` if asked)."""
    total: Optional[np.ndarray] = None
    per: dict[str, np.ndarray] = {}
    for band, T in T_bands.items():
        t0 = time.time()
        Mb = convolve_node_blended(T, K_bands[band], hx, hy, workers=workers)
        if verbose:
            print(f"  band {band}: {time.time() - t0:.1f}s", flush=True)
        total = Mb.copy() if total is None else total + Mb
        if return_per_band:
            per[band] = Mb
    if return_per_band:
        return total, per
    return total
