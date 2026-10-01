# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Iterative MAD σ-clip B-spline fits on a fixed temporal design matrix.

Used to robustify per-FFI Chebyshev coefficient tracks against outlier FFIs
while keeping the same knot / basis as ``TemporalBasis.frame_basis``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MadBSplineFit:
    """Result of an iterative MAD-clipped least-squares B-spline fit."""

    coeff: np.ndarray  # (n_basis,)
    keep: np.ndarray  # (n_frames,) bool
    n_iters: int
    mad: float
    median_resid: float
    prediction: np.ndarray  # (n_frames,) full prediction on all rows


def mad_scale(residuals: np.ndarray) -> tuple[float, float]:
    """Return (median, 1.4826 * MAD) for finite residuals."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(r))
    mad = float(1.4826 * np.median(np.abs(r - med)))
    return med, mad


def iterative_mad_bspline_fit(
    phi: np.ndarray,
    y: np.ndarray,
    *,
    k: float = 3.0,
    max_iters: int = 5,
    min_keep_frac: float = 0.4,
) -> MadBSplineFit:
    """Fit ``y ≈ phi @ coeff`` with iterative MAD residual clipping.

    Parameters
    ----------
    phi
        Design matrix ``(n_frames, n_basis)`` — typically ``wcs_tb.frame_basis``.
    y
        Per-FFI coefficient track ``(n_frames,)``.
    k
        MAD threshold (typical 3–5).
    max_iters
        Maximum fit → clip → refit cycles (final fit always on the last mask).
    min_keep_frac
        Abort further clipping if fewer than this fraction of finite points remain.
    """
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D, got shape {phi.shape}")
    if y.ndim != 1 or y.shape[0] != phi.shape[0]:
        raise ValueError(f"y shape {y.shape} incompatible with phi {phi.shape}")

    finite = np.isfinite(y) & np.isfinite(phi).all(axis=1)
    keep = finite.copy()
    n_finite = int(finite.sum())
    if n_finite < phi.shape[1]:
        raise ValueError(
            f"need at least n_basis={phi.shape[1]} finite points, got {n_finite}"
        )

    coeff = np.zeros(phi.shape[1], dtype=float)
    med = float("nan")
    mad = float("nan")
    n_iters = 0

    for it in range(int(max_iters)):
        n_iters = it + 1
        coeff, *_ = np.linalg.lstsq(phi[keep], y[keep], rcond=None)
        pred = phi @ coeff
        resid = y - pred
        med, mad = mad_scale(resid[keep])
        if not np.isfinite(mad) or mad <= 0.0:
            break
        new_keep = finite & (np.abs(resid - med) <= float(k) * mad)
        if int(new_keep.sum()) < max(phi.shape[1], int(np.ceil(min_keep_frac * n_finite))):
            # Too aggressive — keep previous mask and stop.
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    # Final fit on converged (or last accepted) mask.
    coeff, *_ = np.linalg.lstsq(phi[keep], y[keep], rcond=None)
    pred = phi @ coeff
    resid = y - pred
    med, mad = mad_scale(resid[keep])
    return MadBSplineFit(
        coeff=np.asarray(coeff, dtype=float),
        keep=keep,
        n_iters=n_iters,
        mad=float(mad),
        median_resid=float(med),
        prediction=np.asarray(pred, dtype=float),
    )


def fit_coeff_tracks_mad(
    phi: np.ndarray,
    tracks: np.ndarray,
    *,
    k: float = 3.0,
    max_iters: int = 5,
    min_keep_frac: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, list[MadBSplineFit]]:
    """MAD-clip fit each column of ``tracks`` (n_frames, n_series).

    Returns
    -------
    coeff_matrix
        ``(n_series, n_basis)`` — same layout as ``wcs_coeff`` rows.
    keep_matrix
        ``(n_frames, n_series)`` bool masks.
    results
        Per-column ``MadBSplineFit`` list.
    """
    phi = np.asarray(phi, dtype=float)
    tracks = np.asarray(tracks, dtype=float)
    if tracks.ndim != 2 or tracks.shape[0] != phi.shape[0]:
        raise ValueError(f"tracks shape {tracks.shape} incompatible with phi {phi.shape}")

    n_frames, n_series = tracks.shape
    n_basis = phi.shape[1]
    coeff_matrix = np.zeros((n_series, n_basis), dtype=float)
    keep_matrix = np.ones((n_frames, n_series), dtype=bool)
    results: list[MadBSplineFit] = []
    for j in range(n_series):
        fit = iterative_mad_bspline_fit(
            phi,
            tracks[:, j],
            k=k,
            max_iters=max_iters,
            min_keep_frac=min_keep_frac,
        )
        coeff_matrix[j] = fit.coeff
        keep_matrix[:, j] = fit.keep
        results.append(fit)
    return coeff_matrix, keep_matrix, results
