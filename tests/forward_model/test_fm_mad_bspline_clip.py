# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for iterative MAD B-spline clipping."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.forward_model.init_study.mad_bspline_clip import (
    fit_coeff_tracks_mad,
    iterative_mad_bspline_fit,
    mad_scale,
)
from syndiff_pipeline.forward_model import temporal as T


def test_mad_scale_gaussian():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 1.0, 5000)
    med, mad = mad_scale(r)
    assert abs(med) < 0.05
    assert 0.9 < mad < 1.1


def test_iterative_mad_rejects_spike_and_matches_clean_fit():
    btjd = np.linspace(0.0, 1.0, 120)
    tb = T.build_temporal_basis(btjd, n_interior=6, uniform_knots=True)
    phi = np.asarray(tb.frame_basis, dtype=float)
    # Smooth true curve in the span of the basis.
    true_c = np.zeros(phi.shape[1])
    true_c[0] = 1.0
    true_c[1] = -0.3
    true_c[2] = 0.15
    y = phi @ true_c
    y_spike = y.copy()
    y_spike[10] += 5.0
    y_spike[50] -= 5.0
    y_spike[90] += 4.0

    fit = iterative_mad_bspline_fit(phi, y_spike, k=3.0, max_iters=5)
    assert fit.keep.sum() <= len(y) - 3
    assert not fit.keep[10]
    assert not fit.keep[50]
    assert not fit.keep[90]
    # Recovered curve close to truth on clean points.
    assert np.median(np.abs(fit.prediction[fit.keep] - y[fit.keep])) < 1e-3


def test_fit_coeff_tracks_mad_shape():
    btjd = np.linspace(0.0, 1.0, 80)
    tb = T.build_temporal_basis(btjd, n_interior=4, uniform_knots=True)
    phi = np.asarray(tb.frame_basis, dtype=float)
    tracks = np.column_stack([phi @ np.ones(phi.shape[1]), phi[:, 0]])
    tracks[5, 0] += 10.0
    coeff, keep, results = fit_coeff_tracks_mad(phi, tracks, k=3.0, max_iters=4)
    assert coeff.shape == (2, phi.shape[1])
    assert keep.shape == tracks.shape
    assert len(results) == 2
    assert not keep[5, 0]
