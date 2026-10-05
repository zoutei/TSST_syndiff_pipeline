# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Unit tests for Hotpants-style stamp QA gates."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.forward_model.init_study.stamp_qa import (
    phase1_norm_reject,
    phase2_chi2_reject,
)


def test_phase1_two_sided_clips_outliers():
    rng = np.random.default_rng(0)
    norms = rng.normal(1.0, 0.05, size=200)
    norms[0] = 3.0  # high outlier
    norms[1] = -1.0  # low outlier
    keep, mu, sig, diff = phase1_norm_reject(norms, ker_sig_reject=2.5)
    assert keep[0] is np.False_ or keep[0] == False
    assert keep[1] is np.False_ or keep[1] == False
    assert keep[2:].mean() > 0.9
    assert abs(mu - 1.0) < 0.05
    assert sig > 0
    assert diff[0] >= 2.5


def test_phase2_one_sided_keeps_low_chi2():
    rng = np.random.default_rng(1)
    chi2 = rng.normal(1.0, 0.1, size=200)
    chi2[0] = 5.0  # bad high residual
    chi2[1] = 0.01  # better than average — must keep
    keep, mu, sig = phase2_chi2_reject(chi2, ker_sig_reject=2.5)
    assert keep[0] == False
    assert keep[1] == True
    assert keep.mean() > 0.9
    assert abs(mu - 1.0) < 0.1


def test_phase2_respects_active_mask():
    rng = np.random.default_rng(2)
    chi2 = rng.normal(1.0, 0.05, size=50)
    chi2[0] = 8.0  # clear high outlier among actives
    active = np.ones(50, dtype=bool)
    active[10] = False
    chi2[10] = 8.0  # inactive high value must stay out even if not "clipped"
    keep, _, _ = phase2_chi2_reject(chi2, active=active, ker_sig_reject=2.5)
    assert keep[10] == False
    assert keep[0] == False
    assert keep[1:].mean() > 0.8
