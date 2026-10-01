# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for ``temporal.build_temporal_basis_gap_aware`` (task T1-2).

Opt-in constructor: denser knots in the first ~1.5 days plus an interior
knot cluster at a within-orbit gap, so the temporal ePSF/WCS spline can
represent the fast early-orbit settling and the momentum-dump-gap step
documented in ``docs/TEMPORAL_RESIDUAL_ROOT_CAUSE_20260906.md`` sec 6.
Default ``build_temporal_basis`` behaviour is untouched by this module.
"""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.forward_model import temporal as T


def _synthetic_btjd(n_each: int = 400) -> tuple[np.ndarray, float]:
    gap = 2724.4307
    t1 = np.linspace(2718.64, 2724.40, n_each)
    t2 = np.linspace(2724.435, 2730.35, n_each)
    return np.concatenate([t1, t2]), gap


def test_partition_of_unity_and_identifiable():
    btjd, gap = _synthetic_btjd()
    tb = T.build_temporal_basis_gap_aware(btjd, gap_btjd=(gap - 0.02, gap + 0.02))
    B = np.asarray(tb.frame_basis)
    assert B.shape[0] == btjd.shape[0]
    assert tb.n_basis == B.shape[1]
    # clamped B-spline: rows sum to 1 everywhere in [0, 1].
    np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-5)
    assert btjd.shape[0] > tb.n_basis


def test_gap_knots_are_coincident_and_within_span():
    btjd, gap = _synthetic_btjd()
    tb = T.build_temporal_basis_gap_aware(
        btjd, gap_btjd=(gap - 0.02, gap + 0.02), gap_knot_multiplicity=3,
    )
    tau_gap = (gap - tb.btjd_ref) / tb.btjd_scale
    near = np.isclose(tb.knot_vector, tau_gap, atol=1e-6)
    assert int(near.sum()) == 3


def test_no_gap_still_builds_a_valid_basis():
    btjd, _ = _synthetic_btjd()
    tb = T.build_temporal_basis_gap_aware(btjd, gap_btjd=None)
    B = np.asarray(tb.frame_basis)
    np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-5)


def test_gap_multiplicity_out_of_range_rejected():
    btjd, gap = _synthetic_btjd()
    with pytest.raises(ValueError):
        T.build_temporal_basis_gap_aware(
            btjd, degree=3, gap_btjd=(gap, gap), gap_knot_multiplicity=0,
        )
    with pytest.raises(ValueError):
        T.build_temporal_basis_gap_aware(
            btjd, degree=3, gap_btjd=(gap, gap), gap_knot_multiplicity=5,
        )


def test_higher_multiplicity_fits_a_true_step_better():
    """A genuine value-discontinuity target is fit much better as the gap
    knot's multiplicity rises toward ``degree + 1`` -- this is the whole
    point of the feature (see the module docstring: no data exists inside
    the gap, so the two sides can be nearly independent). Condition number
    also stays small throughout (well-posed, not a numerical artefact).
    """
    rng = np.random.default_rng(0)
    btjd, gap = _synthetic_btjd(500)
    step = np.where(btjd < gap, 0.0, 0.004)
    target = step + rng.normal(0, 1e-5, size=btjd.shape)
    near = np.abs(btjd - gap) < 0.15

    rms_near = {}
    conds = {}
    for m in (1, 2, 3, 4):
        tb = T.build_temporal_basis_gap_aware(
            btjd, degree=3, gap_btjd=(gap - 0.02, gap + 0.02), gap_knot_multiplicity=m,
        )
        B = np.asarray(tb.frame_basis)
        coeff, *_ = np.linalg.lstsq(B, target, rcond=None)
        resid = target - B @ coeff
        rms_near[m] = float(np.std(resid[near]))
        conds[m] = float(np.linalg.cond(B))

    assert rms_near[4] < 0.3 * rms_near[1]
    assert all(c < 1e3 for c in conds.values())


def test_second_difference_matrix_is_shape_agnostic_for_gap_aware_basis():
    """Deliverable T1-2's shape-agnostic requirement: the existing smoothness
    penalty machinery must work unmodified for the new (larger, differently
    structured) basis size.
    """
    btjd, gap = _synthetic_btjd()
    tb = T.build_temporal_basis_gap_aware(btjd, gap_btjd=(gap - 0.02, gap + 0.02))
    D = T.second_difference_matrix(tb.n_basis)
    assert D.shape == (tb.n_basis - 2, tb.n_basis)


def test_dense_start_knots_improve_fast_early_settle_fit():
    btjd, gap = _synthetic_btjd(500)
    t0 = btjd[0]
    settle = 0.002 * np.exp(-(btjd - t0) / 0.3)
    early = (btjd - t0) < 1.5

    default = T.build_temporal_basis(btjd, degree=3, n_interior=10, edge_frac=0.12)
    dense = T.build_temporal_basis_gap_aware(
        btjd, degree=3, dense_start_days=1.5, n_dense_start=6, n_mid=3, n_end=2,
        gap_btjd=None,
    )
    for tb, name in ((default, "default"), (dense, "dense")):
        B = np.asarray(tb.frame_basis)
        coeff, *_ = np.linalg.lstsq(B, settle, rcond=None)
        resid = settle - B @ coeff
        rms = float(np.std(resid[early]))
        if name == "default":
            default_rms = rms
        else:
            dense_rms = rms
    assert dense_rms < default_rms
