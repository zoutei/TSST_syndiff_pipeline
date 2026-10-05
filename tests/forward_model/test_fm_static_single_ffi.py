# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Single-FFI (static) fit: constant temporal basis and the static reject gate.

The static fit drops time entirely: one FFI, one constant column in place of the
B-spline in ``t``. The point of these tests is that ``n_basis == 1`` is a real,
supported configuration rather than an accident that happens to not crash -- and
that the frame-axis reject modes, which are silently inert at one frame, fail
closed instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import stamp_reject as SR
from syndiff_pipeline.forward_model import temporal as T


# ---------------------------------------------------------------------------
# Static temporal basis
# ---------------------------------------------------------------------------


def test_static_basis_is_a_single_constant_column():
    b = T.build_static_basis([1975.75])
    assert np.asarray(b.frame_basis).shape == (1, 1)
    assert np.asarray(b.frame_basis).tolist() == [[1.0]]
    assert b.n_basis == 1
    assert b.degree == 0
    assert b.btjd_ref == pytest.approx(1975.75)
    # Not the frame span: a single frame has zero span, which would divide by zero
    # in _tau. tau is never evaluated for a constant basis.
    assert b.btjd_scale == 1.0


def test_static_basis_accepts_several_frames():
    """Same flag also means "one static model fitted jointly to N frames"."""
    b = T.build_static_basis([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.asarray(b.frame_basis).shape == (5, 1)
    assert np.all(np.asarray(b.frame_basis) == 1.0)
    assert b.n_basis == 1


def test_static_basis_needs_at_least_one_frame():
    with pytest.raises(ValueError, match="at least 1 frame"):
        T.build_static_basis([])


def test_bspline_basis_still_refuses_a_single_frame():
    """The guard that makes --static-basis necessary must stay in place."""
    with pytest.raises(ValueError, match="at least 2 frames"):
        T.build_temporal_basis(np.array([1975.75]))


def test_smoothness_operator_is_empty_at_one_basis_function():
    """Why no special case is needed in the loss: the penalty is a clean zero.

    ``spline_smoothness_penalty`` contracts ``wcs_coeff`` with this operator, so an
    empty (0, 1) operator makes the curvature penalty vanish rather than error --
    which is correct, since a constant has no curvature to penalize.
    """
    D = np.asarray(T.second_difference_matrix(1))
    assert D.shape == (0, 1)
    assert np.asarray(T.second_difference_matrix(14)).shape == (12, 14)


def test_degree_five_cheb_static_has_21_terms_per_axis():
    """The whole-CCD distortion order this study asks for."""
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents

    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([1025.0, 1025.0], dtype=float),
        center=np.array([1024.0, 1024.0], dtype=float),
        half_extents=np.array([1024.0, 1024.0], dtype=float),
        poly_degree=5, exponents=tuple(sci2idl_exponents(5)),
    )
    assert static.n_terms == 21
    coeff = CW.zero_coeff_matrix(static, 1)
    assert np.asarray(coeff).shape == (42, 1)


def test_static_basis_positions_are_frame_independent():
    """A constant basis must give every frame the same distortion."""
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents
    import jax.numpy as jnp

    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([1025.0, 1025.0], dtype=float),
        center=np.array([1024.0, 1024.0], dtype=float),
        half_extents=np.array([1024.0, 1024.0], dtype=float),
        poly_degree=2, exponents=tuple(sci2idl_exponents(2)),
    )
    basis = T.build_static_basis([10.0, 20.0, 30.0])
    rng = np.random.default_rng(0)
    coeff = jnp.asarray(rng.normal(size=(2 * static.n_terms, 1)).astype(np.float32))
    x_lin = jnp.asarray(np.array([900.0, 1100.0], dtype=np.float32))
    y_lin = jnp.asarray(np.array([1000.0, 1200.0], dtype=np.float32))
    star_basis = jnp.asarray(rng.normal(size=(2, static.n_terms)).astype(np.float32))
    x, y = CW.eval_all_positions(
        x_lin, y_lin, star_basis, coeff, basis.frame_basis, static.n_terms,
    )
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == (2, 3)
    assert np.allclose(x, x[:, :1])
    assert np.allclose(y, y[:, :1])


# ---------------------------------------------------------------------------
# Static reject gate
# ---------------------------------------------------------------------------


def _trend(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    brightness = np.sort(10 ** rng.uniform(2.0, 6.0, size=n))
    # chi2_red rises smoothly with brightness: this is the trend a pooled MAD
    # would misread as a bright-end full of outliers.
    log_chi2 = 0.4 * np.log10(brightness) + 0.15 * rng.normal(size=n)
    return log_chi2, brightness


def test_static_gate_does_not_cut_a_pure_brightness_trend():
    log_chi2, brightness = _trend()
    drop, dev = SR.static_rank_outliers(log_chi2, brightness)
    assert drop.sum() == 0
    assert np.isfinite(dev).all()


def test_static_gate_catches_injected_outliers_including_at_the_ends():
    log_chi2, brightness = _trend()
    bad = np.array([5, 700, log_chi2.size - 1])
    log_chi2[bad] += 3.0
    drop, _ = SR.static_rank_outliers(log_chi2, brightness)
    assert drop[bad].all()
    assert drop.sum() == bad.size


def test_static_gate_is_one_sided():
    """An implausibly GOOD fit is not evidence of a bad star."""
    log_chi2, brightness = _trend()
    log_chi2[[10, 900]] -= 4.0
    drop, _ = SR.static_rank_outliers(log_chi2, brightness)
    assert drop.sum() == 0


def test_static_gate_survives_a_degenerate_window():
    """Identical scores give MAD = 0; the floor must stop it cutting everything."""
    n = 64
    log_chi2 = np.zeros(n)
    brightness = np.arange(n, dtype=float)
    drop, _ = SR.static_rank_outliers(log_chi2, brightness, window=9)
    assert drop.sum() == 0


def test_static_gate_needs_a_minimum_sample():
    log_chi2 = np.array([1.0, 2.0, 3.0])
    drop, dev = SR.static_rank_outliers(log_chi2, np.array([1.0, 2.0, 3.0]))
    assert drop.sum() == 0
    assert np.isnan(dev).all()


def test_static_gate_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        SR.static_rank_outliers(np.zeros(5), np.zeros(4))


def test_frame_axis_reject_modes_are_refused_at_one_frame():
    """Fail closed, do not no-op.

    Every frame-axis mode centres log-chi2 per group across frames, so at one frame
    the deviation is identically zero: the gate would log as though it ran and
    reject nothing. That is the silent-wrong-number failure this refuses.
    """
    from syndiff_pipeline.forward_model import fit as FIT

    class _Ctx:
        stamp_active = np.ones((4, 1), dtype=np.float32)

    class _FD:
        ctx = _Ctx()

    for mode in ("hysteresis", "standardized", "legacy", "two-level"):
        with pytest.raises(ValueError, match="ACROSS"):
            FIT.run_stage(
                {}, [_FD()], stage=2, n_steps=0, lr=1e-3,
                reject_every=50, reject_mode=mode,
            )


def test_static_and_audit_modes_are_allowed_at_one_frame():
    """The two modes that mean something at one frame must not trip the guard.

    ``n_steps=0`` returns before any real work, so this exercises the guard only.
    """
    from syndiff_pipeline.forward_model import fit as FIT

    class _Ctx:
        stamp_active = np.ones((4, 1), dtype=np.float32)

    class _FD:
        ctx = _Ctx()

    for mode in ("static", "audit"):
        # Either completes (audit needs nothing else) or fails later on this stub's
        # missing arrays (static reads fd.data for its brightness ordinate). What
        # must not happen is the frame-axis guard firing.
        try:
            FIT.run_stage(
                {}, [_FD()], stage=2, n_steps=0, lr=1e-3,
                reject_every=50, reject_mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 - asserting on the message
            assert "ACROSS" not in str(exc), f"{mode} tripped the frame-axis guard"


# ---------------------------------------------------------------------------
# K = 0 mode init (what --mode-init "" needs)
# ---------------------------------------------------------------------------


def test_mode_init_supports_k_zero():
    """``--mode-init ""`` must build a zero-length mode stack, not raise.

    K=0 is already supported by the loss and the optimizer, but the finite-difference
    init used to ``np.stack([])`` and die -- which is what a single-FFI export hits,
    since one frame makes every ``w_k`` identically zero under the zero-temporal-mean
    gauge and the modes unidentifiable by construction.
    """
    from syndiff_pipeline.forward_model import epsf_model as EM

    rng = np.random.default_rng(0)
    g = EM.NODE_GRID_SIZE  # pixel-integrated default (was a hardcoded 58 pre-representation-change)
    base = np.abs(rng.normal(size=(2, 2, g, g))).astype(np.float32)
    base /= base.sum(axis=(2, 3), keepdims=True)

    p0 = EM.init_epsf_from_base(base, mode_names=())
    assert np.asarray(p0.base).shape == (2, 2, g, g)
    assert np.asarray(p0.modes).shape == (0, 2, 2, g, g)

    p1 = EM.init_epsf_from_base(base, mode_names=("iso_defocus",))
    assert np.asarray(p1.modes).shape == (1, 2, 2, g, g)


def test_k_zero_params_have_empty_mode_leaves():
    import jax.numpy as jnp

    from syndiff_pipeline.forward_model import epsf_model as EM
    from syndiff_pipeline.forward_model import loss as L
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents

    rng = np.random.default_rng(1)
    g = EM.NODE_GRID_SIZE  # pixel-integrated default (was a hardcoded 58 pre-representation-change)
    base = np.abs(rng.normal(size=(2, 2, g, g))).astype(np.float32)
    base /= base.sum(axis=(2, 3), keepdims=True)
    epsf = EM.init_epsf_from_base(base, mode_names=())
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([1025.0, 1025.0], dtype=float),
        center=np.array([1024.0, 1024.0], dtype=float),
        half_extents=np.array([1024.0, 1024.0], dtype=float),
        poly_degree=5, exponents=tuple(sci2idl_exponents(5)),
    )
    params = L.init_params(static, epsf, n_wcs_basis=1, n_w_basis=1)
    assert np.asarray(params["epsf_modes"]).shape[0] == 0
    assert np.asarray(params["w_coeff"]).shape == (0, 1)
    assert np.asarray(params["wcs_coeff"]).shape == (42, 1)
    assert jnp.all(jnp.isfinite(params["epsf_base_raw"]))


def test_chroma_image_has_its_own_optimizer_bucket():
    """Not the parametric colour bucket: different units, and a shared clip.

    ``optax.multi_transform`` evaluates ``clip_by_global_norm`` per bucket, so
    3364 image parameters in the same bucket as 180 parametric numbers would set
    the clip factor for both. Measured 2026-09-16: sharing the bucket at scale 1.0
    took the single-FFI loss from 68.03 to 104.37 in one step.
    """
    from syndiff_pipeline.forward_model import fit as FIT

    labels = FIT._leaf_labels(4, freeze_wcs=False, param_keys=FIT.ALL_LEAVES)
    assert labels["chroma_image"] == "train_chroma_image"
    for k in ("chroma_shift", "chroma_dilation", "chroma_aniso", "chroma_shear"):
        assert labels[k] == "train_chroma", k
    assert FIT.CHROMA_IMAGE_LR_SCALE_DEFAULT < 1.0
    # Must build: every label needs a transform, or multi_transform raises.
    FIT.make_stage_optimizer(4, 1e-4, param_keys=FIT.ALL_LEAVES)


def test_enclose_pad_grows_member_squares_and_default_is_a_no_op():
    """``--irregular-enclose-pad-px`` must widen the PAINTED support, not just the tier.

    The pad exists to make an r^-2 chromatic halo separable from a flat per-stamp
    pedestal, which only works if the extra pixels are actually fitted. Default 0 has
    to stay bit-identical, because every historical bundle was exported without it.
    """
    import numpy as np

    from syndiff_pipeline.forward_model import irregular_stamps as IS

    # One isolated segment: a 5x5 block of label 1 at (20, 20) in a 60x60 map.
    label_map = np.zeros((60, 60), dtype=np.int32)
    label_map[18:23, 18:23] = 1
    cx = np.array([20], dtype=int)
    cy = np.array([20], dtype=int)

    tight = IS.member_stamp_sizes_for_segment(
        label_map, cx, cy, 1, s_max=13, s_min=5, enclose_pad_px=0)
    assert tight.tolist() == [5], tight

    for pad, expected in ((1, 7), (2, 9), (3, 11), (4, 13), (9, 13)):
        sizes = IS.member_stamp_sizes_for_segment(
            label_map, cx, cy, 1, s_max=13, s_min=5, enclose_pad_px=pad)
        # 2*pad on top of the tight enclose, clamped to s_max and kept odd.
        assert sizes.tolist() == [expected], (pad, sizes)

    # The painted mask -- i.e. the fitted pixels -- grows with it.
    n_pix = {}
    for pad in (0, 3):
        mask = np.zeros_like(label_map, dtype=bool)
        size = int(IS.member_stamp_sizes_for_segment(
            label_map, cx, cy, 1, s_max=13, s_min=5, enclose_pad_px=pad)[0])
        IS.paint_stamp_window(mask, 20, 20, stamp_physical=size)
        n_pix[pad] = int(mask.sum())
    assert n_pix[0] == 25
    assert n_pix[3] == 121
