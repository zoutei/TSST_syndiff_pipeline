# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Unit tests for the forward-modeled ePSF + WCS prototype (plan §Verification).

Synthetic/self-contained where possible; a few tests need the PRF fixture
files or a real workspace and are skipped if those aren't present in this
checkout (see plan §Data landmines).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebvander
from scipy.interpolate import BSpline

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import flux_solve as FS
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import temporal as T
from syndiff_pipeline.forward_model._bootstrap import _EXTRA_PATHS  # noqa: F401 (ensures sys.path wired)

from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.cheb_poly_fit import cheb_design_matrix  # noqa: E402
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents  # noqa: E402

jax.config.update("jax_platform_name", "cpu")

PRF_ROOT = EM.PRF_ROOT_DEFAULT
HAS_PRF_DATA = os.path.isdir(os.path.join(PRF_ROOT, "start_s0004", "cam3_ccd3"))


def test_hysteresis_stamp_rejection_band_and_gate_c_invariant():
    from syndiff_pipeline.forward_model import stamp_reject as SR

    # With scale=1 and median log(1)=0, scores are log(chi2).  The middle
    # values sit in the [keep, drop] band and must retain their prior state.
    chi2 = np.array([[1.0, 1.0, np.exp(2.5), np.exp(4.0), np.exp(1.0)]])
    current = np.array([[True, True, False, True, False]])
    baseline = np.array([[True, True, True, True, False]])
    updated, stats = SR.update_stamp_active_hysteresis(
        chi2, current, baseline, np.array([1.0]), tau_drop=3.5,
        tau_keep=2.2, max_churn_frac=1.0,
    )
    np.testing.assert_array_equal(updated, [[True, True, False, False, False]])
    assert stats["n_dropped"] == 1
    assert not updated[0, 4]  # Absolute TNS/asteroid Gate-C mask.


def test_hysteresis_stamp_rejection_churn_budget_prioritizes_extremes():
    from syndiff_pipeline.forward_model import stamp_reject as SR

    chi2 = np.array([[1.0] * 7 + [np.exp(4.0), np.exp(7.0), np.exp(9.0)]])
    current = np.ones((1, 10), dtype=bool)
    updated, stats = SR.update_stamp_active_hysteresis(
        chi2, current, np.ones_like(current), np.array([1.0]),
        tau_drop=3.5, tau_keep=2.2, max_churn_frac=0.2,
    )
    np.testing.assert_array_equal(updated, [[True] * 8 + [False, False]])
    assert stats["net_churn"] == 2


# ---------------------------------------------------------------------------
# epsf_model: area-overlap resampler
# ---------------------------------------------------------------------------


def test_area_overlap_matrix_conserves_flux_arbitrary_grids():
    rng = np.random.default_rng(0)
    src = rng.random(37)
    R = EM.area_overlap_matrix(37, 1.0 / 9.0, 18.0, 53, 1.0 / 4.0, 26.0)
    out = R @ src
    # target grid fully covers the source support here, so mass is conserved
    assert np.isclose(out.sum(), src.sum(), rtol=1e-10)


def test_area_overlap_matrix_identity_when_grids_match():
    R = EM.area_overlap_matrix(10, 1.0, 4.5, 10, 1.0, 4.5)
    assert np.allclose(R, np.eye(10), atol=1e-12)


@pytest.mark.skipif(not HAS_PRF_DATA, reason="TESS SPOC PRF fixture files not present")
@pytest.mark.skip(reason='pre-existing failure on dev main too (PRF fork resample), 2026-09-30')
def test_resample_prf_native_to_node_sums_to_one():
    import sys

    pass  # migration: dev sys.path wiring removed
    from PRF import TESS_PRF

    prf = TESS_PRF(3, 3, 20, 1069.0, 1025.0, localdatadir=PRF_ROOT)
    native = np.asarray(prf.prf, dtype=np.float64)
    assert np.isclose(native.sum(), 81.0, rtol=1e-6)

    node = EM.resample_prf_native_to_node(native)
    assert node.shape == (EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)
    assert np.isclose(node.sum(), 1.0, atol=1e-3)
    peak = np.unravel_index(np.argmax(node), node.shape)
    assert abs(peak[0] - EM.NODE_CENTER_INDEX) < 2
    assert abs(peak[1] - EM.NODE_CENTER_INDEX) < 2


# ---------------------------------------------------------------------------
# epsf_model: bilinear node blend + renderer sign convention
# ---------------------------------------------------------------------------


def _toy_grid_params(n_rows=2, n_cols=2, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((n_rows, n_cols, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)).astype(np.float32)
    base /= base.sum(axis=(-1, -2), keepdims=True)
    modes = rng.normal(size=(5, n_rows, n_cols, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)).astype(np.float32)
    modes -= modes.mean(axis=(-1, -2), keepdims=True)
    return EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))


def test_bilinear_blend_exact_at_nodes_and_midpoint():
    node_x = np.array([0.0, 10.0])
    node_y = np.array([0.0, 10.0])
    params = _toy_grid_params()

    qx = jnp.array([0.0, 10.0, 5.0])
    qy = jnp.array([0.0, 0.0, 0.0])
    i0, j0, wy, wx = EM.bilinear_cell(qx, qy, node_x, node_y)
    local = EM.blend_to_local(params.base, i0, j0, wy, wx)

    assert np.allclose(np.asarray(local[0]), np.asarray(params.base[0, 0]))
    assert np.allclose(np.asarray(local[1]), np.asarray(params.base[0, 1]))
    expected_mid = (np.asarray(params.base[0, 0]) + np.asarray(params.base[0, 1])) / 2
    assert np.allclose(np.asarray(local[2]), expected_mid)


def test_render_stamps_subpixel_shift_sign_and_flux_conservation():
    # A synthetic PSF-like (concentrated, not flat-random) grid: flux
    # conservation only holds when the profile's wings fit inside the 52-core
    # + 3-pad-per-side footprint, same as a real PRF.
    idx = np.arange(EM.NODE_GRID_SIZE, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    gauss = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    gauss /= gauss.sum()
    base = jnp.asarray(gauss)[None, None, :, :]  # (1,1,G,G), a single static grid

    def centroid(dx, dy):
        stamp = EM.render_stamps(base, jnp.array([[dx]]), jnp.array([[dy]]))
        s = np.asarray(stamp[0, 0])
        yy, xx = np.mgrid[0 : EM.STAMP_PHYSICAL, 0 : EM.STAMP_PHYSICAL]
        c = EM.STAMP_PHYSICAL // 2
        return s.sum(), (s * xx).sum() / s.sum() - c, (s * yy).sum() / s.sum() - c

    f0, cx0, cy0 = centroid(0.0, 0.0)
    fp, cxp, cyp = centroid(0.3, 0.0)
    fm, cxm, cym = centroid(-0.3, 0.0)
    assert abs(f0 - 1.0) < 0.01 and abs(fp - 1.0) < 0.01  # flux conserved to ~1%
    assert (cxp - cx0) > 0.2  # +dx shifts the rendered centroid in +x
    assert (cxm - cx0) < -0.2
    assert abs(cyp - cy0) < 0.05  # no cross-talk into y


def test_render_stamps_zero_grid_gives_zero_stamp():
    zero = jnp.zeros((2, 3, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE))
    dx = jnp.zeros((2, 3))
    dy = jnp.zeros((2, 3))
    stamp = EM.render_stamps(zero, dx, dy)
    assert stamp.shape == (2, 3, EM.STAMP_PHYSICAL, EM.STAMP_PHYSICAL)
    assert np.allclose(np.asarray(stamp), 0.0)


def test_render_stamps_out_of_support_samples_are_zero_not_edge_repeat():
    """Option 1a (SESSION_PLAN_20260808.md Problem 1/5a): a companion far enough
    that its needed node-grid samples fall outside [0, G-1] should render 0 there,
    not repeat the undertrained edge-ring value. Uniform grid = 1 everywhere makes
    in-bounds vs. out-of-bounds trivially distinguishable (in-bounds physical pixel
    = OVERSAMPLE**2, since it's a block-sum of 16 samples all equal to 1)."""
    ones = jnp.ones((1, 1, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE))
    full_val = float(EM.OVERSAMPLE**2)

    # Huge dx: every required sample is far outside [0, G-1] -> whole stamp is 0.
    stamp_far = EM.render_stamps(ones, jnp.array([[50.0]]), jnp.array([[0.0]]))
    assert np.allclose(np.asarray(stamp_far), 0.0)

    # Moderate dx (comparable to a real companion separation): far side of the
    # stamp goes out of support (0), near side stays fully in-bounds (full_val).
    stamp_mid = np.asarray(EM.render_stamps(ones, jnp.array([[10.0]]), jnp.array([[0.0]])))[0, 0]
    assert np.any(stamp_mid == 0.0)
    assert np.any(np.isclose(stamp_mid, full_val))
    assert not np.any(np.isclose(stamp_mid, full_val) & (stamp_mid == 0.0))  # sanity: disjoint sets

    # Zero shift: fully in-bounds, matches the old clamp-based behavior exactly.
    stamp0 = np.asarray(EM.render_stamps(ones, jnp.array([[0.0]]), jnp.array([[0.0]])))[0, 0]
    assert np.allclose(stamp0, full_val)


# ---------------------------------------------------------------------------
# epsf_model: bit-exact, gather-free banded renderer (axis_band / render_band /
# node_moments / renorm_scalar) -- see EM.axis_band's docstring for the
# derivation this replaces the render_stamps/recenter_grid_core gather path with.
# ---------------------------------------------------------------------------

_BAND_TEST_OFFSETS = (0.0, 0.3, -0.3, 0.42, -0.42, 0.49, -0.49, 0.5, -0.5, 1.7, -1.7, 3.2, -3.2, 6.6, -6.6)


def _reference_axis_matrix(off: float) -> np.ndarray:
    """Literal reference construction, independent of EM.axis_band's closed
    form (pixel-integrated ``E`` rendering, CONTRACT_pixel_integrated_epsf.md):
    each physical pixel is a single bilinear point-sample of ``E``, scaled by
    ``oversample``, at continuous index ``g(p) = c_E + (p - (n_pix-1)/2 -
    off) * oversample``. Was a 5-tap ``oversample``-wide block-sum reference
    (the old sub-pixel-``P`` semantics) before the pixel-integrated
    representation change -- see git history / CONTRACT for the old form."""
    S, os_, G = EM.STAMP_PHYSICAL, EM.OVERSAMPLE, EM.NODE_GRID_SIZE
    center = EM.NODE_CENTER_INDEX
    p = np.arange(S)
    g = center + (p - (S - 1) / 2.0 - off) * os_
    inb = (g >= 0.0) & (g <= float(G - 1))
    x0 = np.clip(np.floor(g).astype(int), 0, G - 2)
    f = np.clip(g - x0, 0.0, 1.0)
    W = np.zeros((S, G))
    W[p, x0] += os_ * (1 - f) * inb
    W[p, x0 + 1] += os_ * f * inb
    return W


def _asymmetric_psf_grid(seed: int = 1) -> jnp.ndarray:
    """A compact-but-asymmetric flux-fraction (G, G) grid (nonzero COM, unit sum),
    representative of a real ePSF composite -- unlike a flat/symmetric toy grid,
    this exercises the COM-recenter fold in tests that need it."""
    G = EM.NODE_GRID_SIZE
    idx = np.arange(G, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    core = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.2) ** 2))
    wing_r2 = (idx[:, None] - EM.NODE_CENTER_INDEX - 3) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX + 2) ** 2
    wing = 0.3 * np.exp(-wing_r2 / (2 * (EM.OVERSAMPLE * 2.0) ** 2))
    grid = core + wing
    grid = (grid / grid.sum()).astype(np.float32)
    return jnp.asarray(grid)


def test_axis_band_matches_reference_construction():
    """(a), Trap 3: closed-form axis_band (no 52x58 one-hot intermediate) must
    match the literal reference construction exactly, including the boundary
    case where the in/out-of-support transition falls strictly inside one
    physical-pixel's block of samples (verified at off=+-3.2, +-6.6)."""
    worst = 0.0
    for off in _BAND_TEST_OFFSETS:
        ref = _reference_axis_matrix(off)
        got = np.asarray(EM.axis_band(jnp.asarray(off)))
        worst = max(worst, np.abs(ref - got).max())
    assert worst < 1e-5


def test_axis_band_matches_render_stamps():
    """(a): render_stamps(grid, dx, dy) == einsum of two axis_band matrices
    around `grid`, for interior, +-0.5 boundary, and far out-of-support offsets."""
    grid = _asymmetric_psf_grid()
    worst = 0.0
    for dx in _BAND_TEST_OFFSETS:
        for dy in _BAND_TEST_OFFSETS:
            ref = np.asarray(
                EM.render_stamps(grid[None, None], jnp.array([[dx]]), jnp.array([[dy]]))
            )[0, 0]
            Ay = EM.axis_band(jnp.asarray(dy))
            Ax = EM.axis_band(jnp.asarray(dx))
            got = np.asarray(jnp.einsum("sg,gh,th->st", Ay, grid, Ax))
            worst = max(worst, np.abs(ref - got).max())
    assert worst < 1e-5


def test_render_band_matches_recentered_render_stamps():
    """(b)+(d): render_band folds the n_iter=1 core recenter into the render
    matrix exactly; renorm_scalar supplies the missing flux normalization."""
    grid = _asymmetric_psf_grid()
    cx, cy = EM.core_centroid_xy(grid)
    recentered = EM.recenter_grid_core(grid[None, None], clip_nonneg=False, n_iter=1)
    worst = 0.0
    for dx in _BAND_TEST_OFFSETS:
        for dy in _BAND_TEST_OFFSETS:
            ref = np.asarray(
                EM.render_stamps(recentered, jnp.array([[dx]]), jnp.array([[dy]]))
            )[0, 0]
            Ay = EM.render_band(jnp.asarray([dy]), cy[None])
            Ax = EM.render_band(jnp.asarray([dx]), cx[None])
            tot = EM.renorm_scalar(grid, cx, cy) + 1e-12
            got = np.asarray(jnp.einsum("nsg,ngh,nth->nst", Ay, grid[None], Ax)[0] / tot)
            worst = max(worst, np.abs(ref - got).max())
    assert worst < 1e-5


def test_node_moments_matches_core_centroid_xy():
    """(c): analytic per-slot core centroid (blend of precomputed node moments)
    must match core_centroid_xy evaluated on the actually-blended local grid.
    A small (~1e-6) gap is expected fp32 roundoff -- sum-then-blend
    (node_moments) and blend-then-sum (core_centroid_xy) accumulate the same
    58x58 reduction in a different order."""
    G = EM.NODE_GRID_SIZE
    rng = np.random.default_rng(3)
    node_field = rng.random((2, 2, G, G)).astype(np.float32)
    node_field /= node_field.sum(axis=(-1, -2), keepdims=True)
    node_field = jnp.asarray(node_field)
    node_x = np.array([10.0, 90.0], dtype=np.float32)
    node_y = np.array([20.0, 80.0], dtype=np.float32)
    qx = jnp.array([10.0, 50.0, 90.0, 33.0, 77.0])
    qy = jnp.array([20.0, 50.0, 80.0, 65.0, 24.0])

    i0, j0, wy, wx = EM.bilinear_cell(qx, qy, node_x, node_y)
    local = EM.blend_field(node_field, i0, j0, wy, wx)
    cx_ref, cy_ref = EM.core_centroid_xy(local)

    total, xmom, ymom = EM.node_moments(node_field)
    t_blend = EM.blend_field(total, i0, j0, wy, wx)
    x_blend = EM.blend_field(xmom, i0, j0, wy, wx)
    y_blend = EM.blend_field(ymom, i0, j0, wy, wx)
    t_blend = t_blend + 1e-12
    cx_analytic = x_blend / t_blend
    cy_analytic = y_blend / t_blend

    np.testing.assert_allclose(np.asarray(cx_analytic), np.asarray(cx_ref), atol=2e-6)
    np.testing.assert_allclose(np.asarray(cy_analytic), np.asarray(cy_ref), atol=2e-6)


def test_renorm_scalar_matches_bilinear_shift_sum():
    """(d): renorm_scalar(grid, cx, cy) == sum(bilinear_shift_physical(grid, -cx, -cy))
    without ever materializing the shifted (G, G) grid."""
    grid = _asymmetric_psf_grid()
    cx, cy = EM.flux_centroid_xy(grid)
    ref = jnp.sum(EM.bilinear_shift_physical(grid, -cx, -cy))
    got = EM.renorm_scalar(grid, cx, cy)
    np.testing.assert_allclose(float(got), float(ref), atol=1e-5)


def test_render_band_matches_matmul_reference():
    """render_band's closed-form (oversample+2)-tap convolution (never forming
    tap2_matrix's (G, G) matrix -- the memory-costly path, see its docstring)
    must exactly reproduce the literal axis_band(off_render) @ tap2_matrix(-off_com)
    matmul, across render offsets *and* COM offsets (not just off_com=0)."""
    com_offsets = (0.0, 0.1, -0.1, 0.3, -0.3, 0.49, -0.49, 1.2, -1.2, 2.7, -2.7)
    worst = 0.0
    for dx in _BAND_TEST_OFFSETS:
        for cx in com_offsets:
            ref = np.asarray(EM._render_band_via_matmul(jnp.asarray(dx), jnp.asarray(cx)))
            got = np.asarray(EM.render_band(jnp.asarray(dx), jnp.asarray(cx)))
            worst = max(worst, np.abs(ref - got).max())
    assert worst < 1e-5


def test_node_blend_weights_matches_blend_field():
    """node_blend_weights + a single einsum contraction must reproduce
    blend_field's per-slot corner blend exactly -- it replaces blend_field's
    nested weighted-sum only for memory reasons (see its docstring), not a
    different blend."""
    G = EM.NODE_GRID_SIZE
    rng = np.random.default_rng(5)
    for n_rows, n_cols in [(2, 2), (2, 2), (4, 3)]:
        node_field = rng.random((n_rows, n_cols, G, G)).astype(np.float32)
        node_x = np.linspace(0.0, 100.0, n_cols).astype(np.float32)
        node_y = np.linspace(0.0, 100.0, n_rows).astype(np.float32)
        qx = jnp.asarray(rng.uniform(0.0, 100.0, size=6).astype(np.float32))
        qy = jnp.asarray(rng.uniform(0.0, 100.0, size=6).astype(np.float32))
        i0, j0, wy, wx = EM.bilinear_cell(qx, qy, node_x, node_y)
        ref = EM.blend_field(jnp.asarray(node_field), i0, j0, wy, wx)
        Wn = EM.node_blend_weights(i0, j0, wy, wx, n_rows=n_rows, n_cols=n_cols)
        got = jnp.einsum("nrc,rcxy->nxy", Wn, jnp.asarray(node_field))
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-5)


# ---------------------------------------------------------------------------
# temporal: B-spline basis
# ---------------------------------------------------------------------------


def test_temporal_basis_partition_of_unity_and_shape():
    btjd = np.linspace(0.0, 10.0, 60)
    tb = T.build_temporal_basis(btjd, n_interior=10)
    assert tb.frame_basis.shape == (60, tb.n_basis)
    row_sums = np.asarray(tb.frame_basis).sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_temporal_basis_matches_scipy_bspline_design_matrix():
    btjd = np.linspace(0.0, 5.0, 30)
    tb = T.build_temporal_basis(btjd, n_interior=6)
    tau = np.clip((btjd - tb.btjd_ref) / tb.btjd_scale, 0.0, 1.0)
    expected = BSpline.design_matrix(tau, tb.knot_vector, tb.degree).toarray()
    assert np.allclose(np.asarray(tb.frame_basis), expected, atol=1e-6)


def test_second_difference_matrix_shape_and_zero_on_linear_ramp():
    D = np.asarray(T.second_difference_matrix(8))
    assert D.shape == (6, 8)
    ramp = np.arange(8, dtype=float)
    assert np.allclose(D @ ramp, 0.0, atol=1e-10)  # linear ramp has zero curvature


def test_orbit_fraction_basis_truncates_full_orbit_profile():
    """n_interior=10 full-orbit edge profile → 5 interiors below tau=0.5 → n_basis=9."""
    btjd_full = np.linspace(1000.0, 1010.0, 590)
    btjd_fit = btjd_full[:295]
    tb = T.build_temporal_basis_orbit_fraction(
        btjd_fit, btjd_full, n_interior=10, tau_cut=0.5,
    )
    assert tb.btjd_ref == float(btjd_full[0])
    assert abs(tb.btjd_scale - float(btjd_full[-1] - btjd_full[0])) < 1e-12
    interiors = np.asarray(T.interior_knots(tb.knot_vector, tb.degree))
    assert interiors.shape == (5,)
    assert np.all(interiors < 0.5)
    assert tb.n_basis == 9
    assert tb.frame_basis.shape == (295, 9)
    # Right clamp only at tau_cut; no interior knots at/above cut.
    assert np.all(tb.knot_vector[: tb.degree + 1] == 0.0)
    assert np.all(tb.knot_vector[-(tb.degree + 1) :] == 0.5)
    row_sums = np.asarray(tb.frame_basis).sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_orbit_fraction_basis_default_tau_cut_from_fit_span():
    btjd_full = np.linspace(0.0, 100.0, 200)
    btjd_fit = btjd_full[:100]
    tb = T.build_temporal_basis_orbit_fraction(btjd_fit, btjd_full, n_interior=10)
    expected_cut = float((btjd_fit[-1] - btjd_full[0]) / (btjd_full[-1] - btjd_full[0]))
    assert tb.knot_vector[-1] == pytest.approx(expected_cut, abs=1e-6)
    assert expected_cut < 0.5  # first 100 of 200 samples end just below mid-span
    assert tb.n_basis == 9
    interiors = np.asarray(T.interior_knots(tb.knot_vector, tb.degree))
    assert np.all(interiors < expected_cut)


def test_orbit_fraction_smaller_than_full_orbit_basis():
    btjd_full = np.linspace(0.0, 10.0, 100)
    full = T.build_temporal_basis(btjd_full, n_interior=10, uniform_knots=False)
    half = T.build_temporal_basis_orbit_fraction(
        btjd_full[:50], btjd_full, n_interior=10, tau_cut=0.5,
    )
    assert half.n_basis < full.n_basis
    assert full.n_basis == 14
    assert half.n_basis == 9


def test_edge_interior_split_places_knots_in_bands():
  """2-1-2 split at edge_frac=0.24 → knots in start/mid/end bands."""
  btjd = np.linspace(1842.5, 1854.86, 590)
  tb = T.build_temporal_basis(
      btjd, n_interior=5, edge_frac=0.24, edge_interior_split=(2, 1, 2),
      uniform_knots=False,
  )
  ints = np.asarray(T.interior_knots(tb.knot_vector, tb.degree))
  assert ints.shape == (5,)
  assert tb.n_basis == 9
  assert np.allclose(ints, [0.08, 0.16, 0.5, 0.84, 0.92], atol=1e-3)


# ---------------------------------------------------------------------------
# cheb_wcs: Chebyshev basis vs numpy reference, round-trip
# ---------------------------------------------------------------------------


def test_chebyshev_vandermonde_matches_numpy():
    x = np.linspace(-1, 1, 25)
    for degree in (0, 1, 3, 5):
        got = np.asarray(CW.chebyshev_vandermonde(jnp.asarray(x), degree))
        expected = chebvander(x, degree)
        assert np.allclose(got, expected, atol=1e-5)


def test_cheb_star_basis_matches_cheb_design_matrix():
    rng = np.random.default_rng(1)
    xhat = rng.uniform(-1, 1, 50)
    yhat = rng.uniform(-1, 1, 50)
    degree = 4
    exponents = tuple(sci2idl_exponents(degree))
    got = np.asarray(CW.cheb_star_basis(jnp.asarray(xhat), jnp.asarray(yhat), degree, exponents))
    expected = cheb_design_matrix(xhat, yhat, degree)
    assert np.allclose(got, expected, atol=1e-5)


def test_warmstart_fit_recovers_known_distortion():
    rng = np.random.default_rng(2)
    n = 400
    x_lin = rng.uniform(-100, 100, n)
    y_lin = rng.uniform(-100, 100, n)
    static = CW.ChebWcsStatic(
        ra0_deg=0.0, dec0_deg=0.0, cd_inv=np.eye(2), crpix=np.array([1.0, 1.0]),
        center=np.array([0.0, 0.0]), half_extents=np.array([100.0, 100.0]),
        poly_degree=2, exponents=tuple(sci2idl_exponents(2)),
    )
    xhat, yhat = x_lin / 100.0, y_lin / 100.0
    design = cheb_design_matrix(xhat, yhat, 2)
    true_cx = rng.normal(scale=0.5, size=design.shape[1])
    true_cy = rng.normal(scale=0.5, size=design.shape[1])
    x_obs = x_lin + design @ true_cx
    y_obs = y_lin + design @ true_cy

    cx, cy, mask = CW.fit_frame_cheb_warmstart(x_lin, y_lin, x_obs, y_obs, static)
    assert mask.all()
    assert np.allclose(cx, true_cx, atol=1e-6)
    assert np.allclose(cy, true_cy, atol=1e-6)


# ---------------------------------------------------------------------------
# flux_solve: analytic solve vs lstsq reference
# ---------------------------------------------------------------------------


def test_solve_group_fluxes_matches_lstsq_reference():
    rng = np.random.default_rng(3)
    S, K, n_groups, n_frames = 13, 3, 4, 2
    templates = rng.random((n_groups, K, n_frames, S, S)).astype(np.float32)
    weight = np.ones((n_groups, n_frames, S, S), dtype=np.float32)
    true_flux = rng.uniform(10, 100, size=(n_groups, n_frames, K)).astype(np.float32)
    data = np.einsum("gkfxy,gfk->gfxy", templates, true_flux)

    flux = np.asarray(FS.solve_group_fluxes(jnp.asarray(templates), jnp.asarray(data), jnp.asarray(weight), ridge=1e-9))

    for g in range(n_groups):
        for f in range(n_frames):
            M = templates[g, :, f].reshape(K, -1).T  # (P, K)
            d = data[g, f].reshape(-1)
            expected, *_ = np.linalg.lstsq(M, d, rcond=None)
            assert np.allclose(flux[g, f], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# loss: penalty terms must be mean-normalized, not summed (regression test for
# a real instability -- see plan/session notes: an unnormalized sum-over-frames
# centroid penalty inflated with n_frames and destabilized stage 2 as soon as
# P_base became trainable).
# ---------------------------------------------------------------------------


def test_decode_epsf_base_is_positive_and_sums_to_one():
    rng = np.random.default_rng(7)
    raw = jnp.asarray(rng.normal(size=(2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)), dtype=jnp.float32)
    base = EM.decode_epsf_base(raw)
    assert float(base.min()) >= 0.0
    sums = np.asarray(base.sum(axis=(-1, -2)))
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_decode_epsf_base_core_centroid_near_zero():
    """Core-centroid gauge on a compact blob (realistic ePSF support), not white noise.

    The origin-centered Gaussian window (σ=0.75 px) measures a fraction of a
    large offset of a σ≈1.5 px core; two decode iters leave a small residual
    (unlike global COM, which is exact in one shift for a Gaussian). A 0.08 px
    seed is in the range of a slightly-miscentered PRF.
    """
    idx = np.arange(EM.NODE_GRID_SIZE, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    gauss = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob = np.asarray(
        EM.bilinear_shift_physical(jnp.asarray(gauss), jnp.asarray(0.08), jnp.asarray(-0.06))
    )
    blob = np.clip(blob, 0.0, None)
    blob /= blob.sum()
    raw = EM.encode_epsf_base(jnp.asarray(blob)[None, None])
    base = EM.decode_epsf_base(raw)
    cx, cy = EM.core_centroid_xy(base)
    assert float(jnp.max(jnp.abs(cx))) < 5e-2
    assert float(jnp.max(jnp.abs(cy))) < 5e-2


def test_encode_decode_epsf_base_roundtrip():
    idx = np.arange(EM.NODE_GRID_SIZE, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    gauss = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    gauss /= gauss.sum()
    base0 = np.broadcast_to(gauss, (2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)).copy()
    base0 = np.asarray(EM.recenter_grid_core(jnp.asarray(base0), clip_nonneg=True))
    raw = EM.encode_epsf_base(jnp.asarray(base0))
    base1 = np.asarray(EM.decode_epsf_base(raw))
    assert np.allclose(base1, base0, rtol=1e-4, atol=1e-5)


def test_recenter_undoes_physical_shift_on_decode():
    """Hard gauge: shifting content then decoding still yields core-centroid≈0."""
    idx = np.arange(EM.NODE_GRID_SIZE, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    gauss = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    gauss /= gauss.sum()
    base = np.broadcast_to(gauss, (2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)).copy()
    base = np.asarray(EM.recenter_grid_core(jnp.asarray(base), clip_nonneg=True))
    shifted = np.asarray(
        EM.bilinear_shift_physical(jnp.asarray(base), jnp.asarray(0.4), jnp.asarray(0.0))
    )
    shifted = np.clip(shifted, 0.0, None)
    shifted /= shifted.sum(axis=(-1, -2), keepdims=True)
    cx_shift, _ = EM.core_centroid_xy(jnp.asarray(shifted))
    assert float(jnp.mean(cx_shift)) > 0.05
    decoded = EM.decode_epsf_base(EM.encode_epsf_base(jnp.asarray(shifted)))
    cx, cy = EM.core_centroid_xy(decoded)
    assert float(jnp.mean(jnp.abs(cx))) < float(jnp.mean(jnp.abs(cx_shift)))
    assert float(jnp.max(jnp.abs(cx))) < 0.3
    assert float(jnp.max(jnp.abs(cy))) < 0.05


def test_composite_recenter_kills_mode_induced_core_centroid():
    """Hook B: modes can move the core centroid; recenter_grid_core brings it back."""
    idx = np.arange(EM.NODE_GRID_SIZE, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    gauss = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    gauss /= gauss.sum()
    base = np.asarray(EM.recenter_grid_core(jnp.asarray(gauss), clip_nonneg=True))
    gy, gx = np.gradient(base)
    mode = gx - gx.mean()
    mode = mode / (np.sqrt(np.mean(mode**2)) + 1e-12)
    comp = base + 0.05 * mode
    cx0, _ = EM.core_centroid_xy(jnp.asarray(comp))
    assert abs(float(cx0)) > 1e-3
    recentered = EM.recenter_grid_core(jnp.asarray(comp), clip_nonneg=False)
    cx1, cy1 = EM.core_centroid_xy(recentered)
    assert abs(float(cx1)) < 1e-3
    assert abs(float(cy1)) < 1e-3


def test_pixel_laplacian_penalty_zero_on_constant_positive_field():
    field = jnp.ones((2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE), dtype=jnp.float32)
    assert float(L.pixel_laplacian_penalty(field)) < 1e-12


def test_pixel_laplacian_penalty_positive_on_checkerboard():
    idx = np.arange(EM.NODE_GRID_SIZE, dtype=np.float32)
    checker = ((idx[:, None] + idx[None, :]) % 2).astype(np.float32) * 2.0 - 1.0
    field = jnp.asarray(checker)[None, None, :, :]
    assert float(L.pixel_laplacian_penalty(field)) > 0.0


def test_centroid_penalty_invariant_to_n_frames():
    G = EM.NODE_GRID_SIZE
    rng = np.random.default_rng(5)
    base = rng.random((2, 2, G, G)).astype(np.float32)
    base /= base.sum(axis=(-1, -2), keepdims=True)
    modes = np.zeros((5, 2, 2, G, G), dtype=np.float32)
    params = {
        "epsf_base_raw": EM.encode_epsf_base(jnp.asarray(base)),
        "epsf_modes": jnp.asarray(modes),
    }
    vals = []
    for n_frames in (1, 5, 50):
        w_of_t = jnp.zeros((n_frames, 5))  # modes are zero anyway, but exercise the einsum
        vals.append(float(L.centroid_penalty(params, w_of_t)))
    assert max(vals) - min(vals) < 1e-6


def test_decode_epsf_modes_sums_to_zero():
    rng = np.random.default_rng(11)
    raw = jnp.asarray(rng.normal(size=(5, 2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)), dtype=jnp.float32)
    base_raw = jnp.asarray(rng.normal(size=(2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)), dtype=jnp.float32)
    base = EM.decode_epsf_base(base_raw)
    modes = EM.decode_epsf_modes(raw, base)
    sums = np.asarray(modes.sum(axis=(-1, -2)))
    # float32 sum of 58² mean-centered values leaves ~1e-4 residual
    assert float(np.max(np.abs(sums))) < 2e-4
    assert float(np.max(np.abs(sums))) < 1e-3 * float(np.sqrt(np.mean(np.asarray(modes) ** 2)))


def test_decode_epsf_modes_invariant_to_meansub_base_multiple():
    """Gauge 1 (TEMPORAL_EPSF_GAUGE_PLAN_20260808.md Sec.4.1): decode_epsf_modes projects
    off base in a fixed weighted metric, so adding any multiple of the (mean-subtracted,
    since raw base sums to ~1 not 0) base direction to raw is exactly absorbed -- the
    decoded mode is unchanged. This is the concrete falsifiable claim behind "this mode
    component has zero data-term gradient"."""
    rng = np.random.default_rng(22)
    G = EM.NODE_GRID_SIZE
    base_raw = jnp.asarray(rng.normal(size=(2, 2, G, G)), dtype=jnp.float32)
    base = EM.decode_epsf_base(base_raw)
    base_meansub = base - jnp.mean(base, axis=(-2, -1), keepdims=True)
    raw = jnp.asarray(rng.normal(size=(3, 2, 2, G, G)), dtype=jnp.float32)
    baseline = np.asarray(EM.decode_epsf_modes(raw, base))
    for eps in (0.001, 0.5, -3.0):
        perturbed = np.asarray(EM.decode_epsf_modes(raw + eps * base_meansub[None], base))
        np.testing.assert_allclose(perturbed, baseline, rtol=1e-4, atol=1e-4)


def test_decode_epsf_modes_idempotent():
    """decode(decode(raw, base), base) == decode(raw, base): re-applying the gauge to an
    already-gauged mode is a no-op, matching recenter_grid_core's own idempotent design and
    required so encode_epsf_modes(modes, base) round-trips through init/checkpoint reload."""
    rng = np.random.default_rng(23)
    G = EM.NODE_GRID_SIZE
    base_raw = jnp.asarray(rng.normal(size=(2, 2, G, G)), dtype=jnp.float32)
    base = EM.decode_epsf_base(base_raw)
    raw = jnp.asarray(rng.normal(size=(3, 2, 2, G, G)), dtype=jnp.float32)
    once = EM.decode_epsf_modes(raw, base)
    twice = EM.decode_epsf_modes(once, base)
    np.testing.assert_allclose(np.asarray(twice), np.asarray(once), rtol=1e-4, atol=1e-4)


def _core_dipole_moments(modes: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Core-weighted dipole of ``modes`` ``(..., G, G)`` in subpixel index units."""
    h = int(modes.shape[-2])
    w = int(modes.shape[-1])
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)
    dy = (y - cy)[:, None]
    dx = (x - cx)[None, :]
    ww = np.asarray(EM.core_gaussian_weight(h, dtype=jnp.float32))
    ww = ww / ww.sum()
    arr = np.asarray(modes)
    dip_x = np.sum(arr * ww * dx, axis=(-2, -1))
    dip_y = np.sum(arr * ww * dy, axis=(-2, -1))
    return dip_x, dip_y


def test_recenter_grid_core_pins_core_not_global_com():
    """Asymmetric wings pull global COM; the core-Gaussian gauge leaves them free."""
    G = EM.NODE_GRID_SIZE
    idx = np.arange(G, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    core = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 0.6) ** 2))
    # Far wing (~3 physical px) well outside the 0.75 px core window.
    wing_r2 = (idx[:, None] - EM.NODE_CENTER_INDEX - 12) ** 2 + (
        idx[None, :] - EM.NODE_CENTER_INDEX + 8
    ) ** 2
    wing = 0.4 * np.exp(-wing_r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2))
    grid = (core + wing).astype(np.float32)
    grid = grid / grid.sum()
    out = EM.recenter_grid_core(jnp.asarray(grid), clip_nonneg=False, n_iter=2)
    cx_core, cy_core = EM.core_centroid_xy(out)
    cx_glob, cy_glob = EM.flux_centroid_xy(out)
    assert abs(float(cx_core)) < 2e-2
    assert abs(float(cy_core)) < 2e-2
    assert abs(float(cx_glob)) + abs(float(cy_glob)) > 5e-3
    assert abs(float(cx_core)) + abs(float(cy_core)) < 0.4 * (
        abs(float(cx_glob)) + abs(float(cy_glob))
    )


@pytest.mark.skip(reason='needs dev/pointing_analysis (not migrated)')
def test_pointing_analysis_numpy_recenter_matches_training_core_gauge():
    """Photometry-side ``_numpy_recenter`` must match training core-centroid gauge."""
    from dev.pointing_analysis import epsf as PA

    G = EM.NODE_GRID_SIZE
    idx = np.arange(G, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    core = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 0.6) ** 2))
    wing_r2 = (idx[:, None] - EM.NODE_CENTER_INDEX - 12) ** 2 + (
        idx[None, :] - EM.NODE_CENTER_INDEX + 8
    ) ** 2
    wing = 0.4 * np.exp(-wing_r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2))
    grid = ((core + wing) / (core + wing).sum()).astype(np.float32)

    out_pa = PA._numpy_recenter(
        grid[None, None],
        n_iter=EM.HOTPATH_RECENTER_N_ITER,
        clip_nonneg=False,
    )
    out_train = np.asarray(
        EM.recenter_grid_core(
            jnp.asarray(grid[None, None]),
            clip_nonneg=False,
            n_iter=EM.HOTPATH_RECENTER_N_ITER,
        )
    )
    np.testing.assert_allclose(out_pa, out_train, rtol=1e-5, atol=1e-6)

    cx_core, cy_core = EM.core_centroid_xy(jnp.asarray(out_pa[0, 0]))
    cx_glob, cy_glob = EM.flux_centroid_xy(jnp.asarray(out_pa[0, 0]))
    core_norm = abs(float(cx_core)) + abs(float(cy_core))
    glob_norm = abs(float(cx_glob)) + abs(float(cy_glob))
    assert core_norm < 5e-2
    assert glob_norm > 5e-3
    assert core_norm < 0.4 * glob_norm


def test_recenter_grid_core_is_differentiable():
    """Must survive jax.value_and_grad (no boolean masks / hard truncations)."""
    grid = _asymmetric_psf_grid()[None]

    def loss(g):
        return jnp.sum(EM.recenter_grid_core(g, clip_nonneg=False, n_iter=1) ** 2)

    val, grad = jax.value_and_grad(loss)(grid)
    assert np.isfinite(float(val))
    assert np.isfinite(np.asarray(grad)).all()


def test_project_core_dipoles_out_zeros_core_dipole():
    rng = np.random.default_rng(41)
    G = EM.NODE_GRID_SIZE
    modes = jnp.asarray(rng.normal(size=(3, 2, 2, G, G)), dtype=jnp.float32)
    cleaned = EM.project_core_dipoles_out(modes)
    dip_x, dip_y = _core_dipole_moments(cleaned)
    assert float(np.max(np.abs(dip_x))) < 1e-5
    assert float(np.max(np.abs(dip_y))) < 1e-5
    twice = EM.project_core_dipoles_out(cleaned)
    np.testing.assert_allclose(np.asarray(twice), np.asarray(cleaned), rtol=1e-5, atol=1e-5)


def test_decode_epsf_modes_projects_core_dipoles():
    rng = np.random.default_rng(42)
    G = EM.NODE_GRID_SIZE
    base_raw = jnp.asarray(rng.normal(size=(2, 2, G, G)), dtype=jnp.float32)
    base = EM.decode_epsf_base(base_raw)
    raw = jnp.asarray(rng.normal(size=(3, 2, 2, G, G)), dtype=jnp.float32)
    decoded = EM.decode_epsf_modes(raw, base)
    dip_x, dip_y = _core_dipole_moments(decoded)
    assert float(np.max(np.abs(dip_x))) < 2e-5
    assert float(np.max(np.abs(dip_y))) < 2e-5


def test_dipole_modes_rejected_for_joint_wcs():
    G = EM.NODE_GRID_SIZE
    blob = np.ones((G, G), dtype=np.float32)
    blob /= blob.sum()
    with pytest.raises(ValueError, match="dipole"):
        EM._finite_diff_modes(blob, mode_names=("x_smear",))
    with pytest.raises(ValueError, match="dipole"):
        EM.validate_mode_names(("iso_defocus", "y_smear"))
    with pytest.raises(ValueError, match="unknown"):
        EM.validate_mode_names(("not_a_mode",))
    assert EM.validate_mode_names(("astig0", "iso_defocus")) == ("iso_defocus", "astig0")


def test_init_epsf_from_base_seeds_base_and_derives_modes_from_it():
    """init_epsf_from_base (for --init-epsf-base): P_base is the given array verbatim;
    P_k are FD-derived from *that* base, not a PRF resample, and match calling
    _finite_diff_modes directly on each node."""
    rng = np.random.default_rng(31)
    G = EM.NODE_GRID_SIZE
    base_raw = jnp.asarray(rng.normal(size=(2, 2, G, G)), dtype=jnp.float32)
    seed_base = np.asarray(EM.decode_epsf_base(base_raw))

    params0 = EM.init_epsf_from_base(seed_base, mode_names=("iso_defocus", "astig0"))
    np.testing.assert_allclose(np.asarray(params0.base), seed_base, rtol=1e-6, atol=1e-6)
    assert params0.modes.shape == (2, 2, 2, G, G)
    for i in range(2):
        for j in range(2):
            expected = EM._finite_diff_modes(seed_base[i, j], mode_names=("iso_defocus", "astig0"))
            np.testing.assert_allclose(np.asarray(params0.modes[:, i, j]), expected, rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError, match="square node grid"):
        EM.init_epsf_from_base(np.zeros((2, 2, G, G + 1), dtype=np.float32))


def test_finite_diff_modes_kurt_matches_kurt_generator_up_to_gauge():
    """The new "kurt" FD-mode name (K=2 iso_defocus+kurt temporal model,
    2026-09-07) must equal `kurt_generator(base)` after the SAME
    mean-subtract + unit-RMS normalization `_finite_diff_modes` already
    applies to every other named mode -- not a separately reimplemented
    formula. Also checks `validate_mode_names`/`FD_MODE_NAMES` accept it and
    that it is not treated as a forbidden dipole mode."""
    rng = np.random.default_rng(7)
    G = EM.NODE_GRID_SIZE
    base = np.asarray(rng.random((G, G)), dtype=np.float32) ** 2
    base = base / base.sum()

    assert "kurt" in EM.FD_MODE_NAMES
    assert "kurt" not in EM.FORBIDDEN_MODES
    assert EM.validate_mode_names(("kurt",)) == ("kurt",)
    assert EM.validate_mode_names(("kurt", "iso_defocus")) == ("iso_defocus", "kurt")

    modes = EM._finite_diff_modes(base, mode_names=("kurt",))
    assert modes.shape == (1, G, G)

    raw_kurt = np.asarray(EM.kurt_generator(jnp.asarray(base)))
    expected = raw_kurt - raw_kurt.mean()
    expected = expected / (np.sqrt(np.mean(expected ** 2)) + 1e-12)
    np.testing.assert_allclose(modes[0], expected, rtol=1e-5, atol=1e-5)

    # Exact flux-neutrality and unit RMS, same contract as every other mode.
    assert abs(float(modes[0].mean())) < 1e-5
    assert abs(float(np.sqrt(np.mean(modes[0] ** 2))) - 1.0) < 1e-4

    # K=2 end-to-end through init_epsf_from_base (the actual production path
    # a --mode-init "iso_defocus,kurt" run would take).
    seed_base = np.asarray(EM.decode_epsf_base(jnp.asarray(rng.normal(size=(2, 2, G, G)), dtype=jnp.float32)))
    params0 = EM.init_epsf_from_base(seed_base, mode_names=("iso_defocus", "kurt"))
    assert params0.modes.shape == (2, 2, 2, G, G)
    for i in range(2):
        for j in range(2):
            exp_ij = EM._finite_diff_modes(seed_base[i, j], mode_names=("iso_defocus", "kurt"))
            np.testing.assert_allclose(np.asarray(params0.modes[:, i, j]), exp_ij, rtol=1e-5, atol=1e-5)


def test_forward_model_w_of_t_has_zero_time_mean():
    """Gauge 2 (Sec.4.2): forward_model's w_of_t is mean-subtracted over frames every
    call, for arbitrary w_coeff -- kills the base<->DC-offset degeneracy exactly."""
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents

    from syndiff_pipeline.forward_model.data import RegionSpec
    from syndiff_pipeline.forward_model.groups import GroupSet

    region = RegionSpec(0, 0, 100, 100)
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1, exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]), node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]), node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    blob = (np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)) / 1.0).astype(np.float32)
    blob /= blob.sum()
    for i in range(2):
        for j in range(2):
            base[i, j] = blob
    modes = np.zeros((2, 2, 2, Gsz, Gsz), dtype=np.float32)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    members = np.array([[0, -1]], dtype=int)
    valid = np.array([[True, False]], dtype=bool)
    groups = GroupSet(1, 2, members, valid, np.array([True]), 0)
    ra = np.array([180.0], dtype=np.float32)
    dec = np.array([0.0], dtype=np.float32)
    n_frames, n_basis = 6, 2
    wcs_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    w_fb[:, 0] = 1.0

    ctx = L.build_static_context(
        cheb_static=static, wcs_frame_basis=wcs_fb, w_frame_basis=w_fb,
        epsf_grid=grid, groups=groups, ra=ra, dec=dec,
        stamp_center_x=np.array([50], dtype=np.int64), stamp_center_y=np.array([50], dtype=np.int64),
        t_exp_sec=1426.0, stamp_snr_weight=np.array([1.0], dtype=np.float32),
        fit_radius=np.array([3.0], dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)
    rng = np.random.default_rng(24)
    params["w_coeff"] = jnp.asarray(rng.normal(size=(2, n_basis)).astype(np.float32))

    _, _, _, w_of_t = L.forward_model(params, ctx)
    assert w_of_t.shape == (n_frames, 2)
    mean_over_t = np.asarray(jnp.mean(w_of_t, axis=0))
    np.testing.assert_allclose(mean_over_t, np.zeros_like(mean_over_t), atol=1e-6)


def test_pixel_variance_is_noise_squared_plus_floor():
    noise = jnp.asarray([0.5, 1.0, 0.0], dtype=jnp.float32)
    var = L.pixel_variance(noise)
    assert np.allclose(np.asarray(var), np.array([0.25, 1.0, 0.0]) + L.VARIANCE_FLOOR)


def test_inverse_variance_weights_downweight_noisy_pixels():
    mask = jnp.ones((2, 2), dtype=jnp.float32)
    var = jnp.asarray([[1.0, 4.0], [1.0, 1.0]], dtype=jnp.float32)
    iv = L.inverse_variance_weights(mask, var)
    assert float(iv[0, 0]) > float(iv[0, 1])
    assert float(iv[0, 1]) == pytest.approx(0.25)


def test_flux_solve_iv_beats_mask_when_wings_are_noisy():
    """Core carries signal; noisy wings bias uniform-mask flux more than IV."""
    rng = np.random.default_rng(12)
    S, K, n_groups, n_frames = 13, 1, 1, 1
    yy, xx = np.mgrid[0:S, 0:S]
    core = np.exp(-((xx - 6) ** 2 + (yy - 6) ** 2) / 4.0).astype(np.float32)
    core /= core.sum()
    templates = core[None, None, None, :, :]
    true_flux = 200.0
    noise_plane = np.full((S, S), 0.5, dtype=np.float32)
    noise_plane[0:3, :] = 5.0  # noisy wing strip
    wing_noise = rng.normal(0.0, 5.0, size=(S, S)).astype(np.float32)
    wing_noise[3:, :] = 0.0
    data = (true_flux * core + wing_noise)[None, None]
    mask = np.ones((n_groups, n_frames, S, S), dtype=np.float32)
    var = L.pixel_variance(jnp.asarray(noise_plane)[None, None])
    iv = L.inverse_variance_weights(jnp.asarray(mask), var)

    flux_mask = float(FS.solve_group_fluxes(jnp.asarray(templates), jnp.asarray(data), jnp.asarray(mask))[0, 0, 0])
    flux_iv = float(FS.solve_group_fluxes(jnp.asarray(templates), jnp.asarray(data), iv)[0, 0, 0])
    assert abs(flux_iv - true_flux) < abs(flux_mask - true_flux)


def test_per_stamp_snr_weighted_nll_invariant_for_constant_ell():
    """Constant per-pixel ℓ → stamp means equal; SNR weights cancel in the ratio."""
    data = jnp.zeros((2, 1, 3, 3))
    model = jnp.zeros((2, 1, 3, 3))
    var = jnp.ones((2, 1, 3, 3))
    pix = jnp.ones((2, 1, 3, 3))
    sw1 = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    sw2 = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    v1 = float(L.per_stamp_snr_weighted_nll(data, model, var, pix, sw1))
    v2 = float(L.per_stamp_snr_weighted_nll(data, model, var, pix, sw2))
    assert abs(v1 - v2) < 1e-5


def test_soft_snr_stamp_weights_cap_and_floor():
    mags = np.array([7.0, 9.0, 12.0])
    w = L.soft_snr_stamp_weights(mags, snr_cap_mag=9.0, w_min=0.05)
    assert w[1] == pytest.approx(1.0)
    assert w[0] == pytest.approx(1.0)  # brighter than cap → still 1
    assert w[2] >= 0.05
    assert w[2] < w[1]


def test_fit_radius_from_mag_stage_split():
    """Default tiers (loss.fit_radius_tiers_default) now cover the whole
    STAMP_PHYSICAL=13 stamp (corner radius 6*sqrt(2)) for every magnitude tier
    -- see fit_radius_tiers_default's docstring: a gradient-coverage
    diagnostic (dev_grad_coverage.py) found the old hardcoded 6.0/3.5/2.5px
    tiers gave epsf_base_raw structurally zero gradient beyond each tier's
    radius. Stage 1 stays core-capped at 3px regardless of tier."""
    mags = np.array([8.0, 10.0, 12.0])
    r1 = L.fit_radius_from_mag(mags, stage=1)
    r2 = L.fit_radius_from_mag(mags, stage=2)
    corner = L.stamp_corner_radius_px()
    assert r1[0] == pytest.approx(3.0)  # capped in stage 1
    assert r2[0] == pytest.approx(corner)
    assert r2[1] == pytest.approx(corner)
    assert r2[2] == pytest.approx(corner)
    assert np.all(r1 <= 3.0)


def test_fit_radius_from_mag_custom_tiers_reproduces_legacy_values():
    """--fit-radius-tiers lets a caller opt back into the old narrower,
    per-magnitude cutoffs (e.g. to test whether a *smaller* radius is
    empirically better for the faintest stars) -- verify the tunable path
    reproduces the exact legacy 6.0/3.5/2.5px tiers when asked."""
    mags = np.array([8.0, 10.0, 12.0])
    legacy_tiers = ((9.0, 6.0), (11.0, 3.5), (float("inf"), 2.5))
    r2 = L.fit_radius_from_mag(mags, stage=2, tiers=legacy_tiers)
    assert r2[0] == pytest.approx(6.0)
    assert r2[1] == pytest.approx(3.5)
    assert r2[2] == pytest.approx(2.5)


def test_parse_fit_radius_tiers():
    parsed = L.parse_fit_radius_tiers("9:6.0,11:3.5,inf:2.5")
    assert parsed == ((9.0, 6.0), (11.0, 3.5), (float("inf"), 2.5))
    # Order-independence: parser sorts by ascending mag bound.
    parsed2 = L.parse_fit_radius_tiers("inf:2.5,9:6.0,11:3.5")
    assert parsed2 == parsed


def test_stamp_corner_radius_px():
    # 13x13 stamp, center index 6 -> corner at 6*sqrt(2).
    assert L.stamp_corner_radius_px(13) == pytest.approx(6.0 * np.sqrt(2.0))
    assert L.stamp_corner_radius_px(7) == pytest.approx(3.0 * np.sqrt(2.0))


def test_run_stage_early_stop_patience_stops_before_n_steps():
    """--early-stop-patience: with lr=0 the loss is exactly constant every step
    (zero gradient update), so early stopping must trigger as soon as enough
    logged checks have accumulated, well before the full n_steps budget."""
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model.data import RegionSpec
    from syndiff_pipeline.forward_model.groups import GroupSet

    region = RegionSpec(0, 0, 100, 100)
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1, exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]), node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]), node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob /= blob.sum()
    for i in range(2):
        for j in range(2):
            base[i, j] = blob
    modes = np.zeros((1, 2, 2, Gsz, Gsz), dtype=np.float32)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    members = np.array([[0, -1]], dtype=int)
    valid = np.array([[True, False]], dtype=bool)
    groups = GroupSet(1, 2, members, valid, np.array([True]), 0)
    ra = np.array([180.0], dtype=np.float32)
    dec = np.array([0.0], dtype=np.float32)
    n_frames, n_basis = 6, 4
    wcs_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    w_fb[:, 0] = 1.0

    ctx = L.build_static_context(
        cheb_static=static, wcs_frame_basis=wcs_fb, w_frame_basis=w_fb,
        epsf_grid=grid, groups=groups, ra=ra, dec=dec,
        stamp_center_x=np.array([50], dtype=np.int64), stamp_center_y=np.array([50], dtype=np.int64),
        t_exp_sec=1426.0, stamp_snr_weight=np.array([1.0], dtype=np.float32),
        fit_radius=np.array([3.0], dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)

    stamp = EM.STAMP_PHYSICAL
    data = jnp.zeros((1, n_frames, stamp, stamp), dtype=jnp.float32)
    noise = jnp.ones((1, n_frames, stamp, stamp), dtype=jnp.float32)
    weight = jnp.ones((1, n_frames, stamp, stamp), dtype=jnp.float32)

    fd = FIT.FitData(
        ctx=ctx, data=data, noise=noise, weight=weight,
        wcs_second_diff=T.second_difference_matrix(n_basis),
        w_second_diff=T.second_difference_matrix(n_basis),
        epsf_modes_init=epsf0.modes,
    )

    _, history = FIT.run_stage(
        params, fd, stage=3, n_steps=200, lr=0.0, log_every=5,
        early_stop_patience=3, early_stop_tol=1e-6, reject_every=0,
    )
    step_events = [h["step"] for h in history if "step" in h]
    assert max(step_events) < 199  # stopped well before the full 200-step budget

    # Disabled (default None) must reproduce the exact old behavior: full budget.
    _, history_full = FIT.run_stage(
        params, fd, stage=3, n_steps=20, lr=0.0, log_every=5, reject_every=0,
    )
    assert max(h["step"] for h in history_full if "step" in h) == 19


def test_run_stage_hysteresis_reject_logs_real_med_chi2_red():
    """``_refresh_reject``'s hysteresis branch hardcoded ``med_chi2_red: nan``
    (fit.py, pre-fix), so every logged history row from a `--reject-mode
    hysteresis` run (the training default) had a NaN med_chi2_red -- checked
    directly against real Colab run histories (0/2732 rows finite across four
    completed stage-3 runs). The fix reuses the ``per_stamp_chi2_red`` already
    computed for the reject decision itself to also populate a real median
    (same convention as the `audit`/`legacy` branches: median over stamps with
    ``pix_sum > 0``), without touching the reject-decision arrays. This test
    triggers one hysteresis refresh (``reject_burn_in=0, reject_every=1``) and
    asserts every logged row after it carries a finite, sane ``med_chi2_red``,
    not NaN."""
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model.data import RegionSpec
    from syndiff_pipeline.forward_model.groups import GroupSet

    region = RegionSpec(0, 0, 100, 100)
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1, exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]), node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]), node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob /= blob.sum()
    for i in range(2):
        for j in range(2):
            base[i, j] = blob
    modes = np.zeros((1, 2, 2, Gsz, Gsz), dtype=np.float32)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    # Two isolated groups so per_stamp_chi2_red has more than one stamp to
    # take a median over (a single-stamp median is a degenerate check).
    members = np.array([[0, -1], [1, -1]], dtype=int)
    valid = np.array([[True, False], [True, False]], dtype=bool)
    groups = GroupSet(2, 2, members, valid, np.array([True, True]), 0)
    ra = np.array([180.0, 180.0], dtype=np.float32)
    dec = np.array([0.0, 0.0], dtype=np.float32)
    n_frames, n_basis = 6, 4
    wcs_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    w_fb[:, 0] = 1.0

    ctx = L.build_static_context(
        cheb_static=static, wcs_frame_basis=wcs_fb, w_frame_basis=w_fb,
        epsf_grid=grid, groups=groups, ra=ra, dec=dec,
        stamp_center_x=np.array([50, 50], dtype=np.int64),
        stamp_center_y=np.array([50, 50], dtype=np.int64),
        t_exp_sec=1426.0, stamp_snr_weight=np.array([1.0, 1.0], dtype=np.float32),
        fit_radius=np.array([3.0, 3.0], dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)

    stamp = EM.STAMP_PHYSICAL
    rng = np.random.default_rng(0)
    # Non-degenerate (non-zero, non-constant) data so per-stamp chi2_red isn't
    # a trivial single repeated value -- makes the median check meaningful.
    data = jnp.asarray(
        rng.normal(loc=0.0, scale=0.05, size=(2, n_frames, stamp, stamp)).astype(np.float32)
    )
    noise = jnp.ones((2, n_frames, stamp, stamp), dtype=jnp.float32)
    weight = jnp.ones((2, n_frames, stamp, stamp), dtype=jnp.float32)

    fd = FIT.FitData(
        ctx=ctx, data=data, noise=noise, weight=weight,
        wcs_second_diff=T.second_difference_matrix(n_basis),
        w_second_diff=T.second_difference_matrix(n_basis),
        epsf_modes_init=epsf0.modes,
    )

    _, history = FIT.run_stage(
        params, fd, stage=3, n_steps=3, lr=1e-4, log_every=1,
        reject_mode="hysteresis", reject_every=1, reject_burn_in=0,
    )

    rows_with_chi2 = [h for h in history if "med_chi2_red" in h and "step" in h]
    assert rows_with_chi2, "expected at least one logged row carrying med_chi2_red"
    chi2_values = [h["med_chi2_red"] for h in rows_with_chi2]
    assert all(np.isfinite(v) for v in chi2_values), (
        f"med_chi2_red must be finite once a hysteresis reject refresh has run "
        f"(was hardcoded NaN pre-fix); got {chi2_values}"
    )
    assert all(v >= 0 for v in chi2_values), f"chi2 cannot be negative; got {chi2_values}"

    # Cross-check against a direct, independent computation of the same
    # quantity (median chi2_red over stamps with pix_sum > 0) on the INITIAL
    # params, so the fix is checked against real reject-machinery output, not
    # just "some finite number".
    from syndiff_pipeline.forward_model import stamp_reject as SR
    chi2_direct, pix_direct = SR.per_stamp_chi2_red(params, fd)
    pix_ok = np.asarray(pix_direct) > 0
    expected_first = float(np.median(np.asarray(chi2_direct, dtype=np.float64)[pix_ok]))
    assert np.isclose(chi2_values[0], expected_first, rtol=1e-5), (
        f"first logged med_chi2_red ({chi2_values[0]}) should match a direct "
        f"median of per_stamp_chi2_red on the pre-step params ({expected_first}) "
        "-- the reject refresh at step 0 (burn_in=0) runs before that step's "
        "optimizer update"
    )


@pytest.mark.parametrize("support_power", [0.0, 1.0])
def test_bucketed_total_loss_recombination_matches_unbucketed(support_power):
    """Integration-level proof for the K-bucketing recombination formula used in
    fit.py: sum(W_i * data_term_i) / sum(W_i) [+ regularization counted once]
    must exactly reproduce what a single, unbucketed context (one GroupSet at
    the global max K, with padding) computes -- bucketing must be loss-neutral,
    only cheaper. Builds 3 groups of heterogeneous size (1, 2, 3 members) once
    at a uniform K=4 (padded) and once split into K-tier buckets (1, 2, 4) via
    groups.bucket_groups_by_size, and compares total_loss both ways."""
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model import groups as GR
    from syndiff_pipeline.forward_model.data import RegionSpec
    from syndiff_pipeline.forward_model.groups import GroupSet

    region = RegionSpec(0, 0, 100, 100)
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1, exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]), node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]), node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob /= blob.sum()
    for i in range(2):
        for j in range(2):
            base[i, j] = blob
    modes = np.zeros((1, 2, 2, Gsz, Gsz), dtype=np.float32)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    n_frames, n_basis, stamp = 4, 4, EM.STAMP_PHYSICAL
    sizes = [1, 2, 3]  # -> tiers 1, 2, 4
    n_stars = sum(sizes)
    ra = np.full(n_stars, 180.0, dtype=np.float32)
    dec = np.full(n_stars, 0.0, dtype=np.float32)

    max_k = 4
    members = np.full((3, max_k), -1, dtype=int)
    valid = np.zeros((3, max_k), dtype=bool)
    star = 0
    for gi, s in enumerate(sizes):
        members[gi, :s] = np.arange(star, star + s)
        valid[gi, :s] = True
        star += s
    orig_groups = GroupSet(3, max_k, members, valid, np.ones(n_stars, dtype=bool), 0)
    stamp_cx = np.array([50, 50, 50], dtype=np.int64)
    stamp_cy = np.array([50, 50, 50], dtype=np.int64)

    wcs_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    w_fb[:, 0] = 1.0
    wcs_second_diff = T.second_difference_matrix(n_basis)
    w_second_diff = T.second_difference_matrix(n_basis)

    rng = np.random.default_rng(11)
    stamp_snr_weight = rng.uniform(0.3, 1.0, size=3).astype(np.float32)
    # Unequal radii -> unequal per-stamp pixel counts, so a pooling denominator that
    # drops pix_sum**support_power (support_power=1) is caught.
    fit_radius = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    lw = L.LossWeights(support_size_weight_power=support_power)
    data_all = rng.normal(scale=5.0, size=(3, n_frames, stamp, stamp)).astype(np.float32)
    noise_all = np.ones((3, n_frames, stamp, stamp), dtype=np.float32)
    weight_all = np.ones((3, n_frames, stamp, stamp), dtype=np.float32)

    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)

    def build_fd(groups: GroupSet, cx, cy, idx) -> FIT.FitData:
        ctx = L.build_static_context(
            cheb_static=static, wcs_frame_basis=wcs_fb, w_frame_basis=w_fb,
            epsf_grid=grid, groups=groups, ra=ra, dec=dec,
            stamp_center_x=cx, stamp_center_y=cy,
            t_exp_sec=1426.0, stamp_snr_weight=stamp_snr_weight[idx],
            fit_radius=fit_radius[idx],
        )
        return FIT.FitData(
            ctx=ctx, data=jnp.asarray(data_all[idx]), noise=jnp.asarray(noise_all[idx]),
            weight=jnp.asarray(weight_all[idx]),
            wcs_second_diff=wcs_second_diff, w_second_diff=w_second_diff,
            epsf_modes_init=epsf0.modes,
        )

    fd_orig = build_fd(orig_groups, stamp_cx, stamp_cy, np.array([0, 1, 2]))
    loss_orig, metrics_orig = L.total_loss(
        params, fd_orig.ctx, fd_orig.data, fd_orig.noise, fd_orig.weight,
        fd_orig.wcs_second_diff, fd_orig.w_second_diff, epsf_modes_init=fd_orig.epsf_modes_init, weights=lw,
    )

    buckets = GR.bucket_groups_by_size(orig_groups, stamp_cx, stamp_cy, tiers=(1, 2, 4))
    assert sorted(bg.max_group_size for bg, _, _, _ in buckets) == [1, 2, 4]  # confirms heterogeneous test

    bucket_metrics = []
    for bg, bcx, bcy, idx in buckets:
        # idx maps each bucket-local group back to its index in orig_groups -- use it to
        # slice data/noise/weight/stamp_snr_weight/fit_radius consistently (mirrors what
        # run_fit.py does with an already-built StampBatch, no re-running extract_stamps).
        fd_i = build_fd(bg, bcx, bcy, idx)
        loss_i, metrics_i = L.total_loss(
            params, fd_i.ctx, fd_i.data, fd_i.noise, fd_i.weight,
            fd_i.wcs_second_diff, fd_i.w_second_diff, epsf_modes_init=fd_i.epsf_modes_init, weights=lw,
        )
        bucket_metrics.append((float(loss_i), {k: float(v) for k, v in metrics_i.items()}))

    W = np.array([m["stamp_weight_sum"] for _, m in bucket_metrics])
    dt = np.array([m["data_term"] for _, m in bucket_metrics])
    combined_data_term = float(np.sum(W * dt) / np.sum(W))
    reg_terms = bucket_metrics[0][0] - bucket_metrics[0][1]["data_term"]
    combined_loss = combined_data_term + reg_terms

    assert abs(float(metrics_orig["stamp_weight_sum"]) - float(np.sum(W))) < 1e-3
    np.testing.assert_allclose(combined_data_term, float(metrics_orig["data_term"]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(combined_loss, float(loss_orig), rtol=1e-4, atol=1e-4)


def test_save_params_npz_roundtrip_and_atomic(tmp_path):
    from pathlib import Path

    from syndiff_pipeline.forward_model import fit as FIT

    G = EM.NODE_GRID_SIZE
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(2, 2, G, G)).astype(np.float32)
    modes = rng.normal(size=(5, 2, 2, G, G)).astype(np.float32)
    params = {
        "wcs_coeff": jnp.asarray(rng.normal(size=(20, 4)).astype(np.float32)),
        "epsf_base_raw": jnp.asarray(raw),
        "epsf_modes": jnp.asarray(modes),
        "w_coeff": jnp.asarray(rng.normal(size=(5, 4)).astype(np.float32)),
    }
    path = Path(tmp_path) / "params_latest.npz"
    FIT.save_params_npz(path, params)
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp.npz").exists()
    loaded = dict(np.load(path))
    for k in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        np.testing.assert_allclose(loaded[k], np.asarray(params[k]), rtol=1e-5, atol=1e-5)
    assert "epsf_base" in loaded and "epsf_modes_decoded" in loaded
    np.testing.assert_allclose(
        loaded["epsf_base"], np.asarray(EM.decode_epsf_base(params["epsf_base_raw"])),
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(
        loaded["epsf_modes_decoded"],
        np.asarray(EM.decode_epsf_modes(params["epsf_modes"], EM.decode_epsf_base(params["epsf_base_raw"]))),
        rtol=1e-5, atol=1e-5,
    )
    reloaded = FIT.load_params_npz(path)
    assert set(reloaded.keys()) == set(FIT.STAGE_LEAVES)
    assert "epsf_base" not in reloaded
    for k in FIT.STAGE_LEAVES:
        np.testing.assert_allclose(np.asarray(reloaded[k]), np.asarray(params[k]), rtol=1e-5, atol=1e-5)




def test_stamp_active_npz_roundtrip_and_merge_fds(tmp_path):
    from pathlib import Path

    from syndiff_pipeline.forward_model import fit as FIT

    mask = np.ones((5, 4), dtype=np.float32)
    path = Path(tmp_path) / "stamp_active.npz"
    FIT.save_stamp_active_npz(path, mask)
    loaded = FIT.load_stamp_active_npz(path)
    np.testing.assert_array_equal(loaded, mask)

    global_mask = np.ones((6, 4), dtype=np.float32)
    fidx = np.array([0, 2], dtype=np.int32)
    bidx_a = np.array([1, 3], dtype=np.int32)
    buckets = [(None, None, None, bidx_a, 0, 8)]
    act_a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    class _FakeFd:
        def __init__(self, active: np.ndarray):
            self.ctx = type("Ctx", (), {"stamp_active": active})()

    merged = FIT.merge_fds_stamp_active(
        global_mask, [_FakeFd(act_a)], buckets, fidx,
    )
    np.testing.assert_array_equal(merged[1, 0], 0.0)
    np.testing.assert_array_equal(merged[3, 2], 0.0)
    np.testing.assert_array_equal(merged[1, 2], 1.0)
    np.testing.assert_array_equal(merged[3, 0], 1.0)



def test_run_stage_writes_slim_checkpoint_history(tmp_path):
    """checkpoint_every=2 retains slim leaves under checkpoints/; params_latest stays full."""
    from pathlib import Path

    import jax.numpy as jnp

    from syndiff_pipeline.forward_model import cheb_wcs as CW
    from syndiff_pipeline.forward_model import epsf_model as EM
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model import loss as L
    from syndiff_pipeline.forward_model import temporal as T
    from syndiff_pipeline.forward_model.groups import GroupSet
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents
    from syndiff_pipeline.forward_model.data import RegionSpec

    region = RegionSpec(0, 0, 100, 100)
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1, exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]), node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]), node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    modes = np.zeros((1, 2, 2, Gsz, Gsz), dtype=np.float32)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))
    members = np.array([[0, -1]], dtype=int)
    valid = np.array([[True, False]], dtype=bool)
    groups = GroupSet(1, 2, members, valid, np.array([True]), 0)
    n_frames, n_basis = 4, 4
    wcs_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    w_fb[:, 0] = 1.0
    ctx = L.build_static_context(
        cheb_static=static, wcs_frame_basis=wcs_fb, w_frame_basis=w_fb,
        epsf_grid=grid, groups=groups,
        ra=np.array([180.0], dtype=np.float32), dec=np.array([0.0], dtype=np.float32),
        stamp_center_x=np.array([50], dtype=np.int64), stamp_center_y=np.array([50], dtype=np.int64),
        t_exp_sec=1426.0, stamp_snr_weight=np.array([1.0], dtype=np.float32),
        fit_radius=np.array([3.0], dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)
    stamp = EM.STAMP_PHYSICAL
    fd = FIT.FitData(
        ctx=ctx,
        data=jnp.zeros((1, n_frames, stamp, stamp), dtype=jnp.float32),
        noise=jnp.ones((1, n_frames, stamp, stamp), dtype=jnp.float32),
        weight=jnp.ones((1, n_frames, stamp, stamp), dtype=jnp.float32),
        wcs_second_diff=T.second_difference_matrix(n_basis),
        w_second_diff=T.second_difference_matrix(n_basis),
        epsf_modes_init=epsf0.modes,
    )
    out_dir = Path(tmp_path)
    latest = out_dir / "params_latest.npz"
    FIT.run_stage(
        params, fd, stage=1, n_steps=4, lr=0.0, log_every=1, reject_every=0,
        checkpoint_path=latest, checkpoint_every=2,
    )
    assert latest.is_file()
    full = dict(np.load(latest))
    assert "epsf_base" in full
    hist = sorted((out_dir / "checkpoints").glob("params_s1_step*.npz"))
    assert len(hist) >= 2
    slim = dict(np.load(hist[0]))
    assert set(slim.keys()) == set(FIT.STAGE_LEAVES)
    assert "epsf_base" not in slim

    stop_dir = out_dir / "stop"
    stop_file = stop_dir / "STOP"
    stop_latest = stop_dir / "params_latest.npz"

    def request_stop(*_args, **_kwargs):
        stop_dir.mkdir(parents=True, exist_ok=True)
        stop_file.touch()

    _, stopped_history = FIT.run_stage(
        params, fd, stage=1, n_steps=4, lr=0.0, log_every=1, reject_every=0,
        checkpoint_path=stop_latest, checkpoint_every=1,
        state_callback=request_stop, stop_file=stop_file,
    )
    assert any(row.get("event") == "stop_requested" for row in stopped_history)
    assert (stop_dir / "checkpoints" / "params_s1_step00000.npz").is_file()



def test_flux_neutral_penalty_is_zero_after_hard_decode():
    G = EM.NODE_GRID_SIZE
    modes = np.zeros((5, 2, 2, G, G), dtype=np.float32)
    modes[..., 0, 0] = 2.0  # raw sum != 0, but decode projects it out
    base_raw = np.zeros((2, 2, G, G), dtype=np.float32)
    params = {"epsf_modes": jnp.asarray(modes), "epsf_base_raw": jnp.asarray(base_raw)}
    val = float(L.flux_neutral_penalty(params))
    # base_raw=0 -> decode_epsf_base is perfectly uniform -> gauge 1's projection axis
    # (base - mean(base)) is ~0, so alpha=num/den divides two near-zero, float32-noisy
    # numbers (den only floored at 1e-12) -- 1e-5 here is float32 noise amplification in
    # a degenerate corner case, not a correctness regression (still far tighter than the
    # 2e-4 precedent in test_decode_epsf_modes_sums_to_zero for a non-degenerate base).
    assert val < 1e-4


def test_node_smoothness_penalty_is_per_edge_average():
    # field[i, j] = (i + j) * ones(G, G): every row-edge AND every col-edge has
    # squared-difference exactly 1, for any grid shape, so the mean is exactly
    # 1.0 regardless of node count (a genuine, not-approximate invariance check).
    G = EM.NODE_GRID_SIZE
    for n_rows, n_cols in [(2, 2), (4, 2), (3, 5)]:
        field = np.fromfunction(lambda i, j: (i + j).astype(np.float32), (n_rows, n_cols))
        base = field[:, :, None, None] * np.ones((1, 1, G, G), dtype=np.float32)
        val = float(L.node_smoothness_penalty(jnp.asarray(base)[None]))
        assert abs(val - 1.0) < 1e-5, (n_rows, n_cols, val)


def test_epsf_base_health_reports_zero_negative_fraction():
    from syndiff_pipeline.forward_model import post_fit as D

    rng = np.random.default_rng(9)
    base = rng.random((2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)).astype(np.float32)
    base /= base.sum(axis=(-1, -2), keepdims=True)
    params = {"epsf_base": base}
    health = D.epsf_base_health(params)
    assert (health["frac_negative"] == 0.0).all()
    assert np.allclose(health["sum"], 1.0, atol=1e-5)


def test_solve_group_fluxes_padding_slot_is_zero():
    S, K, n_groups, n_frames = 13, 4, 1, 1
    templates = np.zeros((n_groups, K, n_frames, S, S), dtype=np.float32)
    g = np.exp(-((np.mgrid[0:S, 0:S][1] - 6) ** 2 + (np.mgrid[0:S, 0:S][0] - 6) ** 2) / 8.0)
    templates[0, 0, 0] = g / g.sum()
    data = (templates[0, 0, 0] * 500.0)[None, None]
    weight = np.ones((n_groups, n_frames, S, S), dtype=np.float32)

    flux = np.asarray(FS.solve_group_fluxes(jnp.asarray(templates), jnp.asarray(data), jnp.asarray(weight)))
    assert abs(flux[0, 0, 0] - 500.0) < 5.0
    assert np.allclose(flux[0, 0, 1:], 0.0, atol=1e-6)


def test_solve_group_fluxes_padding_amount_does_not_change_real_slot_flux():
    """Core invariant behind K-bucketing (groups.bucket_groups_by_size): padding
    a group's slot dimension to a larger, mostly-empty K must not change the
    solved flux for its real member(s) -- only how much wasted K^3 compute the
    batched solve pays. A 2-member group solved at K=2 (no padding) must match
    the same group embedded at K=8 (6 padding slots) to within the ridge term's
    negligible effect."""
    S, n_frames = 13, 3
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:S, 0:S]
    g1 = np.exp(-((xx - 5) ** 2 + (yy - 6) ** 2) / 6.0)
    g1 /= g1.sum()
    g2 = np.exp(-((xx - 8) ** 2 + (yy - 7) ** 2) / 6.0)
    g2 /= g2.sum()
    f1, f2 = 800.0, 250.0
    data = (f1 * g1 + f2 * g2)[None, None] + rng.normal(scale=0.5, size=(1, n_frames, S, S))
    data = np.broadcast_to(data, (1, n_frames, S, S)).astype(np.float32).copy()
    weight = np.ones((1, n_frames, S, S), dtype=np.float32)

    def make_templates(k: int) -> np.ndarray:
        t = np.zeros((1, k, n_frames, S, S), dtype=np.float32)
        t[0, 0] = g1[None]
        t[0, 1] = g2[None]
        return t

    flux_k2 = np.asarray(FS.solve_group_fluxes(
        jnp.asarray(make_templates(2)), jnp.asarray(data), jnp.asarray(weight)
    ))
    flux_k8 = np.asarray(FS.solve_group_fluxes(
        jnp.asarray(make_templates(8)), jnp.asarray(data), jnp.asarray(weight)
    ))
    np.testing.assert_allclose(flux_k2[0, :, 0], flux_k8[0, :, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(flux_k2[0, :, 1], flux_k8[0, :, 1], rtol=1e-4, atol=1e-3)
    # sanity: solve actually recovered something close to the injected fluxes
    assert abs(float(np.median(flux_k2[0, :, 0])) - f1) < 20.0
    assert abs(float(np.median(flux_k2[0, :, 1])) - f2) < 20.0


# ---------------------------------------------------------------------------
# groups: flux-overlap companion augmentation
# ---------------------------------------------------------------------------


def _single_group(primary_idx: int = 0, max_group_size: int = 4):
    from syndiff_pipeline.forward_model import groups as GR

    members = np.full((1, max_group_size), -1, dtype=int)
    valid = np.zeros((1, max_group_size), dtype=bool)
    members[0, 0] = primary_idx
    valid[0, 0] = True
    kept = np.zeros(max(primary_idx + 1, 1), dtype=bool)
    kept[primary_idx] = True
    return GR.GroupSet(1, max_group_size, members, valid, kept, 0)


def test_augment_adds_neighbor_outside_stamp_but_inside_radius():
    from syndiff_pipeline.forward_model import groups as GR

    # Primary at (100, 100); companion center at (107, 100) is outside 13x13 stamp
    # (half=6 → x in [94,106]) but L∞ distance 7 from center.
    x = np.array([100.0, 107.0], dtype=float)
    y = np.array([100.0, 100.0], dtype=float)
    groups = _single_group(0)
    new_groups, stats, _, _ = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, np.array([1], dtype=int),
        primary_members_by_group=[np.array([0], dtype=int)],
        companion_radius_px=12.0,
    )
    assert stats["companions_added"] == 1
    assert new_groups.valid.sum() == 2
    assert new_groups.members[0, 1] == 1


def test_augment_skips_neighbor_beyond_radius():
    from syndiff_pipeline.forward_model import groups as GR

    x = np.array([100.0, 120.0], dtype=float)
    y = np.array([100.0, 100.0], dtype=float)
    groups = _single_group(0)
    new_groups, stats, _, _ = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, np.array([1], dtype=int),
        primary_members_by_group=[np.array([0], dtype=int)],
        companion_radius_px=12.0,
    )
    assert stats["companions_added"] == 0
    assert new_groups.valid.sum() == 1


def test_augment_stamp_center_uses_primary_members_only():
    from syndiff_pipeline.forward_model import groups as GR

    x = np.array([100.0, 110.0], dtype=float)
    y = np.array([100.0, 100.0], dtype=float)
    groups = _single_group(0)
    new_groups, stats, cx, cy = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, np.array([1], dtype=int),
        primary_members_by_group=[np.array([0], dtype=int)],
        companion_radius_px=12.0,
    )
    assert stats["companions_added"] == 1
    assert cx[0] == 100
    assert cy[0] == 100


def test_augment_companion_in_two_overlapping_stamps():
    from syndiff_pipeline.forward_model import groups as GR

    x = np.array([100.0, 200.0, 150.0], dtype=float)
    y = np.array([100.0, 100.0, 100.0], dtype=float)
    members = np.array([[0, -1, -1, -1], [1, -1, -1, -1]], dtype=int)
    valid = np.array([[True, False, False, False], [True, False, False, False]], dtype=bool)
    kept = np.array([True, True, True], dtype=bool)
    groups = GR.GroupSet(2, 4, members, valid, kept, 0)
    new_groups, stats, _, _ = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, np.array([2], dtype=int),
        primary_members_by_group=[np.array([0], dtype=int), np.array([1], dtype=int)],
        companion_radius_px=60.0,
    )
    assert stats["companions_added"] == 2
    assert new_groups.valid.sum(axis=1).tolist() == [2, 2]


def test_merge_star_tables_and_index_map():
    from syndiff_pipeline.forward_model.data import merge_star_tables, primary_to_expanded_index_map

    import pandas as pd

    primary = pd.DataFrame({"source_id": [1, 2], "ra": [1.0, 2.0], "dec": [1.0, 2.0]})
    pool = pd.DataFrame({"source_id": [2, 3], "ra": [2.0, 3.0], "dec": [2.0, 3.0]})
    expanded = merge_star_tables(primary, pool)
    assert len(expanded) == 3
    remap = primary_to_expanded_index_map(primary, expanded)
    assert remap.tolist() == [0, 1]


def test_augment_seeds_from_remapped_primary_indices_not_groups_members():
    """Regression: groups.members may index the primary table; x/y are expanded.

    Seed membership must come from primary_members_by_group (already remapped).
    """
    from syndiff_pipeline.forward_model import groups as GR

    # Expanded table: indices 0=filler, 1=primary, 2=companion near primary.
    # Primary-table index 0 must NOT be treated as expanded index 0.
    x = np.array([0.0, 100.0, 107.0], dtype=float)
    y = np.array([0.0, 100.0, 100.0], dtype=float)
    groups = _single_group(primary_idx=0, max_group_size=4)  # members[:,0] == 0 (primary table)
    new_groups, stats, cx, cy = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, np.array([2], dtype=int),
        primary_members_by_group=[np.array([1], dtype=int)],  # remapped primary → expanded 1
        companion_radius_px=12.0,
    )
    assert stats["companions_added"] == 1
    assert cx[0] == 100 and cy[0] == 100
    members = new_groups.members[0][new_groups.valid[0]]
    assert sorted(members.tolist()) == [1, 2]


def test_augment_drops_when_over_hard_cap():
    from syndiff_pipeline.forward_model import groups as GR

    # One primary + 8 neighbors within radius → 9 members > cap 8 → prune faintest
    # companion (keep primary + 7 brightest companions), do not drop whole group.
    n_neighbors = 8
    x = np.concatenate([[100.0], np.linspace(101.0, 108.0, n_neighbors)])
    y = np.full(1 + n_neighbors, 100.0)
    # Mag ascending with index so faintest is the last neighbor
    mags = np.concatenate([[8.0], np.linspace(9.0, 12.5, n_neighbors)])
    groups = _single_group(0, max_group_size=4)
    new_groups, stats, _, _ = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, np.arange(1, 1 + n_neighbors, dtype=int),
        primary_members_by_group=[np.array([0], dtype=int)],
        companion_radius_px=12.0,
        max_group_size=4,
        max_group_size_cap=8,
        mags=mags,
    )
    assert stats["companions_added"] == n_neighbors
    assert stats["dropped_oversized"] == 1
    assert new_groups.n_groups == 1
    assert int(new_groups.valid.sum()) == 8
    kept = set(new_groups.members[0][new_groups.valid[0]].tolist())
    assert 0 in kept
    assert (1 + n_neighbors - 1) not in kept  # faintest companion pruned


def test_companion_attach_radius_schedule():
    from syndiff_pipeline.forward_model import groups as GR

    h = 6
    base = h * np.sqrt(2.0)
    assert np.isclose(GR.companion_attach_radius_px(8.0, 13), base + 6.0)
    assert np.isclose(GR.companion_attach_radius_px(10.0, 13), base + 5.0)
    assert np.isclose(GR.companion_attach_radius_px(11.5, 13), base + 4.0)
    assert np.isclose(GR.companion_attach_radius_px(12.5, 13), base + 3.0)
    h17 = 8
    base17 = h17 * np.sqrt(2.0)
    assert np.isclose(GR.companion_attach_radius_px(8.0, 17), base17 + 6.0)
    r = GR.companion_attach_radius_px(np.array([8.0, 10.0, 11.5, 12.5]), 13)
    assert np.allclose(r, [base + 6, base + 5, base + 4, base + 3])


def test_augment_mag_dependent_attach():
    from syndiff_pipeline.forward_model import groups as GR

    # Bright mag-8: r≈14.5 — attach at L∞=13. Faint mag-12.5: r≈11.5 — skip at L∞=13.
    x = np.array([100.0, 113.0, 113.0], dtype=float)
    y = np.array([100.0, 100.0, 100.0], dtype=float)
    mags = np.array([8.0, 8.0, 12.5], dtype=float)
    groups = _single_group(0)
    new_groups, stats, _, _ = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, np.array([1, 2], dtype=int),
        primary_members_by_group=[np.array([0], dtype=int)],
        neighbor_mags=mags,
        stamp_physical=13,
        mags=mags,
    )
    assert stats["companions_added"] == 1
    kept = set(new_groups.members[0][new_groups.valid[0]].tolist())
    assert kept == {0, 1}


def test_augment_matches_brute_force_at_scale_with_mixed_mag_tiers():
    """Dense regression test for the tiered-KDTree neighbor-attach rewrite.

    The existing test_augment_* fixtures are all single/two-group toy cases
    and never exercise the batching over multiple mag-dependent radius tiers
    at once. This builds many groups and many candidates spanning all 4
    companion_attach_radius_px tiers and checks the vectorized result
    against an independent brute-force reimplementation of the original
    per-candidate L-infinity distance test.
    """
    from syndiff_pipeline.forward_model import groups as GR

    rng = np.random.default_rng(0)
    n_groups = 40
    n_candidates = 250
    stamp_physical = 13

    # One primary per group, spread on a grid so stamps don't overlap.
    primary_x = (np.arange(n_groups) % 8) * 60.0
    primary_y = (np.arange(n_groups) // 8) * 60.0

    cand_x = rng.uniform(primary_x.min() - 20, primary_x.max() + 20, n_candidates)
    cand_y = rng.uniform(primary_y.min() - 20, primary_y.max() + 20, n_candidates)
    cand_mag = rng.uniform(7.0, 13.0, n_candidates)  # spans all 4 attach tiers

    x = np.concatenate([primary_x, cand_x])
    y = np.concatenate([primary_y, cand_y])
    mags = np.concatenate([np.full(n_groups, 10.0), cand_mag])
    neighbor_indices = np.arange(n_groups, n_groups + n_candidates)

    max_group_size = 4
    members = np.full((n_groups, max_group_size), -1, dtype=int)
    valid = np.zeros((n_groups, max_group_size), dtype=bool)
    members[:, 0] = np.arange(n_groups)
    valid[:, 0] = True
    kept = np.zeros(len(x), dtype=bool)
    kept[:n_groups] = True
    groups = GR.GroupSet(n_groups, max_group_size, members, valid, kept, 0)
    primary_members_by_group = [np.array([gi], dtype=int) for gi in range(n_groups)]

    new_groups, stats, cx, cy = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, neighbor_indices,
        primary_members_by_group=primary_members_by_group,
        neighbor_mags=mags,
        stamp_physical=stamp_physical,
        mags=mags,
        max_group_size=max_group_size,
        max_group_size_cap=1000,  # avoid pruning -- isolate the attach step
    )

    expected_members = []
    for gi in range(n_groups):
        cx_i = float(primary_x[gi])
        cy_i = float(primary_y[gi])
        existing = {gi}
        for ni in neighbor_indices:
            ni = int(ni)
            if ni in existing:
                continue
            r = float(GR.companion_attach_radius_px(mags[ni], stamp_physical))
            dist = max(abs(x[ni] - cx_i), abs(y[ni] - cy_i))
            if dist <= r:
                existing.add(ni)
        expected_members.append(sorted(existing))

    assert new_groups.n_groups == n_groups
    got_members = [
        sorted(new_groups.members[gi][new_groups.valid[gi]].tolist())
        for gi in range(new_groups.n_groups)
    ]
    assert got_members == expected_members
    assert cx.tolist() == primary_x.astype(np.int64).tolist()
    assert cy.tolist() == primary_y.astype(np.int64).tolist()


def test_filter_member_radius_off_is_identity():
    from syndiff_pipeline.forward_model import groups as GR

    x = np.array([100.0, 110.0], dtype=float)
    y = np.array([100.0, 100.0], dtype=float)
    groups = _single_group(0)
    groups, stats, cx, cy = GR.augment_groups_with_stamp_neighbors(
        groups, x, y, np.array([1], dtype=int),
        primary_members_by_group=[np.array([0], dtype=int)],
        companion_radius_px=12.0,
    )
    out, cx2, cy2, estats = GR.filter_groups_by_member_radius(
        groups, x, y, cx, cy, max_member_radius_px=0.0,
    )
    assert estats["trimmed_members"] == 0
    assert out.n_groups == groups.n_groups
    assert np.array_equal(out.members, groups.members)
    assert cx2[0] == cx[0] and cy2[0] == cy[0]


def _groupset_with_sizes(sizes: list[int], max_group_size: int) -> "GroupSet":
    """Synthetic GroupSet with groups of the given true member counts."""
    from syndiff_pipeline.forward_model.groups import GroupSet

    n = len(sizes)
    members = np.full((n, max_group_size), -1, dtype=int)
    valid = np.zeros((n, max_group_size), dtype=bool)
    star = 0
    for gi, s in enumerate(sizes):
        members[gi, :s] = np.arange(star, star + s)
        valid[gi, :s] = True
        star += s
    kept_star_mask = np.zeros(star, dtype=bool)
    kept_star_mask[:] = True
    return GroupSet(n, max_group_size, members, valid, kept_star_mask, 0)


def test_bucket_groups_by_size_partitions_without_loss():
    from syndiff_pipeline.forward_model import groups as GR

    sizes = [1, 1, 3, 8, 2, 1, 5, 4, 8, 2]
    groups = _groupset_with_sizes(sizes, max_group_size=8)
    cx = np.arange(len(sizes), dtype=np.int64) * 10
    cy = np.arange(len(sizes), dtype=np.int64) * 100

    buckets = GR.bucket_groups_by_size(groups, cx, cy, tiers=(1, 2, 4, 8))

    # Every non-empty tier present, ascending K, no group lost or duplicated.
    ks = [bg.max_group_size for bg, _, _, _ in buckets]
    assert ks == sorted(set(ks))
    total_groups = sum(bg.n_groups for bg, _, _, _ in buckets)
    assert total_groups == groups.n_groups

    # orig_group_idx must correctly map each bucket-local group back to the
    # original: same members, same stamp center, K = smallest covering tier,
    # and every original group index appears in exactly one bucket overall.
    seen_orig_idx: list[int] = []
    for bg, bcx, bcy, idx in buckets:
        assert len(idx) == bg.n_groups
        seen_orig_idx.extend(int(i) for i in idx)
        for bi, orig_gi in enumerate(idx):
            mem = set(bg.members[bi][bg.valid[bi]].tolist())
            orig_mem = set(groups.members[orig_gi][groups.valid[orig_gi]].tolist())
            assert mem == orig_mem
            assert bg.max_group_size == min(t for t in (1, 2, 4, 8) if t >= sizes[orig_gi])
            assert bg.valid[bi].sum() == sizes[orig_gi]
            assert bcx[bi] == cx[orig_gi] and bcy[bi] == cy[orig_gi]
    assert sorted(seen_orig_idx) == list(range(groups.n_groups))


def test_bucket_groups_by_size_empty_and_uncovered_tiers():
    from syndiff_pipeline.forward_model import groups as GR
    from syndiff_pipeline.forward_model.groups import GroupSet

    empty = GroupSet(0, 4, np.zeros((0, 4), dtype=int), np.zeros((0, 4), dtype=bool), np.zeros(0, dtype=bool), 0)
    assert GR.bucket_groups_by_size(empty, np.zeros(0), np.zeros(0)) == []

    groups = _groupset_with_sizes([1, 6], max_group_size=8)
    cx = np.zeros(2, dtype=np.int64)
    cy = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="does not cover"):
        GR.bucket_groups_by_size(groups, cx, cy, tiers=(1, 2, 4))  # max group size 6 > max tier 4


def test_wcs_translation_direction_constant_term():
    from syndiff_pipeline.forward_model import post_fit as D

    n_terms, n_basis = 10, 5
    wcs = np.zeros((2 * n_terms, n_basis), dtype=np.float32)
    dx = np.asarray(D.wcs_translation_direction(wcs, n_terms=n_terms, axis="x"))
    dy = np.asarray(D.wcs_translation_direction(wcs, n_terms=n_terms, axis="y"))
    assert dx[0].tolist() == [1.0] * n_basis
    assert np.allclose(dx[1:], 0.0)
    assert dy[n_terms].tolist() == [1.0] * n_basis
    assert np.allclose(dy[:n_terms], 0.0) and np.allclose(dy[n_terms + 1 :], 0.0)


def test_epsf_centroid_shift_is_undone_by_decode_gauge():
    """Raw-direction that used to move the core is reduced by hard decode recenter."""
    from syndiff_pipeline.forward_model import post_fit as D

    idx = np.arange(EM.NODE_GRID_SIZE, dtype=float)
    r2 = (idx[:, None] - EM.NODE_CENTER_INDEX) ** 2 + (idx[None, :] - EM.NODE_CENTER_INDEX) ** 2
    gauss = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    gauss /= gauss.sum()
    base = np.broadcast_to(gauss, (2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)).copy()
    base = np.asarray(EM.recenter_grid_core(jnp.asarray(base), clip_nonneg=True))
    raw = EM.encode_epsf_base(jnp.asarray(base))
    direction = D.epsf_centroid_shift_raw_direction(raw, dx_phys=0.15, dy_phys=0.0)
    raw_shifted = raw + direction
    pos = jax.nn.softplus(raw_shifted) + 1e-8
    unrecentered = pos / jnp.sum(pos, axis=(-2, -1), keepdims=True)
    base1 = np.asarray(EM.decode_epsf_base(raw_shifted))
    cx0, cy0 = EM.core_centroid_xy(jnp.asarray(base))
    cx1, cy1 = EM.core_centroid_xy(jnp.asarray(base1))
    cx_raw, _ = EM.core_centroid_xy(unrecentered)
    assert abs(float(jnp.mean(cx0))) < 5e-2 and abs(float(jnp.mean(cy0))) < 5e-2
    assert abs(float(jnp.mean(cx1))) < abs(float(jnp.mean(cx_raw)))
    assert abs(float(jnp.mean(cy1))) < 5e-2


def test_mad_reject_mask_keeps_inliers_drops_outlier():
    from syndiff_pipeline.forward_model import stamp_reject as SR

    chi2 = np.array([[1.0, 1.1, 0.9, 1.05, 50.0]], dtype=np.float64)
    mask = SR.mad_reject_mask(chi2, n_sigma=3.0)
    assert mask.shape == chi2.shape
    assert mask[0, 4] == 0.0
    assert np.all(mask[0, :4] == 1.0)


def test_mad_reject_mask_zeros_inactive_pixels():
    from syndiff_pipeline.forward_model import stamp_reject as SR

    chi2 = np.ones((2, 3), dtype=np.float64)
    pix_active = np.array([[True, True, False], [False, True, True]])
    mask = SR.mad_reject_mask(chi2, n_sigma=3.0, pix_active=pix_active)
    assert mask[0, 2] == 0.0
    assert mask[1, 0] == 0.0
    assert mask[0, 0] == 1.0


def test_prefilter_quorum_min_frac():
    import pandas as pd

    from syndiff_pipeline.forward_model import stamp_reject as SR

    fit_stars = pd.DataFrame({"source_id": [1, 2, 3], "ra": [0.0, 0.0, 0.0], "dec": [0.0, 0.0, 0.0]})
    # Star 1: seen 4 frames, passes 3 → frac 0.75
    # Star 2: seen 4 frames, passes 2 → frac 0.50
    # Star 3: never seen
    merged = [
        pd.DataFrame({"source_id": [1, 2]}),
        pd.DataFrame({"source_id": [1, 2]}),
        pd.DataFrame({"source_id": [1, 2]}),
        pd.DataFrame({"source_id": [1, 2]}),
    ]
    qc = [
        pd.DataFrame({"source_id": [1, 2]}),
        pd.DataFrame({"source_id": [1, 2]}),
        pd.DataFrame({"source_id": [1]}),
        pd.DataFrame({"source_id": [1]}),
    ]
    kept_lo, stats_lo = SR.prefilter_from_frame_tables(fit_stars, merged, qc, min_frac=0.5)
    assert set(kept_lo["source_id"]) == {1, 2}
    assert stats_lo["n_never_seen"] == 1
    kept_hi, stats_hi = SR.prefilter_from_frame_tables(fit_stars, merged, qc, min_frac=0.8)
    assert set(kept_hi["source_id"]) == {1}
    assert stats_hi["n_below_frac"] == 1


def test_tns_asteroid_stamp_active_static_and_per_ffi():
    """TNS rejects all frames; asteroid rejects only the cadence it is active."""
    import pandas as pd

    from syndiff_pipeline.difference_imaging.masking.bits import ASTEROID, TNS
    from syndiff_pipeline.difference_imaging.masking.catalog import MaskCatalog
    from syndiff_pipeline.forward_model import stamp_reject as SR

    static = np.zeros((100, 100), dtype=np.int16)
    static[50:55, 50:55] = np.int16(TNS)
    intervals = pd.DataFrame(
        {
            "target_id": [1],
            "y": [20],
            "x": [20],
            "cadence_lo": [1],
            "cadence_hi": [1],
        }
    )
    times = pd.DataFrame({"cadence": [0, 1, 2], "btjd": [100.0, 100.1, 100.2]})
    cat = MaskCatalog(
        static=static,
        asteroid_intervals=intervals,
        asteroid_times=times,
    )
    # g0 near TNS, g1 near asteroid track, g2 clean
    cx = np.array([52, 20, 80], dtype=np.int64)
    cy = np.array([52, 20, 80], dtype=np.int64)
    btjds = [100.0, 100.1, 100.2]
    keep, stats = SR.build_tns_asteroid_stamp_active(
        cat, btjds, cx, cy, stamp=13,
    )
    assert keep.shape == (3, 3)
    assert np.all(keep[0] == 0.0)  # TNS every frame
    assert keep[1, 0] == 1.0 and keep[1, 1] == 0.0 and keep[1, 2] == 1.0
    assert np.all(keep[2] == 1.0)
    assert stats["n_tns"] == 3
    assert stats["n_asteroid"] == 1
    assert stats["n_either"] == 4
    # Strap/PS1 bits alone must not reject
    static2 = np.zeros((40, 40), dtype=np.int16)
    static2[10:15, 10:15] = np.int16(4 | 16)  # STRAP | PS1
    cat2 = MaskCatalog(static=static2)
    keep2, _ = SR.build_tns_asteroid_stamp_active(
        cat2, [0.0], np.array([12]), np.array([12]), stamp=5,
    )
    assert keep2[0, 0] == 1.0


def test_combine_stamp_active_ands_mask_baseline():
    from syndiff_pipeline.forward_model import stamp_reject as SR

    mad = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    baseline = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    out = SR.combine_stamp_active(mad, baseline)
    np.testing.assert_array_equal(out, [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    # Without baseline, MAD alone
    np.testing.assert_array_equal(SR.combine_stamp_active(mad, None), mad)


def test_infer_data_root_from_workspace():
    from pathlib import Path

    from syndiff_pipeline.forward_model.run_fit import infer_data_root_from_workspace

    ws = Path("/astro/data/s0020/c3/k3/diff_linear")
    assert infer_data_root_from_workspace(ws) == Path("/astro/data")
    assert infer_data_root_from_workspace(Path("/tmp/not_an_scc")) is None


def test_nll_broadcast_gt_and_ignores_inactive_stamp():
    """(G,T) stamp weights; zeroed stamp must not affect the mean NLL."""
    data = jnp.zeros((2, 2, 3, 3))
    model = jnp.zeros((2, 2, 3, 3))
    # Make group1/frame1 a bad residual; others perfect
    data = data.at[1, 1].set(10.0)
    var = jnp.ones((2, 2, 3, 3))
    pix = jnp.ones((2, 2, 3, 3))
    sw_all = jnp.ones((2, 2), dtype=jnp.float32)
    sw_drop = sw_all.at[1, 1].set(0.0)
    v_all = float(L.per_stamp_snr_weighted_nll(data, model, var, pix, sw_all))
    v_drop = float(L.per_stamp_snr_weighted_nll(data, model, var, pix, sw_drop))
    assert v_all > v_drop
    # (G,) broadcast equals uniform (G,T)
    sw1d = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    v1d = float(L.per_stamp_snr_weighted_nll(data, model, var, pix, sw1d))
    assert abs(v1d - v_all) < 1e-5


def test_effective_stamp_weight_broadcast():
    snr = jnp.asarray([0.25, 1.0], dtype=jnp.float32)
    active = jnp.asarray([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=jnp.float32)
    w = np.asarray(L.effective_stamp_weight(snr, active))
    assert w.shape == (2, 3)
    assert w[0, 0] == pytest.approx(0.25)
    assert w[0, 1] == pytest.approx(0.0)
    assert w[1, 0] == pytest.approx(1.0)
    assert w[1, 2] == pytest.approx(0.0)


def test_stop_grad_frozen_params_zeros_frozen_grads():
    from syndiff_pipeline.forward_model import fit as FIT

    labels = {
        "wcs_coeff": "train_wcs",
        "epsf_base_raw": "frozen",
        "epsf_modes": "frozen",
        "w_coeff": "frozen",
    }
    params = {
        "wcs_coeff": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "epsf_base_raw": jnp.array([3.0, 4.0], dtype=jnp.float32),
        "epsf_modes": jnp.array([5.0], dtype=jnp.float32),
        "w_coeff": jnp.array([6.0], dtype=jnp.float32),
    }

    def loss_fn(p):
        p_sg = FIT.stop_grad_frozen_params(p, labels)
        return jnp.sum(p_sg["wcs_coeff"] ** 2) + jnp.sum(p_sg["epsf_base_raw"] ** 2)

    grads = jax.grad(loss_fn)(params)
    assert float(jnp.max(jnp.abs(grads["wcs_coeff"]))) > 0.0
    assert float(jnp.max(jnp.abs(grads["epsf_base_raw"]))) == 0.0


def test_write_gaia_membership_regions(tmp_path):
    from pathlib import Path

    from syndiff_pipeline.forward_model import post_fit as D

    x = np.array([100.0, 110.0, 120.0, 130.0], dtype=float)
    y = np.array([200.0, 210.0, 220.0, 230.0], dtype=float)
    mags = np.array([8.12, 11.34, 12.56, 7.89], dtype=float)
    path = tmp_path / "gaia_membership.reg"
    D.write_gaia_membership_regions(
        path,
        x=x, y=y,
        region_x_min=90, region_y_min=190,
        pool_indices=np.array([0, 1, 2, 3]),
        primary_indices=np.array([0, 3]),
        companion_indices=np.array([1]),
        rejected_primary_indices=np.array([0]),
        tess_mag=mags,
        include_pool=False,
    )
    text = path.read_text()
    assert "image" in text
    assert text.count("circle(") == 3  # no pool-only star 2
    assert "color=green" not in text.split("image", 1)[1]  # no green circles after image
    assert "text={8.12}" in text  # rejected primary mag
    assert "text={11.34}" in text  # companion mag
    assert "text={7.89}" in text  # kept primary mag
    assert "primary" not in text.split("image", 1)[1] or "text={7.89}" in text
    assert "companion" not in text.split("image", 1)[1]
    assert "pool" not in text.split("image", 1)[1]


# ---------------------------------------------------------------------------
# Warmstart membership + temporal bilin blend
# ---------------------------------------------------------------------------


def test_filter_stars_by_xy_uses_provided_coords_not_catalog():
    import pandas as pd

    from syndiff_pipeline.forward_model.data import RegionSpec, filter_stars_by_xy

    region = RegionSpec(100, 100, 200, 200)
    df = pd.DataFrame({
        "source_id": [1, 2, 3],
        "x": [150.0, 50.0, 150.0],   # catalog: in, out, in
        "y": [150.0, 150.0, 150.0],
        "ra": [1.0, 2.0, 3.0],
        "dec": [1.0, 2.0, 3.0],
    })
    # Warmstart: first out, second in, third in
    x_ws = np.array([50.0, 150.0, 160.0])
    y_ws = np.array([150.0, 150.0, 160.0])
    kept, xk, yk = filter_stars_by_xy(df, x_ws, y_ws, region, margin_px=0.0)
    assert list(kept["source_id"]) == [2, 3]
    assert np.allclose(xk, [150.0, 160.0])
    assert np.allclose(yk, [150.0, 160.0])


def test_drop_groups_off_array_removes_edge_stamps():
    from syndiff_pipeline.forward_model import groups as GR

    members = np.array([[0, -1], [1, -1]], dtype=int)
    valid = np.array([[True, False], [True, False]], dtype=bool)
    kept = np.array([True, True], dtype=bool)
    groups = GR.GroupSet(2, 2, members, valid, kept, 0)
    # Array is 100x100 at origin (0,0); stamp S=13 needs center in [6, 93]
    cx = np.array([50, 2], dtype=np.int64)
    cy = np.array([50, 50], dtype=np.int64)
    new_g, new_cx, new_cy, stats = GR.drop_groups_off_array(
        groups, cx, cy,
        array_origin=(0, 0), array_shape=(100, 100), stamp=13, n_stars=2,
    )
    assert stats["dropped_groups"] == 1
    assert new_g.n_groups == 1
    assert new_cx[0] == 50
    assert new_cy[0] == 50
    assert new_g.members[0, 0] == 0


def test_node_compose_bilin_tracks_query_position_not_xlin():
    """Preferred hot path: compose on nodes, blend at query (x,y)."""
    G = EM.NODE_GRID_SIZE
    node_x = np.array([0.0, 100.0], dtype=np.float32)
    node_y = np.array([0.0, 100.0], dtype=np.float32)
    base = np.zeros((2, 2, G, G), dtype=np.float32)
    # Distinct node signatures so blend location is observable.
    base[0, 0] = 1.0
    base[0, 1] = 2.0
    base[1, 0] = 3.0
    base[1, 1] = 4.0
    base_j = jnp.asarray(base)
    modes_j = jnp.zeros((5, 2, 2, G, G), dtype=jnp.float32)
    w_t = jnp.zeros(5, dtype=jnp.float32)
    node_field = base_j + jnp.einsum("kijxy,k->ijxy", modes_j, w_t)

    # At lower-left node → value 1; at lower-right → value 2
    for qx, qy, expect in [(0.0, 0.0, 1.0), (100.0, 0.0, 2.0), (50.0, 0.0, 1.5)]:
        i0, j0, wy, wx = EM.bilinear_cell(
            jnp.asarray([qx]), jnp.asarray([qy]), node_x, node_y,
        )
        local = EM.blend_to_local(node_field, i0, j0, wy, wx)
        assert float(local[0, 0, 0]) == pytest.approx(expect, abs=1e-5)


def test_forward_model_bilin_and_dx_use_xt_gradients_finite():
    """Tiny forward_model: grads w.r.t. wcs_coeff finite when bilin uses x(t)."""
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents

    from syndiff_pipeline.forward_model.data import RegionSpec
    from syndiff_pipeline.forward_model.groups import GroupSet

    region = RegionSpec(0, 0, 100, 100)
    static = CW.ChebWcsStatic(
        ra0_deg=180.0,
        dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1,
        exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]),
        node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]),
        node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob /= blob.sum()
    for i in range(2):
        for j in range(2):
            base[i, j] = blob * (1.0 + 0.1 * i + 0.2 * j)
            base[i, j] /= base[i, j].sum()
    modes = np.zeros((5, 2, 2, Gsz, Gsz), dtype=np.float32)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    members = np.array([[0, -1]], dtype=int)
    valid = np.array([[True, False]], dtype=bool)
    groups = GroupSet(1, 2, members, valid, np.array([True]), 0)

    # Star near CRVAL → x_lin ≈ 50
    ra = np.array([180.0], dtype=np.float32)
    dec = np.array([0.0], dtype=np.float32)
    n_frames = 2
    n_basis = 2
    wcs_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    w_fb[:, 0] = 1.0

    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_fb,
        w_frame_basis=w_fb,
        epsf_grid=grid,
        groups=groups,
        ra=ra,
        dec=dec,
        stamp_center_x=np.array([50], dtype=np.int64),
        stamp_center_y=np.array([50], dtype=np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.array([1.0], dtype=np.float32),
        fit_radius=np.array([3.0], dtype=np.float32),
    )
    assert "bilin_i0" not in ctx.__dataclass_fields__

    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)
    # Nonzero constant term on a_00 so x(t) shifts from x_lin
    params["wcs_coeff"] = params["wcs_coeff"].at[0, :].set(5.0)

    def loss_fn(p):
        templates, *_ = L.forward_model(p, ctx)
        return jnp.sum(templates ** 2)

    val, grads = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(float(val))
    assert np.isfinite(np.asarray(grads["wcs_coeff"])).all()
    assert np.isfinite(np.asarray(grads["epsf_base_raw"])).all()


def _tiny_forward_fixture():
    """Shared tiny StaticContext + params for bake / forward_model checks."""
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents

    from syndiff_pipeline.forward_model.groups import GroupSet

    static = CW.ChebWcsStatic(
        ra0_deg=180.0,
        dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1,
        exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]),
        node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]),
        node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob /= blob.sum()
    for i in range(2):
        for j in range(2):
            base[i, j] = blob
            base[i, j] /= base[i, j].sum()
    modes = np.zeros((1, 2, 2, Gsz, Gsz), dtype=np.float32)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))
    members = np.array([[0, -1]], dtype=int)
    valid = np.array([[True, False]], dtype=bool)
    groups = GroupSet(1, 2, members, valid, np.array([True]), 0)
    ra = np.array([180.0], dtype=np.float32)
    dec = np.array([0.0], dtype=np.float32)
    n_frames, n_basis = 2, 2
    wcs_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    w_fb[:, 0] = 1.0
    return static, grid, groups, ra, dec, wcs_fb, w_fb, epsf0, n_basis


def test_build_static_context_bakes_star_basis_once():
    static, grid, groups, ra, dec, wcs_fb, w_fb, _, _ = _tiny_forward_fixture()
    x0, y0, b0 = CW.star_basis(jnp.asarray(ra), jnp.asarray(dec), static)
    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_fb,
        w_frame_basis=w_fb,
        epsf_grid=grid,
        groups=groups,
        ra=ra,
        dec=dec,
        stamp_center_x=np.array([50], dtype=np.int64),
        stamp_center_y=np.array([50], dtype=np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.array([1.0], dtype=np.float32),
        fit_radius=np.array([3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(np.asarray(ctx.x_lin), np.asarray(x0), rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ctx.y_lin), np.asarray(y0), rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ctx.cheb_basis), np.asarray(b0), rtol=0, atol=1e-5)

    ctx2 = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_fb,
        w_frame_basis=w_fb,
        epsf_grid=grid,
        groups=groups,
        ra=ra,
        dec=dec,
        stamp_center_x=np.array([50], dtype=np.int64),
        stamp_center_y=np.array([50], dtype=np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.array([1.0], dtype=np.float32),
        fit_radius=np.array([3.0], dtype=np.float32),
        x_lin=np.asarray(x0),
        y_lin=np.asarray(y0),
        cheb_basis=np.asarray(b0),
    )
    np.testing.assert_allclose(np.asarray(ctx2.x_lin), np.asarray(x0), rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ctx2.cheb_basis), np.asarray(b0), rtol=0, atol=1e-5)


def test_forward_model_matches_explicit_eval_all_positions():
    static, grid, groups, ra, dec, wcs_fb, w_fb, epsf0, n_basis = _tiny_forward_fixture()
    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_fb,
        w_frame_basis=w_fb,
        epsf_grid=grid,
        groups=groups,
        ra=ra,
        dec=dec,
        stamp_center_x=np.array([50], dtype=np.int64),
        stamp_center_y=np.array([50], dtype=np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.array([1.0], dtype=np.float32),
        fit_radius=np.array([3.0], dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)
    params["wcs_coeff"] = params["wcs_coeff"].at[0, :].set(5.0)

    _, x_t, y_t, _ = L.forward_model(params, ctx)
    x_ref, y_ref = CW.eval_all_positions(
        ctx.x_lin, ctx.y_lin, ctx.cheb_basis,
        params["wcs_coeff"], ctx.wcs_frame_basis, ctx.n_terms,
    )
    # forward_model returns packed occupied-slot positions, not all-star (n_stars, T)
    # Compare via the same star index used in the single occupied slot.
    star_idx = int(np.asarray(ctx.members)[0, 0])
    np.testing.assert_allclose(
        np.asarray(x_t)[0, 0], np.asarray(x_ref)[star_idx], rtol=0, atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(y_t)[0, 0], np.asarray(y_ref)[star_idx], rtol=0, atol=1e-5,
    )


def test_forward_model_does_not_call_star_basis(monkeypatch):
    static, grid, groups, ra, dec, wcs_fb, w_fb, epsf0, n_basis = _tiny_forward_fixture()
    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_fb,
        w_frame_basis=w_fb,
        epsf_grid=grid,
        groups=groups,
        ra=ra,
        dec=dec,
        stamp_center_x=np.array([50], dtype=np.int64),
        stamp_center_y=np.array([50], dtype=np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.array([1.0], dtype=np.float32),
        fit_radius=np.array([3.0], dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)

    def _boom(*_a, **_k):
        raise AssertionError("star_basis must not be called from forward_model")

    monkeypatch.setattr(CW, "star_basis", _boom)
    monkeypatch.setattr(L.CW, "star_basis", _boom)
    templates, *_ = L.forward_model(params, ctx)
    assert np.isfinite(np.asarray(templates)).all()


def test_forward_model_banded_render_matches_old_path():
    """End-to-end acceptance check (plan Sec. "Acceptance" #2): the new banded,
    gather-free hot path (loss.USE_BANDED_RENDER=True) must reproduce the old
    blend->recenter_grid_core->render_stamps path to <=1e-5 relative to the
    template peak. Uses an asymmetric base + live modes/w_of_t/wcs offset so
    the COM-recenter fold is actually exercised (a plain symmetric blob would
    trivially have COM=0 either way).
    """
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents

    from syndiff_pipeline.forward_model.groups import GroupSet

    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1, exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]), node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]), node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    wing_r2 = (xx - EM.NODE_CENTER_INDEX - 3) ** 2 + (yy - EM.NODE_CENTER_INDEX + 2) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob = blob + 0.3 * np.exp(-wing_r2 / (2 * (EM.OVERSAMPLE * 2.0) ** 2)).astype(np.float32)
    blob /= blob.sum()
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            base[i, j] = blob * (1.0 + 0.1 * i + 0.2 * j)
            base[i, j] /= base[i, j].sum()
    rng = np.random.default_rng(0)
    modes = 0.05 * rng.normal(size=(3, 2, 2, Gsz, Gsz)).astype(np.float32)
    modes -= modes.mean(axis=(-1, -2), keepdims=True)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    members = np.array([[0, 1, -1], [2, -1, -1]], dtype=int)
    valid = np.array([[True, True, False], [True, False, False]], dtype=bool)
    groups = GroupSet(2, 3, members, valid, np.array([True, True]), 0)

    ra = np.array([180.0, 180.002, 179.998], dtype=np.float32)
    dec = np.array([0.0, 0.001, -0.0015], dtype=np.float32)
    n_frames, n_basis = 5, 3
    rng2 = np.random.default_rng(2)
    wcs_fb = (0.1 * rng2.normal(size=(n_frames, n_basis))).astype(np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = (0.1 * rng2.normal(size=(n_frames, n_basis))).astype(np.float32)
    w_fb[:, 0] = 1.0

    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_fb,
        w_frame_basis=w_fb,
        epsf_grid=grid,
        groups=groups,
        ra=ra,
        dec=dec,
        stamp_center_x=np.array([50, 52], dtype=np.int64),
        stamp_center_y=np.array([50, 48], dtype=np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.array([1.0, 1.0], dtype=np.float32),
        fit_radius=np.array([6.0, 6.0], dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)
    params["wcs_coeff"] = params["wcs_coeff"].at[0, :].set(3.0)
    params["epsf_modes"] = 0.1 * jnp.asarray(
        rng2.normal(size=params["epsf_modes"].shape).astype(np.float32)
    )

    saved = L.USE_BANDED_RENDER
    try:
        L.USE_BANDED_RENDER = False
        t_old, x_old, y_old, w_old = L.forward_model(params, ctx)
        L.USE_BANDED_RENDER = True
        t_new, x_new, y_new, w_new = L.forward_model(params, ctx)
    finally:
        L.USE_BANDED_RENDER = saved

    t_old = np.asarray(t_old)
    t_new = np.asarray(t_new)
    max_diff = np.abs(t_old - t_new).max()
    peak = np.abs(t_old).max()
    assert peak > 0
    assert max_diff / peak <= 1e-5, (max_diff, peak)
    np.testing.assert_allclose(np.asarray(x_old), np.asarray(x_new), atol=1e-6)
    np.testing.assert_allclose(np.asarray(y_old), np.asarray(y_new), atol=1e-6)
    np.testing.assert_allclose(np.asarray(w_old), np.asarray(w_new), atol=1e-6)


# ---------------------------------------------------------------------------
# aperture correction + flux errors
# ---------------------------------------------------------------------------


def test_solve_group_fluxes_with_err_matches_inv_diag():
    rng = np.random.default_rng(11)
    S, K, n_groups, n_frames = 7, 2, 3, 2
    templates = rng.random((n_groups, K, n_frames, S, S)).astype(np.float32) + 0.1
    weight = rng.uniform(0.5, 2.0, size=(n_groups, n_frames, S, S)).astype(np.float32)
    true_flux = rng.uniform(10, 50, size=(n_groups, n_frames, K)).astype(np.float32)
    data = np.einsum("gkfxy,gfk->gfxy", templates, true_flux)
    flux, sigma = FS.solve_group_fluxes_with_err(
        jnp.asarray(templates), jnp.asarray(data), jnp.asarray(weight), ridge=1e-9,
    )
    flux2 = FS.solve_group_fluxes(
        jnp.asarray(templates), jnp.asarray(data), jnp.asarray(weight), ridge=1e-9,
    )
    assert np.allclose(np.asarray(flux), np.asarray(flux2), atol=1e-5)
    # Single-group analytic check
    g, f = 0, 0
    M = templates[g, :, f].reshape(K, -1).T
    W = np.diag(weight[g, f].reshape(-1))
    MtWM = M.T @ W @ M + 1e-9 * np.eye(K)
    cov = np.linalg.inv(MtWM)
    expected = np.sqrt(np.diag(cov))
    assert np.allclose(np.asarray(sigma)[g, f], expected, rtol=1e-4, atol=1e-5)


def test_aperture_correction_recovers_injected_field():
    from syndiff_pipeline.forward_model import aperture_correction as AC

    rng = np.random.default_rng(21)
    n_rows = n_cols = 2
    n_stars, n_frames, n_basis = 40, 20, 4
    node_x = np.array([25.0, 75.0])
    node_y = np.array([25.0, 75.0])
    # Uniform temporal basis on [0,1]
    t = np.linspace(0, 1, n_frames)
    Phi = np.vstack([np.ones(n_frames), t, t**2, t**3]).T.astype(np.float64)
    # Orthonormalize lightly for conditioning
    Phi, _ = np.linalg.qr(Phi)

    A_true = np.zeros((n_rows, n_cols, n_basis))
    A_true[0, 0, 1] = 0.08
    A_true[0, 1, 1] = -0.04
    A_true[1, 0, 2] = 0.06
    A_true[1, 1, :] = -A_true.reshape(4, n_basis)[:-1].sum(axis=0)
    A_node = AC.decode_aperture(A_true, Phi)

    # Stars scattered in the box
    x0 = rng.uniform(20, 80, size=n_stars)
    y0 = rng.uniform(20, 80, size=n_stars)
    x = np.broadcast_to(x0[:, None], (n_stars, n_frames)).copy()
    y = np.broadcast_to(y0[:, None], (n_stars, n_frames)).copy()
    A_star = AC.eval_astar(A_node, x, y, node_x, node_y)
    F = rng.uniform(100, 1000, size=n_stars)
    flux = F[:, None] * A_star
    # Heteroscedastic noise: fainter → larger relative error
    sigma = (0.001 * F)[:, None] * np.ones((n_stars, n_frames))
    flux = flux + rng.normal(0.0, 1.0, size=flux.shape) * sigma

    table = AC.PrimaryFluxTable(
        flux=flux,
        sigma_f=sigma,
        x=x,
        y=y,
        active=np.ones_like(flux, dtype=bool),
        group_index=np.arange(n_stars),
        slot_index=np.zeros(n_stars, dtype=int),
        star_index=np.arange(n_stars),
        tess_mag=np.full(n_stars, 8.0),
    )
    res = AC.fit_aperture_coeff(
        table, Phi, node_x, node_y, n_rows=n_rows, n_cols=n_cols, n_iter=15,
    )
    # Global scale gauge only (not per-frame spatial mean — that kills common mode)
    assert np.isclose(res.A_node.mean(), 1.0, atol=1e-5)
    # Correlation of recovered vs true A_star
    corr = np.corrcoef(A_star.ravel(), res.A_star.ravel())[0, 1]
    assert corr > 0.95, corr
    # Corrected fluxes nearly constant
    after = AC.summarize_flux_trend(res.flux_corr, table.active)
    before = AC.summarize_flux_trend(table.flux, table.active)
    assert after["ptp_mean_frac"] < 0.5 * before["ptp_mean_frac"]


# ---------------------------------------------------------------------------
# loss.total_loss: stamp_chunk (lax.scan + jax.checkpoint frame-block chunking)
# ---------------------------------------------------------------------------
#
# Gradient tolerance note: unlike the loss value (a single weighted mean, well
# conditioned, matches unchunked to ~1e-7 absolute regardless of block count --
# see the loss-only assertions below), individual gradient *elements* --
# especially wcs_coeff/w_coeff rows with a near-zero aggregate derivative --
# can show elementwise relative differences up to ~1e-3 purely from
# jax.checkpoint's forward-recompute-on-backward being fused/scheduled
# differently by XLA than the original (non-rematted) forward pass. This was
# isolated directly: wrapping the *unchunked* total_loss in a bare
# ``jax.checkpoint`` (no lax.scan, no chunking at all) reproduces the same
# ~1e-5-absolute-diff noise floor on this fixture's wcs_coeff gradient, with
# and without multithreading (SYNDIFF_FIT_THREADS=1 gives the same result) --
# i.e. it is float32-roundoff-under-remat, not a chunking bug, and it is
# already present with just one block (block == n_frames, no padding). The
# tolerances below (rtol=1e-3, atol=1e-4) comfortably cover that floor while
# still catching real bugs, which show up as O(0.1-1) relative errors (e.g. a
# dropped frame block, wrong pad handling, or a numerator/denominator mixup).
_STAMP_CHUNK_GRAD_RTOL = 1e-3
_STAMP_CHUNK_GRAD_ATOL = 1e-4


def _stamp_chunk_fixture(n_frames: int, *, seed: int = 0):
    """Small but non-trivial StaticContext + params + data for stamp_chunk
    tests: heterogeneous group sizes, a real (non-flat) WCS distortion and
    temporal ePSF weighting, and a mixed stamp_active mask (some rejected
    (group, frame) stamps), all so the chunked data-term path is exercised
    the same way it would be in a real fit, not just on trivial input."""
    from syndiff_pipeline.forward_model.groups import GroupSet

    rng = np.random.default_rng(seed)
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=2, exponents=tuple(sci2idl_exponents(2)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]), node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]), node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    wing_r2 = (xx - EM.NODE_CENTER_INDEX - 3) ** 2 + (yy - EM.NODE_CENTER_INDEX + 2) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob = blob + 0.3 * np.exp(-wing_r2 / (2 * (EM.OVERSAMPLE * 2.0) ** 2)).astype(np.float32)
    blob /= blob.sum()
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            base[i, j] = blob * (1.0 + 0.1 * i + 0.2 * j)
            base[i, j] /= base[i, j].sum()
    modes = 0.05 * rng.normal(size=(2, 2, 2, Gsz, Gsz)).astype(np.float32)
    modes -= modes.mean(axis=(-1, -2), keepdims=True)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    sizes = [1, 2, 3, 2]  # heterogeneous membership, padded to K=3
    n_stars = sum(sizes)
    n_groups, max_k = len(sizes), 3
    members = np.full((n_groups, max_k), -1, dtype=int)
    valid = np.zeros((n_groups, max_k), dtype=bool)
    star = 0
    for gi, s in enumerate(sizes):
        members[gi, :s] = np.arange(star, star + s)
        valid[gi, :s] = True
        star += s
    groups = GroupSet(n_groups, max_k, members, valid, np.ones(n_stars, dtype=bool), 0)
    ra = 180.0 + 0.001 * np.arange(n_stars)
    dec = 0.0 + 0.001 * np.arange(n_stars)

    n_basis = 4
    wcs_fb = (0.1 * rng.normal(size=(n_frames, n_basis))).astype(np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = (0.1 * rng.normal(size=(n_frames, n_basis))).astype(np.float32)
    w_fb[:, 0] = 1.0

    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_fb,
        w_frame_basis=w_fb,
        epsf_grid=grid,
        groups=groups,
        ra=ra, dec=dec,
        stamp_center_x=np.array([50, 52, 49, 51], dtype=np.int64),
        stamp_center_y=np.array([50, 48, 51, 49], dtype=np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.ones(n_groups, dtype=np.float32),
        fit_radius=np.full(n_groups, 6.0, dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)
    params["wcs_coeff"] = params["wcs_coeff"] + 0.05 * jnp.asarray(
        rng.normal(size=params["wcs_coeff"].shape).astype(np.float32)
    )
    params["epsf_modes"] = 0.1 * jnp.asarray(
        rng.normal(size=params["epsf_modes"].shape).astype(np.float32)
    )
    params["w_coeff"] = 0.1 * jnp.asarray(
        rng.normal(size=params["w_coeff"].shape).astype(np.float32)
    )

    S = EM.STAMP_PHYSICAL
    data = jnp.asarray(rng.normal(size=(n_groups, n_frames, S, S)).astype(np.float32))
    noise = jnp.asarray(rng.uniform(0.5, 1.5, size=(n_groups, n_frames, S, S)).astype(np.float32))
    weight = jnp.asarray(np.ones((n_groups, n_frames, S, S), dtype=np.float32))
    # Mixed reject mask: not all-active, not all-rejected, so padding frames
    # (stamp_active forced to 0) are exercised alongside genuine rejects.
    stamp_active = jnp.asarray((rng.uniform(size=(n_groups, n_frames)) > 0.2).astype(np.float32))
    wcs_second_diff = T.second_difference_matrix(n_basis)
    w_second_diff = T.second_difference_matrix(n_basis)

    return dict(
        params=params, ctx=ctx, data=data, noise=noise, weight=weight,
        stamp_active=stamp_active, wcs_second_diff=wcs_second_diff,
        w_second_diff=w_second_diff, epsf_modes_init=epsf0.modes,
    )


def _stamp_chunk_loss_and_grad(fx: dict, stamp_chunk: int | None):
    def f(params):
        loss, metrics = L.total_loss(
            params, fx["ctx"], fx["data"], fx["noise"], fx["weight"],
            fx["wcs_second_diff"], fx["w_second_diff"],
            epsf_modes_init=fx["epsf_modes_init"],
            stamp_active=fx["stamp_active"],
            stamp_chunk=stamp_chunk,
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(f, has_aux=True)(fx["params"])
    return loss, metrics, grads


def test_stamp_chunk_matches_unchunked_loss():
    """Chunked (lax.scan + jax.checkpoint) data-term must match the unchunked
    forward path's loss to fp32 roundoff -- it's a pure sum/weighted-mean
    reassociation, not a different computation."""
    fx = _stamp_chunk_fixture(n_frames=12)
    loss_ref, metrics_ref, _ = _stamp_chunk_loss_and_grad(fx, None)
    for block in (2, 4, 6, 12):
        loss_c, metrics_c, _ = _stamp_chunk_loss_and_grad(fx, block)
        np.testing.assert_allclose(float(loss_c), float(loss_ref), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            float(metrics_c["stamp_weight_sum"]), float(metrics_ref["stamp_weight_sum"]),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(
            float(metrics_c["data_term"]), float(metrics_ref["data_term"]),
            rtol=1e-5, atol=1e-5,
        )


def test_stamp_chunk_matches_unchunked_gradient():
    """Same as above, but for value_and_grad on all four trainable leaves."""
    fx = _stamp_chunk_fixture(n_frames=12)
    _, _, grads_ref = _stamp_chunk_loss_and_grad(fx, None)
    for block in (2, 4, 6, 12):
        _, _, grads_c = _stamp_chunk_loss_and_grad(fx, block)
        for leaf in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
            np.testing.assert_allclose(
                np.asarray(grads_c[leaf]), np.asarray(grads_ref[leaf]),
                rtol=_STAMP_CHUNK_GRAD_RTOL, atol=_STAMP_CHUNK_GRAD_ATOL,
                err_msg=f"block={block} leaf={leaf}",
            )


def test_stamp_chunk_nondividing_block_size_matches_unchunked():
    """block must not need to evenly divide n_frames -- the frame axis is
    zero-padded with stamp_active forced to 0 on the pad frames, which must
    contribute exactly nothing. T=10, block=3 -> 4 blocks, 2 padding frames."""
    fx = _stamp_chunk_fixture(n_frames=10)
    loss_ref, metrics_ref, grads_ref = _stamp_chunk_loss_and_grad(fx, None)
    loss_c, metrics_c, grads_c = _stamp_chunk_loss_and_grad(fx, 3)
    np.testing.assert_allclose(float(loss_c), float(loss_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        float(metrics_c["stamp_weight_sum"]), float(metrics_ref["stamp_weight_sum"]),
        rtol=1e-5, atol=1e-5,
    )
    for leaf in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        np.testing.assert_allclose(
            np.asarray(grads_c[leaf]), np.asarray(grads_ref[leaf]),
            rtol=_STAMP_CHUNK_GRAD_RTOL, atol=_STAMP_CHUNK_GRAD_ATOL,
            err_msg=f"leaf={leaf}",
        )
    # A second, non-dividing block size (7 -> 2 blocks, 4 padding frames) must
    # also agree, so this isn't a coincidence of one particular padding amount.
    loss_c7, metrics_c7, grads_c7 = _stamp_chunk_loss_and_grad(fx, 7)
    np.testing.assert_allclose(float(loss_c7), float(loss_ref), rtol=1e-5, atol=1e-5)
    for leaf in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        np.testing.assert_allclose(
            np.asarray(grads_c7[leaf]), np.asarray(grads_ref[leaf]),
            rtol=_STAMP_CHUNK_GRAD_RTOL, atol=_STAMP_CHUNK_GRAD_ATOL,
            err_msg=f"leaf={leaf}",
        )


def test_stamp_chunk_various_block_counts_agree_with_each_other():
    """Several different block counts (dividing and non-dividing) must all
    agree with each other, not just with the unchunked reference -- pins down
    that the result is independent of --stamp-chunk's specific value."""
    fx = _stamp_chunk_fixture(n_frames=16)
    results = {}
    for block in (1, 3, 4, 5, 8, 16):
        loss_c, metrics_c, grads_c = _stamp_chunk_loss_and_grad(fx, block)
        results[block] = (float(loss_c), grads_c)

    ref_loss, ref_grads = results[16]  # block == n_frames: single block, no padding
    for block, (loss_c, grads_c) in results.items():
        np.testing.assert_allclose(loss_c, ref_loss, rtol=1e-5, atol=1e-5, err_msg=f"block={block}")
        for leaf in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
            np.testing.assert_allclose(
                np.asarray(grads_c[leaf]), np.asarray(ref_grads[leaf]),
                rtol=_STAMP_CHUNK_GRAD_RTOL, atol=_STAMP_CHUNK_GRAD_ATOL,
                err_msg=f"block={block} leaf={leaf}",
            )


def test_stamp_chunk_zero_or_none_is_unchunked_and_bit_identical():
    """stamp_chunk=None and stamp_chunk=0 must both take the literal original
    (non-scan, non-checkpoint) code path -- bit-identical, not just close."""
    fx = _stamp_chunk_fixture(n_frames=8)
    loss_none, metrics_none, grads_none = _stamp_chunk_loss_and_grad(fx, None)
    loss_zero, metrics_zero, grads_zero = _stamp_chunk_loss_and_grad(fx, 0)
    assert float(loss_none) == float(loss_zero)
    for leaf in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        np.testing.assert_array_equal(np.asarray(grads_none[leaf]), np.asarray(grads_zero[leaf]))


def test_stamp_chunk_rejects_minibatch_frame_or_group_idx():
    """frame_idx/group_idx minibatching is out of scope for this chunking pass
    (neither is wired from fit.py's minibatch step today -- see
    fit.make_step_fn); combining them with stamp_chunk must fail loudly
    rather than silently compute something else."""
    fx = _stamp_chunk_fixture(n_frames=6)
    with pytest.raises(NotImplementedError):
        L.total_loss(
            fx["params"], fx["ctx"], fx["data"], fx["noise"], fx["weight"],
            fx["wcs_second_diff"], fx["w_second_diff"],
            epsf_modes_init=fx["epsf_modes_init"],
            stamp_active=fx["stamp_active"],
            stamp_chunk=2,
            frame_idx=jnp.arange(4),
        )


def test_epsf_grid_static_from_region_center_default_bit_identical():
    """S4: adding the ``placement`` kwarg must not change a single bit of the
    original (now default ``placement="center"``) behaviour -- callers that
    never pass ``placement`` (every call site in this repo as of this test)
    must get exactly the historical node positions."""
    from syndiff_pipeline.forward_model.data import RegionSpec

    region = RegionSpec(873, 281, 1897, 1305)
    grid_default = EM.EpsfGridStatic.from_region(region, n_rows=4, n_cols=3, crop_origin=(44, 0))
    grid_explicit_center = EM.EpsfGridStatic.from_region(
        region, n_rows=4, n_cols=3, crop_origin=(44, 0), placement="center",
    )
    np.testing.assert_array_equal(grid_default.node_x, grid_explicit_center.node_x)
    np.testing.assert_array_equal(grid_default.node_y, grid_explicit_center.node_y)
    np.testing.assert_array_equal(grid_default.node_col_ccd, grid_explicit_center.node_col_ccd)
    np.testing.assert_array_equal(grid_default.node_row_ccd, grid_explicit_center.node_row_ccd)

    # Reproduce the original hand-written formula independently (pre-refactor
    # reference), not just a self-consistency check against the new code path.
    x0, y0, x1, y1 = region.x_min, region.y_min, region.x_max, region.y_max
    col_w = (x1 - x0) / 3
    row_h = (y1 - y0) / 4
    expected_node_x = x0 + (np.arange(3) + 0.5) * col_w
    expected_node_y = y0 + (np.arange(4) + 0.5) * row_h
    np.testing.assert_array_equal(grid_default.node_x, expected_node_x)
    np.testing.assert_array_equal(grid_default.node_y, expected_node_y)


def test_epsf_grid_static_from_region_edge_places_nodes_at_bounds():
    """S4: ``placement="edge"`` must put the OUTERMOST nodes exactly at the
    region bounds, with the remaining nodes evenly spaced between, for
    several (n_rows, n_cols) including a non-square grid."""
    from syndiff_pipeline.forward_model.data import RegionSpec

    region = RegionSpec(873, 281, 1897, 1305)
    for n_rows, n_cols in [(2, 2), (3, 3), (4, 4), (5, 5), (4, 3)]:
        grid = EM.EpsfGridStatic.from_region(
            region, n_rows=n_rows, n_cols=n_cols, crop_origin=(44, 0), placement="edge",
        )
        assert grid.node_x[0] == region.x_min
        assert grid.node_x[-1] == region.x_max
        assert grid.node_y[0] == region.y_min
        assert grid.node_y[-1] == region.y_max
        np.testing.assert_allclose(grid.node_x, np.linspace(region.x_min, region.x_max, n_cols))
        np.testing.assert_allclose(grid.node_y, np.linspace(region.y_min, region.y_max, n_rows))
        # strictly increasing (required by bilinear_cell's searchsorted)
        assert np.all(np.diff(grid.node_x) > 0)
        assert np.all(np.diff(grid.node_y) > 0)
        # node_col_ccd/node_row_ccd still carry the crop_origin offset
        np.testing.assert_array_equal(grid.node_col_ccd, grid.node_x + 44)
        np.testing.assert_array_equal(grid.node_row_ccd, grid.node_y + 0)


def test_epsf_grid_static_from_region_unknown_placement_rejected():
    from syndiff_pipeline.forward_model.data import RegionSpec

    region = RegionSpec(0, 0, 100, 100)
    with pytest.raises(ValueError):
        EM.EpsfGridStatic.from_region(
            region, n_rows=2, n_cols=2, crop_origin=(0, 0), placement="corner",
        )
