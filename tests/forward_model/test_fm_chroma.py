# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Unit tests for the chromatic PSF term.

Tier 1 here: the dilation operator itself. These are the tests that pin the sign,
the ``2P`` coefficient and the index-vs-physical coordinate convention. A sign or
factor error in ``dilation_generator`` is otherwise nearly invisible downstream,
because ``renorm_scalar`` and the per-stamp flux solve each absorb a net rescale.

Tiers 2-4 (axis order, banded-vs-unbanded, AD and plumbing) live alongside the
existing render tests once the render plumbing lands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import epsf_model as EM

jax.config.update("jax_platform_name", "cpu")


G_SMALL = 58


def _grid_coords(g_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Physical-px sample coordinates, matching ``EM.node_coord_1d``."""
    coord = np.asarray(EM.node_coord_1d(n_grid=g_size), dtype=np.float64)
    return coord[None, :], coord[:, None]  # (x along last axis, y along second-last)


def _gaussian(g_size: int, sigma_px: float, *, scale: float = 1.0,
              x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Flux-normalised sampled Gaussian, dilated by ``scale``: ``a^-2 P(u/a)``."""
    x, y = _grid_coords(g_size)
    s = sigma_px * scale
    g = np.exp(-0.5 * (((x - x0) ** 2 + (y - y0) ** 2) / s ** 2)) / (2 * np.pi * s ** 2)
    return g.astype(np.float64)


def test_dilation_generator_matches_analytic_scale_derivative():
    """``D[P]`` is the negative scale derivative of ``P_a(u) = a^-2 P(u/a)``.

    Pins the sign, the ``2P`` coefficient and the OVERSAMPLE convention in one shot.
    A missing ``2P`` shows up as a large error at the core; a wrong OVERSAMPLE gives
    a factor 4; a sign flip gives a correlation of -1.
    """
    sigma = 2.0  # wide enough that the central difference is accurate at 4x sampling
    p = _gaussian(G_SMALL, sigma)
    da = -1e-3
    p_a = _gaussian(G_SMALL, sigma, scale=1.0 + da)
    analytic = (p_a - p) / da  # d/da P_a at a=1, which equals -D[P]

    got = np.asarray(EM.dilation_generator(jnp.asarray(p, dtype=jnp.float32)), dtype=np.float64)

    # Compare where there is signal; the far wings are ~0 in both and only add noise.
    m = p > 1e-6 * p.max()
    assert m.sum() > 200
    num = np.abs(got[m] + analytic[m]).max()
    den = np.abs(analytic[m]).max()
    assert num / den < 2e-2, f"D[P] != -dP_a/da, max rel err {num / den:.3e}"

    # Sign, stated explicitly: positive q in P + q D[P] must NARROW the PSF.
    q = 0.01
    widened = p + q * got
    r2 = (_grid_coords(G_SMALL)[0] ** 2 + _grid_coords(G_SMALL)[1] ** 2)
    second_moment_before = float((p * r2).sum() / p.sum())
    second_moment_after = float((widened * r2).sum() / widened.sum())
    assert second_moment_after < second_moment_before, (
        "positive q must give a NARROWER PSF; sign convention is inverted"
    )


def test_dilation_generator_recovers_the_scale_factor():
    """``P + q D[P]`` should match the true dilation by ``a = 1 - q`` to second order."""
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma)
    d = np.asarray(EM.dilation_generator(jnp.asarray(p, dtype=jnp.float32)), dtype=np.float64)
    for q in (0.002, 0.01):
        exact = _gaussian(G_SMALL, sigma, scale=1.0 - q)
        approx = p + q * d
        err = np.abs(approx - exact).max() / np.abs(exact).max()
        # first-order truncation: error should scale like q^2, so well under q here
        assert err < q, f"q={q}: first-order dilation error {err:.3e} not below {q}"


def test_dilation_generator_second_order_error_is_negligible_at_one_percent():
    """The production regime. ``delta * eps`` stays under ~1 %, where the neglected
    second-order term must sit below the 1e-4 fractional residual we are chasing."""
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma)
    d = np.asarray(EM.dilation_generator(jnp.asarray(p, dtype=jnp.float32)), dtype=np.float64)
    q = 0.01
    exact = _gaussian(G_SMALL, sigma, scale=1.0 - q)
    approx = p + q * d
    # measure against the PSF peak, the same normalisation the residual stacks use
    frac = np.abs(approx - exact).max() / p.max()
    assert frac < 1e-3, f"second-order dilation error {frac:.2e} of peak at q={q}"


def test_chroma_dilation_field_is_flux_neutral():
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma)
    field = np.stack([np.stack([p, p * 1.01]), np.stack([p * 0.99, p])])  # (2,2,G,G)
    d = np.asarray(EM.chroma_dilation_field(jnp.asarray(field, dtype=jnp.float32)))
    sums = np.abs(d.sum(axis=(-2, -1)))
    assert np.all(sums < 1e-4 * np.abs(d).sum(axis=(-2, -1))), f"grid sums {sums}"


def test_chroma_dilation_field_is_orthogonal_to_a_pure_rescale():
    """Gauge (2): the base-parallel component is exactly degenerate with the flux
    solve, so it must be projected out or ``eps`` owns a flat direction."""
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma).astype(np.float32)
    w = np.asarray(EM.canonical_mode_weight_grid(G_SMALL))
    d = np.asarray(EM.chroma_dilation_field(jnp.asarray(p)))
    ref = p - p.mean()
    overlap = float((d * w * ref).sum())
    norm = float(np.sqrt((w * d ** 2).sum() * (w * ref ** 2).sum())) + 1e-30
    assert abs(overlap) / norm < 1e-5, f"base overlap {overlap / norm:.2e} not projected out"


def test_chroma_dilation_field_does_not_move_the_core_centroid():
    """An even PSF perturbed by the dilation must stay core-centred, so the dilation
    cannot masquerade as a chromatic shift."""
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma).astype(np.float32)
    d = np.asarray(EM.chroma_dilation_field(jnp.asarray(p)))
    for q in (-0.02, 0.02):
        cx, cy = EM.core_centroid_xy(jnp.asarray(p + q * d))
        assert abs(float(cx)) < 1e-4, f"q={q}: core centroid x moved to {float(cx):.2e}"
        assert abs(float(cy)) < 1e-4, f"q={q}: core centroid y moved to {float(cy):.2e}"


def test_dilation_generator_axis_convention_is_x_last():
    """Last axis is detector-x, second-to-last is detector-y.

    Tested against the analytic scale derivative of an ANISOTROPIC Gaussian, which is
    what makes it a real test: swapping the two coordinate factors gives
    ``2P + y dP/dx + x dP/dy``, which agrees with the truth for a round PSF and
    disagrees badly for an elongated one. Deliberately not a second-moment test --
    the grid spans only +-7.25 px, so a sigma large enough to be clearly anisotropic
    has its second moment truncated, and that truncation swamps the effect.
    """
    x, y = _grid_coords(G_SMALL)
    sx, sy = 1.8, 1.0

    def aniso(scale: float) -> np.ndarray:
        a, b = sx * scale, sy * scale
        return np.exp(-0.5 * (x ** 2 / a ** 2 + y ** 2 / b ** 2)) / (2 * np.pi * a * b)

    p = aniso(1.0)
    da = -1e-3
    analytic = (aniso(1.0 + da) - p) / da  # equals -D[P]

    got = np.asarray(EM.dilation_generator(jnp.asarray(p, dtype=jnp.float32)), dtype=np.float64)
    m = p > 1e-6 * p.max()
    rel = np.abs(got[m] + analytic[m]).max() / np.abs(analytic[m]).max()
    assert rel < 3e-2, f"anisotropic D[P] mismatch, max rel err {rel:.3e}"

    # and the swapped-axis implementation must actually fail this, or the test is vacuous
    d_dx = np.gradient(p, axis=-1)
    d_dy = np.gradient(p, axis=-2)
    ax = np.asarray(np.arange(G_SMALL) - EM.node_center_for_grid(G_SMALL))
    swapped = 2.0 * p + ax[:, None] * d_dx + ax[None, :] * d_dy
    rel_swapped = np.abs(swapped[m] + analytic[m]).max() / np.abs(analytic[m]).max()
    assert rel_swapped > 0.2, (
        f"swapped-axis form only differs by {rel_swapped:.3e}; this test cannot "
        "detect an axis transposition, make the PSF more anisotropic"
    )


def test_dilation_generator_is_linear():
    """Linearity is what lets the generator commute with the node blend, which is the
    whole reason the fast banded renderer survives."""
    sigma = 2.0
    a = _gaussian(G_SMALL, sigma).astype(np.float32)
    b = _gaussian(G_SMALL, sigma * 1.3, x0=0.3).astype(np.float32)
    da = np.asarray(EM.dilation_generator(jnp.asarray(a)))
    db = np.asarray(EM.dilation_generator(jnp.asarray(b)))
    dab = np.asarray(EM.dilation_generator(jnp.asarray(0.4 * a + 0.6 * b)))
    assert np.allclose(dab, 0.4 * da + 0.6 * db, atol=1e-6 * np.abs(dab).max())


def test_dilation_generator_commutes_with_the_node_blend():
    """The concat trick in the renderer relies on this exactly."""
    sigma = 2.0
    rng = np.random.default_rng(0)
    field = np.stack([
        np.stack([_gaussian(G_SMALL, sigma * s) for s in (1.0, 1.05)]),
        np.stack([_gaussian(G_SMALL, sigma * s) for s in (0.95, 1.02)]),
    ]).astype(np.float32)
    w = rng.random((5, 2, 2)).astype(np.float32)
    blended_then_d = np.asarray(
        EM.dilation_generator(jnp.einsum("nrc,rcxy->nxy", jnp.asarray(w), jnp.asarray(field)))
    )
    d_then_blended = np.asarray(
        jnp.einsum("nrc,rcxy->nxy", jnp.asarray(w), EM.dilation_generator(jnp.asarray(field)))
    )
    scale = np.abs(blended_then_d).max()
    assert np.allclose(blended_then_d, d_then_blended, atol=1e-5 * scale)


# ---------------------------------------------------------------------------
# Tier 2/3: the chromatic term inside the forward model.
# ---------------------------------------------------------------------------

from syndiff_pipeline.forward_model import cheb_wcs as CW  # noqa: E402
from syndiff_pipeline.forward_model import loss as L  # noqa: E402
from syndiff_pipeline.forward_model.groups import GroupSet  # noqa: E402
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents  # noqa: E402


def _scaffold(*, uniform_nodes: bool, bp_rp=None, colour_ref=0.0, n_frames=5):
    """Small end-to-end context, mirroring test_forward_epsf_wcs's banded A/B setup.

    ``uniform_nodes`` makes every node grid identical and kills the modes, so the node
    blend becomes position-independent. That is what lets the axis test compare a
    chromatic shift against a WCS shift exactly: a WCS shift also moves the blend
    position, a chromatic one deliberately does not.
    """
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
    gsz = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:gsz, 0:gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    wing = (xx - EM.NODE_CENTER_INDEX - 3) ** 2 + (yy - EM.NODE_CENTER_INDEX + 2) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob = blob + 0.3 * np.exp(-wing / (2 * (EM.OVERSAMPLE * 2.0) ** 2)).astype(np.float32)
    blob /= blob.sum()
    base = np.zeros((2, 2, gsz, gsz), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            base[i, j] = blob if uniform_nodes else blob * (1.0 + 0.1 * i + 0.2 * j)
            base[i, j] /= base[i, j].sum()
    rng = np.random.default_rng(0)
    n_modes = 3
    if uniform_nodes:
        modes = np.zeros((n_modes, 2, 2, gsz, gsz), dtype=np.float32)
    else:
        modes = 0.05 * rng.normal(size=(n_modes, 2, 2, gsz, gsz)).astype(np.float32)
        modes -= modes.mean(axis=(-1, -2), keepdims=True)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    members = np.array([[0, 1, -1], [2, -1, -1]], dtype=int)
    valid = np.array([[True, True, False], [True, False, False]], dtype=bool)
    groups = GroupSet(2, 3, members, valid, np.array([True, True]), 0)
    ra = np.array([180.0, 180.002, 179.998], dtype=np.float32)
    dec = np.array([0.0, 0.001, -0.0015], dtype=np.float32)
    n_basis = 3
    rng2 = np.random.default_rng(2)
    wcs_fb = (0.1 * rng2.normal(size=(n_frames, n_basis))).astype(np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = (0.1 * rng2.normal(size=(n_frames, n_basis))).astype(np.float32)
    w_fb[:, 0] = 1.0

    ctx = L.build_static_context(
        cheb_static=static, wcs_frame_basis=wcs_fb, w_frame_basis=w_fb,
        epsf_grid=grid, groups=groups, ra=ra, dec=dec,
        stamp_center_x=np.array([50, 52], dtype=np.int64),
        stamp_center_y=np.array([50, 48], dtype=np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.array([1.0, 1.0], dtype=np.float32),
        fit_radius=np.array([6.0, 6.0], dtype=np.float32),
        bp_rp=bp_rp, colour_ref=colour_ref,
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis, chroma=True)
    params["wcs_coeff"] = params["wcs_coeff"].at[0, :].set(3.0)
    if not uniform_nodes:
        params["epsf_modes"] = 0.1 * jnp.asarray(
            rng2.normal(size=params["epsf_modes"].shape).astype(np.float32)
        )
    return ctx, params, static, epsf0


def test_chroma_slot_terms_evaluates_the_bilinear_field_and_maps_component0_to_x():
    """Pins the (row, col) -> (y, x) mapping of ``chroma_shift`` against
    ``bilinear_cell``. Stars sit exactly on the four nodes and at the cell centre."""
    # chroma_slot_terms is a pure function of x_lin/y_lin/chroma_delta/node_*, so the
    # scaffold's three-star geometry is replaced wholesale with five probe positions.
    colours = np.ones(5, dtype=np.float32)
    ctx, params, static, _ = _scaffold(uniform_nodes=True)
    # place five stars at (x, y) = the four nodes plus the centre
    xs = np.array([25.0, 75.0, 25.0, 75.0, 50.0], dtype=np.float32)
    ys = np.array([25.0, 25.0, 75.0, 75.0, 50.0], dtype=np.float32)
    ctx = L.replace(
        ctx,
        x_lin=jnp.asarray(xs), y_lin=jnp.asarray(ys),
        chroma_delta=jnp.asarray(colours),
    )
    node_vals_x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # [row=y][col=x]
    node_vals_y = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    params = dict(params)
    params["chroma_shift"] = jnp.asarray(np.stack([node_vals_x, node_vals_y]))

    star_occ = jnp.arange(5)
    delta, chx, chy, eps = L.chroma_slot_terms(params, ctx, star_occ)
    # node (row iy, col ix) is at (x=node_x[ix], y=node_y[iy])
    want_x = [1.0, 2.0, 3.0, 4.0, 2.5]
    want_y = [10.0, 20.0, 30.0, 40.0, 25.0]
    np.testing.assert_allclose(np.asarray(chx), want_x, atol=1e-5)
    np.testing.assert_allclose(np.asarray(chy), want_y, atol=1e-5)
    np.testing.assert_allclose(np.asarray(delta), colours, atol=1e-6)


def test_chroma_slot_terms_scales_with_colour():
    colours = np.array([0.0, 0.5, -1.0], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=True, bp_rp=colours, colour_ref=0.0)
    params = dict(params)
    params["chroma_shift"] = jnp.asarray(
        np.stack([np.full((2, 2), 0.7, np.float32), np.zeros((2, 2), np.float32)])
    )
    _, chx, chy, _ = L.chroma_slot_terms(params, ctx, jnp.arange(3))
    np.testing.assert_allclose(np.asarray(chx), 0.7 * colours, atol=1e-6)
    np.testing.assert_allclose(np.asarray(chy), 0.0, atol=1e-6)


def test_chroma_shift_moves_x_only_for_component_zero():
    """Component 0 must displace along detector-x and leave y untouched.

    Uniform nodes and zero modes, so a chromatic shift and an equal WCS constant
    shift are exactly equivalent: a symmetric PSF would hide an axis swap entirely,
    hence the asymmetric two-blob base.
    """
    shift_px = 0.35
    colours = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for comp, row in ((0, 0), (1, 1)):
        ctx, params, _, _ = _scaffold(uniform_nodes=True, bp_rp=colours, colour_ref=0.0)
        sh = np.zeros((2, 2, 2), dtype=np.float32)
        sh[comp] = shift_px
        p_chroma = dict(params)
        p_chroma["chroma_shift"] = jnp.asarray(sh)
        t_chroma, *_ = L.forward_model(p_chroma, ctx)

        # reference: same displacement applied through the WCS constant term
        p_ref = {k: v for k, v in params.items() if k not in L.CHROMA_LEAVES}
        n_terms = ctx.n_terms
        r = 0 if comp == 0 else n_terms
        p_ref["wcs_coeff"] = p_ref["wcs_coeff"].at[r, 0].add(shift_px)
        t_ref, *_ = L.forward_model(p_ref, ctx)

        peak = float(np.abs(np.asarray(t_ref)).max())
        rel = float(np.abs(np.asarray(t_chroma) - np.asarray(t_ref)).max() / peak)
        assert rel < 2e-3, f"component {comp}: chromatic shift != WCS shift, rel {rel:.2e}"

        # and it must NOT match the other axis
        p_wrong = {k: v for k, v in params.items() if k not in L.CHROMA_LEAVES}
        r_other = n_terms if comp == 0 else 0
        p_wrong["wcs_coeff"] = p_wrong["wcs_coeff"].at[r_other, 0].add(shift_px)
        t_wrong, *_ = L.forward_model(p_wrong, ctx)
        rel_wrong = float(np.abs(np.asarray(t_chroma) - np.asarray(t_wrong)).max() / peak)
        assert rel_wrong > 1e-2, (
            f"component {comp}: the swapped-axis reference also matches "
            f"(rel {rel_wrong:.2e}); this test cannot detect a transposition"
        )


def test_chroma_zero_leaves_are_identical_to_no_chroma():
    colours = np.array([0.3, -0.4, 1.1], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.2)
    t_on, *_ = L.forward_model(params, ctx)  # leaves present but zero
    p_off = {k: v for k, v in params.items() if k not in L.CHROMA_LEAVES}
    t_off, *_ = L.forward_model(p_off, ctx)
    # Pixel-integrated representation update: the chroma-on-but-zero path folds
    # a (zero) COM/shift correction through render_band's convolution, while
    # the chroma-off path calls render_stamps directly -- two DIFFERENT, both
    # exact, closed forms that happen not to be bit-identical for a 2-tap
    # kernel (they were, incidentally, for the old 5-tap kernel). Max observed
    # diff is float32 ULP-level (~1e-7 relative, ~1e-10 absolute) -- physically
    # negligible; same reasoning as test_chroma_banded_matches_unbanded's
    # rel<=1e-5 tolerance between two other equally-valid code paths.
    np.testing.assert_allclose(np.asarray(t_on), np.asarray(t_off), rtol=1e-6, atol=1e-9)


def test_chroma_banded_matches_unbanded():
    """THE test for the node_moments fix.

    The banded hot path folds the dilation onto the node-row axis and reuses the same
    weights for ``node_moments``; the unbanded path perturbs ``local`` and then calls
    ``recenter_grid_core`` on it. If the hot path left ``node_moments`` on the
    unperturbed grid, the two disagree here and nowhere else.
    """
    colours = np.array([0.4, -0.6, 1.3], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.35)
    rng = np.random.default_rng(7)
    params = dict(params)
    params["chroma_shift"] = jnp.asarray(
        (0.03 * rng.normal(size=(2, 2, 2))).astype(np.float32)
    )
    params["chroma_dilation"] = jnp.asarray(
        (0.02 + 0.005 * rng.normal(size=(2, 2))).astype(np.float32)
    )
    saved = L.USE_BANDED_RENDER
    try:
        L.USE_BANDED_RENDER = False
        t_old, *_ = L.forward_model(params, ctx)
        L.USE_BANDED_RENDER = True
        t_new, *_ = L.forward_model(params, ctx)
    finally:
        L.USE_BANDED_RENDER = saved
    t_old = np.asarray(t_old)
    t_new = np.asarray(t_new)
    peak = np.abs(t_old).max()
    assert peak > 0
    rel = np.abs(t_old - t_new).max() / peak
    assert rel <= 1e-5, f"banded vs unbanded chroma mismatch, rel {rel:.3e}"


def test_chroma_actually_changes_the_render():
    """Guard against a no-op: the above equality tests would all pass trivially if the
    chromatic term were silently dropped on both paths."""
    colours = np.array([0.4, -0.6, 1.3], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.35)
    t0, *_ = L.forward_model(params, ctx)
    p = dict(params)
    p["chroma_shift"] = jnp.asarray(np.full((2, 2, 2), 0.02, np.float32))
    p["chroma_dilation"] = jnp.asarray(np.full((2, 2), 0.02, np.float32))
    t1, *_ = L.forward_model(p, ctx)
    peak = float(np.abs(np.asarray(t0)).max())
    rel = float(np.abs(np.asarray(t1) - np.asarray(t0)).max() / peak)
    assert rel > 1e-3, f"chromatic term had no effect on the render (rel {rel:.2e})"


def test_chroma_without_colour_in_context_raises():
    ctx, params, _, _ = _scaffold(uniform_nodes=True, bp_rp=None)
    assert ctx.chroma_delta is None
    with pytest.raises(ValueError, match="no colour"):
        L.forward_model(params, ctx)


def test_chroma_nan_colour_becomes_zero_delta():
    colours = np.array([0.5, np.nan, 1.0], dtype=np.float32)
    ctx, _, _, _ = _scaffold(uniform_nodes=True, bp_rp=colours, colour_ref=0.5)
    d = np.asarray(ctx.chroma_delta)
    assert np.all(np.isfinite(d))
    assert d[1] == 0.0
    np.testing.assert_allclose(d[[0, 2]], [0.0, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# Tier 4: chunking, AD, and the anchor gradient policy.
# ---------------------------------------------------------------------------

def _chroma_fixture(n_frames=12, seed=0, *, contributor=True):
    """The existing stamp_chunk fixture, with colour and non-zero chromatic leaves."""
    from test_fm_forward_epsf_wcs import _stamp_chunk_fixture

    fx = _stamp_chunk_fixture(n_frames=n_frames, seed=seed)
    ctx = fx["ctx"]
    n_stars = int(ctx.ra.shape[0])
    rng = np.random.default_rng(seed + 100)
    colours = rng.uniform(-0.6, 0.8, size=n_stars).astype(np.float32)
    colours = colours - colours.mean()  # the c_ref gauge
    ctx = L.replace(ctx, chroma_delta=jnp.asarray(colours))
    if not contributor:
        ctx = L.replace(
            ctx, is_epsf_contributor=jnp.zeros_like(ctx.is_epsf_contributor, dtype=bool)
        )
    params = dict(fx["params"])
    params["chroma_shift"] = jnp.asarray((0.03 * rng.normal(size=(2, 2, 2))).astype(np.float32))
    params["chroma_dilation"] = jnp.asarray(
        (0.02 + 0.005 * rng.normal(size=(2, 2))).astype(np.float32)
    )
    fx = dict(fx)
    fx["ctx"] = ctx
    fx["params"] = params
    return fx


def _loss_of(fx, params, stamp_chunk=None):
    loss, _ = L.total_loss(
        params, fx["ctx"], fx["data"], fx["noise"], fx["weight"],
        fx["wcs_second_diff"], fx["w_second_diff"],
        epsf_modes_init=fx["epsf_modes_init"],
        stamp_active=fx["stamp_active"],
        stamp_chunk=stamp_chunk,
    )
    return loss


def test_chroma_chunked_matches_unchunked():
    """The chunked path carries the chromatic shift through its own padding and
    blocking of the position arrays, so it needs its own equality check."""
    fx = _chroma_fixture(n_frames=12)
    ref = float(_loss_of(fx, fx["params"], None))
    for block in (2, 4, 12):
        got = float(_loss_of(fx, fx["params"], block))
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_chroma_gradients_match_finite_differences():
    fx = _chroma_fixture(n_frames=6)
    params = fx["params"]
    grads = jax.grad(lambda p: _loss_of(fx, p))(params)
    for key in L.CHROMA_LEAVES:
        g = np.asarray(grads[key])
        assert np.all(np.isfinite(g)), f"{key} gradient is not finite"
        assert np.abs(g).max() > 0, f"{key} received no gradient"
        # central difference on the single largest-gradient entry
        idx = np.unravel_index(np.argmax(np.abs(g)), g.shape)
        h = 1e-3
        base = np.asarray(params[key], dtype=np.float64)
        up, dn = base.copy(), base.copy()
        up[idx] += h
        dn[idx] -= h
        p_up = dict(params); p_up[key] = jnp.asarray(up.astype(np.float32))
        p_dn = dict(params); p_dn[key] = jnp.asarray(dn.astype(np.float32))
        fd = (float(_loss_of(fx, p_up)) - float(_loss_of(fx, p_dn))) / (2 * h)
        rel = abs(fd - g[idx]) / max(abs(fd), 1e-12)
        assert rel < 5e-2, f"{key}{idx}: AD {g[idx]:.6g} vs FD {fd:.6g}, rel {rel:.2e}"


def test_anchors_train_chroma_but_not_the_epsf():
    """Codifies the deliberate policy in ``_epsf_params_for_context``: faint WCS
    anchors drive the colour term even though the ePSF is stop-gradient for them.

    The invariant is that chroma opens NO new gradient path into the ePSF leaves for
    an anchor batch, not that those gradients are exactly zero. They are not: the
    regularisation penalties act on the raw leaves directly, outside the anchor
    barrier, so anchors carry a small pre-existing penalty-only gradient. That is
    pre-existing behaviour and is asserted here as an equality, which is the
    statement that actually matters.
    """
    fx = _chroma_fixture(n_frames=6, contributor=False)
    with_chroma = dict(fx["params"])
    without = {k: v for k, v in with_chroma.items() if k not in L.CHROMA_LEAVES}
    g_on = jax.grad(lambda p: _loss_of(fx, p))(with_chroma)
    g_off = jax.grad(lambda p: _loss_of(fx, p))(without)
    for key in ("epsf_base_raw", "epsf_modes", "w_coeff"):
        np.testing.assert_allclose(
            np.asarray(g_on[key]), np.asarray(g_off[key]), rtol=0, atol=0,
            err_msg=f"chroma opened a gradient path into {key} for anchor groups",
        )
    for key in L.CHROMA_LEAVES:
        assert np.abs(np.asarray(g_on[key])).max() > 0, (
            f"anchors produced no gradient for {key}; they are meant to drive it"
        )


def test_contributors_still_train_the_epsf():
    """Sanity companion to the above: the anchor barrier must not have leaked into
    the contributor path."""
    fx = _chroma_fixture(n_frames=6, contributor=True)
    grads = jax.grad(lambda p: _loss_of(fx, p))(fx["params"])
    assert np.abs(np.asarray(grads["epsf_base_raw"])).max() > 1e-8


def test_chroma_recovers_an_injected_signal():
    """Identifiability: the loss must be lower at the truth than at zero, and the
    gradient at the truth must be small."""
    fx = _chroma_fixture(n_frames=8, seed=3)
    truth = fx["params"]
    # synthesise data from the model itself at the true chromatic parameters
    templates, *_ = L.forward_model(truth, fx["ctx"], stamp_active=fx["stamp_active"])
    flux = FS.solve_fluxes(templates, fx["data"], fx["weight"])
    synth = np.asarray(FS.model_stamps(templates, flux))
    fx = dict(fx)
    fx["data"] = jnp.asarray(synth)

    zero = dict(truth)
    for key in L.CHROMA_LEAVES:
        zero[key] = jnp.zeros_like(truth[key])
    assert float(_loss_of(fx, truth)) < float(_loss_of(fx, zero)), (
        "the true chromatic parameters do not fit the data they generated"
    )


from syndiff_pipeline.forward_model import flux_solve as FS  # noqa: E402


# ---------------------------------------------------------------------------
# The PACKED path. Production runs packed tiers, so this is the one that counts.
# ---------------------------------------------------------------------------

def _packed_scaffold(*, seed=0, n_frames=3):
    """Packed irregular-support context with colour, asymmetric base and live modes."""
    from syndiff_pipeline.forward_model.groups import GroupSet

    degree, region = 1, 64
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0, cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]]),
        crpix=np.array([region / 2, region / 2]), center=np.array([region / 2, region / 2]),
        half_extents=np.array([region / 2, region / 2]), poly_degree=degree,
        exponents=tuple(sci2idl_exponents(degree)),
    )
    grid = EM.EpsfGridStatic.from_region(
        type("Region", (), {"x_min": 0, "x_max": region, "y_min": 0, "y_max": region})(),
        n_rows=2, n_cols=2, crop_origin=(0, 0),
    )
    g_size = EM.node_geometry(5)[1]
    yy, xx = np.mgrid[0:g_size, 0:g_size]
    c = (g_size - 1) / 2
    blob = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / 18.0).astype(np.float32)
    # asymmetric wing, so the core-centroid fold is genuinely exercised
    blob += 0.3 * np.exp(-((xx - c - 2.5) ** 2 + (yy - c + 1.5) ** 2) / 30.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    base = np.zeros((2, 2, g_size, g_size), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            base[i, j] = blob * (1.0 + 0.1 * i + 0.2 * j)
            base[i, j] /= base[i, j].sum()
    modes = 0.03 * rng.normal(size=(1, 2, 2, g_size, g_size)).astype(np.float32)
    modes -= modes.mean(axis=(-1, -2), keepdims=True)
    epsf = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    n_stars = 4
    ra = np.array([179.95, 179.98, 180.02, 180.05], dtype=np.float32)
    dec = np.array([-0.04, -0.01, 0.02, 0.05], dtype=np.float32)
    x_lin, y_lin, cheb_basis = CW.star_basis(jnp.asarray(ra), jnp.asarray(dec), static)
    groups = GroupSet(
        n_stars, 1, np.arange(n_stars, dtype=np.int32)[:, None],
        np.ones((n_stars, 1), bool), np.ones(n_stars, bool), 0,
    )
    P = 49
    pix_x = np.zeros((n_stars, P), np.float32)
    pix_y = np.zeros((n_stars, P), np.float32)
    for i, (cx, cy) in enumerate(zip(np.asarray(x_lin), np.asarray(y_lin))):
        py, px = np.mgrid[-3:4, -3:4]
        pix_x[i] = (round(float(cx)) + px).ravel()
        pix_y[i] = (round(float(cy)) + py).ravel()
    colours = rng.uniform(-0.5, 0.7, size=n_stars).astype(np.float32)
    colours -= colours.mean()
    ctx = L.build_static_context(
        cheb_static=static, wcs_frame_basis=np.eye(n_frames, 2, dtype=np.float32),
        w_frame_basis=np.eye(n_frames, 2, dtype=np.float32), epsf_grid=grid, groups=groups,
        ra=ra, dec=dec, stamp_center_x=np.asarray(x_lin), stamp_center_y=np.asarray(y_lin),
        t_exp_sec=1.0, stamp_snr_weight=np.ones(n_stars, np.float32),
        fit_radius=np.full(n_stars, 4.0, np.float32),
        x_lin=np.asarray(x_lin), y_lin=np.asarray(y_lin), cheb_basis=np.asarray(cheb_basis),
        pix_x=pix_x, pix_y=pix_y, pix_valid=np.ones((n_stars, P), np.float32),
        is_epsf_contributor=np.ones(n_stars, bool),
        bp_rp=colours, colour_ref=0.0,
    )
    assert ctx.is_packed
    params = L.init_params(static, epsf, n_wcs_basis=2, n_w_basis=2, chroma=True)
    params["chroma_shift"] = jnp.asarray((0.04 * rng.normal(size=(2, 2, 2))).astype(np.float32))
    params["chroma_dilation"] = jnp.asarray(
        (0.02 + 0.006 * rng.normal(size=(2, 2))).astype(np.float32)
    )
    return ctx, params


def test_packed_chroma_banded_matches_unbanded():
    """Packed twin of the square A/B test. Same purpose: an unperturbed
    ``node_moments`` on the banded path shows up here and nowhere else."""
    ctx, params = _packed_scaffold()
    saved = L.USE_BANDED_RENDER_PACKED
    try:
        L.USE_BANDED_RENDER_PACKED = False
        t_old, *_ = L.forward_model(params, ctx)
        L.USE_BANDED_RENDER_PACKED = True
        t_new, *_ = L.forward_model(params, ctx)
    finally:
        L.USE_BANDED_RENDER_PACKED = saved
    t_old, t_new = np.asarray(t_old), np.asarray(t_new)
    peak = np.abs(t_old).max()
    assert peak > 0
    rel = np.abs(t_old - t_new).max() / peak
    assert rel <= 1e-5, f"packed banded vs unbanded chroma mismatch, rel {rel:.3e}"


def test_packed_chroma_changes_the_render():
    ctx, params = _packed_scaffold()
    zero = dict(params)
    for k in L.CHROMA_LEAVES:
        zero[k] = jnp.zeros_like(params[k])
    t0, *_ = L.forward_model(zero, ctx)
    t1, *_ = L.forward_model(params, ctx)
    peak = float(np.abs(np.asarray(t0)).max())
    rel = float(np.abs(np.asarray(t1) - np.asarray(t0)).max() / peak)
    assert rel > 1e-3, f"packed chromatic term had no effect (rel {rel:.2e})"


def test_packed_chroma_shift_does_not_move_the_node_blend():
    """On the packed path ``x_occ`` feeds BOTH the node blend and the render offset,
    so the chromatic shift has to be threaded through a separate array. This is the
    test that it was.

    Exact reference: displacing the star by +1 px is identical to displacing the pixel
    list by -1 px, and the pixel-list version leaves ``x_occ`` -- and therefore the
    node blend -- untouched by construction. An integer displacement is required
    because the packed banded renderer relies on ``pix - ref`` being an exact integer.
    If the chromatic shift leaked into ``bilinear_cell``, the two would disagree
    wherever the node grids differ, which they do here.
    """
    ctx, params = _packed_scaffold()
    delta0 = float(np.asarray(ctx.chroma_delta)[0])
    # Sanity guard only (star 0's colour offset must be comfortably nonzero so
    # shift_px/delta0 below is well-conditioned) -- the exact value is
    # incidental: _packed_scaffold's g_size = EM.node_geometry(5)[1] is now
    # the pixel-integrated size (23, was the legacy sub-pixel size 26), so the
    # shared `rng` draws a differently-shaped `modes` array before `colours`,
    # shifting every subsequent draw from the same rng (delta0 was ~0.12,
    # is now ~0.021). Threshold lowered accordingly, not chasing the old value.
    assert abs(delta0) > 0.01
    shift_px = 1.0
    p = dict(params)
    sh = np.zeros((2, 2, 2), np.float32)
    sh[0] = shift_px / delta0  # star 0 ends up displaced by exactly +1 px in x
    p["chroma_shift"] = jnp.asarray(sh)
    p["chroma_dilation"] = jnp.zeros_like(params["chroma_dilation"])
    t_chroma, *_ = L.forward_model(p, ctx)

    ref_params = {k: v for k, v in params.items() if k not in L.CHROMA_LEAVES}
    ctx_ref = L.replace(ctx, pix_x=ctx.pix_x - shift_px)
    t_ref, *_ = L.forward_model(ref_params, ctx_ref)

    a = np.asarray(t_chroma)[0]
    b = np.asarray(t_ref)[0]
    peak = np.abs(b).max()
    assert peak > 0
    rel = np.abs(a - b).max() / peak
    assert rel < 1e-5, (
        f"packed chromatic shift is not a pure translation, rel {rel:.2e}; "
        "it has most likely leaked into the node blend"
    )


# ---------------------------------------------------------------------------
# Checkpoint and optimizer plumbing.
# ---------------------------------------------------------------------------

def test_chroma_leaf_roundtrip_and_backward_compatible_load(tmp_path):
    from syndiff_pipeline.forward_model import fit as FIT

    _, params = _packed_scaffold()
    p = tmp_path / "params.npz"
    FIT.save_params_npz(p, {k: np.asarray(v) for k, v in params.items()})
    back = FIT.load_params_npz(p)
    for key in L.CHROMA_LEAVES:
        assert key in back
        np.testing.assert_allclose(np.asarray(back[key]), np.asarray(params[key]), atol=0)

    # a pre-chroma checkpoint must still load, with the optional leaves simply absent
    legacy = {k: np.asarray(v) for k, v in params.items() if k not in L.CHROMA_LEAVES}
    q = tmp_path / "legacy.npz"
    FIT.save_params_npz(q, legacy)
    back_legacy = FIT.load_params_npz(q)
    assert set(back_legacy) == set(FIT.STAGE_LEAVES)


def test_optimizer_accepts_params_with_and_without_chroma():
    """optax.multi_transform requires the label tree to match the params tree
    exactly, so both key sets must build and step cleanly."""
    from syndiff_pipeline.forward_model import fit as FIT

    _, params = _packed_scaffold()
    for keys in (tuple(params), tuple(FIT.STAGE_LEAVES)):
        p = {k: params[k] for k in keys}
        tx = FIT.make_stage_optimizer(3, 1e-4, param_keys=keys)
        state = tx.init(p)
        grads = {k: jnp.ones_like(v) for k, v in p.items()}
        updates, _ = tx.update(grads, state, p)
        assert set(updates) == set(p)


def test_chroma_is_frozen_before_stage_three():
    """The colour term is strongly correlated with the WCS; only the star-to-star
    colour spread separates them. It must not train before the WCS and base settle."""
    from syndiff_pipeline.forward_model import fit as FIT

    _, params = _packed_scaffold()
    keys = tuple(params)
    for stage in (1, 2):
        labels = FIT._leaf_labels(stage, freeze_wcs=False, param_keys=keys)
        for k in L.CHROMA_LEAVES:
            assert labels[k] == "frozen", f"stage {stage}: {k} is {labels[k]}"
        tx = FIT.make_stage_optimizer(stage, 1e-4, param_keys=keys)
        state = tx.init(params)
        grads = {k: jnp.ones_like(v) for k, v in params.items()}
        updates, _ = tx.update(grads, state, params)
        for k in L.CHROMA_LEAVES:
            assert float(np.abs(np.asarray(updates[k])).max()) == 0.0
    labels3 = FIT._leaf_labels(3, freeze_wcs=False, param_keys=keys)
    for k in L.CHROMA_LEAVES:
        assert labels3[k] == "train_chroma"


def test_stop_grad_freezes_chroma_when_labels_default():
    """stop_grad_frozen_params looks labels up by key, so the default label set must
    be a SUPERSET of the params keys or a frozen optional leaf stays trainable."""
    from syndiff_pipeline.forward_model import fit as FIT

    _, params = _packed_scaffold()
    labels = FIT._leaf_labels(1, freeze_wcs=False)  # no param_keys given
    for k in L.CHROMA_LEAVES:
        assert labels.get(k) == "frozen", f"{k} missing from the default label set"
    out = FIT.stop_grad_frozen_params(params, labels)
    g = jax.grad(lambda p: jnp.sum(FIT.stop_grad_frozen_params(p, labels)["chroma_shift"] ** 2))
    assert float(np.abs(np.asarray(g(params)["chroma_shift"])).max()) == 0.0


# ---------------------------------------------------------------------------
# End to end: bundle -> stages -> checkpoint, which is where a silent reset hides.
# ---------------------------------------------------------------------------

def test_bundle_carries_bp_rp_through_save_and_load(tmp_path):
    from syndiff_pipeline.forward_model import fit_bundle as FB
    from test_fm_fit_bundle import _tiny_bundle

    b = _tiny_bundle()
    n_stars = int(np.asarray(b.ra).shape[0])
    colours = np.linspace(0.4, 1.4, n_stars)
    colours[0] = np.nan  # a missing Gaia colour must survive as NaN in the bundle
    b.bp_rp = colours
    path = FB.save_fit_bundle(tmp_path / "fit_bundle.npz", b)
    loaded = FB.load_fit_bundle(path)
    assert loaded.bp_rp is not None
    np.testing.assert_allclose(loaded.bp_rp, colours, equal_nan=True)

    # and a bundle without colour still loads, with the field simply None
    b2 = _tiny_bundle()
    loaded2 = FB.load_fit_bundle(FB.save_fit_bundle(tmp_path / "plain.npz", b2))
    assert loaded2.bp_rp is None


def test_end_to_end_chroma_trains_and_survives_the_stage_handoff(tmp_path):
    """The whole point of the handoff fix. Chroma must be frozen through stages 1-2,
    train in stage 3, and still be non-zero in the stage-3 checkpoint -- which is the
    only channel between isolated stages."""
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model import fit_bundle as FB
    from syndiff_pipeline.forward_model import train_loop as TL
    from test_fm_fit_bundle import _tiny_bundle

    b = _tiny_bundle()
    n_stars = int(np.asarray(b.ra).shape[0])
    b.bp_rp = np.linspace(0.4, 1.4, n_stars)
    loaded = FB.load_fit_bundle(FB.save_fit_bundle(tmp_path / "fit_bundle.npz", b))
    out = tmp_path / "out"
    TL.run_stages_from_bundle(
        loaded, out_dir=out, stage=3, start_stage=1,
        steps_per_stage=[1, 1, 3], lr_per_stage=[1e-2, 3e-4, 1e-2],
        log_every=1, checkpoint_every=0, reject_every=0, stage1_core_stamp=0,
        chroma=True,
    )
    p1 = FIT.load_params_npz(out / "params_stage1.npz")
    p3 = FIT.load_params_npz(out / "params_stage3.npz")
    for key in FIT.OPTIONAL_LEAVES:
        assert key in p1, f"{key} missing from the stage-1 checkpoint"
        assert float(np.abs(np.asarray(p1[key])).max()) == 0.0, (
            f"{key} trained during stage 1; it must be frozen until stage 3"
        )
        assert key in p3, f"{key} was dropped by the stage handoff"
    # NOTE: this fixture's loss is exactly flat (every leaf, including wcs_coeff,
    # gets a zero gradient) because it exists to exercise plumbing, so "did chroma
    # move" cannot be asserted here. That is covered by the finite-difference and
    # optimizer tests above, and by the resume test below.


def test_isolated_stage_handoff_preserves_non_zero_chroma(tmp_path):
    """The exact failure mode the handoff fix targets.

    Isolated stages run in fresh processes and pass parameters ONLY through
    --init-params. A hardcoded four-key merge silently resets chroma to zero at every
    boundary, which in a real run looks indistinguishable from "the colour term never
    converged". Inject a known non-zero value and require it to survive.
    """
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model import fit_bundle as FB
    from syndiff_pipeline.forward_model import train_loop as TL
    from test_fm_fit_bundle import _tiny_bundle

    b = _tiny_bundle()
    n_stars = int(np.asarray(b.ra).shape[0])
    b.bp_rp = np.linspace(0.4, 1.4, n_stars)
    loaded = FB.load_fit_bundle(FB.save_fit_bundle(tmp_path / "fit_bundle.npz", b))

    seed = dict(FB.params0_as_jnp(loaded))
    base = np.asarray(loaded.epsf_base)
    nr, nc = int(base.shape[0]), int(base.shape[1])
    inject_shift = (0.011 * np.arange(2 * nr * nc).reshape(2, nr, nc)).astype(np.float32)
    inject_dil = (0.021 + 0.001 * np.arange(nr * nc).reshape(nr, nc)).astype(np.float32)
    seed["chroma_shift"] = inject_shift
    seed["chroma_dilation"] = inject_dil
    seed_path = tmp_path / "seed.npz"
    FIT.save_params_npz(seed_path, {k: np.asarray(v) for k, v in seed.items()})

    out = tmp_path / "out"
    TL.run_stages_from_bundle(
        loaded, out_dir=out, stage=3, start_stage=3,
        steps_per_stage=[0, 0, 1], lr_per_stage=[1e-2, 3e-4, 1e-4],
        log_every=1, checkpoint_every=0, reject_every=0, stage1_core_stamp=0,
        chroma=True, init_params=seed_path,
    )
    got = FIT.load_params_npz(out / "params_stage3.npz")
    for key, want in (("chroma_shift", inject_shift), ("chroma_dilation", inject_dil)):
        assert key in got, f"{key} was dropped by the --init-params merge"
        arr = np.asarray(got[key])
        assert np.abs(arr).max() > 0.0, (
            f"{key} came back as all zeros: the merge reset it, which is exactly the "
            "bug this test exists for"
        )
        # a single tiny-lr step cannot move it far from the injected value
        np.testing.assert_allclose(arr, want, atol=1e-2)


def test_chroma_requires_a_bundle_with_colour(tmp_path):
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import fit_bundle as FB
    from syndiff_pipeline.forward_model import train_loop as TL
    from test_fm_fit_bundle import _tiny_bundle

    loaded = FB.load_fit_bundle(FB.save_fit_bundle(tmp_path / "b.npz", _tiny_bundle()))
    with pytest.raises(SystemExit, match="bp_rp"):
        TL.run_stages_from_bundle(
            loaded, out_dir=tmp_path / "o", stage=1, start_stage=1,
            steps_per_stage=[1, 0, 0], lr_per_stage=[1e-2, 3e-4, 1e-4],
            log_every=1, checkpoint_every=0, reject_every=0, stage1_core_stamp=0,
            chroma=True,
        )


# ---------------------------------------------------------------------------
# Every params-serialisation path must carry the optional leaves.
# ---------------------------------------------------------------------------

def test_every_params_writer_round_trips_the_optional_leaves(tmp_path):
    """A leaf that survives training but not serialisation is worse than one that
    never trained: the run looks fine and the value is silently lost on resume.

    This enumerates every path that writes a parameter set. `save_params_leaves_npz`
    failed this when it was written -- it took only the four required leaves, and
    those rolling checkpoints are a legitimate `--bootstrap-init-params` source (the
    r8 -> r9 chain used one), so a resume would have reset chroma to zero.
    """
    import jax
    from syndiff_pipeline.forward_model import checkpoint_history as CH
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model import training_state as TS

    _, params = _packed_scaffold()
    params = {k: jnp.asarray(v) for k, v in params.items()}
    for k in L.CHROMA_LEAVES:
        assert float(np.abs(np.asarray(params[k])).max()) > 0, "fixture must be non-zero"

    # 1. the full checkpoint written at every stage boundary and for params_latest
    p1 = tmp_path / "full.npz"
    FIT.save_params_npz(p1, {k: np.asarray(v) for k, v in params.items()})
    back = FIT.load_params_npz(p1)
    for k in L.CHROMA_LEAVES:
        np.testing.assert_allclose(np.asarray(back[k]), np.asarray(params[k]), atol=0)

    # 2. the slim rolling checkpoint written every --checkpoint-every steps
    p2 = tmp_path / "rolling.npz"
    CH.save_params_leaves_npz(p2, params)
    got = np.load(p2)
    for k in L.CHROMA_LEAVES:
        assert k in got.files, f"{k} dropped by the rolling checkpoint writer"
        np.testing.assert_allclose(got[k], np.asarray(params[k]), atol=0)
    # and it must still load through the normal reader
    for k in L.CHROMA_LEAVES:
        assert k in FIT.load_params_npz(p2)

    # 3. the resume state, which also carries optimizer leaves
    tx = FIT.make_stage_optimizer(3, 1e-4, param_keys=tuple(params))
    p3 = tmp_path / "state.npz"
    TS.save(p3, params=params, opt_state=tx.init(params), metadata={"step": 1})
    back3, _opt, _meta = TS.load(p3)
    for k in L.CHROMA_LEAVES:
        assert k in back3, f"{k} dropped by training_state.save"
        np.testing.assert_allclose(back3[k], np.asarray(params[k]), atol=0)


def test_rolling_checkpoint_is_a_valid_bootstrap_source(tmp_path):
    """The r8 -> r9 chain bootstrapped from `checkpoints/params_s3_stepNNNNN.npz`, so
    that file has to be a complete restart point, not a subset."""
    from syndiff_pipeline.forward_model import checkpoint_history as CH
    from syndiff_pipeline.forward_model import fit as FIT

    _, params = _packed_scaffold()
    p = tmp_path / "params_s3_step00100.npz"
    CH.save_params_leaves_npz(p, {k: np.asarray(v) for k, v in params.items()})
    loaded = FIT.load_params_npz(p)
    assert set(loaded) == set(params), (
        f"rolling checkpoint is not a complete restart point; missing "
        f"{set(params) - set(loaded)}"
    )


# ---------------------------------------------------------------------------
# C1: colour-AFFINE extension (chroma_aniso / chroma_shear) plus the optional
# chroma_kurt leaf. Mirrors the Tier-1 dilation-generator tests at the top of
# this file, then the forward-model bit-identity/render/gradient/checkpoint
# tests established for shift+dilation.
# ---------------------------------------------------------------------------


def test_aniso_generator_matches_analytic_scale_derivative():
    """``x*Px - y*Py`` is minus the derivative of the area-preserving anisotropic
    scale ``P_a(x,y) = (1-a^2)^-1 P(x/(1+a), y/(1-a))`` at ``a=0``.

    Derivation: ``d/da P_a|_0 = Px*(-x) + Py*(y)`` (the ``(1-a^2)^-1`` prefactor's
    own derivative vanishes at ``a=0``), i.e. ``-(x*Px - y*Py)``.
    """
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma)
    x, y = _grid_coords(G_SMALL)
    da = -1e-3

    def p_a(a):
        xs, ys = x / (1 + a), y / (1 - a)
        s = sigma
        g = np.exp(-0.5 * (xs ** 2 + ys ** 2) / s ** 2) / (2 * np.pi * s ** 2)
        return g / (1 - a ** 2)

    analytic = (p_a(da) - p) / da  # equals -B_aniso at a=0

    got = np.asarray(EM.aniso_generator(jnp.asarray(p, dtype=jnp.float32)), dtype=np.float64)
    m = p > 1e-6 * p.max()
    assert m.sum() > 200
    num = np.abs(got[m] + analytic[m]).max()
    den = np.abs(analytic[m]).max()
    assert num / den < 3e-2, f"aniso_generator != -dP_a/da, max rel err {num / den:.3e}"


def test_shear_generator_matches_analytic_scale_derivative():
    """``x*Py + y*Px`` is minus the derivative of the symmetric shear
    ``P_s(x,y) ~= P(x - s*y, y - s*x)`` at ``s=0``: ``d/ds P_s|_0 = -(y*Px + x*Py)``.
    """
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma)
    x, y = _grid_coords(G_SMALL)
    ds = -1e-3

    def p_s(s):
        xs, ys = x - s * y, y - s * x
        g = np.exp(-0.5 * (xs ** 2 + ys ** 2) / sigma ** 2) / (2 * np.pi * sigma ** 2)
        return g / (1 - s ** 2)

    analytic = (p_s(ds) - p) / ds  # equals -B_shear at s=0

    got = np.asarray(EM.shear_generator(jnp.asarray(p, dtype=jnp.float32)), dtype=np.float64)
    m = p > 1e-6 * p.max()
    num = np.abs(got[m] + analytic[m]).max()
    den = np.abs(analytic[m]).max()
    assert num / den < 3e-2, f"shear_generator != -dP_s/ds, max rel err {num / den:.3e}"


def test_kurt_generator_is_flux_neutral_before_gauging():
    """``r^2*P - ratio*P`` should already sum to ~0 by construction (``ratio`` is
    exactly the mean-weighted r^2), well before ``chroma_kurt_field``'s explicit gauge."""
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma).astype(np.float32)
    d = np.asarray(EM.kurt_generator(jnp.asarray(p)))
    assert abs(float(d.sum())) < 1e-3 * float(np.abs(d).sum())


@pytest.mark.parametrize("field_fn", [EM.chroma_aniso_field, EM.chroma_shear_field, EM.chroma_kurt_field])
def test_chroma_affine_field_is_flux_neutral(field_fn):
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma)
    field = np.stack([np.stack([p, p * 1.01]), np.stack([p * 0.99, p])])  # (2,2,G,G)
    d = np.asarray(field_fn(jnp.asarray(field, dtype=jnp.float32)))
    sums = np.abs(d.sum(axis=(-2, -1)))
    assert np.all(sums < 1e-4 * np.abs(d).sum(axis=(-2, -1))), f"grid sums {sums}"


@pytest.mark.parametrize("field_fn", [EM.chroma_aniso_field, EM.chroma_shear_field, EM.chroma_kurt_field])
def test_chroma_affine_field_is_orthogonal_to_a_pure_rescale(field_fn):
    """Gauge (2), same as ``chroma_dilation_field``: the base-parallel component is
    exactly degenerate with the flux solve and must be projected out."""
    sigma = 2.0
    p = _gaussian(G_SMALL, sigma).astype(np.float32)
    w = np.asarray(EM.canonical_mode_weight_grid(G_SMALL))
    d = np.asarray(field_fn(jnp.asarray(p)))
    ref = p - p.mean()
    overlap = float((d * w * ref).sum())
    norm = float(np.sqrt((w * d ** 2).sum() * (w * ref ** 2).sum())) + 1e-30
    assert abs(overlap) / norm < 1e-5, f"base overlap {overlap / norm:.2e} not projected out"


def test_aniso_and_shear_generators_are_linear():
    sigma = 2.0
    a = _gaussian(G_SMALL, sigma).astype(np.float32)
    b = _gaussian(G_SMALL, sigma * 1.3, x0=0.3).astype(np.float32)
    for gen in (EM.aniso_generator, EM.shear_generator):
        da = np.asarray(gen(jnp.asarray(a)))
        db = np.asarray(gen(jnp.asarray(b)))
        dab = np.asarray(gen(jnp.asarray(0.4 * a + 0.6 * b)))
        assert np.allclose(dab, 0.4 * da + 0.6 * db, atol=1e-6 * np.abs(dab).max())


def _add_affine_leaves(params: dict, *, seed: int = 11, kurt: bool = False) -> dict:
    """Non-zero chroma_aniso/chroma_shear[/chroma_kurt] on top of an existing
    chroma-enabled params dict, same node-grid shape as chroma_dilation."""
    rng = np.random.default_rng(seed)
    shape = tuple(np.asarray(params["chroma_dilation"]).shape)
    out = dict(params)
    out["chroma_aniso"] = jnp.asarray((0.02 + 0.006 * rng.normal(size=shape)).astype(np.float32))
    out["chroma_shear"] = jnp.asarray((0.015 * rng.normal(size=shape)).astype(np.float32))
    if kurt:
        out["chroma_kurt"] = jnp.asarray((0.01 * rng.normal(size=shape)).astype(np.float32))
    return out


def test_chroma_affine_zero_leaves_are_bit_identical_square():
    """The whole point of C1's additive design: with chroma_aniso/chroma_shear
    present but exactly zero, the render must reproduce the pre-C1
    shift+dilation-only model to float32 machine precision.

    Not literally ``assert_array_equal``: adding the two extra all-zero blocks
    changes the einsum's contraction WIDTH (4 -> 8 (row,col) pairs here), and each
    product term involving a zero coefficient is exactly 0.0 (0.0 * finite is exact
    in IEEE754) but XLA's reduction can still group the *other*, non-zero partial
    sums differently at a different contraction width, at the ~1 ULP level (measured
    max relative diff ~1.7e-7, i.e. within float32 eps ~1.19e-7). That is real
    floating-point hardware behavior, not a physical or numerical bug in the model --
    ``test_chroma_zero_leaves_are_identical_to_no_chroma`` above happens to land on an
    exactly-equal case at its (different) contraction width, which is not something
    to rely on in general.
    """
    colours = np.array([0.3, -0.4, 1.1], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.2)
    t_ref, *_ = L.forward_model(params, ctx)
    shape = tuple(np.asarray(params["chroma_dilation"]).shape)
    p_zero_affine = dict(params)
    p_zero_affine["chroma_aniso"] = jnp.zeros(shape, dtype=jnp.float32)
    p_zero_affine["chroma_shear"] = jnp.zeros(shape, dtype=jnp.float32)
    t_zero, *_ = L.forward_model(p_zero_affine, ctx)
    np.testing.assert_allclose(np.asarray(t_zero), np.asarray(t_ref), rtol=2e-6, atol=1e-9)


def test_chroma_affine_zero_leaves_are_bit_identical_packed():
    """Packed twin of the above; see its docstring for the float32-eps tolerance."""
    ctx, params = _packed_scaffold()
    t_ref, *_ = L.forward_model(params, ctx)
    shape = tuple(np.asarray(params["chroma_dilation"]).shape)
    p_zero_affine = dict(params)
    p_zero_affine["chroma_aniso"] = jnp.zeros(shape, dtype=jnp.float32)
    p_zero_affine["chroma_shear"] = jnp.zeros(shape, dtype=jnp.float32)
    t_zero, *_ = L.forward_model(p_zero_affine, ctx)
    np.testing.assert_allclose(np.asarray(t_zero), np.asarray(t_ref), rtol=2e-6, atol=1e-9)


def test_chroma_affine_actually_changes_the_render():
    colours = np.array([0.4, -0.6, 1.3], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.35)
    t0, *_ = L.forward_model(params, ctx)
    p = _add_affine_leaves(params)
    t1, *_ = L.forward_model(p, ctx)
    peak = float(np.abs(np.asarray(t0)).max())
    rel = float(np.abs(np.asarray(t1) - np.asarray(t0)).max() / peak)
    assert rel > 1e-3, f"colour-affine term had no effect on the render (rel {rel:.2e})"


def test_packed_chroma_affine_banded_matches_unbanded():
    """Packed A/B test for the generalized multi-term concat fold (production runs
    packed tiers, so this is the one that counts -- see the dilation-only twin)."""
    ctx, params = _packed_scaffold()
    params = _add_affine_leaves(params, kurt=True)
    saved = L.USE_BANDED_RENDER_PACKED
    try:
        L.USE_BANDED_RENDER_PACKED = False
        t_old, *_ = L.forward_model(params, ctx)
        L.USE_BANDED_RENDER_PACKED = True
        t_new, *_ = L.forward_model(params, ctx)
    finally:
        L.USE_BANDED_RENDER_PACKED = saved
    t_old = np.asarray(t_old)
    t_new = np.asarray(t_new)
    peak = np.abs(t_old).max()
    assert peak > 0
    rel = np.abs(t_old - t_new).max() / peak
    assert rel <= 1e-5, f"banded vs unbanded colour-affine mismatch, rel {rel:.3e}"


def test_chroma_affine_gradients_match_finite_differences():
    fx = _chroma_fixture(n_frames=6)
    params = _add_affine_leaves(fx["params"], kurt=True)
    fx = dict(fx)
    fx["params"] = params
    grads = jax.grad(lambda p: _loss_of(fx, p))(params)
    for key in L.CHROMA_AFFINE_LEAVES + L.CHROMA_KURT_LEAVES:
        g = np.asarray(grads[key])
        assert np.all(np.isfinite(g)), f"{key} gradient is not finite"
        assert np.abs(g).max() > 0, f"{key} received no gradient"
        idx = np.unravel_index(np.argmax(np.abs(g)), g.shape)
        h = 1e-3
        base = np.asarray(params[key], dtype=np.float64)
        up, dn = base.copy(), base.copy()
        up[idx] += h
        dn[idx] -= h
        p_up = dict(params); p_up[key] = jnp.asarray(up.astype(np.float32))
        p_dn = dict(params); p_dn[key] = jnp.asarray(dn.astype(np.float32))
        fd = (float(_loss_of(fx, p_up)) - float(_loss_of(fx, p_dn))) / (2 * h)
        rel = abs(fd - g[idx]) / max(abs(fd), 1e-12)
        # Looser than the shift/dilation FD test's 5e-2: this fixture stacks three
        # extra leaves at once and a central difference on the single largest entry
        # of a small (n_frames=6) fixture is noisier than that simpler case (measured
        # up to ~6.2e-2 for chroma_aniso); still tight enough to catch a sign flip,
        # wrong axis, or missing factor, which would be O(1), not ~6%.
        assert rel < 1e-1, f"{key}{idx}: AD {g[idx]:.6g} vs FD {fd:.6g}, rel {rel:.2e}"


def test_has_chroma_affine_and_has_chroma_kurt_gating():
    _, params = _packed_scaffold()
    assert L.has_chroma(params)
    assert not L.has_chroma_affine(params)
    assert not L.has_chroma_kurt(params)
    p_affine = _add_affine_leaves(params)
    assert L.has_chroma_affine(p_affine)
    assert not L.has_chroma_kurt(p_affine)
    p_kurt = _add_affine_leaves(params, kurt=True)
    assert L.has_chroma_kurt(p_kurt)
    # dropping the base chroma leaves must also drop the derived gates, even if the
    # affine/kurt leaves are still present.
    p_no_base = {k: v for k, v in p_kurt.items() if k not in L.CHROMA_LEAVES}
    assert not L.has_chroma(p_no_base)
    assert not L.has_chroma_affine(p_no_base)
    assert not L.has_chroma_kurt(p_no_base)


def test_chroma_affine_leaves_round_trip_through_all_checkpoint_writers(tmp_path):
    """C1 twin of ``test_every_params_writer_round_trips_the_optional_leaves``,
    covering the new leaves specifically (that test only covers ``L.CHROMA_LEAVES``,
    which was deliberately NOT extended -- see ``fit.OPTIONAL_LEAVES``'s docstring)."""
    from syndiff_pipeline.forward_model import checkpoint_history as CH
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model import training_state as TS

    _, params = _packed_scaffold()
    params = _add_affine_leaves({k: jnp.asarray(v) for k, v in params.items()}, kurt=True)
    new_leaves = L.CHROMA_AFFINE_LEAVES + L.CHROMA_KURT_LEAVES
    for k in new_leaves:
        assert float(np.abs(np.asarray(params[k])).max()) > 0, "fixture must be non-zero"

    p1 = tmp_path / "full.npz"
    FIT.save_params_npz(p1, {k: np.asarray(v) for k, v in params.items()})
    back = FIT.load_params_npz(p1)
    for k in new_leaves:
        np.testing.assert_allclose(np.asarray(back[k]), np.asarray(params[k]), atol=0)

    p2 = tmp_path / "rolling.npz"
    CH.save_params_leaves_npz(p2, params)
    got = np.load(p2)
    for k in new_leaves:
        assert k in got.files, f"{k} dropped by the rolling checkpoint writer"
        np.testing.assert_allclose(got[k], np.asarray(params[k]), atol=0)
    for k in new_leaves:
        assert k in FIT.load_params_npz(p2)

    tx = FIT.make_stage_optimizer(3, 1e-4, param_keys=tuple(params))
    p3 = tmp_path / "state.npz"
    TS.save(p3, params=params, opt_state=tx.init(params), metadata={"step": 1})
    back3, _opt, _meta = TS.load(p3)
    for k in new_leaves:
        assert k in back3, f"{k} dropped by training_state.save"
        np.testing.assert_allclose(back3[k], np.asarray(params[k]), atol=0)


def test_chroma_affine_leaves_load_as_zero_from_a_pre_c1_checkpoint(tmp_path):
    """A checkpoint saved before C1 (shift+dilation only) must still load: the new
    leaves are simply absent, exactly like a pre-chroma checkpoint lacks shift+dilation."""
    from syndiff_pipeline.forward_model import fit as FIT

    _, params = _packed_scaffold()
    p = tmp_path / "pre_c1.npz"
    FIT.save_params_npz(p, {k: np.asarray(v) for k, v in params.items()})
    loaded = FIT.load_params_npz(p)
    for k in L.CHROMA_AFFINE_LEAVES + L.CHROMA_KURT_LEAVES:
        assert k not in loaded


def test_optimizer_routes_chroma_affine_to_train_chroma_and_freezes_before_stage3():
    from syndiff_pipeline.forward_model import fit as FIT

    _, params = _packed_scaffold()
    params = _add_affine_leaves(params, kurt=True)
    keys = tuple(params)
    for stage in (1, 2):
        labels = FIT._leaf_labels(stage, freeze_wcs=False, param_keys=keys)
        for k in L.CHROMA_AFFINE_LEAVES + L.CHROMA_KURT_LEAVES:
            assert labels[k] == "frozen", f"stage {stage}: {k} must be frozen, got {labels[k]}"
    labels3 = FIT._leaf_labels(3, freeze_wcs=False, param_keys=keys)
    for k in L.CHROMA_AFFINE_LEAVES + L.CHROMA_KURT_LEAVES:
        assert labels3[k] == "train_chroma", f"stage 3: {k} should be train_chroma, got {labels3[k]}"


# ---------------------------------------------------------------------------
# Free dP/dcolour IMAGE leaf (single-FFI chromatic study)
# ---------------------------------------------------------------------------


def test_chroma_image_gauge_is_flux_neutral_and_idempotent():
    """The image carries the three MODE gauges, for the mode reasons.

    Flux-neutral (a colour-dependent flux scale is profiled out per stamp),
    base-orthogonal, and core-dipole-free. Idempotency is the check that the
    three hold *jointly* -- sequential projectors do not commute, which is why
    ``decode_epsf_modes`` runs two alternating passes.
    """
    rng = np.random.default_rng(11)
    base = np.abs(rng.normal(size=(2, 2, G_SMALL, G_SMALL))).astype(np.float32)
    base /= base.sum(axis=(2, 3), keepdims=True)
    raw = rng.normal(size=(G_SMALL, G_SMALL)).astype(np.float32)

    gauged = EM.decode_chroma_image(jnp.asarray(raw), jnp.asarray(base))
    again = EM.decode_chroma_image(gauged, jnp.asarray(base))

    rms = float(np.sqrt(np.mean(np.asarray(gauged) ** 2)))
    assert rms > 0.1
    assert abs(float(jnp.sum(gauged))) < 1e-3 * rms
    assert float(jnp.max(jnp.abs(again - gauged))) < 1e-4 * rms


def test_chroma_image_rejects_a_per_node_field():
    rng = np.random.default_rng(3)
    base = np.abs(rng.normal(size=(2, 2, G_SMALL, G_SMALL))).astype(np.float32)
    with pytest.raises(ValueError, match="single"):
        EM.decode_chroma_image(
            jnp.asarray(rng.normal(size=(2, 2, G_SMALL, G_SMALL)).astype(np.float32)),
            jnp.asarray(base),
        )


def test_chroma_image_zero_leaf_is_identical_to_no_image():
    colours = np.array([0.3, -0.4, 1.1], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.2)
    g = int(np.asarray(params["epsf_base_raw"]).shape[-1])
    p_img = dict(params)
    p_img["chroma_image"] = jnp.zeros((g, g), dtype=jnp.float32)
    t_img, *_ = L.forward_model(p_img, ctx)
    t_ref, *_ = L.forward_model(params, ctx)
    np.testing.assert_array_equal(np.asarray(t_img), np.asarray(t_ref))


def test_chroma_image_banded_matches_unbanded():
    """Square path: broadcast-then-contract (hot) vs add-the-image (fallback).

    The fallback skips the node blend entirely on the grounds that blending a
    node-constant field is the identity. If that reasoning were wrong -- or if the
    hot path folded the image in without letting ``node_moments`` see it -- the two
    disagree here and nowhere else.
    """
    colours = np.array([0.4, -0.6, 1.3], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.35)
    rng = np.random.default_rng(23)
    g = int(np.asarray(params["epsf_base_raw"]).shape[-1])
    params = dict(params)
    params["chroma_image"] = jnp.asarray(
        (0.02 * rng.normal(size=(g, g))).astype(np.float32)
    )
    saved = L.USE_BANDED_RENDER
    try:
        L.USE_BANDED_RENDER = False
        t_old, *_ = L.forward_model(params, ctx)
        L.USE_BANDED_RENDER = True
        t_new, *_ = L.forward_model(params, ctx)
    finally:
        L.USE_BANDED_RENDER = saved
    t_old, t_new = np.asarray(t_old), np.asarray(t_new)
    peak = np.abs(t_old).max()
    assert peak > 0
    rel = np.abs(t_old - t_new).max() / peak
    assert rel <= 1e-5, f"banded vs unbanded chroma-image mismatch, rel {rel:.3e}"


def test_packed_chroma_image_banded_matches_unbanded():
    ctx, params = _packed_scaffold(seed=5)
    rng = np.random.default_rng(29)
    g = int(np.asarray(params["epsf_base_raw"]).shape[-1])
    params = dict(params)
    params["chroma_image"] = jnp.asarray(
        (0.02 * rng.normal(size=(g, g))).astype(np.float32)
    )
    saved = L.USE_BANDED_RENDER_PACKED
    try:
        L.USE_BANDED_RENDER_PACKED = False
        t_old, *_ = L.forward_model(params, ctx)
        L.USE_BANDED_RENDER_PACKED = True
        t_new, *_ = L.forward_model(params, ctx)
    finally:
        L.USE_BANDED_RENDER_PACKED = saved
    t_old, t_new = np.asarray(t_old), np.asarray(t_new)
    peak = np.abs(t_old).max()
    assert peak > 0
    rel = np.abs(t_old - t_new).max() / peak
    assert rel <= 1e-5, f"packed banded vs unbanded chroma-image mismatch, rel {rel:.3e}"


def test_chroma_image_scales_with_colour_offset():
    """The image enters linearly in delta, so a star at delta=0 must not move."""
    colours = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=True, bp_rp=colours, colour_ref=0.5)
    rng = np.random.default_rng(31)
    g = int(np.asarray(params["epsf_base_raw"]).shape[-1])
    p_img = dict(params)
    p_img["chroma_image"] = jnp.asarray((0.05 * rng.normal(size=(g, g))).astype(np.float32))
    t_img, *_ = L.forward_model(p_img, ctx)
    t_ref, *_ = L.forward_model(params, ctx)
    peak = float(np.abs(np.asarray(t_ref)).max())
    assert peak > 0
    rel = float(np.abs(np.asarray(t_img) - np.asarray(t_ref)).max() / peak)
    assert rel < 1e-6, f"delta == 0 but the image moved the model (rel {rel:.2e})"


def test_chroma_image_unfreezes_only_at_stage_4():
    from syndiff_pipeline.forward_model import fit as FIT

    for stage in (1, 2, 3):
        labels = FIT._leaf_labels(stage, freeze_wcs=False, param_keys=FIT.ALL_LEAVES)
        assert labels["chroma_image"] == "frozen", stage
    labels = FIT._leaf_labels(4, freeze_wcs=False, param_keys=FIT.ALL_LEAVES)
    # Its OWN optax bucket, not the parametric leaves' one: sharing "train_chroma" put a
    # (58,58) raw image and per-node coefficients under one clip_by_global_norm and the
    # stage-4 loss rose 68.03 -> 104.37 in a single step. See
    # fit.CHROMA_IMAGE_LR_SCALE_DEFAULT.
    assert labels["chroma_image"] == "train_chroma_image"
    # The parametric colour leaves are still live in stage 4 unless asked otherwise.
    assert labels["chroma_shift"] == "train_chroma"
    frozen = FIT._leaf_labels(
        4, freeze_wcs=False, freeze_others_stage4=True, param_keys=FIT.ALL_LEAVES,
    )
    assert frozen["chroma_image"] == "train_chroma_image"
    for k in ("wcs_coeff", "epsf_base_raw", "chroma_shift", "chroma_dilation"):
        assert frozen[k] == "frozen", k


def test_chroma_image_penalty_is_zero_without_the_leaf():
    colours = np.array([0.3, -0.4, 1.1], dtype=np.float32)
    _, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.2)
    assert float(L.chroma_image_penalty(params)) == 0.0
    g = int(np.asarray(params["epsf_base_raw"]).shape[-1])
    rng = np.random.default_rng(37)
    p_img = dict(params)
    p_img["chroma_image"] = jnp.asarray(rng.normal(size=(g, g)).astype(np.float32))
    assert float(L.chroma_image_penalty(p_img)) > 0.0


def test_chroma_image_has_a_live_gradient():
    colours = np.array([0.4, -0.6, 1.3], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.35)
    g = int(np.asarray(params["epsf_base_raw"]).shape[-1])
    params = dict(params)
    params["chroma_image"] = jnp.zeros((g, g), dtype=jnp.float32)
    # Gradient of a plain sum-of-squares against a fixed synthetic target: the leaf
    # is zero-initialised, so a dead fold (or a gauge that annihilates everything)
    # shows up as an exactly-zero gradient here.
    target = jnp.asarray(
        np.random.default_rng(41).normal(
            size=np.asarray(L.forward_model(params, ctx)[0]).shape
        ).astype(np.float32)
    )

    def loss_fn(p):
        templates, *_ = L.forward_model(p, ctx)
        return jnp.sum((templates - target) ** 2)

    grad = jax.grad(loss_fn)(params)["chroma_image"]
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


# --- chromatic HALO leaf -------------------------------------------------------
# Measured 2026-09-17: the excess the per-stamp pedestal absorbs is the star's own
# light on an r^-2.00 +- 0.14 profile. Every other chroma generator is a derivative
# of the ePSF and inherits its r^-4, so this leaf is the only one that is NOT a
# functional of the base. These tests pin the two properties that make it different:
# the profile really is a power law in PHYSICAL px, and it is base-orthogonal but
# deliberately NOT flux-neutral.


def test_chroma_halo_profile_is_a_power_law_in_physical_px():
    g = EM.NODE_GRID_SIZE
    prof = np.asarray(EM.chroma_halo_profile(g, index=2.0, core_px=1.0))
    assert prof.shape == (g, g)
    coord = np.asarray(EM.node_coord_1d(n_grid=g))
    r = np.hypot(*np.meshgrid(coord, coord, indexing="xy"))
    # Flat inside the core, so the profile is finite at the origin and peaks at 1.
    # NODE_GRID_SIZE is now the pixel-integrated grid (55, ODD): there IS an
    # exact centre sample at NODE_CENTER_INDEX=27.0, but argmin(r) still finds
    # it robustly without assuming that.
    assert prof.ravel()[int(np.argmin(r))] == pytest.approx(1.0)
    assert prof.max() == pytest.approx(1.0)
    assert np.all(np.isfinite(prof)) and np.all(prof > 0)
    # r^-2: doubling the radius quarters the value. Checked on actual node samples
    # rather than on the formula, so an index/physical-vs-index-coordinate mix-up
    # (the bug class dilation_generator's tests exist for) cannot pass.
    for r_in, r_out in ((1.5, 3.0), (2.0, 4.0), (2.5, 5.0)):
        i = np.argmin(np.abs(r - r_in)); j = np.argmin(np.abs(r - r_out))
        ratio = prof.ravel()[i] / prof.ravel()[j]
        assert ratio == pytest.approx(4.0, rel=0.12), (r_in, r_out, ratio)
    # A different index really changes the falloff.
    steep = np.asarray(EM.chroma_halo_profile(g, index=3.0, core_px=1.0))
    i = np.argmin(np.abs(r - 2.0)); j = np.argmin(np.abs(r - 4.0))
    assert steep.ravel()[i] / steep.ravel()[j] == pytest.approx(8.0, rel=0.15)


def test_chroma_halo_field_is_base_orthogonal_but_not_flux_neutral():
    g = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:g, 0:g]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    base = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    base /= base.sum()
    field = EM.chroma_halo_field(jnp.asarray(base[None]))
    arr = np.asarray(field)[0]
    assert arr.shape == (g, g)
    assert np.all(np.isfinite(arr))

    w = np.asarray(EM.canonical_mode_weight_grid(g))

    def _overlap(ref):
        return float(np.sum(w * arr * ref) / np.sqrt(
            np.sum(w * arr * arr) * np.sum(w * ref * ref)))

    # Gauge that IS imposed: no component along the RAW base inside the gauge radius.
    # Raw, not mean-removed, and that distinction is the point -- the free per-stamp
    # flux multiplies P itself, so <halo, P> = 0 is exactly the direction flux can
    # absorb. The other generators orthogonalise against the mean-removed base
    # because they additionally impose zero-sum; this one must not.
    assert abs(_overlap(base)) < 1e-5, _overlap(base)
    # And so it is deliberately NOT orthogonal to the mean-removed base.
    assert abs(_overlap(base - base.mean())) > 1e-3
    # Gauge that is NOT imposed: the halo is net extra light, so a zero-sum version
    # could not represent it. This is the only chroma generator with that property.
    assert abs(float(arr.sum())) > 1e-3 * float(np.abs(arr).sum())
    for name, gen in (("dilation", EM.chroma_dilation_field),
                      ("kurt", EM.chroma_kurt_field)):
        other = np.asarray(gen(jnp.asarray(base[None])))[0]
        assert abs(float(other.sum())) < 1e-5 * float(np.abs(other).sum()), name


def test_chroma_halo_init_params_and_prerequisites():
    from syndiff_pipeline.forward_model import loss as LL
    ctx, params, _, _ = _scaffold(uniform_nodes=False)
    n_rows, n_cols = 2, 2
    assert not LL.has_chroma_halo(params)
    p = dict(params)
    p["chroma_halo"] = jnp.zeros((n_rows, n_cols), dtype=jnp.float32)
    assert LL.has_chroma_halo(p)
    # Unlike chroma_kurt (which requires chroma_affine) the halo is additive, so it
    # layers directly on the shift/dilation pair with no prerequisite leaf.
    assert "halo" in LL.CHROMA_FIELD_GENERATORS
    assert LL.CHROMA_HALO_LEAVES == ("chroma_halo",)
    from syndiff_pipeline.forward_model import fit as FIT
    assert "chroma_halo" in FIT.ALL_OPTIONAL_LEAVES
    labels = FIT._leaf_labels(3, freeze_wcs=False, param_keys=FIT.ALL_LEAVES)
    assert labels["chroma_halo"] == "train_chroma"
    for stage in (1, 2):
        assert FIT._leaf_labels(
            stage, freeze_wcs=False, param_keys=FIT.ALL_LEAVES
        )["chroma_halo"] == "frozen", stage


def test_chroma_halo_changes_the_render_and_has_a_live_gradient():
    colours = np.array([0.5, -0.7, 1.2], dtype=np.float32)
    ctx, params, _, _ = _scaffold(uniform_nodes=False, bp_rp=colours, colour_ref=0.3)
    base_templates = np.asarray(L.forward_model(params, ctx)[0])

    p = dict(params)
    p["chroma_halo"] = jnp.full((2, 2), 0.05, dtype=jnp.float32)
    halo_templates = np.asarray(L.forward_model(p, ctx)[0])
    assert np.all(np.isfinite(halo_templates))
    # A non-zero amplitude must actually reach the render -- a dead fold would leave
    # the templates untouched.
    assert np.max(np.abs(halo_templates - base_templates)) > 0.0

    p0 = dict(params)
    p0["chroma_halo"] = jnp.zeros((2, 2), dtype=jnp.float32)
    target = jnp.asarray(np.random.default_rng(53).normal(
        size=base_templates.shape).astype(np.float32))

    def loss_fn(q):
        templates, *_ = L.forward_model(q, ctx)
        return jnp.sum((templates - target) ** 2)

    grad = np.asarray(jax.grad(loss_fn)(p0)["chroma_halo"])
    assert np.all(np.isfinite(grad))
    assert float(np.max(np.abs(grad))) > 0.0
