"""Radial colour family (branch colour-radial-20261008): chroma_g8 extras rb{j}_0/_r, rq{j}, rc{j} and --chroma-g8-drop."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import scene_fit as SF

jax.config.update("jax_platform_name", "cpu")

@pytest.fixture(autouse=True, params=["add", "mult"])
def radial_mode(request):
    """Run every test in both radial modes (the mode is a trace-time module constant)."""
    old = EM.get_radial_mode()
    EM.set_radial_mode(request.param)
    EM._RADIAL_GRID_CACHE.clear()
    yield request.param
    EM.set_radial_mode(old)


AXIS = (-55.66, 2098.93)
A3 = ("dil_r", "sq0", "sq1", "q1_0", "q1_x", "q1_y", "q2_0", "q2_x", "q2_y")
A3_VALS = [0.012, -0.03, 0.004, -0.02, 0.006, -0.015, 0.02, -0.004,      # 8 base
           0.01, 0.003, -0.002, 0.004, 0.001, -0.002, 0.003, 0.002, -0.001]   # 9 A3 extras
NEW = ("rb1_0", "rb1_r", "rb2_0", "rb2_r", "rb3_0", "rb3_r", "rc1", "rc2", "rq1")


def _fixture(n_frames=4):
    from test_fm_chroma import _chroma_fixture
    return _chroma_fixture(n_frames=n_frames)


def _g8_ctx(ctx, extras, drop=()):
    return L.replace(ctx, chroma_axis=AXIS, chroma_g8_gauge="raw", chroma_g8_extras=tuple(extras),
                     chroma_g8_drop=L.g8_drop_tuple(drop), chroma_delta2_mean=0.2)


def _g8_params(params, values):
    p = {k: v for k, v in params.items() if k not in L.CHROMA_LEAVES}
    p["chroma_g8"] = jnp.asarray(np.asarray(values, np.float32))
    return p


def _loss(fx, params, ctx):
    loss, _ = L.total_loss(params, ctx, fx["data"], fx["noise"], fx["weight"], fx["wcs_second_diff"],
                           fx["w_second_diff"], epsf_modes_init=fx["epsf_modes_init"], stamp_active=fx["stamp_active"])
    return loss


def _fake_ctx(n=60, seed=0, extras=NEW):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        chroma_delta=jnp.asarray(rng.normal(0, 0.6, n), jnp.float32), chroma_axis=AXIS,
        x_lin=jnp.asarray(rng.uniform(0, 2048, n), jnp.float32), y_lin=jnp.asarray(rng.uniform(0, 2048, n), jnp.float32),
        chroma_g8_gauge="raw", chroma_g8_no_dil=False, chroma_g8_extras=tuple(extras), chroma_delta2_mean=0.2)


def _node_field():
    g = EM.node_geometry(5)[1]
    yy, xx = np.mgrid[0:g, 0:g]
    c = EM.node_center_for_grid(g)
    P = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / 18.0) + 0.3 * np.exp(-((xx - c - 2.5) ** 2 + (yy - c + 1.5) ** 2) / 30.0)
    return jnp.asarray((P / P.sum())[None, None].astype(np.float32))


# ------------------------------------------------------------------ basis

def test_bump_basis_values_partition_and_plateau():
    k = EM.RADIAL_KNOTS_DEFAULT
    assert k == (0.0, 0.7, 1.5, 3.0, 5.5) and EM.COMA_KNOTS_DEFAULT == (0.8, 2.2, 5.0)
    assert EM.n_radial_basis() == 5 and EM.n_coma_basis() == 3
    B = EM.radial_bspline_basis(np.array(k), k)                       # at knot j: bump j = 2/3, neighbours 1/6
    for j in range(5):
        assert B[j, j] == pytest.approx(2 / 3)
        for jj in (j - 1, j + 1):
            if 0 <= jj < 5:
                assert B[jj, j] == pytest.approx(1 / 6)
    rho = np.linspace(k[1], k[-2], 301)                               # interior: partition of unity
    np.testing.assert_allclose(EM.radial_bspline_basis(rho, k).sum(0), 1.0, atol=1e-12)
    far = EM.radial_bspline_basis(np.array([5.5, 9.0, 40.0]), k)       # plateau beyond the last knot
    np.testing.assert_allclose(far[:3], 0.0, atol=1e-14)
    np.testing.assert_allclose(far[3], 1 / 6); np.testing.assert_allclose(far[4], 2 / 3)
    assert (EM.radial_bspline_basis(np.linspace(0, 9, 500), k) >= 0).all()
    # matches the step-0 reference construction (rb.py): bspline3 of the warped coordinate
    rho = np.linspace(0, 8, 97)
    s_ = np.interp(rho, np.array(k), np.arange(5))
    ref = np.stack([np.where(abs(s_ - j) < 1, 2 / 3 - (s_ - j) ** 2 + abs(s_ - j) ** 3 / 2,
                             np.where(abs(s_ - j) < 2, (2 - abs(s_ - j)) ** 3 / 6, 0.0)) for j in range(5)])
    np.testing.assert_allclose(EM.radial_bspline_basis(rho, k), ref, atol=1e-14)
    with pytest.raises(ValueError):
        EM.parse_radial_knots("0,2,1")


# ------------------------------------------------------------------ generators

@pytest.mark.parametrize("j", [1, 2, 3, 4, 5])
def test_radial_generator_flux_neutral_and_orthogonal_to_p0(j):
    P = _node_field()
    g = EM.radial_generator(P, j)
    np.testing.assert_allclose(float(jnp.sum(g)), 0.0, atol=1e-5 * float(jnp.abs(g).sum()))     # ungauged: flux-neutral
    f = EM.chroma_radial_field_raw(P, j)
    wg = EM.canonical_mode_weight_grid(int(P.shape[-1]))
    dot = float(jnp.sum(f * wg * P))
    assert abs(dot) < 1e-5 * float(jnp.sqrt(jnp.sum(f ** 2 * wg) * jnp.sum(P ** 2 * wg)) + 1e-12)
    assert float(jnp.abs(f).max()) > 0


def test_coma_generators_orthogonal_to_p0_and_its_gradients():
    P = _node_field()
    wg = EM.canonical_mode_weight_grid(int(P.shape[-1]))
    px, py = EM._grad_phys(P)
    for j in (1, 2, 3):
        for w in "ab":
            f = EM.chroma_radial_coma_field_raw(P, j, w)
            scale = float(jnp.sqrt(jnp.sum(f ** 2 * wg)))
            assert scale > 0
            for ref in (P, px, py):
                dot = float(jnp.sum(f * wg * ref))
                assert abs(dot) < 1e-5 * scale * float(jnp.sqrt(jnp.sum(ref ** 2 * wg))), (j, w, dot)
    # sign flips under 180 deg rotation of the grid (X, Y odd; B(rho) and P rotated alike)
    c = EM.radial_coma_generator(P, 2, "a")
    cr = EM.radial_coma_generator(P[..., ::-1, ::-1], 2, "a")
    np.testing.assert_allclose(np.asarray(cr), -np.asarray(c)[..., ::-1, ::-1], atol=1e-7)
    # smooth at the origin: the generator vanishes at rho = 0 even for the innermost spline
    g = P.shape[-1]; c0 = (g - 1) // 2
    assert float(EM.radial_coma_generator(P, 1, "a")[0, 0, c0, c0]) == 0.0


def test_degeneracy_matrix_shape_and_round_terms_overlap_rb():
    P = _node_field()
    rows, cols, M = EM.colour_radial_degeneracy(P[0, 0])
    assert rows == ["blur", "dil", "kurt", "shift_x", "shift_y"] and len(cols) == 5 + 6
    assert M.shape == (5, 11) and np.all(np.abs(M) <= 1 + 1e-6)
    # shift is orthogonal to every coma generator by construction
    assert np.abs(M[3:, 5:]).max() < 1e-4


# ------------------------------------------------------------------ slot terms

def test_slot_term_weights_and_coma_sign_flip():
    ctx = _fake_ctx(extras=A3 + NEW); occ = jnp.arange(60)
    vals = dict(rb1_0=0.01, rb1_r=-0.02, rb2_0=0.03, rb2_r=0.04, rb3_0=0.0, rb3_r=0.0, rc1=0.05, rc2=-0.06, rq1=0.07)
    ex = A3 + NEW
    c = jnp.asarray(A3_VALS + [vals[n] for n in NEW], jnp.float32)
    f = L._chroma_g8_slot_terms({"chroma_g8": c}, ctx, occ)[3]
    d = np.asarray(ctx.chroma_delta)
    x, y = np.asarray(ctx.x_lin), np.asarray(ctx.y_lin)
    vx, vy = AXIS[0] - x, AXIS[1] - y
    rr = np.hypot(vx, vy); r = rr / 1000.0; nx, ny = vx / rr, vy / rr
    np.testing.assert_allclose(np.asarray(f["rad1_raw"]), d * (0.01 - 0.02 * r) + (d * d - 0.2) * 0.07, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(np.asarray(f["rad2_raw"]), d * (0.03 + 0.04 * r), rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(np.asarray(f["radc1_a_raw"]), d * 0.05 * nx, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(np.asarray(f["radc1_b_raw"]), d * 0.05 * ny, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(np.asarray(f["radc2_b_raw"]), d * -0.06 * ny, rtol=1e-4, atol=1e-7)
    assert "rad3_raw" in f and "radc3_a_raw" not in f
    # 180 degree rotation of the axis direction (axis mirrored through the star) flips every coma weight
    # mirror each star through the axis (x -> 2 axis - x): the direction to the axis flips by 180 degrees
    ctx3 = _fake_ctx(extras=A3 + NEW); ctx3.x_lin = jnp.asarray(2 * AXIS[0] - x, jnp.float32); ctx3.y_lin = jnp.asarray(2 * AXIS[1] - y, jnp.float32)
    f3 = L._chroma_g8_slot_terms({"chroma_g8": c}, ctx3, occ)[3]
    for k in ("radc1_a_raw", "radc1_b_raw", "radc2_a_raw", "radc2_b_raw"):
        np.testing.assert_allclose(np.asarray(f3[k]), -np.asarray(f[k]), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(np.asarray(f3["rad2_raw"]), np.asarray(f["rad2_raw"]), rtol=1e-4, atol=1e-7)


def test_absent_extras_create_no_fields_and_unknown_raise():
    ctx = _fake_ctx(extras=A3); occ = jnp.arange(60)
    f = L._chroma_g8_slot_terms({"chroma_g8": jnp.asarray(A3_VALS, jnp.float32)}, ctx, occ)[3]
    assert not any(k.startswith("rad") for k in f)
    ctx.chroma_g8_extras = A3 + ("rb2_0",)
    f = L._chroma_g8_slot_terms({"chroma_g8": jnp.asarray(A3_VALS + [0.1], jnp.float32)}, ctx, occ)[3]
    assert [k for k in f if k.startswith("rad")] == ["rad2_raw"]
    for bad in ("rb6_0", "rb0_0", "rbx_0", "rc4"):
        ctx.chroma_g8_extras = A3 + (bad,)
        if bad in ("rb6_0", "rc4"):          # syntactically fine, beyond the 5 radial / 3 coma bumps
            assert not L.is_valid_g8_extra(bad)
        with pytest.raises(ValueError):
            L._chroma_g8_slot_terms({"chroma_g8": jnp.asarray(A3_VALS + [0.1], jnp.float32)}, ctx, occ)
    ctx.chroma_g8_extras = A3 + ("rb1_0",); ctx.chroma_g8_gauge = "mean"
    with pytest.raises(ValueError, match="raw"):
        L._chroma_g8_slot_terms({"chroma_g8": jnp.asarray(A3_VALS + [0.1], jnp.float32)}, ctx, occ)


def test_drop_removes_exactly_the_named_fields():
    ctx = _fake_ctx(extras=A3); occ = jnp.arange(60)
    c = {"chroma_g8": jnp.asarray(A3_VALS, jnp.float32)}
    full = set(L._chroma_g8_slot_terms(c, ctx, occ)[3])
    assert {"blur_raw", "dilation_raw", "kurt_plain_raw"} <= full
    for drop, gone in (("blur", {"blur_raw"}), ("dil", {"dilation_raw"}), ("kurt", {"kurt_plain_raw"}),
                       ("blur,dil,kurt", {"blur_raw", "dilation_raw", "kurt_plain_raw"})):
        ctx.chroma_g8_drop = L.g8_drop_tuple(drop)
        assert set(L._chroma_g8_slot_terms(c, ctx, occ)[3]) == full - gone
    with pytest.raises(ValueError):
        L.g8_drop_tuple("blur,foo")
    # shift, trefoil, astig planes and the radial family are untouched
    ctx.chroma_g8_drop = L.g8_drop_tuple("blur,dil,kurt")
    out = L._chroma_g8_slot_terms(c, ctx, occ)
    assert {"tre_a_raw", "tre_b_raw", "aniso_raw", "shear_raw"} <= set(out[3])


# ------------------------------------------------------------------ render / loss

def _ctx_params(extras, values, drop=()):
    fx = _fixture()
    return fx, _g8_ctx(fx["ctx"], extras, drop), _g8_params(fx["params"], values)


def test_zero_new_coefficients_bit_identical_to_a3():
    fx = _fixture()
    base_ctx = _g8_ctx(fx["ctx"], A3)
    base_p = _g8_params(fx["params"], A3_VALS)
    ctx = _g8_ctx(fx["ctx"], A3 + NEW)
    p = _g8_params(fx["params"], A3_VALS + [0.0] * len(NEW))
    v0, g0 = jax.value_and_grad(lambda q: _loss(fx, q, base_ctx))(base_p)
    v1, g1 = jax.value_and_grad(lambda q: _loss(fx, q, ctx))(p)
    t0 = np.asarray(L.forward_model(base_p, base_ctx)[0]); t1 = np.asarray(L.forward_model(p, ctx)[0])
    # NOT bitwise: the extra (zero-weight) node rows change the GEMM reduction order, so float32 rounding differs
    # (~1e-7 relative); the program WITHOUT the extras is the A3 program exactly (test_absent_extras_add_no_ops).
    np.testing.assert_allclose(t1, t0, rtol=1e-6, atol=1e-7 * np.abs(t0).max())
    np.testing.assert_allclose(float(v1), float(v0), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(g1["chroma_g8"])[:17], np.asarray(g0["chroma_g8"]), rtol=1e-4,
                               atol=1e-6 * np.abs(np.asarray(g0["chroma_g8"])).max())


def test_absent_extras_add_no_ops():
    fx = _fixture()
    ctx = _g8_ctx(fx["ctx"], A3)
    p = _g8_params(fx["params"], A3_VALS)
    j_plain = str(jax.make_jaxpr(lambda q: _loss(fx, q, ctx))(p))
    ctx_r = _g8_ctx(fx["ctx"], A3 + ("rb1_0",))
    p_r = _g8_params(fx["params"], A3_VALS + [0.0])
    j_rad = str(jax.make_jaxpr(lambda q: _loss(fx, q, ctx_r))(p_r))
    assert len(j_rad) > len(j_plain)
    ctx_d = L.replace(ctx, chroma_g8_drop=())      # an explicit empty drop is the same program
    assert str(jax.make_jaxpr(lambda q: _loss(fx, q, ctx_d))(p)) == j_plain


def test_gradients_nonzero_and_match_finite_differences():
    ex = A3 + NEW
    vals = A3_VALS + [0.004, 0.003, -0.004, 0.002, 0.003, -0.002, 0.004, -0.003, 0.003]
    fx, ctx, p = _ctx_params(ex, vals)
    f = lambda q: _loss(fx, q, ctx)
    g = np.asarray(jax.grad(f)(p)["chroma_g8"])
    base = np.asarray(p["chroma_g8"], np.float64)
    for i, name in enumerate(NEW):
        gi = g[17 + i]
        assert np.isfinite(gi) and abs(gi) > 0, name
        h = 1e-4          # the additive terms are strongly curved: FD only converges at small h
        up, dn = base.copy(), base.copy(); up[17 + i] += h; dn[17 + i] -= h
        fd = (float(f({**p, "chroma_g8": jnp.asarray(up, jnp.float32)}))
              - float(f({**p, "chroma_g8": jnp.asarray(dn, jnp.float32)}))) / (2 * h)
        assert abs(fd - gi) <= 0.1 * abs(fd) + 2e-3 * abs(g).max(), (name, gi, fd)


def test_render_changes_with_new_coefficients_square_and_unbanded_agree():
    ex = A3 + ("rb2_0", "rb2_r", "rc1")
    fx, ctx, p = _ctx_params(ex, A3_VALS + [0.01, 0.01, 0.02])
    t_new = np.asarray(L.forward_model(p, ctx)[0])
    t_a3 = np.asarray(L.forward_model(_g8_params(fx["params"], A3_VALS), _g8_ctx(fx["ctx"], A3))[0])
    assert np.abs(t_new - t_a3).max() > 1e-6 * np.abs(t_a3).max()
    saved = L.USE_BANDED_RENDER
    try:
        L.USE_BANDED_RENDER = False
        t_old = np.asarray(L.forward_model(p, ctx)[0])
    finally:
        L.USE_BANDED_RENDER = saved
    assert np.abs(t_old - t_new).max() / np.abs(t_old).max() < 1e-5


def test_packed_path_handles_radial_fields():
    from test_fm_chroma import _packed_scaffold
    ctx, params = _packed_scaffold()
    ex = A3 + ("rb1_0", "rb3_r", "rc2")
    ctx = _g8_ctx(ctx, ex)
    p = _g8_params(params, A3_VALS + [0.02, 0.02, 0.03])
    ctx0 = _g8_ctx(ctx, A3); p0 = _g8_params(params, A3_VALS)
    t = np.asarray(L.forward_model(p, ctx)[0]); t0 = np.asarray(L.forward_model(p0, ctx0)[0])
    assert np.abs(t - t0).max() > 1e-7 * np.abs(t0).max()
    saved = L.USE_BANDED_RENDER_PACKED
    try:
        L.USE_BANDED_RENDER_PACKED = False
        t_old = np.asarray(L.forward_model(p, ctx)[0])
    finally:
        L.USE_BANDED_RENDER_PACKED = saved
    assert np.abs(t_old - t).max() / np.abs(t_old).max() < 1e-5
    pz = _g8_params(params, A3_VALS + [0.0] * 3)
    np.testing.assert_allclose(np.asarray(L.forward_model(pz, ctx)[0]), t0, rtol=1e-6, atol=1e-7 * np.abs(t0).max())


def test_drop_changes_render_and_gives_inert_coefficients():
    fx = _fixture()
    p = _g8_params(fx["params"], A3_VALS)
    ctx = _g8_ctx(fx["ctx"], A3, drop="blur,dil,kurt")
    g = np.asarray(jax.grad(lambda q: _loss(fx, q, ctx))(p)["chroma_g8"])
    for i in (3, 4, 5, 6, 8):           # k0, k1, blur, dil, dil_r
        assert g[i] == 0.0, i
    assert abs(g[0]) > 0 and abs(g[7]) > 0
    t_a3 = np.asarray(L.forward_model(p, _g8_ctx(fx["ctx"], A3))[0]); t_d = np.asarray(L.forward_model(p, ctx)[0])
    assert np.abs(t_a3 - t_d).max() > 0


def test_nonraw_gauge_with_drop_ok_and_context_defaults():
    fx = _fixture()
    assert fx["ctx"].chroma_g8_drop == () and fx["ctx"].chroma_radial_knots == EM.RADIAL_KNOTS_DEFAULT


# ------------------------------------------------------------------ scene_fit plumbing

def test_warm_start_from_a3_pads_zeros_and_reproduces_a3_loss():
    fx = _fixture()
    a3_p = _g8_params(fx["params"], A3_VALS)
    new = SF.carry_g8(a3_p["chroma_g8"], A3, A3 + NEW)
    assert new.shape == (17 + len(NEW),) and not new[17:].any()
    np.testing.assert_array_equal(new[:17], np.asarray(A3_VALS, np.float32))
    p = _g8_params(fx["params"], new)
    v_new = float(_loss(fx, p, _g8_ctx(fx["ctx"], A3 + NEW)))
    v_a3 = float(_loss(fx, a3_p, _g8_ctx(fx["ctx"], A3)))
    np.testing.assert_allclose(v_new, v_a3, rtol=1e-6)
    with pytest.raises(ValueError, match="must start with"):      # extras order: new ones must FOLLOW the warm start's
        SF.carry_g8(a3_p["chroma_g8"], A3, NEW + A3)


def test_cli_flags_and_recipes():
    from syndiff_pipeline.forward_model import recipe
    for name, drop in (("paper1_final_radial", None), ("paper1_final_radial_drop", "blur,dil,kurt"),
                       ("paper1_final_radial_frozenshift", "blur,dil,kurt")):
        a = SF.build_parser().parse_args(recipe.recipe_argv(name) + ["--scene-dir", "s", "--out-dir", "o"])
        ex = SF.g8_extras_tuple(a.chroma_g8_extras)
        if name != "x":
            assert ex[:9] == A3 and ex[9:] == tuple(f"rb{j}_{s}" for j in range(1, 6) for s in ("0", "r")) + \
                tuple(f"rq{j}" for j in range(1, 6)) + ("rc1", "rc2", "rc3")
            assert a.chroma_radial_mode == "add"
        assert all(L.is_valid_g8_extra(e) for e in ex)
        assert a.chroma_g8_drop == drop
    a = SF.build_parser().parse_args(["--scene-dir", "s", "--out-dir", "o"])
    assert a.chroma_g8_drop is None and a.chroma_radial_knots is None
    assert SF.meta_colour_kwargs({}) == dict(chroma_g8_drop="", chroma_radial_knots=None, chroma_radial_mode=None, chroma_coma_knots=None)


def test_resolve_defaults_records_knots_and_resume(tmp_path):
    a = SF.build_parser().parse_args(["--scene-dir", "s", "--out-dir", str(tmp_path)])
    SF.resolve_g8_defaults(a, tmp_path)
    assert a.chroma_g8_drop == "" and EM.parse_radial_knots(a.chroma_radial_knots) == EM.RADIAL_KNOTS_DEFAULT
    assert EM.parse_radial_knots(a.chroma_coma_knots) == EM.COMA_KNOTS_DEFAULT
    (tmp_path / "fit_meta.json").write_text(json.dumps({"chroma_g8_extras": "dil_r,rb1_0", "chroma_g8_gauge": "raw",
                                                       "chroma_g8_drop": "blur", "chroma_radial_knots": "0.0,1.0,5.0"}))
    a = SF.build_parser().parse_args(["--scene-dir", "s", "--out-dir", str(tmp_path), "--resume"])
    SF.resolve_g8_defaults(a, tmp_path)
    assert a.chroma_g8_drop == "blur" and a.chroma_radial_knots == "0.0,1.0,5.0"


def test_custom_knots_change_basis():
    old = EM.get_radial_knots()
    try:
        EM.set_radial_knots("0,1,3")
        assert EM.n_radial_basis() == 3
        P = _node_field()
        a = EM.radial_generator(P, 2)
        EM.set_radial_knots(EM.RADIAL_KNOTS_DEFAULT)
        b = EM.radial_generator(P, 2)
        assert float(jnp.abs(a - b).max()) > 0
        with pytest.raises(ValueError):
            EM.set_radial_knots("0,1,3"); EM.radial_generator(P, 4)
    finally:
        EM.set_radial_knots(old)


# ------------------------------------------------------------------ freeze

def test_freeze_mask_names():
    m = SF.g8_freeze_mask("s0,s1,s2,sq0,sq1", A3 + ("rb1_0",))
    assert m.shape == (18,) and np.flatnonzero(m).tolist() == [0, 1, 2, 9, 10]
    assert SF.g8_freeze_mask("", A3) is None
    with pytest.raises(ValueError):
        SF.g8_freeze_mask("s0,nope", A3)


def test_frozen_slots_unchanged_after_steps_and_others_move():
    import optax
    from syndiff_pipeline.forward_model import fit as FIT
    ex = A3 + ("rb1_0", "rc1")
    fx, ctx, p = _ctx_params(ex, A3_VALS + [0.01, 0.01])
    mask = SF.g8_freeze_mask("s0,s1,s2,sq0,sq1", ex)
    labels = FIT._leaf_labels(3, freeze_wcs=True, param_keys=tuple(p))
    assert labels["wcs_coeff"] == "frozen"
    opt = FIT.make_stage_optimizer(3, 1e-3, param_keys=tuple(p), chroma_lr_scale=10, freeze_wcs=True)
    state = opt.init(p)

    def f(q):
        return _loss(fx, SF.freeze_g8_slots(FIT.stop_grad_frozen_params(q, labels), mask), ctx)

    q = p
    for _ in range(4):
        g = jax.grad(f)(q)
        upd, state = opt.update(g, state, q)
        q = optax.apply_updates(q, upd)
    c0, c1 = np.asarray(p["chroma_g8"]), np.asarray(q["chroma_g8"])
    assert np.array_equal(c1[mask], c0[mask])                       # bit-identical
    assert np.array_equal(np.asarray(q["wcs_coeff"]), np.asarray(p["wcs_coeff"]))
    for i in (3, 4, 7, 17, 18):                                     # unfrozen slots move
        assert c1[i] != c0[i], i


def test_freeze_wcs_flag_parses():
    a = SF.build_parser().parse_args(["--scene-dir", "s", "--out-dir", "o", "--freeze-wcs", "--chroma-g8-freeze", "s0,sq0"])
    assert a.freeze_wcs and a.chroma_g8_freeze == "s0,sq0"


def test_generators_first_called_inside_jit_and_outside():
    """Regression: the grid constants were built from a jax array and failed when first filled during a trace."""
    EM._RADIAL_GRID_CACHE.clear()
    P = _node_field()
    inside = jax.jit(lambda q: (EM.chroma_radial_field_raw(q, 2), EM.chroma_radial_coma_field_raw(q, 2, "a")))(P)
    EM._RADIAL_GRID_CACHE.clear()
    outside = (EM.chroma_radial_field_raw(P, 2), EM.chroma_radial_coma_field_raw(P, 2, "a"))
    for a, b in zip(inside, outside):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-7)


def test_radial_mode_flag_default_meta_and_ctx(tmp_path, radial_mode):
    a = SF.build_parser().parse_args(["--scene-dir", "s", "--out-dir", str(tmp_path)])
    assert a.chroma_radial_mode is None
    SF.resolve_g8_defaults(a, tmp_path)
    assert a.chroma_radial_mode == "add"                                  # new runs
    (tmp_path / "fit_meta.json").write_text(json.dumps({"chroma_g8_extras": "dil_r", "chroma_g8_gauge": "raw"}))
    a = SF.build_parser().parse_args(["--scene-dir", "s", "--out-dir", str(tmp_path), "--resume"])
    SF.resolve_g8_defaults(a, tmp_path)
    assert a.chroma_radial_mode == "mult"                                 # a pre-mode resume keeps the old form
    assert SF.meta_colour_kwargs({"chroma_radial_mode": "add"})["chroma_radial_mode"] == "add"
    assert _fixture()["ctx"].chroma_radial_mode == radial_mode
    with pytest.raises(ValueError):
        EM.set_radial_mode("nope")


def test_add_and_mult_generators_differ_and_add_ignores_p0_scale():
    P = _node_field()
    EM.set_radial_mode("add"); a = EM.radial_generator(P, 3); a2 = EM.radial_generator(2 * P, 3)
    EM.set_radial_mode("mult"); m = EM.radial_generator(P, 3)
    assert float(jnp.abs(a - m).max()) > 0
    np.testing.assert_allclose(np.asarray(a), np.asarray(a2))
    # additive profile reaches the wings where P0 is tiny
    ring = np.asarray(EM._radial_grid_consts(int(P.shape[-1]), EM.get_radial_knots())[0][3]) > 0.05
    assert float(jnp.abs(a[0, 0])[ring].max()) > 1.5 * float(jnp.abs(m[0, 0])[ring].max())


def test_coma_knots_independent_of_radial_knots():
    P = _node_field()
    old_r, old_c = EM.get_radial_knots(), EM.get_coma_knots()
    try:
        EM.set_coma_knots("0.5,1.5"); EM.set_radial_knots("0,2,4")
        assert EM.n_coma_basis() == 2 and EM.n_radial_basis() == 3
        a = EM.radial_coma_generator(P, 1, "a")
        EM.set_radial_knots("0,1,2,3,4,5")
        np.testing.assert_allclose(np.asarray(EM.radial_coma_generator(P, 1, "a")), np.asarray(a))
        with pytest.raises(ValueError):
            EM.radial_coma_generator(P, 3, "a")
    finally:
        EM.set_radial_knots(old_r); EM.set_coma_knots(old_c)


def _ring_minus_psf_projection(P, j):
    """Independent float64 numpy: gauge of rb.py's  ring_j / sum(ring_j) - P0 / sum(P0)  (P0-orthogonal complement)."""
    P = np.asarray(P, np.float64)[0, 0]
    B = EM._radial_grid_consts(P.shape[-1], EM.get_radial_knots())[0][j - 1].astype(np.float64)
    x = B / B.sum() - P / P.sum()
    wg = np.asarray(EM.canonical_mode_weight_grid(P.shape[-1]), np.float64)
    return x - (np.sum(x * wg * P) / np.sum(wg * P * P)) * P


def test_add_mode_equals_ring_minus_psf_projection_and_has_no_pedestal():
    EM.set_radial_mode("add")
    nodes = [_node_field()]
    real = Path("/astro/armin/koji/syndiff/dev_runs/paper1_final_fits_20261007/F1/fits/fold0/params.npz")
    if real.exists():
        from syndiff_pipeline.forward_model import fit as FIT
        raw = FIT.load_params_npz(real)["epsf_base_raw"]
        nodes.append(EM.decode_epsf_base(jnp.asarray(raw))[3:4, 3:4])
    for P in nodes:
        wg = np.asarray(EM.canonical_mode_weight_grid(P.shape[-1]), np.float64)
        for j in range(1, 6):
            g = np.asarray(EM.chroma_radial_field_raw(P, j), np.float64)[0, 0]
            ref = _ring_minus_psf_projection(P, j)
            assert np.abs(g - ref).max() <= 1e-5 * np.abs(ref).max(), j
            # no uniform pedestal beyond the ring's own mean: LS fit of the generator by {P0, const} under the weight grid
            A = np.stack([np.asarray(P, np.float64)[0, 0].ravel(), np.ones(g.size)], 1) * np.sqrt(wg.ravel())[:, None]
            coef = np.linalg.lstsq(A, g.ravel() * np.sqrt(wg.ravel()), rcond=None)[0]
            ring = EM._radial_grid_consts(P.shape[-1], EM.get_radial_knots())[0][j - 1]
            ring = ring / ring.sum()
            c_ring = np.linalg.lstsq(A, ring.ravel() * np.sqrt(wg.ravel()), rcond=None)[0]
            assert abs(coef[1] - c_ring[1]) <= 1e-4 * abs(c_ring[1]) + 1e-9, (j, coef[1], c_ring[1])   # = the ring's own
            if j <= 2:
                assert abs(coef[1]) < 0.05 * np.abs(g).max()
