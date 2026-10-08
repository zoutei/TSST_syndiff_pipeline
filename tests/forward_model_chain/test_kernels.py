"""Kernels stage: linearity K_b = K0 + delta_b K1, option-(a) linearisation, render glue, brightness-width no-op/formula;
slow parity tests against the e2e F1 kernel products."""
import json
from pathlib import Path

import numpy as np
import pytest

import jax  # noqa: F401  (before pandas/pyarrow)
import jax.numpy as jnp

from syndiff_pipeline.forward_model import bright_width as BW
from syndiff_pipeline.template_creation.processing import chromatic_kernels as CK

import b_fixtures as BF
from syndiff_pipeline.forward_model.chain import kernels as K

NR, NC, G = 2, 2, 31


def _gauss(g, sig, cx=0.0, cy=0.0, os=4):
    c = (g - 1) / 2
    t = (np.arange(g) - c) / os
    X, Y = np.meshgrid(t, t)
    a = np.exp(-0.5 * (((X - cx) ** 2 + (Y - cy) ** 2) / sig ** 2))
    return a / a.sum()


def _nodes(seed=0):
    rng = np.random.default_rng(seed)
    P0 = np.stack([[_gauss(G, 0.9 + 0.1 * (i + j), 0.02 * i, -0.02 * j) for j in range(NC)] for i in range(NR)])
    P1 = rng.normal(size=P0.shape) * 1e-3
    P1 = P1 - P1.sum(axis=(-2, -1), keepdims=True) * P0            # zero-sum, as the chain's P1
    sig = np.broadcast_to(np.array([[0.6, 0.05], [0.05, 0.5]]), (NR, NC, 2, 2)).copy()
    eps = np.full((NR, NC), 1e-3)
    tri = np.array([[True, False], [False, True]])
    return P0, P1, sig, eps, tri


def test_band_kernels_are_linear_in_delta():
    P0, P1, sig, eps, tri = _nodes()
    delta = np.array([-0.8, -0.2, 0.3, 0.9])
    E_bands = P0[None] + delta[:, None, None, None, None] * P1[None]
    Kb = CK.band_kernels(E_bands, sig, eps, tri, n_kernel=63)
    K0 = CK.band_kernels(P0[None], sig, eps, tri, n_kernel=63)[0]
    K1 = CK.band_kernels(P1[None], sig, eps, tri, n_kernel=63)[0]
    for i in range(4):
        assert np.abs(Kb[i] - (K0 + delta[i] * K1)).max() < 1e-12 * np.abs(K0).max()
    assert np.allclose(Kb.sum(axis=(-2, -1)), 1.0, atol=5e-3)      # dc normalisation: sums ~1


def test_band_linearisation_exact_for_linear_family():
    P0, P1, *_ = _nodes()
    xq = np.linspace(-0.5, 0.7, 25)
    dq = xq.copy()
    Eq = P0[:, :, None] + xq[None, None, :, None, None] * P1[:, :, None]
    E0 = P0
    T1 = P1                                               # the tangent of a linear family
    le = K.band_linearisation(Eq, E0, P0, P1, T1, xq, dq)
    assert le["err_ls"].max() < 1e-14 and le["err_tan"].max() < 1e-14
    assert le["dcen"].max() < 1e-9 and np.abs(le["dtr"]).max() < 1e-9
    assert le["err_ach"].max() > 1e-6                        # the achromatic model is NOT exact


def test_fourier_kernel_unit_sum_and_node_independent_eps_rule_shapes():
    P0, _, sig, *_ = _nodes()
    k = K.fourier_kernel(P0[0, 0], sig[0, 0], 1e-3, N=63, with_tri=True)
    assert k.shape == (63, 63) and abs(k.sum() - 1) < 1e-12


def test_moments_of_gaussian():
    E = _gauss(G, 1.0)
    mx, my, Ixx, Iyy, Ixy = K.moments(E, sigma_w_px=2.0)
    assert abs(mx) < 1e-12 and abs(my) < 1e-12 and abs(Ixy) < 1e-12
    assert abs(Ixx - 0.8) < 2e-3 and abs(Iyy - 0.8) < 2e-3       # sigma^2 sigma_w^2/(sigma^2+sigma_w^2) = 0.8


def test_fourier_shift_moves_peak_and_conserves_flux():
    E = _gauss(G, 0.8)
    out = K.fourier_shift(E, 1.0, -0.5)                   # +1 px = +4 subcells in x, -0.5 px = -2 in y
    py, px = np.unravel_index(out.argmax(), out.shape)
    assert (px, py) == (G // 2 + 4, G // 2 - 2)
    assert abs(out.sum() - 1) < 5e-4                       # only the tail shifted off the zero-padded grid is lost
    assert np.abs(K.fourier_shift(E, 0.0, 0.0) - E).max() < 1e-14


def test_shares_from_mags():
    m = np.array([[15.0, 14.5, 14.2, 14.1], [16.0, 15.0, 14.0, 13.5]])
    s = K.shares_from_mags(m)
    assert np.allclose(s.sum(-1), 1.0) and (s > 0).all()
    assert s[1, 3] > s[0, 3]                              # redder star: more y share


def test_bright_width_absent_is_noop_and_present_has_formula():
    params = {"chroma_g8": jnp.zeros(8)}
    assert K.bright_width_info(params, {"chroma_g8_gauge": "raw"}, 0.0) is None
    q_ref = 17453.0
    params = {"bright_width": jnp.asarray([0.158])}
    meta = {"bright_width": "lin", "bright_width_model": {"q_ref": q_ref}, "chroma_g8_gauge": "raw"}
    info = K.bright_width_info(params, meta, 0.0)
    b = 0.158 * BW.LEAF_UNIT
    assert info["blur_weight"] == pytest.approx(BW.GEN_PER_DSIGMA * b / BW.Q_UNIT * (0.0 - q_ref))
    assert info["delta_sigma_per_axis_px2"] == pytest.approx(b / BW.Q_UNIT * (0.0 - q_ref))
    assert K.bright_width_info(params, meta, q_ref)["blur_weight"] == pytest.approx(0.0)    # q = q_ref: no change
    with pytest.raises(ValueError):
        K.bright_width_info(params, {"bright_width": "lin", "chroma_g8_gauge": "raw"}, 0.0)   # no q_ref anywhere


def test_slot_ctx_carries_kernel_q():
    A = dict(kernel_q=3.5, bw=None, axis=(1.0, 2.0), extras=(), gauge="raw", d2mean=0.0)
    ctx = K.slot_ctx(A, [1.0, 2.0], [3.0, 4.0], [0.1, 0.2], np.arange(3.0), np.arange(3.0))
    assert np.all(np.asarray(ctx.bright_q) == 3.5) and ctx.bright_q_ref == 0.0
    A["bw"] = {"q_ref": 99.0}
    assert K.slot_ctx(A, [1.0], [3.0], [0.1], np.arange(3.0), np.arange(3.0)).bright_q_ref == 99.0


def test_unsupported_kernel_source_raises(tmp_path):
    cfg = BF.make_cfg(tmp_path, kernels={"source": "phasea"}, extra_inputs={"xp_synth": str(tmp_path / "x.csv")})
    with pytest.raises(NotImplementedError):
        K.run_k02(cfg, A={})


def test_colour_map_from_config(tmp_path):
    cfg = BF.make_cfg(tmp_path, extra_inputs={"colour_map": {"a": -26.0, "b": 0.034}})
    assert K._colour_map(cfg) == (-26.0, 0.034, "config")
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps({"X": {"linear_map": {"a": 1.5, "b": 2.5}}}))
    cfg = BF.make_cfg(tmp_path, extra_inputs={"colour_map": {"summary_json": str(sj), "key": "X"}})
    assert K._colour_map(cfg) == (1.5, 2.5, str(sj))
    with pytest.raises(KeyError, match="colour_map"):
        K._colour_map(BF.make_cfg(tmp_path))


# ------------------------------------------------------------------ slow parity (e2e F1)
slow = pytest.mark.slow
KE = BF.E2E_RUN / "kernels/F1"


def _f1_kernels_cfg(tmp_path):
    BF.need(KE / "band_epsf_a3.npz", BF.E2E / "s3_fit/fit/params.npz", BF.E2E / "s3_fit/scene", BF.E2E_RUN / "mapping/oversampling_4")
    cfg = BF.f1_cfg(tmp_path)
    BF.link(tmp_path / "F1/fit", BF.E2E / "s3_fit/fit")
    BF.link(tmp_path / "F1/scene_boot", BF.E2E / "s3_fit/scene")
    BF.link(tmp_path / "F1/mapping", BF.E2E_RUN / "mapping")
    return cfg


@slow
def test_k01_validation_matches_e2e(tmp_path):
    cfg = _f1_kernels_cfg(tmp_path)
    out = K.run_k01(cfg)
    ref = json.loads((KE / "validate_a3.json").read_text())
    for k in ("n_cases", "mine_vs_banded_max", "mine_vs_unbanded_max", "banded_vs_unbanded_max", "fourier_vs_render_max",
              "mine_vs_banded_median", "fourier_vs_render_median", "base_decode_vs_stored_max", "render_pass_1e-6",
              "colour_ref_recomputed", "delta2_recomputed", "kernel_bright_q", "local_poly_hard"):
        assert out[k] == ref[k], (k, out[k], ref[k])
    assert out["bright_width"] == ref["bright_width"]
    assert out["render_pass_1e-6"] and out["mine_vs_banded_max"] < 1e-6
    for a, b in zip(out["cases"], ref["cases"]):
        assert a == b
    BF.arrays_equal(tmp_path / "F1/kernels/scene_colours.npz", KE / "scene_colours_a3.npz")


@slow
def test_k_sigma_sigma_matches_e2e_bitwise(tmp_path):
    cfg = _f1_kernels_cfg(tmp_path)
    K.run_k_sigma(cfg, do_eps=False)                      # Sigma_G only (the eps/tri rule is ~25 min of CPU)
    a = np.load(tmp_path / "F1/kernels/kin_sigonly.npz")
    b = np.load(KE / "kin_a3.npz")
    for k in ("sigma_G_tess", "ra", "dec", "J_tess", "node_x", "node_y"):
        assert np.array_equal(a[k], b[k]), k


@slow
def test_k02_band_epsf_matches_e2e_bitwise(tmp_path):
    """k02 from the e2e kin (eps/tri included) and scene colours: every array of band_epsf / aux / K_achrom bitwise."""
    cfg = _f1_kernels_cfg(tmp_path)
    kd = tmp_path / "F1/kernels"
    kd.mkdir(parents=True)
    BF.link(kd / "kin.npz", KE / "kin_a3.npz")
    BF.link(kd / "scene_colours.npz", KE / "scene_colours_a3.npz")
    K.run_k02(cfg)
    n1 = BF.arrays_equal(kd / "band_epsf.npz", KE / "band_epsf_a3.npz",
                         keys=["K_bands", "K0", "K1", "P0", "P1", "E_bands", "delta_b", "node_x", "node_y", "sigma_G_tess", "eps", "tri"])
    n2 = BF.arrays_equal(kd / "band_epsf_aux.npz", KE / "band_epsf_a3_aux.npz")
    n3 = BF.arrays_equal(kd / "K_achrom.npz", KE / "K_achrom_a3.npz", keys=["K", "E", "node_x", "node_y", "sigma_G_tess", "eps", "tri"])
    assert len(n1) + len(n2) + len(n3) >= 30
    old = json.loads(str(np.load(KE / "band_epsf_a3.npz")["meta"]))
    new = json.loads(str(np.load(kd / "band_epsf.npz")["meta"]))
    assert new["linearisation_errors"] == old["linearisation_errors"] and new["kernel_checks"] == old["kernel_checks"]
    assert new["delta_b"] == old["delta_b"] and new["kernel_bright_q"] == old["kernel_bright_q"]


@slow
def test_k03_mixture_matches_e2e(tmp_path):
    cfg = _f1_kernels_cfg(tmp_path)
    kd = tmp_path / "F1/kernels"
    kd.mkdir(parents=True)
    for new, old in (("band_epsf.npz", "band_epsf_a3.npz"), ("band_epsf_aux.npz", "band_epsf_a3_aux.npz"),
                     ("scene_colours.npz", "scene_colours_a3.npz")):
        BF.link(kd / new, KE / old)
    out = K.run_k03(cfg)
    ref = json.loads((KE / "mixture_error_a3.json").read_text())
    assert out["summary"] == ref["summary"] and out["colour_check"] == ref["colour_check"]
    BF.arrays_equal(kd / "mixture_error_stars.npz", KE / "mixture_error_stars_a3.npz")


def test_truncation_aware_eps_keeps_passing_nodes_and_falls_back_on_overflow():
    """Nodes within tolerance keep the rule's eps/tri; a node over it takes the first rms-ordered table entry within it."""
    P0, P1, sig, eps, tri = _nodes()
    table = {f"{i},{j}": [[1e-8, True, 1.0], [1e-2, True, 2.0], [1e-3, bool(tri[i, j]), 3.0]]
             for i in range(NR) for j in range(NC)}
    l_rule = max(abs(K.truncation_loss(E[i, j], sig[i, j], eps[i, j], tri[i, j])) for E in (P0, P1)
                 for i in range(NR) for j in range(NC))
    e2, t2, ch, ml = K.truncation_aware_eps(eps, tri, table, sig, (P0, P1), tol=max(l_rule, 1e-12) * 10)
    assert ch == [] and np.array_equal(e2, eps) and np.array_equal(t2, tri) and ml.shape == (NR, NC)
    eps_bad = eps.copy()
    eps_bad[1, 0] = 1e-8
    lb = max(abs(K.truncation_loss(E[1, 0], sig[1, 0], 1e-8, tri[1, 0])) for E in (P0, P1))
    l2 = max(abs(K.truncation_loss(E[1, 0], sig[1, 0], 1e-2, True)) for E in (P0, P1))
    if not lb > l2:
        pytest.skip("synthetic node does not overflow at eps=1e-8")
    tol = 0.5 * (lb + l2)
    e3, t3, ch3, _ = K.truncation_aware_eps(eps_bad, tri, table, sig, (P0, P1), tol=tol)
    assert [c["node"] for c in ch3] == [[1, 0]] and e3[1, 0] == 1e-2 and t3[1, 0]
    m = np.ones_like(eps, bool)
    m[1, 0] = False
    assert np.array_equal(e3[m], eps_bad[m]) and np.array_equal(t3[m], tri[m])
    with pytest.raises(RuntimeError, match="no eps/tri"):
        K.truncation_aware_eps(eps_bad, tri, table, sig, (P0, P1), tol=0.0)


def test_kernel_sum_gate_passes_unit_kernels_and_flags_bad_node():
    k = np.zeros((NR, NC, 5, 5))
    k[..., 2, 2] = 1.0
    z = np.zeros_like(k)
    out = K.kernel_sum_gate(k, z, k, np.zeros((NR, NC)))
    assert out["max"]["K0"] == 0.0 and out["max"]["trunc"] == 0.0
    z2 = z.copy()
    z2[1, 1, 0, 0] = 0.0106                                       # the perband_v3 F2 (3,2) sum K1
    with pytest.raises(RuntimeError, match=r"'K1': \[\[1, 1\]\]"):
        K.kernel_sum_gate(k, z2, k, np.zeros((NR, NC)))
