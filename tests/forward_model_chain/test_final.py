"""Final-image stage: blended convolution, moment form, match/difference assembly, adopted-weight rescale (both store
cases), delta' recalibration; slow bitwise parity against the e2e F1 band_a3w products."""
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import b_fixtures as BF
from syndiff_pipeline.forward_model.chain import _tk as TK
from syndiff_pipeline.forward_model.chain import final as FN
from syndiff_pipeline.forward_model.chain.perband import paths as PP

BANDS = ("r", "i", "z", "y")
PROD = {"r": 0.238, "i": 0.344, "z": 0.283, "y": 0.135}
D13 = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}


def fake_grid(n=32, os=4):
    return types.SimpleNamespace(oversampling=os, width_os=n, height_os=n, ffi_xmin=0, ffi_ymin=0,
                                 science_xmin_ffi=0, science_ymin_ffi=0)


def _kernel_set(rng, nr=2, nc=2, N=5):
    K0 = np.abs(rng.normal(size=(nr, nc, N, N))) + 0.1
    K0 /= K0.sum(axis=(-2, -1), keepdims=True)
    K1 = rng.normal(size=(nr, nc, N, N)) * 1e-2
    K1 -= K1.mean(axis=(-2, -1), keepdims=True)
    return K0, K1


# ------------------------------------------------------------------ convolution
def test_hat_weights_partition_of_unity():
    nodes = np.array([1.0, 6.0, 11.0])
    c = np.linspace(-3, 15, 101)
    H = TK.hat_weights(c, nodes)
    assert np.allclose(H.sum(axis=0), 1.0) and (H >= 0).all()


def test_convolve_blended_identity_and_flux():
    rng = np.random.default_rng(0)
    g = fake_grid()
    T = np.zeros((32, 32))
    T[14:18, 14:18] = rng.random((4, 4)) + 1
    K = np.zeros((2, 2, 5, 5))
    K[..., 2, 2] = 1.0
    out = TK.convolve_blended(T, K, g, [1.0, 6.0], [1.0, 6.0], workers=1)
    assert np.allclose(out, T, atol=1e-14)
    K0, _ = _kernel_set(rng)
    out = TK.convolve_blended(T, K0, g, [1.0, 6.0], [1.0, 6.0], workers=1)
    assert abs(out.sum() - T.sum()) < 1e-12 * T.sum()      # sum_n w_n = 1, K_n sum to 1, interior source


def test_moment_form_equals_per_band_kernels():
    """C = K0 (*) sum_b T_b + K1 (*) sum_b delta_b T_b  ==  sum_b K_b (*) T_b  for K_b = K0 + delta_b K1."""
    rng = np.random.default_rng(1)
    g = fake_grid()
    Tb = [rng.random((32, 32)) for _ in range(4)]
    K0, K1 = _kernel_set(rng)
    delta = np.array([-0.9, -0.3, 0.4, 0.8])
    nodes = [1.0, 6.0]
    M0, M1 = sum(Tb), sum(d * t for d, t in zip(delta, Tb))
    mom = TK.convolve_blended(M0, K0, g, nodes, nodes, workers=1) + TK.convolve_blended(M1, K1, g, nodes, nodes, workers=1)
    per = sum(TK.convolve_blended(t, K0 + d * K1, g, nodes, nodes, workers=1) for d, t in zip(delta, Tb))
    assert np.abs(mom - per).max() < 1e-12 * np.abs(per).max()


# ------------------------------------------------------------------ match / difference assembly
def _synthetic_frame(n=64, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    C = 100 + 40 * np.sin(xx / 7.0) * np.cos(yy / 9.0) + rng.random((n, n)) * 5
    return C, rng


def test_match_recovers_known_a_and_b_and_difference_is_zero():
    C, _ = _synthetic_frame()
    coef = np.array([1.05, 0.02, -0.01, 0.005, 0.003, -0.004])          # a(x, y): Chebyshev total degree 2
    a_true = TK.eval_a(coef, 2, C.shape)
    target = a_true * C + 7.5
    noise = np.full(C.shape, 0.5)
    mask = np.zeros(C.shape, np.int32)
    stars = pd.DataFrame(dict(x=[30.0], y=[30.0], tmag=[9.0]))          # footprint radius 15 px excluded from the fit
    from syndiff_pipeline.forward_model.chain.final import final_image
    D, res = final_image(target, C, noise, mask, stars, 2)
    assert np.abs(D).max() < 1e-8 and abs(res["b"] - 7.5) < 1e-8
    assert np.allclose(res["coef_a"], coef, atol=1e-9)
    assert res["n_bright_excl"] > 0 and res["n_keep"] < C.size
    # a pure-noise perturbation passes straight through to D
    noisy = target + np.random.default_rng(0).normal(size=C.shape) * 0.5
    D2, res2 = final_image(noisy, C, noise, mask, stars, 2)
    assert abs(np.std(D2) - 0.5) < 0.05 and 0.8 < res2["chi2_red_keep"] < 1.2


def test_selection_accepts_mask_0_and_64_only():
    noise = np.ones((8, 8))
    mask = np.zeros((8, 8), np.int32)
    mask[0, 0], mask[0, 1], mask[0, 2] = 64, 1, 32
    good, sel = TK.selection(noise, mask, pd.DataFrame(dict(x=[], y=[], tmag=[])))
    assert good[0, 0] and not good[0, 1] and not good[0, 2] and sel["n_bright_excl"] == 0


def test_write_fits_roundtrip(tmp_path):
    from astropy.io import fits
    D = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    TK.write_fits(tmp_path / "x/hp_d.fits.fz", fits.Header(), [(D, fits.Header()), (D * 2, fits.Header())])
    with fits.open(tmp_path / "x/hp_d.fits.fz") as h:
        assert np.array_equal(h[1].data, D) and np.array_equal(h[2].data, D * 2)


# ------------------------------------------------------------------ adopted-weight rescale, both store cases
def _write_adopted(path):
    s = [D13[b] / PROD[b] for b in BANDS]
    path.write_text(json.dumps(dict(weights_rizy=[D13[b] for b in BANDS], scale_vs_production=s,
                                    colour_map_label_from_new=[29.64, 0.98])))
    return s


def _setup_stage(tmp_path, rng, *, adopted_store):
    """Tiny per-band templates + kernels in a real config's stage dirs; returns (cfg, T_production, scales, kernels)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    extra = {"combined_store_weights": "adopted"} if adopted_store else {}
    cfg = BF.make_cfg(tmp_path, extra_inputs=extra)
    P = PP.chain_paths(cfg)
    s = _write_adopted(tmp_path / "adopted.json")
    Tprod = {b: rng.random((32, 32)) for b in BANDS}
    P.band_templates.mkdir(parents=True)
    for b, k in zip(BANDS, s):
        # an adopted-weight store carries T'_b = s_b T_b already; a production store carries T_b
        np.save(P.band_templates / f"T_{b}.npy", Tprod[b] * float(k) if adopted_store else Tprod[b])
    if adopted_store:
        P.store_weights_json.write_text(json.dumps(dict(store_band_weights=D13)))
    K0, K1 = _kernel_set(rng)
    P.kernels.mkdir(parents=True)
    meta = json.dumps(dict(u_ref_nm=790.0, du_dc_nm_per_mag=25.0))
    delta_old = np.array([-0.9, -0.3, 0.4, 0.8])
    np.savez(P.kernels / "band_epsf.npz", K0=K0, K1=K1, delta_b=delta_old, node_x=np.array([1.0, 6.0]),
             node_y=np.array([1.0, 6.0]), meta=meta)
    np.savez(P.kernels / "K_achrom.npz", K=K0, node_x=np.array([1.0, 6.0]), node_y=np.array([1.0, 6.0]), meta=meta)
    return cfg, Tprod, np.array(s), (K0, K1, delta_old)


@pytest.fixture
def tiny_tk(monkeypatch):
    monkeypatch.setattr(TK, "load_grid", lambda mapping_root: fake_grid())
    monkeypatch.setattr(TK, "trim", lambda a, grid: np.asarray(a))


def test_band_scales_both_store_cases(tmp_path):
    cfg_p = BF.make_cfg(tmp_path)
    s = _write_adopted(tmp_path / "adopted.json")
    assert list(FN.band_scales(cfg_p, True)) == s                      # production store: w'/w, = ADOPTED scale_vs_production
    assert list(FN.band_scales(cfg_p, False)) == [1.0] * 4             # unweighted variants: no rescale
    cfg_a = BF.make_cfg(tmp_path, extra_inputs={"combined_store_weights": "adopted"})
    assert FN.store_weights(cfg_a) == D13                              # no store_weights.json yet: the chain assumption
    assert list(FN.band_scales(cfg_a, True)) == [1.0] * 4              # adopted store: nothing left to rescale
    P = PP.chain_paths(cfg_a)
    P.band_templates.mkdir(parents=True)
    P.store_weights_json.write_text(json.dumps(dict(store_band_weights=PROD)))
    assert FN.store_weights(cfg_a) == PROD                             # f04's recorded weights win
    assert list(FN.band_scales(cfg_a, True)) == s


def test_w_variant_identical_for_production_and_adopted_store(tmp_path, tiny_tk):
    """The rescale is relative to the store: T (production store) * s  ==  T' (adopted store) * 1  -> same image, bitwise."""
    rng = np.random.default_rng(5)
    cfg_p, Tprod, s, _ = _setup_stage(tmp_path / "p", rng, adopted_store=False)
    rng = np.random.default_rng(5)
    cfg_a, _, _, _ = _setup_stage(tmp_path / "a", rng, adopted_store=True)
    for v in ("band_w", "achrom_w"):
        Cp = FN.model_image(cfg_p, v, workers=1)
        Ca = FN.model_image(cfg_a, v, workers=1)
        assert np.array_equal(Cp, Ca), v
    info = json.loads((PP.chain_paths(cfg_p).final / "cache/C_band_w.json").read_text())
    assert info["scale_b"] == s.tolist() and info["store_band_weights"] == PROD
    info = json.loads((PP.chain_paths(cfg_a).final / "cache/C_band_w.json").read_text())
    assert info["scale_b"] == [1.0] * 4 and info["store_band_weights"] == D13


def test_band_variants_match_manual_moment_form_with_delta_recalibration(tmp_path, tiny_tk):
    rng = np.random.default_rng(6)
    cfg, Tprod, s, (K0, K1, delta_old) = _setup_stage(tmp_path, rng, adopted_store=False)
    g, nodes = fake_grid(), [1.0, 6.0]
    # unweighted: templates as built, delta_b from the kernel file
    M0 = sum(Tprod[b] for b in BANDS)
    M1 = sum(float(d) * Tprod[b] for d, b in zip(delta_old, BANDS))
    ref = TK.convolve_blended(M0, K0, g, nodes, nodes, workers=1) + TK.convolve_blended(M1, K1, g, nodes, nodes, workers=1)
    C = FN.model_image(cfg, "band", workers=1)
    assert np.array_equal(C, TK.block_sum(ref, 4))
    # weighted: T' = s T and delta' = (alpha + beta lambda_b - u_ref)/(du/dc)  (recalibrated, differs from delta_old)
    dnew = (29.64 + 0.98 * np.array([617.0, 752.0, 866.0, 962.0]) - 790.0) / 25.0
    Tw = {b: Tprod[b] * float(k) for b, k in zip(BANDS, s)}
    M0 = sum(Tw.values())
    M1 = sum(float(d) * Tw[b] for d, b in zip(dnew, BANDS))
    ref = TK.convolve_blended(M0, K0, g, nodes, nodes, workers=1) + TK.convolve_blended(M1, K1, g, nodes, nodes, workers=1)
    Cw = FN.model_image(cfg, "band_w", workers=1)
    assert np.array_equal(Cw, TK.block_sum(ref, 4)) and not np.array_equal(Cw, C)
    assert not np.allclose(dnew, delta_old)
    # achromatic: sum_b T_b (*) K, rescaled for _w
    Ca = FN.model_image(cfg, "achrom_w", workers=1)
    assert np.array_equal(Ca, TK.block_sum(TK.convolve_blended(sum(Tw.values()), K0, g, nodes, nodes, workers=1), 4))
    # cache: second call returns the stored array
    assert np.array_equal(FN.model_image(cfg, "band_w", workers=1), Cw)


def test_unknown_variant_rejected(tmp_path):
    cfg = BF.make_cfg(tmp_path)
    from syndiff_pipeline.forward_model.chain import config as C
    C.mark_done(cfg.stage_dir("kernels"))
    C.mark_done(cfg.stage_dir("hotpants"))
    with pytest.raises(ValueError, match="unknown variant"):
        FN.run(cfg, ["band_a3w"])


# ------------------------------------------------------------------ slow parity (e2e F1)
slow = pytest.mark.slow
DR = BF.E2E_RUN / "perband/dryrun/F1"


@slow
def test_band_w_hp_d_matches_e2e_bitwise(tmp_path):
    """Final band_a3w hp_d (both frames) re-assembled from the e2e T_b, kernels and Hotpants references: every pixel bitwise."""
    from astropy.io import fits
    from syndiff_pipeline.forward_model.chain import config as C
    BF.need(BF.E2E_RUN / "perband/out/band_templates/F1/a3/T_r.npy", BF.E2E_RUN / "kernels/F1/band_epsf_a3.npz",
            DR / BF.STEM / "band_a3w/hp_d", BF.E2E / "s3_fit/scene", BF.E2E_RUN / "mapping/oversampling_4",
            BF.E2E_RUN / "hotpants" / BF.STEM)
    cfg = BF.f1_cfg(tmp_path)
    root = tmp_path / "F1"
    BF.link(root / "perband/band_templates", BF.E2E_RUN / "perband/out/band_templates/F1/a3")
    kd = root / "kernels"
    kd.mkdir(parents=True)
    BF.link(kd / "band_epsf.npz", BF.E2E_RUN / "kernels/F1/band_epsf_a3.npz")
    BF.link(kd / "K_achrom.npz", BF.E2E_RUN / "kernels/F1/K_achrom_a3.npz")
    C.mark_done(kd)
    for s in (BF.STEM, BF.UNSEEN):
        BF.link(root / "hotpants" / s, BF.E2E_RUN / "hotpants" / s)
    C.mark_done(root / "hotpants")
    BF.link(root / "scene_boot", BF.E2E / "s3_fit/scene")
    BF.link(root / "mapping", BF.E2E_RUN / "mapping")
    FN.run(cfg, ["band_w"], workers=16)
    # the convolved native template, bitwise
    assert np.array_equal(np.load(root / "final/cache/C_band_w.npy"), np.load(DR / "cache/C_band_a3w.npy"))
    old_sum = json.loads((DR / "summary.json").read_text())
    new_sum = json.loads((root / "final/summary.json").read_text())
    for stem in (BF.STEM, BF.UNSEEN):
        a = root / "final" / stem / "band_w/hp_d" / f"{stem}_hp_d.fits.fz"
        b = DR / stem / "band_a3w/hp_d" / f"{stem}_hp_d.fits.fz"
        with fits.open(a) as ha, fits.open(b) as hb:
            for i in (1, 2, 3):
                assert ha[i].data.dtype == hb[i].data.dtype
                assert np.array_equal(ha[i].data, hb[i].data, equal_nan=True), (stem, i)
        n, o = new_sum[f"{stem}/band_w"], old_sum[f"{stem}/band_a3w"]
        assert n["chi2_red_keep"] == o["chi2_red_keep"] and n["b"] == o["b"] and n["a"] == o["a"]
