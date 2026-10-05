# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Round-trip tests for fit_bundle + train_loop (no hp_d)."""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import fit_bundle as FB
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents


def _tiny_bundle() -> FB.FitBundle:
    n_g, n_t, s = 2, 3, 7
    n_stars = 3
    k = 2
    degree = 2
    exps = tuple(sci2idl_exponents(degree))
    cheb = CW.ChebWcsStatic(
        ra0_deg=200.0,
        dec0_deg=80.0,
        cd_inv=np.eye(2),
        crpix=np.array([100.0, 100.0]),
        center=np.array([1600.0, 1600.0]),
        half_extents=np.array([256.0, 256.0]),
        poly_degree=degree,
        exponents=exps,
    )
    node_x = np.array([1500.0, 1700.0])
    node_y = np.array([1500.0, 1700.0])
    grid = EM.EpsfGridStatic(
        node_x=node_x, node_y=node_y,
        node_col_ccd=node_x + 44.0, node_row_ccd=node_y,
    )
    _, node, _ = EM.node_geometry(s)
    rng = np.random.default_rng(0)
    base = rng.random((2, 2, node, node)).astype(np.float32)
    base /= base.reshape(2, 2, -1).sum(axis=-1)[..., None, None]
    modes = rng.normal(size=(1, 2, 2, node, node)).astype(np.float32)
    n_wcs, n_w = 4, 3
    params0 = {
        "wcs_coeff": np.zeros((2 * len(exps), n_wcs), dtype=np.float32),
        "epsf_base_raw": np.asarray(EM.encode_epsf_base(base)),
        "epsf_modes": np.asarray(EM.encode_epsf_modes(modes, base)),
        "w_coeff": np.zeros((1, n_w), dtype=np.float32),
    }
    members = np.array([[0, 1], [2, -1]], dtype=np.int32)
    valid = np.array([[True, True], [True, False]], dtype=bool)
    ra = np.linspace(199.0, 201.0, n_stars)
    dec = np.linspace(79.0, 81.0, n_stars)
    x_lin, y_lin, cheb_basis = CW.star_basis(
        np.asarray(ra, dtype=np.float32),
        np.asarray(dec, dtype=np.float32),
        cheb,
    )
    return FB.FitBundle(
        data=rng.random((n_g, n_t, s, s)).astype(np.float32),
        noise=np.ones((n_g, n_t, s, s), dtype=np.float32),
        weight=np.ones((n_g, n_t, s, s), dtype=np.float32),
        stamp_center_x=np.array([1600.0, 1610.0]),
        stamp_center_y=np.array([1600.0, 1610.0]),
        mask_active=np.ones((n_g, n_t), dtype=np.float32),
        ra=ra,
        dec=dec,
        x_lin=np.asarray(x_lin, dtype=np.float32),
        y_lin=np.asarray(y_lin, dtype=np.float32),
        cheb_basis=np.asarray(cheb_basis, dtype=np.float32),
        members=members,
        valid=valid,
        kept_star_mask=np.ones(n_stars, dtype=bool),
        max_group_size=k,
        stamp_snr_weight=np.ones(n_g, dtype=np.float32),
        fit_radius_stage1=np.full(n_g, 3.0, dtype=np.float32),
        fit_radius_stage23=np.full(n_g, 3.5, dtype=np.float32),
        cheb_static=cheb,
        epsf_grid=grid,
        wcs_frame_basis=np.eye(n_t, n_wcs, dtype=np.float32),
        w_frame_basis=np.eye(n_t, n_w, dtype=np.float32),
        epsf_base=base,
        epsf_modes=modes,
        params0=params0,
        t_exp_sec=1425.6,
        stamp_physical=s,
        k_tiers=(k,),
        meta={"test": True},
    )


def test_fit_bundle_roundtrip(tmp_path):
    bundle = _tiny_bundle()
    path = FB.save_fit_bundle(tmp_path / "fit_bundle.npz", bundle)
    loaded = FB.load_fit_bundle(path)
    assert loaded.n_groups == bundle.n_groups
    assert loaded.n_frames == bundle.n_frames
    assert loaded.stamp_physical == bundle.stamp_physical
    np.testing.assert_allclose(loaded.data, bundle.data)
    np.testing.assert_allclose(loaded.wcs_frame_basis, bundle.wcs_frame_basis)
    np.testing.assert_allclose(loaded.x_lin, bundle.x_lin)
    np.testing.assert_allclose(loaded.y_lin, bundle.y_lin)
    np.testing.assert_allclose(loaded.cheb_basis, bundle.cheb_basis)
    assert loaded.cheb_static.poly_degree == bundle.cheb_static.poly_degree
    assert loaded.cheb_static.exponents == bundle.cheb_static.exponents
    assert loaded.epsf_grid.n_rows == 2
    assert set(loaded.params0) == set(bundle.params0)


def test_train_loop_few_steps_from_bundle(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import train_loop as TL

    bundle = _tiny_bundle()
    path = FB.save_fit_bundle(tmp_path / "fit_bundle.npz", bundle)
    loaded = FB.load_fit_bundle(path)
    out = tmp_path / "out"
    TL.run_stages_from_bundle(
        loaded,
        out_dir=out,
        stage=1,
        start_stage=1,
        steps_per_stage=[2, 0, 0],
        lr_per_stage=[1e-2, 3e-4, 1e-4],
        log_every=1,
        checkpoint_every=0,
        reject_every=0,
        stage1_core_stamp=0,
    )
    assert (out / "params_stage1.npz").exists()
    assert (out / "fit_meta.json").exists()


def test_graceful_stop_exports_bounded_gpu_flux(tmp_path, monkeypatch):
    """STOP preserves the checkpoint and still emits the GPU flux solution."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import fit as FIT
    from syndiff_pipeline.forward_model import gpu_flux_export as GFE
    from syndiff_pipeline.forward_model import train_loop as TL

    bundle = _tiny_bundle()
    out = tmp_path / "out"
    stop_file = out / "STOP"
    seen = {}

    def fake_run_stage(params, *_args, **_kwargs):
        stop_file.touch()
        return params, [{"stage": 1, "step": 0, "loss": 1.0}]

    def fake_flux_export(*_args, **kwargs):
        seen.update(kwargs)
        return out / "flux_solved.npz"

    monkeypatch.setattr(FIT, "run_stage", fake_run_stage)
    monkeypatch.setattr(GFE, "export_gpu_flux_solution", fake_flux_export)
    meta = TL.run_stages_from_bundle(
        bundle,
        out_dir=out,
        stage=1,
        start_stage=1,
        steps_per_stage=[2, 0, 0],
        lr_per_stage=[1e-2, 3e-4, 1e-4],
        log_every=1,
        checkpoint_every=1,
        reject_every=0,
        stage1_core_stamp=0,
        stop_file=stop_file,
    )

    assert meta["fit_complete"] is False
    assert meta["gpu_flux_exported"] is True
    assert seen["frame_block"] == 8
    assert (out / "params_stage1.npz").is_file()
    assert (out / "params.npz").is_file()
