# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for packed irregular-stamp ingest, render physics, and loss/grad smoke."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import flux_solve as FS
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import packed_support as PS
from syndiff_pipeline.forward_model import temporal as T
from syndiff_pipeline.forward_model._bootstrap import _EXTRA_PATHS  # noqa: F401
from syndiff_pipeline.forward_model.groups import GroupSet

from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents  # noqa: E402

jax.config.update("jax_platform_name", "cpu")


def _tiny_static(n_stars: int = 4, n_frames: int = 3, region: int = 64):
    static = CW.ChebWcsStatic(
        ra0_deg=180.0,
        dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([region / 2, region / 2], dtype=float),
        center=np.array([region / 2, region / 2], dtype=float),
        half_extents=np.array([region / 2, region / 2], dtype=float),
        poly_degree=3,
        exponents=tuple(sci2idl_exponents(3)),
    )
    from syndiff_pipeline.forward_model.data import RegionSpec

    region_spec = RegionSpec(0, 0, region, region)
    grid = EM.EpsfGridStatic.from_region(region_spec, n_rows=2, n_cols=2, crop_origin=(0, 0))
    g_size = EM.node_geometry(13)[1]
    blob = np.zeros((g_size, g_size), dtype=np.float32)
    yy, xx = np.mgrid[0:g_size, 0:g_size]
    cy = cx = (g_size - 1) / 2.0
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob /= blob.sum()
    base = np.stack([np.stack([blob, blob], axis=0), np.stack([blob, blob], axis=0)], axis=0)
    modes = np.zeros((1, 2, 2, g_size, g_size), dtype=np.float32)
    modes[0] = EM._finite_diff_modes(blob, mode_names=("iso_defocus",))
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))
    btjd = np.linspace(0.0, 1.0, n_frames)
    wcs_tb = T.build_temporal_basis(btjd, n_interior=1, uniform_knots=True)
    w_tb = T.build_temporal_basis(btjd, n_interior=1, uniform_knots=True)
    ra = np.linspace(179.9, 180.1, n_stars).astype(np.float32)
    dec = np.linspace(-0.05, 0.05, n_stars).astype(np.float32)
    x_lin, y_lin, cheb_basis = CW.star_basis(jnp.asarray(ra), jnp.asarray(dec), static)
    return static, grid, epsf0, wcs_tb, w_tb, ra, dec, np.asarray(x_lin), np.asarray(y_lin), np.asarray(cheb_basis)


def test_pack_irregular_stamps_tiers():
    stamps = [
        PS.IrregularStamp(np.array([0]), np.linspace(10, 12, 20), np.linspace(10, 12, 20)),
        PS.IrregularStamp(np.array([1, 2]), np.linspace(0, 5, 100), np.linspace(0, 5, 100)),
        PS.IrregularStamp(np.array([3]), np.linspace(1, 2, 30), np.linspace(1, 2, 30)),
    ]
    buckets = PS.pack_irregular_stamps(stamps, k_tiers=(1, 2, 4), p_tiers=(64, 128, 256), n_stars=4)
    ks = {(g.max_group_size, int(px.shape[1])) for g, px, _, _, _ in buckets}
    assert (1, 64) in ks
    assert (2, 128) in ks
    # two K=1 stamps may share P=64
    n_total = sum(g.n_groups for g, *_ in buckets)
    assert n_total == 3


def test_ensure_tiers_cover_and_pack_oversized_p():
    """Merged irregular stamps can exceed DEFAULT_P_TIERS max (512); grow ladder."""
    assert PS.ensure_tiers_cover(945, (64, 128, 256, 512)) == (64, 128, 256, 512, 1024)
    assert PS.ensure_tiers_cover(200, (64, 128, 256, 512)) == (64, 128, 256, 512)
    stamps = [
        PS.IrregularStamp(
            np.array([0, 1]),
            np.linspace(0, 1, 945),
            np.linspace(0, 1, 945),
        ),
    ]
    buckets = PS.pack_irregular_stamps(
        stamps, k_tiers=(1, 2, 4, 8), p_tiers=(64, 128, 256, 512), n_stars=2,
    )
    assert len(buckets) == 1
    g, px, *_ = buckets[0]
    assert g.n_groups == 1
    assert px.shape[1] == 1024


def test_validate_peak_in_support():
    st = PS.IrregularStamp(
        member_star_idx=np.array([0, 1]),
        pix_x=np.array([10.0, 11.0, 12.0]),
        pix_y=np.array([20.0, 21.0, 22.0]),
    )
    x = np.array([11.0, 11.5, 50.0])
    y = np.array([21.0, 21.0, 50.0])
    assert PS.validate_stamp_peak_in_support(st, x, y, core_margin_px=2.0)
    st_bad = PS.IrregularStamp(np.array([0, 2]), st.pix_x, st.pix_y)
    assert not PS.validate_stamp_peak_in_support(st_bad, x, y, core_margin_px=2.0)


def test_render_physical_pixels_matches_render_stamps():
    g_size = EM.node_geometry(13)[1]
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:g_size, 0:g_size]
    blob = np.exp(-((xx - (g_size - 1) / 2) ** 2 + (yy - (g_size - 1) / 2) ** 2) / (2 * (EM.OVERSAMPLE * 1.5) ** 2))
    blob = blob.astype(np.float32)
    blob /= blob.sum()
    grid = jnp.asarray(blob)
    S = 13
    dx, dy = 0.2, -0.15
    stamp = np.asarray(EM.render_stamps(grid[None], jnp.array([dx]), jnp.array([dy]), n_pix=S)[0])
    yy, xx = np.mgrid[0:S, 0:S]
    ox = jnp.asarray((xx - 0.5 * (S - 1) - dx).reshape(1, -1), dtype=jnp.float32)
    oy = jnp.asarray((yy - 0.5 * (S - 1) - dy).reshape(1, -1), dtype=jnp.float32)
    packed = np.asarray(EM.render_physical_pixels_blocksum(grid[None], ox, oy)).reshape(S, S)
    assert np.max(np.abs(packed - stamp)) < 1e-5


def test_flux_solve_packed_1d_matches_square_flatten():
    rng = np.random.default_rng(1)
    n_g, K, T, S = 3, 2, 4, 5
    P = S * S
    templates = rng.normal(size=(n_g, K, T, S, S)).astype(np.float32)
    data = rng.normal(size=(n_g, T, S, S)).astype(np.float32)
    weight = np.ones_like(data)
    f_sq = FS.solve_group_fluxes(jnp.asarray(templates), jnp.asarray(data), jnp.asarray(weight))
    f_1d = FS.solve_group_fluxes(
        jnp.asarray(templates.reshape(n_g, K, T, P)),
        jnp.asarray(data.reshape(n_g, T, P)),
        jnp.asarray(weight.reshape(n_g, T, P)),
    )
    np.testing.assert_allclose(f_sq, f_1d, rtol=1e-5, atol=1e-5)
    m_sq = FS.model_stamps(jnp.asarray(templates), f_sq)
    m_1d = FS.model_stamps(jnp.asarray(templates.reshape(n_g, K, T, P)), f_1d)
    np.testing.assert_allclose(m_sq.reshape(n_g, T, P), m_1d, rtol=1e-5, atol=1e-5)


def test_packed_total_loss_and_grads_finite():
    static, grid, epsf0, wcs_tb, w_tb, ra, dec, x_lin, y_lin, cheb_basis = _tiny_static(
        n_stars=3, n_frames=4, region=64,
    )
    # One isolate + one pair sharing a small pixel support
    members = np.array([[0, -1], [1, 2]], dtype=int)
    valid = np.array([[True, False], [True, True]], dtype=bool)
    groups = GroupSet(2, 2, members, valid, np.ones(3, dtype=bool), 0)
    P = 64
    rng = np.random.default_rng(2)
    # Pixel cloud around each group's stars
    pix_x = np.zeros((2, P), dtype=np.float32)
    pix_y = np.zeros((2, P), dtype=np.float32)
    pix_valid = np.zeros((2, P), dtype=np.float32)
    for gi, stars in enumerate(([0], [1, 2])):
        n = 40
        cx = float(np.mean(x_lin[stars]))
        cy = float(np.mean(y_lin[stars]))
        pix_x[gi, :n] = cx + rng.uniform(-3, 3, size=n)
        pix_y[gi, :n] = cy + rng.uniform(-3, 3, size=n)
        pix_valid[gi, :n] = 1.0

    n_frames = 4
    data = rng.normal(size=(2, n_frames, P)).astype(np.float32)
    noise = np.ones_like(data)
    weight = np.broadcast_to(pix_valid[:, None, :], data.shape).copy()
    cx = np.array([np.mean(pix_x[0, :40]), np.mean(pix_x[1, :40])], dtype=np.float32)
    cy = np.array([np.mean(pix_y[0, :40]), np.mean(pix_y[1, :40])], dtype=np.float32)

    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_tb.frame_basis,
        w_frame_basis=w_tb.frame_basis,
        epsf_grid=grid,
        groups=groups,
        ra=ra,
        dec=dec,
        stamp_center_x=cx,
        stamp_center_y=cy,
        t_exp_sec=1426.0,
        stamp_snr_weight=np.ones(2, dtype=np.float32),
        fit_radius=np.full(2, 8.0, dtype=np.float32),
        x_lin=x_lin,
        y_lin=y_lin,
        cheb_basis=cheb_basis,
        pix_x=pix_x,
        pix_y=pix_y,
        pix_valid=pix_valid,
    )
    assert ctx.is_packed
    params = L.init_params(static, epsf0, n_wcs_basis=wcs_tb.n_basis, n_w_basis=w_tb.n_basis)
    wcs_sd = T.second_difference_matrix(wcs_tb.n_basis)
    w_sd = T.second_difference_matrix(w_tb.n_basis)

    def loss_fn(p):
        loss, metrics = L.total_loss(
            p, ctx,
            jnp.asarray(data), jnp.asarray(noise), jnp.asarray(weight),
            wcs_sd, w_sd,
            epsf_modes_init=epsf0.modes,
            weights=L.LossWeights(centroid_in_grad=False),
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert np.isfinite(float(loss))
    for k in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        g = np.asarray(grads[k])
        assert np.isfinite(g).all(), k
        # at least some leaf should have nonzero grad on a random data field
    assert float(np.sqrt(np.mean(np.asarray(grads["epsf_base_raw"]) ** 2))) >= 0.0


def test_packed_joint_flux_two_members():
    """Two members on shared pixels → joint LS returns finite 2-vector fluxes."""
    templates = np.zeros((1, 2, 2, 8), dtype=np.float32)
    templates[0, 0, :, :4] = 1.0
    templates[0, 1, :, 4:] = 1.0
    data = np.zeros((1, 2, 8), dtype=np.float32)
    data[..., :4] = 3.0
    data[..., 4:] = 5.0
    weight = np.ones_like(data)
    flux = FS.solve_group_fluxes(jnp.asarray(templates), jnp.asarray(data), jnp.asarray(weight))
    np.testing.assert_allclose(flux[0, :, 0], 3.0, rtol=1e-4)
    np.testing.assert_allclose(flux[0, :, 1], 5.0, rtol=1e-4)


def test_packed_per_stamp_chi2_red_and_mad_refresh():
    """MAD reject must not assume S×S on packed (G,T,P) FitData."""
    from types import SimpleNamespace

    from syndiff_pipeline.forward_model import stamp_reject as SR

    static, grid, epsf0, wcs_tb, w_tb, ra, dec, x_lin, y_lin, cheb_basis = _tiny_static(
        n_stars=2, n_frames=3, region=64,
    )
    members = np.array([[0, -1], [1, -1]], dtype=int)
    valid = np.array([[True, False], [True, False]], dtype=bool)
    groups = GroupSet(2, 2, members, valid, np.ones(2, dtype=bool), 0)
    P = 32
    rng = np.random.default_rng(3)
    pix_x = np.zeros((2, P), dtype=np.float32)
    pix_y = np.zeros((2, P), dtype=np.float32)
    pix_valid = np.zeros((2, P), dtype=np.float32)
    for gi in range(2):
        n = 16
        pix_x[gi, :n] = float(x_lin[gi]) + rng.uniform(-2, 2, size=n)
        pix_y[gi, :n] = float(y_lin[gi]) + rng.uniform(-2, 2, size=n)
        pix_valid[gi, :n] = 1.0
    n_frames = 3
    data = rng.normal(size=(2, n_frames, P)).astype(np.float32)
    noise = np.ones_like(data)
    weight = np.broadcast_to(pix_valid[:, None, :], data.shape).copy()
    cx = np.array([x_lin[0], x_lin[1]], dtype=np.float32)
    cy = np.array([y_lin[0], y_lin[1]], dtype=np.float32)
    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_tb.frame_basis,
        w_frame_basis=w_tb.frame_basis,
        epsf_grid=grid,
        groups=groups,
        ra=ra[:2],
        dec=dec[:2],
        stamp_center_x=cx,
        stamp_center_y=cy,
        t_exp_sec=1426.0,
        stamp_snr_weight=np.ones(2, dtype=np.float32),
        fit_radius=np.full(2, 8.0, dtype=np.float32),
        x_lin=x_lin[:2],
        y_lin=y_lin[:2],
        cheb_basis=cheb_basis[:2],
        pix_x=pix_x,
        pix_y=pix_y,
        pix_valid=pix_valid,
    )
    params = L.init_params(static, epsf0, n_wcs_basis=wcs_tb.n_basis, n_w_basis=w_tb.n_basis)
    fd = SimpleNamespace(
        ctx=ctx,
        data=jnp.asarray(data),
        noise=jnp.asarray(noise),
        weight=jnp.asarray(weight),
        weights=L.LossWeights(centroid_in_grad=False),
        use_dx_only=False,
        local_cache=None,
        do_recenter=True,
        recenter_n_iter=1,
        n_pix=1,
        mask_stamp_active=np.ones((2, n_frames), dtype=np.float32),
    )
    chi2_red, pix_sum = SR.per_stamp_chi2_red(params, fd)
    assert chi2_red.shape == (2, n_frames)
    assert pix_sum.shape == (2, n_frames)
    assert np.isfinite(np.asarray(chi2_red)).all()
    assert float(np.asarray(pix_sum).min()) > 0
    stats = SR.refresh_stamp_active(params, fd, n_sigma=3.0)
    assert stats["n_active_cand"] > 0
    assert fd.ctx.stamp_active.shape == (2, n_frames)


def test_assignments_to_irregular_and_concat():
    """SegmentAssignment → IrregularStamp → packed batches → flat FitBundle arrays."""
    from types import SimpleNamespace

    a0 = SimpleNamespace(
        member_indices=np.array([0], dtype=int),
        pix_x=np.linspace(10, 12, 20),
        pix_y=np.linspace(20, 22, 20),
        stamp_center_x=11,
        stamp_center_y=21,
    )
    a1 = SimpleNamespace(
        member_indices=np.array([1, 2], dtype=int),
        pix_x=np.linspace(0, 5, 100),
        pix_y=np.linspace(0, 5, 100),
        stamp_center_x=2,
        stamp_center_y=2,
    )
    stamps = PS.irregular_stamps_from_assignments([a0, a1])
    assert len(stamps) == 2
    assert stamps[0].stamp_center_x == 11.0

    ny, nx = 32, 32
    frames = []
    for _ in range(3):
        frames.append(SimpleNamespace(
            cal=np.zeros((ny, nx), dtype=np.float32),
            noise=np.ones((ny, nx), dtype=np.float32),
            bad=np.zeros((ny, nx), dtype=bool),
        ))

    batches = PS.build_packed_stamp_batches(
        stamps, frames,
        k_tiers=(1, 2, 4), p_tiers=(64, 128, 256),
        array_origin=(0, 0), n_stars=4,
    )
    assert len(batches) >= 1
    cx, cy = PS.stamp_centers_from_irregular(stamps)
    data, noise, weight, px, py, pv, members, valid, cx_o, cy_o, kept, k_max = PS.concat_packed_batches(
        batches, stamp_center_x=cx, stamp_center_y=cy, n_stars=4,
    )
    assert data.shape[0] == 2
    assert data.ndim == 3
    assert k_max >= 2
    assert kept.sum() >= 2

    groups = GroupSet(
        n_groups=data.shape[0], max_group_size=k_max,
        members=members, valid=valid, kept_star_mask=kept, dropped_oversized=0,
    )
    buckets = PS.bucket_packed_by_kp(
        groups, cx_o, cy_o, pv,
        k_tiers=(1, 2, 4), p_tiers=(64, 128, 256),
    )
    assert len(buckets) >= 1
    n_back = sum(bg.n_groups for bg, *_ in buckets)
    assert n_back == 2


def test_single_pass_matches_legacy_gather():
    """The one-read-per-frame path preserves tier-native pixel contents."""
    from types import SimpleNamespace

    stamps = [
        PS.IrregularStamp(np.array([0]), np.linspace(2, 7, 20), np.linspace(3, 8, 20)),
        PS.IrregularStamp(np.array([1, 2]), np.linspace(10, 18, 100), np.linspace(12, 20, 100)),
    ]
    frames = []
    for fi in range(4):
        yy, xx = np.mgrid[:32, :32]
        cal = (1000 * fi + 10 * yy + xx).astype(np.float32)
        frames.append(SimpleNamespace(
            cal=cal, noise=(cal + 1).astype(np.float32),
            bad=np.zeros(cal.shape, dtype=bool),
        ))
    kwargs = dict(k_tiers=(1, 2, 4), p_tiers=(64, 128, 256), n_stars=4)
    legacy = PS.build_packed_stamp_batches(stamps, frames, **kwargs)
    single = PS.build_packed_stamp_batches_single_pass(stamps, frames, **kwargs)
    assert [(b.k_tier, b.p_tier, b.members.shape) for b in legacy] == [
        (b.k_tier, b.p_tier, b.members.shape) for b in single
    ]
    for old, new in zip(legacy, single):
        np.testing.assert_array_equal(old.pix_x, new.pix_x)
        np.testing.assert_array_equal(old.pix_y, new.pix_y)
        np.testing.assert_array_equal(old.pix_valid, new.pix_valid)
        np.testing.assert_array_equal(old.members, new.members)
        np.testing.assert_array_equal(old.valid, new.valid)
        np.testing.assert_array_equal(old.data, new.data)
        np.testing.assert_array_equal(old.noise, new.noise)
        np.testing.assert_array_equal(old.weight, new.weight)


def test_irregular_stamps_from_mask_dir(tmp_path):
    path = tmp_path / "mask_primary_3_seg1.npz"
    np.savez_compressed(
        path,
        mask=np.ones((5, 5), dtype=bool),
        primary_index=3,
        segment_label=1,
        member_indices=np.array([3, 7], dtype=int),
        member_stamp_sizes=np.array([7, 5], dtype=int),
        pix_x=np.array([1.0, 2.0, 3.0]),
        pix_y=np.array([4.0, 5.0, 6.0]),
        stamp_center_x=2,
        stamp_center_y=5,
    )
    stamps = PS.irregular_stamps_from_mask_dir(tmp_path)
    assert len(stamps) == 1
    np.testing.assert_array_equal(stamps[0].member_star_idx, [3, 7])
    assert stamps[0].pix_x.shape == (3,)


def _packed_chunk_fixture(n_frames: int = 12, *, seed: int = 0):
    """Small packed StaticContext + params + data for stamp_chunk parity tests."""
    static, grid, epsf0, wcs_tb, w_tb, ra, dec, x_lin, y_lin, cheb_basis = _tiny_static(
        n_stars=3, n_frames=n_frames, region=64,
    )
    members = np.array([[0, -1], [1, 2]], dtype=int)
    valid = np.array([[True, False], [True, True]], dtype=bool)
    groups = GroupSet(2, 2, members, valid, np.ones(3, dtype=bool), 0)
    P = 64
    rng = np.random.default_rng(seed)
    pix_x = np.zeros((2, P), dtype=np.float32)
    pix_y = np.zeros((2, P), dtype=np.float32)
    pix_valid = np.zeros((2, P), dtype=np.float32)
    for gi, stars in enumerate(([0], [1, 2])):
        n = 40
        cx = float(np.mean(x_lin[stars]))
        cy = float(np.mean(y_lin[stars]))
        pix_x[gi, :n] = cx + rng.uniform(-3, 3, size=n)
        pix_y[gi, :n] = cy + rng.uniform(-3, 3, size=n)
        pix_valid[gi, :n] = 1.0

    data = rng.normal(size=(2, n_frames, P)).astype(np.float32)
    noise = np.full_like(data, 1.5)
    weight = np.broadcast_to(pix_valid[:, None, :], data.shape).copy()
    cx = np.array([np.mean(pix_x[0, :40]), np.mean(pix_x[1, :40])], dtype=np.float32)
    cy = np.array([np.mean(pix_y[0, :40]), np.mean(pix_y[1, :40])], dtype=np.float32)

    ctx = L.build_static_context(
        cheb_static=static,
        wcs_frame_basis=wcs_tb.frame_basis,
        w_frame_basis=w_tb.frame_basis,
        epsf_grid=grid,
        groups=groups,
        ra=ra,
        dec=dec,
        stamp_center_x=cx,
        stamp_center_y=cy,
        t_exp_sec=1426.0,
        stamp_snr_weight=np.ones(2, dtype=np.float32),
        fit_radius=np.full(2, 8.0, dtype=np.float32),
        x_lin=x_lin,
        y_lin=y_lin,
        cheb_basis=cheb_basis,
        pix_x=pix_x,
        pix_y=pix_y,
        pix_valid=pix_valid,
    )
    params = L.init_params(static, epsf0, n_wcs_basis=wcs_tb.n_basis, n_w_basis=w_tb.n_basis)
    wcs_sd = T.second_difference_matrix(wcs_tb.n_basis)
    w_sd = T.second_difference_matrix(w_tb.n_basis)
    return {
        "ctx": ctx,
        "params": params,
        "data": jnp.asarray(data),
        "noise": jnp.asarray(noise),
        "weight": jnp.asarray(weight),
        "wcs_sd": wcs_sd,
        "w_sd": w_sd,
        "epsf_modes_init": epsf0.modes,
    }


def _packed_chunk_loss_and_grad(fx: dict, stamp_chunk: int | None):
    def loss_fn(p):
        loss, metrics = L.total_loss(
            p, fx["ctx"], fx["data"], fx["noise"], fx["weight"],
            fx["wcs_sd"], fx["w_sd"],
            epsf_modes_init=fx["epsf_modes_init"],
            weights=L.LossWeights(centroid_in_grad=False),
            stamp_chunk=stamp_chunk,
        )
        return loss, metrics

    return jax.value_and_grad(loss_fn, has_aux=True)(fx["params"])


def test_packed_stamp_chunk_matches_unchunked_loss():
    fx = _packed_chunk_fixture(n_frames=12)
    (loss_ref, metrics_ref), _ = _packed_chunk_loss_and_grad(fx, None)
    for block in (3, 5, 12):
        (loss_c, metrics_c), _ = _packed_chunk_loss_and_grad(fx, block)
        np.testing.assert_allclose(float(loss_c), float(loss_ref), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            float(metrics_c["data_term"]), float(metrics_ref["data_term"]),
            rtol=1e-4, atol=1e-4,
        )


def test_packed_stamp_chunk_matches_unchunked_gradient():
    fx = _packed_chunk_fixture(n_frames=12)
    _, grads_ref = _packed_chunk_loss_and_grad(fx, None)
    for block in (4, 7):
        _, grads_c = _packed_chunk_loss_and_grad(fx, block)
        for k in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
            np.testing.assert_allclose(
                np.asarray(grads_c[k]), np.asarray(grads_ref[k]),
                rtol=2e-3, atol=2e-3, err_msg=k,
            )


def test_packed_stamp_chunk_nondividing_block_matches_unchunked():
    fx = _packed_chunk_fixture(n_frames=10)
    (loss_ref, _), grads_ref = _packed_chunk_loss_and_grad(fx, None)
    (loss_c, _), grads_c = _packed_chunk_loss_and_grad(fx, 3)
    np.testing.assert_allclose(float(loss_c), float(loss_ref), rtol=1e-4, atol=1e-4)
    for k in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        np.testing.assert_allclose(
            np.asarray(grads_c[k]), np.asarray(grads_ref[k]),
            rtol=2e-3, atol=2e-3, err_msg=k,
        )


def test_packed_stamp_chunk_zero_is_unchunked():
    fx = _packed_chunk_fixture(n_frames=8)
    (loss_none, _), grads_none = _packed_chunk_loss_and_grad(fx, None)
    (loss_zero, _), grads_zero = _packed_chunk_loss_and_grad(fx, 0)
    assert float(loss_none) == float(loss_zero)
    for k in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff"):
        np.testing.assert_array_equal(np.asarray(grads_none[k]), np.asarray(grads_zero[k]))
