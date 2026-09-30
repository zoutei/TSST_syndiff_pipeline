# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""CPU invariants for isolated faint WCS anchors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import fit_bundle as FB
from syndiff_pipeline.forward_model import irregular_stamps as IS
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model.groups import GroupSet
from test_fm_fit_bundle import _tiny_bundle


jax.config.update("jax_platform_name", "cpu")


def _mixed_context():
    """Four K=1 packed groups: two bright and two faint."""
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents

    degree, region, n_frames = 1, 64, 2
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
    # Keep the ePSF grid deliberately tiny: the test exercises routing, not
    # high-resolution profile fidelity, and should remain a CPU unit test.
    g_size = EM.node_geometry(3)[1]
    yy, xx = np.mgrid[0:g_size, 0:g_size]
    blob = np.exp(-((xx - (g_size - 1) / 2) ** 2 + (yy - (g_size - 1) / 2) ** 2) / 18).astype(np.float32)
    blob /= blob.sum()
    base = np.broadcast_to(blob, (2, 2, g_size, g_size)).copy()
    modes = np.zeros((1, 2, 2, g_size, g_size), dtype=np.float32)
    modes[0] = EM._finite_diff_modes(blob, mode_names=("iso_defocus",))
    epsf = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))
    ra = np.array([179.95, 179.98, 180.02, 180.05], dtype=np.float32)
    dec = np.array([-0.04, -0.01, 0.02, 0.05], dtype=np.float32)
    x_lin, y_lin, cheb_basis = CW.star_basis(jnp.asarray(ra), jnp.asarray(dec), static)
    groups = GroupSet(4, 1, np.arange(4, dtype=np.int32)[:, None], np.ones((4, 1), bool), np.ones(4, bool), 0)
    P = 49
    pix_x = np.zeros((4, P), np.float32)
    pix_y = np.zeros((4, P), np.float32)
    for i, (cx, cy) in enumerate(zip(np.asarray(x_lin), np.asarray(y_lin))):
        py, px = np.mgrid[-3:4, -3:4]
        pix_x[i] = (round(float(cx)) + px).ravel()
        pix_y[i] = (round(float(cy)) + py).ravel()
    ctx = L.build_static_context(
        cheb_static=static, wcs_frame_basis=np.eye(n_frames, 2, dtype=np.float32),
        w_frame_basis=np.eye(n_frames, 2, dtype=np.float32), epsf_grid=grid, groups=groups,
        ra=ra, dec=dec, stamp_center_x=np.asarray(x_lin), stamp_center_y=np.asarray(y_lin),
        t_exp_sec=1.0, stamp_snr_weight=np.ones(4, np.float32), fit_radius=np.full(4, 4.0, np.float32),
        x_lin=np.asarray(x_lin), y_lin=np.asarray(y_lin), cheb_basis=np.asarray(cheb_basis),
        pix_x=pix_x, pix_y=pix_y, pix_valid=np.ones((4, P), np.float32),
        is_epsf_contributor=np.array([True, True, False, False]),
    )
    return ctx, L.init_params(static, epsf, n_wcs_basis=2, n_w_basis=2)


def test_faint_pixels_change_only_wcs_data_gradient():
    ctx, params = _mixed_context()
    rng = np.random.default_rng(7)
    data = jnp.asarray(rng.normal(size=(4, 2, 49)).astype(np.float32))
    def objective(p, d):
        # A simple pixel residual isolates the renderer's routing from the
        # profile-flux solve; it keeps this mandatory CPU test lightweight.
        templates, *_ = L.forward_model(p, ctx, do_recenter=False)
        return jnp.sum((templates[:, 0] - d) ** 2)

    grad_a = jax.grad(objective)(params, data)
    data_faint_changed = data.at[2:].add(0.7)
    grad_b = jax.grad(objective)(params, data_faint_changed)
    for key in ("epsf_base_raw", "epsf_modes", "w_coeff"):
        np.testing.assert_array_equal(np.asarray(grad_a[key]), np.asarray(grad_b[key]))
    assert not np.allclose(np.asarray(grad_a["wcs_coeff"]), np.asarray(grad_b["wcs_coeff"]))


def test_faint_anchor_pixel_exclusion_is_complete():
    # Star 1 is six pixels from star 0: it clears r_iso=5 but its S=7 square
    # overlaps star 0's square, so ordering must reject it by claimed pixel.
    stamps, stats = IS.select_isolated_faint_anchor_stamps(
        np.array([30.0, 36.0]), np.array([30.0, 30.0]), np.array([11.2, 11.4]),
        isolation_radius_px=5.0, stamp_size=7,
    )
    assert stats["retained"] == 1
    assert stats["rejected_pixel_overlap"] == 1
    # A claimed irregular-support pixel also rejects a catalog-isolated square.
    stamps2, stats2 = IS.select_isolated_faint_anchor_stamps(
        np.array([20.0]), np.array([20.0]), np.array([11.3]),
        claimed_pixels={(17, 20)}, isolation_radius_px=5.0, stamp_size=7,
    )
    assert not stamps2
    assert stats2["rejected_pixel_overlap"] == 1
    assert len({(int(x), int(y)) for x, y in zip(stamps[0].pix_x, stamps[0].pix_y)}) == 49


def test_cap_faint_anchors_by_mag_keeps_brightest():
    stamps, stats = IS.select_isolated_faint_anchor_stamps(
        np.array([10.0, 40.0, 70.0, 100.0]), np.array([10.0, 10.0, 10.0, 10.0]),
        np.array([12.9, 11.1, 12.0, 11.6]), isolation_radius_px=5.0, stamp_size=7,
    )
    assert stats["retained"] == 4
    mag = np.array([12.9, 11.1, 12.0, 11.6])

    kept, cap_stats = IS.cap_faint_anchors_by_mag(stamps, mag, n_bright=0, max_stars=2)
    assert cap_stats == {"capped": True, "dropped": 2, "effective_mag_hi": 11.6}
    assert len(kept) == 2
    kept_mags = sorted(mag[a.primary_index] for a in kept)
    assert kept_mags == [11.1, 11.6]  # the two brightest of the four

    # n_bright already consumes the whole budget: every faint anchor is dropped.
    kept0, cap_stats0 = IS.cap_faint_anchors_by_mag(stamps, mag, n_bright=5, max_stars=5)
    assert kept0 == []
    assert cap_stats0 == {"capped": True, "dropped": 4, "effective_mag_hi": None}

    # Under the cap: no-op, same list identity semantics (unchanged).
    kept1, cap_stats1 = IS.cap_faint_anchors_by_mag(stamps, mag, n_bright=0, max_stars=10)
    assert kept1 == stamps
    assert cap_stats1 == {"capped": False, "dropped": 0, "effective_mag_hi": None}

    # <= 0 disables the cap even when nominally "over".
    kept2, cap_stats2 = IS.cap_faint_anchors_by_mag(stamps, mag, n_bright=100, max_stars=0)
    assert kept2 == stamps
    assert cap_stats2["capped"] is False


def test_bundle_roundtrip_preserves_faint_anchor_flags(tmp_path):
    bundle = _tiny_bundle()
    bundle.is_epsf_contributor = np.array([True, False], dtype=bool)
    path = FB.save_fit_bundle(tmp_path / "fit_bundle.npz", bundle)
    loaded = FB.load_fit_bundle(path)
    assert loaded.is_epsf_contributor.dtype == np.bool_
    np.testing.assert_array_equal(loaded.is_epsf_contributor, [True, False])

    # v2 tier-native payload carries the same global flags and a redundant
    # per-tier copy, so lean packed training does not need a dense rebuild.
    tier = FB.PackedTier(
        data=np.zeros((2, bundle.n_frames, 64), dtype=np.float32),
        noise=np.ones((2, bundle.n_frames, 64), dtype=np.float32),
        weight_u8=np.ones((2, bundle.n_frames, 64), dtype=np.uint8),
        pix_x=np.zeros((2, 64), dtype=np.float32), pix_y=np.zeros((2, 64), dtype=np.float32),
        pix_valid=np.ones((2, 64), dtype=np.float32), group_idx=np.array([0, 1], dtype=np.int32),
        k_tier=2, p_tier=64, is_epsf_contributor=np.array([True, False], dtype=bool),
    )
    v2 = FB.bundle_from_tiers(bundle, [tier])
    loaded_v2 = FB.load_fit_bundle(FB.save_fit_bundle(tmp_path / "fit_bundle_v2.npz", v2))
    np.testing.assert_array_equal(loaded_v2.is_epsf_contributor, [True, False])
    np.testing.assert_array_equal(loaded_v2.packed_tiers[0].is_epsf_contributor, [True, False])
