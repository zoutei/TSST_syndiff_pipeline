# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Round-trip tests for tier-segmented (bundle_version=2) packed bundles.

Covers: PackedTier round-trip save/load, a hand-built version-1 bundle still
loading (no dependence on the real files under output/bundles/), uint8
weight round-tripping to identical float32 values, per-bucket arrays
reconstructing exactly what the old dense-padded path produced, and square
(non-packed) bundles being entirely unaffected by the tier-segmented code
path.
"""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import fit_bundle as FB
from syndiff_pipeline.forward_model import packed_support as PS
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents


def _static_model_kwargs(n_t: int, n_stars: int, n_g: int):
    """Shared WCS/ePSF/params0 scaffolding, independent of packed vs square."""
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
    s = 7
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
    ra = np.linspace(199.0, 201.0, n_stars)
    dec = np.linspace(79.0, 81.0, n_stars)
    x_lin, y_lin, cheb_basis = CW.star_basis(
        np.asarray(ra, dtype=np.float32),
        np.asarray(dec, dtype=np.float32),
        cheb,
    )
    return dict(
        ra=ra,
        dec=dec,
        x_lin=np.asarray(x_lin, dtype=np.float32),
        y_lin=np.asarray(y_lin, dtype=np.float32),
        cheb_basis=np.asarray(cheb_basis, dtype=np.float32),
        cheb_static=cheb,
        epsf_grid=grid,
        wcs_frame_basis=np.eye(n_t, n_wcs, dtype=np.float32),
        w_frame_basis=np.eye(n_t, n_w, dtype=np.float32),
        epsf_base=base,
        epsf_modes=modes,
        params0=params0,
        t_exp_sec=1425.6,
        stamp_physical=s,
        mask_active=np.ones((n_g, n_t), dtype=np.float32),
        stamp_snr_weight=np.ones(n_g, dtype=np.float32),
        fit_radius_stage1=np.full(n_g, 3.0, dtype=np.float32),
        fit_radius_stage23=np.full(n_g, 3.5, dtype=np.float32),
    )


def _square_bundle() -> FB.FitBundle:
    n_g, n_t, s = 2, 3, 7
    n_stars = 3
    k = 2
    kw = _static_model_kwargs(n_t, n_stars, n_g)
    kw["stamp_physical"] = s
    members = np.array([[0, 1], [2, -1]], dtype=np.int32)
    valid = np.array([[True, True], [True, False]], dtype=bool)
    rng = np.random.default_rng(1)
    return FB.FitBundle(
        data=rng.random((n_g, n_t, s, s)).astype(np.float32),
        noise=np.ones((n_g, n_t, s, s), dtype=np.float32),
        weight=np.ones((n_g, n_t, s, s), dtype=np.float32),
        stamp_center_x=np.array([1600.0, 1610.0]),
        stamp_center_y=np.array([1600.0, 1610.0]),
        members=members,
        valid=valid,
        kept_star_mask=np.ones(n_stars, dtype=bool),
        max_group_size=k,
        k_tiers=(k,),
        meta={"test": True},
        **kw,
    )


def _packed_dense_bundle_and_tiers(n_t: int = 2):
    """Hand-built G=3 packed bundle, split into a K=1/P=4 and a K=2/P=8 tier.

    Returns ``(dense_bundle, [tier_a, tier_b], p_max)`` where ``dense_bundle``
    is a legacy (bundle_version=1) FitBundle whose ``data``/``noise``/
    ``weight``/``pix_*`` are the dense, globally-padded arrays -- i.e. exactly
    what ``concat_packed_batches`` would have produced -- and ``tier_a``/
    ``tier_b`` are the equivalent tier-segmented ``PackedTier`` objects (no
    padding beyond each tier's own P).
    """
    n_g = 3
    n_stars = 5
    k_max = 2
    p_max = 8  # max(P_tier) across tiers = tier B's P
    kw = _static_model_kwargs(n_t, n_stars, n_g)

    # Global (small, un-tiered) group membership: group0 has 1 real member
    # (tier K=1), groups 1/2 have 2 real members each (tier K=2).
    members = np.array([[0, -1], [1, 2], [3, 4]], dtype=np.int32)
    valid = np.array([[True, False], [True, True], [True, True]], dtype=bool)

    rng = np.random.default_rng(2)
    data_dense = np.zeros((n_g, n_t, p_max), dtype=np.float32)
    noise_dense = np.ones((n_g, n_t, p_max), dtype=np.float32)
    weight_dense = np.zeros((n_g, n_t, p_max), dtype=np.float32)
    pix_x_dense = np.zeros((n_g, p_max), dtype=np.float32)
    pix_y_dense = np.zeros((n_g, p_max), dtype=np.float32)
    pix_valid_dense = np.zeros((n_g, p_max), dtype=np.float32)

    # group0 (tier K=1/P=4): 3 of 4 slots valid, rest is concat-style padding.
    data_dense[0, :, :3] = rng.random((n_t, 3))
    noise_dense[0, :, :3] = rng.random((n_t, 3)) + 0.5
    weight_dense[0, :, :3] = 1.0
    pix_x_dense[0, :3] = [10.0, 11.0, 12.0]
    pix_y_dense[0, :3] = [20.0, 21.0, 22.0]
    pix_valid_dense[0, :3] = 1.0

    # group1 (tier K=2/P=8): 5 of 8 slots valid.
    data_dense[1, :, :5] = rng.random((n_t, 5))
    noise_dense[1, :, :5] = rng.random((n_t, 5)) + 0.5
    weight_dense[1, :, :5] = 1.0
    pix_x_dense[1, :5] = np.arange(5.0)
    pix_y_dense[1, :5] = np.arange(5.0) + 100.0
    pix_valid_dense[1, :5] = 1.0

    # group2 (tier K=2/P=8): all 8 slots valid (no padding at all in-tier).
    data_dense[2, :, :] = rng.random((n_t, 8))
    noise_dense[2, :, :] = rng.random((n_t, 8)) + 0.5
    weight_dense[2, :, :] = 1.0
    pix_x_dense[2, :] = np.arange(8.0)
    pix_y_dense[2, :] = np.arange(8.0) + 200.0
    pix_valid_dense[2, :] = 1.0

    dense_bundle = FB.FitBundle(
        data=data_dense,
        noise=noise_dense,
        weight=weight_dense,
        pix_x=pix_x_dense,
        pix_y=pix_y_dense,
        pix_valid=pix_valid_dense,
        p_tiers=(4, 8),
        stamp_center_x=np.array([1600.0, 1610.0, 1620.0]),
        stamp_center_y=np.array([1600.0, 1610.0, 1620.0]),
        members=members,
        valid=valid,
        kept_star_mask=np.ones(n_stars, dtype=bool),
        max_group_size=k_max,
        k_tiers=(1, 2),
        meta={"test": True, "packed": True},
        **kw,
    )

    tier_a = FB.PackedTier(
        data=data_dense[[0], :, :4].copy(),
        noise=noise_dense[[0], :, :4].copy(),
        weight_u8=weight_dense[[0], :, :4].astype(np.uint8),
        pix_x=pix_x_dense[[0], :4].copy(),
        pix_y=pix_y_dense[[0], :4].copy(),
        pix_valid=pix_valid_dense[[0], :4].copy(),
        group_idx=np.array([0], dtype=np.int32),
        k_tier=1,
        p_tier=4,
    )
    tier_b = FB.PackedTier(
        data=data_dense[[1, 2], :, :8].copy(),
        noise=noise_dense[[1, 2], :, :8].copy(),
        weight_u8=weight_dense[[1, 2], :, :8].astype(np.uint8),
        pix_x=pix_x_dense[[1, 2], :8].copy(),
        pix_y=pix_y_dense[[1, 2], :8].copy(),
        pix_valid=pix_valid_dense[[1, 2], :8].copy(),
        group_idx=np.array([1, 2], dtype=np.int32),
        k_tier=2,
        p_tier=8,
    )
    return dense_bundle, [tier_a, tier_b], p_max


# --------------------------------------------------------------------------


def test_square_bundle_unaffected():
    bundle = _square_bundle()
    assert bundle.is_packed is False
    assert bundle.pix_x is None
    assert bundle.packed_tiers is None


def test_square_bundle_roundtrip(tmp_path):
    bundle = _square_bundle()
    path = FB.save_fit_bundle(tmp_path / "fit_bundle.npz", bundle)
    raw = np.load(path)
    assert int(np.asarray(raw["bundle_version"])) == FB.BUNDLE_VERSION
    loaded = FB.load_fit_bundle(path)
    assert loaded.is_packed is False
    assert loaded.packed_tiers is None
    np.testing.assert_allclose(loaded.data, bundle.data)
    assert loaded.n_groups == bundle.n_groups
    assert loaded.n_frames == bundle.n_frames


def test_v1_dense_packed_bundle_still_loads(tmp_path):
    """A hand-built, self-contained version-1 packed bundle -- no dependence
    on the real bundles under output/bundles/ -- still round-trips exactly.
    """
    dense_bundle, _tiers, _p_max = _packed_dense_bundle_and_tiers()
    path = FB.save_fit_bundle(tmp_path / "fit_bundle.npz", dense_bundle)
    raw = np.load(path)
    assert int(np.asarray(raw["bundle_version"])) == FB.BUNDLE_VERSION
    loaded = FB.load_fit_bundle(path)
    assert loaded.is_packed is True
    assert loaded.packed_tiers is None
    np.testing.assert_array_equal(loaded.data, dense_bundle.data)
    np.testing.assert_array_equal(loaded.noise, dense_bundle.noise)
    np.testing.assert_array_equal(loaded.weight, dense_bundle.weight)
    np.testing.assert_array_equal(loaded.pix_x, dense_bundle.pix_x)
    np.testing.assert_array_equal(loaded.pix_y, dense_bundle.pix_y)
    np.testing.assert_array_equal(loaded.pix_valid, dense_bundle.pix_valid)
    assert loaded.n_groups == 3
    assert loaded.n_frames == dense_bundle.n_frames


def test_tiered_bundle_dense_view_matches_legacy_padding():
    """bundle_from_tiers' lazy dense reconstruction must be byte-identical
    to the hand-built dense (legacy concat_packed_batches-style) arrays.
    """
    dense_bundle, tiers, _p_max = _packed_dense_bundle_and_tiers()
    tiered = FB.bundle_from_tiers(dense_bundle, tiers)
    assert tiered.packed_tiers is not None
    assert tiered.is_packed is True
    np.testing.assert_array_equal(tiered.data, dense_bundle.data)
    np.testing.assert_array_equal(tiered.noise, dense_bundle.noise)
    np.testing.assert_array_equal(tiered.weight, dense_bundle.weight)
    np.testing.assert_array_equal(tiered.pix_x, dense_bundle.pix_x)
    np.testing.assert_array_equal(tiered.pix_y, dense_bundle.pix_y)
    np.testing.assert_array_equal(tiered.pix_valid, dense_bundle.pix_valid)
    assert tiered.n_groups == dense_bundle.n_groups
    assert tiered.n_frames == dense_bundle.n_frames


def test_tiered_bundle_roundtrip_save_load(tmp_path):
    dense_bundle, tiers, _p_max = _packed_dense_bundle_and_tiers()
    tiered = FB.bundle_from_tiers(dense_bundle, tiers)
    path = FB.save_fit_bundle(tmp_path / "fit_bundle.npz", tiered)
    raw = np.load(path)
    assert int(np.asarray(raw["bundle_version"])) == FB.BUNDLE_VERSION_PACKED_TIERS
    assert raw["pt0_weight_u8"].dtype == np.uint8
    assert raw["pt1_weight_u8"].dtype == np.uint8

    loaded = FB.load_fit_bundle(path)
    assert loaded.packed_tiers is not None
    assert len(loaded.packed_tiers) == 2
    assert loaded.is_packed is True
    assert loaded.n_groups == dense_bundle.n_groups
    assert loaded.n_frames == dense_bundle.n_frames

    # Dense reconstruction from the loaded (from-disk) tiers must still
    # match the original hand-built dense arrays exactly.
    np.testing.assert_array_equal(loaded.data, dense_bundle.data)
    np.testing.assert_array_equal(loaded.noise, dense_bundle.noise)
    np.testing.assert_array_equal(loaded.weight, dense_bundle.weight)
    np.testing.assert_array_equal(loaded.pix_x, dense_bundle.pix_x)
    np.testing.assert_array_equal(loaded.pix_y, dense_bundle.pix_y)
    np.testing.assert_array_equal(loaded.pix_valid, dense_bundle.pix_valid)


def test_uint8_weight_roundtrips_to_identical_float32():
    dense_bundle, tiers, _p_max = _packed_dense_bundle_and_tiers()
    for tier, gi in zip(tiers, (np.array([0]), np.array([1, 2]))):
        assert tier.weight_u8.dtype == np.uint8
        assert set(np.unique(tier.weight_u8).tolist()) <= {0, 1}
        expected = dense_bundle.weight[gi][:, :, : tier.p_tier]
        np.testing.assert_array_equal(tier.weight_f32, expected)
        assert tier.weight_f32.dtype == np.float32


def test_packed_bucket_plan_matches_legacy_bucketing():
    """packed_bucket_plan() (tier-native) must partition groups identically
    to packed_support.bucket_packed_by_kp() run on the equivalent dense
    (legacy, byte-identical) pix_valid/groups -- i.e. per-bucket arrays
    reconstruct exactly what the old dense-padded path produced.
    """
    dense_bundle, tiers, _p_max = _packed_dense_bundle_and_tiers()
    tiered = FB.bundle_from_tiers(dense_bundle, tiers)

    native_plan = tiered.packed_bucket_plan()
    legacy_plan = PS.bucket_packed_by_kp(
        dense_bundle.group_set(),
        dense_bundle.stamp_center_x,
        dense_bundle.stamp_center_y,
        dense_bundle.pix_valid,
        k_tiers=(1, 2),
        p_tiers=(4, 8),
    )

    assert len(native_plan) == len(legacy_plan) == 2
    for (bg_n, cx_n, cy_n, gi_n, kt_n, pt_n), (bg_l, cx_l, cy_l, gi_l, kt_l, pt_l) in zip(
        native_plan, legacy_plan,
    ):
        assert kt_n == kt_l
        assert pt_n == pt_l
        np.testing.assert_array_equal(gi_n, gi_l)
        np.testing.assert_array_equal(cx_n, cx_l)
        np.testing.assert_array_equal(cy_n, cy_l)
        np.testing.assert_array_equal(bg_n.members, bg_l.members)
        np.testing.assert_array_equal(bg_n.valid, bg_l.valid)
        np.testing.assert_array_equal(bg_n.kept_star_mask, bg_l.kept_star_mask)


def test_packed_tiers_ordered_with_packed_bucket_plan():
    dense_bundle, tiers, _p_max = _packed_dense_bundle_and_tiers()
    tiered = FB.bundle_from_tiers(dense_bundle, tiers)
    plan = tiered.packed_bucket_plan()
    assert len(plan) == len(tiers)
    for (bg, _cx, _cy, gi, kt, pt), tier in zip(plan, tiers):
        assert kt == tier.k_tier
        assert pt == tier.p_tier
        np.testing.assert_array_equal(gi, tier.group_idx)
        # The dense reconstruction sliced back down to this tier's own P
        # must equal the tier's own (unpadded) stamp arrays exactly.
        np.testing.assert_array_equal(
            tiered.data[gi][:, :, :pt], tier.data,
        )
        np.testing.assert_array_equal(
            tiered.noise[gi][:, :, :pt], tier.noise,
        )
        np.testing.assert_array_equal(
            tiered.weight[gi][:, :, :pt], tier.weight_f32,
        )


def test_train_loop_runs_from_tiered_bundle(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import train_loop as TL

    dense_bundle, tiers, _p_max = _packed_dense_bundle_and_tiers()
    tiered = FB.bundle_from_tiers(dense_bundle, tiers)
    path = FB.save_fit_bundle(tmp_path / "fit_bundle.npz", tiered)
    loaded = FB.load_fit_bundle(path)
    out = tmp_path / "out"
    TL.run_stages_from_bundle(
        loaded,
        out_dir=out,
        stage=1,
        start_stage=1,
        steps_per_stage=[1, 0, 0],
        lr_per_stage=[1e-2, 3e-4, 1e-4],
        log_every=1,
        checkpoint_every=0,
        reject_every=0,
        stage1_core_stamp=0,
        no_stage1_dx_only=True,
    )
    assert (out / "params_stage1.npz").exists()
    assert (out / "fit_meta.json").exists()
