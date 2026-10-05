# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Correctness + numerical regression tests for ``gpu_flux_export``.

Regression coverage for three bugs found in the previous (un-jitted, plain
Python block loop) implementation of ``export_gpu_flux_solution``, all
verified against source before this rewrite:

1. Group rows were scattered under the wrong id (``zip(fds, fidx)`` instead
   of ``zip(fds, buckets)``) -- only the first group of each bucket
   survived, mislabeled with a frame index instead of its real group id.
2. Local block position was written as if it were already a global frame
   column, wrong whenever a stage's frame subsample (``frame_indices``) is
   a non-zero-offset window rather than ``[0, n_frames)``.
3. The zero-time-mean ``w_of_t`` gauge was recomputed per block instead of
   once over the whole (subsampled) orbit, silently changing the ePSF model
   whenever ``frame_block < n_frames``.

Each fixture below uses a NONZERO, frame-varying ``w_coeff`` and randomized
(non-identity) frame bases specifically so bug 3 is numerically observable
-- with an all-zero ``w_coeff`` (as in ``test_fit_bundle.py``'s
``_tiny_bundle``), ``w_of_t`` is zero regardless of gauge/blocking and the
bug would not show up in any comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import fit_bundle as FB
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents


def _static_kwargs(n_t: int, n_stars: int, n_g: int, *, seed: int = 0):
    """WCS/ePSF/params0 scaffolding with a nonzero w_coeff and randomized
    (non-identity) frame bases -- see module docstring for why."""
    degree = 2
    exps = tuple(sci2idl_exponents(degree))
    cheb = CW.ChebWcsStatic(
        ra0_deg=200.0, dec0_deg=80.0, cd_inv=np.eye(2),
        crpix=np.array([100.0, 100.0]), center=np.array([1600.0, 1600.0]),
        half_extents=np.array([256.0, 256.0]), poly_degree=degree, exponents=exps,
    )
    node_x = np.array([1500.0, 1700.0])
    node_y = np.array([1500.0, 1700.0])
    grid = EM.EpsfGridStatic(
        node_x=node_x, node_y=node_y,
        node_col_ccd=node_x + 44.0, node_row_ccd=node_y,
    )
    s = 7
    _, node, _ = EM.node_geometry(s)
    rng = np.random.default_rng(seed)
    base = rng.random((2, 2, node, node)).astype(np.float32)
    base /= base.reshape(2, 2, -1).sum(axis=-1)[..., None, None]
    modes = rng.normal(size=(1, 2, 2, node, node)).astype(np.float32)
    n_wcs, n_w = 4, 3
    params0 = {
        "wcs_coeff": np.zeros((2 * len(exps), n_wcs), dtype=np.float32),
        "epsf_base_raw": np.asarray(EM.encode_epsf_base(base)),
        "epsf_modes": np.asarray(EM.encode_epsf_modes(modes, base)),
        "w_coeff": (rng.normal(size=(1, n_w)) * 0.5).astype(np.float32),
    }
    ra = np.linspace(199.0, 201.0, n_stars)
    dec = np.linspace(79.0, 81.0, n_stars)
    x_lin, y_lin, cheb_basis = CW.star_basis(
        np.asarray(ra, dtype=np.float32), np.asarray(dec, dtype=np.float32), cheb,
    )
    rng_fb = np.random.default_rng(seed + 100)
    return dict(
        ra=ra, dec=dec,
        x_lin=np.asarray(x_lin, dtype=np.float32),
        y_lin=np.asarray(y_lin, dtype=np.float32),
        cheb_basis=np.asarray(cheb_basis, dtype=np.float32),
        cheb_static=cheb,
        epsf_grid=grid,
        wcs_frame_basis=rng_fb.normal(size=(n_t, n_wcs)).astype(np.float32),
        w_frame_basis=rng_fb.normal(size=(n_t, n_w)).astype(np.float32),
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


def _square_bundle(n_t: int) -> FB.FitBundle:
    """4 groups, single K=2 tier, square (non-packed) stamps."""
    n_g, s, k = 4, 7, 2
    n_stars = n_g
    kw = _static_kwargs(n_t, n_stars, n_g)
    members = np.array([[i, (i + 1) % n_stars] for i in range(n_g)], dtype=np.int32)
    valid = np.ones((n_g, k), dtype=bool)
    rng = np.random.default_rng(1)
    return FB.FitBundle(
        data=rng.random((n_g, n_t, s, s)).astype(np.float32),
        noise=np.ones((n_g, n_t, s, s), dtype=np.float32),
        weight=np.ones((n_g, n_t, s, s), dtype=np.float32),
        stamp_center_x=1600.0 + 10.0 * np.arange(n_g),
        stamp_center_y=1600.0 + 10.0 * np.arange(n_g),
        members=members,
        valid=valid,
        kept_star_mask=np.ones(n_stars, dtype=bool),
        max_group_size=k,
        k_tiers=(k,),
        meta={"test": True},
        **kw,
    )


def _packed_bundle(n_t: int) -> FB.FitBundle:
    """3 groups across two K-tiers (K=1 and K=2), dense packed layout --
    same construction as test_fit_bundle_ragged.py's
    ``_packed_dense_bundle_and_tiers``, with a nonzero w_coeff/randomized
    frame bases (see module docstring)."""
    n_g, n_stars, k_max, p_max = 3, 5, 2, 8
    kw = _static_kwargs(n_t, n_stars, n_g, seed=2)

    members = np.array([[0, -1], [1, 2], [3, 4]], dtype=np.int32)
    valid = np.array([[True, False], [True, True], [True, True]], dtype=bool)

    rng = np.random.default_rng(3)
    data = np.zeros((n_g, n_t, p_max), dtype=np.float32)
    noise = np.ones((n_g, n_t, p_max), dtype=np.float32)
    weight = np.zeros((n_g, n_t, p_max), dtype=np.float32)
    pix_x = np.zeros((n_g, p_max), dtype=np.float32)
    pix_y = np.zeros((n_g, p_max), dtype=np.float32)
    pix_valid = np.zeros((n_g, p_max), dtype=np.float32)

    data[0, :, :3] = rng.random((n_t, 3))
    noise[0, :, :3] = rng.random((n_t, 3)) + 0.5
    weight[0, :, :3] = 1.0
    pix_x[0, :3] = [10.0, 11.0, 12.0]
    pix_y[0, :3] = [20.0, 21.0, 22.0]
    pix_valid[0, :3] = 1.0

    data[1, :, :5] = rng.random((n_t, 5))
    noise[1, :, :5] = rng.random((n_t, 5)) + 0.5
    weight[1, :, :5] = 1.0
    pix_x[1, :5] = np.arange(5.0)
    pix_y[1, :5] = np.arange(5.0) + 100.0
    pix_valid[1, :5] = 1.0

    data[2, :, :] = rng.random((n_t, 8))
    noise[2, :, :] = rng.random((n_t, 8)) + 0.5
    weight[2, :, :] = 1.0
    pix_x[2, :] = np.arange(8.0)
    pix_y[2, :] = np.arange(8.0) + 200.0
    pix_valid[2, :] = 1.0

    return FB.FitBundle(
        data=data, noise=noise, weight=weight,
        pix_x=pix_x, pix_y=pix_y, pix_valid=pix_valid,
        p_tiers=(4, 8),
        stamp_center_x=np.array([1600.0, 1610.0, 1620.0]),
        stamp_center_y=np.array([1600.0, 1610.0, 1620.0]),
        members=members, valid=valid,
        kept_star_mask=np.ones(n_stars, dtype=bool),
        max_group_size=k_max,
        k_tiers=(1, 2),
        meta={"test": True, "packed": True},
        **kw,
    )


def _run_export(bundle, tmp_path, *, n_want: int, gpu_flux_frame_block: int, tag: str):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import train_loop as TL

    out = tmp_path / tag
    TL.run_stages_from_bundle(
        bundle,
        out_dir=out,
        stage=1,
        start_stage=1,
        steps_per_stage=[0],
        lr_per_stage=[1e-2],
        log_every=1,
        checkpoint_every=0,
        reject_every=0,
        stage1_core_stamp=0,
        n_frames_per_stage=[n_want],
        gpu_flux_frame_block=gpu_flux_frame_block,
    )
    assert (out / "flux_solved.npz").is_file()
    return np.load(out / "flux_solved.npz")


@pytest.mark.parametrize("bundle_fn,n_g", [(_square_bundle, 4), (_packed_bundle, 3)])
def test_all_groups_populated_at_correct_offset_columns(tmp_path, bundle_fn, n_g):
    """Regression test for bugs 1 (wrong group id) + 2 (local block position
    used as a global frame column): every group must end up with nonzero
    data, under its own true id, at the true (non-zero-offset) global frame
    columns -- not just the first group of each bucket, and not at columns
    ``[0, n_want)``."""
    n_t, n_want = 8, 4
    bundle = bundle_fn(n_t)
    start = max(0, (n_t - n_want) // 2)  # matches train_loop._middle_frame_index
    assert start > 0, "test requires a genuinely offset window"

    npz = _run_export(bundle, tmp_path, n_want=n_want, gpu_flux_frame_block=2, tag="offset")
    pix_sum = npz["pix_sum"]
    assert pix_sum.shape == (n_g, n_t)

    # Outside the true window: nothing should have been written.
    np.testing.assert_array_equal(pix_sum[:, :start], 0.0)
    np.testing.assert_array_equal(pix_sum[:, start + n_want:], 0.0)

    # Inside the true window: every group (not just the bucket's first) has data.
    inside = pix_sum[:, start:start + n_want]
    nonzero_groups = np.where(inside.sum(axis=1) > 0)[0]
    assert set(nonzero_groups.tolist()) == set(range(n_g)), (
        f"expected all {n_g} groups populated, got {sorted(nonzero_groups.tolist())}"
    )
    assert np.all(inside > 0), "every frame in the true window should have pix_sum > 0 for every group"


@pytest.mark.parametrize("bundle_fn", [_square_bundle, _packed_bundle])
def test_gauge_invariant_to_frame_block(tmp_path, bundle_fn):
    """Regression test for bug 3 (per-block gauge instead of whole-orbit
    gauge): with a nonzero, frame-varying w_coeff, splitting the same
    frame window into multiple blocks must not change the result -- the
    whole-orbit zero-time-mean gauge is computed once per bucket,
    independent of ``frame_block``."""
    n_t, n_want = 8, 6
    bundle = bundle_fn(n_t)

    ref = _run_export(bundle, tmp_path, n_want=n_want, gpu_flux_frame_block=n_want, tag="unblocked")
    blocked = _run_export(bundle, tmp_path, n_want=n_want, gpu_flux_frame_block=2, tag="blocked")

    for key in ("flux", "chi2_red", "pix_sum"):
        np.testing.assert_allclose(
            ref[key], blocked[key], atol=1e-4, rtol=1e-4,
            err_msg=f"{key} differs between unblocked and blocked export -- gauge is block-dependent",
        )
