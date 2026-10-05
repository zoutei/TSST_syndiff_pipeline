# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""--fine-nbr-sigma: the fine-part neighbour coupling as a Gaussian prior normalised
by the POOLED data-term denominator, on every code path (single context, bucketed
_combined_loss_fn, frame-block accumulation)."""

import numpy as np
import jax.numpy as jnp
import optax
import pytest

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import fit as FIT
from syndiff_pipeline.forward_model import groups as GR
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import temporal as T
from syndiff_pipeline.forward_model.groups import GroupSet
from syndiff_pipeline.forward_model._bootstrap import _EXTRA_PATHS  # noqa: F401 (wires sys.path)
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents  # noqa: E402

SIGMA = 3e-4
N_FRAMES, N_BASIS = 4, 4


def _setup(sigma=SIGMA, power=1.0):
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
    G = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:G, 0:G]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob /= blob.sum()
    base = np.broadcast_to(blob, (2, 2, G, G)).copy()
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.zeros((1, 2, 2, G, G), jnp.float32))

    sizes = [1, 2, 3]
    n_stars = sum(sizes)
    members = np.full((3, 4), -1, dtype=int)
    valid = np.zeros((3, 4), dtype=bool)
    star = 0
    for gi, s in enumerate(sizes):
        members[gi, :s] = np.arange(star, star + s)
        valid[gi, :s] = True
        star += s
    orig = GroupSet(3, 4, members, valid, np.ones(n_stars, dtype=bool), 0)
    cx = np.array([50, 50, 50], dtype=np.int64)
    cy = np.array([50, 50, 50], dtype=np.int64)
    ra = np.full(n_stars, 180.0, dtype=np.float32)
    dec = np.full(n_stars, 0.0, dtype=np.float32)
    fb = np.zeros((N_FRAMES, N_BASIS), dtype=np.float32)
    fb[:, 0] = 1.0

    rng = np.random.default_rng(3)
    snr = np.ones(3, dtype=np.float32)
    radius = np.array([2.0, 3.0, 5.0], dtype=np.float32)  # unequal pixel counts
    S = EM.STAMP_PHYSICAL
    data = rng.normal(scale=5.0, size=(3, N_FRAMES, S, S)).astype(np.float32)
    ones = np.ones((3, N_FRAMES, S, S), dtype=np.float32)
    lw = L.LossWeights(support_size_weight_power=power, fine_nbr_sigma=sigma)

    params = L.init_params(static, epsf0, n_wcs_basis=N_BASIS, n_w_basis=N_BASIS)
    # Nodes must differ at the pixel scale or the prior is identically zero.
    params["epsf_base_raw"] = params["epsf_base_raw"] + jnp.asarray(
        rng.normal(scale=0.3, size=params["epsf_base_raw"].shape).astype(np.float32))

    def build(groups, bcx, bcy, idx):
        ctx = L.build_static_context(
            cheb_static=static, wcs_frame_basis=fb, w_frame_basis=fb, epsf_grid=grid,
            groups=groups, ra=ra, dec=dec, stamp_center_x=bcx, stamp_center_y=bcy,
            t_exp_sec=1426.0, stamp_snr_weight=snr[idx], fit_radius=radius[idx],
        )
        return FIT.FitData(
            ctx=ctx, data=jnp.asarray(data[idx]), noise=jnp.asarray(ones[idx]),
            weight=jnp.asarray(ones[idx]), wcs_second_diff=T.second_difference_matrix(N_BASIS),
            w_second_diff=T.second_difference_matrix(N_BASIS), epsf_modes_init=epsf0.modes,
            weights=lw,
        )

    fd_all = build(orig, cx, cy, np.arange(3))
    buckets = [build(bg, bcx, bcy, idx)
               for bg, bcx, bcy, idx in GR.bucket_groups_by_size(orig, cx, cy, tiers=(1, 2, 4))]
    return params, fd_all, buckets, lw


def _total(params, fd, lw):
    return L.total_loss(params, fd.ctx, fd.data, fd.noise, fd.weight, fd.wcs_second_diff,
                        fd.w_second_diff, epsf_modes_init=fd.epsf_modes_init, weights=lw)


def test_single_context_prior_is_sum_over_pooled_denominator():
    params, fd, _, lw = _setup()
    loss_on, m = _total(params, fd, lw)
    loss_off, _ = _total(params, fd, L.LossWeights(support_size_weight_power=1.0))
    prior_sum = float(L.fine_nbr_prior_sum(params, SIGMA))
    assert prior_sum > 0
    expected = prior_sum / float(m["stamp_weight_sum"])
    np.testing.assert_allclose(float(loss_on) - float(loss_off), expected, rtol=1e-4)
    # lambda_eff is the fixed-lambda value that would give the same term.
    np.testing.assert_allclose(float(m["lambda_fine_nbr_eff"]) * float(m["fine_nbr"]), expected, rtol=1e-4)


def test_sigma_mode_equals_fixed_lambda_at_lambda_eff():
    params, fd, _, lw = _setup()
    loss_sigma, m = _total(params, fd, lw)
    lam = L.LossWeights(support_size_weight_power=1.0, lambda_fine_nbr=float(m["lambda_fine_nbr_eff"]))
    loss_lam, _ = _total(params, fd, lam)
    np.testing.assert_allclose(float(loss_sigma), float(loss_lam), rtol=1e-5)


def _labels():
    return FIT._leaf_labels(3, freeze_wcs=False)


def test_bucketed_combined_loss_matches_unbucketed_in_sigma_mode():
    params, fd_all, buckets, lw = _setup()
    loss_all, m_all = _total(params, fd_all, lw)
    fn = FIT._combined_loss_fn(buckets, _labels())
    sa = tuple(jnp.asarray(b.ctx.stamp_active, jnp.float32) for b in buckets)
    arr = tuple((b.data, b.noise, b.weight) for b in buckets)
    loss_b, m_b = fn(params, sa, arr)
    np.testing.assert_allclose(float(loss_b), float(loss_all), rtol=1e-4)
    np.testing.assert_allclose(float(m_b["lambda_fine_nbr_eff"]), float(m_all["lambda_fine_nbr_eff"]), rtol=1e-4)


@pytest.mark.parametrize("frame_block", [1, 2])
def test_frame_block_step_matches_single_shot_step_in_sigma_mode(frame_block):
    params, _, buckets, lw = _setup()
    tx = optax.sgd(1e-3)
    sa = tuple(jnp.asarray(b.ctx.stamp_active, jnp.float32) for b in buckets)
    arr = tuple((b.data, b.noise, b.weight) for b in buckets)
    step, _ = FIT.make_step_fn(buckets, tx, stage=3)
    p_ref, _, m_ref = step(params, tx.init(params), sa, arr)
    astep, _ = FIT.make_accum_step_fn(buckets, tx, stage=3, frame_block=frame_block)
    p_acc, _, m_acc = astep(params, tx.init(params), sa, arr)
    np.testing.assert_allclose(float(m_acc["loss"]), float(m_ref["loss"]), rtol=1e-4)
    for k in ("epsf_base_raw", "wcs_coeff"):
        np.testing.assert_allclose(np.asarray(p_acc[k]), np.asarray(p_ref[k]), rtol=1e-4, atol=1e-7)
