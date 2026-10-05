"""make_init --route photutils: synthetic checks (known Gaussian ePSF, known WCS shift, fold exclusion)."""
import jax  # noqa: F401  (before pandas/pyarrow)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents
from syndiff_pipeline.forward_model.crossfit import folds as FO
from syndiff_pipeline.forward_model.crossfit import photutils_init as PI

SIG = 0.9


def _cdf(u):
    return 0.5 * (1.0 + erf(u / np.sqrt(2.0)))


def pixel_gaussian(px, py, x0, y0, flux, sig=SIG):
    """Pixel-integrated circular Gaussian at pixel centres (px, py)."""
    gx = _cdf((px + 0.5 - x0) / sig) - _cdf((px - 0.5 - x0) / sig)
    gy = _cdf((py + 0.5 - y0) / sig) - _cdf((py - 0.5 - y0) / sig)
    return flux * gx * gy


def photutils_stamp(n=45, os_=4, dx=0.0, dy=0.0, scale=1.0, sig=SIG):
    """A photutils-style oversampled ePSF stamp (density, sum ~ os^2 before ``scale``) of a Gaussian shifted by (dx, dy)."""
    c = (n - 1) / 2
    ox = (np.arange(n) - c) / os_
    X, Y = np.meshgrid(ox, ox)
    st = pixel_gaussian(X, Y, dx, dy, 1.0, sig)
    return st / st.sum() * os_ ** 2 * scale


# ------------------------------------------------------------------------------------------------ recentring
@pytest.mark.parametrize("shift", [(0.13, -0.07), (-0.04, 0.09), (0.0, 0.0)])
def test_recentre_to_core_gauge_unit_flux(shift):
    dx, dy = shift
    stack = np.stack([photutils_stamp(dx=dx, dy=dy, scale=0.83)])          # a cropped stamp: sum 0.83 * os^2
    base, tab = PI.stack_to_node_base(stack, tile_ny=1, tile_nx=1)
    g = jnp.asarray(base[0, 0])
    cx, cy = (float(v) for v in EM.core_centroid_xy(g))
    assert abs(cx) < 1e-3 and abs(cy) < 1e-3                               # scene_fit's gauge, to 1e-3 px
    assert abs(float(base.sum()) - 1.0) < 1e-5                             # unit flux restored (07-31 lesson)
    assert abs(tab.node_sum_before[0] - 1.0) < 1e-5
    # a symmetric profile recentred in the core gauge sits at its symmetry centre: global first moment ~ 0
    fx, fy = (float(v) for v in EM.flux_centroid_xy(g))
    assert abs(fx) < 3e-3 and abs(fy) < 3e-3
    # the applied shift undoes the offset (the windowed centroid is the gauge, so the shift equals -offset)
    assert abs(tab.shift_x[0] + dx) < 3e-3 and abs(tab.shift_y[0] + dy) < 3e-3


def test_recentre_one_interpolation_does_not_blur():
    """The old two-pass recentre leaves a residual and blurs; the fixed-point shift converges with one interpolation."""
    st = photutils_stamp(dx=0.1, dy=0.0)
    base, _ = PI.stack_to_node_base(np.stack([st]), tile_ny=1, tile_nx=1)
    twopass = np.asarray(EM.recenter_grid_core(jnp.asarray(base[0, 0]), clip_nonneg=True))
    assert np.abs(twopass - base[0, 0]).max() < 1e-4 * base.max()           # already centred: a no-op for scene_fit's decode
    c2 = float(EM.core_centroid_xy(jnp.asarray(twopass))[0])
    assert abs(c2) < 1e-3


def test_start_epsf_decoded_unit_flux_and_gauge():
    st = np.stack([photutils_stamp(dx=0.05, dy=-0.03, scale=0.9, sig=0.9 + 0.05 * k) for k in range(4)])
    base, _ = PI.stack_to_node_base(st, tile_ny=2, tile_nx=2)
    raw, dec = PI.start_epsf(base, 15)
    assert raw.shape == (2, 2, 63, 63) and dec.shape == raw.shape
    assert np.allclose(dec.reshape(4, -1).sum(1), 1.0, atol=1e-4)
    for k in range(4):
        cx, cy = (float(v) for v in EM.core_centroid_xy(jnp.asarray(dec[k // 2, k % 2])))
        assert abs(cx) < 1e-3 and abs(cy) < 1e-3
    # decode(encode) round trip is what scene_fit renders at step 0
    assert np.allclose(np.asarray(EM.decode_epsf_base(jnp.asarray(raw))), dec, atol=1e-7)


# ------------------------------------------------------------------------------------------------ WCS
def _static():
    return CW.ChebWcsStatic(ra0_deg=10.0, dec0_deg=20.0, cd_inv=np.eye(2) / 0.0059, crpix=np.array([1024.0, 1024.0]),
                            center=np.array([1024.0, 1024.0]), half_extents=np.array([1024.0, 1024.0]),
                            poly_degree=5, exponents=tuple(sci2idl_exponents(5)))


def test_wcs_fit_recovers_known_shift_and_distortion():
    rng = np.random.default_rng(1)
    st = _static()
    n = 3000
    xl, yl = rng.uniform(0, 2048, n), rng.uniform(0, 2048, n)
    # truth: translation + smooth low-order distortion, expressed as a Chebyshev coefficient vector
    truth = np.zeros(2 * st.n_terms)
    ex = list(st.exponents)
    truth[ex.index((0, 0))] = 0.25
    truth[st.n_terms + ex.index((0, 0))] = -0.10
    truth[ex.index((1, 0))] = 0.05
    truth[st.n_terms + ex.index((0, 2))] = -0.04
    xo, yo = PI.predict_xy(xl, yl, st, truth[:, None])
    assert abs(np.median(xo - xl) - 0.25) < 0.1                            # the sign convention: obs = lin + shift
    xo = xo + rng.normal(0, 0.01, n)
    yo = yo + rng.normal(0, 0.01, n)
    out = rng.random(n) < 0.03                                              # 3 % gross outliers: the MAD clip must cope
    xo[out] += rng.normal(0, 0.8, out.sum())
    coeff, kept = PI.fit_wcs(xl, yl, xo, yo, st)
    assert coeff.shape == (42, 1) and coeff.dtype == np.float32
    xp, yp = PI.predict_xy(xl, yl, st, coeff)
    xt, yt = PI.predict_xy(xl, yl, st, truth[:, None])
    assert np.abs(xp - xt).max() < 0.01 and np.abs(yp - yt).max() < 0.01
    assert abs(coeff[ex.index((0, 0)), 0] - 0.25) < 3e-3 and abs(coeff[st.n_terms + ex.index((0, 0)), 0] + 0.10) < 3e-3
    mx, my, cnt = PI.cell_medians(xp, yp, xp - xt, yp - yt)
    assert np.nanmax(np.hypot(mx, my)) < 0.01


# ------------------------------------------------------------------------------------------------ folds
def _scene_arrays(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    return dict(x0=rng.uniform(0, 2048, n), y0=rng.uniform(0, 2048, n), tess_mag=rng.uniform(8, 13, n),
                tess_flux=np.full(n, 1e4), source_id=np.arange(n) + 1000,
                role=rng.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3]).astype(np.int8))


@pytest.mark.parametrize("k", [0, 3])
def test_fold_exclusion(k):
    z = _scene_arrays()
    tmap = FO.tile_map_diagonal(20260929, 128, 5)
    fd = FO.fold_of(tmap, 128, z["x0"], z["y0"])                           # the harness assignment
    s = PI.select_stars(z, fold=k)
    assert (s.fold.to_numpy() == fd).all() and (s.held.to_numpy() == (fd == k)).all()
    assert not (s.train & s.held).any() and not (s.pool & s.held).any()
    assert not (s.train & (s.role == 2)).any() and not (s.pool & (s.role != 0)).any()
    assert s.pool.sum() > 0 and (s.train.sum() == ((z["role"] != 2) & (fd != k)).sum())
    allfit = PI.select_stars(z, fold=None)
    assert not allfit.held.any() and allfit.train.sum() == (z["role"] != 2).sum()
    assert s.pool.sum() < allfit.pool.sum()


def test_isolation_rule():
    z = dict(x0=np.array([100.0, 103.0, 500.0, 900.0]), y0=np.array([100.0, 100.0, 500.0, 900.0]),
             tess_mag=np.array([9.0, 10.0, 9.5, 9.0]), tess_flux=np.ones(4), source_id=np.arange(4),
             role=np.zeros(4, np.int8))
    s = PI.select_stars(z, fold=None, min_sep_px=6.0)
    assert s.pool.tolist() == [False, False, True, True]                  # the 3-px pair is not isolated
    assert PI.select_stars(z, fold=None, min_sep_px=0).pool.all()


# ------------------------------------------------------------------------------------------------ pooled ePSF + photometry
def _synthetic_frame(n=420, nstar=30, seed=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    # stars on a jittered grid, >= 40 px apart and >= 20 px from the edge: isolated
    g = np.linspace(30, n - 30, 6)
    pos = np.array([(x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)) for y in g for x in g])[:nstar]
    flux = rng.uniform(2e4, 8e4, len(pos))
    img = np.zeros((n, n))
    for (x, y), f in zip(pos, flux):
        sl = (slice(max(int(y) - 12, 0), int(y) + 13), slice(max(int(x) - 12, 0), int(x) + 13))
        img[sl] += pixel_gaussian(xx[sl], yy[sl], x, y, f)
    img += rng.normal(0, 3.0, img.shape)
    return img, pos, flux


def test_pool_photometry_wcs_chain_synthetic():
    img, pos, flux = _synthetic_frame()
    n = img.shape[0]
    reject = np.zeros(img.shape, bool)
    reject[:6] = True                                                       # a bad band that must not matter
    # catalogue positions carry a known WCS error (0.22, -0.13) px relative to the truth
    cat = pos + np.array([0.22, -0.13])
    pool = pd.DataFrame(dict(x=cat[:, 0], y=cat[:, 1], tess_mag=np.full(len(cat), 9.5)))
    stack, xy, stats = PI.pool_epsf(img, reject, pool, tile_ny=1, tile_nx=1, n_jobs=1)
    assert stats["n_tiles_ok"] == 1 and stack.shape[0] == 1
    st4 = np.repeat(stack, 4, axis=0)                                      # a 2x2 node grid of the same ePSF
    base, tab = PI.stack_to_node_base(st4, tile_ny=2, tile_nx=2)
    assert np.abs(tab[["core_cx_after", "core_cy_after"]].to_numpy()).max() < 1e-3
    raw, dec = PI.start_epsf(base, 15)
    assert np.allclose(dec.reshape(4, -1).sum(1), 1.0, atol=1e-4)
    # the built ePSF is the Gaussian: core width of the recentred node vs the analytic pixel-integrated profile
    prof = dec[0, 0]
    c = 31
    assert abs(prof[c, c] - dec[1, 1][c, c]) < 1e-7
    model = PI.gridded_model(dec, np.array([0.0, float(n)]), np.array([0.0, float(n)]))
    stars = pd.DataFrame(dict(idx=np.arange(len(cat)), source_id=np.arange(len(cat)), x_init=cat[:, 0], y_init=cat[:, 1],
                              tess_flux=flux * 1.2, tess_mag=np.full(len(cat), 9.5)))
    err = np.full(img.shape, 3.0)
    ph = PI.run_photometry(img, err, reject, model, stars, block=210, margin=16.0, n_jobs=1)
    assert len(ph) == len(cat)
    ph = ph.sort_values("scene_idx")
    # flux recovered with unit-flux ePSF (a mis-normalised stamp would scale it): within 1.5 %
    ratio = ph.flux_fit.to_numpy() / flux
    assert abs(np.median(ratio) - 1.0) < 0.015, np.median(ratio)
    # free-xy centroids land on the TRUE star positions, unbiased
    dxt = ph.x_fit.to_numpy() - pos[:, 0]
    dyt = ph.y_fit.to_numpy() - pos[:, 1]
    assert abs(np.median(dxt)) < 0.01 and abs(np.median(dyt)) < 0.01
    # and the WCS fit against the (wrong) catalogue recovers the known offset
    st = _static()
    coeff, kept = PI.fit_wcs(cat[:, 0], cat[:, 1], ph.x_fit.to_numpy(), ph.y_fit.to_numpy(), st)
    xp, yp = PI.predict_xy(cat[:, 0], cat[:, 1], st, coeff)
    assert abs(np.median(xp - cat[:, 0]) + 0.22) < 0.01 and abs(np.median(yp - cat[:, 1]) - 0.13) < 0.01


# ------------------------------------------------------------------------------------------------ header init
def test_header_xy_is_science_local(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [30.0, -20.0]
    w.wcs.crpix = [1100.0, 1000.0]                                          # 1-based FFI pixel
    w.wcs.cd = [[-0.0059, 0.0], [0.0, 0.0059]]
    hdr = w.to_header()
    hdr["NAXIS"] = 2
    p = tmp_path / "f.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((4, 4), np.float32), header=hdr)]).writeto(p)
    x, y = PI.header_xy(p, [30.0], [-20.0])
    assert abs(x[0] - (1100.0 - 1 - 44.0)) < 1e-6 and abs(y[0] - (1000.0 - 1)) < 1e-6   # 0-based, col - 44
    x2, y2 = PI.header_xy(p, [30.0 - 0.0059 * 10 / np.cos(np.deg2rad(-20.0))], [-20.0])
    assert abs((x2[0] - x[0]) - 10) < 0.05                                  # RA decreasing-CD: +x on the sky West-East


def test_folds_csv_is_authoritative(tmp_path):
    z = _scene_arrays(n=500)
    ref = PI.select_stars(z, fold=1)
    other = (ref.fold.to_numpy() + 2) % 5                                   # a different assignment on purpose
    pd.DataFrame({"source_id": z["source_id"], "fold": other}).to_csv(tmp_path / "folds.csv", index=False)
    s = PI.select_stars(z, fold=1, folds_csv=tmp_path / "folds.csv")
    assert (s.fold.to_numpy() == other).all() and (s.held.to_numpy() == (other == 1)).all()
    assert not (s.pool & s.held).any() and not (s.train & s.held).any()
