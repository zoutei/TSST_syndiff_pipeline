"""Stage ``kernels``: per-band ePSF kernels from the calibration fit (e2e ``kc.py``, ``k_sigma.py``, ``k01``, ``k02``, ``k03``).

Reads ``fit/`` (params.npz, fit_meta.json), ``scene_boot/`` (the scene the fit was trained on), ``mapping/`` (skycell
list for the PS1 TAN headers) and the config (colour file + linear colour map, XP-synthetic PS1 table); writes
``kernels/``:

  kin.npz / kin.json              k_sigma:  Sigma_G per node + Phase A per-node eps/tri rule (kernel inputs)
  validate.json, scene_colours.npz   k01:   grid-level PSF vs the fitter's own render; model colours
  band_epsf.npz (+ _aux.npz), K_achrom.npz   k02:  per-band ePSFs/kernels  K_b = K0 + delta_b K1,  achromatic kernel
  mixture_error.json (+ _stars.npz)  k03:   option-(a) mixture error on real T<13 stars

The render glue (``slot_ctx``, ``local_grid``, ``fourier_shift``, ``E_of``, ``moments``) is the e2e ``kc.py`` logic
unchanged.  Differences from the e2e scripts (numerics identical; products compared bitwise in
tests/forward_model_chain/test_kernels.py):

* one fit / one field, taken from the config; the dead ``a3abl`` / ``d14`` branches, the F1-only Phase A / W3 / C5
  reference comparisons and ``kernel_inputs.source = "phasea"`` are not ported (``kernels.source`` must be ``k_sigma``);
* product file names no longer carry the historic ``_a3`` label (``band_epsf.npz``, ``kin.npz`` ...), and live
  directly in the stage dir ``kernels/``;
* ``kernel_bright_q`` is ``cfg.kernels.kernel_bright_q`` (default 0: template stars are the faint limit); fits without a
  ``bright_width`` leaf ignore it (no-op), as before;
* provenance records the chain code sha instead of the e2e worktree git probe.

jax is imported first (pyarrow before jax segfaults XLA); x64 is switched on process-wide as in the e2e scripts.
"""
from __future__ import annotations

import json
import os
import time
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"
import jax  # noqa: E402  (before pandas/pyarrow)

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from syndiff_pipeline.forward_model import bright_width as BW  # noqa: E402
from syndiff_pipeline.forward_model import cheb_wcs as CW  # noqa: E402
from syndiff_pipeline.forward_model import epsf_model as EM  # noqa: E402
from syndiff_pipeline.forward_model import fit_bundle as FB  # noqa: E402
from syndiff_pipeline.forward_model import loss as L  # noqa: E402
from syndiff_pipeline.template_creation.processing import chromatic_kernels as CK  # noqa: E402

from .config import is_done, mark_done, write_provenance  # noqa: E402
from .perband.paths import BANDS, LAM, MODEL, chain_paths, opt  # noqa: E402

WB = np.array([0.238, 0.344, 0.283, 0.135])   # colour-definition band weights (u = sum_b s_b lambda_b): fixed by the fit's colour file
OS = 4
MAGS = [f"Panstarrs1Std_mag_{x}p" for x in "rizy"]
NK = 127                                      # kernel size [OS4 subcells]


# ------------------------------------------------------------------ model loading (kc.py)
def kernel_dir(cfg) -> Path:
    p = chain_paths(cfg).kernels
    p.mkdir(parents=True, exist_ok=True)
    return p


def _colour_map(cfg):
    """(a, b, source) of the linear colour map c = a + b u_nm: explicit ``{a, b}`` or ``{summary_json, key}``."""
    cm = opt(cfg, "inputs.colour_map", None)
    if cm is None:
        raise KeyError("config field 'inputs.colour_map' ({a, b} or {summary_json, key}) is required by the kernels stage")
    if "a" in cm and "b" in cm:
        return float(cm["a"]), float(cm["b"]), "config"
    sf = Path(cm["summary_json"])
    lm = json.loads(sf.read_text())[cm["key"]]["linear_map"]
    return float(lm["a"]), float(lm["b"]), str(sf)


def bright_width_info(params, meta, kernel_q):
    """Brightness-width model of the fit + what the kernel chain applies (None when the leaf is absent).

    The fitter (forward_model.bright_width.slot_weight) adds to the gauge's blur field, per rendered star s,
        w_s = GEN_PER_DSIGMA * b / Q_UNIT * (q_s - q_ref),   b = leaf * LEAF_UNIT   [px^2 per 1e4 e-],
    with q_s the star's peak charge per 2-s read and q_ref the loss-weighted mean q of the ePSF contributors
    (fit_meta bright_width_model.q_ref, also stored in <fit>/bright_width_q.npz).  The template stars are the
    faint limit, so the kernel chain renders with q_s = kernel_q (default 0) for every node/colour:
        w_kernel = GEN_PER_DSIGMA * b / Q_UNIT * (kernel_q - q_ref)   (< 0: sharper than the q_ref ePSF).
    """
    if not BW.has_bright_width(params):
        return None
    form = meta.get("bright_width")
    assert form == "lin", f"params carry a bright_width leaf but fit_meta bright_width={form!r} (need 'lin')"
    bwm = meta.get("bright_width_model") or {}
    q_ref = bwm.get("q_ref")
    qf = Path(meta.get("bright_width_q_file") or bwm.get("q_file") or "")
    if q_ref is None and qf.is_file():
        q_ref = float(np.load(qf)["q_ref"])
    if q_ref is None:
        raise ValueError("bright_width leaf present but fit_meta has no q_ref (bright_width_model.q_ref)")
    q_ref = float(q_ref)
    if qf.is_file():
        assert abs(float(np.load(qf)["q_ref"]) - q_ref) < 1e-6 * max(abs(q_ref), 1.0), (qf, q_ref)
    assert abs(float(meta.get("bright_width_gen_per_dsigma", BW.GEN_PER_DSIGMA)) - BW.GEN_PER_DSIGMA) < 1e-12
    b = BW.leaf_to_b(params["bright_width"])
    return dict(form=form, b_px2_per_1e4e=b, q_ref=q_ref, kernel_q=float(kernel_q),
                blur_weight=BW.GEN_PER_DSIGMA * b / BW.Q_UNIT * (float(kernel_q) - q_ref),
                delta_sigma_per_axis_px2=b / BW.Q_UNIT * (float(kernel_q) - q_ref),
                generator=BW.generator_for_gauge(meta["chroma_g8_gauge"]))


def load_model(cfg):
    """Params (float64 jnp), fit_meta, decoded base, colour definition constants, brightness-width info."""
    P = chain_paths(cfg)
    md = P.fit_dir
    z = np.load(md / "params.npz")
    params = {k: jnp.asarray(np.asarray(z[k], np.float64)) for k in z.files if k != "epsf_repr"}
    meta = json.loads((md / "fit_meta.json").read_text())
    assert not int(meta.get("epsf_nodes") or 0), "fits with --epsf-nodes (coarser ePSF grid) are not supported here"
    # process-wide model switch: AK-hard decode of the ePSF, exactly as the fit used it (absent in old fits -> off)
    L.set_local_poly_hard(int(meta.get("local_poly_hard") or 0))
    extras = tuple(meta["chroma_g8_extras"].split(",")) if meta.get("chroma_g8_extras") else ()
    g8_trained = np.asarray(params["chroma_g8"]).copy()
    base = np.asarray(L.decoded_epsf_base(params), np.float64)
    cref = float(meta["colour_ref"])
    kq = float(cfg.kernels.kernel_bright_q)
    A = dict(model=MODEL, model_dir=md, params=params, meta=meta, base=base,
             stored_base=np.asarray(z["epsf_base"], np.float64), cref=cref, extras=extras,
             axis=tuple(meta["chroma_axis"]), d2mean=float(meta.get("chroma_delta2_mean", 0.0)),
             gauge=meta["chroma_g8_gauge"], g8_trained=g8_trained,
             local_poly_hard=int(meta.get("local_poly_hard") or 0), kernel_q=kq,
             bw=bright_width_info(params, meta, kq))
    a, bcoef, src = _colour_map(cfg)
    A.update(colour_file=P.colour_file, colour_summary=src, a=a, b=bcoef)
    A["u_ref"] = (cref - A["a"]) / A["b"]
    A["dudc"] = 1.0 / A["b"]
    assert Path(meta["colour_file"]) == A["colour_file"], meta["colour_file"]
    return A


def load_bundle(cfg):
    meta = json.loads((chain_paths(cfg).scene_dir / "scene_meta.json").read_text())
    return FB.load_fit_bundle(Path(meta["source_bundle"]))


# ------------------------------------------------------------------ render glue (kc.py, unchanged)
def slot_ctx(A, x, y, delta, node_x, node_y):
    """Minimal context carrying exactly the fields chroma_slot_terms / _render_occ_templates read.

    bright_q / bright_q_ref: per-star brightness the brightness-width term is evaluated at (the kernel chain
    always renders the TEMPLATE-star limit: q = cfg kernels.kernel_bright_q for every star). Only read when the fit
    has a bright_width leaf; without the leaf they are ignored and the render is unchanged."""
    x = jnp.atleast_1d(jnp.asarray(x, jnp.float64))
    y = jnp.atleast_1d(jnp.asarray(y, jnp.float64))
    bw = A.get("bw")
    return types.SimpleNamespace(
        bright_q=jnp.full(x.shape, A["kernel_q"], jnp.float64),
        bright_q_ref=(bw["q_ref"] if bw else 0.0),
        chroma_delta=jnp.atleast_1d(jnp.asarray(delta, jnp.float64)),
        chroma_axis=A["axis"], x_lin=x, y_lin=y, chroma_g8_extras=A["extras"],
        chroma_g8_gauge=A["gauge"], chroma_g8_no_dil=False, chroma_delta2_mean=A["d2mean"],
        node_x=jnp.asarray(node_x, jnp.float64), node_y=jnp.asarray(node_y, jnp.float64))


def local_grid(A, x, y, delta, node_x, node_y, chunk=512):
    """forward_model slot terms (colour + brightness width) + unbanded fold + hot-path recenter (W3). Returns (local (n,g,g), shift_x, shift_y)."""
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))
    delta = np.broadcast_to(np.asarray(delta, float), x.shape)
    if x.size > chunk:
        parts = [local_grid(A, x[s:s + chunk], y[s:s + chunk], delta[s:s + chunk], node_x, node_y, chunk)
                 for s in range(0, x.size, chunk)]
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))
    ctx = slot_ctx(A, x, y, delta, node_x, node_y)
    n = int(ctx.x_lin.shape[0])
    occ = jnp.arange(n)
    d_occ, shx, shy, fields = L.chroma_slot_terms(A["params"], ctx, occ)
    base = jnp.asarray(A["base"])
    g = base.shape[-1]
    i0, j0, wy, wx = EM.bilinear_cell(ctx.x_lin, ctx.y_lin, ctx.node_x, ctx.node_y)
    i0, j0, wy, wx = (jnp.reshape(v, (n,)) for v in (i0, j0, wy, wx))
    local = jnp.reshape(EM.blend_field(base, i0, j0, wy, wx), (n, g, g))
    for name, coeff in fields.items():
        assert coeff.ndim == 1, name
        gen = L.CHROMA_FIELD_GENERATORS[name](base)
        local = local + coeff[:, None, None] * jnp.reshape(EM.blend_field(gen, i0, j0, wy, wx), (n, g, g))
    local = EM.recenter_grid_core(local, clip_nonneg=False, n_iter=EM.HOTPATH_RECENTER_N_ITER)
    return np.asarray(local), np.asarray(shx), np.asarray(shy)


def fourier_shift(E, sx_px, sy_px, os=OS, pad=64):
    """Shift grid content by (+sx, +sy) physical px (band-limited, zero-padded). E: (..., g, g)."""
    E = np.asarray(E)
    g = E.shape[-1]
    Pd = np.zeros(E.shape[:-2] + (g + 2 * pad, g + 2 * pad))
    Pd[..., pad:pad + g, pad:pad + g] = E
    f = np.fft.fftfreq(Pd.shape[-1])
    FX, FY = np.meshgrid(f, f)
    sx = np.asarray(sx_px, float)[..., None, None]
    sy = np.asarray(sy_px, float)[..., None, None]
    ph = np.exp(-2j * np.pi * (FX * sx * os + FY * sy * os))
    out = np.real(np.fft.ifft2(np.fft.fft2(Pd) * ph))
    return out[..., pad:pad + g, pad:pad + g]


def E_of(A, x, y, delta, node_x, node_y):
    """Star PSF grid (pixel-integrated E, OS4) for stars at (x, y) with colour offsets delta."""
    loc, sx, sy = local_grid(A, x, y, delta, node_x, node_y)
    return fourier_shift(loc, sx, sy), sx, sy


def shares_from_mags(m):
    """m: (..., 4) PS1 Std r,i,z,y mags -> template band shares."""
    f = WB * 10 ** (-0.4 * np.asarray(m, float))
    return f / f.sum(axis=-1, keepdims=True)


def model_wcs_pix(A, bundle, ra, dec):
    """Science-local pixel position under the model's own static WCS (pinned cheb_wcs, float64)."""
    coeff = jnp.asarray(np.asarray(A["params"]["wcs_coeff"], np.float64))
    st = bundle.cheb_static
    xl, yl, basis = CW.star_basis(jnp.asarray(ra, jnp.float64), jnp.asarray(dec, jnp.float64), st)
    x, y = CW.eval_all_positions(xl, yl, basis, coeff, jnp.ones((1, 1), jnp.float64), st.n_terms)
    return np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)


def moments(E, sigma_w_px=2.0, os=OS):
    """Gaussian-windowed (sigma_w physical px, centred on the grid centre) moments (mx, my, Ixx, Iyy, Ixy)."""
    g = E.shape[-1]
    c = (g - 1) / 2.0
    t = (np.arange(g) - c) / os
    X, Y = np.meshgrid(t, t)
    W = np.exp(-0.5 * (X ** 2 + Y ** 2) / sigma_w_px ** 2)
    f = (E * W).sum(axis=(-2, -1))
    mx = (E * W * X).sum(axis=(-2, -1)) / f
    my = (E * W * Y).sum(axis=(-2, -1)) / f
    Ixx = (E * W * X ** 2).sum(axis=(-2, -1)) / f - mx ** 2
    Iyy = (E * W * Y ** 2).sum(axis=(-2, -1)) / f - my ** 2
    Ixy = (E * W * X * Y).sum(axis=(-2, -1)) / f - mx * my
    return mx, my, Ixx, Iyy, Ixy


def xp_u(cfg):
    """XP-synthetic PS1 table -> (source_id, u_nm = sum_b s_b lambda_b)."""
    import pandas as pd
    xp = pd.read_csv(cfg.inputs.xp_synth, usecols=["source_id"] + MAGS)
    m = xp[MAGS].values
    ok = np.isfinite(m).all(1)
    s = shares_from_mags(m[ok])
    return pd.DataFrame(dict(source_id=xp.source_id.values[ok], u_nm=s @ np.asarray(LAM)))


def _need_xp(cfg):
    if opt(cfg, "inputs.xp_synth", None) is None:
        raise KeyError("config field 'inputs.xp_synth' (XP-synthetic PS1 r,i,z,y table) is required by the kernels stage")


# ------------------------------------------------------------------ k_sigma: kernel inputs
PS1_PSF_SIGMA_PS1PX = 40.0
HK = ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2",
      "PC1_1", "PC1_2", "PC2_1", "PC2_2", "NAXIS1", "NAXIS2"]


def make_pix(coeff, st):
    coeff = jnp.asarray(np.asarray(coeff, np.float64))

    def pix(ra, dec):
        xl, yl, basis = CW.star_basis(jnp.atleast_1d(jnp.asarray(ra, jnp.float64)),
                                      jnp.atleast_1d(jnp.asarray(dec, jnp.float64)), st)
        x, y = CW.eval_all_positions(xl, yl, basis, coeff, jnp.ones((1, 1), jnp.float64), st.n_terms)
        return np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    return pix


def jac(pix, ra, dec, step_arcsec=2.0):
    """d(x,y)/d(xi,eta) [px/arcsec] (Phase A scene_jacobian)."""
    h = step_arcsec / 3600.0
    cosd = np.cos(np.deg2rad(dec))
    xp, yp = pix(ra + h / cosd, dec)
    xm, ym = pix(ra - h / cosd, dec)
    xn, yn = pix(ra, dec + h)
    xs, ys = pix(ra, dec - h)
    J = np.empty((ra.size, 2, 2))
    J[:, 0, 0] = (xp - xm) / (2 * step_arcsec)
    J[:, 1, 0] = (yp - ym) / (2 * step_arcsec)
    J[:, 0, 1] = (xn - xs) / (2 * step_arcsec)
    J[:, 1, 1] = (yn - ys) / (2 * step_arcsec)
    return J


def node_sky(pix, bundle, nx, ny):
    """Invert the model WCS at each node (Newton, Phase A node_sky_positions); start at the nearest star."""
    X, Y = np.meshgrid(nx, ny)
    ra_s, dec_s = np.asarray(bundle.ra, float), np.asarray(bundle.dec, float)
    xs, ys = pix(ra_s, dec_s)
    k = np.argmin((xs[None] - X.ravel()[:, None]) ** 2 + (ys[None] - Y.ravel()[:, None]) ** 2, axis=1)
    ra, dec = ra_s[k].copy(), dec_s[k].copy()
    for _ in range(8):
        x, y = pix(ra, dec)
        J = jac(pix, ra, dec)
        r = np.stack([X.ravel() - x, Y.ravel() - y], -1)
        d = np.linalg.solve(J, r[..., None])[..., 0]
        dec = dec + d[:, 1] / 3600.0
        ra = ra + d[:, 0] / 3600.0 / np.cos(np.deg2rad(dec))
    x, y = pix(ra, dec)
    err = float(np.hypot(x - X.ravel(), y - Y.ravel()).max())
    assert err < 1e-6, err
    return ra.reshape(Y.shape), dec.reshape(Y.shape), err


def ps1_jacobian(w, ra, dec, step_arcsec=2.0):
    cosd = np.cos(np.deg2rad(dec))
    s = step_arcsec / 3600.0
    pts = np.array([[ra + s / cosd, dec], [ra - s / cosd, dec], [ra, dec + s], [ra, dec - s]])
    uv = w.all_world2pix(pts, 0)
    J = np.empty((2, 2))
    J[:, 0] = (uv[0] - uv[1]) / (2 * step_arcsec)
    J[:, 1] = (uv[2] - uv[3]) / (2 * step_arcsec)
    return J


def skycell_headers(csv_path, ra0, dec0, radius_deg=None, names=None):
    import pandas as pd
    from astropy.io import fits
    df = pd.read_csv(csv_path, usecols=["NAME"] + HK + (["RA", "DEC"] if radius_deg else []))
    if names is not None:
        df = df[df.NAME.isin(names)]
    if radius_deg:
        cosd = np.cos(np.deg2rad(dec0))
        df = df[np.hypot((((df.RA - ra0 + 180) % 360) - 180) * cosd, df.DEC - dec0) < radius_deg]
    heads = {}
    for _, r in df.iterrows():
        h = fits.Header()
        for k in HK:
            h[k] = r[k]
        heads[r["NAME"]] = h
    return heads


def compute_sigma(pix, bundle, nx, ny, heads):
    """Sigma_tess = 40^2 M M^T, M = J_tess J_ps1^{-1}, at each node; median over containing skycells (Phase A rule)."""
    from astropy.wcs import WCS
    ra, dec, inv_err = node_sky(pix, bundle, nx, ny)
    nr, nc = ra.shape
    Jt = jac(pix, ra.ravel(), dec.ravel()).reshape(nr, nc, 2, 2)
    wcs = {k: WCS(h) for k, h in heads.items()}
    sig = np.empty((nr, nc, 2, 2))
    ncells = np.zeros((nr, nc), int)
    spread = np.zeros((nr, nc))
    fallback = np.zeros((nr, nc), bool)
    used = {}
    for i in range(nr):
        for j in range(nc):
            sigs, names = [], []
            for k, w in wcs.items():
                u, v = w.all_world2pix([[ra[i, j], dec[i, j]]], 0)[0]
                if -0.5 <= u < heads[k]["NAXIS1"] - 0.5 and -0.5 <= v < heads[k]["NAXIS2"] - 0.5:
                    Mx = Jt[i, j] @ np.linalg.inv(ps1_jacobian(w, ra[i, j], dec[i, j]))
                    sigs.append(PS1_PSF_SIGMA_PS1PX ** 2 * Mx @ Mx.T)
                    names.append(k)
            if not sigs:
                best = min(wcs, key=lambda k: np.hypot(*(np.array(wcs[k].all_world2pix([[ra[i, j], dec[i, j]]], 0)[0])
                                                        - np.array([heads[k]["NAXIS1"], heads[k]["NAXIS2"]]) / 2)))
                Mx = Jt[i, j] @ np.linalg.inv(ps1_jacobian(wcs[best], ra[i, j], dec[i, j]))
                sigs.append(PS1_PSF_SIGMA_PS1PX ** 2 * Mx @ Mx.T)
                names.append(best)
                fallback[i, j] = True
            sigs = np.array(sigs)
            ncells[i, j] = len(sigs)
            sig[i, j] = np.median(sigs, axis=0)
            s0 = np.sqrt(np.linalg.eigvalsh(sig[i, j]))
            spread[i, j] = max(abs(np.sqrt(np.linalg.eigvalsh(s)) - s0).max() for s in sigs) / s0.mean()
            used[f"{i},{j}"] = names
    return dict(sigma_G_tess=sig, ra=ra, dec=dec, J_tess=Jt, ncells=ncells, proj_spread_frac=spread,
                nearest_fallback=fallback, inv_err=inv_err), used


# eps/tri rule (verbatim Phase A code)
N_STAMP = 15
N_PHASE = 16
PHASES = (np.arange(N_PHASE) + 0.5) / N_PHASE - 0.5
GL_Q = 8
NPAD = 256
EPS_LIST = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7, 1e-8]
CE = 31


def render_grid(Erc, sx, sy, M):
    sx = np.atleast_1d(np.asarray(sx, float))
    sy = np.atleast_1d(np.asarray(sy, float))
    nx_, ny_ = np.round(sx).astype(int), np.round(sy).astype(int)
    dx, dy = sx - nx_, sy - ny_
    comp = jnp.broadcast_to(jnp.asarray(Erc), (sx.size,) + Erc.shape)
    st = np.asarray(EM.render_stamps(comp, jnp.asarray(dx), jnp.asarray(dy), n_pix=N_STAMP))
    h = (M - 1) // 2
    hs = (N_STAMP - 1) // 2
    out = np.zeros((sx.size, M, M))
    for k in range(sx.size):
        y0, x0 = h - hs + ny_[k], h - hs + nx_[k]
        out[k, y0:y0 + N_STAMP, x0:x0 + N_STAMP] = st[k]
    return out


def phase_grid():
    PX, PY = np.meshgrid(PHASES, PHASES)
    return PX.ravel(), PY.ravel()


def gint_subcells(Sigma, sx, sy, t0, Ln, M, q=GL_Q):
    h = (M - 1) // 2
    xg, wg = np.polynomial.legendre.leggauss(q)
    t = np.arange(t0, t0 + Ln)
    lo = -h - 0.5 + t / OS
    pts = lo[:, None] + (xg[None, :] + 1) / (2 * OS)
    ww = wg / (2 * OS)
    Si = np.linalg.inv(Sigma)
    norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    xx = pts.ravel() - sx
    yy = pts.ravel() - sy
    Q = Si[0, 0] * xx[None, :] ** 2 + 2 * Si[0, 1] * xx[None, :] * yy[:, None] + Si[1, 1] * yy[:, None] ** 2
    dens = norm * np.exp(-0.5 * Q)
    dens = dens.reshape(Ln, q, Ln, q) * ww[None, :, None, None] * ww[None, None, None, :]
    return dens.sum(axis=(1, 3))


class Operator:
    def __init__(self, Sigma, M, N):
        self.Sigma, self.M, self.N = np.asarray(Sigma, float), M, N
        self.c = (N - 1) // 2
        self.t0 = -self.c
        self.L = OS * M + 2 * self.c

    def A_ext(self, sx, sy):
        e = OS - 1
        T = gint_subcells(self.Sigma, sx, sy, self.t0 - e, self.L + e + 3, self.M)
        cs = np.cumsum(np.cumsum(np.pad(T, ((1, 0), (1, 0))), 0), 1)
        C = cs[OS:, OS:] - cs[:-OS, OS:] - cs[OS:, :-OS] + cs[:-OS, :-OS]
        p = np.arange(self.M)
        jj = np.arange(self.N + e)
        I = OS * p[:, None] - jj[None, :] + self.c - self.t0 + e
        A = C[I[:, None, :, None], I[None, :, None, :]]
        return A.reshape(self.M * self.M, (self.N + e) ** 2)


def normal_equations_fast(op, targets):
    N, M = op.N, op.M
    Ne = N + OS - 1
    H4 = np.zeros((N, N, N, N))
    g2 = np.zeros((N, N))
    bb = 0.0
    tg = targets.reshape(OS, OS, OS, OS, M * M)
    for ry in range(OS):
        for rx in range(OS):
            Ae = op.A_ext(PHASES[rx], PHASES[ry])
            He = (Ae.T @ Ae).reshape(Ne, Ne, Ne, Ne)
            for ay in range(OS):
                for ax in range(OS):
                    b = tg[ay, ry, ax, rx]
                    ge = (Ae.T @ b).reshape(Ne, Ne)
                    H4 += He[ay:ay + N, ax:ax + N, ay:ay + N, ax:ax + N]
                    g2 += ge[ay:ay + N, ax:ax + N]
                    bb += b @ b
    return H4.reshape(N * N, N * N), g2.ravel(), bb


def fourier_kernel(E, Sigma, eps, N=63, with_tri=True):
    O = OS
    Ep = np.zeros((NPAD, NPAD))
    Ep[:E.shape[0], :E.shape[1]] = E
    Ep = np.roll(Ep, (-CE, -CE), axis=(0, 1))
    Ef = np.fft.fft2(Ep)
    f = np.fft.fftfreq(NPAD, d=1.0 / O)
    FX, FY = np.meshgrid(f, f)
    G = np.exp(-2 * np.pi ** 2 * (Sigma[0, 0] * FX ** 2 + 2 * Sigma[0, 1] * FX * FY + Sigma[1, 1] * FY ** 2))
    H = G * np.sinc(FX) * np.sinc(FY)
    tri = (np.sinc(FX / O) * np.sinc(FY / O)) ** 2 if with_tri else 1.0
    Kf = Ef * tri * np.conj(H) / (np.abs(H) ** 2 + eps)
    k = np.real(np.fft.ifft2(Kf))
    k = np.roll(k, ((N - 1) // 2, (N - 1) // 2), axis=(0, 1))[:N, :N]
    return k / k.sum()


def eps_rule(E_nodes, sig, M=23, N=63, log=print):
    nr, nc = E_nodes.shape[:2]
    eps = np.zeros((nr, nc))
    tri = np.zeros((nr, nc), bool)
    rms_best = np.zeros((nr, nc))
    table = {}
    px, py = phase_grid()
    for i in range(nr):
        for j in range(nc):
            t = time.time()
            op = Operator(sig[i, j], M, N)
            tg = render_grid(E_nodes[i, j], px, py, M)
            H, g, bb = normal_equations_fast(op, tg)
            best, rows = None, []
            for with_tri in (True, False):
                for e in EPS_LIST:
                    k = fourier_kernel(E_nodes[i, j], sig[i, j], e, N, with_tri).ravel()
                    rms = float(np.sqrt(max(k @ H @ k - 2 * k @ g + bb, 0) / tg.size))
                    rows.append((e, with_tri, rms))
                    if best is None or rms < best[2]:
                        best = (e, with_tri, rms)
            eps[i, j], tri[i, j], rms_best[i, j] = best
            table[f"{i},{j}"] = rows
            log(f"node {i},{j}: {time.time() - t:.1f}s eps={best[0]:.0e} tri={best[1]} rms={best[2]:.3e}")
    return eps, tri, rms_best, table


# Kernel-sum gate (dev_runs/kernel_sum_gate_20261008). The eps rule scores a 63-subcell, sum-normalised kernel, so it
# cannot see light the 127-subcell "dc" product kernel loses past its box edge; at eps <= 3e-6 the Wiener ringing can
# overflow the box (perband_v3 F2 node (3,2): sum K1 = 0.0106 instead of 0).  A node whose product kernels lose more
# than TRUNC_TOL takes the next eps/tri of its own rule table (rms order) that keeps every loss within TRUNC_TOL; all
# other nodes keep the rule's eps/tri, so their kernels are unchanged.  Thresholds calibrated on perband_v3 F1/F2/S22/C1/C4.
N_FULL = 255                                  # near-full FFT plane (npad 256): reference sum for the truncation loss
TRUNC_TOL = 5e-3                              # max |sum K^NK - sum K^N_FULL| for K0, K1 and K_achrom
SUM_TOL = dict(K0=3e-3, K1=5e-3, mix=5e-3, Kach=3e-3)   # |sum K0 - 1|, |sum K1|, max_x |sum(K0 + x K1) - 1|, |sum Kach - 1|
MIX_X = (-0.4, 0.9)                           # colour coordinate x of real star mixtures (training colour range)


def truncation_loss(E, sigma, eps, with_tri, n_kernel=NK) -> float:
    """sum of the n_kernel "dc" kernel minus the sum of the same kernel on the near-full FFT plane."""
    kw = dict(with_tri=bool(with_tri), normalize="dc")
    return float(CK.fourier_kernel(E, sigma, float(eps), n_kernel=n_kernel, **kw).sum()
                 - CK.fourier_kernel(E, sigma, float(eps), n_kernel=N_FULL, **kw).sum())


def truncation_aware_eps(eps, tri, table, sig, E_sets, tol=TRUNC_TOL, log=print):
    """Per node: keep the rule's (eps, tri) unless a kernel of any grid in ``E_sets`` (each (nr, nc, g, g)) loses more
    than ``tol`` to truncation; then take the first (eps, tri) of the node's rule table, in rms order, that does not.
    -> (eps, tri, changes, max_loss) with ``max_loss`` (nr, nc) the largest |loss| at the adopted (eps, tri)."""
    eps = np.array(eps, float)
    tri = np.array(tri, bool)
    nr, nc = eps.shape
    changes, max_loss = [], np.zeros((nr, nc))

    def losses(i, j, e, t):
        return [truncation_loss(E[i, j], sig[i, j], e, t) for E in E_sets]

    for i in range(nr):
        for j in range(nc):
            ls = losses(i, j, eps[i, j], tri[i, j])
            if max(abs(v) for v in ls) <= tol:
                max_loss[i, j] = max(abs(v) for v in ls)
                continue
            for e, t, rms in sorted(table[f"{i},{j}"], key=lambda r: r[2]):
                ls2 = losses(i, j, e, t)
                if max(abs(v) for v in ls2) <= tol:
                    changes.append(dict(node=[i, j], eps_rule=float(eps[i, j]), tri_rule=bool(tri[i, j]),
                                        loss_rule=ls, eps=float(e), tri=bool(t), loss=ls2, rms=float(rms)))
                    log(f"node {i},{j}: eps {eps[i, j]:.0e} tri={bool(tri[i, j])} loses {ls} past the "
                        f"{NK}-subcell box -> eps {e:.0e} tri={bool(t)} (rms {rms:.3e}, loss {ls2})")
                    eps[i, j], tri[i, j] = e, t
                    max_loss[i, j] = max(abs(v) for v in ls2)
                    break
            else:
                raise RuntimeError(f"kernel node {i},{j}: no eps/tri in the rule table keeps the truncation loss "
                                   f"within {tol:g} (rule eps {eps[i, j]:g}: {ls})")
    return eps, tri, changes, max_loss


def kernel_sum_gate(K0, K1, Kach, max_loss) -> dict:
    """Per-node sum checks of the product kernels; raises if any exceeds SUM_TOL / TRUNC_TOL."""
    s0, s1, sa = (np.asarray(K).sum(axis=(-2, -1)) for K in (K0, K1, Kach))
    mix = np.maximum(np.abs(s0 + MIX_X[0] * s1 - 1), np.abs(s0 + MIX_X[1] * s1 - 1))
    vals = dict(K0=np.abs(s0 - 1), K1=np.abs(s1), mix=mix, Kach=np.abs(sa - 1), trunc=np.asarray(max_loss))
    tols = dict(SUM_TOL, trunc=TRUNC_TOL)
    out = dict(tolerances=tols, mix_x=list(MIX_X), max={k: float(v.max()) for k, v in vals.items()})
    bad = {k: [[int(i), int(j)] for i, j in zip(*np.where(v > tols[k]))] for k, v in vals.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise RuntimeError(f"kernel-sum gate FAIL (nodes per check): {bad}; maxima {out['max']}")
    return out


def run_k_sigma(cfg, A=None, do_eps: bool = True) -> dict:
    """Sigma_G per node from this fit's WCS + the mapping's skycell TAN headers, and the Phase A eps/tri rule on the
    model's E(c_ref).  -> ``kernels/kin.npz`` (+ ``kin.json``)."""
    P = chain_paths(cfg)
    KD = kernel_dir(cfg)
    A = A or load_model(cfg)
    b = load_bundle(cfg)
    st = b.cheb_static
    nx, ny = np.asarray(b.epsf_grid.node_x, float), np.asarray(b.epsf_grid.node_y, float)
    out = dict(field=P.field, model=MODEL, rule_sigma="Phase A distill.compute_sigma_G generalised (model WCS, 40 PS1 px)",
               rule_eps="Phase A distill_fourier.py per-node eps/tri (EPS_LIST, M=23, N=63, 16x16 phases)")
    coeff = np.asarray(A["params"]["wcs_coeff"], np.float64)
    mp = P.mapping_dir
    cands = sorted(mp.glob("*_master_skycells_list_os4.csv")) if mp.exists() else []
    if cands:
        csv, prov = cands[0], False
    else:   # provisional: the SCC's production OS4 mapping skycell list (same PS1 TAN headers)
        csv = P.scc / "mapping/oversampling_4" / f"tess_s{P.sector:04d}_{P.camera}_{P.ccd}_master_skycells_list_os4.csv"
        prov = True
    pix = make_pix(coeff, st)
    X, Y = np.meshgrid(nx, ny)
    ra_s, dec_s = np.asarray(b.ra, float), np.asarray(b.dec, float)
    ra0, dec0 = float(np.median(ra_s)), float(np.median(dec_s))
    heads = skycell_headers(csv, ra0, dec0)
    res, used = compute_sigma(pix, b, nx, ny, heads)
    out.update(skycell_csv=str(csv), provisional_csv=prov, n_headers=len(heads), inv_err=res["inv_err"],
               ncells=res["ncells"].tolist(), nearest_fallback=int(res["nearest_fallback"].sum()),
               proj_spread_frac_max=float(res["proj_spread_frac"].max()))
    ev = np.sqrt(np.linalg.eigvalsh(res["sigma_G_tess"]))
    out["sig_minor_px"] = ev[..., 0].round(5).tolist()
    out["sig_major_px"] = ev[..., 1].round(5).tolist()
    if do_eps:
        E_all, _, _ = E_of(A, X.ravel(), Y.ravel(), np.zeros(X.size), nx, ny)
        E_nodes = E_all.reshape(len(ny), len(nx), *E_all.shape[-2:])
        (KD / "logs").mkdir(parents=True, exist_ok=True)
        with open(KD / "logs" / "k_sigma.eps.log", "w") as logf:
            eps, tri, rmsb, table = eps_rule(E_nodes, res["sigma_G_tess"], log=lambda s: print(s, file=logf, flush=True))
        out["eps"] = eps.tolist()
        out["tri"] = tri.tolist()
        out["rms_best"] = rmsb.tolist()
        np.savez(KD / "kin.npz", sigma_G_tess=res["sigma_G_tess"], eps=eps, tri=tri, rms_best=rmsb,
                 node_x=nx, node_y=ny, ra=res["ra"], dec=res["dec"], J_tess=res["J_tess"], E_nodes=E_nodes,
                 table=json.dumps(table), meta=json.dumps(out))
    else:
        np.savez(KD / "kin_sigonly.npz", sigma_G_tess=res["sigma_G_tess"], node_x=nx, node_y=ny,
                 ra=res["ra"], dec=res["dec"], J_tess=res["J_tess"], meta=json.dumps(out))
    (KD / ("kin.json" if do_eps else "kin_sigonly.json")).write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ k01: validation
def run_k01(cfg, A=None) -> dict:
    """Validate the grid-level PSF against the fitter's own render (incl. the brightness-width blur at q = kernel_bright_q
    when the fit has the leaf) and recompute the model's colours with forward_model.scene_fit.Scene.
    -> ``kernels/validate.json``, ``kernels/scene_colours.npz``."""
    from syndiff_pipeline.forward_model import scene_fit as SF
    P = chain_paths(cfg)
    KD = kernel_dir(cfg)
    A = A or load_model(cfg)
    out = {"field": P.field, "model": MODEL, "model_dir": str(A["model_dir"]),
           "code": {"sha": cfg.code_sha(), "forward_model_root": str(cfg.code.forward_model_root)}, "loss_file": L.__file__,
           "local_poly_hard": A["local_poly_hard"], "bright_width": A["bw"], "kernel_bright_q": A["kernel_q"]}
    out["base_decode_vs_stored_max"] = float(np.abs(A["base"] - A["stored_base"]).max())

    sc = SF.Scene(P.scene_dir)
    out["colour_counts"] = sc.use_colour_file(A["colour_file"])
    cref = sc.colour_ref()
    d2 = sc.delta2_mean(cref)
    out.update(colour_source=sc.colour_source, colour_ref_recomputed=cref, colour_ref_fit_meta=A["cref"],
               delta2_recomputed=d2, delta2_fit_meta=A["meta"].get("chroma_delta2_mean"))
    assert abs(cref - A["cref"]) < 1e-9, (cref, A["cref"])
    if A["meta"].get("chroma_delta2_mean") is not None:
        assert abs(d2 - A["d2mean"]) < 1e-9
    c = sc.colour[sc.z["star_bundle_index"]]
    train = (sc.role != SF.ROLE_NUISANCE) & np.isfinite(c)
    np.savez(KD / "scene_colours.npz", colour=c, role=sc.role, train=train,
             star_bundle_index=sc.z["star_bundle_index"], source_id=sc.z["source_id"], tess_mag=sc.z["tess_mag"])

    b = sc.src
    nx, ny = np.asarray(b.epsf_grid.node_x, float), np.asarray(b.epsf_grid.node_y, float)
    out["node_x"], out["node_y"] = nx.tolist(), ny.tolist()
    cq = np.quantile(c[train], [0.02, 0.5, 0.98])
    rng = np.random.default_rng(0)
    nr, nc = len(ny), len(nx)
    pts = [(nx[0], ny[0]), (nx[nc // 2 - 1], ny[nr // 2]), (nx[-1], ny[-1]), (nx[0], ny[-1]), (nx[-1], ny[0])]
    pts += [tuple(rng.uniform(0, 2048, 2)) for _ in range(4)]
    res = []
    for (x, y) in pts:
        for cc in cq:
            d = cc - A["cref"]
            for _ in range(3):
                dx, dy = rng.uniform(-0.5, 0.5, 2)
                loc, sx, sy = local_grid(A, [x], [y], [d], nx, ny)
                mine = np.asarray(EM.render_stamps(jnp.asarray(loc), jnp.asarray(dx + sx), jnp.asarray(dy + sy), n_pix=15))[0]
                ctx = slot_ctx(A, [x], [y], [d], nx, ny)
                dd, chx, chy, fields = L.chroma_slot_terms(A["params"], ctx, jnp.arange(1))
                pin = {}
                for banded in (True, False):
                    L.USE_BANDED_RENDER = banded
                    st = L._render_occ_templates(
                        ctx, ctx.x_lin[:, None], ctx.y_lin[:, None],
                        jnp.asarray([[dx]]) + chx[:, None], jnp.asarray([[dy]]) + chy[:, None],
                        jnp.zeros((1, 0)), base=jnp.asarray(A["base"]), modes=jnp.zeros((0,) + A["base"].shape),
                        local_cache_arr=None, n_pix=15, do_recenter=True, recenter_n_iter=EM.HOTPATH_RECENTER_N_ITER,
                        chroma_delta_occ=dd, chroma_fields=fields)
                    pin[banded] = np.asarray(st)[0, 0]
                L.USE_BANDED_RENDER = True
                fs = fourier_shift(loc, sx, sy)
                four = np.asarray(EM.render_stamps(jnp.asarray(fs), jnp.asarray([dx]), jnp.asarray([dy]), n_pix=15))[0]
                pk = pin[True].max()
                res.append(dict(x=float(x), y=float(y), colour=float(cc), dx=float(dx), dy=float(dy),
                                shift=[float(sx[0]), float(sy[0])],
                                mine_vs_banded=float(np.abs(mine - pin[True]).max() / pk),
                                mine_vs_unbanded=float(np.abs(mine - pin[False]).max() / pk),
                                banded_vs_unbanded=float(np.abs(pin[True] - pin[False]).max() / pk),
                                fourier_vs_render=float(np.abs(four - mine).max() / pk),
                                flux_mine=float(mine.sum()), flux_pin=float(pin[True].sum())))
    out["n_cases"] = len(res)
    for k in ("mine_vs_banded", "mine_vs_unbanded", "banded_vs_unbanded", "fourier_vs_render"):
        v = np.array([r[k] for r in res])
        out[k + "_max"] = float(v.max())
        out[k + "_median"] = float(np.median(v))
    out["render_pass_1e-6"] = bool(max(out["mine_vs_banded_max"], out["mine_vs_unbanded_max"]) <= 1e-6)
    out["cases"] = res
    (KD / "validate.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ k02: band ePSFs and kernels
def band_linearisation(Eq, E0, P0, P1, T1, xq, dq):
    """Linearisation errors of the option-(a) model E_b = P0 + x P1 against the rendered colour quantiles."""
    pk = E0.max(axis=(-2, -1))
    Lmod = P0[:, :, None] + xq[None, None, :, None, None] * P1[:, :, None]
    lin = Eq - Lmod
    err_ls = np.abs(lin).max(axis=(-2, -1)) / pk[..., None]
    err_tan = np.abs(Eq - (E0[:, :, None] + dq[None, None, :, None, None] * T1[:, :, None])).max(axis=(-2, -1)) / pk[..., None]
    err_ach = np.abs(Eq - E0[:, :, None]).max(axis=(-2, -1)) / pk[..., None]
    rms_ls = np.sqrt((lin ** 2).mean(axis=(-2, -1))) / pk[..., None]
    mE = moments(Eq)
    mL = moments(Lmod)
    mA = moments(np.broadcast_to(E0[:, :, None], Eq.shape))
    dcen = np.hypot(mL[0] - mE[0], mL[1] - mE[1]) * 1e3
    dtr = ((mL[2] + mL[3]) - (mE[2] + mE[3])) * 1e3
    dcen_ach = np.hypot(mA[0] - mE[0], mA[1] - mE[1]) * 1e3
    dtr_ach = ((mA[2] + mA[3]) - (mE[2] + mE[3])) * 1e3
    return dict(err_ls=err_ls, err_tan=err_tan, err_ach=err_ach, rms_ls=rms_ls, dcen=dcen, dtr=dtr,
                dcen_ach=dcen_ach, dtr_ach=dtr_ach)


def run_k02(cfg, A=None) -> dict:
    """Per-band ePSFs at all nodes, kernels (K_b = K0 + delta_b K1) and the achromatic kernel.

    Colour coordinate: the model's colour is linear in u, c = a + b u_nm, so x = delta = c - c_ref; the 25 colour quantiles
    (2-98%) of the model colour over its training stars are rendered, a least-squares line P0 + x P1 is fitted through the
    star PSF at those colours, and delta_b = a + b lambda_b - c_ref.  Hence sum_b s_b E_b = P0 + x(u*) P1 exactly for a star
    with shares s (sum s = 1, sum s lambda = u*).  Kernel inputs: ``kin.npz`` (k_sigma).
    -> ``kernels/band_epsf.npz``, ``band_epsf_aux.npz``, ``K_achrom.npz``."""
    P = chain_paths(cfg)
    KD = kernel_dir(cfg)
    if cfg.kernels.source != "k_sigma":
        raise NotImplementedError(f"kernels.source={cfg.kernels.source!r}: only 'k_sigma' is implemented in the chain")
    _need_xp(cfg)
    t0 = time.time()
    A = A or load_model(cfg)
    b = load_bundle(cfg)
    nx, ny = np.asarray(b.epsf_grid.node_x, float), np.asarray(b.epsf_grid.node_y, float)
    base = A["base"]
    nr, nc, g, _ = base.shape
    sc = np.load(KD / "scene_colours.npz")
    qs = np.linspace(0.02, 0.98, 25)
    nq = len(qs)

    ctrain = sc["colour"][sc["train"]]
    cq = np.quantile(ctrain, qs)
    dq = cq - A["cref"]                    # colour offsets fed to the model
    xq = dq.copy()                         # linear coordinate (== delta for A3)
    delta_b = A["a"] + A["b"] * np.asarray(LAM) - A["cref"]
    assert np.allclose(delta_b, (np.asarray(LAM) - A["u_ref"]) / A["dudc"])
    u_ref, dudc = A["u_ref"], A["dudc"]
    uq = (cq - A["a"]) / A["b"]
    # colour-file check: c from XP shares == colour file for XP stars
    import pandas as pd
    u = xp_u(cfg)
    d = pd.DataFrame(dict(source_id=sc["source_id"], c=sc["colour"], tr=sc["train"])).merge(u, on="source_id", how="inner")
    cinfo = dict(colour_map_a=A["a"], colour_map_b_mag_per_nm=A["b"], n_train=int(ctrain.size),
                 n_xp_matched=int(len(d)),
                 colour_file_vs_shares_absmax=float(np.abs(d.c - (A["a"] + A["b"] * d.u_nm)).max()))
    print(cinfo)
    n_train = int(ctrain.size)

    # ---- 1. E at all nodes x quantiles (+ delta = 0)
    XX, YY = np.meshgrid(nx, ny)
    xs = np.repeat(XX.ravel(), nq + 1)
    ys = np.repeat(YY.ravel(), nq + 1)
    ds = np.tile(np.r_[0.0, dq], nr * nc)
    E_all, sx_all, sy_all = E_of(A, xs, ys, ds, nx, ny)
    E_all = E_all.reshape(nr, nc, nq + 1, g, g)
    shift = np.stack([sx_all, sy_all], -1).reshape(nr, nc, nq + 1, 2)
    E0 = E_all[:, :, 0]
    Eq = E_all[:, :, 1:]
    print(f"rendered {E_all.shape} in {time.time() - t0:.1f}s")

    # ---- 2. LS line in x (u-linear coordinate)
    Am = np.stack([np.ones(nq), xq], 1)
    coef, *_ = np.linalg.lstsq(Am, Eq.transpose(2, 0, 1, 3, 4).reshape(nq, -1), rcond=None)
    a0 = coef[0].reshape(nr, nc, g, g)
    a1 = coef[1].reshape(nr, nc, g, g)
    P0 = a0 / a0.sum(axis=(-2, -1), keepdims=True)
    P1 = a1 - a1.sum(axis=(-2, -1), keepdims=True) * P0
    lsum = dict(a0_sum_minmax=[float(a0.sum(axis=(-2, -1)).min()), float(a0.sum(axis=(-2, -1)).max())],
                a1_sum_absmax=float(np.abs(a1.sum(axis=(-2, -1))).max()))
    H = 0.02
    Ep, _, _ = E_of(A, XX.ravel(), YY.ravel(), np.full(nr * nc, H), nx, ny)
    Em, _, _ = E_of(A, XX.ravel(), YY.ravel(), np.full(nr * nc, -H), nx, ny)
    T1 = ((Ep - Em) / (2 * H)).reshape(nr, nc, g, g)
    T1 = T1 - T1.sum(axis=(-2, -1), keepdims=True) * E0
    E_bands = P0[None] + delta_b[:, None, None, None, None] * P1[None]

    # ---- 3. linearisation error
    pk = E0.max(axis=(-2, -1))
    le = band_linearisation(Eq, E0, P0, P1, T1, xq, dq)
    err_ls, err_tan, err_ach, rms_ls = le["err_ls"], le["err_tan"], le["err_ach"], le["rms_ls"]
    dcen, dtr, dcen_ach, dtr_ach = le["dcen"], le["dtr"], le["dcen_ach"], le["dtr_ach"]
    linerr = dict(
        definition="max over grid |E(c_k) - (P0 + x_k P1)| / max E(c_ref), per node, per colour quantile (2-98%, 25)",
        ls_max=float(err_ls.max()), ls_median_all=float(np.median(err_ls)),
        ls_median_over_nodes_of_node_max=float(np.median(err_ls.max(axis=2))),
        ls_rms_frac_peak_max=float(rms_ls.max()),
        ls_worst_node=[int(v) for v in np.unravel_index(err_ls.max(axis=2).argmax(), (nr, nc))],
        tangent_max=float(err_tan.max()), tangent_median_all=float(np.median(err_tan)),
        achromatic_max=float(err_ach.max()), achromatic_median_all=float(np.median(err_ach)),
        extreme_quantile_ls_max={"q02": float(err_ls[:, :, 0].max()), "q98": float(err_ls[:, :, -1].max())},
        centroid_err_mpx_max=float(dcen.max()), width_trace_err_1e3px2_max_abs=float(np.abs(dtr).max()),
        achromatic_centroid_err_mpx_max=float(dcen_ach.max()),
        achromatic_width_trace_err_1e3px2_max_abs=float(np.abs(dtr_ach).max()),
        moments_window="Gaussian window sigma = 2 px about the grid centre; width = Ixx + Iyy",
    )
    print(json.dumps(linerr, indent=1))

    # ---- 4. kernel inputs (k_sigma): sigma_G from this fit's WCS + mapping, eps/tri from the rule on E(c_ref)
    ks = np.load(KD / "kin.npz")
    sig = np.asarray(ks["sigma_G_tess"], float)
    eps = np.asarray(ks["eps"], float)
    tri = np.asarray(ks["tri"], bool)
    assert np.allclose(ks["node_x"], nx) and np.allclose(ks["node_y"], ny)
    kin_src = dict(sigma_G=str(KD / "kin.npz"), sigma_G_meta=json.loads(str(ks["meta"])), eps_tri=str(KD / "kin.npz"),
                   note="sigma_G from this model's own WCS + mapping skycell TAN headers; eps/tri = Phase A rule on E(c_ref)")
    kin_src["sigma_G_meta"] = {k: v for k, v in kin_src["sigma_G_meta"].items() if k not in ("eps", "tri", "rms_best")}
    eps, tri, eps_changes, max_loss = truncation_aware_eps(eps, tri, json.loads(str(ks["table"])), sig, (P0, P1, E0))
    Kb = CK.band_kernels(E_bands, sig, eps, tri, n_kernel=NK)
    K0 = CK.band_kernels(P0[None], sig, eps, tri, n_kernel=NK)[0]
    K1 = CK.band_kernels(P1[None], sig, eps, tri, n_kernel=NK)[0]
    Kach = CK.band_kernels(E0[None], sig, eps, tri, n_kernel=NK)[0]
    gate = dict(kernel_sum_gate(K0, K1, Kach, max_loss), eps_changes=eps_changes,
                eps_rule_source=str(KD / "kin.npz"),
                note="eps/tri here = kin.npz rule, except eps_changes (truncation-aware fallback, kernel_sum_gate_20261008)")
    print(json.dumps(dict(gate, eps_changes=len(eps_changes)), indent=1))
    lin_k = max(float(np.abs(Kb[i] - (K0 + delta_b[i] * K1)).max()) for i in range(4))
    ksum = Kb.sum(axis=(-2, -1))
    K0c = K0[:, :, 32:95, 32:95]
    kchk = dict(
        n_kernel=NK, normalize="dc", kernel_linearity_abs_max=lin_k, kernel_linearity_rel=lin_k / float(np.abs(K0).max()),
        kernel_sums_band_min=[float(ksum[i].min()) for i in range(4)], kernel_sums_band_max=[float(ksum[i].max()) for i in range(4)],
        K0_sum_minmax=[float(K0.sum(axis=(-2, -1)).min()), float(K0.sum(axis=(-2, -1)).max())],
        K1_sum_absmax=float(np.abs(K1.sum(axis=(-2, -1))).max()),
        K0_sum_inside_central63_minmax=[float(K0c.sum(axis=(-2, -1)).min()), float(K0c.sum(axis=(-2, -1)).max())],
        P0_vs_Ecref_rel_max=float((np.abs(P0 - E0).max(axis=(-2, -1)) / pk).max()),
    )
    print(json.dumps(kchk, indent=1))

    prov = {"sha": cfg.code_sha(), "forward_model_root": str(cfg.code.forward_model_root)}
    meta = dict(
        note=("Option (a): E_b = P0 + delta_b P1, least-squares line in u through the model's star PSF at 25 colour "
              "quantiles (2-98%) of its T<13 training stars."),
        field=P.field, model=MODEL, sector=P.sector, camera=P.camera, ccd=P.ccd, frame_stem=P.stem,
        model_path=str(A["model_dir"]), model_params=str(A["model_dir"] / "params.npz"),
        model_code=prov, model_code_sha=prov["sha"],
        model_trained_with=A["meta"].get("started", "see fit_meta.json"),
        local_poly_hard=A["local_poly_hard"], kernel_bright_q=A["kernel_q"], bright_width=A["bw"],
        chromatic_kernels=str(CK.__file__),
        scene_dir=str(P.scene_dir), colour_kind="u", colour_file=str(A["colour_file"]),
        colour_definition=("model colour c = a + b*u_nm (XP-synthetic PS1 Std band-mix wavelength u_nm = sum_b s_b lambda_b, "
                           "s_b = w_b F_b / sum w F); stars without XP fall back to Gaia BP-RP"),
        colour_ref=A["cref"], u_ref_nm=u_ref, du_dc_nm_per_mag=dudc, delta2_mean=A["d2mean"],
        linear_coordinate="x = (u - u_ref)/(du/dc) [BP-RP units]; x == c - c_ref exactly",
        colour_info=cinfo, xp_synth=str(cfg.inputs.xp_synth),
        delta_b=dict(zip(BANDS, delta_b.tolist())), lambda_b_nm=list(LAM), band_weights=WB.tolist(),
        chroma_g8_used=np.asarray(A["params"]["chroma_g8"]).tolist(), chroma_g8_trained=A["g8_trained"].tolist(),
        chroma_g8_extras=list(A["extras"]), chroma_axis=list(A["axis"]), chroma_g8_gauge=A["gauge"],
        colour_quantiles=cq.tolist(), x_quantiles=xq.tolist(), u_quantiles_nm=np.asarray(uq).tolist(),
        quantile_levels=qs.tolist(), n_training_stars=n_train,
        shift_representation="colour shift applied as band-limited Fourier shift of the recentred grid (W3 shift_repr.json)",
        mapping=str(P.mapping_dir), kernel_inputs=kin_src, n_kernel=NK, normalize="dc", band_order=list(BANDS),
        axes="K_bands (band, node row=y, node col=x, N, N) OS4 subcells; node_x/node_y science-local px",
        linearisation_errors=linerr, kernel_checks=kchk, kernel_sum_gate=gate, lstsq_sums=lsum,
        render_validation=str(KD / "validate.json"),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    np.savez(KD / "band_epsf.npz", K_bands=Kb, K0=K0, K1=K1, P0=P0, P1=P1, E_bands=E_bands, delta_b=delta_b,
             node_x=nx, node_y=ny, sigma_G_tess=sig, eps=eps, tri=tri, meta=json.dumps(meta))
    np.savez(KD / "band_epsf_aux.npz", E0=E0, Eq=Eq, T1=T1, shift=shift, cq=cq, dq=dq, xq=xq, err_ls=err_ls,
             err_tan=err_tan, err_ach=err_ach, rms_ls=rms_ls, dcen=dcen, dtr=dtr, dcen_ach=dcen_ach, dtr_ach=dtr_ach)
    ksa = Kach.sum(axis=(-2, -1))
    am = dict(meta, note="Achromatic control: 127-subcell kernel of the star PSF at its colour reference E(c_ref) "
                         "(delta = 0), dc normalisation, same sigma_G/eps/tri.",
              K_sum_minmax=[float(ksa.min()), float(ksa.max())],
              K_vs_K0_rel_max=float(np.abs(Kach - K0).max() / np.abs(K0).max()))
    for k in ("linearisation_errors",):
        am.pop(k, None)
    np.savez(KD / "K_achrom.npz", K=Kach, E=E0, node_x=nx, node_y=ny, sigma_G_tess=sig, eps=eps, tri=tri,
             meta=json.dumps(am))
    print("K_achrom sums", am["K_sum_minmax"], "vs K0", am["K_vs_K0_rel_max"])
    print(f"done {time.time() - t0:.1f}s")
    return dict(meta=meta, kernel_checks=kchk, linearisation_errors=linerr)


# ------------------------------------------------------------------ k03: mixture error
def run_k03(cfg, A=None) -> dict:
    """Option-(a) error for the band MIXTURE real stars get.

    Stars: the scene's T<13 training stars with XP-synthetic PS1 r,i,z,y (band shares s_b); sample 400 random with
    BP-RP <= 1.6 + 300 red (BP-RP > 1.6).  Positions: the model's own WCS.  Truth = the model's own PSF for the star,
    E(c*) with c* = a + b u*.  Per star: node = nearest node, sum_b s_b E_b[n] vs truth at the node; pos = node-blended
    mixture at the star position vs truth at the position; pos0 = achromatic node-blend floor; achro = E_n(c_ref) vs truth.
    -> ``kernels/mixture_error.json``, ``mixture_error_stars.npz``."""
    import pandas as pd
    P = chain_paths(cfg)
    KD = kernel_dir(cfg)
    _need_xp(cfg)
    A = A or load_model(cfg)
    b = load_bundle(cfg)
    z = np.load(KD / "band_epsf.npz")
    aux = np.load(KD / "band_epsf_aux.npz")
    meta = json.loads(str(z["meta"]))
    nx, ny = z["node_x"], z["node_y"]
    E_bands = z["E_bands"]
    E0n = aux["E0"]
    u_ref, dudc = meta["u_ref_nm"], meta["du_dc_nm_per_mag"]
    sc = np.load(KD / "scene_colours.npz")
    sbi = sc["star_bundle_index"]
    bprp_b = np.asarray(b.bp_rp, float)
    syn = pd.read_csv(cfg.inputs.xp_synth, usecols=["source_id"] + MAGS)
    df = pd.DataFrame(dict(source_id=sc["source_id"], colour_model=sc["colour"], bprp=bprp_b[sbi], T=sc["tess_mag"],
                           ra=np.asarray(b.ra, float)[sbi], dec=np.asarray(b.dec, float)[sbi]))[sc["train"]]
    df = df.merge(syn, on="source_id", how="inner")
    df = df[np.isfinite(df[MAGS].values).all(1) & np.isfinite(df.bprp.values)].copy()
    s = shares_from_mags(df[MAGS].values)
    df["u_nm"] = s @ np.asarray(LAM)
    df["x_star"] = (df.u_nm - u_ref) / dudc                 # the linear model's coordinate for this star
    df["c_true"] = A["a"] + A["b"] * df.u_nm
    df["c_ofu"] = df.c_true
    for i, bb in enumerate(BANDS):
        df["s_" + bb] = s[:, i]
    df["x"], df["y"] = model_wcs_pix(A, b, df.ra.values, df.dec.values)
    df = df[(df.x >= 0) & (df.x < 2048) & (df.y >= 0) & (df.y < 2048)]
    colchk = dict(n_matched=int(len(df)), model_colour_file_vs_truth_absmax=float(np.nanmax(np.abs(df.colour_model - df.c_true))))
    print(colchk)
    rng = np.random.default_rng(1)
    red = df[df.bprp > 1.6]
    rest = df[df.bprp <= 1.6]
    S = pd.concat([rest.sample(min(400, len(rest)), random_state=rng.integers(1 << 30)),
                   red.sample(min(300, len(red)), random_state=rng.integers(1 << 30))]).reset_index(drop=True)
    x, y = S.x.values, S.y.values
    n = len(S)
    dtrue = S.c_true.values - A["cref"]
    shares = S[["s_" + bb for bb in BANDS]].values
    jn = np.abs(x[:, None] - nx[None]).argmin(1)
    inn = np.abs(y[:, None] - ny[None]).argmin(1)
    E_node_true, _, _ = E_of(A, nx[jn], ny[inn], dtrue, nx, ny)
    mix_node = np.einsum("nb,bnxy->nxy", shares, E_bands[:, inn, jn])
    E_pos, _, _ = E_of(A, x, y, dtrue, nx, ny)
    E_pos0, _, _ = E_of(A, x, y, np.zeros(n), nx, ny)
    Wx = CK.hat_weights(x, nx)
    Wy = CK.hat_weights(y, ny)
    Wn = Wy.T[:, :, None] * Wx.T[:, None, :]
    mix_pos = np.einsum("nij,nb,bijxy->nxy", Wn, shares, E_bands)
    blend0 = np.einsum("nij,ijxy->nxy", Wn, E0n)
    E0_nearest = E0n[inn, jn]

    def cmp(M, R):
        pk = R.max(axis=(-2, -1))
        e = np.abs(M - R).max(axis=(-2, -1)) / pk
        mM, mR = moments(M), moments(R)
        vx, vy = A["axis"][0] - x, A["axis"][1] - y
        rr = np.hypot(vx, vy)
        ux, uy = vx / rr, vy / rr
        ddx, ddy = mM[0] - mR[0], mM[1] - mR[1]
        return dict(err=e, dcen=np.hypot(ddx, ddy) * 1e3, dcen_axis=(ddx * ux + ddy * uy) * 1e3,
                    dtr=((mM[2] + mM[3]) - (mR[2] + mR[3])) * 1e3)

    R = {"node": cmp(mix_node, E_node_true), "pos": cmp(mix_pos, E_pos), "pos0": cmp(blend0, E_pos0),
         "achro": cmp(E0_nearest, E_node_true)}
    bp = S.bprp.values
    summ = {}
    for sel_name, sel in (("all", np.ones(n, bool)), ("red_bprp_gt_1.6", bp > 1.6), ("blue_bprp_lt_0.8", bp < 0.8),
                          ("in_range", (S.c_true.values >= meta["colour_quantiles"][0]) & (S.c_true.values <= meta["colour_quantiles"][-1]))):
        if not sel.any():
            continue
        d = {"n": int(sel.sum())}
        for k, r in R.items():
            d[k] = {"err_median": float(np.median(r["err"][sel])), "err_p95": float(np.percentile(r["err"][sel], 95)),
                    "err_max": float(r["err"][sel].max()),
                    "dcen_mpx_median": float(np.median(r["dcen"][sel])), "dcen_mpx_p95": float(np.percentile(r["dcen"][sel], 95)),
                    "dcen_axis_mpx_median": float(np.median(r["dcen_axis"][sel])),
                    "dtr_1e3px2_median": float(np.median(r["dtr"][sel])), "dtr_1e3px2_p95abs": float(np.percentile(np.abs(r["dtr"][sel]), 95))}
        summ[f"T<13|{sel_name}"] = d
    out = dict(field=P.field, model=MODEL, colour_check=colchk, summary=summ,
               notes="err = max|model - truth|/peak(truth); dcen = |centroid diff| "
                     "(Gaussian window sigma 2 px); dcen_axis = component toward the optical axis; dtr = (Ixx+Iyy) model - truth")
    (KD / "mixture_error.json").write_text(json.dumps(out, indent=1))
    np.savez(KD / "mixture_error_stars.npz", x=x, y=y, bprp=bp, c_true=S.c_true.values, c_ofu=S.c_ofu.values,
             u_nm=S.u_nm.values, T=S["T"].values, shares=shares,
             **{f"{k}_{q}": v[q] for k, v in R.items() for q in ("err", "dcen", "dcen_axis", "dtr")})
    return out


# ------------------------------------------------------------------ stage driver
def run(cfg, force: bool = False) -> Path:
    """Stage ``kernels``: needs ``fit``, ``scene_boot`` and ``mapping`` done.  k_sigma -> k01 -> k02 -> k03."""
    stage = cfg.stage_dir("kernels")
    if is_done(stage) and not force:
        return stage
    for dep in ("fit", "scene_boot", "mapping"):
        if not is_done(cfg.stage_dir(dep)):
            raise FileNotFoundError(f"stage {dep} not done: {cfg.stage_dir(dep)}")
    A = load_model(cfg)
    kernel_dir(cfg)
    run_k_sigma(cfg, A)
    run_k01(cfg, A)
    run_k02(cfg, A)
    run_k03(cfg, A)
    P = chain_paths(cfg)
    write_provenance(stage, cfg, {"fit_params": P.fit_dir / "params.npz", "fit_meta": P.fit_dir / "fit_meta.json",
                                  "scene_meta": P.scene_dir / "scene_meta.json", "colour_file": P.colour_file,
                                  "xp_synth": cfg.inputs.xp_synth, "mapping": P.mapping_dir,
                                  "kernel_bright_q": {"value": A["kernel_q"]}})
    mark_done(stage)
    return stage
