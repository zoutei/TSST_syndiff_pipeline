"""Per-node template-offset correction of the per-band kernels (PS1 stack image vs header WCS, plus proper motion).

The PS1 3pi stack images are offset from their own header WCS by about (+0.3..0.45 W, +0.5..0.65 N) PS1 px, and
the PS1 epoch (~2012.8) differs from the TESS epoch, so every template star sits away from the model position by

    t(x) = J_T(x) [ Delta_alpha_stack(s(x)) + mu (t_PS1 - t_TESS) ]          [TESS px]

with J_T the trained TESS WCS Jacobian (px per arcsec, east/north), Delta_alpha_stack the measured offset of the
star's skycell s (arcsec), mu the Gaia proper motion. The kernels are translated by -t per kernel node (band-limited
Fourier shift, ``kernels.fourier_shift``), which moves the convolved template back onto the model. Record:
dev_runs/bandw_kernel_shift_20261009 (README "Per-node geometric kernel shift").

Stage ``template_shift`` (after ``fit`` and ``scene_boot``): node values of t from this run's own fitted WCS ->
``<out_root>/template_shift/spec.json``. Inputs:
  * ``inputs.template_shift_ps1``: per-skycell offsets, the JSON list written by
    ``bandw_kernel_shift_20261009/ps1_wcs_allfields/measure_combined.py`` (keys n, ra_c, dec_c, mjd, east_mas, north_mas);
  * ``inputs.template_shift_gaia``: directory of Gaia DR3 ``proj_<projection>.parquet`` (source_id, ra, dec, pmra, pmdec).
The proper-motion term is averaged over the Gaia stars on the CCD with 13.6 <= T < 16 (T from G, BP-RP) (the node shift can only remove its node
mean; each star's own motion relative to that mean stays as per-star scatter).

``inputs.template_shift`` selects what ``kernels`` applies: unset -> nothing (kernels bit-identical to before);
``auto`` -> this stage's spec.json; otherwise a spec.json path. Spec: {"node_x", "node_y", "tx_mpx", "ty_mpx"}
([iy][ix], mpx = 1e-3 TESS px; t is the template offset, the kernels move by -t)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .config import is_done, mark_done, write_provenance

STAGE = "template_shift"
PM_TMAG = (13.6, 16.0)
MIN_SKYCELL_STARS = 20


def spec_path(cfg) -> Path | None:
    """The spec ``kernels`` should apply, or None (``inputs.template_shift`` unset)."""
    v = cfg.inputs.template_shift
    if v is None:
        return None
    if str(v) == "auto":
        d = cfg.stage_dir(STAGE)
        if not is_done(d):
            raise FileNotFoundError(f"inputs.template_shift = auto but stage {STAGE} is not done: {d}")
        return d / "spec.json"
    return Path(v)


def load_spec(path: Path, node_x: np.ndarray, node_y: np.ndarray) -> dict:
    s = json.loads(Path(path).read_text())
    tx, ty = np.asarray(s["tx_mpx"], float), np.asarray(s["ty_mpx"], float)
    shape = (len(node_y), len(node_x))
    if tx.shape != shape or ty.shape != shape:
        raise ValueError(f"template shift spec {path}: shape {tx.shape}/{ty.shape}, kernel nodes {shape}")
    if not (np.allclose(s["node_x"], node_x) and np.allclose(s["node_y"], node_y)):
        raise ValueError(f"template shift spec {path}: nodes differ from the kernel nodes")
    if not (np.isfinite(tx).all() and np.isfinite(ty).all()):
        raise ValueError(f"template shift spec {path}: non-finite node values")
    return dict(tx_mpx=tx, ty_mpx=ty, path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def apply(K: np.ndarray, spec: dict) -> np.ndarray:
    """Translate kernels ``K[..., iy, ix, N, N]`` by -t per node. An all-zero spec returns ``K`` unchanged (bit-identical)."""
    from .kernels import fourier_shift
    tx, ty = spec["tx_mpx"], spec["ty_mpx"]
    if not (np.any(tx) or np.any(ty)):
        return K
    return fourier_shift(K, -tx / 1e3, -ty / 1e3)


def _hat_node_mean(x, y, v, nx, ny, wmin=0.5):
    from ._tk import hat_weights
    hx = hat_weights(np.clip(x, nx[0], nx[-1]), nx)
    hy = hat_weights(np.clip(y, ny[0], ny[-1]), ny)
    w = hy[:, None, :] * hx[None, :, :]
    ws = w.sum(-1)
    return np.where(ws > wmin, (w * v).sum(-1) / np.maximum(ws, 1e-12), np.nan), ws


def _tess_epoch(stem: str) -> float:
    from astropy.time import Time
    s = stem.split("-")[0].replace("tess", "")[:13]
    return float(Time(f"{s[:4]}:{s[4:7]}:{s[7:9]}:{s[9:11]}:{s[11:13]}", format="yday").jyear)


def compute(cfg) -> dict:
    """Node values of the template offset from this run's fitted WCS (``fit``) and boot scene."""
    import pandas as pd
    from . import kernels as KN

    ps1 = cfg.inputs.template_shift_ps1
    gdir = cfg.inputs.template_shift_gaia
    if ps1 is None or gdir is None:
        raise ValueError("stage template_shift needs inputs.template_shift_ps1 and inputs.template_shift_gaia")
    A = KN.load_model(cfg)
    b = KN.load_bundle(cfg)
    pix = KN.make_pix(np.asarray(A["params"]["wcs_coeff"]), b.cheb_static)
    nx_nodes = np.asarray(b.epsf_grid.node_x, float)                       # the kernel node grid (kernels.run_k02)
    ny_nodes = np.asarray(b.epsf_grid.node_y, float)
    cells = [c for c in json.loads(Path(ps1).read_text()) if c.get("n", 0) >= MIN_SKYCELL_STARS and "east_mas" in c]
    if not cells:
        raise ValueError(f"no usable skycells in {ps1}")
    ra = np.array([c["ra_c"] for c in cells])
    dec = np.array([c["dec_c"] for c in cells])
    x, y = pix(ra, dec)
    x, y = np.asarray(x), np.asarray(y)
    J = np.asarray(KN.jac(pix, ra, dec))                                   # px / arcsec, (east, north)
    e = np.c_[[c["east_mas"] for c in cells], [c["north_mas"] for c in cells]] / 1e3
    d_ps1 = np.einsum("nij,nj->ni", J, e) * 1e3                             # mpx
    ins = (x > -150) & (x < 2200) & (y > -150) & (y < 2200)
    if ins.sum() < 10:
        raise ValueError(f"only {int(ins.sum())} measured skycells on this CCD")
    ep_cells = 2000.0 + (np.array([c["mjd"] for c in cells]) - 51544.5) / 365.25
    t_tess = _tess_epoch(cfg.stem)
    # proper-motion term over Gaia stars with 13.6 <= T < 16 on this CCD (the stars the template shift is for; the boot
    # scene holds only T <= 13). T from G and BP-RP (Stassun et al. 2019, TIC v8 eq. 1).
    projs = sorted({str(c["proj"]) for c in cells if "proj" in c})
    gfiles = [Path(gdir) / f"proj_{p}.parquet" for p in projs]
    gfiles = [f for f in gfiles if f.exists()] or sorted(Path(gdir).glob("proj_*.parquet"))
    cols = ["source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
    g = pd.concat([pd.read_parquet(f, columns=cols) for f in gfiles]).drop_duplicates("source_id")
    c = (g.phot_bp_mean_mag - g.phot_rp_mean_mag).values
    T = g.phot_g_mean_mag.values - 0.00522555 * c ** 3 + 0.0891337 * c ** 2 - 0.633923 * c + 0.0324473
    st = g[np.isfinite(T) & (T >= PM_TMAG[0]) & (T < PM_TMAG[1]) & np.isfinite(g.pmra.values) & np.isfinite(g.pmdec.values)]
    gx, gy = pix(st.ra.values, st.dec.values)
    gx, gy = np.asarray(gx), np.asarray(gy)
    on = (gx >= 0) & (gx < 2048) & (gy >= 0) & (gy < 2048)
    st = st[on]
    sx, sy = gx[on], gy[on]
    xi, yi = x[ins], y[ins]
    nn = np.argmin((sx[:, None] - xi[None]) ** 2 + (sy[:, None] - yi[None]) ** 2, axis=1)
    dt = ep_cells[ins][nn] - t_tess
    pm = np.c_[st.pmra.values, st.pmdec.values] * dt[:, None] / 1e3                          # arcsec (east, north)
    Js = np.asarray(KN.jac(pix, st.ra.values, st.dec.values))
    d_pm = np.einsum("nij,nj->ni", Js, pm) * 1e3                                              # mpx
    psx, w_cells = _hat_node_mean(xi, yi, d_ps1[ins, 0], nx_nodes, ny_nodes)
    psy, _ = _hat_node_mean(xi, yi, d_ps1[ins, 1], nx_nodes, ny_nodes)
    pmx, w_stars = _hat_node_mean(sx, sy, d_pm[:, 0], nx_nodes, ny_nodes)
    pmy, _ = _hat_node_mean(sx, sy, d_pm[:, 1], nx_nodes, ny_nodes)
    n_fill = int((~np.isfinite(psx)).sum() + (~np.isfinite(pmx)).sum())
    for a in (psx, psy, pmx, pmy):                                          # unsupported nodes -> field mean
        a[~np.isfinite(a)] = np.nanmean(a) if np.isfinite(a).any() else 0.0
    tx, ty = psx + pmx, psy + pmy
    return dict(
        field=cfg.field, stem=cfg.stem, node_x=nx_nodes.tolist(), node_y=ny_nodes.tolist(), layout="[iy][ix]",
        tx_mpx=np.round(tx, 4).tolist(), ty_mpx=np.round(ty, 4).tolist(),
        ps1_x_mpx=np.round(psx, 4).tolist(), ps1_y_mpx=np.round(psy, 4).tolist(),
        pm_x_mpx=np.round(pmx, 4).tolist(), pm_y_mpx=np.round(pmy, 4).tolist(),
        cell_weight=np.round(w_cells, 2).tolist(), star_weight=np.round(w_stars, 2).tolist(), n_nodes_filled=n_fill,
        n_skycells=int(ins.sum()), n_pm_stars=int(len(st)), t_tess=t_tess, t_ps1_median=float(np.median(ep_cells[ins])),
        field_mean_mpx=[float(tx.mean()), float(ty.mean())],
        definition="t = J_T [PS1 stack image - header WCS + mu (t_PS1 - t_TESS)], template - model offset; kernels move by -t",
        created=time.strftime("%Y-%m-%dT%H:%M:%S"))


def run(cfg, force: bool = False) -> Path:
    """Stage ``template_shift``: needs ``fit`` and ``scene_boot``."""
    stage = cfg.stage_dir(STAGE)
    if is_done(stage) and not force:
        return stage
    for dep in ("fit", "scene_boot"):
        if not is_done(cfg.stage_dir(dep)):
            raise FileNotFoundError(f"stage {dep} not done: {cfg.stage_dir(dep)}")
    stage.mkdir(parents=True, exist_ok=True)
    spec = compute(cfg)
    (stage / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(f"[{STAGE}] {cfg.field}: field mean t = ({spec['field_mean_mpx'][0]:+.2f}, {spec['field_mean_mpx'][1]:+.2f}) mpx, "
          f"{spec['n_skycells']} skycells, {spec['n_pm_stars']} PM stars", flush=True)
    from .perband.paths import chain_paths
    write_provenance(stage, cfg, {"fit_params": chain_paths(cfg).fit_dir / "params.npz",
                                  "ps1_offsets": cfg.inputs.template_shift_ps1,
                                  "gaia_dir": {"value": str(cfg.inputs.template_shift_gaia)}})
    mark_done(stage)
    return stage
