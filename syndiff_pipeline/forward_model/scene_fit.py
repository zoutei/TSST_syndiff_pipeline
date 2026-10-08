# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Static single-FFI *scene* fit: 15×15 squares on every star, exact island flux solve.

Plan: ``docs/SCENE_MODE_PLAN_20260923.md``. Input: ``scene_export.py`` output.

Model at every union pixel p:  m_p = sum_i f_i T_i(p), each star's unit template
``T_i`` rendered on its own S×S square by the existing square-path renderer
(``loss.forward_model``: WCS, ePSF node blend, core recentre, chroma). Every pixel
enters the NLL once (``owner`` copy); the default is plain L2 (``--huber-delta 1e6``).
Fluxes are the exact weighted least-squares solution per overlap island,
differentiated through (with a finite Huber delta the L2 fluxes are not stationary
for the loss and must not be held fixed).
No per-star pedestal. Optional smooth background (``--bg-cheb-order k >= 0``, ported
2026-09-30 from branch scene-bg-cheb 5855d64): a Chebyshev polynomial of total degree <= k
over the CCD, b(x, y) = sum_m c_m T_i(u) T_j(v) with u = (x - 1024) / 1024 (science px),
solved EXACTLY with the fluxes at every step (the global coefficients couple every island:
per-island solves against [rhs | basis columns], then an M x M Schur complement for c, then
back-substitution) and differentiated through like the fluxes.

Roles: 0 = ePSF contributor (all gradients), 1 = WCS anchor (ePSF stop-gradient,
same barrier as the packed path), 2 = nuisance (template fully stop-gradient; flux
only, with a weak ridge toward ``scale * tess_flux``).

Rejection (per star, stages 2-3): score = chi2_red on the star's own core disk
(r <= core_radius) under the full scene model; brightness-rank-local median/MAD of
log chi2 (``stamp_reject.static_rank_outliers``); Schmitt trigger drop > tau_drop,
readmit < tau_keep; churn cap per refresh. A rejected star's core pixels leave the
likelihood, and the star stays in the model with its flux frozen so its wings are
still subtracted from its neighbours.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import _bootstrap  # noqa: F401
from . import bright_width as BW
from . import epsf_model as EM
from . import stamp_bg as SB
from . import fit as FIT
from . import fit_bundle as FB
from . import loss as L
from .groups import GroupSet
from .stamp_reject import static_rank_outliers

ROLE_CONTRIB, ROLE_ANCHOR, ROLE_NUISANCE = 0, 1, 2


def _log(msg: str) -> None:
    print(f"[scene_fit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


class Scene:
    def __init__(self, scene_dir: Path):
        self.dir = Path(scene_dir)
        self.meta = json.loads((self.dir / "scene_meta.json").read_text())
        z = np.load(self.dir / "scene_bundle.npz")
        self.z = {k: z[k] for k in z.files}
        self.S = int(self.z["stamp"])
        self.N = int(self.z["role"].shape[0])
        self.U = int(self.z["n_union"])
        self.role = self.z["role"].astype(np.int64)
        self.src = FB.load_fit_bundle(Path(self.meta["source_bundle"]))
        # EXPERIMENT (2026-09-30): coarser ePSF node grid, from --epsf-nodes / SYNDIFF_EPSF_NODES (0 = the bundle's grid)
        n_nodes = int(os.environ.get("SYNDIFF_EPSF_NODES", "0") or 0)
        if n_nodes and n_nodes != len(self.src.epsf_grid.node_x):
            self.coarsen_nodes(n_nodes)
        # colour per bundle star (BP-RP units) used everywhere in the colour model;
        # use_colour_file() overrides it for the scene stars
        self.colour = np.asarray(self.src.bp_rp, dtype=np.float64).copy()
        self.colour_source = "bp_rp"

    def coarsen_nodes(self, n: int) -> None:
        """Replace the bundle's ePSF node grid by an n x n grid spanning the same outermost nodes (edge placement kept),
        and resample the seed ePSF onto it. EXPERIMENT (prior bake-off 2026-09-30)."""
        g = self.src.epsf_grid
        old_x, old_y = np.asarray(g.node_x, float), np.asarray(g.node_y, float)
        new = dataclasses.replace(
            g, node_x=np.linspace(old_x[0], old_x[-1], n), node_y=np.linspace(old_y[0], old_y[-1], n),
            node_col_ccd=np.linspace(g.node_col_ccd[0], g.node_col_ccd[-1], n),
            node_row_ccd=np.linspace(g.node_row_ccd[0], g.node_row_ccd[-1], n))
        self.src.epsf_grid = new
        self.z["epsf_seed"] = resample_nodes(self.z["epsf_seed"], old_x, old_y, np.asarray(new.node_x), np.asarray(new.node_y))
        self.coarse_from = (old_x, old_y)

    def use_colour_file(self, path) -> dict:
        """Replace the scene stars' colour by ``path`` (CSV source_id,colour); returns counts."""
        sbi = self.z["star_bundle_index"]
        c, counts = colour_from_file(path, self.z["source_id"], self.colour[sbi])
        self.colour[sbi] = c
        self.colour_source = str(path)
        return counts

    def colour_ref(self) -> float:
        c = self.colour[self.z["star_bundle_index"]]
        m = (self.role != ROLE_NUISANCE) & np.isfinite(c)
        return float(np.mean(c[m])) if m.any() else 0.0

    def _population_weights(self, c) -> np.ndarray:
        """Per-star weight ~S/N^2 of the per-pixel likelihood (0 without a colour)."""
        f = np.asarray(self.z["tess_flux"], dtype=np.float64)
        core = np.asarray(self.z["core"], dtype=bool)
        nz = np.asarray(self.z["noise"], dtype=np.float64)[:, core]
        s2 = np.nanmedian(nz ** 2, axis=1)
        return np.where(np.isfinite(c) & (s2 > 0), f ** 2 / np.maximum(s2, 1e-12), 0.0)

    def colour_ref_per_population(self) -> dict:
        """Reference colour per role, weighted like the per-pixel likelihood (~S/N^2).

        Zeroes the loss-weighted mean colour offset separately for ePSF contributors
        and anchors, so a delta-weighted colour term cannot absorb colour-INDEPENDENT
        shape error (the 2026-09-16 chroma_kurt failure: contributors +0.067 vs
        anchors -0.039 under a single reference). Nuisance stars reuse the
        contributor value (their templates carry no gradient).
        """
        c = self.colour[self.z["star_bundle_index"]]
        w = self._population_weights(c)
        out = {}
        for r in (ROLE_CONTRIB, ROLE_ANCHOR):
            m = (self.role == r) & (w > 0)
            out[r] = float(np.sum(w[m] * c[m]) / np.sum(w[m])) if m.any() else self.colour_ref()
        out[ROLE_NUISANCE] = out[ROLE_CONTRIB]
        return out

    def delta2_mean(self, colour_ref):
        """<(c - c_ref)^2> over the population and weights that define ``colour_ref``.

        Float for the global gauge (unweighted, scored stars), per-role dict for the
        per_population gauge (S/N^2-weighted per role, nuisance = contributor value).
        """
        c = self.colour[self.z["star_bundle_index"]]
        if not isinstance(colour_ref, dict):
            m = (self.role != ROLE_NUISANCE) & np.isfinite(c)
            return float(np.mean((c[m] - colour_ref) ** 2)) if m.any() else 0.0
        w = self._population_weights(c)
        out = {}
        for r in (ROLE_CONTRIB, ROLE_ANCHOR):
            m = (self.role == r) & (w > 0)
            out[r] = (float(np.sum(w[m] * (c[m] - colour_ref[r]) ** 2) / np.sum(w[m])) if m.any()
                      else self.delta2_mean(self.colour_ref()))
        out[ROLE_NUISANCE] = out[ROLE_CONTRIB]
        return out

    def optical_axis(self) -> tuple:
        """Camera optical axis on this CCD, science px, from tess-point's Levine FPG."""
        import tess_stars2px as TP
        sp = TP.TESS_Spacecraft_Pointing_Data()
        k = list(sp.sectors).index(int(self.meta["sector"]))
        xy = sp.fpgObjs[k].mm_to_pix_single_ccd(int(self.meta["camera"]) - 1, np.zeros(2),
                                                 int(self.meta["ccd"]) - 1)
        return (float(np.asarray(xy).ravel()[0]), float(np.asarray(xy).ravel()[1]))

    def contexts(self, colour_ref, chroma_axis=None, chroma_g8_gauge="mean", chroma_g8_no_dil=False,
                 chroma_g8_extras=(), delta2_mean=0.0, bright_q=None, bright_q_ref=0.0,
                 chroma_g8_drop=(), chroma_radial_knots=None, chroma_radial_mode=None,
                 chroma_coma_knots=None):
        """One homogeneous square-path context per role (K=1 groups).

        ``colour_ref`` and ``delta2_mean`` are floats or per-role dicts. ``bright_q`` is the
        per-SCENE-star peak charge per 2-s read (bright_width.py), scattered here onto the
        bundle-star index the contexts use; None leaves the contexts without it.
        """
        b = self.src
        out = []
        bq = None
        if bright_q is not None:
            bq = np.zeros(int(np.asarray(b.ra).shape[0]), np.float32)
            bq[self.z["star_bundle_index"]] = np.asarray(bright_q, np.float32)
            bq = jnp.asarray(bq)
        for r in (ROLE_CONTRIB, ROLE_ANCHOR, ROLE_NUISANCE):
            idx = np.flatnonzero(self.role == r)
            if idx.size == 0:
                continue
            members = self.z["star_bundle_index"][idx][:, None].astype(np.int64)
            gs = GroupSet(
                n_groups=idx.size, max_group_size=1, members=members,
                valid=np.ones((idx.size, 1), dtype=bool),
                kept_star_mask=np.ones(int(np.asarray(b.ra).shape[0]), dtype=bool),
                dropped_oversized=0,
            )
            ctx = L.build_static_context(
                cheb_static=b.cheb_static,
                wcs_frame_basis=np.asarray(b.wcs_frame_basis),
                w_frame_basis=np.asarray(b.w_frame_basis),
                epsf_grid=b.epsf_grid,
                groups=gs,
                ra=np.asarray(b.ra), dec=np.asarray(b.dec),
                stamp_center_x=self.z["cx"][idx].astype(np.float32),
                stamp_center_y=self.z["cy"][idx].astype(np.float32),
                t_exp_sec=float(b.t_exp_sec),
                stamp_snr_weight=np.ones(idx.size, np.float32),
                fit_radius=np.full(idx.size, 99.0, np.float32),
                x_lin=np.asarray(b.x_lin), y_lin=np.asarray(b.y_lin),
                cheb_basis=np.asarray(b.cheb_basis),
                is_epsf_contributor=np.full(idx.size, r == ROLE_CONTRIB),
                bp_rp=self.colour,
                colour_ref=(colour_ref[r] if isinstance(colour_ref, dict) else colour_ref),
                chroma_axis=chroma_axis,
                chroma_g8_gauge=chroma_g8_gauge,
                chroma_g8_no_dil=chroma_g8_no_dil,
                chroma_g8_extras=chroma_g8_extras,
                chroma_g8_drop=chroma_g8_drop,
                chroma_radial_knots=chroma_radial_knots,
                chroma_radial_mode=chroma_radial_mode,
                chroma_coma_knots=chroma_coma_knots,
                chroma_delta2_mean=(delta2_mean[r] if isinstance(delta2_mean, dict) else delta2_mean),
            )
            if bq is not None:
                ctx = L.replace(ctx, bright_q=bq, bright_q_ref=float(bright_q_ref))
            out.append((r, jnp.asarray(idx, dtype=jnp.int32), ctx))
        return out

    def tier_tables(self):
        """Static index tables for the batched per-island solve."""
        z = self.z
        tiers = [int(k) for k in z["tiers"]]
        st_tier, st_row, st_slot = z["star_tier"], z["star_row"], z["star_slot"]
        pi, pj = z["pair_i"], z["pair_j"]
        out = []
        for ti, K in enumerate(tiers):
            n_isl = int(z[f"tier{K}_star_idx"].shape[0])
            s = np.flatnonzero(st_tier == ti)
            p = np.flatnonzero(st_tier[pi] == ti)
            pad = (z[f"tier{K}_star_idx"] < 0)
            out.append(dict(
                K=K, n=n_isl,
                s=jnp.asarray(s, jnp.int32), row=jnp.asarray(st_row[s], jnp.int32),
                slot=jnp.asarray(st_slot[s], jnp.int32),
                p=jnp.asarray(p, jnp.int32), prow=jnp.asarray(st_row[pi[p]], jnp.int32),
                psi=jnp.asarray(st_slot[pi[p]], jnp.int32),
                psj=jnp.asarray(st_slot[pj[p]], jnp.int32),
                pad=jnp.asarray(pad, jnp.float32),
            ))
        return out


def colour_from_file(path, source_id, fallback):
    """Per-star colour from a CSV ``source_id,colour`` (BP-RP units).

    ``source_id``/``fallback`` are per scene star. A star missing from the file, or with
    a NaN colour there, keeps ``fallback`` (its BP-RP). Returns (colour, counts).
    """
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    table = {int(r["source_id"]): float(r["colour"] or "nan") for r in rows}
    got = np.array([table.get(int(s), np.nan) for s in np.asarray(source_id)], dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    matched = np.isfinite(got)
    c = np.where(matched, got, fallback)
    counts = {"n_file_rows": len(rows), "matched": int(matched.sum()),
              "fallback": int(np.sum(~matched & np.isfinite(fallback))),
              "no_colour": int(np.sum(~np.isfinite(c)))}
    return c, counts


def resample_nodes(field, old_x, old_y, new_x, new_y):
    """Bilinear resample of a (R, C, ...) node field from node positions (old_y, old_x) to (new_y, new_x)."""
    field = np.asarray(field, np.float64)
    fy = np.interp(new_y, old_y, np.arange(len(old_y))); fx = np.interp(new_x, old_x, np.arange(len(old_x)))
    y0 = np.clip(np.floor(fy).astype(int), 0, len(old_y) - 2); x0 = np.clip(np.floor(fx).astype(int), 0, len(old_x) - 2)
    wy = fy - y0; wx = fx - x0
    out = np.empty((len(new_y), len(new_x)) + field.shape[2:])
    for i in range(len(new_y)):
        for j in range(len(new_x)):
            a, b = y0[i], x0[j]
            out[i, j] = ((1 - wy[i]) * ((1 - wx[j]) * field[a, b] + wx[j] * field[a, b + 1])
                         + wy[i] * ((1 - wx[j]) * field[a + 1, b] + wx[j] * field[a + 1, b + 1]))
    return out


def coarsen_params(params: dict, scene: Scene) -> dict:
    """Resample an init params file's per-node ePSF leaves onto a coarsened scene grid (decoded, then re-encoded)."""
    if not hasattr(scene, "coarse_from") or params["epsf_base_raw"].shape[0] == len(scene.src.epsf_grid.node_y):
        return params
    ox, oy = scene.coarse_from; nx, ny = np.asarray(scene.src.epsf_grid.node_x), np.asarray(scene.src.epsf_grid.node_y)
    p = dict(params)
    base = np.asarray(EM.decode_epsf_base(params["epsf_base_raw"]), np.float64)
    p["epsf_base_raw"] = EM.encode_epsf_base(jnp.asarray(resample_nodes(base, ox, oy, nx, ny), jnp.float32))
    g = params["epsf_modes"].shape[-1]
    p["epsf_modes"] = jnp.zeros((params["epsf_modes"].shape[0], len(ny), len(nx), g, g), jnp.float32) \
        if params["epsf_modes"].shape[0] == 0 else jnp.asarray(np.stack([resample_nodes(m, ox, oy, nx, ny) for m in np.asarray(params["epsf_modes"])]), jnp.float32)
    for k in ("chroma_shift",):
        if k in params:
            p[k] = jnp.asarray(np.stack([resample_nodes(v, ox, oy, nx, ny) for v in np.asarray(params[k])]), jnp.float32)
    for k in ("chroma_dilation", "chroma_halo"):
        if k in params:
            p[k] = jnp.asarray(resample_nodes(np.asarray(params[k]), ox, oy, nx, ny), jnp.float32)
    return p


def init_params(scene: Scene) -> dict:
    b = scene.src
    n_rows, n_cols = scene.z["epsf_seed"].shape[:2]
    g = scene.z["epsf_seed"].shape[-1]
    return {
        "wcs_coeff": jnp.asarray(b.params0["wcs_coeff"], jnp.float32),
        "epsf_base_raw": EM.encode_epsf_base(jnp.asarray(scene.z["epsf_seed"])),
        "epsf_modes": jnp.zeros((0, n_rows, n_cols, g, g), jnp.float32),
        "w_coeff": jnp.zeros((0, 1), jnp.float32),
        "chroma_shift": jnp.zeros((2, n_rows, n_cols), jnp.float32),
        "chroma_dilation": jnp.zeros((n_rows, n_cols), jnp.float32),
    }


def carry_g8(old, old_extras, new_extras) -> np.ndarray:
    """Warm-start global8 coefficients: keep ``old`` and zero-pad the extras it lacks.

    ``old_extras`` are the extras the source was trained with; they must be a prefix of
    ``new_extras`` (the slots are positional).
    """
    old = np.asarray(old, dtype=float)
    old_extras, new_extras = tuple(old_extras), tuple(new_extras)
    if old.shape != (8 + len(old_extras),):
        raise ValueError(f"warm-start chroma_g8 has {old.shape[0]} values but its extras "
                         f"{old_extras} need {8 + len(old_extras)}")
    if new_extras[:len(old_extras)] != old_extras:
        raise ValueError(f"--chroma-g8-extras {new_extras} must start with the warm start's "
                         f"extras {old_extras}")
    return np.concatenate([old, np.zeros(len(new_extras) - len(old_extras))])


def set_chroma_model(params: dict, model: str, *, halo: bool, g8_init=None, g8_blur_order: int = 0,
                     g8_extras=(), g8_source_extras=None) -> dict:
    """Swap the colour leaves for the requested model; every other leaf is untouched.

    global8: ``g8_init`` wins; else an incoming ``chroma_g8`` (trained with extras
    ``g8_source_extras``) is kept and zero-padded to ``g8_extras``; else zeros.
    """
    p = {k: v for k, v in params.items()
         if k not in ("chroma_shift", "chroma_dilation", "chroma_g8", "chroma_halo")}
    n_rows, n_cols = p["epsf_base_raw"].shape[:2]
    if model == "nodes":
        p["chroma_shift"] = params.get("chroma_shift", jnp.zeros((2, n_rows, n_cols), jnp.float32))
        p["chroma_dilation"] = params.get("chroma_dilation", jnp.zeros((n_rows, n_cols), jnp.float32))
    elif model == "global8":
        if g8_blur_order and g8_extras:
            raise ValueError("use either --chroma-g8-blur-order or --chroma-g8-extras, not both")
        n8 = 8 + int(g8_blur_order) + len(g8_extras)
        if g8_init is not None:
            init = np.asarray(g8_init, float)
        elif "chroma_g8" in params:
            if g8_source_extras is None:
                raise ValueError("keeping a warm-start chroma_g8 needs its g8_source_extras")
            init = carry_g8(params["chroma_g8"], g8_source_extras, g8_extras)
        else:
            init = np.zeros(n8)
        if init.shape != (n8,):
            raise ValueError(f"--chroma-g8-init needs {n8} values for blur order {g8_blur_order}")
        p["chroma_g8"] = jnp.asarray(init, jnp.float32)
    elif model != "none":
        raise ValueError(f"unknown --chroma-model {model}")
    if halo:
        if model == "none":
            raise ValueError("--chroma-halo needs a colour model")
        p["chroma_halo"] = params.get("chroma_halo", jnp.zeros((n_rows, n_cols), jnp.float32))
    return p


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


BG_CENTRE, BG_SCALE = 1024.0, 1024.0


def bg_terms(order: int):
    """(i, j) Chebyshev degree pairs with i + j <= order (x degree i, y degree j)."""
    return [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]


def bg_basis(x, y, order: int):
    """Chebyshev background basis at science-px (x, y); returns (..., M)."""
    u = (np.asarray(x, np.float64) - BG_CENTRE) / BG_SCALE
    v = (np.asarray(y, np.float64) - BG_CENTRE) / BG_SCALE
    Tu, Tv = [np.ones_like(u), u], [np.ones_like(v), v]
    for _ in range(2, order + 1):
        Tu.append(2 * u * Tu[-1] - Tu[-2]); Tv.append(2 * v * Tv[-1] - Tv[-2])
    return np.stack([Tu[i] * Tv[j] for i, j in bg_terms(order)], axis=-1)


def island_solve(T, st, a, tiers, *, ridge: float, prior_kappa: float, phi=None, ped=None):
    """Exact weighted-L2 fluxes per overlap island; with ``phi`` (N, S2, M) also the global
    Chebyshev background c via a Schur complement; with ``ped`` also one free constant per stamp
    (``--stamp-bg fit``) solved inside each island's system. Returns (f, c, b); c / b are None
    when the term is off.

    ``a``: the scene arrays (data, var, valid, finite, owner, uid, pi, pj, lj, pmask, nuis).
    ``ped``: dict with ``cell_star`` (N, S2) scene index of the stamp whose pedestal each copy's
    pixel belongs to (stamp_bg.cell_owner), ``cell_slot`` (N, S2) that stamp's island slot,
    ``own_cell`` (N, S2) 1 on the copy inside its own cell (counts each union pixel once),
    ``prior_w`` (N,) = 1/sigma^2 (0 = no prior) and ``prior_mu`` (N,) the prior mean.
    Without ``phi``/``ped`` the ops are exactly the pre-background solve.
    """
    uid, var, pi, pj = a["uid"], a["var"], a["pi"], a["pj"]
    N = T.shape[0]
    w = a["valid"] * st["pix_active_u"][uid] / var
    diag = jnp.sum(w * T * T, axis=1)
    bvec = jnp.sum(w * T * a["data"], axis=1)
    Tj = jnp.take_along_axis(T[pj], a["lj"], axis=1)
    aij = jnp.sum(a["pmask"] * w[pi] * T[pi] * Tj, axis=1)
    lam = prior_kappa * a["nuis"] * jnp.sum(a["finite"] / var * T * T, axis=1)
    diag = diag * (1.0 + ridge) + lam + 1e-20
    bvec = bvec + lam * st["f_prior"]
    f = jnp.zeros((N,), jnp.float32)
    use_bg = phi is not None
    use_ped = ped is not None
    if use_bg:
        # background normal equations: star-background coupling Cm (N, M) over each star's
        # square; background Gram and rhs over OWNED pixels (each union pixel once); known
        # (rejected/fixed) star fluxes moved to the rhs.
        M = phi.shape[-1]
        Cm = jnp.einsum("np,npm->nm", w * T, phi)
        wo = w * a["owner"]
        Sbg = jnp.einsum("np,npm,npl->ml", wo, phi, phi)
        rbg = jnp.einsum("np,npm->m", wo * a["data"], phi)
        rbg = rbg - jnp.einsum("n,nm->m", (1.0 - st["star_free"]) * st["f_fixed"], Cm)
        parts = []
    if use_ped:
        wT = w * T
        wc = w * ped["own_cell"]
        Dp = jnp.sum(wc, axis=1) + ped["prior_w"]
        rp = jnp.sum(wc * a["data"], axis=1) + ped["prior_w"] * ped["prior_mu"]
        if use_bg:
            Cp = jnp.einsum("np,npm->nm", wc, phi)
        b_out = jnp.zeros((N,), jnp.float32)
    for t in tiers:
        K, n = t["K"], t["n"]
        A = jnp.zeros((n, K, K), jnp.float32)
        A = A.at[t["row"], t["slot"], t["slot"]].add(diag[t["s"]])
        A = A.at[t["prow"], t["psi"], t["psj"]].add(aij[t["p"]])
        A = A.at[t["prow"], t["psj"], t["psi"]].add(aij[t["p"]])
        bb = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(bvec[t["s"]])
        free = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(st["star_free"][t["s"]])
        ff = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(st["f_fixed"][t["s"]])
        fixed = 1.0 - free  # includes padding slots (ff = 0 there)
        A2 = A * free[:, :, None] * free[:, None, :] + jax.vmap(jnp.diag)(fixed)
        b2 = free * (bb - jnp.einsum("nij,nj->ni", A, ff * fixed)) + fixed * ff
        if use_ped:
            # [[A2, B2], [B2^T, D]] over (fluxes, pedestals) of the island. B[i, j] couples star
            # i's flux to the pedestal of cell j over star i's square; fixed fluxes -> rhs.
            s_ = t["s"]
            Bm = jnp.zeros((n, K, K), jnp.float32).at[
                t["row"][:, None], t["slot"][:, None], ped["cell_slot"][s_]].add(wT[s_])
            occ = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(1.0)
            Dt = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(Dp[s_]) + (1.0 - occ)
            Dt = jnp.where(Dt > 0, Dt, 1.0)          # a stamp with no weight and no prior -> 0
            rpt = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(rp[s_])
            rpt = rpt - jnp.einsum("nij,ni->nj", Bm, ff * fixed)
            B2 = Bm * free[:, :, None]
            A2 = jnp.concatenate([jnp.concatenate([A2, B2], axis=2),
                                  jnp.concatenate([jnp.swapaxes(B2, 1, 2), jax.vmap(jnp.diag)(Dt)], axis=2)],
                                 axis=1)
            b2 = jnp.concatenate([b2, rpt], axis=1)
        if not use_bg:
            x = jnp.linalg.solve(A2, b2[..., None])[..., 0]
            f = f.at[t["s"]].set(x[t["row"], t["slot"]])
            if use_ped:
                b_out = b_out.at[t["s"]].set(x[:, K:][t["row"], t["slot"]])
            continue
        # background columns for this tier's free stars (fixed/padding rows are zero)
        Ct = jnp.zeros((n, K, M), jnp.float32).at[t["row"], t["slot"]].set(Cm[t["s"]])
        Ct = Ct * free[:, :, None]
        if use_ped:
            Ct = jnp.concatenate(
                [Ct, jnp.zeros((n, K, M), jnp.float32).at[t["row"], t["slot"]].set(Cp[t["s"]])], axis=1)
        sol = jnp.linalg.solve(A2, jnp.concatenate([b2[..., None], Ct], axis=-1))
        X0, XC = sol[..., 0], sol[..., 1:]
        Sbg = Sbg - jnp.einsum("nkm,nkl->ml", Ct, XC)
        rbg = rbg - jnp.einsum("nkm,nk->m", Ct, X0)
        parts.append((t, X0, XC))
    if not use_bg:
        return f, None, (b_out if use_ped else None)
    Sbg = Sbg + 1e-9 * jnp.trace(Sbg) / M * jnp.eye(M, dtype=jnp.float32)
    c = jnp.linalg.solve(Sbg, rbg)
    for t, X0, XC in parts:
        x = X0 - jnp.einsum("nkm,m->nk", XC, c)
        f = f.at[t["s"]].set(x[t["row"], t["slot"]])
        if use_ped:
            b_out = b_out.at[t["s"]].set(x[:, t["K"]:][t["row"], t["slot"]])
    return f, c, (b_out if use_ped else None)


def bright_q_by_source(path, source_id) -> np.ndarray:
    """Per-star q from a bright_width_q.npz, matched by source_id (every star must be there)."""
    zq = np.load(path)
    idx = {int(s): i for i, s in enumerate(np.asarray(zq["source_id"]))}
    sid = np.asarray(source_id)
    missing = [int(s) for s in sid if int(s) not in idx]
    if missing:
        raise KeyError(f"{len(missing)} scene stars missing from {path} (e.g. {missing[:3]})")
    return np.asarray(zq["q"], np.float64)[[idx[int(s)] for s in sid]]


def scene_stamp_bg(scene, *, r_in=5.0, r_out=7.0, clip=3.0, nb_frac=0.01, nb_radius=3.0):
    """Annulus estimates + pixel cells for ``--stamp-bg`` (deterministic from the scene)."""
    z = scene.z
    est = SB.annulus_estimates(z["data"], z["valid"], z["x0"], z["y0"], z["cx"], z["cy"],
                               BW.catalogue_flux(z["tess_mag"]), S=scene.S, r_in=r_in, r_out=r_out,
                               clip=clip, nb_frac=nb_frac, nb_radius=nb_radius)
    cells = SB.cell_owner(z["uid"], scene.U, scene.S)
    return est, cells


def make_model(scene: Scene, *, colour_ref, huber_delta: float, ridge: float,
               prior_kappa: float, lambda_lap: float, lambda_pixel: float, chroma_axis=None,
               lambda_fine_nbr: float = 0.0, fine_nbr_mode: str = L.FINE_NBR_MODE_DEFAULT,
               fine_nbr_basis_sigma: float = 0.0, lambda_local_poly: float = 0.0,
               local_poly_window: int = L.LOCAL_POLY_WINDOW, local_poly_orders=L.LOCAL_POLY_ORDERS,
               local_poly_radii=L.LOCAL_POLY_RADII_PX,
               chroma_g8_gauge: str = "mean",
               chroma_g8_no_dil: bool = False, chroma_g8_extras=(), delta2_mean=0.0,
               chroma_g8_drop=(), chroma_radial_knots=None, chroma_radial_mode=None, chroma_coma_knots=None,
               bright_q=None, bright_q_ref: float = 0.0, return_templates: bool = False,
               bright_width: str = "none", bright_width_q_file=None, bright_width_q_ref: float = 0.0,
               bright_width_gen_per_dsigma: float = BW.GEN_PER_DSIGMA,
               bg_cheb_order: int = -1,
               stamp_bg: str = "none", stamp_bg_r_in: float = 5.0, stamp_bg_r_out: float = 7.0,
               stamp_bg_clip: float = 3.0, stamp_bg_prior_sigma=None, stamp_bg_nb_frac: float = 0.01,
               stamp_bg_nb_radius: float = 3.0, penalty_nref: float = 0.0):
    """(loss_fn, diagnose), plus ``render_and_solve(params, st) -> (T, f)`` when
    ``return_templates``.

    Every optional term is a keyword named exactly like its fit_meta.json key, so a fit can be
    rebuilt from its fit_meta (the OOF renderer passes every make_model keyword it finds there):

    - brightness width (bright_width.py): ``bright_width`` 'lin' with ``bright_width_q_file``
      (npz: source_id, q) and ``bright_width_q_ref``; or pass per-scene-star ``bright_q`` /
      ``bright_q_ref`` directly. ``bright_width_gen_per_dsigma`` is checked against the code.
    - ``bg_cheb_order`` >= 0: Chebyshev background (``island_solve``); aux ``bg_coef``.
    - ``stamp_bg`` 'annulus' (fixed per-stamp offset, subtracted from the data) or 'fit' (free
      per-stamp constant with a Gaussian prior toward the annulus estimate, sigma
      ``stamp_bg_prior_sigma`` e-/s; None = 3 x the estimate's robust SE; <= 0 = no prior);
      aux ``stamp_bg`` (the per-stamp values used). See stamp_bg.py.

    ``st`` may carry ``bg_coef_fixed`` (M,) and/or ``stamp_bg_fixed`` (N,) to hold those terms
    at given values instead of solving them (e.g. to split a model into flux-independent and
    per-star parts). ``diagnose.components(params, st) -> (f, c, b)`` exposes the solve.

    Model-error floor: if ``st`` carries ``var_eff`` (N, S2), it replaces the stored variance in the
    flux solve and in the loss (both the chi^2 and the 0.5 ln var term). It is state, not a parameter,
    so no gradient flows through it; ``run`` refreshes it from ``diagnose.model``. ``diagnose`` still
    reports chi^2_core against the stored variance, so scores stay comparable across noise models.

    ``penalty_nref`` > 0 multiplies every ePSF penalty by penalty_nref / N_pix (N_pix = valid owned
    pixels of this scene). The data term is a mean over N_pix, so this keeps the penalty's strength
    relative to the data the same in every scene and fold. 0 = off (penalties unscaled).
    """
    S2 = scene.S * scene.S
    N, U = scene.N, scene.U
    if bright_width not in BW.FORMS:
        raise ValueError(f"unknown bright_width {bright_width!r}")
    if bright_width != "none" and bright_q is None:
        if abs(float(bright_width_gen_per_dsigma) - BW.GEN_PER_DSIGMA) > 1e-12:
            raise ValueError(f"fit used bright_width_gen_per_dsigma={bright_width_gen_per_dsigma}, this code "
                             f"has {BW.GEN_PER_DSIGMA}")
        if bright_width_q_file is None:
            raise ValueError("bright_width='lin' needs bright_width_q_file (or bright_q)")
        bright_q = bright_q_by_source(bright_width_q_file, scene.z["source_id"])
        bright_q_ref = float(bright_width_q_ref)
    if stamp_bg not in SB.MODES:
        raise ValueError(f"unknown stamp_bg {stamp_bg!r}")
    ctxs = scene.contexts(colour_ref, chroma_axis=chroma_axis, chroma_g8_gauge=chroma_g8_gauge,
                          chroma_g8_no_dil=chroma_g8_no_dil, chroma_g8_extras=chroma_g8_extras,
                          chroma_g8_drop=chroma_g8_drop, chroma_radial_knots=chroma_radial_knots,
                          chroma_radial_mode=chroma_radial_mode, chroma_coma_knots=chroma_coma_knots,
                          delta2_mean=delta2_mean, bright_q=bright_q, bright_q_ref=bright_q_ref)
    tiers = scene.tier_tables()
    z = scene.z
    data = jnp.asarray(z["data"])
    var = L.pixel_variance(jnp.asarray(z["noise"]))
    valid = jnp.asarray(z["valid"], jnp.float32)
    finite = jnp.asarray(z["finite"], jnp.float32)
    owner = jnp.asarray(z["owner"], jnp.float32)
    uid = jnp.asarray(z["uid"], jnp.int32)
    pi = jnp.asarray(z["pair_i"], jnp.int32)
    pj = jnp.asarray(z["pair_j"], jnp.int32)
    lj = jnp.asarray(z["pair_lj"].astype(np.int32))
    pmask = jnp.asarray(z["pair_mask"], jnp.float32)
    nuis = jnp.asarray(scene.role == ROLE_NUISANCE, jnp.float32)
    core = jnp.asarray(z["core"], jnp.float32)
    n_pix_static = float(np.sum(np.asarray(z["valid"]) & np.asarray(z["owner"])))
    pen_fac = float(penalty_nref) / max(n_pix_static, 1.0) if penalty_nref > 0 else 1.0

    def templates(params):
        T = jnp.zeros((N, S2), jnp.float32)
        for r, idx, ctx in ctxs:
            t = L.forward_model(params, ctx)[0].reshape(idx.shape[0], S2)
            if r == ROLE_NUISANCE:
                t = jax.lax.stop_gradient(t)
            T = T.at[idx].set(t)
        return T

    use_bg = bg_cheb_order >= 0
    phi = None
    if use_bg:
        S = scene.S
        kk = np.arange(S2)
        bx = z["cx"][:, None].astype(np.float64) + (kk % S - S // 2)[None]
        by = z["cy"][:, None].astype(np.float64) + (kk // S - S // 2)[None]
        phi = jnp.asarray(bg_basis(bx, by, bg_cheb_order), jnp.float32)  # (N, S2, M)
    use_ped = stamp_bg != "none"
    ped = None
    if use_ped:
        est, cells = scene_stamp_bg(scene, r_in=stamp_bg_r_in, r_out=stamp_bg_r_out, clip=stamp_bg_clip,
                                    nb_frac=stamp_bg_nb_frac, nb_radius=stamp_bg_nb_radius)
        a_est = np.nan_to_num(est["est"], nan=0.0)
        cs = jnp.asarray(cells, jnp.int32)
        if stamp_bg == "annulus":
            data = data - jnp.asarray(a_est, jnp.float32)[cs]    # FIXED offset: no gradient
        else:
            if stamp_bg_prior_sigma is None:
                sig = 3.0 * est["se"]
            else:
                sig = np.full(N, float(stamp_bg_prior_sigma))
            pw_ = np.where(np.isfinite(est["est"]) & np.isfinite(sig) & (sig > 0), 1.0 / np.maximum(sig, 1e-30) ** 2, 0.0)
            island = np.stack([z["star_tier"], z["star_row"]], 1)
            if not np.array_equal(island[cells], np.broadcast_to(island[:, None, :], cells.shape + (2,))):
                raise AssertionError("a stamp's pixel cell left its overlap island")
            ped = dict(cell_star=cs, cell_slot=jnp.asarray(z["star_slot"][cells], jnp.int32),
                       own_cell=jnp.asarray(cells == np.arange(N)[:, None], jnp.float32),
                       prior_w=jnp.asarray(pw_, jnp.float32), prior_mu=jnp.asarray(a_est, jnp.float32))
    arrs = dict(data=data, var=var, valid=valid, finite=finite, owner=owner, uid=uid, pi=pi, pj=pj,
                lj=lj, pmask=pmask, nuis=nuis)

    def solve(T, st):
        # Differentiated THROUGH (as the packed path does): the loss is Huber but the
        # fluxes are the weighted-L2 optimum, so they are not stationary points of the
        # loss and holding them fixed would drop a real gradient term (measured 4% on
        # wcs_coeff and >10x on epsf_base_raw on the smoke scene).
        a_ = dict(arrs, var=st["var_eff"]) if "var_eff" in st else arrs
        phi_, ped_, c_fix, b_fix = phi, ped, None, None
        if use_bg and "bg_coef_fixed" in st:
            c_fix = st["bg_coef_fixed"]
            a_ = dict(a_, data=a_["data"] - jnp.einsum("npm,m->np", phi, c_fix))
            phi_ = None
        if use_ped and stamp_bg == "fit" and "stamp_bg_fixed" in st:
            b_fix = st["stamp_bg_fixed"]
            a_ = dict(a_, data=a_["data"] - b_fix[ped["cell_star"]])
            ped_ = None
        f, c, b = island_solve(T, st, a_, tiers, ridge=ridge, prior_kappa=prior_kappa, phi=phi_, ped=ped_)
        return f, (c_fix if c_fix is not None else c), (b_fix if b_fix is not None else b)

    def scene_model(T, f, c=None, b=None):
        mu = jnp.zeros((U + 1,), jnp.float32).at[uid].add(f[:, None] * T)
        m = mu[uid]
        if use_bg:
            m = m + jnp.einsum("npm,m->np", phi, c)
        if b is not None:
            m = m + b[ped["cell_star"]]
        return m

    def loss_fn(params, st):
        T = templates(params)
        f, c, b = solve(T, st)
        m = scene_model(T, f, c, b)
        var_l = st["var_eff"] if "var_eff" in st else var
        chi = (data - m) / jnp.sqrt(var_l)
        pw = valid * owner * st["pix_active_u"][uid]
        ell = 0.5 * L.huber_rho(chi, huber_delta) + 0.5 * jnp.log(var_l)
        data_term = jnp.sum(pw * ell) / jnp.clip(jnp.sum(pw), 1.0, None)
        base = L.decoded_epsf_base(params)
        lap = L.node_smoothness_penalty(base[None])
        pix_lap = L.pixel_laplacian_penalty(base)
        fine = (L.fine_nbr_penalty(base[None], fine_nbr_mode, basis_sigma=fine_nbr_basis_sigma)
                if lambda_fine_nbr > 0
                else jnp.zeros((), jnp.float32))
        lpoly = (L.local_poly_penalty(base, local_poly_window, local_poly_orders, local_poly_radii)
                 if lambda_local_poly > 0 else jnp.zeros((), jnp.float32))
        if penalty_nref > 0:
            loss = data_term + pen_fac * (lambda_lap * lap + lambda_pixel * pix_lap + lambda_fine_nbr * fine
                                          + lambda_local_poly * lpoly)
        else:   # historical expression, kept verbatim so the default path stays bit-identical
            loss = (data_term + lambda_lap * lap + lambda_pixel * pix_lap + lambda_fine_nbr * fine
                    + lambda_local_poly * lpoly)
        aux = {"loss": loss, "data_term": data_term, "lap": lap, "pixel_lap": pix_lap,
               "fine_nbr": fine, "local_poly": lpoly, "n_pix": jnp.sum(pw)}
        if use_bg:
            aux["bg_coef"] = c
        if b is not None:
            aux["stamp_bg"] = b
        return loss, aux

    def diagnose(params, st):
        T = templates(params)
        f, c, b = solve(T, st)
        m = scene_model(T, f, c, b)
        chi = (data - m) / jnp.sqrt(var)
        cv = core[None] * valid
        chi2_core = jnp.sum(cv * chi * chi, axis=1) / jnp.clip(jnp.sum(cv, axis=1), 1.0, None)
        return f, chi2_core, chi

    def components(params, st):
        return solve(templates(params), st)

    def model_pixels(params, st):
        """Scene model (N, S2) in e-/s with the current fluxes; input to the model-error floor."""
        T = templates(params)
        f, c, b = solve(T, st)
        return scene_model(T, f, c, b)

    pos_ctx = [(idx, ctx) for r, idx, ctx in ctxs if r != ROLE_NUISANCE]

    def positions(params):
        """Detector x, y (px) of every contributor and anchor from the WCS leaf (stop-rule check)."""
        xs, ys = [], []
        for idx, ctx in pos_ctx:
            x_t, y_t = L.CW.eval_all_positions(ctx.x_lin, ctx.y_lin, ctx.cheb_basis, params["wcs_coeff"],
                                               ctx.wcs_frame_basis, ctx.n_terms)
            xs.append(jnp.ravel(x_t)); ys.append(jnp.ravel(y_t))
        return jnp.concatenate(xs), jnp.concatenate(ys)

    diagnose.components = components
    diagnose.model = model_pixels
    diagnose.positions = positions
    diagnose.n_pix_static = n_pix_static
    diagnose.pen_fac = pen_fac

    if return_templates:
        def render_and_solve(params, st):
            T = templates(params)
            return T, solve(T, st)[0]
        return loss_fn, diagnose, render_and_solve
    return loss_fn, diagnose


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------


def refresh_due(stage: int, step: int, *, reject_every: int, burn_in: int, converged: bool,
                last_refresh_changed) -> bool:
    """Whether to refresh the rejection after ``step`` steps. ``reject_every`` 0 = never
    (the convergence-forced refresh used to fire anyway and drop ~300 stars at a
    run-dependent step, so eval star sets differed)."""
    if stage < 2 or reject_every <= 0:
        return False
    if step >= burn_in and (step - burn_in) % reject_every == 0:
        return True
    return converged and last_refresh_changed != 0


def refresh_rejection(chi2_core, f_now, st, scene: Scene, *, tau_drop, tau_keep,
                      window, max_churn):
    """Schmitt-trigger whole-star gate; returns (new_state, stats)."""
    role = scene.role
    tmag = scene.z["tess_mag"].astype(np.float64)
    scored = role != ROLE_NUISANCE
    chi2 = np.asarray(chi2_core, dtype=np.float64)
    usable = scored & np.isfinite(chi2) & (chi2 > 0)
    _, dev = static_rank_outliers(np.log(np.maximum(chi2, 1e-6)), tmag,
                                  tau=np.inf, window=window, usable=usable)
    rejected = np.asarray(st["star_free"]) < 0.5
    want = rejected.copy()
    want[usable & ~rejected & (dev > tau_drop)] = True
    want[usable & rejected & (dev < tau_keep)] = False
    change = np.flatnonzero(want != rejected)
    cap = max(1, int(max_churn * usable.sum()))
    if change.size > cap:
        # most decisive changes first
        margin = np.where(want[change], dev[change] - tau_drop, tau_keep - dev[change])
        change = change[np.argsort(-margin)][:cap]
    new_rej = rejected.copy()
    new_rej[change] = ~new_rej[change]
    f_fixed = np.asarray(st["f_fixed"]).copy()
    newly = change[new_rej[change]]
    f_fixed[newly] = np.asarray(f_now)[newly]
    # pixel activity: a rejected star's core pixels leave the likelihood
    uid = scene.z["uid"]
    core = scene.z["core"]
    pix = np.ones(scene.U + 1, np.float32)
    rs = np.flatnonzero(new_rej)
    if rs.size:
        pix[uid[rs][:, core].reshape(-1)] = 0.0
    pix[scene.U] = 0.0
    new_st = dict(st)
    new_st["star_free"] = jnp.asarray(~new_rej, jnp.float32)
    new_st["f_fixed"] = jnp.asarray(f_fixed, jnp.float32)
    new_st["pix_active_u"] = jnp.asarray(pix)
    stats = {
        "n_scored": int(usable.sum()),
        "n_rejected": int(new_rej.sum()),
        "n_changed": int(change.size),
        "n_wanted_changes": int(np.sum(want != rejected)),
        "rej_contrib": int(np.sum(new_rej & (role == ROLE_CONTRIB))),
        "rej_anchor": int(np.sum(new_rej & (role == ROLE_ANCHOR))),
        "med_chi2_core": float(np.median(chi2[usable])) if usable.any() else float("nan"),
        "n_pix_active": int(pix.sum()),
    }
    return new_st, stats


def update_flux_prior(f_now, st, scene: Scene):
    tf = scene.z["tess_flux"].astype(np.float64)
    f = np.asarray(f_now, dtype=np.float64)
    m = (scene.role == ROLE_CONTRIB) & np.isfinite(tf) & (tf > 0) & (f > 0)
    scale = float(np.median(f[m] / tf[m])) if m.any() else 0.0
    new_st = dict(st)
    new_st["f_prior"] = jnp.asarray(np.nan_to_num(scale * tf), jnp.float32)
    return new_st, scale


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def save_state(path: Path, st, extra=None):
    arrs = {k: np.asarray(v) for k, v in st.items()}
    if extra:
        arrs.update({k: np.asarray(v) for k, v in extra.items()})
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


# Chosen colour model A3 (2026-09-29, docs/COLOUR_MODEL_V2_DECISION_20260929.md): #14 (raw-P gauge + dil_r) plus a
# shift quadratic in colour (sq0, sq1) and CCD-frame colour elongation planes (q1_*, q2_*). Pass the Gaia-XP u colour
# with --colour-file where available; without it the colour is BP-RP.
G8_DEFAULT_GAUGE = "raw"
G8_DEFAULT_EXTRAS = "dil_r,sq0,sq1,q1_0,q1_x,q1_y,q2_0,q2_x,q2_y"
# Colour model assumed for a warm-start params file with no fit_meta.json beside it: #14, the default from
# 2026-09-24 until A3 (every run since writes its extras to fit_meta, so this only covers older files).
G8_NO_META_EXTRAS = "dil_r"


def resolve_g8_defaults(args, out: Path) -> None:
    """Fill unset --chroma-g8-gauge/--chroma-g8-extras/--fine-nbr-mode in place.

    Explicit flags always win. A --resume of an existing run keeps the model it was
    started with: values come from its fit_meta.json, where a missing key means the
    pre-2026-09-24 behaviour (gauge 'mean', no extras); #14 runs record extras 'dil_r'. Otherwise the
    A3 defaults apply.
    """
    meta = {}
    if getattr(args, "resume", False) and (out / "fit_meta.json").exists():
        meta = json.loads((out / "fit_meta.json").read_text())
    if args.chroma_g8_gauge is None:
        args.chroma_g8_gauge = meta.get("chroma_g8_gauge", "mean") if meta else G8_DEFAULT_GAUGE
    if args.chroma_g8_extras is None:
        args.chroma_g8_extras = meta.get("chroma_g8_extras", "") if meta else G8_DEFAULT_EXTRAS
    # radial colour family: a resume keeps the run's drop list and knots
    if getattr(args, "chroma_g8_drop", None) is None:
        args.chroma_g8_drop = (meta.get("chroma_g8_drop") or "") if meta else ""
    if getattr(args, "chroma_g8_freeze", None) is None:
        args.chroma_g8_freeze = (meta.get("chroma_g8_freeze") or "") if meta else ""
    if getattr(args, "chroma_radial_mode", None) is None:
        args.chroma_radial_mode = (meta.get("chroma_radial_mode") or "mult") if meta else "add"   # pre-mode resumes were 'mult'
    if getattr(args, "chroma_coma_knots", None) is None:
        args.chroma_coma_knots = (meta.get("chroma_coma_knots") if meta else None) \
            or ",".join(repr(v) for v in EM.COMA_KNOTS_DEFAULT)
    if getattr(args, "chroma_radial_knots", None) is None:
        args.chroma_radial_knots = (meta.get("chroma_radial_knots") if meta else None) \
            or ",".join(repr(v) for v in EM.RADIAL_KNOTS_DEFAULT)
    # fine-neighbour coupling: a resumed pre-2026-09-29 run (no key) keeps "plain"
    if args.fine_nbr_mode is None:
        args.fine_nbr_mode = meta.get("fine_nbr_mode", "plain") if meta else L.FINE_NBR_MODE_DEFAULT
    if getattr(args, "colour_file", None) is None:
        args.colour_file = meta.get("colour_file")
    if getattr(args, "stamp_bg", "none") is None:
        args.stamp_bg = (meta.get("stamp_bg") if meta else None) or "none"
    if getattr(args, "bg_cheb_order", -1) is None:
        m_bg = meta.get("bg_cheb_order") if meta else None
        args.bg_cheb_order = -1 if m_bg is None else int(m_bg)
    # brightness width: a resume keeps the run's form/trainability; new runs default to off
    if getattr(args, "bright_width", "none") is None:
        args.bright_width = meta.get("bright_width") or "none"
    if getattr(args, "bright_width_train", True) is None:
        args.bright_width_train = meta.get("bright_width_train", True)
        if args.bright_width_train is None:
            args.bright_width_train = True


G8_BASE_NAMES = ("s0", "s1", "s2", "k0", "k1", "b", "eps", "t")


def g8_freeze_mask(freeze, extras) -> np.ndarray | None:
    """Boolean (8 + len(extras),) mask of the chroma_g8 slots named in ``freeze`` (base names s0,s1,s2,k0,k1,b,eps,t
    or any extras name), or None when nothing is frozen. Unknown names raise."""
    names = [v.strip() for v in (freeze.split(",") if isinstance(freeze, str) else freeze) if v.strip()]
    if not names:
        return None
    slots = list(G8_BASE_NAMES) + list(extras)
    bad = [n for n in names if n not in slots]
    if bad:
        raise ValueError(f"--chroma-g8-freeze names {bad} are not chroma_g8 slots {slots}")
    m = np.zeros(len(slots), bool)
    for n in names:
        m[slots.index(n)] = True
    return m


def freeze_g8_slots(params: dict, mask) -> dict:
    """stop_gradient on the masked chroma_g8 slots: their gradient is exactly zero, so Adam (fresh state per stage)
    never moves them and they stay bit-identical to the warm start."""
    if mask is None or "chroma_g8" not in params:
        return params
    c = params["chroma_g8"]
    return dict(params, chroma_g8=jnp.where(jnp.asarray(mask), jax.lax.stop_gradient(c), c))


def meta_colour_kwargs(meta: dict) -> dict:
    """make_model / Scene.contexts keywords for the radial-family colour options recorded in a fit_meta.json
    (older fits have neither key: nothing dropped, default knots)."""
    return dict(chroma_g8_drop=meta.get("chroma_g8_drop") or "",
                chroma_radial_knots=meta.get("chroma_radial_knots") or None,
                chroma_radial_mode=meta.get("chroma_radial_mode") or None,
                chroma_coma_knots=meta.get("chroma_coma_knots") or None)


def g8_extras_tuple(extras: str, blur_order: int = 0, no_dil: bool = False) -> tuple:
    """The global8 extras a run actually uses, from its --chroma-g8-* flags."""
    ex = tuple(s.strip() for s in extras.split(",") if s.strip())
    ex += tuple(e for e in ("blur_r", "blur_r2")[:blur_order] if e not in ex)
    if no_dil:
        ex = tuple(e for e in ex if e != "dil_r")   # no dilation -> no radial dilation
    return ex


def warm_start_source(args) -> dict | None:
    """Colour model the warm-start params were trained with.

    Read from the ``fit_meta.json`` beside ``--init-params-file`` (else in ``--init-from``);
    a meta without g8 keys is a pre-2026-09-24 run (gauge 'mean', no extras); without a
    meta the #14 defaults are assumed.
    """
    if args.init_params_file:
        params_path = Path(args.init_params_file)
        meta_path = params_path.parent / "fit_meta.json"
    elif args.init_from:
        params_path = Path(args.init_from) / "params.npz"
        meta_path = Path(args.init_from) / "fit_meta.json"
    else:
        return None
    if not meta_path.exists():
        return {"params": str(params_path), "fit_meta": None, "gauge": G8_DEFAULT_GAUGE,
                "extras": g8_extras_tuple(G8_NO_META_EXTRAS)}
    m = json.loads(meta_path.read_text())
    return {"params": str(params_path), "fit_meta": str(meta_path),
            "gauge": m.get("chroma_g8_gauge", "mean"),
            "radial_knots": m.get("chroma_radial_knots"), "radial_mode": m.get("chroma_radial_mode"), "coma_knots": m.get("chroma_coma_knots"),
            "extras": g8_extras_tuple(m.get("chroma_g8_extras", ""),
                                      int(m.get("chroma_g8_blur_order", 0)),
                                      bool(m.get("chroma_g8_no_dil", False)))}


def _flux_solved_beside(args) -> Path | None:
    """flux_solved.npz of the warm-start fit (--init-from dir, else beside --init-params-file)."""
    for d in ([Path(args.init_from)] if args.init_from else []) + (
            [Path(args.init_params_file).parent] if args.init_params_file else []):
        if (d / "flux_solved.npz").exists():
            return d / "flux_solved.npz"
    return None


def bright_width_q(scene: Scene, params, st, render_and_solve, args):
    """Per-scene-star peak charge per 2-s read at setup, and its provenance.

    Peak fraction: max pixel of the star's unit-flux template rendered with the STARTING
    params (without any bright_width leaf). Flux (e-/s), in order of preference:
    the warm start's flux_solved.npz 'flux' (same scene stars); else, with a warm start,
    the exact island solve under the warm-start params (= the warm-start fluxes); else the
    catalogue (15000 e-/s at Tmag 10), with a warning. Stars with a non-finite flux use the
    catalogue value. q is capped at ``args.bright_width_qmax`` (default 2e5 e-, ~the TESS
    pixel full well: beyond it charge bleeds and the linear law, measured to q ~ 6e4, has no
    meaning; 0 = no cap). Returns (q, flux, peak_fraction, meta).
    """
    p0 = {k: v for k, v in params.items() if k not in BW.BRIGHT_WIDTH_LEAVES}
    T, f_solve = jax.jit(render_and_solve)(p0, st)
    peak = np.asarray(jnp.max(T, axis=1), np.float64)
    f_cat = BW.catalogue_flux(scene.z["tess_mag"])
    src = None
    fpath = _flux_solved_beside(args)
    if fpath is not None:
        z = np.load(fpath)
        if "flux" in z.files and ("source_id" not in z.files
                                   or np.array_equal(z["source_id"], scene.z["source_id"])):
            flux, src = np.asarray(z["flux"], np.float64), f"flux_solved:{fpath}"
        else:
            _log(f"WARNING: {fpath} has no flux or other scene stars; not used for bright_width q")
    if src is None and (args.init_from or args.init_params_file):
        # same two-pass order as run(): the nuisance ridge needs the flux prior set first
        st2, _ = update_flux_prior(f_solve, st, scene)
        f_solve = jax.jit(render_and_solve)(p0, st2)[1]
        flux, src = np.asarray(f_solve, np.float64), "warm_start_initial_solve"
    if src is None:
        flux, src = f_cat, "catalogue"
        _log("WARNING: bright_width q from CATALOGUE fluxes (no warm start): 15000 e-/s at Tmag 10. "
             "Warm-start from a fit (--init-from / --init-params-file) for data fluxes.")
    bad = ~np.isfinite(flux)
    flux = np.where(bad, f_cat, flux)
    q = BW.peak_charge(flux, peak)
    q = np.where(np.isfinite(q), q, 0.0)
    qmax = float(getattr(args, "bright_width_qmax", BW.Q_MAX_DEFAULT) or 0.0)
    n_capped = int(np.sum(q > qmax)) if qmax > 0 else 0
    q_uncapped_max = float(q.max())
    if qmax > 0:
        q = np.minimum(q, qmax)
    meta = {"q_source": src, "n_catalogue_fallback": int(bad.sum()),
            "q_cap": qmax, "n_capped": n_capped, "q_uncapped_max": q_uncapped_max,
            "q_median": float(np.median(q)), "q_p99": float(np.percentile(q, 99)),
            "q_max": float(q.max()), "peak_fraction_median": float(np.median(peak))}
    return q, flux, peak, meta


def bright_width_q_ref(scene: Scene, q, flux) -> float:
    """Loss-weighted mean q of the ePSF contributors (weight ~S/N^2 = f^2 / median core var)."""
    core = np.asarray(scene.z["core"], dtype=bool)
    nz = np.asarray(scene.z["noise"], dtype=np.float64)[:, core]
    s2 = np.nanmedian(nz ** 2, axis=1)
    w = np.where(s2 > 0, np.asarray(flux, np.float64) ** 2 / np.maximum(s2, 1e-12), 0.0)
    return BW.weighted_q_ref(q, w, scene.role == ROLE_CONTRIB)


def run(args):
    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    _log(f"jax backend: {jax.default_backend()} devices={jax.devices()} "
         f"cpus={os.cpu_count()}")
    # EXPERIMENT options that change the model itself are exported to the environment, so every consumer that
    # rebuilds this model (crossfit scorers, raster scripts) decodes it identically when run with the same env
    if getattr(args, "epsf_nodes", 0):
        os.environ["SYNDIFF_EPSF_NODES"] = str(int(args.epsf_nodes))
    if getattr(args, "local_poly_hard", 0):
        L.set_local_poly_hard(int(args.local_poly_hard))
    scene = Scene(Path(args.scene_dir))
    if getattr(args, "anchors_train_epsf", False):
        # option 3 (training_fixes 2026-10-06): faint WCS anchors also train the ePSF shape (no anchor barrier)
        n_a = int(np.sum(scene.role == ROLE_ANCHOR))
        scene.role = np.where(scene.role == ROLE_ANCHOR, ROLE_CONTRIB, scene.role).astype(scene.role.dtype)
        _log(f"anchors_train_epsf: {n_a} anchors relabelled as ePSF contributors")
    resolve_g8_defaults(args, out)
    g8_extras = g8_extras_tuple(args.chroma_g8_extras, args.chroma_g8_blur_order, args.chroma_g8_no_dil)
    args.chroma_g8_extras = ",".join(g8_extras)                     # fit_meta records what was used
    bad_extras = [e for e in g8_extras if not L.is_valid_g8_extra(e)]
    if bad_extras:
        raise ValueError(f"unknown --chroma-g8-extras {bad_extras}")
    g8_drop = L.g8_drop_tuple(args.chroma_g8_drop)
    args.chroma_g8_drop = ",".join(g8_drop)                         # fit_meta records what was used
    g8_freeze = g8_freeze_mask(args.chroma_g8_freeze, g8_extras)
    args.chroma_g8_freeze = ",".join(n.strip() for n in args.chroma_g8_freeze.split(",") if n.strip())
    radial_knots = EM.set_radial_knots(args.chroma_radial_knots)    # trace-time constant for rb*/rq*/rc* extras
    args.chroma_radial_knots = ",".join(repr(v) for v in radial_knots)
    coma_knots = EM.set_coma_knots(args.chroma_coma_knots)
    args.chroma_coma_knots = ",".join(repr(v) for v in coma_knots)
    radial_mode = EM.set_radial_mode(args.chroma_radial_mode)       # recorded in fit_meta; a warm start must match
    if any(L.parse_radial_extra(e) for e in g8_extras):
        bad_j = [e for e in g8_extras if L.parse_radial_extra(e) and L.parse_radial_extra(e)[1] >
                 (EM.n_coma_basis() if L.parse_radial_extra(e)[0] in ("rc", "rcq") else EM.n_radial_basis())]
        if bad_j:
            raise ValueError(f"extras {bad_j} exceed the {EM.n_radial_basis()} radial / {EM.n_coma_basis()} coma "
                             f"bumps of knots {radial_knots} / {coma_knots}")
        if args.chroma_g8_gauge != "raw":
            raise ValueError("the radial colour family needs --chroma-g8-gauge raw")
    if args.colour_file:
        colour_counts = scene.use_colour_file(args.colour_file)
    else:
        c = scene.colour[scene.z["star_bundle_index"]]
        colour_counts = {"bp_rp": int(np.isfinite(c).sum()), "no_colour": int((~np.isfinite(c)).sum())}
        if args.chroma_model == "global8" and args.colour_file is None:
            # A3 was chosen with the Gaia-XP u colour; BP-RP is its fallback, not its input
            _log("WARNING: no --colour-file: the colour model runs on Gaia BP-RP, not the XP u colour "
                 "the chosen model (A3) was validated with. Pass --colour-file <ccd>_u.csv, or "
                 "--colour-file '' to use BP-RP on purpose.")
    _log(f"colour source {scene.colour_source}: {colour_counts}")
    colour_ref = (scene.colour_ref_per_population() if args.colour_gauge == "per_population"
                  else scene.colour_ref())
    delta2_mean = scene.delta2_mean(colour_ref)
    chroma_axis = None
    if args.chroma_model == "global8":
        chroma_axis = (scene.optical_axis() if args.chroma_axis == "tesspoint"
                       else tuple(float(v) for v in args.chroma_axis.split(",")))
        _log(f"chroma_g8 optical axis (science px): ({chroma_axis[0]:.2f}, {chroma_axis[1]:.2f})")
    _log(f"scene: N={scene.N} U={scene.U} stamp={scene.S} roles={scene.meta['n_roles']} "
         f"islands={scene.meta['n_islands']} max_island={scene.meta['max_island']} "
         f"colour_ref={colour_ref} <delta^2>={delta2_mean}")
    model_kw = dict(
        colour_ref=colour_ref, huber_delta=args.huber_delta, ridge=args.ridge,
        prior_kappa=args.prior_kappa, lambda_lap=args.lambda_lap,
        lambda_pixel=args.lambda_pixel_lap, chroma_axis=chroma_axis,
        lambda_fine_nbr=args.lambda_fine_nbr, fine_nbr_mode=args.fine_nbr_mode,
        fine_nbr_basis_sigma=args.fine_nbr_basis_sigma,
        lambda_local_poly=args.lambda_local_poly, local_poly_window=args.local_poly_window,
        local_poly_orders=tuple(int(v) for v in args.local_poly_orders.split(",")),
        local_poly_radii=tuple(float(v) for v in args.local_poly_radii.split(",")),
        chroma_g8_gauge=args.chroma_g8_gauge,
        chroma_g8_no_dil=args.chroma_g8_no_dil, chroma_g8_extras=g8_extras, delta2_mean=delta2_mean,
        chroma_g8_drop=g8_drop, chroma_radial_knots=radial_knots, chroma_radial_mode=radial_mode, chroma_coma_knots=coma_knots,
        bg_cheb_order=args.bg_cheb_order,
        stamp_bg=args.stamp_bg, stamp_bg_r_in=args.stamp_bg_r_in, stamp_bg_r_out=args.stamp_bg_r_out,
        stamp_bg_clip=args.stamp_bg_clip, stamp_bg_prior_sigma=args.stamp_bg_prior_sigma,
        stamp_bg_nb_frac=args.stamp_bg_nb_frac, stamp_bg_nb_radius=args.stamp_bg_nb_radius,
        penalty_nref=args.penalty_nref,
    )
    loss_fn, diagnose = make_model(scene, **model_kw)
    if args.bg_cheb_order >= 0:
        _log(f"background: Chebyshev total degree <= {args.bg_cheb_order}, terms (x,y deg) "
             f"{bg_terms(args.bg_cheb_order)}, u=(x-{BG_CENTRE})/{BG_SCALE}, solved exactly per step")
    sb_meta = {"mode": args.stamp_bg}
    if args.stamp_bg != "none":
        sb_est, _ = scene_stamp_bg(scene, r_in=args.stamp_bg_r_in, r_out=args.stamp_bg_r_out,
                                   clip=args.stamp_bg_clip, nb_frac=args.stamp_bg_nb_frac,
                                   nb_radius=args.stamp_bg_nb_radius)
        sb_meta.update(SB.summary(sb_est), cells="nearest stamp centre (stamp_bg.cell_owner)",
                       file=str((out / "stamp_bg.npz").resolve()))
        np.savez(out / "stamp_bg.npz", source_id=scene.z["source_id"], **sb_est)
        _log(f"stamp background '{args.stamp_bg}': annulus {args.stamp_bg_r_in}-{args.stamp_bg_r_out} px, "
             f"clip {args.stamp_bg_clip} sigma: {SB.summary(sb_est)}"
             + (f"; prior sigma {'3 x SE' if args.stamp_bg_prior_sigma is None else args.stamp_bg_prior_sigma}"
                if args.stamp_bg == "fit" else ""))
    params = init_params(scene)
    pix = np.ones(scene.U + 1, np.float32)
    pix[scene.U] = 0.0
    st = {
        "pix_active_u": jnp.asarray(pix),
        "star_free": jnp.ones((scene.N,), jnp.float32),
        "f_fixed": jnp.zeros((scene.N,), jnp.float32),
        "f_prior": jnp.zeros((scene.N,), jnp.float32),
    }
    start_stage, start_step = 1, 0
    if args.init_from:
        # warm start from another fit dir's final params + rejection state (a fresh run
        # dir; pair with --steps-per-stage 0,0,N to continue only stage 3)
        src = Path(args.init_from)
        params = FIT.load_params_npz(src / "params.npz")
        sz = np.load(src / "flux_solved.npz")
        st = {k: jnp.asarray(sz[k]) for k in st}
        _log(f"initialised from {src} ({int((np.asarray(st['star_free']) == 0).sum())} "
             f"stars rejected)")
    if args.init_params_file:
        params = coarsen_params(FIT.load_params_npz(Path(args.init_params_file)), scene)
        _log(f"params from {args.init_params_file}")
    if args.init_state_file:
        sz = np.load(args.init_state_file)
        st = {k: jnp.asarray(sz[k]) for k in st}
        _log(f"state from {args.init_state_file} "
             f"({int((np.asarray(st['star_free']) == 0).sum())} stars rejected)")
    g8_init = None if not args.chroma_g8_init else [float(v) for v in args.chroma_g8_init.split(",")]
    g8_warm = None
    if args.chroma_model == "global8":
        if g8_init is not None:
            g8_warm = "--chroma-g8-init"
        elif "chroma_g8" in params:
            g8_warm = warm_start_source(args)
            if g8_warm["gauge"] != args.chroma_g8_gauge:
                raise ValueError(f"warm start {g8_warm['params']} has chroma_g8 gauge "
                                 f"{g8_warm['gauge']!r}, this run asks for {args.chroma_g8_gauge!r}")
            g8_warm["padded"] = list(g8_extras[len(g8_warm["extras"]):])
            wk = g8_warm.get("radial_knots")
            if any(L.parse_radial_extra(e) for e in g8_warm["extras"]) or wk:
                if wk is None or EM.parse_radial_knots(wk) != radial_knots:
                    raise ValueError(f"warm start {g8_warm['params']} was trained with radial knots {wk}, this run "
                                     f"uses {radial_knots}; pass --chroma-radial-knots {wk}")
                wc = g8_warm.get("coma_knots")
                if any(L.parse_radial_extra(e) for e in g8_warm["extras"]) and (
                        wc is None or EM.parse_radial_knots(wc) != coma_knots):
                    raise ValueError(f"warm start {g8_warm['params']} was trained with coma knots {wc}, this run "
                                     f"uses {coma_knots}; pass --chroma-coma-knots {wc}")
                wm = g8_warm.get("radial_mode")
                if any(L.parse_radial_extra(e) for e in g8_warm["extras"]) and (wm or "mult") != radial_mode:
                    raise ValueError(f"warm start {g8_warm['params']} was trained with --chroma-radial-mode {wm or 'mult'}, "
                                     f"this run uses {radial_mode}")
    params = set_chroma_model(params, args.chroma_model, halo=args.chroma_halo, g8_init=g8_init,
                              g8_extras=g8_extras,
                              g8_source_extras=g8_warm["extras"] if isinstance(g8_warm, dict) else None)
    if isinstance(g8_warm, dict):
        _log(f"chroma_g8 kept from {g8_warm['params']} (extras {g8_warm['extras']}, zero-padded "
             f"{g8_warm['padded']}): {np.round(np.asarray(params['chroma_g8']), 6).tolist()}")
    _log(f"colour model: {args.chroma_model}{' + halo' if args.chroma_halo else ''}; "
         f"colour gauge {args.colour_gauge}; leaves {sorted(k for k in params if k.startswith('chroma'))}")
    prog_path = out / "progress.json"
    if args.resume and prog_path.exists():
        prog = json.loads(prog_path.read_text())
        params = FIT.load_params_npz(out / "params_latest.npz")
        sz = np.load(out / "state_latest.npz")
        st = {k: jnp.asarray(sz[k]) for k in st}
        start_stage, start_step = int(prog["stage"]), int(prog["step"])
        _log(f"resumed from stage {start_stage} step {start_step} (Adam state reset)")

    # Brightness-dependent width (bright_width.py): static per-star q from the starting
    # params, then the model is rebuilt with q on its contexts.
    bw_meta = {"form": args.bright_width}
    if args.bright_width != "none":
        resumed = args.resume and prog_path.exists()
        init_leaf = (None if (resumed or args.bright_width_init is None)
                     else args.bright_width_init / BW.LEAF_UNIT)
        params, init_src = BW.set_bright_width(params, args.bright_width, init_leaf)
        qfile = (out / "bright_width_q.npz").resolve()
        if resumed and qfile.exists():
            zq = np.load(qfile)
            q, q_ref = np.asarray(zq["q"], np.float64), float(zq["q_ref"])
            qm = json.loads(str(zq["meta"]))
        else:
            _, _, render_and_solve = make_model(scene, **model_kw, return_templates=True)
            q, flux_q, peak_q, qm = bright_width_q(scene, params, st, render_and_solve, args)
            q_ref = (float(args.bright_width_qref) if args.bright_width_qref is not None
                     else bright_width_q_ref(scene, q, flux_q))
            qm["q_ref_source"] = "cli" if args.bright_width_qref is not None else \
                "loss_weighted_mean_contributors"
            np.savez(qfile, source_id=scene.z["source_id"], q=q, flux=flux_q, peak_fraction=peak_q,
                     q_ref=q_ref, meta=json.dumps(qm))
        loss_fn, diagnose = make_model(scene, **model_kw, bright_width=args.bright_width,
                                       bright_width_q_file=str(qfile), bright_width_q_ref=q_ref)
        bw_meta.update(
            qm, q_ref=q_ref, trainable=bool(args.bright_width_train), init_source=init_src,
            b_init=BW.leaf_to_b(params["bright_width"]),
            units="b = DeltaSigma (extra blur covariance per axis, px^2) per 1e4 e- per 2-s read",
            leaf_unit=BW.LEAF_UNIT, gen_per_dsigma=BW.GEN_PER_DSIGMA,
            generator=BW.generator_for_gauge(args.chroma_g8_gauge),
            q_file=str(qfile))
        _log(f"bright_width: b_init={bw_meta['b_init']:.4g} px^2/1e4e- ({init_src}), "
             f"q_ref={q_ref:.4g} e- ({bw_meta['q_ref_source'] if 'q_ref_source' in bw_meta else 'resumed'}), "
             f"q source {qm['q_source']}, q median/p99/max {qm['q_median']:.3g}/{qm['q_p99']:.3g}/"
             f"{qm['q_max']:.3g}, {'trainable' if args.bright_width_train else 'FIXED'}, "
             f"generator {bw_meta['generator']}")
    elif BW.has_bright_width(params):
        params, _ = BW.set_bright_width(params, "none", None)
        _log("bright_width: warm-start leaf dropped (--bright-width none)")

    star_w = None
    if getattr(args, "star_weight_cap_tmag", None) is not None:
        # option D (training_fixes 2026-10-06): cap each star's total weight at that of a T = cap star by dividing
        # its stamp's inverse variance by (f_i / f_cap)^2 for stars brighter than the cap; the weighting INSIDE a
        # star (core vs wing) is unchanged, unlike the per-pixel floor
        tm = np.asarray(scene.z["tess_mag"], np.float64)
        ratio2 = 10.0 ** (-0.8 * (tm - args.star_weight_cap_tmag))          # (f / f_cap)^2
        if getattr(args, "star_weight_cap_form", "hard") == "smooth":
            star_w = np.where(np.isfinite(tm), 1.0 / (1.0 + ratio2), 1.0)       # -> 1 faint, (f_cap/f)^2 bright
        else:
            star_w = np.where(np.isfinite(tm), np.minimum(1.0, 1.0 / ratio2), 1.0)
        # one weight per PHYSICAL pixel (its owner star's), identical in every stamp that contains it: per-stamp
        # weights make the island solve's normal equations inconsistent for overlapping stamps
        uid_ = np.asarray(scene.z["uid"]); own_ = np.asarray(scene.z["owner"], bool)
        w_u = np.ones(int(uid_.max()) + 1, np.float64)
        w_u[uid_[own_]] = np.broadcast_to(star_w[:, None], uid_.shape)[own_]
        star_w = w_u[uid_]                                             # (N, S2) per-pixel weights
        _log(f"star weight cap at T={args.star_weight_cap_tmag}: {int(np.sum(np.isfinite(tm) & (tm < args.star_weight_cap_tmag)))} "
             f"stars down-weighted, min pixel weight {star_w.min():.3g}")
    if star_w is not None and not ((args.model_floor_eps > 0 or args.noise_scale != 1.0)
                                   and args.floor_source == "data"):
        v0 = np.asarray(L.pixel_variance(jnp.asarray(scene.z["noise"])), np.float64)
        st = dict(st, var_eff=jnp.asarray(v0 / star_w, jnp.float32))
    if (args.model_floor_eps > 0 or args.noise_scale != 1.0) and args.floor_source == "data":
        # data-based floor: computed ONCE from the measured pixels, so the weights never feed back on the fluxes
        # (the model-based refresh oscillates even at fixed params, training_fixes_20261005 irls_fixedparams)
        v0 = np.asarray(L.pixel_variance(jnp.asarray(scene.z["noise"])), np.float64)
        d0 = np.clip(np.asarray(scene.z["data"], np.float64), 0.0, None)
        ve0 = args.noise_scale ** 2 * v0 + (args.model_floor_eps * d0) ** 2
        if star_w is not None:
            ve0 = ve0 / star_w
        st = dict(st, var_eff=jnp.asarray(ve0, jnp.float32))
        okv = np.asarray(scene.z["valid"]) & np.isfinite(ve0)
        _log(f"data-based floor eps={args.model_floor_eps} noise_scale={args.noise_scale}: median sigma_eff/sigma "
             f"{np.median(np.sqrt(ve0[okv] / v0[okv])):.4f}, p99 {np.percentile(np.sqrt(ve0[okv] / v0[okv]), 99):.3f}")
    diag_j = jax.jit(diagnose)
    t0 = time.time()
    f0, chi2_0, _ = diag_j(params, st)
    f0.block_until_ready()
    _log(f"initial solve+diagnose (incl. compile) {time.time() - t0:.1f}s")
    st, scale = update_flux_prior(f0, st, scene)
    _log(f"flux prior scale (solved/tess_flux, contributors) = {scale:.4g}")

    keys = tuple(params.keys())
    hist_path = out / "history.jsonl"
    run_meta = dict(vars(args), colour_ref=colour_ref, colour_source=scene.colour_source,
                    colour_counts=colour_counts, chroma_delta2_mean=delta2_mean,
                    g8_warm_start=g8_warm, chroma_axis=chroma_axis, scene_meta=scene.meta,
                    bright_width_model=bw_meta, stamp_bg_model=sb_meta,
                    # make_model keywords, recorded under their own names (see make_model)
                    bright_width_q_file=bw_meta.get("q_file"), bright_width_q_ref=bw_meta.get("q_ref", 0.0),
                    bright_width_gen_per_dsigma=BW.GEN_PER_DSIGMA,
                    bg_terms=bg_terms(args.bg_cheb_order) if args.bg_cheb_order >= 0 else [],
                    bg_centre=BG_CENTRE, bg_scale=BG_SCALE,
                    started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / "fit_meta.json").write_text(json.dumps(run_meta, indent=1, default=str))

    steps = [int(s) for s in args.steps_per_stage.split(",")]
    lrs = [float(s) for s in args.lr_per_stage.split(",")]
    global_step = 0
    smoothed = args.stop_rule == "smoothed"
    # use_floor: the MODEL-based floor, refreshed during training. The data-based floor is fixed in st already.
    use_floor = (args.model_floor_eps > 0 or args.noise_scale != 1.0) and args.floor_source == "model"
    if use_floor or smoothed:
        var0 = np.asarray(L.pixel_variance(jnp.asarray(scene.z["noise"])), np.float64)
        model_j = jax.jit(diagnose.model)
        lossval_j = jax.jit(lambda p_, s_: loss_fn(p_, s_)[0])
        pos_j = jax.jit(diagnose.positions)
        free_mask = np.asarray(scene.role) == ROLE_CONTRIB
        _log(f"stop rule {args.stop_rule}; floor eps={args.model_floor_eps} noise_scale={args.noise_scale}; "
             f"penalty_nref={args.penalty_nref} (factor {diagnose.pen_fac:.6g}, N_pix {diagnose.n_pix_static:.0f})")
    stage_outcomes = []

    def refresh_floor(p, s):
        """var_eff = (s sigma)^2 + (eps m)^2 from the current model. Returns (new state, relative loss jump
        at fixed params, median sigma_eff change on floor-dominated pixels, median sigma_eff/sigma)."""
        m = np.asarray(model_j(p, s), np.float64)
        ve = args.noise_scale ** 2 * var0 + (args.model_floor_eps * m) ** 2
        prev = np.asarray(s["var_eff"], np.float64) if "var_eff" in s else None
        new_s = dict(s)
        new_s["var_eff"] = jnp.asarray(ve, jnp.float32)
        ok = np.asarray(scene.z["valid"]) & np.isfinite(ve) & (ve > 0)
        # judge the change where the floor matters (floor term > scaled noise term); sky pixels barely move
        fl = ok & ((args.model_floor_eps * m) ** 2 > args.noise_scale ** 2 * var0)
        sel = fl if fl.sum() >= 100 else ok
        chg = (float(np.median(np.abs(np.sqrt(ve[sel] / prev[sel]) - 1.0))) if prev is not None
               else float("inf"))
        l_old, l_new = float(lossval_j(p, s)), float(lossval_j(p, new_s))
        jump = abs(l_new - l_old) / max(abs(l_old), 1e-12) if prev is not None else float("inf")
        return new_s, jump, chg, float(np.median(np.sqrt(ve[sel] / var0[sel])))

    def hist_event(rec):
        with hist_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        _log(f"  event: {rec}")

    for stage in range(start_stage, 4):
        n_steps = steps[stage - 1]
        if n_steps <= 0:
            continue
        labels = FIT._leaf_labels(stage, freeze_wcs=bool(args.freeze_wcs), param_keys=keys)
        if "bright_width" in labels and not args.bright_width_train:
            # fixed b: stop-gradient here, so its bucket's Adam update is exactly zero
            labels["bright_width"] = "frozen"
        opt = FIT.make_stage_optimizer(stage, lrs[stage - 1], epsf_lr_scale=args.epsf_lr_scale,
                                       param_keys=keys, chroma_lr_scale=args.chroma_lr_scale,
                                       grad_clip=args.grad_clip, freeze_wcs=bool(args.freeze_wcs))
        opt_state = opt.init(params)

        def stage_loss(p, s, _labels=labels):
            return loss_fn(freeze_g8_slots(FIT.stop_grad_frozen_params(p, _labels), g8_freeze), s)

        # floor: refreshed every --floor-refresh-every steps (smoothed rule) as part of the iteration. Each refresh
        # changes the loss scale; its jump at fixed params is subtracted from the stored history so the plateau
        # test compares like with like. A stage only ends when the last refresh moved the loss < --floor-settle-tol.
        last_jump = float("inf")
        if use_floor:
            st, jump, chg, med = refresh_floor(params, st)
            hist_event({"event": "floor_refresh", "stage": stage, "step": 0, "median_sigma_eff_over_sigma": med})

        buckets = sorted({v for k, v in labels.items() if k in keys and v != "frozen"})

        @jax.jit
        def step_fn_smoothed(p, o, s, freeze_wcs, lr_mult):
            (lv, met), g = jax.value_and_grad(stage_loss, has_aux=True)(p, s)
            g = dict(g)
            g["wcs_coeff"] = g["wcs_coeff"] * (1.0 - freeze_wcs)
            upd, o = opt.update(g, o, p)
            # Adam's update is linear in its learning rate, so scaling the update IS the lr decay
            upd = jax.tree_util.tree_map(lambda u: u * lr_mult, upd)
            p = optax.apply_updates(p, upd)
            gn = optax.global_norm(g)
            gb = {bk: optax.global_norm({k: g[k] for k in g if labels.get(k) == bk}) for bk in buckets}
            return p, o, met, gn, gb

        @jax.jit
        def step_fn(p, o, s, freeze_wcs):
            (lv, met), g = jax.value_and_grad(stage_loss, has_aux=True)(p, s)
            g = dict(g)
            g["wcs_coeff"] = g["wcs_coeff"] * (1.0 - freeze_wcs)
            upd, o = opt.update(g, o, p)
            p = optax.apply_updates(p, upd)
            gn = optax.global_norm(g)
            return p, o, met, gn

        _log(f"=== stage {stage}: {n_steps} steps, lr {lrs[stage - 1]}, trainable "
             f"{sorted(k for k, v in labels.items() if v != 'frozen' and k in keys)}")
        losses: list[float] = []
        last_refresh_changed = None
        lr_mult, n_decay, reason = 1.0, 0, "cap"
        win: list[float] = []          # smoothed rule: losses since the last restart
        snaps: dict[int, tuple] = {}   # smoothed rule: step-since-restart -> (x, y, flux)
        since = 0
        first = start_step if stage == start_stage else 0
        t_stage = time.time()
        step = first
        while step < n_steps:
            freeze = 1.0 if (stage == 2 and step < args.stage2_freeze_wcs_steps) else 0.0
            ts = time.time()
            if smoothed:
                params, opt_state, met, gn, gb = step_fn_smoothed(params, opt_state, st, freeze, lr_mult)
            else:
                params, opt_state, met, gn = step_fn(params, opt_state, st, freeze)
            lv = float(met["loss"])
            dt = time.time() - ts
            if not np.isfinite(lv):
                raise FloatingPointError(f"non-finite loss at stage {stage} step {step}")
            losses.append(lv)
            rec = {"stage": stage, "step": step, "global_step": global_step, "loss": lv,
                   "data_term": float(met["data_term"]), "lap": float(met["lap"]),
                   "pixel_lap": float(met["pixel_lap"]), "fine_nbr": float(met["fine_nbr"]),
                   "local_poly": float(met["local_poly"]), "n_pix": float(met["n_pix"]),
                   "grad_norm": float(gn), "step_s": dt, "t": time.time()}
            if smoothed:
                rec["grad_norm_bucket"] = {k: float(v) for k, v in gb.items()}
                rec["lr_mult"] = lr_mult
            if "bright_width" in params:
                rec["bright_width_b"] = BW.leaf_to_b(params["bright_width"])
            if args.bg_cheb_order >= 0:
                rec["bg_coef"] = [float(v) for v in np.asarray(met["bg_coef"])]
            if "stamp_bg" in met:
                rec["stamp_bg_median"] = float(np.median(np.asarray(met["stamp_bg"])))
            with hist_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if step % args.log_every == 0:
                _log(f"s{stage} step {step:5d} loss {lv:.6f} data {rec['data_term']:.6f} "
                     f"|g| {float(gn):.3g} {dt:.2f}s/step")
            step += 1
            global_step += 1

            # smoothed stop rule (see --stop-rule): plateau of the w-step running mean over lag K AND
            # physical quantities still -> lr decay, after --lr-decays decays the next plateau ends the stage
            plateau_stop = False
            if smoothed and use_floor and (step - first) % args.floor_refresh_every == 0:
                l_old = float(lossval_j(params, st))
                st, last_jump, chg, med = refresh_floor(params, st)
                offset = float(lossval_j(params, st)) - l_old
                win = [x + offset for x in win]
                lv += offset
                hist_event({"event": "floor_refresh", "stage": stage, "step": step, "loss_jump": last_jump,
                            "median_rel_change": chg, "median_sigma_eff_over_sigma": med})
            if smoothed:
                win.append(lv)
                since += 1
                K, w = args.stop_lag, args.stop_window
                if since % args.stop_check == 0:
                    f_now, _, _ = diag_j(params, st)
                    xs, ys = pos_j(params)
                    snaps[since] = (np.asarray(xs), np.asarray(ys), np.asarray(f_now, np.float64))
                    for kk in [k for k in snaps if k < since - K]:
                        del snaps[kk]
                    if len(win) >= K + w and (since - K) in snaps:
                        lt, lk = float(np.mean(win[-w:])), float(np.mean(win[-K - w:-K]))
                        s_rel = (lk - lt) / max(abs(lt), 1e-12)
                        x0_, y0_, f0_ = snaps[since - K]
                        x1_, y1_, f1_ = snaps[since]
                        dpos = float(np.max(np.hypot(x1_ - x0_, y1_ - y0_))) * 1e3
                        okf = free_mask & np.isfinite(f1_) & np.isfinite(f0_) & (np.abs(f1_) > 0)
                        dflux = float(np.median(np.abs((f1_[okf] - f0_[okf]) / f1_[okf])))
                        hist_event({"event": "stop_check", "stage": stage, "step": step, "s_rel": s_rel,
                                    "dpos_mpx": dpos, "dflux": dflux, "lr_mult": lr_mult})
                        # lr decay on a LOSS plateau alone (ReduceLROnPlateau: a stalled or creeping loss at
                        # full lr is an oscillation the decay damps); the stage ENDS only when the loss is flat,
                        # WCS / fluxes / floor weights are still, and at least --lr-decays decays have happened.
                        loss_flat = s_rel < args.stop_tol
                        floor_ok = (not use_floor) or last_jump < args.floor_settle_tol
                        phys_ok = dpos < args.stop_wcs_mpx and dflux < args.stop_flux_tol and floor_ok
                        if loss_flat and phys_ok and n_decay >= args.lr_decays:
                            plateau_stop, reason = True, "plateau"
                        elif loss_flat and n_decay < max(args.lr_decays_max, args.lr_decays):
                            n_decay += 1
                            lr_mult *= args.lr_decay
                            hist_event({"event": "lr_decay", "stage": stage, "step": step, "lr_mult": lr_mult,
                                        "phys_ok": phys_ok, "floor_ok": floor_ok})
                            win, snaps, since = [], {}, 0

            # rejection refresh
            converged = False
            p_ = args.early_stop_patience
            if not smoothed and len(losses) > p_:
                rel = abs(losses[-1 - p_] - losses[-1]) / max(abs(losses[-1 - p_]), 1e-12)
                converged = rel < args.early_stop_tol
            if refresh_due(stage, step, reject_every=args.reject_every, burn_in=args.reject_burn_in,
                           converged=converged, last_refresh_changed=last_refresh_changed):
                f_now, chi2_core, _ = diag_j(params, st)
                st, rstats = refresh_rejection(
                    chi2_core, f_now, st, scene, tau_drop=args.reject_tau_drop,
                    tau_keep=args.reject_tau_keep, window=args.reject_window,
                    max_churn=args.reject_max_churn)
                st, scale = update_flux_prior(f_now, st, scene)
                last_refresh_changed = rstats["n_changed"]
                rstats.update(stage=stage, step=step, flux_prior_scale=scale, event="refresh")
                with hist_path.open("a") as fh:
                    fh.write(json.dumps(rstats) + "\n")
                _log(f"  refresh: {rstats}")
                if rstats["n_changed"]:
                    losses = []  # the pixel set changed; loss history is not comparable
                    converged = False
            if step % args.checkpoint_every == 0 or step == n_steps or converged or plateau_stop:
                FIT.save_params_npz(out / "params_latest.npz", params)
                save_state(out / "state_latest.npz", st)
                prog_path.write_text(json.dumps({"stage": stage, "step": step}))
                if step % (args.checkpoint_every * 5) == 0:
                    FIT.save_params_npz(out / "checkpoints" / f"params_s{stage}_step{step:05d}.npz",
                                        params)
            if converged:
                reason = "plateau"
                _log(f"  early stop at stage {stage} step {step} "
                     f"(rel change < {args.early_stop_tol} over {p_} steps)")
                break
            if plateau_stop:
                _log(f"  plateau stop at stage {stage} step {step} after {n_decay} lr decays")
                break
        _log(f"stage {stage} done in {(time.time() - t_stage) / 60:.1f} min")
        outcome = {"stage": stage, "steps": step, "cap": n_steps, "reason": reason,
                   "converged": reason == "plateau", "lr_mult": lr_mult, "n_lr_decays": n_decay}
        stage_outcomes.append(outcome)
        if smoothed or use_floor:
            hist_event(dict(outcome, event="stage_end"))
        FIT.save_params_npz(out / f"params_stage{stage}.npz", params)
        save_state(out / f"state_stage{stage}.npz", st)
        prog_path.write_text(json.dumps({"stage": stage + 1, "step": 0}))

    f_fin, chi2_fin, _ = diag_j(params, st)
    FIT.save_params_npz(out / "params.npz", params)
    extra = {
        "flux": f_fin, "chi2_core": chi2_fin, "source_id": scene.z["source_id"],
        "role": scene.role, "tess_mag": scene.z["tess_mag"],
        "x0": scene.z["x0"], "y0": scene.z["y0"],
    }
    if args.bg_cheb_order >= 0 or args.stamp_bg == "fit":
        _, c_fin, b_fin = jax.jit(diagnose.components)(params, st)
    if args.bg_cheb_order >= 0:
        bg_fin = np.asarray(c_fin)
        extra["bg_coef"] = bg_fin
        run_meta["bg_coef_final"] = [float(v) for v in bg_fin]
        _log(f"background coefficients (e-/s): {np.round(bg_fin, 5).tolist()}")
    if args.stamp_bg == "fit":
        b_fin = np.asarray(b_fin)
        extra["stamp_bg"] = b_fin
        run_meta["stamp_bg_model"]["fit_median"] = float(np.median(b_fin))
        run_meta["stamp_bg_model"]["fit_p16_p84"] = [float(v) for v in np.percentile(b_fin, [16, 84])]
        _log(f"stamp background (fit) median {np.median(b_fin):.4g} e-/s")
    save_state(out / "flux_solved.npz", st, extra=extra)
    if "bright_width" in params:
        run_meta["bright_width_model"]["b_final"] = BW.leaf_to_b(params["bright_width"])
    if args.stop_rule == "smoothed" or args.model_floor_eps > 0 or args.noise_scale != 1.0:
        run_meta["stage_outcomes"] = stage_outcomes
    if ("bright_width" in params or args.bg_cheb_order >= 0 or args.stamp_bg != "none"
            or "stage_outcomes" in run_meta):
        (out / "fit_meta.json").write_text(json.dumps(run_meta, indent=1, default=str))
        if "bright_width" in params:
            _log(f"bright_width: b_final={run_meta['bright_width_model']['b_final']:.4g} px^2 per 1e4 e-")
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    _log("done")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default=None,
                   help="fit dir whose params.npz + flux_solved.npz seed this run")
    p.add_argument("--steps-per-stage", default="300,2000,800")
    p.add_argument("--lr-per-stage", default="1e-3,3e-4,1e-4")
    p.add_argument("--epsf-lr-scale", type=float, default=2.5)
    p.add_argument("--chroma-lr-scale", type=float, default=10.0,
                   help="Adam lr multiplier for colour leaves (1.0 is step-limited for global8)")
    p.add_argument("--chroma-model", choices=("nodes", "global8", "none"), default="global8",
                   help="colour model: node fields (shift+dilation, legacy), the global "
                        "8-parameter model, or none")
    p.add_argument("--chroma-halo", action="store_true", help="add the chroma_halo node leaf")
    p.add_argument("--chroma-axis", default="tesspoint",
                   help="optical axis for global8: 'tesspoint' or 'x,y' in science px")
    # Defaults = the chosen colour model A3 (2026-09-29, docs/COLOUR_MODEL_V2_DECISION_20260929.md):
    # raw-P gauge + dil_r + sq0,sq1 + q1_*,q2_*. #14 (2026-09-24) = extras 'dil_r'. Checkpoints written
    # before 2026-09-24 used gauge 'mean' and no extras; their fit_meta.json has no chroma_g8_gauge key.
    p.add_argument("--chroma-g8-gauge", choices=("mean", "raw"), default=None,
                   help="global8 base gauge: 'raw' (default) projects off P itself; 'mean' "
                        "projects off P - mean(P) and leaves a flat colour-dependent sheet "
                        "(the pre-2026-09-24 behaviour)")
    p.add_argument("--chroma-g8-blur-order", type=int, choices=(0, 1, 2), default=0,
                   help="global8 blur as a polynomial in r (distance to the axis, 1000 px): "
                        "0 = global, 1 = b0 + b1 r, 2 = b0 + b1 r + b2 r^2 "
                        "(adds blur_r[,blur_r2] to --chroma-g8-extras)")
    p.add_argument("--chroma-g8-extras", default=None,
                   help="comma list (default A3: 'dil_r,sq0,sq1,q1_0,q1_x,q1_y,q2_0,q2_x,q2_y'; "
                        "'dil_r' = model #14; '' = none) of extra global8 coefficients appended after "
                        "the 8 base slots: blur_r, blur_r2, dil_r (dilation d0 + d1 r), astig0, astig_r "
                        "(astigmatism toward the optical axis), sq0, sq1 (shift quadratic in colour), "
                        "q1_0/x/y, q2_0/x/y (colour elongation planes in CCD X, Y), dilp_x, dilp_y")
    p.add_argument("--chroma-g8-no-dil", action="store_true",
                   help="global8 without the dilation term (blur and dilation are ~95%% collinear)")
    p.add_argument("--chroma-g8-drop", default=None,
                   help="comma list of round A3 colour fields to REMOVE from the render: blur, dil, kurt "
                        "(default none). Their coefficients stay in the vector but are inert (zero gradient), so "
                        "checkpoints keep their layout. Generalises --chroma-g8-no-dil.")
    p.add_argument("--chroma-radial-knots", default=None,
                   help="knots (px, comma list, >= 2) of the warped-coordinate cubic B-spline bumps behind the radial "
                        "colour extras rb{j}_0, rb{j}_r (colour-linear profile, amplitude a0 + a1 r) and rq{j} "
                        "(quadratic in colour): J = len(knots) bumps, j = 1..J, B_j = bspline3(s(rho) - (j-1)) with s "
                        "the piecewise-linear map knots -> 0..J-1 (constant plateau beyond the last knot). Default "
                        "0,0.7,1.5,3.0,5.5. Recorded in fit_meta.json; a warm start with radial extras must match.")
    p.add_argument("--chroma-coma-knots", default=None,
                   help="knots (px, comma list) of the warped-coordinate bumps behind the coma extras rc{j} "
                        "(j = 1..len(knots)); default 0.8,2.2,5.0. Coma is always multiplicative (P0 C_j rho cos/sin "
                        "theta, then Gram-Schmidt off P0, dP0/dx, dP0/dy). Recorded in fit_meta; a warm start with "
                        "radial extras must match.")
    p.add_argument("--chroma-radial-mode", choices=EM.RADIAL_MODES, default=None,
                   help="form of the RADIAL profiles (rb*/rq*): 'add' (default for new runs) = additive B_j(rho) - "
                        "mean(B_j) projected off P0; 'mult' = P0-weighted P0 (B_j - m_j). The coma (rc*) is always "
                        "multiplicative. Recorded in fit_meta.json; a warm start with radial extras must match.")
    p.add_argument("--chroma-g8-freeze", default=None,
                   help="comma list of chroma_g8 slots to hold at their starting (warm-start) values: base names "
                        "s0,s1,s2,k0,k1,b,eps,t and any --chroma-g8-extras name (e.g. sq0,sq1). Their gradient is "
                        "stop_gradient-ed, so Adam never moves them (bit-identical to the warm start). Recorded in fit_meta.")
    p.add_argument("--freeze-wcs", action="store_true",
                   help="hold wcs_coeff fixed in every stage (its leaf is labelled frozen: no gradient, no update). "
                        "Pair with --steps-per-stage 0,0,N to continue a converged fit with the WCS fixed.")
    p.add_argument("--chroma-g8-init", default="",
                   help="starting values [s0,s1,s2,k0,k1,b,eps,t] followed by one value per "
                        "--chroma-g8-extras entry (17 values with the A3 defaults)")
    p.add_argument("--colour-file", default=None,
                   help="CSV source_id,colour (BP-RP units) replacing BP-RP in the colour model; "
                        "unmatched/NaN stars keep BP-RP. --resume reuses the run's file; '' = BP-RP")
    p.add_argument("--colour-gauge", choices=("global", "per_population"), default="global",
                   help="keep 'global': per_population gives same-colour stars different "
                        "chroma shifts by role ((c_ref_contrib - c_ref_anchor) * s(r), ~10 mpx "
                        "radial on S24/S20), which the shared WCS cannot absorb (2026-09-24)")
    p.add_argument("--bright-width", choices=BW.FORMS, default=None,
                   help="brightness-dependent PSF width (bright_width.py): 'lin' = isotropic blur "
                        "DeltaSigma = b (q - q_ref) with q the star's peak e- per 2-s read; default "
                        "'none' (a --resume keeps the run's form)")
    p.add_argument("--bright-width-init", type=float, default=None,
                   help="initial b in px^2 (DeltaSigma per axis) per 1e4 e- per read (bfwidth_20260929: "
                        "0.8e-3); default: the warm start's leaf, else 0")
    p.add_argument("--bright-width-train", action=argparse.BooleanOptionalAction, default=None,
                   help="train b at stage 3 in the colour lr bucket (default) or hold it fixed "
                        "(--no-bright-width-train)")
    p.add_argument("--bright-width-qref", type=float, default=None,
                   help="q_ref in e- per read; default = loss-weighted (~S/N^2) mean q of the ePSF "
                        "contributors, computed at setup and recorded in fit_meta.json")
    p.add_argument("--bright-width-qmax", type=float, default=BW.Q_MAX_DEFAULT,
                   help="cap on q (e- per read), default 2e5 ~ pixel full well (saturated stars); 0 = none")
    p.add_argument("--init-params-file", default=None,
                   help="start from this params npz (e.g. <fit>/params_stage2.npz)")
    p.add_argument("--init-state-file", default=None,
                   help="rejection/flux-prior state npz (e.g. <fit>/state_stage2.npz)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stage2-freeze-wcs-steps", type=int, default=20)
    p.add_argument("--huber-delta", type=float, default=1e6,
                   help="default 1e6 = plain L2 (inf NaNs float32 grads); delta=3 clips bright "
                        "cores and biases the even colour terms (2026-09-23/24)")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--prior-kappa", type=float, default=0.1,
                   help="nuisance-flux ridge toward scale*tess_flux, as a fraction of the "
                        "star's unmasked-equivalent information")
    p.add_argument("--lambda-lap", type=float, default=1e-2)
    p.add_argument("--lambda-pixel-lap", type=float, default=1e-2)
    p.add_argument("--lambda-fine-nbr", type=float, default=1e8,
                   help="fine-part neighbour coupling (pixel-integrated ePSF)")
    p.add_argument("--fine-nbr-mode", choices=L.FINE_NBR_MODES, default=None,
                   help="moment_blind (default for new runs): the coupling ignores node-to-node "
                        "flux/shift/width differences. plain = pre-2026-09-29 coupling, which drags "
                        "core width toward neighbours (2-5 permil optical-axis flux bias). "
                        "A --resume keeps the mode the run started with. moment_blind_pair: "
                        "pair-mean generators; less junk but width-BIASED like plain (reference only)")
    p.add_argument("--fine-nbr-basis-sigma", type=float, default=0.0,
                   help="moment-blind modes: Gaussian-smooth (grid samples) the ePSF the "
                        "low-order generators are taken from; 0 = exact (>0 re-introduces width bias)")
    p.add_argument("--lambda-local-poly", type=float, default=0.0,
                   help="EXPERIMENT: per-node local-polynomial (Anderson & King) smoothness, "
                        "mean (E - Q(E))^2; 0 = off. No coupling between nodes")
    p.add_argument("--local-poly-window", type=int, default=7, help="smoothing window (grid samples, odd)")
    p.add_argument("--local-poly-hard", type=int, default=0,
                   help="EXPERIMENT 'AK hard': the ePSF used is always its per-node local-polynomial-smoothed version "
                        "with this window (5 = Anderson & King 2000 eq. 8, 7); 0 = off. Exported as "
                        "SYNDIFF_EPSF_LOCAL_POLY_HARD: set the same env var when re-evaluating the fit")
    p.add_argument("--epsf-nodes", type=int, default=0,
                   help="EXPERIMENT: n x n ePSF node grid spanning the bundle's outermost nodes (0 = the bundle's grid); "
                        "init params are resampled. Exported as SYNDIFF_EPSF_NODES: set it when re-evaluating the fit")
    p.add_argument("--local-poly-orders", default="4,2,1",
                   help="polynomial order inside / between / beyond --local-poly-radii")
    p.add_argument("--local-poly-radii", default="3,5", help="zone radii in native px")
    p.add_argument("--bg-cheb-order", type=int, default=None,
                   help="smooth background: Chebyshev polynomial of total degree <= this over the CCD "
                        "(u = (x-1024)/1024), solved exactly with the fluxes (-1 = off, the default; "
                        "2 = 6 terms). A --resume keeps the run's order")
    p.add_argument("--stamp-bg", choices=SB.MODES, default=None,
                   help="per-stamp local background (stamp_bg.py): 'annulus' = fixed offset per stamp, a "
                        "clipped median at r_in..r_out px before the fit; 'fit' = free constant per stamp solved "
                        "with the fluxes, Gaussian prior toward the annulus value; default 'none' "
                        "(a --resume keeps the run's mode)")
    p.add_argument("--stamp-bg-r-in", type=float, default=5.0, help="annulus inner radius, px")
    p.add_argument("--stamp-bg-r-out", type=float, default=7.0, help="annulus outer radius, px")
    p.add_argument("--stamp-bg-clip", type=float, default=3.0, help="annulus sigma clip (MAD sigma)")
    p.add_argument("--stamp-bg-prior-sigma", type=float, default=None,
                   help="'fit' prior width, e-/s; default 3 x the annulus estimate's robust standard error; "
                        "<= 0 = no prior (unconstrained least-squares pedestal)")
    p.add_argument("--stamp-bg-nb-frac", type=float, default=0.01,
                   help="annulus: mask neighbours brighter than this fraction of the target (catalogue flux)")
    p.add_argument("--stamp-bg-nb-radius", type=float, default=3.0,
                   help="annulus: neighbour mask radius, px (grows 1.5x per decade of brightness ratio above 1)")
    p.add_argument("--reject-every", type=int, default=0,
                   help="0 = no star rejection (default; mask the scene instead); >0 = refresh period")
    p.add_argument("--reject-burn-in", type=int, default=50)
    p.add_argument("--reject-tau-drop", type=float, default=4.0)
    p.add_argument("--reject-tau-keep", type=float, default=2.5)
    p.add_argument("--reject-window", type=int, default=201)
    p.add_argument("--reject-max-churn", type=float, default=0.02)
    p.add_argument("--early-stop-patience", type=int, default=200,
                   help="stop a stage when the loss changed by < --early-stop-tol (relative) over this many "
                        "steps. 200 since 2026-09-29 (was 30): with rejection really off, Adam's oscillation "
                        "met the 30-step rule after 37-425 steps and left new colour terms untrained")
    p.add_argument("--early-stop-tol", type=float, default=1e-5)
    # training fixes 2026-10-05 (dev/forward_epsf_wcs/docs/TRAINING_FIXES_PLAN_20261005.md); all off by default
    p.add_argument("--stop-rule", choices=("point", "smoothed"), default="point",
                   help="point: the --early-stop-* two-point rule (historical). smoothed: plateau when the "
                        "--stop-window running mean of the loss falls by < --stop-tol (relative) over --stop-lag "
                        "steps AND the WCS moves < --stop-wcs-mpx and contributor fluxes < --stop-flux-tol over the "
                        "same lag. A loss plateau alone multiplies every lr by --lr-decay (at most --lr-decays-max times); the "
                        "stage ends when all criteria hold after >= --lr-decays decays. Stage lengths in "
                        "--steps-per-stage become safety caps")
    p.add_argument("--stop-window", type=int, default=50)
    p.add_argument("--stop-lag", type=int, default=500)
    p.add_argument("--stop-check", type=int, default=100, help="steps between smoothed-rule checks")
    p.add_argument("--stop-tol", type=float, default=1e-4)
    p.add_argument("--stop-wcs-mpx", type=float, default=0.5)
    p.add_argument("--stop-flux-tol", type=float, default=1e-4)
    p.add_argument("--lr-decay", type=float, default=0.3)
    p.add_argument("--lr-decays", type=int, default=2, help="lr decays required before a stage may end")
    p.add_argument("--lr-decays-max", type=int, default=4,
                   help="most lr decays per stage; after that a stage that never meets the stop criteria runs "
                        "to its cap and is labelled not converged")
    p.add_argument("--model-floor-eps", type=float, default=0.0,
                   help="fractional model-error floor: var_eff = (noise_scale sigma)^2 + (eps m)^2, m = scene model "
                        "frozen at each refresh (stage start and every plateau); used in flux solve and loss")
    p.add_argument("--noise-scale", type=float, default=1.0, help="multiplies the stored noise in var_eff")
    p.add_argument("--floor-source", choices=("model", "data"), default="model",
                   help="m in the floor: 'model' = scene model, refreshed (feeds back on the fluxes; can oscillate); "
                        "'data' = max(measured pixel, 0), computed once, so the loss stays one fixed function")
    p.add_argument("--floor-refresh-every", type=int, default=50,
                   help="smoothed rule: refresh the floor weights every this many steps")
    p.add_argument("--floor-settle-tol", type=float, default=1e-4,
                   help="a stage can only end when the last refresh changed the loss (at fixed params) by less than "
                        "this relative amount")
    p.add_argument("--anchors-train-epsf", action="store_true",
                   help="faint WCS anchors also train the ePSF shape (relabelled as contributors at scene load)")
    p.add_argument("--star-weight-cap-tmag", type=float, default=None,
                   help="cap each star's total likelihood weight at that of a star of this Tmag (stamp inverse "
                        "variance divided by (f/f_cap)^2 for brighter stars); weighting inside a star unchanged")
    p.add_argument("--star-weight-cap-form", choices=("hard", "smooth"), default="hard",
                   help="hard: w = min(1, (f_cap/f)^2); smooth: w = 1 / (1 + (f/f_cap)^2)")
    p.add_argument("--penalty-nref", type=float, default=0.0,
                   help="> 0: scale every ePSF penalty by penalty_nref / N_pix so lambda means the same in every "
                        "scene/fold (F1 fold-0 N_pix 1581636 keeps the 10-04 F1 fold-0 balance)")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=20)
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
