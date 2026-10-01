# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Export a single-FFI *scene* bundle: a fixed S×S square on every Tmag <= 13 star.

Plan: ``docs/SCENE_MODE_PLAN_20260923.md``. Companion trainer: ``scene_fit.py``.

Unlike the packed/irregular bundle, every catalogue star gets the same S×S square
(S = 15 by default, the largest square the ePSF node grid can fill), all stars are
in the model, and each image pixel enters the likelihood exactly once even when
several squares cover it. Fluxes are solved exactly per *overlap island* (connected
component of stars whose squares overlap), so there is no group-size cap.

This module deliberately reuses a finished packed bundle (``--source-bundle``) for
everything that is not pixel layout: the epoch-propagated Gaia positions/colours,
the frozen TAN + Chebyshev static WCS, the initial ``wcs_coeff``, the ePSF node
grid and the seed ePSF. Only the pixels, the mask, the star roles and the
overlap/island tables are new.

Star roles (all stars contribute light to the model; the role only decides which
parameters its pixels may train):

    0  ePSF contributor : 8 <= Tmag < 11 and a member of a source-bundle ePSF group
                          (i.e. passed the photutils QC pre-filter)
    1  WCS anchor       : 11 <= Tmag <= 13
    2  nuisance         : everything else (Tmag < 8, QC failures, masked or
                          off-array centres, too few unmasked core pixels) --
                          flux only, no gradients, weak flux prior toward Gaia
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# jax MUST be imported before pandas/pyarrow: pyarrow loads the system
# /lib64/libstdc++.so.6, which is too old for jaxlib (GLIBCXX_3.4.30), and the
# first XLA CPU compile then segfaults. Importing jax first binds conda's libstdc++.
import jax  # noqa: F401  (import order is load-bearing)
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import _bootstrap  # noqa: F401
from . import cheb_wcs as CW
from . import epsf_model as EM
from . import fit_bundle as FB

SCENE_VERSION = 1

ROLE_CONTRIB = 0
ROLE_ANCHOR = 1
ROLE_NUISANCE = 2

# shared-mask bits (syndiff_pipeline/difference_imaging/masking/bits.py)
BIT_BRIGHT = 1
BIT_EDGE = 8
BIT_PS1 = 16
BIT_TNS = 64
BIT_ASTEROID = 128


def _log(msg: str) -> None:
    print(f"[scene_export {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pow2_ceil(n: int) -> int:
    return 1 << max(0, int(np.ceil(np.log2(max(1, n)))))


def extend_epsf_seed(base58: np.ndarray, stamp: int, *, r_fit=(4.0, 5.0),
                     r_start: float = 5.0, index: float = 3.0,
                     floor: float = 1e-7) -> np.ndarray:
    """Zero-pad a node grid to the ``stamp`` geometry and give it a live power-law wing.

    The production epsf_r1 seed stops at ~5.6 px, and the base is a softplus of the
    raw leaf, so an exactly-zero outer ring sits where Adam moves it by only
    ~lr per step in log space and it never trains (measured: the minbg run ends with
    ~6e-8 per subpixel beyond 6.5 px). Beyond ``r_start`` each node gets
    ``A r^-index`` with ``A`` fitted to its own [r_fit] annulus, then Σ = 1.
    """
    base58 = np.asarray(base58, dtype=np.float64)
    _, g_new, _ = EM.node_geometry(stamp)
    g_old = base58.shape[-1]
    pad = (g_new - g_old) // 2
    if pad < 0 or (g_new - g_old) % 2:
        raise ValueError(f"cannot pad grid {g_old} -> {g_new}")
    out = np.pad(base58, [(0, 0), (0, 0), (pad, pad), (pad, pad)])
    c = (g_new - 1) / 2.0
    yy, xx = np.mgrid[:g_new, :g_new]
    r = np.hypot(xx - c, yy - c) / EM.OVERSAMPLE
    ann = (r >= r_fit[0]) & (r < r_fit[1])
    outer = r >= r_start
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            g = out[i, j]
            amp = float(np.median(g[ann] * r[ann] ** index))
            g[outer] = np.maximum(amp * r[outer] ** (-index), floor)
            g /= g.sum()
    return out.astype(np.float32)


def match_catalog(ra, dec, gaia: pd.DataFrame, tol_arcsec: float = 10.0):
    """Nearest-neighbour sky match of bundle stars to the lane Gaia catalogue."""
    def unit(r, d):
        r = np.deg2rad(np.asarray(r, float))
        d = np.deg2rad(np.asarray(d, float))
        return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]

    tree = cKDTree(unit(gaia["ra"], gaia["dec"]))
    dist, idx = tree.query(unit(ra, dec))
    sep = np.rad2deg(dist) * 3600.0
    ok = sep < tol_arcsec
    return idx, ok, sep


def load_frame_mask(workspace: Path, *, data_root, sector, camera, ccd, btjd, shape):
    try:
        from syndiff_pipeline.difference_imaging.masking.ffi_mask import (
            load_catalog_for_scc_lane,
        )
        cat = load_catalog_for_scc_lane(workspace, data_root=data_root, sector=sector,
                                        camera=camera, ccd=ccd)
        m = np.asarray(cat.mask_at(float(btjd), which="full"), dtype=np.int64)
        src = "MaskCatalog.mask_at(full)"
    except Exception as exc:  # noqa: BLE001
        _log(f"MaskCatalog unavailable ({exc!r}); falling back to shared_mask.fits.fz")
        with fits.open(workspace / "shared_mask.fits.fz") as h:
            m = [x.data for x in h if x.data is not None][0].astype(np.int64)
        src = "shared_mask.fits.fz (static only)"
    if m.shape != tuple(shape):
        raise ValueError(f"mask shape {m.shape} != image shape {shape}")
    return m, src


def build_scene(args) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = int(args.stamp)
    h = S // 2

    _log(f"loading source bundle {args.source_bundle}")
    b = FB.load_fit_bundle(Path(args.source_bundle))
    meta_src = dict(b.meta)
    stem = meta_src["selected_frame_stems"][0]
    btjd = float(meta_src["selected_frame_btjd"][0])
    workspace = Path(meta_src["workspace"])
    sector, camera, ccd = int(meta_src["sector"]), int(meta_src["camera"]), int(meta_src["ccd"])
    region = meta_src["region"]
    if region[0] != 0 or region[1] != 0:
        raise NotImplementedError("scene export assumes a full-CCD source region")

    n_all = int(np.asarray(b.ra).shape[0])
    wcs0 = np.asarray(b.params0["wcs_coeff"], dtype=np.float32)
    x_t, y_t = CW.eval_all_positions(
        np.asarray(b.x_lin), np.asarray(b.y_lin), np.asarray(b.cheb_basis),
        wcs0, np.asarray(b.wcs_frame_basis), b.cheb_static.n_terms,
    )
    x0 = np.asarray(x_t)[:, 0].astype(np.float64)
    y0 = np.asarray(y_t)[:, 0].astype(np.float64)

    _log("matching bundle stars to the lane Gaia catalogue")
    gaia = pd.read_csv(workspace / "gaia_catalog_pipeline.csv")
    gidx, gok, sep = match_catalog(b.ra, b.dec, gaia)
    tmag = np.where(gok, gaia["tess_mag"].to_numpy(float)[gidx], np.nan)
    tflux = np.where(gok, gaia["tess_flux"].to_numpy(float)[gidx], np.nan)
    source_id = np.where(gok, gaia["source_id"].to_numpy(np.int64)[gidx], -1)
    _log(f"  matched {gok.sum()}/{n_all} (median sep {np.median(sep[gok]):.3f}\")")

    # ePSF-contributor membership in the source bundle (QC-passed primaries + their
    # joint-fit companions); we keep only the 8-11 mag part as contributors.
    in_contrib_group = np.zeros(n_all, dtype=bool)
    members = np.asarray(b.members)
    valid = np.asarray(b.valid, dtype=bool)
    contrib_g = np.asarray(b.is_epsf_contributor, dtype=bool)
    in_contrib_group[members[contrib_g][valid[contrib_g]]] = True

    lo, hi = args.contrib_mag
    role = np.full(n_all, ROLE_NUISANCE, dtype=np.int8)
    role[(tmag >= lo) & (tmag < hi) & in_contrib_group] = ROLE_CONTRIB
    role[(tmag >= hi) & (tmag <= args.tmag_max)] = ROLE_ANCHOR
    keep = gok & np.isfinite(tmag) & (tmag <= args.tmag_max)

    _log(f"loading hp_d frame {stem}")
    with fits.open(workspace / "hp_d" / f"{stem}_hp_d.fits.fz") as hd:
        cal = np.asarray(hd[1].data, dtype=np.float32)
        noise = np.asarray(hd[2].data, dtype=np.float32)
    ny, nx = cal.shape
    mask, mask_src = load_frame_mask(workspace, data_root=Path(args.data_root), sector=sector,
                                     camera=camera, ccd=ccd, btjd=btjd, shape=cal.shape)
    bright = (mask & BIT_BRIGHT) != 0
    if args.bright_dilate > 0:
        bright = ndimage.binary_dilation(bright, iterations=int(args.bright_dilate))
    other = (mask & (BIT_EDGE | BIT_PS1 | BIT_TNS | BIT_ASTEROID)) != 0
    finite = np.isfinite(cal) & np.isfinite(noise) & (noise > 0)
    good = finite & ~bright & ~other
    _log(f"  mask source: {mask_src}; bad pixels {100 * (1 - good.mean()):.2f}% "
         f"(bright+{args.bright_dilate}px {100 * bright.mean():.2f}%, "
         f"edge/ps1/tns/asteroid {100 * other.mean():.2f}%)")

    cx = np.round(x0).astype(np.int64)
    cy = np.round(y0).astype(np.int64)
    keep &= (cx >= -h) & (cx < nx + h) & (cy >= -h) & (cy < ny + h)
    if args.subregion:
        sx0, sy0, sx1, sy1 = [int(v) for v in args.subregion.split(",")]
        keep &= (cx >= sx0) & (cx < sx1) & (cy >= sy0) & (cy < sy1)
    sel = np.flatnonzero(keep)
    N = sel.size
    _log(f"scene stars: {N}")

    off = np.arange(S) - h
    gx = cx[sel][:, None, None] + off[None, None, :]
    gy = cy[sel][:, None, None] + off[None, :, None]
    gx = np.broadcast_to(gx, (N, S, S))
    gy = np.broadcast_to(gy, (N, S, S))
    inarr = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    gxc = np.clip(gx, 0, nx - 1)
    gyc = np.clip(gy, 0, ny - 1)
    data_loc = np.where(inarr, cal[gyc, gxc], 0.0).astype(np.float32).reshape(N, S * S)
    noise_loc = np.where(inarr, noise[gyc, gxc], 1.0).astype(np.float32).reshape(N, S * S)
    finite_loc = (inarr & finite[gyc, gxc]).reshape(N, S * S)
    valid_loc = (inarr & good[gyc, gxc]).reshape(N, S * S)
    noise_loc = np.where(finite_loc, noise_loc, 1.0).astype(np.float32)
    data_loc = np.where(finite_loc, data_loc, 0.0).astype(np.float32)

    lr = np.hypot(*np.meshgrid(off, off)).reshape(-1)
    core = lr <= float(args.core_radius)
    n_core_valid = (valid_loc & core[None]).sum(1)
    role_s = role[sel].copy()
    centre_ok = valid_loc[:, (S * S) // 2]
    demote = (role_s != ROLE_NUISANCE) & ((n_core_valid < args.min_core_valid) | ~centre_ok)
    role_s[demote] = ROLE_NUISANCE
    _log(f"  roles: contrib {np.sum(role_s == 0)}, anchor {np.sum(role_s == 1)}, "
         f"nuisance {np.sum(role_s == 2)} (demoted for masked core: {demote.sum()})")

    # union pixel ids; each union pixel gets exactly one owner copy for the loss
    uid_raw = np.where(inarr, gy * nx + gx, -1).reshape(N, S * S)
    flat = uid_raw.reshape(-1)
    on = flat >= 0
    uniq, inv = np.unique(flat[on], return_inverse=True)
    U = uniq.size
    uid = np.full(flat.shape, U, dtype=np.int32)  # U = dummy slot for off-array
    uid[on] = inv.astype(np.int32)
    first = np.full(U, -1, dtype=np.int64)
    pos = np.flatnonzero(on)
    # np.unique(return_index) on the inverse gives the first occurrence per union pixel
    _, first_idx = np.unique(inv, return_index=True)
    owner = np.zeros(flat.shape, dtype=bool)
    owner[pos[first_idx]] = True
    uid = uid.reshape(N, S * S)
    owner = owner.reshape(N, S * S)
    _log(f"  union pixels {U} (sum of squares {N * S * S}); owned+valid "
         f"{int((owner & valid_loc).sum())}")

    # overlapping pairs (Chebyshev distance of square centres < S)
    P = np.c_[cx[sel], cy[sel]].astype(float)
    pairs = cKDTree(P).query_pairs(S - 0.5, p=np.inf, output_type="ndarray")
    if pairs.size == 0:
        pairs = np.zeros((0, 2), dtype=np.int64)
    pi, pj = pairs[:, 0], pairs[:, 1]
    ox = (cx[sel][pj] - cx[sel][pi]).astype(np.int32)
    oy = (cy[sel][pj] - cy[sel][pi]).astype(np.int32)
    # local-flat gather indices: pixel (a, c) of i is pixel (a - oy, c - ox) of j
    A_, C_ = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    ja = A_[None] - oy[:, None, None]
    jc = C_[None] - ox[:, None, None]
    pmask = (ja >= 0) & (ja < S) & (jc >= 0) & (jc < S)
    lj = (np.clip(ja, 0, S - 1) * S + np.clip(jc, 0, S - 1)).reshape(-1, S * S).astype(np.int16)
    pmask = pmask.reshape(-1, S * S)
    _log(f"  overlapping pairs {len(pi)}")

    # islands
    A = coo_matrix((np.ones(len(pi)), (pi, pj)), shape=(N, N))
    n_isl, isl = connected_components(A, directed=False)
    sizes = np.bincount(isl, minlength=n_isl)
    tier_k = np.array([_pow2_ceil(s) for s in sizes])
    tiers = np.unique(tier_k)
    star_tier = np.zeros(N, np.int16)
    star_row = np.zeros(N, np.int32)
    star_slot = np.zeros(N, np.int16)
    tier_star_idx = {}
    for ti, K in enumerate(tiers):
        isl_ids = np.flatnonzero(tier_k == K)
        mat = np.full((isl_ids.size, K), -1, dtype=np.int32)
        row_of = {int(v): r for r, v in enumerate(isl_ids)}
        fill = np.zeros(isl_ids.size, dtype=np.int32)
        for s in np.flatnonzero(np.isin(isl, isl_ids)):
            r = row_of[int(isl[s])]
            mat[r, fill[r]] = s
            star_tier[s], star_row[s], star_slot[s] = ti, r, fill[r]
            fill[r] += 1
        tier_star_idx[int(K)] = mat
    _log(f"  islands {n_isl}: sizes max {sizes.max()}, tiers "
         + ", ".join(f"K{K}x{v.shape[0]}" for K, v in tier_star_idx.items()))

    # seed ePSF on the S-geometry grid
    base_src = np.asarray(EM.decode_epsf_base(np.asarray(b.params0["epsf_base_raw"])))
    seed = extend_epsf_seed(base_src, S)

    arrays = dict(
        scene_version=np.int32(SCENE_VERSION),
        stamp=np.int32(S),
        star_bundle_index=sel.astype(np.int32),
        source_id=source_id[sel],
        tess_mag=tmag[sel].astype(np.float32),
        tess_flux=tflux[sel].astype(np.float32),
        role=role_s,
        cx=cx[sel].astype(np.int32),
        cy=cy[sel].astype(np.int32),
        x0=x0[sel].astype(np.float32),
        y0=y0[sel].astype(np.float32),
        data=data_loc,
        noise=noise_loc,
        valid=valid_loc,
        finite=finite_loc,
        uid=uid,
        owner=owner,
        n_union=np.int64(U),
        core=core,
        pair_i=pi.astype(np.int32),
        pair_j=pj.astype(np.int32),
        pair_lj=lj,
        pair_mask=pmask,
        island=isl.astype(np.int32),
        star_tier=star_tier,
        star_row=star_row,
        star_slot=star_slot,
        tiers=tiers.astype(np.int32),
        epsf_seed=seed,
    )
    for K, mat in tier_star_idx.items():
        arrays[f"tier{K}_star_idx"] = mat
    out = out_dir / "scene_bundle.npz"
    np.savez(out, **arrays)
    meta = dict(
        scene_version=SCENE_VERSION,
        source_bundle=str(Path(args.source_bundle).resolve()),
        workspace=str(workspace),
        frame_stem=stem,
        frame_btjd=btjd,
        sector=sector, camera=camera, ccd=ccd,
        stamp=S,
        tmag_max=args.tmag_max,
        contrib_mag=list(args.contrib_mag),
        bright_dilate=args.bright_dilate,
        masked_bits=[BIT_BRIGHT, BIT_EDGE, BIT_PS1, BIT_TNS, BIT_ASTEROID],
        straps_masked=False,
        mask_source=mask_src,
        core_radius=args.core_radius,
        min_core_valid=args.min_core_valid,
        subregion=args.subregion,
        n_stars=int(N),
        n_roles={"contrib": int(np.sum(role_s == 0)), "anchor": int(np.sum(role_s == 1)),
                 "nuisance": int(np.sum(role_s == 2))},
        n_demoted_masked_core=int(demote.sum()),
        n_union=int(U),
        n_owned_valid=int((owner & valid_loc).sum()),
        n_pairs=int(len(pi)),
        n_islands=int(n_isl),
        max_island=int(sizes.max()),
        tiers={str(K): int(v.shape[0]) for K, v in tier_star_idx.items()},
        epsf_seed="source params0 base, zero-padded, r^-3 wing beyond 5 px",
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    _log(f"wrote {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source-bundle", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-root", default="/astro/armin/koji/syndiff/data")
    p.add_argument("--stamp", type=int, default=15)
    p.add_argument("--tmag-max", type=float, default=13.0)
    p.add_argument("--contrib-mag", type=float, nargs=2, default=(8.0, 11.0))
    p.add_argument("--bright-dilate", type=int, default=3)
    p.add_argument("--core-radius", type=float, default=3.0)
    p.add_argument("--min-core-valid", type=int, default=15)
    p.add_argument("--subregion", default=None,
                   help="x0,y0,x1,y1: keep only stars whose square centre is inside (smoke tests)")
    build_scene(p.parse_args(argv))


if __name__ == "__main__":
    main()
