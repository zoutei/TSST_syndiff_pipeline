# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Ad hoc diagnostic: radial residual profile (data-model)/noise before/after a
stage-2 checkpoint, on the fast mag7-9/20-frame smoke config. Not part of the
CLI; rebuilds the same setup as run_fit.main() up through ctx/stamp_batch,
then evaluates the forward model at two saved checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .. import _bootstrap  # noqa: F401
from .. import cheb_wcs as CW
from .. import epsf_model as EM
from .. import fit as FIT
from .. import groups as G
from .. import loss as L
from .. import stamp_reject as SR
from .. import temporal as T
from ..data import (
    RegionSpec,
    companion_candidates_by_mag,
    filter_stars_by_xy,
    list_orbit_frames,
    load_region_stack,
    merge_star_tables,
    preload_merged_stars,
    primary_candidates_by_mag,
    primary_to_expanded_index_map,
    select_middle_frames,
    xy_in_region_mask,
)
from ..shared_wcs import fit_region_shared_wcs

from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_gaia_catalog  # noqa: E402


def build_fd(
    *, mag_lo=7.0, mag_hi=9.0, n_frames=20, region_str="1536,1536,2048,2048", stamp_physical=None,
    epsf_grid=(2, 2),
):
    return build_full_context(
        mag_lo=mag_lo, mag_hi=mag_hi, n_frames=n_frames, region_str=region_str,
        stamp_physical=stamp_physical, epsf_grid=epsf_grid,
    )["fd"]


def build_full_context(
    *, mag_lo=7.0, mag_hi=9.0, n_frames=20, region_str="1536,1536,2048,2048", stamp_physical=None,
    epsf_grid=(2, 2),
):
    """Same setup as ``build_fd`` but returns every intermediate object needed
    for mosaic/FITS export and region-file writing (groups, stamp_batch,
    x_ws/y_ws, mags, primary_index_set, frame_imgs, region, ...), not just
    the trainable ``FitData``.

    ``stamp_physical`` defaults to ``EM.STAMP_PHYSICAL`` (13); pass a
    different odd value to evaluate a checkpoint fit with a non-default
    ``--stamp-physical`` (e.g. 11).

    ``epsf_grid`` is ``(n_rows, n_cols)`` for the spatial-variation node grid
    (``--epsf-grid`` on ``run_fit.py``); must match the checkpoint being
    evaluated (``epsf_base_raw.shape[:2]``) or ``L.build_static_context``'s
    bilinear-cell lookup will silently blend against the wrong node field.
    """
    ws = Path("data/data/s0020/c3/k3/diff_linear")
    region = RegionSpec.parse(region_str)
    margin = 8.0
    region_margin = RegionSpec(
        int(region.x_min - margin), int(region.y_min - margin),
        int(region.x_max + margin), int(region.y_max + margin),
    )
    n_rows, n_cols = int(epsf_grid[0]), int(epsf_grid[1])
    stamp_physical = EM.STAMP_PHYSICAL if stamp_physical is None else int(stamp_physical)

    frames_all, _ = list_orbit_frames(ws, sector=20, orbit_index=1)
    frames = select_middle_frames(frames_all, n_frames)

    gaia_full = load_gaia_catalog(ws)
    wcs, region_qc = fit_region_shared_wcs(frames[0], gaia_full, region, margin_px=margin)
    cheb_static = CW.ChebWcsStatic.from_wcs(wcs, region, poly_degree=3)

    btjd = np.array([f.btjd for f in frames])
    wcs_tb = T.build_temporal_basis(btjd, n_interior=10, uniform_knots=True)
    w_tb = T.build_temporal_basis(btjd, n_interior=3, uniform_knots=True)

    merged_by_stem = preload_merged_stars(frames, gaia_full, n_workers=16)
    wcs_coeff_ws = FIT.warmstart_wcs_coeff(
        frames, gaia_full, cheb_static, np.asarray(wcs_tb.frame_basis),
        merged_by_stem=merged_by_stem,
    )
    ref_frame_index = len(frames) // 2

    primary_cand = primary_candidates_by_mag(gaia_full, tess_mag_range=(mag_lo, mag_hi))
    companion_cand = companion_candidates_by_mag(gaia_full, tess_mag_max=13.0)

    ra_pc = primary_cand["ra"].to_numpy(dtype=float)
    dec_pc = primary_cand["dec"].to_numpy(dtype=float)
    x_ws_pc, y_ws_pc = CW.eval_positions_at_frame_index(
        ra_pc, dec_pc, wcs_coeff_ws, cheb_static, wcs_tb.frame_basis, ref_frame_index,
    )
    gaia_fit, x_ws_fit, y_ws_fit = filter_stars_by_xy(primary_cand, x_ws_pc, y_ws_pc, region, margin_px=margin)
    fit_stars = gaia_fit.loc[~gaia_fit.too_bright & ~gaia_fit.too_faint].reset_index(drop=True)
    fit_mask = (~gaia_fit.too_bright & ~gaia_fit.too_faint).to_numpy()
    x_ws_fit = x_ws_fit[fit_mask]
    y_ws_fit = y_ws_fit[fit_mask]

    fit_stars, _ = SR.prefilter_primaries_by_centroids_qc(
        frames, fit_stars, gaia_full, min_frac=0.5, merged_by_stem=merged_by_stem,
    )
    x_ws_fit, y_ws_fit = CW.eval_positions_at_frame_index(
        fit_stars["ra"].to_numpy(dtype=float), fit_stars["dec"].to_numpy(dtype=float),
        wcs_coeff_ws, cheb_static, wcs_tb.frame_basis, ref_frame_index,
    )

    ra_cc = companion_cand["ra"].to_numpy(dtype=float)
    dec_cc = companion_cand["dec"].to_numpy(dtype=float)
    x_ws_cc, y_ws_cc = CW.eval_positions_at_frame_index(
        ra_cc, dec_cc, wcs_coeff_ws, cheb_static, wcs_tb.frame_basis, ref_frame_index,
    )
    companion_pool, _, _ = filter_stars_by_xy(companion_cand, x_ws_cc, y_ws_cc, region, margin_px=margin)
    expanded_stars = merge_star_tables(fit_stars, companion_pool)

    groups = G.build_groups(x_ws_fit, y_ws_fit, max_sep_px=7.0, max_group_size=4)
    primary_members_by_group = [groups.members[gi][groups.valid[gi]].copy() for gi in range(groups.n_groups)]

    ra = expanded_stars["ra"].to_numpy(dtype=float)
    dec = expanded_stars["dec"].to_numpy(dtype=float)
    mags = expanded_stars["tess_mag"].to_numpy(dtype=float)
    x_ws, y_ws = CW.eval_positions_at_frame_index(
        ra, dec, wcs_coeff_ws, cheb_static, wcs_tb.frame_basis, ref_frame_index,
    )
    primary_remap = primary_to_expanded_index_map(fit_stars, expanded_stars)
    primary_members_expanded = [primary_remap[m] for m in primary_members_by_group]
    primary_index_set = set(int(i) for i in primary_remap)
    neighbor_indices = np.array([i for i in range(len(expanded_stars)) if i not in primary_index_set], dtype=int)
    groups, augment_stats, stamp_cx, stamp_cy = G.augment_groups_with_stamp_neighbors(
        groups, x_ws, y_ws, neighbor_indices,
        primary_members_by_group=primary_members_expanded,
        neighbor_mags=mags, stamp_physical=stamp_physical,
        companion_radius_px=None, max_group_size=4, mags=mags,
    )
    groups, stamp_cx, stamp_cy, _ = G.filter_groups_by_member_radius(
        groups, x_ws, y_ws, stamp_cx, stamp_cy, max_member_radius_px=0.0,
    )

    frame_imgs = load_region_stack(frames, region_margin, n_workers=16)
    groups, stamp_cx, stamp_cy, _ = G.drop_groups_off_array(
        groups, stamp_cx, stamp_cy,
        array_origin=(region_margin.x_min, region_margin.y_min),
        array_shape=frame_imgs[0].cal.shape, stamp=stamp_physical, n_stars=len(expanded_stars),
    )

    def group_mag(groups, expanded_mags, primary_index_set):
        out = np.full(groups.n_groups, 12.0, dtype=float)
        for gi in range(groups.n_groups):
            idx = groups.members[gi][groups.valid[gi]]
            prim = [i for i in idx if int(i) in primary_index_set]
            use = prim if prim else list(idx)
            if use:
                out[gi] = float(np.min(expanded_mags[use]))
        return out

    gmag = group_mag(groups, mags, primary_index_set)
    stamp_w = L.soft_snr_stamp_weights(gmag, snr_cap_mag=9.0, w_min=0.05)
    r_stage23 = L.fit_radius_from_mag(gmag, stage=2)

    stamp_batch = G.extract_stamps(
        groups, frame_imgs, x_ws, y_ws,
        array_origin=(region_margin.x_min, region_margin.y_min),
        stamp=stamp_physical, stamp_center_x=stamp_cx, stamp_center_y=stamp_cy,
    )

    epsf_grid = EM.EpsfGridStatic.from_region(region, n_rows=n_rows, n_cols=n_cols, crop_origin=(44, 0))
    epsf_params0 = EM.init_epsf_from_prf(
        camera=3, ccd=3, sector=20, grid=epsf_grid, stamp_physical=stamp_physical,
    )
    t_exp_sec = float(frame_imgs[0].exposure_days * 86400.0)
    ctx = L.build_static_context(
        cheb_static=cheb_static, wcs_frame_basis=wcs_tb.frame_basis, w_frame_basis=w_tb.frame_basis,
        epsf_grid=epsf_grid, groups=groups, ra=ra, dec=dec,
        stamp_center_x=stamp_batch.stamp_center_x, stamp_center_y=stamp_batch.stamp_center_y,
        t_exp_sec=t_exp_sec, stamp_snr_weight=stamp_w, fit_radius=r_stage23,
    )
    fd = FIT.FitData(
        ctx=ctx,
        data=jnp.asarray(stamp_batch.data),
        noise=jnp.asarray(stamp_batch.noise),
        weight=jnp.asarray(stamp_batch.weight),
        wcs_second_diff=T.second_difference_matrix(wcs_tb.n_basis),
        w_second_diff=T.second_difference_matrix(w_tb.n_basis),
        epsf_modes_init=epsf_params0.modes,
    )
    return {
        "fd": fd,
        "ctx": ctx,
        "groups": groups,
        "stamp_batch": stamp_batch,
        "frame_imgs": frame_imgs,
        "frames": frames,
        "region": region,
        "region_margin": region_margin,
        "x_ws": x_ws,
        "y_ws": y_ws,
        "mags": mags,
        "expanded_stars": expanded_stars,
        "fit_stars": fit_stars,
        "primary_index_set": primary_index_set,
        "epsf_params0": epsf_params0,
        "sector": 20,
        "camera": 3,
        "ccd": 3,
    }


def radial_profile(fd, params, *, n_bins=13):
    """Robust per-pixel chi = (data-model)/sqrt(var) statistics by radius bin.

    Uses median (not weighted mean) so a single bright/outlier stamp cannot
    dominate the bin -- matches the Huber-chi quantity the loss itself uses.
    """
    templates, x_t, y_t, w_of_t = L.forward_model(params, fd.ctx)
    from . import flux_solve as FS
    var = L.pixel_variance(fd.noise)
    rmask = L.radius_pixel_mask(fd.ctx.fit_radius, stamp=fd.data.shape[-1])
    pix_w = fd.weight * rmask[:, None, :, :]
    iv = L.inverse_variance_weights(pix_w, var)
    flux = FS.solve_group_fluxes(templates, fd.data, iv, ridge=1e-6)
    model = FS.model_stamps(templates, flux)
    chi = np.asarray((fd.data - model) / jnp.sqrt(var))
    w = np.asarray(pix_w) > 0

    S = chi.shape[-1]
    c = S // 2
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    bins = np.linspace(0, r.max() + 1e-6, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        vals = chi[:, :, m][w[:, :, m]]
        if vals.size == 0:
            out.append((0.5 * (bins[i] + bins[i + 1]), float("nan"), float("nan"), 0))
            continue
        out.append((0.5 * (bins[i] + bins[i + 1]), float(np.median(vals)), float(np.std(vals)), int(vals.size)))
    return out


def radial_profile_fractional(fd, params, *, n_bins: int = 13) -> list[tuple[float, float, int]]:
    """Radial profile of (model-data) as a fraction of each star's *total* flux.

    Chi (sigma-normalized) doesn't tell you how big the mismatch is relative
    to how bright the star actually was -- a chi of 50 means something
    different for a mag-7 star than a mag-11 one. This instead sums
    (model-data) within each radius bin and divides by that (group, frame)'s
    total fitted flux, so every bin's number is directly interpretable as
    "% of the star's own light misallocated at this radius" -- positive means
    over-subtraction (model brighter than data) in the original hp_d sense.
    """
    templates, x_t, y_t, w_of_t = L.forward_model(params, fd.ctx)
    from . import flux_solve as FS
    var = L.pixel_variance(fd.noise)
    rmask = L.radius_pixel_mask(fd.ctx.fit_radius, stamp=fd.data.shape[-1])
    pix_w = fd.weight * rmask[:, None, :, :]
    iv = L.inverse_variance_weights(pix_w, var)
    flux = FS.solve_group_fluxes(templates, fd.data, iv, ridge=1e-6)
    model = FS.model_stamps(templates, flux)

    data_np = np.asarray(fd.data)
    model_np = np.asarray(model)
    w = np.asarray(pix_w) > 0

    # Total flux per (group, frame): sum of *all* member fluxes (primary +
    # companions), i.e. the total light actually present in that stamp.
    flux_np = np.asarray(flux)  # (n_groups, n_frames, K)
    total_flux = flux_np.sum(axis=-1)  # (n_groups, n_frames)
    total_flux_safe = np.where(np.abs(total_flux) > 1e-6, total_flux, np.nan)

    S = data_np.shape[-1]
    c = S // 2
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    bins = np.linspace(0, r.max() + 1e-6, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        diff_sum = np.where(w[:, :, m], model_np[:, :, m] - data_np[:, :, m], 0.0).sum(axis=-1)
        n_active = w[:, :, m].sum(axis=-1)
        active = n_active > 0
        frac = diff_sum[active] / total_flux_safe[active]
        frac = frac[np.isfinite(frac)]
        if frac.size == 0:
            out.append((0.5 * (bins[i] + bins[i + 1]), float("nan"), 0))
            continue
        out.append((0.5 * (bins[i] + bins[i + 1]), float(np.median(frac)) * 100.0, int(frac.size)))
    return out


def wing_bias_summary(fd, params, *, r_min: float = 3.5) -> dict:
    """Dedicated wing-only (r>=r_min) bias/scatter summary.

    Separate from ``radial_profile`` so wing convergence can be tracked
    without the (much larger, separately-tracked) core numbers drowning it
    out. Reports both the robust chi median/std (matches the training loss's
    own quantity) and a direct flux-domain over/under-subtraction fraction
    -- ``(sum(model)-sum(data))/sum(data)`` in the wing mask, positive means
    the model over-predicts (over-subtracts in the original hp_d framing).
    """
    templates, x_t, y_t, w_of_t = L.forward_model(params, fd.ctx)
    from . import flux_solve as FS
    var = L.pixel_variance(fd.noise)
    rmask = L.radius_pixel_mask(fd.ctx.fit_radius, stamp=fd.data.shape[-1])
    pix_w = fd.weight * rmask[:, None, :, :]
    iv = L.inverse_variance_weights(pix_w, var)
    flux = FS.solve_group_fluxes(templates, fd.data, iv, ridge=1e-6)
    model = FS.model_stamps(templates, flux)

    S = fd.data.shape[-1]
    c = S // 2
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    wing = r >= r_min

    chi = np.asarray((fd.data - model) / jnp.sqrt(var))
    w = np.asarray(pix_w) > 0
    wing_mask = w[:, :, wing]
    chi_wing = chi[:, :, wing][wing_mask]

    data_np = np.asarray(fd.data)[:, :, wing][wing_mask]
    model_np = np.asarray(model)[:, :, wing][wing_mask]
    over_sub_frac = float((model_np.sum() - data_np.sum()) / np.clip(data_np.sum(), 1e-6, None))

    return {
        "r_min": r_min,
        "n_pix": int(chi_wing.size),
        "median_chi": float(np.median(chi_wing)) if chi_wing.size else float("nan"),
        "std_chi": float(np.std(chi_wing)) if chi_wing.size else float("nan"),
        "over_subtraction_frac": over_sub_frac,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("stage_a", type=str, nargs="?", default="params_stage1.npz")
    p.add_argument("stage_b", type=str, nargs="?", default="params_stage2.npz")
    p.add_argument("--mag-lo", type=float, default=7.0)
    p.add_argument("--mag-hi", type=float, default=9.0)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--region", type=str, default="1536,1536,2048,2048")
    p.add_argument("--epsf-grid", type=str, default=None,
                    help="n_rowsxn_cols override; default: infer from stage_b checkpoint's "
                         "epsf_base_raw shape")
    args = p.parse_args()

    run = args.run_dir
    if args.epsf_grid:
        n_rows, n_cols = (int(v) for v in args.epsf_grid.lower().split("x"))
    else:
        probe = FIT.load_params_npz(run / args.stage_b)
        n_rows, n_cols = int(probe["epsf_base_raw"].shape[0]), int(probe["epsf_base_raw"].shape[1])
    print(f"epsf_grid inferred/used: {n_rows}x{n_cols}")

    fd = build_fd(
        mag_lo=args.mag_lo, mag_hi=args.mag_hi, n_frames=args.n_frames, region_str=args.region,
        epsf_grid=(n_rows, n_cols),
    )
    for name in (args.stage_a, args.stage_b):
        params = FIT.load_params_npz(run / name)
        prof = radial_profile(fd, params)
        print(f"--- {name} ---")
        print(f"{'r':>6} {'median_chi':>12} {'std_chi':>12} {'n_pix':>10}")
        for r, med, std, n in prof:
            print(f"{r:6.2f} {med:12.4f} {std:12.4f} {n:10d}")
