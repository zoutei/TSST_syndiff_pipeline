# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Diagnose the excess-variance floor found at the mag9-11 stage-2 plateau
(see HANDOFF_epsf_wing_stage2.md, "The plateau: found it" / "Excess-variance
diagnosis"). Median chi ~ 0 at every radius (unbiased), but std chi is
11-54x too wide vs. the NOISE plane in r in [1.5, 3.5) -- this script asks
whether that excess is (a) a handful of bad frames, (b-flux) analytic
flux-solve scatter beyond what a naive aperture sum shows, or (b-traj) a
jittery x(t)/y(t) trajectory.

Read-only: loads a frozen checkpoint (params_stage2.npz), rebuilds the exact
same context via dev_radial_profile.build_full_context, and evaluates
L.forward_model + flux_solve.solve_group_fluxes exactly as
dev_radial_profile.py / dev_export_fits.py do. Does not touch model/loss/
optimizer code.

Usage:
    python -m syndiff_pipeline.forward_model.diagnostics.excess_variance_diag \
        [run_dir] [params_name] [--mag-lo 9] [--mag-hi 11] [--n-clean 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .. import fit as FIT
from .. import flux_solve as FS
from .. import loss as L
from .. import stamp_reject as SR
from .radial_profile import build_full_context

R_ANNULUS_LO = 1.5
R_ANNULUS_HI = 3.5
NAIVE_APERTURE_R = 3.5


def _forward_and_flux(fd, params):
    from .export_fits import pixel_weights

    templates, x_slot, y_slot, w_of_t = L.forward_model(params, fd.ctx)
    var = L.pixel_variance(fd.noise)
    pix_w = pixel_weights(fd)
    iv = L.inverse_variance_weights(pix_w, var)
    flux = FS.solve_group_fluxes(templates, fd.data, iv, ridge=1e-6)
    model = FS.model_stamps(templates, flux)
    chi = np.asarray((fd.data - model) / jnp.sqrt(var))
    w_mask = np.asarray(pix_w) > 0
    return {
        "templates": templates,
        "x_slot": np.asarray(x_slot),
        "y_slot": np.asarray(y_slot),
        "flux": np.asarray(flux),
        "model": np.asarray(model),
        "chi": chi,
        "w_mask": w_mask,
        "var": np.asarray(var),
    }


def _radius_grid(S: int) -> np.ndarray:
    c = S // 2
    yy, xx = np.mgrid[0:S, 0:S]
    return np.sqrt((xx - c) ** 2 + (yy - c) ** 2)


def step2_per_frame_annulus_stats(fd, out, *, r_lo=R_ANNULUS_LO, r_hi=R_ANNULUS_HI):
    """Per-frame chi summary pooled over all groups/pixels in the r in [r_lo,r_hi)
    annulus. Answers (a) few-bad-frames vs (b) all-frames-similarly-bad."""
    S = fd.data.shape[-1]
    r = _radius_grid(S)
    annulus = (r >= r_lo) & (r < r_hi)
    chi = out["chi"]
    w_mask = out["w_mask"]
    n_frames = chi.shape[1]

    rows = []
    for t in range(n_frames):
        combined = w_mask[:, t, :, :] & annulus[None, :, :]
        vals = chi[:, t, :, :][combined]
        if vals.size == 0:
            rows.append((t, float("nan"), float("nan"), 0))
            continue
        rows.append((t, float(np.median(np.abs(vals))), float(np.std(vals)), int(vals.size)))
    return rows


def step2_with_reject(fd, params):
    """Per-frame MAD-reject fraction (n_sigma=3.0), for extra context on
    whether excess scatter correlates with which stamps get MAD-clipped."""
    chi2_red, pix_sum = SR.per_stamp_chi2_red(params, fd)
    chi2_red = np.asarray(chi2_red)
    pix_sum = np.asarray(pix_sum)
    pix_active = pix_sum > 0
    stamp_active = SR.mad_reject_mask(chi2_red, n_sigma=3.0, pix_active=pix_active)
    n_frames = chi2_red.shape[1]
    rows = []
    for t in range(n_frames):
        act = pix_active[:, t]
        n_cand = int(act.sum())
        n_rej = int((act & (stamp_active[:, t] == 0)).sum())
        rows.append((t, n_rej, n_cand, n_rej / max(n_cand, 1)))
    return rows


def pick_clean_groups(groups, mags, n_clean: int):
    """Return group indices with exactly one valid member (isolated primary,
    no companions attached), sorted brightest (smallest mag) first."""
    n_members = groups.valid.sum(axis=1)
    clean = np.where(n_members == 1)[0]
    slot0 = np.array([int(np.where(groups.valid[gi])[0][0]) for gi in clean])
    star_idx = groups.members[clean, slot0]
    gmag = mags[star_idx]
    order = np.argsort(gmag)
    chosen = clean[order][:n_clean]
    chosen_slots = slot0[order][:n_clean]
    chosen_mag = gmag[order][:n_clean]
    return list(zip(chosen.tolist(), chosen_slots.tolist(), chosen_mag.tolist()))


def step2c_temporal_vs_spatial_variance(fd, out, *, r_lo=R_ANNULUS_LO, r_hi=R_ANNULUS_HI):
    """ANOVA-style decomposition of the pooled annulus chi variance into a
    'spatial' term (variance of each (group,pixel)'s own time-mean chi --
    i.e. a fixed-in-time bias that differs from star/pixel to star/pixel)
    and a 'temporal' term (the average, within a (group,pixel), of the
    variance across the 20 frames -- i.e. genuine frame-to-frame scatter).
    Only (group,pixel) locations valid in *every* frame are used, so the two
    terms are computed on exactly the same population and sum exactly to the
    pooled total variance (law of total variance)."""
    S = fd.data.shape[-1]
    r = _radius_grid(S)
    annulus = (r >= r_lo) & (r < r_hi)
    chi = out["chi"]  # (n_groups, n_frames, S, S)
    w_mask = out["w_mask"]  # (n_groups, n_frames, S, S)
    n_groups, n_frames = chi.shape[0], chi.shape[1]

    always_valid = np.all(w_mask, axis=1) & annulus[None, :, :]  # (n_groups, S, S)
    n_locations_any = int(np.sum(w_mask[:, :, :, :] & annulus[None, None, :, :]))
    n_locations_always = int(np.sum(always_valid)) * n_frames

    time_means = []
    within_vars = []
    for gi in range(n_groups):
        m = always_valid[gi]
        if not m.any():
            continue
        vals = chi[gi][:, m]  # (n_frames, n_loc_g)
        time_means.append(vals.mean(axis=0))  # (n_loc_g,)
        within_vars.append(vals.var(axis=0))  # (n_loc_g,)
    time_means = np.concatenate(time_means)
    within_vars = np.concatenate(within_vars)

    spatial_var = float(np.var(time_means))  # between-(group,pixel) variance of the fixed component
    temporal_var = float(np.mean(within_vars))  # average genuine frame-to-frame variance
    total_var = spatial_var + temporal_var
    return {
        "n_loc": int(time_means.size),
        "n_frames": n_frames,
        "frac_locations_always_valid": n_locations_always / max(n_locations_any, 1),
        "spatial_var": spatial_var,
        "temporal_var": temporal_var,
        "total_var": total_var,
        "spatial_std": float(np.sqrt(spatial_var)),
        "temporal_std": float(np.sqrt(temporal_var)),
        "total_std": float(np.sqrt(total_var)),
        "frac_var_spatial": spatial_var / total_var if total_var > 0 else float("nan"),
    }


def step3_spatial_pattern_stability(out, gi: int, slot: int):
    """Row/column chi vectors across all frames for one group; mean pairwise
    correlation across frames tells us whether the residual *pattern* is a
    fixed shape/position error (high correlation, same pixels always +/-)
    or changes frame to frame (low correlation, real per-frame discrepancy)."""
    chi = out["chi"][gi]  # (n_frames, S, S)
    n_frames, S, _ = chi.shape
    c = S // 2
    row = chi[:, c, :]  # (n_frames, S)
    col = chi[:, :, c]  # (n_frames, S)

    def mean_pairwise_corr(mat):
        # mat: (n_frames, S); drop frames with zero variance (shouldn't happen).
        finite = np.all(np.isfinite(mat), axis=1)
        mat = mat[finite]
        if mat.shape[0] < 2:
            return float("nan")
        C = np.corrcoef(mat)
        iu = np.triu_indices_from(C, k=1)
        return float(np.mean(C[iu]))

    return {
        "row_mean_pairwise_corr": mean_pairwise_corr(row),
        "col_mean_pairwise_corr": mean_pairwise_corr(col),
        "row": row,
        "col": col,
    }


def step4_flux_vs_naive(fd, out, gi: int, slot: int, *, aperture_r: float = NAIVE_APERTURE_R):
    """Compare the analytic flux-solve f(t) to a naive, model-free aperture
    sum over the same stamp, both restricted to r < aperture_r. Reports
    frame-to-frame relative scatter (std/median) of each."""
    S = fd.data.shape[-1]
    r = _radius_grid(S)
    aperture = r < aperture_r
    data_np = np.asarray(fd.data)[gi]  # (n_frames, S, S)
    naive = data_np[:, aperture].sum(axis=-1)  # (n_frames,)
    f_t = out["flux"][gi, :, slot]  # (n_frames,)

    def rel_scatter(x):
        med = np.median(x)
        if med == 0:
            return float("nan")
        return float(np.std(x) / med)

    return {
        "f_t": f_t,
        "naive_t": naive,
        "f_rel_scatter": rel_scatter(f_t),
        "naive_rel_scatter": rel_scatter(naive),
    }


def step5_trajectory_jitter(out, gi: int, slot: int):
    """x(t)/y(t) trajectory for one group's primary: first/second differences
    to spot jumps or high-frequency wiggle vs. a smooth drift."""
    x_t = out["x_slot"][gi, slot, :]
    y_t = out["y_slot"][gi, slot, :]
    dx1 = np.diff(x_t)
    dy1 = np.diff(y_t)
    dx2 = x_t[2:] - 2 * x_t[1:-1] + x_t[:-2]
    dy2 = y_t[2:] - 2 * y_t[1:-1] + y_t[:-2]
    return {
        "x_t": x_t, "y_t": y_t,
        "x_range": float(x_t.max() - x_t.min()),
        "y_range": float(y_t.max() - y_t.min()),
        "std_dx1": float(np.std(dx1)), "std_dy1": float(np.std(dy1)),
        "max_abs_dx1": float(np.max(np.abs(dx1))), "max_abs_dy1": float(np.max(np.abs(dy1))),
        "std_dx2": float(np.std(dx2)), "std_dy2": float(np.std(dy2)),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, nargs="?",
                   default=Path("dev/forward_epsf_wcs/output/runs/scale1_mag911_600"))
    p.add_argument("params_name", type=str, nargs="?", default="params_stage2.npz")
    p.add_argument("--mag-lo", type=float, default=9.0)
    p.add_argument("--mag-hi", type=float, default=11.0)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--region", type=str, default="1536,1536,2048,2048")
    p.add_argument("--n-clean", type=int, default=5)
    p.add_argument("--epsf-grid", type=str, default=None,
                   help="n_rowsxn_cols override; default: infer from the loaded checkpoint's "
                        "epsf_base_raw shape")
    args = p.parse_args(argv)

    print(f"loading params from {args.run_dir / args.params_name} ...")
    params = FIT.load_params_npz(args.run_dir / args.params_name)

    if args.epsf_grid:
        n_rows, n_cols = (int(v) for v in args.epsf_grid.lower().split("x"))
    else:
        n_rows, n_cols = int(params["epsf_base_raw"].shape[0]), int(params["epsf_base_raw"].shape[1])
    print(f"epsf_grid inferred/used: {n_rows}x{n_cols}")

    print(f"rebuilding context (mag {args.mag_lo}-{args.mag_hi}, {args.n_frames} frames)...")
    rebuilt = build_full_context(
        mag_lo=args.mag_lo, mag_hi=args.mag_hi, n_frames=args.n_frames, region_str=args.region,
        epsf_grid=(n_rows, n_cols),
    )
    fd = rebuilt["fd"]
    groups = rebuilt["groups"]
    mags = rebuilt["mags"]
    print(f"n_groups={groups.n_groups} n_frames={args.n_frames}")

    out = _forward_and_flux(fd, params)

    print("\n=== Step 2: per-frame chi stats, annulus r in "
          f"[{R_ANNULUS_LO},{R_ANNULUS_HI}), pooled over all groups ===")
    rows = step2_per_frame_annulus_stats(fd, out)
    print(f"{'t':>3} {'median|chi|':>12} {'std_chi':>10} {'n_pix':>8}")
    for t, medabs, std, n in rows:
        print(f"{t:3d} {medabs:12.3f} {std:10.3f} {n:8d}")
    meds = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])
    print(f"\nacross-frame spread of per-frame median|chi|: "
          f"min={meds.min():.2f} max={meds.max():.2f} median={np.median(meds):.2f} "
          f"(max/median ratio={meds.max()/np.median(meds):.2f})")
    print(f"across-frame spread of per-frame std_chi:      "
          f"min={stds.min():.2f} max={stds.max():.2f} median={np.median(stds):.2f} "
          f"(max/median ratio={stds.max()/np.median(stds):.2f})")

    print("\n=== Step 2c: temporal-vs-spatial variance decomposition, same annulus ===")
    dec = step2c_temporal_vs_spatial_variance(fd, out)
    print(f"n_(group,pixel) locations valid in all {dec['n_frames']} frames: {dec['n_loc']} "
          f"({dec['frac_locations_always_valid']:.1%} of any-frame-valid annulus samples)")
    print(f"spatial (fixed per group,pixel) std : {dec['spatial_std']:.3f}  "
          f"({dec['frac_var_spatial']:.1%} of total variance)")
    print(f"temporal (frame-to-frame) std       : {dec['temporal_std']:.3f}  "
          f"({1 - dec['frac_var_spatial']:.1%} of total variance)")
    print(f"total std (spatial+temporal, sqrt)  : {dec['total_std']:.3f}")

    print("\n=== Step 2b: per-frame MAD-reject rate (n_sigma=3.0, full stamp) ===")
    rej_rows = step2_with_reject(fd, params)
    print(f"{'t':>3} {'n_rej':>6} {'n_cand':>7} {'frac':>7}")
    for t, n_rej, n_cand, frac in rej_rows:
        print(f"{t:3d} {n_rej:6d} {n_cand:7d} {frac:7.3f}")

    print(f"\n=== Steps 3-5: clean (n_members==1) brightest {args.n_clean} groups ===")
    clean = pick_clean_groups(groups, mags, args.n_clean)
    for gi, slot, mag in clean:
        print(f"\n--- group {gi} (mag={mag:.2f}, slot={slot}) ---")
        s3 = step3_spatial_pattern_stability(out, gi, slot)
        print(f"  step3 spatial-pattern stability: "
              f"row mean pairwise corr={s3['row_mean_pairwise_corr']:.3f}, "
              f"col mean pairwise corr={s3['col_mean_pairwise_corr']:.3f}")

        s4 = step4_flux_vs_naive(fd, out, gi, slot)
        print(f"  step4 flux vs naive aperture (r<{NAIVE_APERTURE_R}): "
              f"f(t) rel scatter={s4['f_rel_scatter']:.4f}, "
              f"naive rel scatter={s4['naive_rel_scatter']:.4f}, "
              f"ratio f/naive={s4['f_rel_scatter']/s4['naive_rel_scatter']:.2f}")

        s5 = step5_trajectory_jitter(out, gi, slot)
        print(f"  step5 trajectory: x_range={s5['x_range']:.4f}px y_range={s5['y_range']:.4f}px "
              f"std(dx1)={s5['std_dx1']:.5f} std(dy1)={s5['std_dy1']:.5f} "
              f"max|dx1|={s5['max_abs_dx1']:.5f} max|dy1|={s5['max_abs_dy1']:.5f} "
              f"std(dx2)={s5['std_dx2']:.5f} std(dy2)={s5['std_dy2']:.5f}")


if __name__ == "__main__":
    main()
