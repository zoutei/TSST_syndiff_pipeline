# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Forced photometry of isolated Gaia stars with a frozen WCS+ePSF fit.

Selects stars in a tess_mag window that are isolated by a minimum pixel
separation from brighter/contaminating Gaia neighbors, cuts stamps from the
same hp_d stack used for the fit, solves fluxes with the frozen forward
model, and applies a previously-fit aperture field A(x,y,t).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import aperture_correction as AC
from . import cheb_wcs as CW
from . import fit as FIT
from . import fit_bundle as FB
from . import flux_solve as FS
from . import groups as G
from . import loss as L
from .data import (
    RegionSpec,
    list_orbit_frames,
    select_middle_frames,
)
from .diagnostics.run_labels import resolve_run_id
from .diagnostics.export_fits import _infer_frame_offset, paste_stamps


@dataclass
class IsolatedTargetSet:
    """Catalog + reference-frame positions for isolated forced-phot targets."""

    stars: pd.DataFrame  # rows kept; includes ra, dec, tess_mag, source_id, x_ref, y_ref
    nn_dist_px: np.ndarray  # (n,) distance to nearest contaminating neighbor
    primary_index_set: set[int]  # all indices 0..n-1 (single-star groups)
    region: RegionSpec
    mag_lo: float
    mag_hi: float
    min_sep_px: float
    neighbor_mag_max: float


def _frames_for_meta(meta: dict, *, frame_offset: str | None = None):
    ws = Path(meta["workspace"])
    sector = int(meta["sector"])
    orbit_index = int(meta.get("orbit_index", 1))
    n_frames = int(meta["n_frames"])
    frames_all, _ = list_orbit_frames(ws, sector=sector, orbit_index=orbit_index)
    offset = frame_offset or _infer_frame_offset(meta)
    frames = frames_all[:n_frames] if offset == "start" else select_middle_frames(frames_all, n_frames)
    if len(frames) != n_frames:
        raise RuntimeError(f"listed {len(frames)} frames, meta expects {n_frames}")
    return frames


def select_isolated_stars(
    *,
    gaia: pd.DataFrame,
    x_all: np.ndarray,
    y_all: np.ndarray,
    region: RegionSpec,
    mag_lo: float = 11.0,
    mag_hi: float = 12.0,
    min_sep_px: float = 10.0,
    neighbor_mag_max: float = 13.0,
    edge_margin_px: float = 8.0,
) -> IsolatedTargetSet:
    """Keep mag-window stars whose nearest brighter Gaia neighbor is ≥ min_sep_px.

    Isolation neighbors are Gaia with ``tess_mag < neighbor_mag_max`` (default 13:
    brighter than 13th mag). ``x_all, y_all`` are reference-frame detector
    positions for every Gaia row (same length/order as ``gaia``). Targets must
    also sit inset from the region edge by ``edge_margin_px`` so stamps fit.
    """
    mag = gaia["tess_mag"].to_numpy(dtype=float)
    finite = np.isfinite(mag) & np.isfinite(x_all) & np.isfinite(y_all)
    in_region = (
        (x_all >= region.x_min + edge_margin_px)
        & (x_all < region.x_max - edge_margin_px)
        & (y_all >= region.y_min + edge_margin_px)
        & (y_all < region.y_max - edge_margin_px)
    )
    cand = finite & in_region & (mag >= mag_lo) & (mag <= mag_hi)
    neigh = finite & in_region & (mag < neighbor_mag_max)

    xy_n = np.column_stack([x_all[neigh], y_all[neigh]])
    if xy_n.shape[0] < 2:
        raise RuntimeError("not enough Gaia neighbors to assess isolation")
    tree = cKDTree(xy_n)
    # k=2: nearest neighbor that isn't the star itself (exact self-match → dist 0)
    dist, _ = tree.query(np.column_stack([x_all[cand], y_all[cand]]), k=2)
    nn = dist[:, 1] if dist.ndim == 2 else np.full(int(cand.sum()), np.inf)
    keep_local = nn >= float(min_sep_px)

    cand_idx = np.flatnonzero(cand)[keep_local]
    stars = gaia.iloc[cand_idx].copy().reset_index(drop=True)
    stars["x_ref"] = x_all[cand_idx]
    stars["y_ref"] = y_all[cand_idx]
    nn_kept = nn[keep_local]
    return IsolatedTargetSet(
        stars=stars,
        nn_dist_px=np.asarray(nn_kept, dtype=float),
        primary_index_set=set(range(len(stars))),
        region=region,
        mag_lo=float(mag_lo),
        mag_hi=float(mag_hi),
        min_sep_px=float(min_sep_px),
        neighbor_mag_max=float(neighbor_mag_max),
    )


def build_singleton_groups(n_stars: int) -> G.GroupSet:
    """One star per group, K=1."""
    members = np.arange(n_stars, dtype=int).reshape(n_stars, 1)
    valid = np.ones((n_stars, 1), dtype=bool)
    kept = np.ones(n_stars, dtype=bool)
    return G.GroupSet(
        n_groups=n_stars,
        max_group_size=1,
        members=members,
        valid=valid,
        kept_star_mask=kept,
        dropped_oversized=0,
    )


def forced_phot_isolated(
    *,
    fit_bundle: FB.FitBundle,
    params: dict,
    targets: IsolatedTargetSet,
    frames,
    frame_imgs,
    aperture_coeff_path: Path | None = None,
    region_margin: RegionSpec | None = None,
    stamp_physical: int | None = None,
    ref_frame_index: int | None = None,
) -> dict:
    """Cut stamps, forced-phot with frozen WCS+ePSF, optionally apply A."""
    stamp = int(stamp_physical or fit_bundle.stamp_physical)
    region = targets.region
    if region_margin is None:
        margin = 8
        region_margin = RegionSpec(
            region.x_min - margin, region.y_min - margin,
            region.x_max + margin, region.y_max + margin,
        )
    meta = dict(fit_bundle.meta)
    ref_i = int(meta.get("ref_frame_index", len(frames) // 2) if ref_frame_index is None else ref_frame_index)

    stars = targets.stars
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    mags = stars["tess_mag"].to_numpy(dtype=float)
    source_ids = stars["source_id"].to_numpy(dtype=np.int64) if "source_id" in stars.columns else None

    # Positions from frozen WCS for stamp centers (ref frame) + photometry (all frames)
    x_lin, y_lin, cheb_basis = CW.star_basis(
        jnp.asarray(ra, dtype=jnp.float32),
        jnp.asarray(dec, dtype=jnp.float32),
        fit_bundle.cheb_static,
    )
    x_t, y_t = CW.eval_all_positions(
        x_lin, y_lin, cheb_basis,
        params["wcs_coeff"],
        jnp.asarray(fit_bundle.wcs_frame_basis, dtype=jnp.float32),
        fit_bundle.cheb_static.n_terms,
    )
    x_np = np.asarray(x_t)
    y_np = np.asarray(y_t)
    x_ref = x_np[:, ref_i]
    y_ref = y_np[:, ref_i]
    cx = np.rint(x_ref).astype(np.int64)
    cy = np.rint(y_ref).astype(np.int64)

    groups = build_singleton_groups(len(stars))
    groups, cx, cy, off_stats = G.drop_groups_off_array(
        groups, cx, cy,
        array_origin=(region_margin.x_min, region_margin.y_min),
        array_shape=frame_imgs[0].cal.shape,
        stamp=stamp,
        n_stars=len(stars),
    )
    if groups.n_groups == 0:
        raise RuntimeError("all isolated targets dropped (stamps off array)")

    # drop_groups_off_array keeps original member indices into the star list.
    star_order = np.array(
        [int(groups.members[gi, 0]) for gi in range(groups.n_groups)], dtype=int,
    )

    stars_kept = stars.iloc[star_order].reset_index(drop=True)
    mags_kept = mags[star_order]
    source_ids_kept = None if source_ids is None else source_ids[star_order]
    x_lin_k = np.asarray(x_lin)[star_order]
    y_lin_k = np.asarray(y_lin)[star_order]
    cheb_k = np.asarray(cheb_basis)[star_order]
    # Remap members to dense 0..n_groups-1
    members = np.arange(groups.n_groups, dtype=int).reshape(groups.n_groups, 1)
    groups = G.GroupSet(
        n_groups=groups.n_groups,
        max_group_size=1,
        members=members,
        valid=np.ones((groups.n_groups, 1), dtype=bool),
        kept_star_mask=np.ones(groups.n_groups, dtype=bool),
        dropped_oversized=0,
    )
    nn_kept = targets.nn_dist_px[star_order]

    stamp_batch = G.extract_stamps(
        groups, frame_imgs, x_ref[star_order], y_ref[star_order],
        array_origin=(region_margin.x_min, region_margin.y_min),
        stamp=stamp,
        stamp_center_x=cx, stamp_center_y=cy,
    )

    r_fit = L.fit_radius_from_mag(mags_kept, stage=2)
    stamp_w = L.soft_snr_stamp_weights(mags_kept, snr_cap_mag=9.0, w_min=0.05)
    mask_active = np.ones((groups.n_groups, len(frames)), dtype=np.float32)

    ctx = L.build_static_context(
        bp_rp=getattr(fit_bundle, "bp_rp", None),
        colour_ref=L.colour_ref_from_bundle(fit_bundle),
        cheb_static=fit_bundle.cheb_static,
        wcs_frame_basis=fit_bundle.wcs_frame_basis,
        w_frame_basis=fit_bundle.w_frame_basis,
        epsf_grid=fit_bundle.epsf_grid,
        groups=groups,
        ra=stars_kept["ra"].to_numpy(dtype=float),
        dec=stars_kept["dec"].to_numpy(dtype=float),
        stamp_center_x=stamp_batch.stamp_center_x,
        stamp_center_y=stamp_batch.stamp_center_y,
        t_exp_sec=fit_bundle.t_exp_sec,
        stamp_snr_weight=stamp_w,
        fit_radius=r_fit,
        stamp_active=mask_active,
        x_lin=x_lin_k,
        y_lin=y_lin_k,
        cheb_basis=cheb_k,
    )

    table = AC.extract_primary_fluxes(
        params, ctx, stamp_batch.data, stamp_batch.noise, stamp_batch.weight,
        primary_index_set=set(range(groups.n_groups)),
        mags=mags_kept,
        source_ids=source_ids_kept,
        btjd=np.array([f.btjd for f in frames], dtype=np.float64),
        stamp_active=mask_active,
    )

    # Recompute model stamps for residual export
    templates, x_slot, y_slot, _ = L.forward_model(params, ctx, n_pix=stamp)
    var = L.pixel_variance(stamp_batch.noise)
    rmask = L.radius_pixel_mask(ctx.fit_radius, stamp=stamp)
    pix_w = stamp_batch.weight * rmask[:, None, :, :]
    iv = L.inverse_variance_weights(pix_w, var)
    flux = FS.solve_group_fluxes(templates, stamp_batch.data, iv, ridge=1e-6)
    model = FS.model_stamps(templates, flux)

    A_star = None
    flux_corr = None
    A_node = None
    if aperture_coeff_path is not None:
        ac = np.load(aperture_coeff_path)
        Phi = np.asarray(fit_bundle.w_frame_basis, dtype=np.float64)
        if "A_node" in ac.files:
            A_node = np.asarray(ac["A_node"], dtype=np.float64)
            if A_node.shape[-1] != Phi.shape[0]:
                A_node = AC.decode_aperture(np.asarray(ac["A_coeff"]), Phi)
        else:
            A_node = AC.decode_aperture(np.asarray(ac["A_coeff"]), Phi)
        node_x = np.asarray(ac["node_x"]) if "node_x" in ac.files else np.asarray(ctx.node_x)
        node_y = np.asarray(ac["node_y"]) if "node_y" in ac.files else np.asarray(ctx.node_y)
        A_star = AC.eval_astar(A_node, table.x, table.y, node_x, node_y)
        flux_corr = table.flux / np.clip(A_star, 1e-6, None)

    return {
        "table": table,
        "flux_corr": flux_corr,
        "A_star": A_star,
        "A_node": A_node,
        "stamp_batch": stamp_batch,
        "model": np.asarray(model),
        "data": np.asarray(stamp_batch.data),
        "groups": groups,
        "stars": stars_kept,
        "nn_dist_px": nn_kept,
        "off_stats": off_stats,
        "ctx": ctx,
        "frames": frames,
        "x_slot": np.asarray(x_slot),
        "y_slot": np.asarray(y_slot),
        "fit_radius": r_fit,
        "region": region,
        "region_margin": region_margin,
        "stamp_physical": stamp,
        "ref_frame_index": ref_i,
    }


def save_isolated_outputs(data_dir: Path, result: dict, *, meta: dict) -> Path:
    """Write npz + catalog csv under ``data_dir``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    table: AC.PrimaryFluxTable = result["table"]
    payload = {
        "flux_ls": table.flux,
        "sigma_f": table.sigma_f,
        "x": table.x,
        "y": table.y,
        "active": table.active,
        "tess_mag": table.tess_mag,
        "star_index": table.star_index,
        "nn_dist_px": result["nn_dist_px"],
        "stamp_center_x": result["stamp_batch"].stamp_center_x,
        "stamp_center_y": result["stamp_batch"].stamp_center_y,
        "fit_radius": result["fit_radius"],
    }
    if table.source_id is not None:
        payload["source_id"] = table.source_id
    if table.btjd is not None:
        payload["btjd"] = table.btjd
    if result["flux_corr"] is not None:
        payload["flux_corr"] = result["flux_corr"]
        payload["A_star"] = result["A_star"]
    np.savez_compressed(data_dir / "isolated_flux_table.npz", **payload)

    # Compact residual example (middle frame stamps) for notebook reload
    fi = int(result["ref_frame_index"])
    np.savez_compressed(
        data_dir / "isolated_stamps_example.npz",
        data=result["data"][:, fi],
        model=result["model"][:, fi],
        residual=result["data"][:, fi] - result["model"][:, fi],
        stamp_center_x=result["stamp_batch"].stamp_center_x,
        stamp_center_y=result["stamp_batch"].stamp_center_y,
        frame_index=fi,
        stem=str(result["frames"][fi].stem),
        btjd=float(result["frames"][fi].btjd),
        stamp_physical=int(result["stamp_physical"]),
        region=np.asarray(
            [result["region"].x_min, result["region"].y_min,
             result["region"].x_max, result["region"].y_max],
            dtype=np.int32,
        ),
    )

    cat = result["stars"].copy()
    cat["nn_dist_px"] = result["nn_dist_px"]
    cat["stamp_center_x"] = result["stamp_batch"].stamp_center_x
    cat["stamp_center_y"] = result["stamp_batch"].stamp_center_y
    cat.to_csv(data_dir / "isolated_targets.csv", index=False)

    summary = {
        **meta,
        "n_targets": int(len(result["stars"])),
        "n_frames": int(table.flux.shape[1]),
        "off_array_drop": result["off_stats"],
        "has_aperture_corr": result["flux_corr"] is not None,
        "mag_lo": float(meta.get("mag_lo", np.nan)),
        "mag_hi": float(meta.get("mag_hi", np.nan)),
        "min_sep_px": float(meta.get("min_sep_px", np.nan)),
    }
    if result["flux_corr"] is not None:
        before = AC.summarize_flux_trend(table.flux, table.active)
        after = AC.summarize_flux_trend(result["flux_corr"], table.active)
        summary["before_ptp_mean_frac"] = before["ptp_mean_frac"]
        summary["after_ptp_mean_frac"] = after["ptp_mean_frac"]
        summary["before_pc1_ve"] = before["pc1_variance_explained"]
        summary["after_pc1_ve"] = after["pc1_variance_explained"]
    import json
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return data_dir


def _load_training_stamp_active(run_dir: Path, bundle_path: Path) -> np.ndarray | None:
    """Final training keep-mask (G, T), or ``None`` if not on disk."""
    run_dir = Path(run_dir)
    for name in ("stamp_active.npz", "stamp_active_latest.npz"):
        path = run_dir / name
        if path.is_file():
            return FIT.load_stamp_active_npz(path)
    bundle = FB.load_fit_bundle(bundle_path)
    return np.asarray(bundle.mask_active, dtype=np.float32)


def _training_reject_rects_region_local(
    stamp_active: np.ndarray,
    bundle: FB.FitBundle,
    frame_index: int,
    region: RegionSpec,
    stamp: int,
) -> list[tuple[float, float, float, float]]:
    """Rejected training stamps as region-local ``(x0, y0, width, height)``."""
    fi = int(frame_index)
    if stamp_active.shape[0] != len(bundle.stamp_center_x) or fi >= stamp_active.shape[1]:
        return []
    half = int(stamp) // 2
    cx = np.asarray(bundle.stamp_center_x, dtype=float)
    cy = np.asarray(bundle.stamp_center_y, dtype=float)
    rej = np.asarray(stamp_active[:, fi], dtype=np.float32) <= 0.5
    rects: list[tuple[float, float, float, float]] = []
    for gi in np.flatnonzero(rej):
        x0 = float(cx[gi]) - region.x_min - half
        y0 = float(cy[gi]) - region.y_min - half
        rects.append((x0, y0, float(stamp), float(stamp)))
    return rects


def export_isolated_figures(
    result: dict,
    out_dir: Path,
    *,
    run_id: str,
    max_stars_plot: int = 40,
    run_dir: Path | None = None,
    bundle_path: Path | None = None,
) -> dict[str, Path]:
    if run_dir is not None:
        run_id = resolve_run_id(run_dir, run_id)
    """Write LC + residual panel PNGs into ``out_dir`` (fits_export)."""
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    from astropy.visualization import AsinhStretch, ImageNormalize

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table: AC.PrimaryFluxTable = result["table"]
    t = table.btjd
    flux = table.flux
    corr = result["flux_corr"]
    sig = table.sigma_f
    A = result["A_star"]
    active = table.active
    mag = table.tess_mag
    sid = table.source_id
    paths: dict[str, Path] = {}

    # Population trend
    before = AC.summarize_flux_trend(flux, active)
    after = AC.summarize_flux_trend(corr, active) if corr is not None else None
    fig, axes = plt.subplots(1, 2 if after else 1, figsize=(11 if after else 6, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(t, before["mean_frac"], "o-", ms=2.5, color="0.35",
                 label=f"raw ptp={before['ptp_mean_frac']:.4f}")
    axes[0].fill_between(t, before["mean_frac"] - before["sem_frac"],
                         before["mean_frac"] + before["sem_frac"], color="0.35", alpha=0.18)
    axes[0].axhline(0, color="k", lw=0.5, alpha=0.4)
    axes[0].set_title(r"Raw $f_{\mathrm{LS}}$ (isolated mag window)")
    axes[0].set_xlabel("BTJD")
    axes[0].set_ylabel("mean fractional residual")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    if after is not None:
        axes[1].plot(t, after["mean_frac"], "s-", ms=2.5, color="C3",
                     label=f"corr ptp={after['ptp_mean_frac']:.4f}")
        axes[1].fill_between(t, after["mean_frac"] - after["sem_frac"],
                             after["mean_frac"] + after["sem_frac"], color="C3", alpha=0.18)
        axes[1].axhline(0, color="k", lw=0.5, alpha=0.4)
        axes[1].set_title(r"Aperture-corrected $f_{\mathrm{corr}}=f_{\mathrm{LS}}/A_\star$")
        axes[1].set_xlabel("BTJD")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{run_id}: isolated forced phot  n={flux.shape[0]}", y=1.02)
    fig.tight_layout()
    paths["population"] = out_dir / "isolated_lc_population.png"
    fig.savefig(paths["population"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-star (brightest first / limited count)
    order = np.argsort(mag)[:max_stars_plot]
    n_show = len(order)
    ncols = 4
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for panel, s in enumerate(order):
        ax = axes[panel]
        m = active[s] & np.isfinite(flux[s]) & (flux[s] > 0)
        if corr is not None:
            m &= np.isfinite(corr[s]) & np.isfinite(A[s]) & (A[s] > 0)
        if m.sum() < 2:
            ax.set_visible(False)
            continue
        w = 1.0 / np.clip(sig[s, m], 1e-12, None) ** 2
        f0 = float(np.sum(w * flux[s, m]) / np.sum(w))
        ax.plot(t[m], flux[s, m] / f0, "o", ms=2.0, color="0.65", alpha=0.65, label="raw", zorder=1)
        if corr is not None:
            yerr = (sig[s, m] / f0) / np.clip(A[s, m], 1e-6, None)
            ax.errorbar(t[m], corr[s, m] / f0, yerr=yerr, fmt="s-", ms=2.2, lw=0.8,
                        color="C3", alpha=0.9, label="corrected", zorder=2)
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.35)
        lab = f"mag={mag[s]:.2f} nn={result['nn_dist_px'][s]:.1f}px"
        if sid is not None:
            lab = f"id={int(sid[s])} " + lab
        ax.set_title(lab, fontsize=7)
        ax.grid(True, alpha=0.25)
        if panel == 0:
            ax.legend(fontsize=7)
    for ax in axes[n_show:]:
        ax.set_visible(False)
    fig.suptitle(f"{run_id}: isolated per-star LCs (up to {max_stars_plot})", y=1.01)
    fig.tight_layout()
    paths["per_star"] = out_dir / "isolated_lc_per_star.png"
    fig.savefig(paths["per_star"], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Example residual mosaic (middle frame)
    fi = int(result["ref_frame_index"])
    region = result["region"]
    ny, nx = region.shape
    S = int(result["stamp_physical"])
    cx = result["stamp_batch"].stamp_center_x
    cy = result["stamp_batch"].stamp_center_y
    data_m = paste_stamps(result["data"][:, fi], cx, cy,
                          x_min=region.x_min, y_min=region.y_min, ny=ny, nx=nx, stamp=S)
    model_m = paste_stamps(result["model"][:, fi], cx, cy,
                           x_min=region.x_min, y_min=region.y_min, ny=ny, nx=nx, stamp=S)
    resid_m = paste_stamps(result["data"][:, fi] - result["model"][:, fi], cx, cy,
                           x_min=region.x_min, y_min=region.y_min, ny=ny, nx=nx, stamp=S)
    stem = result["frames"][fi].stem
    btjd = float(result["frames"][fi].btjd)
    finite = np.isfinite(data_m) & np.isfinite(model_m)
    if finite.any():
        vmin, vmax = np.nanpercentile(data_m[finite], [1, 99])
    else:
        vmin, vmax = -1.0, 1.0
    m = np.isfinite(resid_m)
    rv = float(np.nanpercentile(np.abs(resid_m[m]), 98)) if m.any() else 1.0
    rv = max(rv, 1e-3)
    asinh = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.1))
    reject_rects: list[tuple[float, float, float, float]] = []
    if run_dir is not None and bundle_path is not None:
        stamp_active = _load_training_stamp_active(run_dir, bundle_path)
        if stamp_active is not None:
            bundle = FB.load_fit_bundle(bundle_path)
            reject_rects = _training_reject_rects_region_local(
                stamp_active, bundle, fi, region, S,
            )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, arr, title, cmap, norm in [
        (axes[0], data_m, "hp_d stamps", "gray", asinh),
        (axes[1], model_m, "frozen WCS+ePSF model", "gray", asinh),
        (axes[2], resid_m, "residual (data−model)", "coolwarm",
         TwoSlopeNorm(vcenter=0.0, vmin=-rv, vmax=rv)),
    ]:
        im = ax.imshow(arr, origin="lower", cmap=cmap, norm=norm,
                       interpolation="nearest", aspect="equal")
        for x0, y0, rw, rh in reject_rects:
            ax.add_patch(Rectangle(
                (x0, y0), rw, rh, fill=False, ec="orange", lw=1.2, alpha=0.95, zorder=10,
            ))
        ax.set_title(title)
        ax.set_xlabel("x (region-local)")
        ax.set_ylabel("y (region-local)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{run_id} isolated  {stem}  BTJD={btjd:.5f}  frame={fi}")
    paths["residual"] = out_dir / f"isolated_residual_{stem}.png"
    fig.savefig(paths["residual"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    return paths
