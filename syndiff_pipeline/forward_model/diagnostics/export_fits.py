# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Export hp_d / model / residual mosaics as FITS + a DS9 membership region file
for a saved checkpoint.

Two ways to rebuild the fit context:

1. ``--from-bundle`` (preferred for real runs): load the exact stamps + static
   model that produced the checkpoint. No hp_d/centroids/Gaia reload.
2. Legacy rebuild via ``build_full_context`` (smoke / mag7-9 defaults).

Adapted from the mosaic/FITS + DS9-region cells in
``notebooks/results_smoke_mag79_companions.ipynb``.

Usage:
    python -m syndiff_pipeline.forward_model.diagnostics.export_fits <run_dir> params_latest.npz \\
        --from-bundle dev/forward_epsf_wcs/output/bundles/orbit1_half_mag710 \\
        [--frame-index N] [--out-dir DIR] [--png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .. import fit as FIT
from .. import fit_bundle as FB
from .. import post_fit as DIAG
from .run_labels import resolve_run_id
from .. import flux_solve as FS
from .. import loss as L
from .. import stamp_reject as SR
from .. import temporal as T
from ..data import RegionSpec, list_orbit_frames, load_frame_region, select_middle_frames
from .radial_profile import build_full_context


def pixel_weights(fd: FIT.FitData) -> jnp.ndarray:
    """Active pixel weights for flux solve (square S×S or packed P)."""
    if bool(fd.ctx.is_packed):
        return fd.weight * fd.ctx.pix_valid[:, None, :]
    rmask = L.radius_pixel_mask(fd.ctx.fit_radius, stamp=int(fd.data.shape[-1]))
    return fd.weight * rmask[:, None, :, :]


def paste_packed_stamps(
    values_gp: np.ndarray,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    pix_valid: np.ndarray,
    *,
    x_min: int,
    y_min: int,
    ny: int,
    nx: int,
) -> np.ndarray:
    """Mean-accumulate packed (G,P) stamps into a region mosaic."""
    acc = np.zeros((ny, nx), dtype=np.float64)
    wsum = np.zeros((ny, nx), dtype=np.float64)
    for gi in range(values_gp.shape[0]):
        m = np.asarray(pix_valid[gi]) > 0
        if not np.any(m):
            continue
        ix = np.rint(np.asarray(pix_x[gi][m]) - x_min).astype(np.int64)
        iy = np.rint(np.asarray(pix_y[gi][m]) - y_min).astype(np.int64)
        vals = np.asarray(values_gp[gi][m], dtype=np.float64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.add.at(acc, (iy[ok], ix[ok]), vals[ok])
        np.add.at(wsum, (iy[ok], ix[ok]), 1.0)
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    hit = wsum > 0
    out[hit] = (acc[hit] / wsum[hit]).astype(np.float32)
    return out


def paste_stamps(
    stamps: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    x_min: int,
    y_min: int,
    ny: int,
    nx: int,
    stamp: int,
) -> np.ndarray:
    """Mean-accumulate stamps into a region mosaic; NaN outside stamp footprints."""
    half = stamp // 2
    acc = np.zeros((ny, nx), dtype=np.float64)
    wsum = np.zeros((ny, nx), dtype=np.float64)
    for gi in range(stamps.shape[0]):
        x0 = int(cx[gi]) - x_min - half
        y0 = int(cy[gi]) - y_min - half
        x1, y1 = x0 + stamp, y0 + stamp
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            continue
        acc[y0:y1, x0:x1] += stamps[gi]
        wsum[y0:y1, x0:x1] += 1.0
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    m = wsum > 0
    out[m] = (acc[m] / wsum[m]).astype(np.float32)
    return out


def write_mosaic_fits(
    path: Path, data: np.ndarray, *,
    region, sector: int, camera: int, ccd: int,
    stem: str, btjd: float, frame_idx: int, run_id: str, kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["STEM"] = stem
    hdr["BTJD"] = float(btjd)
    hdr["FRAMEIDX"] = int(frame_idx)
    hdr["RUN_ID"] = run_id
    hdr["KIND"] = kind
    hdr["XMIN"] = int(region.x_min)
    hdr["YMIN"] = int(region.y_min)
    hdr["XMAX"] = int(region.x_max)
    hdr["YMAX"] = int(region.y_max)
    hdr["SECTOR"] = sector
    hdr["CAMERA"] = camera
    hdr["CCD"] = ccd
    hdr["BUNIT"] = "e-/s"
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=hdr).writeto(path, overwrite=True)


def export_full_hp_d(
    frame, region: RegionSpec, *,
    out_dir: Path, sector: int, camera: int, ccd: int, frame_idx: int, run_id: str,
) -> Path:
    """Write the real observed hp_d cal plane for ``frame``, cropped to
    ``region`` (the same crop the bundle/fit was built from).

    Unlike ``hp_d_{stem}.fits`` (``paste_stamps``/``paste_packed_stamps``:
    mean-accumulated stamp pixels only, NaN everywhere a group's stamp/
    aperture never touched), this is the full crop -- every pixel in
    ``region``, including ones no stamp ever used (masked out, outside any
    group's footprint, rejected companions, etc).
    """
    img = load_frame_region(frame, region)
    path = out_dir / f"hp_d_full_{frame.stem}.fits"
    write_mosaic_fits(
        path, img.cal, region=region, sector=sector, camera=camera, ccd=ccd,
        stem=frame.stem, btjd=frame.btjd, frame_idx=frame_idx, run_id=run_id,
        kind="hp_d_full",
    )
    return path


def save_panel_png(
    path: Path,
    *,
    hp_d: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    stem: str,
    btjd: float,
    frame_idx: int,
    n_frames: int,
    run_id: str,
) -> None:
    """Notebook-style 1×3 hp_d / model / residual preview."""
    finite_dm = np.isfinite(hp_d) & np.isfinite(model)
    if finite_dm.any():
        vmin, vmax = np.nanpercentile(hp_d[finite_dm], [1, 99])
    else:
        vmin, vmax = -1.0, 1.0
    m = np.isfinite(residual)
    rv = float(np.nanpercentile(np.abs(residual[m]), 98)) if m.any() else 1.0
    rv = max(rv, 1e-3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    asinh = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.1))
    panels = [
        (axes[0], hp_d, "hp_d (input)", "gray", asinh),
        (axes[1], model, "forward model", "gray", asinh),
        (axes[2], residual, "residual (data−model)", "coolwarm",
         TwoSlopeNorm(vcenter=0.0, vmin=-rv, vmax=rv)),
    ]
    for ax, arr, title, cmap, norm in panels:
        im = ax.imshow(arr, origin="lower", cmap=cmap, norm=norm,
                       interpolation="nearest", aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x (region-local)")
        ax.set_ylabel("y (region-local)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{run_id}  {stem}  BTJD={btjd:.5f}  frame={frame_idx}/{n_frames - 1}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ctx_from_bundle(bundle: FB.FitBundle, *, frame_indices=None) -> L.StaticContext:
    """Static WCS/ePSF context only -- never touches bundle.data/noise/weight,
    which for a tier-segmented bundle force a dense (G, T, p_max) reconstruction
    padded to the single largest tier across the *whole* bundle
    (FitBundle._dense_stamp_array) -- multi-GB to 100+ GB at full-orbit scale
    for callers that only need WCS/ePSF context, not per-frame pixel data.

    ``frame_indices``, if given, bounds wcs_frame_basis/w_frame_basis/
    stamp_active to those frame columns -- the returned context's frame axis
    is then LOCAL positions ``0..len(frame_indices)-1`` in that order, not
    the original global frame numbers.
    """
    groups = bundle.group_set()
    wcs_fb = np.asarray(bundle.wcs_frame_basis)
    w_fb = np.asarray(bundle.w_frame_basis)
    mask_active = np.asarray(bundle.mask_active)
    if frame_indices is not None:
        wcs_fb = wcs_fb[frame_indices]
        w_fb = w_fb[frame_indices]
        mask_active = mask_active[:, frame_indices]
    ctx_kw: dict = dict(
        bp_rp=getattr(bundle, "bp_rp", None),
        colour_ref=L.colour_ref_from_bundle(bundle),
        cheb_static=bundle.cheb_static,
        wcs_frame_basis=wcs_fb,
        w_frame_basis=w_fb,
        epsf_grid=bundle.epsf_grid,
        groups=groups,
        ra=bundle.ra,
        dec=bundle.dec,
        stamp_center_x=bundle.stamp_center_x,
        stamp_center_y=bundle.stamp_center_y,
        t_exp_sec=bundle.t_exp_sec,
        stamp_snr_weight=bundle.stamp_snr_weight,
        fit_radius=bundle.fit_radius_stage23,
        stamp_active=mask_active,
        x_lin=bundle.x_lin,
        y_lin=bundle.y_lin,
        cheb_basis=bundle.cheb_basis,
    )
    if bundle.is_packed:
        ctx_kw["pix_x"] = bundle.pix_x
        ctx_kw["pix_y"] = bundle.pix_y
        ctx_kw["pix_valid"] = bundle.pix_valid
    return L.build_static_context(**ctx_kw)


def _fd_from_bundle(bundle: FB.FitBundle, *, frame_indices=None) -> FIT.FitData:
    """FitData over ``frame_indices`` frame columns (default: the full frame
    set, stage-2/3 radii).

    For a tier-segmented bundle, bounding ``frame_indices`` keeps the dense
    (G, T, p_max) stamp reconstruction (FitBundle._dense_stamp_array) to
    T = len(frame_indices) instead of the bundle's full frame count -- the
    difference between ~tens of MB and tens of GB per array at full-orbit
    scale. The returned FitData's frame axis is then LOCAL positions
    ``0..len(frame_indices)-1``, matching ``frame_indices``' order -- callers
    that need the original global frame number for labeling/output must keep
    their own ``enumerate(frame_indices)`` mapping.
    """
    ctx = _ctx_from_bundle(bundle, frame_indices=frame_indices)
    wcs_n = int(bundle.wcs_frame_basis.shape[1])
    w_n = int(bundle.w_frame_basis.shape[1])
    if frame_indices is not None and bundle.packed_tiers is not None:
        # Tier-segmented storage: build the dense reconstruction bounded to
        # just these frame columns instead of via the unbounded `bundle.data`
        # cached_property (see FitBundle._dense_stamp_array).
        data = jnp.asarray(bundle._dense_stamp_array("data", frame_indices=frame_indices))
        noise = jnp.asarray(bundle._dense_stamp_array("noise", frame_indices=frame_indices))
        weight = jnp.asarray(bundle._dense_stamp_array("weight", frame_indices=frame_indices))
    elif frame_indices is not None:
        data = jnp.asarray(np.asarray(bundle.data)[:, frame_indices])
        noise = jnp.asarray(np.asarray(bundle.noise)[:, frame_indices])
        weight = jnp.asarray(np.asarray(bundle.weight)[:, frame_indices])
    else:
        data = jnp.asarray(bundle.data)
        noise = jnp.asarray(bundle.noise)
        weight = jnp.asarray(bundle.weight)
    return FIT.FitData(
        ctx=ctx,
        data=data,
        noise=noise,
        weight=weight,
        wcs_second_diff=T.second_difference_matrix(wcs_n),
        w_second_diff=T.second_difference_matrix(w_n),
        epsf_modes_init=jnp.asarray(bundle.epsf_modes),
        mask_stamp_active=np.asarray(ctx.stamp_active, dtype=np.float32),
    )


def _infer_frame_offset(meta: dict) -> str:
    """Guess start vs middle when meta omitted ``frame_offset``.

    Half-orbit ``tau_cut`` + ``knots_anchor=full-orbit`` runs use the leading
    window (``--frame-offset start``); see ``export_fit_bundle`` example.
    """
    if "frame_offset" in meta:
        return str(meta["frame_offset"])
    if meta.get("knots_anchor") == "full-orbit" and meta.get("tau_cut") is not None:
        return "start"
    return "middle"


def _frame_records_from_meta(meta: dict, *, frame_offset: str | None = None) -> list:
    """FrameRecord list (stem, btjd, hp_d_path, ...) for the frames that went
    into the bundle / run -- same selection ``export_fit_bundle`` used."""
    ws = Path(meta["workspace"])
    sector = int(meta["sector"])
    orbit_index = int(meta.get("orbit_index", 1))
    n_frames = int(meta["n_frames"])
    frames_all, _ = list_orbit_frames(ws, sector=sector, orbit_index=orbit_index)
    offset = frame_offset or _infer_frame_offset(meta)
    if offset == "start":
        frames = frames_all[:n_frames]
    else:
        frames = select_middle_frames(frames_all, n_frames)
    if len(frames) != n_frames:
        raise SystemExit(
            f"workspace listed {len(frames)} frames, meta expects {n_frames}"
        )
    return frames


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("params_name", type=str, default="params_latest.npz", nargs="?")
    p.add_argument(
        "--from-bundle", type=Path, default=None,
        help="fit_bundle.npz (or its directory); preferred for real run checkpoints",
    )
    p.add_argument("--mag-lo", type=float, default=7.0)
    p.add_argument("--mag-hi", type=float, default=9.0)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--region", type=str, default="1536,1536,2048,2048")
    p.add_argument("--frame-index", type=int, default=None, help="default: middle frame")
    p.add_argument(
        "--frame-indices", type=str, default=None,
        help="comma-separated list of frame indices; overrides --frame-index, exports all "
             "of them from a single context build (no rebuild per frame)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--reject-n-sigma", type=float, default=3.0)
    p.add_argument(
        "--skip-rejection", action="store_true",
        help="skip full-bundle chi2/rejection scoring (level-1 audit handles it)",
    )
    p.add_argument(
        "--frame-offset", choices=["start", "middle"], default=None,
        help="frame window used at prep (default: infer from bundle meta)",
    )
    p.add_argument(
        "--png", action="store_true",
        help="also write panel_{stem}.png (hp_d / model / residual)",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "plots" / "fits_export")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = resolve_run_id(run_dir)

    params_path = run_dir / args.params_name
    if not params_path.exists() and Path(args.params_name).exists():
        params_path = Path(args.params_name)
    print(f"loading params from {params_path} ...")
    params = FIT.load_params_npz(params_path)

    membership = None  # optional DS9 region inputs
    fd = None
    if args.from_bundle is not None:
        print(f"loading fit bundle from {args.from_bundle} ...")
        bundle = FB.load_fit_bundle(args.from_bundle)
        meta = dict(bundle.meta)
        region = RegionSpec(*[int(v) for v in meta["region"]])
        groups = bundle.group_set()
        cx = np.asarray(bundle.stamp_center_x)
        cy = np.asarray(bundle.stamp_center_y)
        stamp_physical = int(bundle.stamp_physical)
        sector = int(meta["sector"])
        camera = int(meta["camera"])
        ccd = int(meta["ccd"])
        frame_records = _frame_records_from_meta(meta, frame_offset=args.frame_offset)
        frame_info = [(f.stem, float(f.btjd)) for f in frame_records]
        n_frames = bundle.n_frames
        print(
            f"  bundle G={bundle.n_groups} T={n_frames} S={stamp_physical} "
            f"region={list(meta['region'])}"
        )
    else:
        print(f"rebuilding context (mag {args.mag_lo}-{args.mag_hi}, {args.n_frames} frames)...")
        rebuilt = build_full_context(
            mag_lo=args.mag_lo, mag_hi=args.mag_hi, n_frames=args.n_frames,
            region_str=args.region,
        )
        fd = rebuilt["fd"]
        groups = rebuilt["groups"]
        stamp_batch = rebuilt["stamp_batch"]
        frames = rebuilt["frames"]
        region = rebuilt["region"]
        cx = np.asarray(stamp_batch.stamp_center_x)
        cy = np.asarray(stamp_batch.stamp_center_y)
        stamp_physical = int(fd.data.shape[-1])
        sector = rebuilt["sector"]
        camera = rebuilt["camera"]
        ccd = rebuilt["ccd"]
        frame_records = frames
        frame_info = [(f.stem, float(f.btjd)) for f in frames]
        n_frames = len(frames)
        membership = {
            "x_ws": rebuilt["x_ws"],
            "y_ws": rebuilt["y_ws"],
            "mags": rebuilt["mags"],
            "expanded_stars": rebuilt["expanded_stars"],
            "primary_index_set": rebuilt["primary_index_set"],
            "region_margin": rebuilt["region_margin"],
        }

    if args.frame_indices is not None:
        frame_indices = [int(s) for s in args.frame_indices.split(",") if s.strip()]
    else:
        frame_indices = [n_frames // 2 if args.frame_index is None else int(args.frame_index)]
    for out_i, fi in enumerate(frame_indices):
        if fi < 0 or fi >= n_frames:
            raise SystemExit(f"frame index {fi} out of range [0, {n_frames})")

    # Rejection scoring below (unless --skip-rejection) needs chi2 over every
    # frame the bundle holds, not just the ones being exported here -- only
    # bind fd's frame axis to frame_indices when that full-T pass is skipped
    # (the case run_postfit.py always uses). This is what avoids the ~18GB/
    # array dense (G, T, p_max) reconstruction (FitBundle._dense_stamp_array)
    # for a tier-segmented bundle when only a handful of frames are wanted.
    fd_bound_to_frames = args.from_bundle is not None and args.skip_rejection
    if args.from_bundle is not None:
        fd = _fd_from_bundle(bundle, frame_indices=frame_indices if fd_bound_to_frames else None)

    packed = bool(getattr(fd.ctx, "is_packed", False))
    print("forward model + flux solve (group-chunked for memory safety)...")
    from .chunked_forward import render_and_solve_chunks
    # FITS output only uses requested preview frames; avoid rendering all
    # 590 orbit frames merely to write three mosaics. When fd is already
    # bound to frame_indices, it IS that subset already (local positions
    # 0..len(frame_indices)-1) -- passing frame_indices again would index
    # the wrong (now out-of-range) columns.
    flux, model, *_ = render_and_solve_chunks(
        params, fd, group_chunk=16,
        frame_indices=None if fd_bound_to_frames else frame_indices,
    )

    data_np = np.asarray(fd.data) if fd_bound_to_frames else np.asarray(fd.data)[:, frame_indices]
    model_np_full = np.asarray(model)
    ny, nx = region.shape

    if args.skip_rejection:
        stamp_active = np.ones((len(bundle.group_set().valid), n_frames), dtype=bool)
        print("rejection scoring skipped (use level-1 audit for rejection diagnostics)")
    else:
        n_sigma = float(args.reject_n_sigma)
        # Keep rejection scoring on bounded group chunks; the monolithic χ²
        # graph can otherwise allocate tens of GB.
        fd.stamp_chunk = 16
        chi2_red, pix_sum = SR.per_stamp_chi2_red(params, fd)
        pix_ok = np.asarray(pix_sum) > 0
        stamp_active = SR.mad_reject_mask(np.asarray(chi2_red), n_sigma=n_sigma, pix_active=pix_ok)
        n_rej_all = int(((stamp_active == 0) & pix_ok).sum())
        n_cand_all = int(pix_ok.sum())
        print(f"reject n_sigma={n_sigma}: {n_rej_all}/{n_cand_all} ({n_rej_all / max(n_cand_all, 1):.1%}) rejected (all frames)")

    for out_i, fi in enumerate(frame_indices):
        stem, btjd = frame_info[fi]
        print(f"\n--- frame fi={fi} stem={stem} btjd={btjd:.5f} ---")

        data_stamp = data_np[:, out_i]
        model_stamp = model_np_full[:, out_i]
        resid_stamp = data_stamp - model_stamp

        print("pasting mosaics...")
        if packed:
            px = np.asarray(bundle.pix_x)
            py = np.asarray(bundle.pix_y)
            pv = np.asarray(bundle.pix_valid)
            hp_d_mosaic = paste_packed_stamps(
                data_stamp, px, py, pv,
                x_min=region.x_min, y_min=region.y_min, ny=ny, nx=nx,
            )
            model_mosaic = paste_packed_stamps(
                model_stamp, px, py, pv,
                x_min=region.x_min, y_min=region.y_min, ny=ny, nx=nx,
            )
            resid_mosaic = paste_packed_stamps(
                resid_stamp, px, py, pv,
                x_min=region.x_min, y_min=region.y_min, ny=ny, nx=nx,
            )
        else:
            hp_d_mosaic = paste_stamps(
                data_stamp, cx, cy, x_min=region.x_min, y_min=region.y_min,
                ny=ny, nx=nx, stamp=stamp_physical,
            )
            model_mosaic = paste_stamps(
                model_stamp, cx, cy, x_min=region.x_min, y_min=region.y_min,
                ny=ny, nx=nx, stamp=stamp_physical,
            )
            resid_mosaic = paste_stamps(
                resid_stamp, cx, cy, x_min=region.x_min, y_min=region.y_min,
                ny=ny, nx=nx, stamp=stamp_physical,
            )

        for kind, arr in [("hp_d", hp_d_mosaic), ("model", model_mosaic), ("residual", resid_mosaic)]:
            path = out_dir / f"{kind}_{stem}.fits"
            write_mosaic_fits(
                path, arr, region=region, sector=sector, camera=camera, ccd=ccd,
                stem=stem, btjd=btjd, frame_idx=fi, run_id=run_id, kind=kind,
            )
            print(f"  wrote {path}")

        full_path = export_full_hp_d(
            frame_records[fi], region, out_dir=out_dir, sector=sector, camera=camera,
            ccd=ccd, frame_idx=fi, run_id=run_id,
        )
        print(f"  wrote {full_path}")

        if args.png:
            png_path = out_dir / f"panel_{stem}.png"
            save_panel_png(
                png_path,
                hp_d=hp_d_mosaic, model=model_mosaic, residual=resid_mosaic,
                stem=stem, btjd=btjd, frame_idx=fi, n_frames=n_frames, run_id=run_id,
            )
            print(f"  wrote {png_path}")

        # Rejected stamp regions (bundle path).
        rej_gi = np.where(stamp_active[:, fi] == 0)[0]
        reg_path = out_dir / f"reject_stamps_{stem}.reg"
        with reg_path.open("w", encoding="utf-8") as fh:
            fh.write(f"# rejected stamp groups fi={fi} n={len(rej_gi)}\n")
            fh.write("global color=red\n")
            rad = max(2.0, 0.45 * stamp_physical)
            for gi in rej_gi:
                x = float(cx[gi]) - region.x_min
                y = float(cy[gi]) - region.y_min
                fh.write(f"circle({x:.3f},{y:.3f},{rad:.3f})\n")
        print(f"  wrote {reg_path} ({len(rej_gi)} rejected groups)")

        if membership is not None:
            primary_index_set = membership["primary_index_set"]
            expanded_stars = membership["expanded_stars"]
            x_ws = membership["x_ws"]
            y_ws = membership["y_ws"]
            mags = membership["mags"]
            comp_idx = []
            for gi in range(groups.n_groups):
                for si in groups.members[gi][groups.valid[gi]]:
                    si = int(si)
                    if si not in primary_index_set:
                        comp_idx.append(si)
            comp_idx = np.unique(np.array(comp_idx, dtype=int))
            pool_idx = np.arange(len(expanded_stars), dtype=int)
            prim_idx = np.array(sorted(primary_index_set), dtype=int)

            rej_gi = np.where(stamp_active[:, fi] == 0)[0]
            rej_prim = []
            for gi in rej_gi:
                for si in groups.members[gi][groups.valid[gi]]:
                    if int(si) in primary_index_set:
                        rej_prim.append(int(si))
            rej_prim = np.unique(np.array(rej_prim, dtype=int)) if rej_prim else np.zeros(0, dtype=int)

            reg_path = out_dir / f"gaia_membership_{stem}.reg"
            DIAG.write_gaia_membership_regions(
                reg_path,
                x=x_ws, y=y_ws,
                region_x_min=region.x_min, region_y_min=region.y_min,
                pool_indices=pool_idx,
                primary_indices=prim_idx,
                companion_indices=comp_idx,
                rejected_primary_indices=rej_prim,
                tess_mag=mags,
                include_pool=False,
            )
            print(f"wrote {reg_path}")
            print(
                f"  pool={len(pool_idx)} primary_kept={len(prim_idx) - len(rej_prim)} "
                f"primary_rej={len(rej_prim)} companions={len(comp_idx)}"
            )
            print(f"Load in DS9 with: {out_dir}/hp_d_{stem}.fits + {reg_path.name}")
        else:
            print(f"Load in DS9: {out_dir}/hp_d_{stem}.fits  (+ model_/residual_ same stem)")


if __name__ == "__main__":
    main()
