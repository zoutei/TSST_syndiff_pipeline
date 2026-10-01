"""BOOTSTRAP stage of the Paper 1 chain: harmonic background -> F=4 header-WCS template -> OS-aware Hotpants.

One science frame per field.  Outputs (all under ``cfg.stage_dir("bootstrap")``)::

    bkg/<stem>_ks_b.fits.fz            OPTIONAL local ks_b regeneration (step ``bkg``; dry run only). Real runs read ks_b,
                                       shared_mask and the substamp stars from the F=1 lane dir (``inputs.lane_dir``,
                                       default out_root/lane_f1) built by the lane stage.
    data_priv/                         private data_root (symlinks to read-only inputs) so nothing under data_root is written
    remap/oversampling_4/              field remap store (header WCS, single frame)
    templates/oversampling_4/          band-combined F=4 field template store (+ materialised FITS)
    diff/{hp_d,hp_c,hp_b,hp_d_kernels} OS-aware Hotpants products on the native grid
    step_*.json, DONE, provenance.json

Route (see CONTRACT / dry-run README): the F=4 mapping is built from the science frame's own FITS header WCS
(``mapping`` stage, ``tess_wcs_override=None``).  The remap therefore uses ``drift_source="point_ffi_wcs"`` with
``ref_ffi_path`` = the science frame and an ``ffi_dir`` holding only that frame: the frame IS the mapping reference,
so its drift is exactly zero, there is one group (gid 0), and no temporal-WCS store is involved.  Downsample is the
unmodified production ``run_field_downsample_scc`` (psf_sigma 40, remove_saturated_stars, bright threshold 13) and
reads the combined/convolved PS1 store in ``<data_root>/ps1_skycells_zarr`` (whatever recipe/weights that store holds).

Numerics follow ``dev_runs/e2e_f1_20260930/s12_bootstrap/scripts/{build,finish}.py`` unless noted.
Run:  ``python -m syndiff_pipeline.forward_model.chain.bootstrap --config F.yaml --step {bkg,template,hotpants,all,submit}``
(``all`` = template + hotpants.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_LANE = "diff_linear"  # old lane providing tmpl_conv (OS1 linear convolved template), shared_mask, substamp stars
SCIENCE_BOUNDS = dict(x_min=44, x_max=2092, y_min=0, y_max=2048, shape=(2048, 2048))
OVERSAMPLING = 4
PSF_SIGMA = 40.0
HP_RECIPE = dict(  # config/pipeline_sn2020hvq_tvwcs_os4.yaml, as in finish.py
    hp_sigma_gauss=[0.752, 1.88, 3.76], hp_ko=2, hp_bgo=0, hp_nstampx=10, hp_nstampy=10, hp_nss=100,
    hp_ngauss=3, hp_deg_fixe=[6, 4, 2], hp_kf_spread_mask1=0.0, hp_ks=3.0, hp_kfm=0.75, hp_fitthresh=5.0,
    hp_stat_sig=3.0, hp_force_convolve="t", hp_normalize="t", write_convolved=True, write_bkg=True,
    write_stamps=False, write_kernel_solutions=True,
)
TEMPLATE_RESOURCES = dict(request_cpus=16, request_memory_mb=200000)


# ---------------------------------------------------------------------------------------------- pure helpers
def robust_stats(x: np.ndarray) -> dict:
    """median, 1.4826*MAD, std and n of the finite entries of ``x``."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, median=float("nan"), mad_sigma=float("nan"), std=float("nan"))
    med = float(np.median(x))
    return dict(n=int(x.size), median=med, mad_sigma=float(1.4826 * np.median(np.abs(x - med))), std=float(x.std()))


def hpd_good_mask(diff: np.ndarray, noise: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixels used for hp_d statistics: ``mask==0`` (bit 32 is a FIT pixel, so also accept ``mask==32``) & finite & noise>0."""
    m = (mask == 0) | (mask == 32)
    return m & np.isfinite(diff) & np.isfinite(noise) & (noise > 0)


def hpd_stats(diff: np.ndarray, noise: np.ndarray, mask: np.ndarray) -> dict:
    """Robust stats of hp_d/noise on the good (unmasked) pixels."""
    g = hpd_good_mask(diff, noise, mask)
    z = np.where(g, diff / np.where(noise > 0, noise, np.nan), np.nan)
    out = robust_stats(z)
    out["n_strict_mask0"] = int(((mask == 0) & g).sum())
    return out


def frame_group_offsets(lane_root: Path, stem: str) -> tuple[float, float]:
    """(group_dx, group_dy) of ``stem`` in the linear-mode (OS1) grouping.

    Source: ``<scc>/remap_linear/oversampling_1/point_drift_table.csv`` (present for every SCC; column ``filename``),
    else ``<lane>/frames.csv`` (column ``ffi_basename``; only some lanes have it).  ``lane_root`` is ``<scc>/<lane>``."""
    import pandas as pd

    lane_root = Path(lane_root)
    for path, col in ((lane_root.parent / "remap_linear" / "oversampling_1" / "point_drift_table.csv", "filename"),
                      (lane_root / "frames.csv", "ffi_basename")):
        if not path.is_file():
            continue
        fr = pd.read_csv(path)
        hit = fr[fr[col].astype(str).str.startswith(stem)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        if not np.isfinite(float(row["group_dx"])) or not np.isfinite(float(row["group_dy"])):
            raise ValueError(f"{stem} has no linear group offsets in {path} (group_id={row['group_id']})")
        return float(row["group_dx"]), float(row["group_dy"])
    raise KeyError(f"{stem} not found in point_drift_table.csv / frames.csv of {lane_root}")


def convolved_template_for_frame(lane_root: Path, stem: str) -> Path:
    """Path of the lane's convolved (OS1 linear) template for the frame's group offsets."""
    import pandas as pd

    from syndiff_pipeline.difference_imaging.stages.convolved_templates import lookup_convolved_path

    dx, dy = frame_group_offsets(lane_root, stem)
    table = pd.read_csv(Path(lane_root) / "tmpl_conv" / "convolved_templates.csv")
    return Path(lookup_convolved_path(table, dx, dy))


def make_private_data_root(priv: Path, data_root: Path, sector: int, camera: int, ccd: int, ffi_path: Path,
                           mapping_dir: Path | None = None) -> Path:
    """Private data_root: symlinks to read-only inputs, own bookkeeping and ffi_list, so the stage writes nothing
    under the production ``data_root``.  Idempotent.

    ``mapping_dir`` (``.../oversampling_<F>``) is linked as ``<scc>/mapping/oversampling_<F>``: the downsample reads the
    master skycells list from ``data_root`` (``scc_mapping_master_skycells_csv``), and the schema-v2 convolved
    fingerprint includes each cell's neighbour set from that list, so without it every cell is a miss."""
    priv = Path(priv)
    scc_src = Path(data_root) / f"s{sector:04d}" / f"c{camera}" / f"k{ccd}"
    scc = priv / f"s{sector:04d}" / f"c{camera}" / f"k{ccd}"
    scc.mkdir(parents=True, exist_ok=True)
    if mapping_dir is not None:
        dst = scc / "mapping" / Path(mapping_dir).name
        dst.parent.mkdir(exist_ok=True)
        if dst.is_symlink() and Path(os.readlink(dst)) != Path(mapping_dir):
            raise FileExistsError(f"{dst} links to {os.readlink(dst)}, not {mapping_dir}")
        if not dst.is_symlink():
            dst.symlink_to(Path(mapping_dir))
    for name in ("catalogs", "ffi", "wcs"):
        if (scc_src / name).exists() and not (scc / name).exists():
            (scc / name).symlink_to(scc_src / name)
    for name in ("ffi_list.parquet", "ffi_list.csv"):
        if (scc_src / name).is_file() and not (scc / name).exists():
            shutil.copy2(scc_src / name, scc / name)
    zsrc = Path(data_root) / "ps1_skycells_zarr"
    zdst = priv / "ps1_skycells_zarr"
    zdst.mkdir(exist_ok=True)
    for p in sorted(zsrc.iterdir()):
        if not (zdst / p.name).exists() and not (zdst / p.name).is_symlink():
            (zdst / p.name).symlink_to(p)
    (priv / "bookkeeping").mkdir(exist_ok=True)
    return priv


# ---------------------------------------------------------------------------------------------- step 1: background
def estimate_ks_b(ffi: np.ndarray, convolved: np.ndarray, mask: np.ndarray, *, fill_method: str = "harmonic",
                  star_mask_pad_px: int = 0, **tess_kwargs: Any) -> np.ndarray:
    """ks_b = TESSreduce residual background of ``ffi - convolved`` (what ``background_estimate`` writes as phot_bkg).

    ``star_mask_pad_px`` (chain config ``background.star_mask_pad_px``, default 0 = current production behaviour) grows
    the catalogue star masks by that many px before they are excluded from the fit (branch bkg-star-pad 2ecf558,
    ``tessreduce_star_mask_pad_px``).  It is forwarded to the estimator as ``star_mask_pad_px`` ONLY when non-zero, so
    the same code runs on code bases with and without that branch; with pad > 0 on a code base without it a
    ``TypeError`` is raised rather than silently ignoring the setting.  Note the lane's convolved template (kernel_fit
    output) is reused as is; its own kernel_fit background is not re-padded here.
    """
    if int(star_mask_pad_px) < 0:
        raise ValueError("star_mask_pad_px must be >= 0")
    if int(star_mask_pad_px) > 0:
        tess_kwargs["star_mask_pad_px"] = int(star_mask_pad_px)
    from syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual import (
        estimate_tessreduce_residual_background,
    )

    if ffi.shape != convolved.shape or ffi.shape != mask.shape:
        raise ValueError(f"shape mismatch ffi {ffi.shape} conv {convolved.shape} mask {mask.shape}")
    bkg, _, _ = estimate_tessreduce_residual_background(
        np.asarray(ffi, float) - np.asarray(convolved, float), mask, fill_method=fill_method, **tess_kwargs)
    return bkg


def step_bkg(*, ffi_path: Path, lane_root: Path, stem: str, out_dir: Path, fill_method: str = "harmonic",
             star_mask_pad_px: int = 0, bounds: Mapping | None = None, convolved_path: Path | None = None) -> dict:
    """Regenerate ks_b for the science frame (no asteroid/temporal layer: the static shared_mask is the frame mask,
    as in ``background_estimate`` when the SCC has no asteroid sidecars -- checked: reproduces the stored
    biharmonic ks_b to float32 rounding when ``fill_method='biharmonic'``)."""
    from astropy.io import fits

    from syndiff_pipeline.common import wcs_grouping
    from syndiff_pipeline.difference_imaging.stages.hotpants import _load_ffi_cropped, _write_image_fits

    t0 = time.time()
    b = dict(bounds or SCIENCE_BOUNDS)
    conv_path = Path(convolved_path) if convolved_path else convolved_template_for_frame(lane_root, stem)
    ffi, _ = _load_ffi_cropped(str(ffi_path), b)
    conv = np.asarray(fits.getdata(conv_path), dtype=np.float64)
    mask = np.asarray(fits.getdata(Path(lane_root) / "shared_mask.fits.fz"), dtype=np.int16)
    bkg = estimate_ks_b(ffi, conv, mask, fill_method=fill_method, star_mask_pad_px=star_mask_pad_px)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}_ks_b.fits.fz"
    hdr = wcs_grouping.crop_ffi_header(str(ffi_path), b)
    hdr["BKGFILL"] = (fill_method, "gap-fill method of the background")
    hdr["BKGPAD"] = (int(star_mask_pad_px), "star-mask padding (px) excluded from the bkg fit")
    _write_image_fits(str(out), bkg, header=hdr)
    res = dict(step="bkg", ks_b=str(out), convolved=str(conv_path), shared_mask=str(Path(lane_root) / "shared_mask.fits.fz"),
               fill_method=fill_method, star_mask_pad_px=int(star_mask_pad_px), seconds=time.time() - t0)
    (out_dir.parent / "step_bkg.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


# ---------------------------------------------------------------------------------------------- step 2: template
def step_template(*, sector: int, camera: int, ccd: int, ffi_path: Path, mapping_dir: Path, data_root: Path,
                  work: Path, band_weights: dict | None, n_jobs: int = 16) -> dict:
    """Header-WCS F=4 remap + production field downsample for the single science frame.

    ``band_weights`` are the r,i,z,y weights the combined store was built with (the chain's
    ``perband.paths.chain_band_weights``).  They enter the combined recipe, so the downsample loads only cells of that
    recipe; ``None`` means the production defaults.  Passing the wrong set makes every cell a miss, never a substitute.

    ``mapping_dir`` is the ``.../oversampling_4`` header-WCS mapping built from this very frame (reference == frame)."""
    from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
    from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path
    from syndiff_pipeline.template_creation.processing.combined_store import production_combined_recipe
    from syndiff_pipeline.template_creation.processing.field_downsample import run_field_downsample_scc
    from syndiff_pipeline.template_creation.processing.field_remap import run_field_remap_scc

    work = Path(work)
    priv = make_private_data_root(work / "data_priv", data_root, sector, camera, ccd, ffi_path, mapping_dir=mapping_dir)
    from syndiff_pipeline.common.scc_paths import scc_mapping_master_skycells_csv
    csv = scc_mapping_master_skycells_csv(priv, sector, camera, ccd, oversampling_factor=OVERSAMPLING)
    if not csv.is_file():
        raise FileNotFoundError(f"master skycells list {csv} missing: the downsample cannot resolve canonical cells")
    ffi_in = work / "ffi"
    ffi_in.mkdir(parents=True, exist_ok=True)
    link = ffi_in / Path(ffi_path).name
    if not link.exists():
        link.symlink_to(ffi_path)
    mapping_dir = Path(mapping_dir)
    master = sorted(mapping_dir.glob(f"tess_s{sector:04d}_{camera}_{ccd}_master_pixels2skycells_os{OVERSAMPLING}.fits*"))
    if not master:
        raise FileNotFoundError(f"header-WCS master mapping missing in {mapping_dir}")
    grid = load_mapping_grid_from_master(master[0])
    t0 = time.time()
    remap_root = work / "remap" / f"oversampling_{OVERSAMPLING}"
    res_remap = run_field_remap_scc(
        sector=sector, camera=camera, ccd=ccd, data_root=priv, event_dir=work, mapping_root=mapping_dir,
        base_tess_shape=grid.array_shape_os(), oversampling_factor=OVERSAMPLING, store_root=remap_root, scc_only=True,
        ffi_dir=ffi_in, ref_ffi_path=ffi_path, n_jobs=n_jobs, progress_path=work / "remap_progress.json",
        stage_regmaps_to_scratch=False, drift_source="point_ffi_wcs",
        target_drift=np.zeros((1, 2)))  # one frame == the mapping reference: zero drift by construction
    t_remap = time.time() - t0
    (work / "remap_result.json").write_text(json.dumps(res_remap, indent=2, default=str) + "\n")

    t1 = time.time()
    recipe_cfg = {"remove_saturated_stars": True, "enable_saturation_correction": False}
    if band_weights is not None:
        recipe_cfg["band_weights"] = {b: float(band_weights[b]) for b in ("r", "i", "z", "y")}
    recipe = production_combined_recipe(recipe_cfg, data_root=priv, sector=sector, camera=camera, ccd=ccd)
    template_root = work / "templates" / f"oversampling_{OVERSAMPLING}"
    res_ds = run_field_downsample_scc(
        sector=sector, camera=camera, ccd=ccd, data_root=priv, event_dir=work, mapping_root=mapping_dir,
        convolved_dir=ps1_convolved_zarr_path(priv), roi_bounds=(grid.ffi_xmin, grid.ffi_ymin, grid.ffi_xmax, grid.ffi_ymax),
        base_tess_shape=grid.array_shape_os(), oversampling_factor=OVERSAMPLING, ignore_mask_bits=[12], n_jobs=n_jobs,
        update_frames_csv=False, store_root=template_root, remap_store_root=remap_root, stage_regmaps_to_scratch=False,
        scc_only=True, mapping_grid=grid, psf_sigma=PSF_SIGMA, combined_recipe=recipe, materialize_fits=True,
        progress_path=work / "downsample_progress.json")
    t_ds = time.time() - t1
    (work / "downsample_result.json").write_text(json.dumps(res_ds, indent=2, default=str) + "\n")
    from syndiff_pipeline.template_creation.processing.combined_store import combined_recipe_id
    res = dict(step="template", template_root=str(template_root), mapping=str(master[0]), combined_recipe=recipe,
               combined_recipe_id=combined_recipe_id(recipe),
               seconds_remap=t_remap, seconds_downsample=t_ds, drift_source="point_ffi_wcs")
    (work / "step_template.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


# ---------------------------------------------------------------------------------------------- step 3: Hotpants
def step_hotpants(*, ffi_path: Path, stem: str, lane_root: Path, ks_b_path: Path, template_root: Path,
                  mapping_master: Path, out_dir: Path, lane_label: str = "bootstrap") -> dict:
    """OS-aware Hotpants (F=4): science = FFI - ks_b, mask = shared_mask (bit 32 ignored), lane substamp stars."""
    import pandas as pd
    from astropy.io import fits

    from syndiff_pipeline.common import wcs_grouping
    from syndiff_pipeline.common.grid_pairing import trim_padded_products
    from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
    from syndiff_pipeline.difference_imaging.stages import hotpants as HP
    from syndiff_pipeline.difference_imaging.support.template_resolution import (
        build_field_mode_template_loader,
        load_field_mode_template_context_from_store,
    )

    t0 = time.time()
    out_dir = Path(out_dir)
    grid = load_mapping_grid_from_master(mapping_master)
    ctx = load_field_mode_template_context_from_store(str(template_root))
    bounds = dict(SCIENCE_BOUNDS)
    tmpl = build_field_mode_template_loader(ctx, bounds, crop_to_science=False)(0)
    expect = tuple(grid.array_shape_os())
    if tmpl.shape != expect:
        raise ValueError(f"template shape {tmpl.shape} != mapping grid {expect}")
    sci, err = HP._load_ffi_cropped(str(ffi_path), bounds)
    bkg = np.asarray(fits.getdata(ks_b_path), dtype=float)
    if bkg.shape != sci.shape:
        raise ValueError(f"ks_b shape {bkg.shape} != science {sci.shape}")
    sci = sci - bkg
    mask = HP._resolve_hotpants_mask_array(fits.getdata(Path(lane_root) / "shared_mask.fits.fz"), None, None)
    stars = pd.read_csv(Path(lane_root) / "hotpants_substamp_stars.csv")[["x", "y"]].to_numpy(float)
    sci, tmpl, err, mask, pad = HP._pair_hotpants_inputs(sci, tmpl, err, mask, grid, 0)
    hp = HotpantsParams(**HP_RECIPE)
    (out_dir.parent / "hotpants_params.json").write_text(json.dumps(asdict(hp), indent=2, default=str) + "\n")
    cfg = HP.build_hotpants_config(hp, str(out_dir / "hp_d"), str(out_dir / "hp_c"), stem, write_stamps=False,
                                   sci_shape=sci.shape)
    t_run = time.time()
    res = HP.run_hotpants_frame(sci, err, tmpl, mask, stars + pad, cfg, oversample=OVERSAMPLING, collect_kernel_params=True)
    if not res["success"]:
        raise RuntimeError(res["error_msg"])
    t_run = time.time() - t_run

    def trimmed(key):
        a = np.asarray(trim_padded_products(res[key], grid=grid))
        if a.shape != (2048, 2048):
            raise ValueError(f"{key} trimmed to {a.shape}")
        return a

    src = Path(lane_root) / "hp_d" / f"{stem}_hp_d.fits.fz"
    if src.is_file():  # reuse same-exposure native headers (timing + cropped WCS), as the reference run does
        with fits.open(src) as h:
            primary = h[0].header.copy()
            headers = [x.header.copy() for x in h[1:4]]
    else:
        primary = fits.Header()
        headers = [wcs_grouping.crop_ffi_header(str(ffi_path), bounds)] * 3
    primary["TMPL_OS"] = OVERSAMPLING
    primary["DIFFLANE"] = lane_label
    primary["FFISTEM"] = stem
    primary.add_history("Bootstrap: single-FFI F4 header-WCS template, harmonic ks_b; native-grid output.")
    (out_dir / "hp_d").mkdir(parents=True, exist_ok=True)
    out = out_dir / "hp_d" / f"{stem}_hp_d.fits.fz"
    hdus = [fits.PrimaryHDU(header=primary)]
    for key, hdr in zip(["diff", "noise", "mask"], headers):
        hdus.append(fits.CompImageHDU(data=trimmed(key), header=hdr, compression_type="GZIP_1", quantize_level=0))
    fits.HDUList(hdus).writeto(out, overwrite=True, checksum=True)
    for key, label in [("convolved", "hp_c"), ("bkg", "hp_b")]:
        if res.get(key) is not None:
            dest = out_dir / label / f"{stem}_{label}.fits.fz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            fits.HDUList([fits.PrimaryHDU(header=primary), fits.CompImageHDU(
                data=trimmed(key), header=headers[0], compression_type="GZIP_1", quantize_level=0)]).writeto(
                dest, overwrite=True, checksum=True)
    if res.get("kernel_params_arrays"):
        (out_dir / "hp_d_kernels").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "hp_d_kernels" / f"{stem}_kernel.npz", **res["kernel_params_arrays"])
    stats = hpd_stats(trimmed("diff"), trimmed("noise"), trimmed("mask"))
    if stats["n"] < 100000:
        raise RuntimeError(f"only {stats['n']} good pixels in hp_d")
    r = dict(step="hotpants", hp_d=str(out), hotpants_seconds=t_run, seconds=time.time() - t0, stats=stats, n_stars=int(len(stars)))
    (out_dir.parent / "step_hotpants.json").write_text(json.dumps(r, indent=1) + "\n")
    return r


# ---------------------------------------------------------------------------------------------- config wiring
def lane_dir(cfg, override: str | Path | None = None) -> Path:
    """The F=1 linear lane supplying ks_b/, shared_mask.fits.fz, hotpants_substamp_stars.csv (and tmpl_conv/ for the
    optional ``bkg`` step).  Order: explicit override, ``cfg.inputs.lane_dir``, default ``out_root/lane_f1``
    (built by the lane stage from scratch on the pinned SHA)."""
    if override:
        return Path(override)
    v = getattr(cfg.inputs, "lane_dir", None) or ((cfg.raw or {}).get("inputs") or {}).get("lane_dir")
    return Path(v) if v else cfg.out_root / "lane_f1"


def check_lane(lane: Path, stem: str, ks_b: Path | None = None) -> dict:
    """Fail early (with every missing path listed) if the lane lacks what the OS-aware Hotpants needs."""
    need = {"shared_mask": lane / "shared_mask.fits.fz", "substamp_stars": lane / "hotpants_substamp_stars.csv",
            "ks_b": ks_b or lane / "ks_b" / f"{stem}_ks_b.fits.fz"}
    missing = [f"{k}: {v}" for k, v in need.items() if not Path(v).is_file()]
    if missing:
        raise FileNotFoundError("F=1 lane incomplete (lane stage not run?):\n  " + "\n  ".join(missing))
    return need


def run_stage(cfg, step: str = "all", *, lane_override: str | Path | None = None, local_bkg: bool = False) -> dict:
    """Run bootstrap step(s) for a loaded ``ChainConfig``.

    ``all`` = template + hotpants (writes provenance + DONE): the background is NOT regenerated, it is read from the
    F=1 lane (``lane_dir``).  ``bkg`` regenerates ks_b locally (needs the lane's tmpl_conv + frame offsets; used by the
    dry run), and ``local_bkg=True`` makes ``hotpants`` read that local ks_b instead of the lane's."""
    from syndiff_pipeline.forward_model.chain.config import mark_done, write_provenance
    from syndiff_pipeline.forward_model.chain.perband.paths import chain_band_weights

    sd = cfg.stage_dir("bootstrap")
    sd.mkdir(parents=True, exist_ok=True)
    ffi = cfg.ffi_path()
    lane = lane_dir(cfg, lane_override)
    s = cfg.scc
    mapping = Path(cfg.need("inputs.bootstrap_mapping"))
    master = None
    out: dict = {}
    local_ks_b = sd / "bkg" / f"{cfg.stem}_ks_b.fits.fz"
    if step == "bkg":
        bk = cfg.background
        out["bkg"] = step_bkg(ffi_path=ffi, lane_root=lane, stem=cfg.stem, out_dir=sd / "bkg", fill_method=bk.fill,
                              star_mask_pad_px=bk.star_mask_pad_px)
    if step in ("template", "all"):
        out["template"] = step_template(sector=s.sector, camera=s.camera, ccd=s.ccd, ffi_path=ffi, mapping_dir=mapping,
                                        data_root=cfg.data_root, work=sd, band_weights=chain_band_weights(cfg),
                                        n_jobs=int(os.environ.get("BOOTSTRAP_NJOBS", 16)))
    if step in ("hotpants", "all"):
        ks_b = local_ks_b if local_bkg else lane / "ks_b" / f"{cfg.stem}_ks_b.fits.fz"
        check_lane(lane, cfg.stem, ks_b)
        master = sorted(mapping.glob(f"tess_s{s.sector:04d}_{s.camera}_{s.ccd}_master_pixels2skycells_os4.fits*"))[0]
        out["hotpants"] = step_hotpants(ffi_path=ffi, stem=cfg.stem, lane_root=lane, ks_b_path=ks_b,
                                        template_root=sd / "templates" / "oversampling_4", mapping_master=master,
                                        out_dir=sd / "diff")
    if step == "all":
        write_provenance(sd, cfg, {"ffi": ffi, "ks_b": ks_b, "shared_mask": lane / "shared_mask.fits.fz",
                                   "substamp_stars": lane / "hotpants_substamp_stars.csv", "lane_dir": {"value": str(lane)},
                                   "mapping_master": master, "steps": json.loads(json.dumps(out, default=str))})
        mark_done(sd)
    return out


def submit_template(cfg, step: str = "template", extra: list[str] | None = None) -> str:
    """Write + submit the Condor job for a step (``template`` by default; ``all`` runs template + hotpants on one node)."""
    from syndiff_pipeline.forward_model.chain import condor

    argv = ["python", "-m", "syndiff_pipeline.forward_model.chain.bootstrap", "--config", str(cfg.config_path),
            "--step", step, *(extra or [])]
    sub = condor.write_submit(cfg, "bootstrap", argv, tag=f"bootstrap_{step}", **TEMPLATE_RESOURCES)
    return condor.submit(sub)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--step", default="all", choices=["bkg", "template", "hotpants", "all", "submit"])
    ap.add_argument("--submit-step", default="template", choices=["bkg", "template", "hotpants", "all"],
                    help="with --step submit: which step the Condor job runs")
    ap.add_argument("--lane-dir", default=None, help="override inputs.lane_dir (default out_root/lane_f1)")
    ap.add_argument("--local-bkg", action="store_true", help="hotpants reads bootstrap/bkg/ks_b (dry run) instead of the lane's")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from syndiff_pipeline.forward_model.chain.config import load_config

    cfg = load_config(a.config)
    cfg.check_code_sha()
    extra = (["--lane-dir", a.lane_dir] if a.lane_dir else []) + (["--local-bkg"] if a.local_bkg else [])
    if a.step == "submit":
        print(submit_template(cfg, a.submit_step, extra))
        return 0
    print(json.dumps(run_stage(cfg, a.step, lane_override=a.lane_dir, local_bkg=a.local_bkg), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
