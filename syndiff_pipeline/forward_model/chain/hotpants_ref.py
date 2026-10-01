"""Stage ``hotpants``: the F=4 oversampling-aware Hotpants baseline (production recipe: ko=4, connected_regions) on the band-sum template ``T_sum``.

Port of e2e ``hotpants/hp_build.py`` (+ ``hp_job.sh``): HOTPANTS stage of the minbg_tvwcs_f4 recipe, unchanged except
that the template is ``perband/band_templates/T_sum.npy`` (full OS4 grid incl. padding; cast float64 -> float32 -> float64
exactly as ``build_field_mode_template_loader`` does) and the mapping grid is this chain's mapping master.

Per frame writes ``hotpants/<stem>/{hp_d,hp_c,hp_b,kernels}/`` + ``hotpants_params.json`` + ``validation.json``; the
``hp_d`` file (diff, noise, mask planes) is the noise/mask reference the final images and the scorer use.

Differences from the e2e script: paths/geometry come from the config (no ``e2e_cfg``), the numba cache defaults to
``hotpants/numba_cache``, and a Condor submit helper replaces ``hp_job.sh``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import is_done, mark_done, write_provenance
from .perband.paths import chain_paths

# ko 4 + connected_regions = the 09-08 decided production recipe (was the SN2020hvq ko 2 grid copy until 2026-10-01)
HP_KWARGS = dict(hp_sigma_gauss=[0.752, 1.88, 3.76], hp_ko=4, hp_bgo=0, stamp_mode="connected_regions",
                 hp_nstampx=10, hp_nstampy=10, hp_nss=100, hp_ngauss=3, hp_deg_fixe=[6, 4, 2],
                 hp_kf_spread_mask1=0.0, hp_ks=3.0, hp_kfm=0.75, hp_fitthresh=5.0, hp_stat_sig=3.0,
                 hp_force_convolve="t", hp_normalize="t", write_convolved=True, write_bkg=True,
                 write_stamps=False, write_kernel_solutions=True)
SCIENCE_BOUNDS = dict(x_min=44, x_max=2092, y_min=0, y_max=2048, shape=(2048, 2048))


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build_frame(cfg, stem: str) -> dict:
    """Run Hotpants for one frame; returns the validation dict (also written to ``validation.json``)."""
    P = chain_paths(cfg)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(P.hotpants / "numba_cache"))
    os.environ.setdefault("HOTPANTS_OS_N_JOBS", "4")
    import pandas as pd
    from astropy.io import fits
    from syndiff_pipeline.common.grid_pairing import trim_padded_products
    from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
    from syndiff_pipeline.difference_imaging.stages import hotpants as HP

    scc = P.scc
    master = P.mapping_dir / P.master_name
    tp = P.band_templates / "T_sum.npy"
    ffi = P.ffi(stem)
    out_root = P.hotpants / stem
    for d in ("hp_d", "hp_c", "hp_b", "kernels"):
        (out_root / d).mkdir(parents=True, exist_ok=True)
    grid = load_mapping_grid_from_master(master)
    tmpl = np.load(tp).astype(np.float32).astype(np.float64)   # production loader precision
    assert tmpl.shape == tuple(grid.array_shape_os()) == (8256, 8256), (tmpl.shape, grid.array_shape_os())
    sci, err = HP._load_ffi_cropped(str(ffi), SCIENCE_BOUNDS)
    bkgpath = P.background(stem)
    bkg = np.asarray(fits.getdata(bkgpath, 1), dtype=float)
    assert bkg.shape == sci.shape
    sci -= bkg
    maskraw = fits.getdata(scc / "diff_linear/shared_mask.fits.fz", 1)
    mask = HP._resolve_hotpants_mask_array(maskraw, None, None)
    stars = pd.read_csv(scc / "diff_linear/hotpants_substamp_stars.csv")[["x", "y"]].to_numpy(float)
    sci, tmpl, err, mask, pad = HP._pair_hotpants_inputs(sci, tmpl, err, mask, grid, 0)
    hp = HotpantsParams(**HP_KWARGS)
    (out_root / "hotpants_params.json").write_text(json.dumps(asdict(hp), indent=2, default=str) + "\n")
    hcfg = HP.build_hotpants_config(hp, str(out_root / "hp_d"), str(out_root / "hp_c"), stem,
                                    write_stamps=False, sci_shape=sci.shape)
    print(f"HOTPANTS {P.field} {stem}: science {sci.shape}; template {tmpl.shape}; F=4; stars {len(stars)}; pad {pad}", flush=True)
    t0 = time.time()
    res = HP.run_hotpants_frame(sci, err, tmpl, mask, stars + pad, hcfg, oversample=4, collect_kernel_params=True)
    if not res["success"]:
        raise RuntimeError(res["error_msg"])
    hp_sec = time.time() - t0

    def trimmed(key):
        a = np.asarray(trim_padded_products(res[key], grid=grid))
        assert a.shape == (2048, 2048)
        return a
    source = scc / "diff_linear/hp_d" / f"{stem}_hp_d.fits.fz"
    with fits.open(source) as h:
        primary = h[0].header.copy()
        headers = [x.header.copy() for x in h[1:4]]
    primary["TMPL_OS"] = 4
    primary["DIFFLANE"] = f"chain_{P.field}"
    primary["FFISTEM"] = stem
    primary.add_history(f"forward_model.chain hotpants: ko2 bgo0 F=4 on the band-sum template T_sum; native-grid output.")
    out = out_root / "hp_d" / f"{stem}_hp_d.fits.fz"
    hdus = [fits.PrimaryHDU(header=primary)]
    for key, hdr in zip(["diff", "noise", "mask"], headers):
        hdus.append(fits.CompImageHDU(data=trimmed(key), header=hdr, compression_type="GZIP_1", quantize_level=0))
    fits.HDUList(hdus).writeto(out, overwrite=True, checksum=True)
    rt, zq = {}, None
    with fits.open(out, checksum=True) as check:
        for i, key in enumerate(["diff", "noise", "mask"], 1):
            assert np.array_equal(check[i].data, trimmed(key), equal_nan=True), key
            rt[key] = True
            zq = check[i].header.get("ZQUANTIZ")
    extra = {}
    for key, label in [("convolved", "hp_c"), ("bkg", "hp_b")]:
        if res.get(key) is not None:
            dest = out_root / label / f"{stem}_{label}.fits.fz"
            fits.HDUList([fits.PrimaryHDU(header=primary), fits.CompImageHDU(data=trimmed(key), header=headers[0],
                          compression_type="GZIP_1", quantize_level=0)]).writeto(dest, overwrite=True, checksum=True)
            with fits.open(dest) as check:
                assert np.array_equal(check[1].data, trimmed(key), equal_nan=True), label
            extra[label] = dict(path=str(dest), sha256=sha256(dest), round_trip_exact=True)
    if res.get("kernel_params_arrays"):
        dest = out_root / "kernels" / f"{stem}_kernel.npz"
        np.savez_compressed(dest, **res["kernel_params_arrays"])
        extra["kernels"] = dict(path=str(dest), sha256=sha256(dest))
    D, N, M = trimmed("diff"), trimmed("noise"), trimmed("mask")
    good = (M == 0) & np.isfinite(D) & (N > 0)
    g0 = M == 0
    chi = D[good] / N[good]
    val = dict(field=P.field, stem=stem, output=str(out), sha256=sha256(out),
               template=str(tp), template_sha256=sha256(tp), mapping=str(P.mapping_dir), ffi=str(ffi), background=str(bkgpath),
               mask=str(scc / "diff_linear/shared_mask.fits.fz"), substamp_stars=len(stars),
               finite_fraction=dict(diff=float(np.isfinite(D).mean()), noise=float(np.isfinite(N).mean())),
               good_pixels_mask0=int(g0.sum()), good_pixels=int(good.sum()),
               noise_pos_frac_on_mask0=float((N[g0] > 0).mean()),
               median_noise_good=float(np.median(N[good])),
               robust_std_chi=float(1.4826 * np.median(np.abs(chi - np.median(chi)))),
               diff_quantiles_1_16_50_84_99=np.percentile(D[good], [1, 16, 50, 84, 99]).tolist(),
               round_trip_exact=rt, zquantiz=zq, compression="GZIP_1 quantize_level=0",
               hotpants_seconds=hp_sec, products=extra, code_sha=cfg.code_sha())
    assert good.sum() > 100000 and val["noise_pos_frac_on_mask0"] == 1.0
    (out_root / "validation.json").write_text(json.dumps(val, indent=1) + "\n")
    print("COMPLETED", out, json.dumps({k: val[k] for k in ("good_pixels", "robust_std_chi", "median_noise_good")}), flush=True)
    return val


def frames_for(cfg) -> list[str]:
    return chain_paths(cfg).frames


def run(cfg, stems: Optional[Sequence[str]] = None, force: bool = False, condor: bool = False) -> Path:
    """Stage ``hotpants``: needs ``perband`` done (T_sum).  One Hotpants run per frame (fit frame + unseen frame).
    ``condor``: submit one job per frame instead (see ``submit``)."""
    if condor:
        submit(cfg, stems, do_submit=True)
        return cfg.stage_dir("hotpants")
    P = chain_paths(cfg)
    stage = cfg.stage_dir("hotpants")
    if is_done(stage) and not force and stems is None:
        return stage
    if not (P.band_templates / "T_sum.npy").is_file():
        raise FileNotFoundError(f"{P.band_templates / 'T_sum.npy'} missing: run the perband stage first")
    todo = list(stems) if stems else frames_for(cfg)
    for stem in todo:
        build_frame(cfg, stem)
    if all(P.hotpants_ref(s).is_file() for s in frames_for(cfg)):
        write_provenance(stage, cfg, {"T_sum": P.band_templates / "T_sum.npy", "mapping": P.mapping_dir})
        mark_done(stage)
    return stage


def submit(cfg, stems: Optional[Sequence[str]] = None, *, do_submit: bool = False) -> list[Path]:
    """One Condor job per frame (replaces ``hp_job.sh``): ``python -m ...hotpants_ref --config F.yaml --stem S``."""
    from . import condor
    if cfg.config_path is None:
        raise ValueError("config has no file path; Condor jobs need `--config F.yaml`")
    subs = []
    for stem in (list(stems) if stems else frames_for(cfg)):
        argv = ["python", "-m", "syndiff_pipeline.forward_model.chain.hotpants_ref", "--config", str(cfg.config_path),
                "--stem", stem]
        sub = condor.write_submit(cfg, "hotpants", argv, tag=f"hp_{stem}", request_cpus=8, request_memory_mb=64000,
                                  omp_threads=4)
        subs.append(sub)
        if do_submit:
            print(condor.submit(sub))
    return subs


def main(argv: Optional[list] = None) -> int:
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--stem", action="append", help="frame stem (repeatable); default: fit + unseen frame")
    ap.add_argument("--submit", action="store_true", help="condor_submit one job per frame instead of running")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.submit:
        submit(cfg, a.stem, do_submit=True)
    else:
        run(cfg, a.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
