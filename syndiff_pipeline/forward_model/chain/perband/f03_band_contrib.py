"""f03: blur each band cell exactly as production blurs the combined cell, seam-correct cross-projection cells, and
bin every band through the OS4 regmap.  One Condor job = one chunk of cells (``queue N``, chunk ``k/n``).

Per cell: choose the skycell list whose row-path blur of the combined cell (sum of band cells) reproduces the STORED
canonical cell best -- candidates in publisher order from ``publisher_lists.json``.  Then per band b:
  canonical_b = perband.blur_cell_row_path(...) with that list's projection metadata
  seam_b      = production padding_correction._location_correction(...) with the COMBINED-cell loader swapped for the
                band-b cell loader (the correction is linear in the combined images), summed over locations
  blurred_b   = canonical_b + seam_b  (finite pixels only, as production)
Checks: sum_b blurred_b vs production's padding-aware convolved cell (linear_downsample._load_ps1_skycell).
Binning: field_downsample._bin_skycell_contrib (shift 0, ignore bit 12, production cell mask for every band), full OS4
grid; sparse (pix, sums) per key -> ``perband/contrib/<name>.npz`` (keys ``a3_pix_<k>``, ``a3_sum_<k>``, ``a3_count``).
Reduce with f04_reduce.

Port of e2e ``f03_band_contrib.py``; numerics unchanged.  Removed: the S22 fallbacks for cells without a production
convolved cell / without a cross-projection source in the store (never hit for F1: 986/986 ok); such a cell is now an
error instead of a silently weaker check.  The store recipe (and hence the production cell it is compared with) carries
the chain's band weights (``paths.chain_band_weights``).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .paths import BANDS, MODEL, PSF_SIGMA, RADIUS, chain_band_weights, chain_paths

KEYS = ("prod",) + BANDS
_PUB: dict = {}
_LISTDF: dict = {}


def _publisher_lists(P) -> dict:
    key = str(P.publisher_lists)
    if key not in _PUB:
        _PUB[key] = json.loads(P.publisher_lists.read_text())
    return _PUB[key]


def band_fetcher(P):
    """``get(name) -> {band: float32 cell}`` (every band given the combined NaN pattern) or None; cached."""
    cache = {}

    def get(name):
        if name not in cache:
            p = P.band_cells / f"{name}.npz"
            if not p.exists():
                cache[name] = None
            else:
                z = np.load(p)
                d = {b: z[b] for b in BANDS if b in z.files}
                # production's combined cell is NaN wherever ANY band is NaN, and the blur treats NaN as 0 and
                # restores it -> give every band the combined NaN pattern, else the other bands' light leaks
                # into those pixels (found on 2484.045: 1 px, 1% of peak after the blur)
                nan = np.zeros(next(iter(d.values())).shape, bool)
                for v in d.values():
                    nan |= np.isnan(v)
                cache[name] = {b: np.where(nan, np.float32(np.nan), v) for b, v in d.items()}
        return cache[name]
    return get


def seam_correction(P, name, band, get, df, shape):
    """Sum over locations of production's cross-projection correction, fed band-`band` cells."""
    from syndiff_pipeline.template_creation.processing import padding_correction as PC
    spec = PC.cross_projection_padding_spec(df.loc[name])
    if not spec:
        return None
    own = get(name)
    own_img = np.asarray(own[band], dtype=np.float64)
    orig = PC._load_combined_image

    def band_loader(data_root, projection, cell, *, combined_recipe=None):
        p = str(projection)   # production passes "skycell.PROJ" (combined_store._projection_and_cell)
        c = get(f"{p}.{cell}" if p.startswith("skycell.") else f"skycell.{p}.{cell}")
        return None if c is None or band not in c else np.asarray(c[band], dtype=np.float64)

    PC._load_combined_image = band_loader
    try:
        wcs = PC._cell_wcs(df.loc[name])
        tot = np.zeros(shape, dtype=np.float64)
        for loc, nbrs in PC._grouped_padding_spec(spec).items():
            tot += PC._location_correction(location=loc, neighbors=nbrs, skycell=name, recipient_wcs=wcs,
                                           cell_shape=shape, own_combined=own_img, data_root=P.data,
                                           skycell_df=df, psf_sigma=PSF_SIGMA, kernel_radius=RADIUS)
    finally:
        PC._load_combined_image = orig
    return tot


def one(cfg, name):
    import pandas as pd
    from astropy.io import fits
    from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
    from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path
    from syndiff_pipeline.template_creation.processing import linear_downsample as LD
    from syndiff_pipeline.template_creation.processing import perband as PB
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _bin_skycell_contrib, _try_load_shared_convolved_arrays)
    from syndiff_pipeline.template_creation.processing.field_remap import _find_regmap
    from syndiff_pipeline.template_creation.processing.ps1_process import extract_projection_metadata
    from .f02_band_cells import store_recipe

    P = chain_paths(cfg)
    out = P.contrib / f"{name}.npz"
    if out.exists():
        return name, json.loads(str(np.load(out)["check"]))
    t0 = time.time()
    try:
        pub = _publisher_lists(P)
        df = pd.read_csv(P.skylist).set_index("NAME", drop=False)
        proj = str(df.loc[name, "projection"])
        get = band_fetcher(P)
        if get(name) is None:
            return name, {"error": "no band cell"}
        recipe = store_recipe(cfg, chain_band_weights(cfg))
        stored = _try_load_shared_convolved_arrays(P.data, name, psf_sigma=PSF_SIGMA, combined_recipe=recipe)
        stored = None if stored is None else np.asarray(stored[0], np.float64)
        pl = pub["cells"][name]
        tries, md, chosen = {}, None, None

        def comb(n):
            c = get(n)
            return None if c is None else PB.sum_bands(c)
        for key in pl["order"]:
            d2 = pd.read_csv(pub["lists"][key]).set_index("NAME", drop=False) if key not in _LISTDF else _LISTDF[key]
            _LISTDF[key] = d2
            md2 = extract_projection_metadata(d2.reset_index(drop=True), proj)
            if stored is None:
                md, chosen = md2, key
                break
            r = PB.blur_cell_row_path(name, md2, comb, PSF_SIGMA, RADIUS)
            fin = np.isfinite(r) & np.isfinite(stored)
            e = float(np.abs(r[fin] - stored[fin]).max() / np.nanmax(np.abs(stored))) if fin.any() else 1.0
            tries[key] = e
            if chosen is None or e < tries[chosen]:
                md, chosen = md2, key
            if e <= 1e-5:
                break
        blurred = PB.convolve_band_cells(name, md, get, PSF_SIGMA, RADIUS)
        if blurred is None:
            return name, {"error": "row-path blur returned None"}
        shape = next(iter(blurred.values())).shape
        seam_flux = {}
        for b in BANDS:
            corr = seam_correction(P, name, b, get, df, shape)
            if corr is not None:
                img = blurred[b].astype(np.float64)
                fin = np.isfinite(img)
                img[fin] += corr[fin]
                blurred[b] = img.astype(np.float32)
                seam_flux[b] = float(corr[fin].sum())
        zp, shared, legacy = LD._resolve_convolved_source(ps1_convolved_zarr_path(P.data), data_root=P.data,
                                                          sector=P.sector, camera=P.camera, ccd=P.ccd)
        got = LD._load_ps1_skycell(name, data_root=P.data, shared_convolved_store=shared, legacy_zarr_path=legacy,
                                   zstore_cache={}, skycell_df=df, psf_sigma=PSF_SIGMA, combined_recipe=recipe)
        if got is None:
            return name, {"error": "no production convolved cell in the store"}
        prod, pmask = got
        s = sum(blurred[b].astype(np.float64) for b in BANDS if b in blurred)
        fin = np.isfinite(s) & np.isfinite(prod)
        d = np.abs(s[fin] - prod[fin])
        pk = float(np.nanmax(np.abs(prod))) if fin.any() else 0.0
        chk = dict(n_finite=int(fin.sum()), nan_mismatch=int((np.isfinite(s) != np.isfinite(prod)).sum()),
                   max_rel_to_peak=float(d.max() / pk) if d.size and pk > 0 else 0.0,
                   p999_rel=float(np.quantile(d, 0.999) / pk) if d.size and pk > 0 else 0.0,
                   frac_px_gt_1e5=float(np.mean(d > 1e-5 * pk)) if d.size and pk > 0 else 0.0,
                   xproj=bool(seam_flux), seam_flux=seam_flux, bands=sorted(blurred),
                   list_chosen=chosen, list_tries=tries, no_production_cell=False, seam_source_not_in_store=False)
        arrs = {}
        grid = load_mapping_grid_from_master(P.mapping_dir / P.master_name)
        with fits.open(_find_regmap(P.mapping_dir, P.sector, P.camera, P.ccd, name, oversampling_factor=4)) as h:
            asg = np.asarray(h["TESS_PIXEL_MAP"].data if "TESS_PIXEL_MAP" in h else h[1].data)
        kw = dict(assignment=asg, ps1_mask=pmask, sx_int=0, sy_int=0, base_tess_shape=grid.array_shape_os(),
                  roi_bounds=(grid.ffi_xmin, grid.ffi_ymin, grid.ffi_xmax, grid.ffi_ymax),
                  ignore_mask=LD._ignore_mask_from_bits([12]), mapping_grid=grid)
        for key, img in [("prod", prod)] + [(b, blurred[b]) for b in BANDS if b in blurred]:
            res = _bin_skycell_contrib(ps1_data=img, **kw)
            if res is None:
                continue
            pix, sums, counts, _ = res
            arrs[f"{MODEL}_pix_{key}"] = pix.astype(np.int64)
            arrs[f"{MODEL}_sum_{key}"] = sums.astype(np.float64)
            if key == "prod":
                arrs[f"{MODEL}_count"] = counts
        chk["maps"] = [MODEL]
        chk["seconds"] = time.time() - t0
        P.contrib.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **arrs, check=json.dumps(chk))
        tmp.rename(out)
        return name, chk
    except Exception as e:
        import traceback
        return name, {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}


def run(cfg, k: int = 0, n: int = 1, n_jobs: int = 4) -> dict:
    """Process chunk ``k`` of ``n`` (cells ``cells[k::n]``) with ``n_jobs`` loky workers."""
    from joblib import Parallel, delayed

    P = chain_paths(cfg)
    P.contrib.mkdir(parents=True, exist_ok=True)
    cells = json.loads(P.cells_json.read_text())["cells"]
    mine = cells[k::n]
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(delayed(one)(cfg, c) for c in mine)
    (P.contrib / f"chunk_{k:03d}.json").write_text(json.dumps(dict(res), indent=1))
    print(sum("error" in v for _, v in res), "errors of", len(res))
    return dict(res)


def submit(cfg, n_chunks: int = 40, n_jobs: int = 4, *, do_submit: bool = False) -> Path:
    """Write the Condor array submit file (``queue n_chunks``; one job per chunk) and optionally submit it.

    Uses ``chain.condor`` (executable = repo condor_wrapper, PYTHONPATH = cfg.code.forward_model_root, resources from
    ``condor.request_cpus/request_memory_mb['f03']``).  The shared output/error/log names get ``$(Process)`` so jobs
    do not overwrite each other."""
    from .. import condor
    if cfg.config_path is None:
        raise ValueError("config has no file path; Condor jobs need `--config F.yaml`")
    argv = ["python", "-m", "syndiff_pipeline.forward_model.chain.perband.f03_band_contrib",
            "--config", str(cfg.config_path), "--chunk", "$(Process)", str(n_chunks), "--jobs", str(n_jobs)]
    logs = Path(cfg.out_root) / "condor"
    logs.mkdir(parents=True, exist_ok=True)
    text = condor.submit_text(cfg, "contrib", argv, logs, tag="f03", omp_threads=1, queue=f"queue {n_chunks}")
    text = (text.replace(f"{logs}/f03.out", f"{logs}/f03_$(Process).out")
                .replace(f"{logs}/f03.err", f"{logs}/f03_$(Process).err"))
    sub = logs / "f03.sub"
    sub.write_text(text)
    if do_submit:
        print(condor.submit(sub))
    return sub


def main(argv: Optional[list] = None) -> int:
    import argparse
    from ..config import load_config
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--chunk", nargs=2, type=int, metavar=("K", "N"), default=[0, 1])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--submit", type=int, metavar="N_CHUNKS", help="write + condor_submit an N-chunk array instead of running")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.submit:
        submit(cfg, a.submit, a.jobs, do_submit=True)
        return 0
    run(cfg, a.chunk[0], a.chunk[1], a.jobs)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
