"""f03: blur each band cell exactly as production blurs the combined cell, seam-correct cross-projection cells, and
bin every band through the OS4 regmap.  One Condor job = one chunk of cells (``queue N``, chunk ``k/n``).

Per cell: choose the skycell list whose row-path blur of the combined cell (sum of band cells) reproduces the STORED
canonical cell best -- candidates in publisher order from ``publisher_lists.json``.  Without ``publisher_lists.json``
(canonical-cell stores, schema v3) the field's own mapping list is the publisher: no search, and
``list_tries["own"]`` records sum_b blur(C_b) vs the stored canonical cell.  Then per band b:
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
import os
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
        pub = _publisher_lists(P) if P.publisher_lists.exists() else None
        df = pd.read_csv(P.skylist).set_index("NAME", drop=False)
        proj = str(df.loc[name, "projection"])
        get = band_fetcher(P)
        if get(name) is None:
            return name, {"error": "no band cell"}
        recipe = store_recipe(cfg, chain_band_weights(cfg))
        stored = _try_load_shared_convolved_arrays(P.data, name, psf_sigma=PSF_SIGMA, combined_recipe=recipe,
                                                   mapping_df=df)
        stored = None if stored is None else np.asarray(stored[0], np.float64)
        tries, md, chosen = {}, None, None

        def comb(n):
            c = get(n)
            return None if c is None else PB.sum_bands(c)
        # Canonical cells (schema v3): the stored cell is keyed by this mapping list's neighbour set, so the field's
        # own list IS the publisher; no candidate search (f01b) is needed and a miss is an error, not a fallback.
        own_list = pub is None
        order = ["own"] if own_list else pub["cells"][name]["order"]
        for key in order:
            if own_list:
                md2 = extract_projection_metadata(df.reset_index(drop=True), proj)
                if stored is None:
                    return name, {"error": "no v3 canonical convolved cell for this mapping list"}
                md, chosen = md2, key   # checked below on sum_b of the band blurs (= blur of the sum; linear)
                break
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
        if chosen == "own":
            r = sum(blurred[b].astype(np.float64) for b in BANDS if b in blurred)
            fin = np.isfinite(r) & np.isfinite(stored)
            tries["own"] = float(np.abs(r[fin] - stored[fin]).max() / np.nanmax(np.abs(stored))) if fin.any() else 1.0
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


# ---------------------------------------------------------------------------------------------------------------------
# Projection mode (2026-10-05): one job = one projection (or a contiguous block of its rows), cells in row order,
# one band at a time, each band-cell file read from NFS once and decompressed bands kept in a size-capped LRU cache
# (consecutive cells share 6 of their 9 neighbourhood cells). Outputs are identical in form to ``one``.
# ---------------------------------------------------------------------------------------------------------------------
class BandCache:
    """Band cells for a job: compressed file bytes read once; ``band(name, b)`` = band b with the cell's combined NaN
    pattern (union over bands, as ``band_fetcher``), decompressed on demand and kept in an LRU capped at ``cap_bytes``.
    Returned arrays are read-only."""

    def __init__(self, band_dir, cap_bytes: float = 6e9):
        from collections import OrderedDict
        self.dir = Path(band_dir)
        self.cap = float(cap_bytes)
        self.raw: dict = {}
        self.nan: dict = {}
        self.lru = OrderedDict()
        self.size = 0
        self.stats = dict(reads=0, decompress=0, hits=0)

    def _bytes(self, name):
        if name not in self.raw:
            p = self.dir / f"{name}.npz"
            self.raw[name] = p.read_bytes() if p.exists() else None
            self.stats["reads"] += 1
        return self.raw[name]

    def _nan(self, name):
        if name not in self.nan:
            import io
            m = None
            with np.load(io.BytesIO(self._bytes(name))) as z:
                for b in BANDS:
                    if b in z.files:
                        v = np.isnan(z[b])
                        m = v if m is None else (m | v)
            self.nan[name] = (np.packbits(m), m.shape)
        return self.nan[name]

    def band(self, name, b):
        key = (name, b)
        if key in self.lru:
            self.lru.move_to_end(key)
            self.stats["hits"] += 1
            return self.lru[key]
        raw = self._bytes(name)
        if raw is None:
            return None
        import io
        with np.load(io.BytesIO(raw)) as z:
            if b not in z.files:
                return None
            a = np.array(z[b], dtype=np.float32)
        self.stats["decompress"] += 1
        pk, shape = self._nan(name)
        a[np.unpackbits(pk, count=shape[0] * shape[1]).reshape(shape).astype(bool)] = np.nan
        a.setflags(write=False)
        self.lru[key] = a
        self.size += a.nbytes
        while self.size > self.cap and len(self.lru) > 1:
            _, old = self.lru.popitem(last=False)
            self.size -= old.nbytes
        return a

    def cells_view(self):
        """``get(name) -> {band: array}``-like lazy view (for ``seam_correction``)."""
        cache = self

        class _View:
            def __init__(self, name):
                self.name = name

            def __contains__(self, b):
                return cache.band(self.name, b) is not None

            def __getitem__(self, b):
                v = cache.band(self.name, b)
                if v is None:
                    raise KeyError(b)
                return v

        return lambda name: (_View(name) if cache._bytes(name) is not None else None)


def one_projection_cell(cfg, name, cache: "BandCache", df, ctx: dict):
    """Per-band contribution of one cell (projection mode). Same outputs and checks as :func:`one`."""
    from syndiff_pipeline.template_creation.processing import linear_downsample as LD
    from syndiff_pipeline.template_creation.processing import padding_correction as PC
    from syndiff_pipeline.template_creation.processing import perband as PB
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _bin_skycell_contrib, _try_load_shared_convolved_arrays)
    from syndiff_pipeline.template_creation.processing.field_remap import _find_regmap
    from astropy.io import fits

    P = ctx["P"]
    out = P.contrib / f"{name}.npz"
    if out.exists():
        return name, json.loads(str(np.load(out)["check"]))
    t0 = time.time()
    try:
        if cache._bytes(name) is None:
            return name, {"error": "no band cell"}
        got = _try_load_shared_convolved_arrays(P.data, name, psf_sigma=PSF_SIGMA, combined_recipe=ctx["recipe"],
                                                mapping_df=df)
        if got is None:
            return name, {"error": "no v3 canonical convolved cell for this mapping list"}
        stored_raw, smask = got
        stored = np.asarray(stored_raw, np.float64)
        xproj = bool(PC.cross_projection_padding_spec(df.loc[name]))
        if xproj:
            g2 = LD._load_ps1_skycell(name, data_root=P.data, shared_convolved_store=ctx["shared"],
                                      legacy_zarr_path=ctx["legacy"], zstore_cache={}, skycell_df=df,
                                      psf_sigma=PSF_SIGMA, combined_recipe=ctx["recipe"])
            if g2 is None:
                return name, {"error": "no production convolved cell in the store"}
            prod, pmask = g2
        else:
            prod, pmask = stored_raw, smask   # no cross-projection padding: the production cell IS the canonical cell
        md = ctx["md"][str(df.loc[name, "projection"])]
        with fits.open(_find_regmap(P.mapping_dir, P.sector, P.camera, P.ccd, name, oversampling_factor=4)) as h:
            asg = np.asarray(h["TESS_PIXEL_MAP"].data if "TESS_PIXEL_MAP" in h else h[1].data)
        grid = ctx["grid"]
        kw = dict(assignment=asg, ps1_mask=pmask, sx_int=0, sy_int=0, base_tess_shape=grid.array_shape_os(),
                  roi_bounds=(grid.ffi_xmin, grid.ffi_ymin, grid.ffi_xmax, grid.ffi_ymax),
                  ignore_mask=LD._ignore_mask_from_bits([12]), mapping_grid=grid)
        arrs, seam_flux, bands_done = {}, {}, []
        acc_pre = acc_post = None
        view = cache.cells_view()
        for b in BANDS:
            r = PB.blur_cell_row_path(name, md, lambda n, _b=b: cache.band(n, _b), PSF_SIGMA, RADIUS)
            if r is None:
                continue
            bands_done.append(b)
            r64 = r.astype(np.float64)
            acc_pre = r64.copy() if acc_pre is None else acc_pre + r64
            if xproj:
                corr = seam_correction(P, name, b, view, df, r.shape)
                if corr is not None:
                    fin = np.isfinite(r64)
                    r64[fin] += corr[fin]
                    seam_flux[b] = float(corr[fin].sum())
                    r = r64.astype(np.float32)
            acc_post = r.astype(np.float64) if acc_post is None else acc_post + r.astype(np.float64)
            res = _bin_skycell_contrib(ps1_data=r, **kw)
            if res is not None:
                pix, sums, _, _ = res
                arrs[f"{MODEL}_pix_{b}"] = pix.astype(np.int64)
                arrs[f"{MODEL}_sum_{b}"] = sums.astype(np.float64)
            del r, r64
        if not bands_done:
            return name, {"error": "row-path blur returned None"}
        res = _bin_skycell_contrib(ps1_data=prod, **kw)
        if res is not None:
            pix, sums, counts, _ = res
            arrs[f"{MODEL}_pix_prod"] = pix.astype(np.int64)
            arrs[f"{MODEL}_sum_prod"] = sums.astype(np.float64)
            arrs[f"{MODEL}_count"] = counts

        def rel(a, ref):
            fin = np.isfinite(a) & np.isfinite(ref)
            d = np.abs(a[fin] - ref[fin])
            pk = float(np.nanmax(np.abs(ref))) if fin.any() else 0.0
            return d, pk, fin
        d0, pk0, _ = rel(acc_pre, stored)
        prod64 = np.asarray(prod, np.float64)
        d, pk, fin = rel(acc_post, prod64)
        chk = dict(n_finite=int(fin.sum()), nan_mismatch=int((np.isfinite(acc_post) != np.isfinite(prod64)).sum()),
                   max_rel_to_peak=float(d.max() / pk) if d.size and pk > 0 else 0.0,
                   p999_rel=float(np.quantile(d, 0.999) / pk) if d.size and pk > 0 else 0.0,
                   frac_px_gt_1e5=float(np.mean(d > 1e-5 * pk)) if d.size and pk > 0 else 0.0,
                   xproj=xproj, seam_flux=seam_flux, bands=sorted(bands_done), list_chosen="own",
                   list_tries={"own": float(d0.max() / pk0) if d0.size and pk0 > 0 else 1.0},
                   no_production_cell=False, seam_source_not_in_store=False, maps=[MODEL], mode="projection",
                   blur_impl=os.environ.get("SYNDIFF_BLUR_METHOD", "dask"))
        chk["seconds"] = time.time() - t0
        P.contrib.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **arrs, check=json.dumps(chk))
        tmp.rename(out)
        return name, chk
    except Exception as e:
        import traceback
        return name, {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}


def projection_tasks(cfg, max_cells: int = 60) -> list[tuple[str, int, int]]:
    """``[(projection, part, n_parts)]``: every projection of ``cells.json``, split into contiguous row blocks of at
    most ~``max_cells`` cells."""
    import pandas as pd
    P = chain_paths(cfg)
    cells = json.loads(P.cells_json.read_text())["cells"]
    df = pd.read_csv(P.skylist).set_index("NAME", drop=False).loc[cells]
    out = []
    for proj, g in df.groupby(df["projection"].astype(str)):
        n_parts = max(1, int(np.ceil(len(g) / max_cells)))
        out += [(proj, k, n_parts) for k in range(n_parts)]
    return out


def run_projection(cfg, projection: str, part: int = 0, n_parts: int = 1, cache_gb: float = 10.0) -> dict:
    """All cells of ``projection`` (row block ``part`` of ``n_parts``), sequentially in (row, x) order."""
    import pandas as pd
    from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
    from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path
    from syndiff_pipeline.template_creation.processing import linear_downsample as LD
    from syndiff_pipeline.template_creation.processing.ps1_process import extract_projection_metadata
    from .f02_band_cells import store_recipe
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass

    P = chain_paths(cfg)
    P.contrib.mkdir(parents=True, exist_ok=True)
    cells = json.loads(P.cells_json.read_text())["cells"]
    df = pd.read_csv(P.skylist).set_index("NAME", drop=False)
    mine = df.loc[[c for c in cells if str(df.loc[c, "projection"]) == str(projection)]]
    mine = mine.sort_values(["y", "x"])
    rows = sorted(mine["y"].unique())
    blocks = np.array_split(np.array(rows), n_parts)
    mine = mine[mine["y"].isin(blocks[part])]
    zp, shared, legacy = LD._resolve_convolved_source(ps1_convolved_zarr_path(P.data), data_root=P.data,
                                                      sector=P.sector, camera=P.camera, ccd=P.ccd)
    ctx = dict(P=P, recipe=store_recipe(cfg, chain_band_weights(cfg)), shared=shared, legacy=legacy,
               grid=load_mapping_grid_from_master(P.mapping_dir / P.master_name),
               md={str(projection): extract_projection_metadata(df.reset_index(drop=True), str(projection))})
    cache = BandCache(P.band_cells, cap_bytes=cache_gb * 1e9)
    t0 = time.time()
    res = {}
    for name in mine["NAME"]:
        n, chk = one_projection_cell(cfg, name, cache, df, ctx)
        res[n] = chk
        print(n, "ERR " + chk["error"] if "error" in chk else
              f"ok max_rel={chk['max_rel_to_peak']:.1e} canon={chk['list_tries']['own']:.1e} t={chk.get('seconds', 0):.0f}s",
              flush=True)
    summary = dict(projection=str(projection), part=part, n_parts=n_parts, n_cells=len(res),
                   n_errors=sum("error" in v for v in res.values()), seconds=time.time() - t0, cache=cache.stats)
    (P.contrib / f"proj_{projection}_{part:02d}of{n_parts:02d}.json").write_text(json.dumps(dict(summary=summary, cells=res), indent=1))
    print(json.dumps(summary))
    return res


def submit_projections(cfg, *, max_cells: int = 60, request_memory_mb: int = 16000, do_submit: bool = False) -> Path:
    """Condor: one single-core job per (projection, row block) of this field (``queue ... from`` a task list)."""
    from .. import condor
    if cfg.config_path is None:
        raise ValueError("config has no file path; Condor jobs need `--config F.yaml`")
    logs = Path(cfg.out_root) / "condor"
    logs.mkdir(parents=True, exist_ok=True)
    tasks = projection_tasks(cfg, max_cells)
    (logs / "f03p_tasks.txt").write_text("".join(f"{p} {k} {n}\n" for p, k, n in tasks))
    argv = ["python", "-m", "syndiff_pipeline.forward_model.chain.perband.f03_band_contrib",
            "--config", str(cfg.config_path), "--projection", "$(proj)", "--part", "$(part)", "$(nparts)"]
    text = condor.submit_text(cfg, "contrib", argv, logs, tag="f03p", omp_threads=1, request_cpus=1,
                              request_memory_mb=request_memory_mb,
                              queue=f"queue proj,part,nparts from {logs / 'f03p_tasks.txt'}")
    text = (text.replace(f"{logs}/f03p.out", f"{logs}/f03p_$(proj)_$(part).out")
                .replace(f"{logs}/f03p.err", f"{logs}/f03p_$(proj)_$(part).err"))
    text = text.replace('environment = "', 'environment = "SYNDIFF_BLUR_METHOD=fft SYNDIFF_FFT_WORKERS=1 ', 1)
    sub = logs / "f03p.sub"
    sub.write_text(text)
    if do_submit:
        print(condor.submit(sub))
    return sub


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
    ap.add_argument("--projection", help="projection mode: run every cell of this projection (row order)")
    ap.add_argument("--part", nargs=2, type=int, metavar=("K", "N"), default=[0, 1], help="row block K of N (projection mode)")
    ap.add_argument("--cache-gb", type=float, default=10.0, help="decompressed band-cell LRU cap; one cell needs 36 band arrays (~5.8 GB), consecutive cells share 24")
    ap.add_argument("--submit-projections", action="store_true", help="write + condor_submit the projection-mode jobs")
    ap.add_argument("--request-memory-mb", type=int, default=16000)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.submit_projections:
        submit_projections(cfg, request_memory_mb=a.request_memory_mb, do_submit=True)
        return 0
    if a.projection:
        res = run_projection(cfg, a.projection, a.part[0], a.part[1], a.cache_gb)
        print("DONE" if not any("error" in v for v in res.values()) else "ERRORS")
        return 0 if not any("error" in v for v in res.values()) else 1
    if a.submit:
        submit(cfg, a.submit, a.jobs, do_submit=True)
        return 0
    run(cfg, a.chunk[0], a.chunk[1], a.jobs)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
