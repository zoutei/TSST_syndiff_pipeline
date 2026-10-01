"""OS4 PS1->TESS mapping (e2e ``m02_mapping.py`` + ``make_sub.py``) and the mapping gates C/D (``m03``/``m04``).

Two modes (``mode`` / CLI ``--header-wcs``):

* ``fitted``  (default; stage ``mapping``): geometry = the exported fitted-WCS store (``wcs/<version>``) passed to
  pancakes as ``tess_wcs_override``, reference FFI = the science FFI (zero drift). Output ``mapping/oversampling_4``.
* ``header``  (the bootstrap): no override, i.e. the SPOC FFI header WCS. Output = ``inputs.bootstrap_mapping`` if set,
  else ``bootstrap/mapping/oversampling_4``.

Both: ``MappingStageParams(oversampling_factor=4, n_threads=condor.request_cpus.mapping)``; no remap / downsample.
The mapping takes ~86 min on 48 cpus, so it is normally run with ``--condor``; the foreground entry point is what the
job executes.

Gates (fitted mode; ``gates`` stage; same numbers as the e2e ``gates.json``):
  C1  master pixel->skycell array + skycell list vs the reference mapping (``reference.old_mapping``)
  C2  regmap-implied offset vs the fitter's WCS difference (needs ``reference.c5_fit`` and ``reference.old_store``)
  C3  exact prediction of the regmap subcell from each WCS store
  D   every skycell of the new master has a regmap file (needs no reference)
C1-C3 are skipped (``null``) when their references are not configured. gates.json also folds in gates A/B.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .condor import stage_argv, submit, write_submit
from .config import ChainConfig, is_done, mark_done, write_provenance

OS = 4


def _names(cfg: ChainConfig) -> tuple[str, str]:
    s = cfg.scc
    return (f"tess_s{s.sector:04d}_{s.camera}_{s.ccd}_master_pixels2skycells_os4.fits.fz",
            f"tess_s{s.sector:04d}_{s.camera}_{s.ccd}_master_skycells_list_os4.csv")


def mapping_out_dir(cfg: ChainConfig, mode: str) -> tuple[Path, Path]:
    """(stage dir holding DONE/provenance/results, mapping dir ``.../oversampling_4``)."""
    if mode == "fitted":
        st = cfg.stage_dir("mapping")
        return st, st / "oversampling_4"
    if mode == "header":
        if cfg.inputs.bootstrap_mapping:
            return cfg.stage_dir("bootstrap") / "mapping", cfg.inputs.bootstrap_mapping
        st = cfg.stage_dir("bootstrap") / "mapping"
        return st, st / "oversampling_4"
    raise ValueError(f"mode must be 'fitted' or 'header', got {mode!r}")


def _status(man: Path, state: str, **extra) -> None:
    d = json.loads(man.read_text()) if man.exists() else {}
    d.update(status=state, updated=time.strftime("%Y-%m-%dT%H:%M:%S"), **extra)
    man.write_text(json.dumps(d, indent=2, default=str) + "\n")


def do_mapping(cfg: ChainConfig, mode: str = "fitted", n_threads: int | None = None) -> dict:
    """Foreground mapping (the Condor job body). Skips if ``mapping_result.json`` exists."""
    stage, mapdir = mapping_out_dir(cfg, mode)
    stage.mkdir(parents=True, exist_ok=True)
    res_p, man_p = stage / "mapping_result.json", stage / "run_manifest.json"
    if res_p.exists():
        print("mapping_result.json exists; skipping")
        return json.loads(res_p.read_text())
    from syndiff_pipeline.common.download import manifest_basename_from_local
    from syndiff_pipeline.template_creation.orchestration.bundled_assets import skycell_wcs_csv
    from syndiff_pipeline.template_creation.orchestration.stage_params import MappingStageParams
    from syndiff_pipeline.template_creation.processing import pancakes

    mp = MappingStageParams(oversampling_factor=OS,
                            n_threads=int(n_threads or cfg.condor.request_cpus["mapping"]))
    ffi = cfg.ffi_path()
    extra = {}
    override = None
    if mode == "fitted":
        from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import TemporalChebWcsStore
        store = cfg.wcs_store_dir()
        override, btjd = TemporalChebWcsStore(store).for_stem(manifest_basename_from_local(str(ffi)))
        extra = dict(mapping_btjd=btjd, wcs_store=str(store))
    _status(man_p, "mapping_running", mode=mode, mapping_reference=str(ffi), mapping_params=repr(mp), **extra)
    t0 = time.time()
    res = pancakes.process_tess_image_optimized(
        tess_file=str(ffi), skycell_wcs_csv=str(skycell_wcs_csv()), output_path=str(mapdir),
        pad_distance=mp.pad_distance, edge_exclusion=mp.edge_exclusion,
        edge_buffer_large=mp.edge_buffer_large, edge_buffer_small=mp.edge_buffer_small,
        buffer=mp.buffer, tess_buffer=mp.tess_buffer, n_threads=mp.n_threads, overwrite=True,
        max_workers=mp.max_workers, oversampling_factor=OS, x_left_dead=mp.x_left_dead,
        x_right_dead=mp.x_right_dead, y_edge_strip=mp.y_edge_strip,
        template_conv_pad_spare_px=mp.template_conv_pad_spare_px, sci_fwhm=mp.sci_fwhm,
        mapgrid_version=int(mp.mapgrid_version), tess_wcs_override=override)
    res_p.write_text(json.dumps(res, indent=2, default=str) + "\n")
    _status(man_p, "mapping_complete", mapping_seconds=time.time() - t0)
    return res


def run_mapping(cfg: ChainConfig, mode: str = "fitted", *, force: bool = False, condor: bool = False) -> Path:
    stage, mapdir = mapping_out_dir(cfg, mode)
    if is_done(stage) and not force:
        print(f"[mapping:{mode}] already done: {stage}")
        return mapdir
    if mode == "fitted" and not is_done(cfg.stage_dir("wcs")):
        raise FileNotFoundError(f"stage wcs not done: {cfg.stage_dir('wcs')}")
    if force:
        (stage / "DONE").unlink(missing_ok=True)
        (stage / "mapping_result.json").unlink(missing_ok=True)
    if condor:
        extra = (["--header-wcs"] if mode == "header" else []) + (["--force"] if force else [])
        sub = write_submit(cfg, "mapping", stage_argv(cfg, "mapping", extra), tag="mapping" + ("_header" if mode == "header" else ""),
                           omp_threads=4)
        print(f"[mapping:{mode}] {submit(sub)}  ({sub})")
        return mapdir
    do_mapping(cfg, mode)
    ins = {"ffi": cfg.ffi_path()}
    if mode == "fitted":
        ins["wcs_store_manifest"] = cfg.wcs_store_dir() / "manifest.json"
    write_provenance(stage, cfg, {**ins, "mode": {"value": mode}})
    mark_done(stage)
    return mapdir


# ---------------------------------------------------------------------- gates
def _csv_geo_equal(lo, ln, cols):
    if set(lo.NAME) != set(ln.NAME):
        return None
    a = lo.set_index("NAME")[cols].sort_index()
    b = ln.set_index("NAME")[cols].sort_index()
    return bool((a.values == b.values).all())


def _exists(find, mapdir, cfg, name):
    s = cfg.scc
    try:
        return Path(find(mapdir, s.sector, s.camera, s.ccd, name, oversampling_factor=4)).is_file()
    except FileNotFoundError:
        return False


def mapping_gates(cfg: ChainConfig, *, fit_dir=None, scene_dir=None, mapdir=None, out_dir=None,
                  n_sky: int = 70, patch: int = 512) -> dict:
    """Gates C and D -> ``gates_cd.json`` (+ ``gateC_patches.csv`` and the check figure)."""
    from .wcs_export import WcsInputs, _jax, fitter_sky_to_pix, wcs_coeff
    _jax()
    import numpy as np
    import pandas as pd
    from astropy.io import fits
    from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import TemporalChebWcsStore
    from syndiff_pipeline.template_creation.processing.field_remap import _find_regmap, _master_skycell_id_map
    from syndiff_pipeline.template_creation.processing.pancakes import get_ps1_wcs_information

    rng = np.random.default_rng(20260929)
    ref = cfg.reference
    fit_dir = Path(fit_dir or cfg.stage_dir("fit"))
    scene_dir = Path(scene_dir or cfg.stage_dir("scene_boot"))
    MAP = Path(mapdir or mapping_out_dir(cfg, "fitted")[1])
    W5 = Path(out_dir or cfg.stage_dir("mapping"))
    inp = WcsInputs.from_dirs(fit_dir, scene_dir)
    MASTER, SKYLIST = _names(cfg)
    SECTOR, CAMERA, CCD = cfg.scc.sector, cfg.scc.camera, cfg.scc.ccd
    have_ref = bool(ref.old_mapping and ref.old_store and ref.c5_fit)
    out: dict = {}

    mn, n2i_n = _master_skycell_id_map(MAP / MASTER)
    hn = fits.getheader(MAP / MASTER, 1)
    sn = set(n2i_n)
    ln = pd.read_csv(MAP / SKYLIST)
    i2n_n = {v: k for k, v in n2i_n.items()}
    # ------------------------------------------------ C1
    if have_ref:
        OLD_MAP = ref.old_mapping
        mo, n2i_o = _master_skycell_id_map(OLD_MAP / MASTER)
        ho = fits.getheader(OLD_MAP / MASTER, 1)
        geo_keys = [k for k in ho if any(t in k for t in ("XMIN", "XMAX", "YMIN", "YMAX", "PAD", "OVERSAMP", "MAPGRID", "GEOMFP", "COORDFRM"))]
        geo_diff = {k: (ho.get(k), hn.get(k)) for k in geo_keys if ho.get(k) != hn.get(k)}
        i2n_o = {v: k for k, v in n2i_o.items()}
        lut_o = np.array([i2n_o.get(i, "") for i in range(-1, max(i2n_o) + 1)], dtype=object)
        lut_n = np.array([i2n_n.get(i, "") for i in range(-1, max(i2n_n) + 1)], dtype=object)
        names_o = lut_o[mo.astype(np.int64) + 1]
        names_n = lut_n[mn.astype(np.int64) + 1]
        differ = names_o != names_n
        so = set(n2i_o)
        lo = pd.read_csv(OLD_MAP / SKYLIST)
        geo_cols = ["TMPL_XMIN", "TMPL_XMAX", "TMPL_YMIN", "TMPL_YMAX", "PADL", "PADR", "PADB", "PADT", "MAPGRID", "GEOMFP"]
        out["C1"] = {
            "master_shape_old": list(mo.shape), "master_shape_new": list(mn.shape),
            "header_geometry_keys_compared": geo_keys, "header_geometry_differences": geo_diff,
            "n_subcells": int(mo.size), "n_subcells_skycell_differs": int(differ.sum()),
            "frac_subcells_skycell_differs": float(differ.mean()),
            "n_subcells_assigned_old": int((mo >= 0).sum()), "n_subcells_assigned_new": int((mn >= 0).sum()),
            "master_skycells_old": len(so), "master_skycells_new": len(sn),
            "master_skycells_only_old": sorted(so - sn), "master_skycells_only_new": sorted(sn - so),
            "csv_rows_old": int(len(lo)), "csv_rows_new": int(len(ln)),
            "csv_names_only_old": sorted(set(lo.NAME) - set(ln.NAME)), "csv_names_only_new": sorted(set(ln.NAME) - set(lo.NAME)),
            "csv_geometry_columns_equal": _csv_geo_equal(lo, ln, geo_cols),
        }
        if differ.any():
            rr, cc = np.nonzero(differ)
            out["C1"]["differing_subcells_native_px_bbox_scilocal"] = [
                float(cc.min() / OS - 8), float(cc.max() / OS - 8), float(rr.min() / OS - 8), float(rr.max() / OS - 8)]
        del names_o, names_n
        out["C1"]["pass"] = bool(mo.shape == mn.shape and not geo_diff and so == sn and set(lo.NAME) == set(ln.NAME))
    else:
        out["C1"] = out["C2"] = out["C3"] = None
    # ------------------------------------------------ D
    miss_master = [n for n in sorted(sn) if not _exists(_find_regmap, MAP, cfg, n)]
    miss_csv = [n for n in ln.NAME if not _exists(_find_regmap, MAP, cfg, n)]
    nfiles = len(list(MAP.glob(f"tess_s{SECTOR}_{CAMERA}_{CCD}_skycell.*_os4.fits*")))
    out["D"] = {"n_master_skycells": len(sn), "missing_regmap_master": miss_master,
                "n_csv_skycells": int(len(ln)), "n_csv_without_regmap": len(miss_csv),
                "csv_without_regmap_all_absent_from_master": bool(set(miss_csv).isdisjoint(sn)),
                "n_regmap_files": nfiles, "pass": len(miss_master) == 0}
    if have_ref:
        out["D"]["n_regmap_files_old"] = len(list(OLD_MAP.glob(f"tess_s{SECTOR}_{CAMERA}_{CCD}_skycell.*_os4.fits*")))
        # ------------------------------------------------ C2 / C3
        st_new, bt = TemporalChebWcsStore(cfg.wcs_store_dir()).for_stem(cfg.stem)
        st_old, _ = TemporalChebWcsStore(ref.old_store).for_stem(cfg.stem)
        cA3, cC5 = wcs_coeff(fit_dir), wcs_coeff(ref.c5_fit)
        tx0, ty0 = int(hn.get("XMIN", 36)), int(hn.get("YMIN", -8))
        W = mn.shape[1]
        common = sorted(sn & so)
        pick = [common[i] for i in rng.choice(len(common), size=min(n_sky, len(common)), replace=False)]
        # plus the skycells owning the CCD corners / edge midpoints / centre (where W3's field peaks), OS4 subcells
        for r in (40, 4128, 8215):
            for c in (40, 4128, 8215):
                nm = i2n_n[int(mn[r, c])]
                if nm not in pick:
                    pick.append(nm)
        ln_i = ln.set_index("NAME")
        rows = []
        pred = {"new_vs_A3": [0, 0], "new_vs_C5": [0, 0], "old_vs_C5": [0, 0], "old_vs_A3": [0, 0]}

        def subcell(adapter, ra, dec):
            X, Y = adapter.world_to_pixel_values(ra, dec)
            col = np.floor((np.asarray(X) - tx0 + 0.5) * OS).astype(np.int64)
            row = np.floor((np.asarray(Y) - ty0 + 0.5) * OS).astype(np.int64)
            return row * W + col

        for k, name in enumerate(pick):
            rn = fits.getdata(_find_regmap(MAP, SECTOR, CAMERA, CCD, name, oversampling_factor=4), 1).astype(np.int64)
            ro = fits.getdata(_find_regmap(OLD_MAP, SECTOR, CAMERA, CCD, name, oversampling_factor=4), 1).astype(np.int64)
            _, pw, shape = get_ps1_wcs_information(ln_i.loc[name])
            assert rn.shape == shape == ro.shape
            # ---- C3: 4000 random PS1 pixels valid in both
            vy, vx = np.nonzero((rn >= 0) & (ro >= 0))
            if vy.size == 0:
                continue
            s = rng.choice(vy.size, size=min(4000, vy.size), replace=False)
            ra, dec = pw.pixel_to_world_values(vx[s].astype(float), vy[s].astype(float))
            pn, po = subcell(st_new, ra, dec), subcell(st_old, ra, dec)
            for key, reg, p in (("new_vs_A3", rn, pn), ("new_vs_C5", rn, po), ("old_vs_C5", ro, po), ("old_vs_A3", ro, pn)):
                pred[key][0] += int((reg[vy[s], vx[s]] == p).sum())
                pred[key][1] += int(s.size)
            # ---- C2: patches
            for py in range(0, shape[0] - patch + 1, patch):
                for px in range(0, shape[1] - patch + 1, patch):
                    a = rn[py:py + patch, px:px + patch]
                    b = ro[py:py + patch, px:px + patch]
                    ok = (a >= 0) & (b >= 0)
                    if ok.mean() < 0.95:
                        continue
                    dcol = (a[ok] % W) - (b[ok] % W)
                    drow = (a[ok] // W) - (b[ok] // W)
                    yy, xx = np.mgrid[py:py + patch:32, px:px + patch:32]
                    r2, d2 = pw.pixel_to_world_values(xx.ravel().astype(float) + 15.5, yy.ravel().astype(float) + 15.5)
                    xa, ya = fitter_sky_to_pix(inp, r2, d2, cA3)
                    xc, yc = fitter_sky_to_pix(inp, r2, d2, cC5)
                    rows.append(dict(skycell=name, px=px, py=py, n=int(ok.sum()), x=float(xc.mean()), y=float(yc.mean()),
                                     dx_map=float(dcol.mean() / OS), dy_map=float(drow.mean() / OS),
                                     frac_changed=float(np.mean((dcol != 0) | (drow != 0))),
                                     max_abs_step=int(max(np.abs(dcol).max(), np.abs(drow).max())),
                                     dx_w3=float((xa - xc).mean()), dy_w3=float((ya - yc).mean())))
            print(f"{k + 1}/{len(pick)} {name} patches={len(rows)}", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(W5 / "gateC_patches.csv", index=False)
        rx, ry = (df.dx_map - df.dx_w3) * 1e3, (df.dy_map - df.dy_w3) * 1e3
        mag_w3 = np.hypot(df.dx_w3, df.dy_w3) * 1e3
        mag_map = np.hypot(df.dx_map, df.dy_map) * 1e3
        # null: the regmap-implied offset if the mapping had NOT changed its WCS is 0 -> residual = -W3 field
        out["C2"] = {"n_patches": int(len(df)), "patch_ps1_px": patch, "n_skycells_sampled": len(pick),
                     "w3_field_at_patches_mpx": {"median": float(np.median(mag_w3)), "p99": float(np.percentile(mag_w3, 99)), "max": float(mag_w3.max())},
                     "regmap_implied_mpx": {"median": float(np.median(mag_map)), "p99": float(np.percentile(mag_map, 99)), "max": float(mag_map.max())},
                     "resid_dx_mpx": {"median": float(np.median(rx)), "rms": float(np.sqrt(np.mean(rx ** 2))), "maxabs": float(np.abs(rx).max())},
                     "resid_dy_mpx": {"median": float(np.median(ry)), "rms": float(np.sqrt(np.mean(ry ** 2))), "maxabs": float(np.abs(ry).max())},
                     "corr_dx": float(np.corrcoef(df.dx_map, df.dx_w3)[0, 1]), "corr_dy": float(np.corrcoef(df.dy_map, df.dy_w3)[0, 1]),
                     "slope_dx": float(np.polyfit(df.dx_w3, df.dx_map, 1)[0]), "slope_dy": float(np.polyfit(df.dy_w3, df.dy_map, 1)[0]),
                     "frac_ps1_pixels_changed_median": float(df.frac_changed.median()), "max_abs_subcell_step": int(df.max_abs_step.max())}
        out["C3"] = {k: {"match": v[0], "n": v[1], "frac": v[0] / max(v[1], 1)} for k, v in pred.items()}
        out["C2"]["pass"] = bool(np.sqrt(np.mean(rx ** 2 + ry ** 2)) < 0.25 * np.sqrt(np.mean(mag_w3 ** 2))
                                 and out["C2"]["corr_dx"] > 0.9 and out["C2"]["corr_dy"] > 0.9)
        out["C3"]["pass"] = bool(out["C3"]["new_vs_A3"]["frac"] > out["C3"]["new_vs_C5"]["frac"]
                                 and out["C3"]["old_vs_C5"]["frac"] > out["C3"]["old_vs_A3"]["frac"])
        if ref.w3_wcs_offset_npz:
            _figure(df, W5, patch, ref.w3_wcs_offset_npz)
    (W5 / "gates_cd.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def _figure(df, W5, patch, w3_npz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    w3 = np.load(w3_npz)
    gxx, gyy = w3["grid_x"], w3["grid_y"]
    gm = np.hypot(w3["grid_dx"], w3["grid_dy"]) * 1e3
    fig, ax = plt.subplots(2, 3, figsize=(18, 11.5))
    vmax = 14
    a = ax[0, 0]
    im = a.pcolormesh(gxx, gyy, gm, cmap="viridis", vmin=0, vmax=vmax, shading="nearest")
    a.quiver(df.x, df.y, df.dx_w3, df.dy_w3, angles="xy", scale=0.25, scale_units="width", color="w", width=0.003)
    a.set_title("W3: |A3 - C5| fitter WCS (colour) + at patches (arrows)")
    plt.colorbar(im, ax=a, label="mpx")
    a = ax[0, 1]
    sc = a.scatter(df.x, df.y, c=np.hypot(df.dx_map, df.dy_map) * 1e3, cmap="viridis", vmin=0, vmax=vmax, s=18)
    a.quiver(df.x, df.y, df.dx_map, df.dy_map, angles="xy", scale=0.25, scale_units="width", color="k", width=0.003)
    a.set_title(f"regmap-implied (new - old mapping), {len(df)} patches of {patch}$^2$ PS1 px")
    plt.colorbar(sc, ax=a, label="mpx")
    a = ax[0, 2]
    r = np.hypot(df.dx_map - df.dx_w3, df.dy_map - df.dy_w3) * 1e3
    sc = a.scatter(df.x, df.y, c=r, cmap="magma", s=18, vmin=0, vmax=max(2.0, np.percentile(r, 99)))
    a.set_title("|regmap-implied - W3| [mpx]")
    plt.colorbar(sc, ax=a, label="mpx")
    for a in ax[0]:
        a.set_xlim(-60, 2110)
        a.set_ylim(-60, 2110)
        a.set_aspect("equal")
        a.set_xlabel("science-local x [px]")
        a.set_ylabel("y [px]")
    for a, c, lab in ((ax[1, 0], "x", "dx"), (ax[1, 1], "y", "dy")):
        u, v = df[f"d{c}_w3"] * 1e3, df[f"d{c}_map"] * 1e3
        a.scatter(u, v, s=8, alpha=0.6)
        lim = [min(u.min(), v.min()) - 1, max(u.max(), v.max()) + 1]
        a.plot(lim, lim, "k--", lw=1)
        a.axhline(0, color="0.7", lw=0.8)
        a.set_xlabel(f"W3 {lab} (fitter A3 - C5) [mpx]")
        a.set_ylabel(f"regmap-implied {lab} [mpx]")
        a.set_title(f"{lab}: corr {np.corrcoef(u, v)[0, 1]:.3f}, rms resid {np.sqrt(np.mean((v - u) ** 2)):.2f} mpx")
    a = ax[1, 2]
    a.hist((df.dx_map - df.dx_w3) * 1e3, bins=40, histtype="step", label="dx resid")
    a.hist((df.dy_map - df.dy_w3) * 1e3, bins=40, histtype="step", label="dy resid")
    a.hist(np.hypot(df.dx_w3, df.dy_w3) * 1e3, bins=40, histtype="step", color="0.5", label="|W3 field| (null: mapping unchanged)")
    a.set_xlabel("mpx")
    a.legend()
    a.set_title("residuals vs the size of the effect")
    fig.suptitle("OS4 mapping check: regmap-implied offset vs the fitter's WCS difference to the reference")
    fig.tight_layout()
    fig.savefig(W5 / "mapping_check.png", dpi=95)
    plt.close(fig)


def collect_gates(cfg: ChainConfig, wcs_dir=None, map_dir=None) -> dict:
    """Fold gates A-D + mapping timing into ``gates.json`` (e2e m04)."""
    W = Path(wcs_dir or cfg.stage_dir("wcs"))
    M = Path(map_dir or cfg.stage_dir("mapping"))
    w = json.loads((W / "wcs_export_gates.json").read_text())
    cd = json.loads((M / "gates_cd.json").read_text())
    mr = json.loads((M / "mapping_result.json").read_text())
    g = {"store": w["store"], "mapping": str(M / "oversampling_4"),
         "mapping_seconds": mr["processing_time_seconds"], "mapping_skycells": mr["processed_skycells"],
         "A": w["gateA"], "B": w["gateB"], "C": {"C1_grid_and_skycells": cd["C1"], "C2_regmap_offset_vs_W3": cd["C2"],
                                                  "C3_exact_subcell_prediction": cd["C3"]}, "D": cd["D"]}
    flags = [g["A"]["pass"], g["B"]["pass"], cd["D"]["pass"]] + [cd[k]["pass"] for k in ("C1", "C2", "C3") if cd[k]]
    g["evaluated"] = {"B": g["B"]["pass"] is not None, "C": cd["C1"] is not None}
    g["all_pass"] = all(f for f in flags if f is not None)
    (M / "gates.json").write_text(json.dumps(g, indent=2) + "\n")
    return g


def run_gates(cfg: ChainConfig, force: bool = False) -> Path:
    """Stage ``mapping`` gates (after mapping is done): writes ``mapping/gates_cd.json`` + ``gates.json``."""
    st = cfg.stage_dir("mapping")
    if (st / "gates.json").exists() and not force:
        print(f"[gates] already done: {st / 'gates.json'}")
        return st
    if not is_done(st):
        raise FileNotFoundError(f"mapping not done: {st}")
    mapping_gates(cfg)
    g = collect_gates(cfg)
    print("all_pass", g["all_pass"])
    if not g["all_pass"]:
        raise RuntimeError(f"mapping gates failed; see {st / 'gates.json'}")
    return st
