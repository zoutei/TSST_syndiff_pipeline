"""Init, neighbour and K-fold check stages (Paper-1 final recipe, user 2026-10-07).

``which`` is ``boot`` (bootstrap image: ``scene_boot``) or ``final`` (final de-crowded image: ``scene_final``).

* ``init_<which>``  photutils init (``crossfit.photutils_init.build_photutils_init``) on ``scene_<which>`` with the
  hp_d that scene was swapped from; ``fit.init: photutils`` makes ``fit`` / ``refit`` start from it.
* ``nbr_<which>``   ``scene_<which>`` + Gaia neighbours (``neighbours`` config, see ``chain/neighbours.py``), placed
  with the WCS of ``init_<which>``. ``fit`` / ``refit`` train on it when ``neighbours`` is configured.
* ``folds_<which>`` strict K-fold held-out check of the same recipe. Ported from
  ``dev_runs/epsf_model_closure_20261004/crossfit/code/{prepare,init}.py`` and
  ``dev_runs/paper1_final_fits_20261007/code/{prep_base,prep_fold}.py`` + ``training_fixes_20261005/code/
  {evaluate,raster_all}.py`` (numerics unchanged):

  1. ``build_folds`` (seed/tile/pattern from ``crossfit``) on ``scene_<which>``; per fold, the held tiles' stamp
     pixels are blanked (valid = finite = 0, data 0, noise 1) and an evaluation contract is written. Optionally
     asserted equal to ``crossfit.reference_folds``.
  2. per fold k (``--fold k``; ``--condor`` submits one job per fold; without it each fold still runs in its own
     process, because two photutils inits in one process do not reproduce): strict photutils init on the hp_d with the held
     tiles blanked and the held tiles in the reject mask -> neighbour fold scene (that init's WCS) -> fit with the
     recipe -> held-out score (``score_oof``; fixed ePSF/WCS, fluxes re-solved on the untouched evaluation scene
     ``nbr_<which>``) -> rasters + residual grids (held-out, in-fold, all).
  3. ``--summarise``: per-fold medians, held/in-fold gap and the optical-axis bullseye into ``summary.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import ChainConfig, ConfigError, is_done, mark_done, write_provenance

WHICH = ("boot", "final")
T_BINS = tuple((lo, lo + 1) for lo in range(8, 13))


def _check(which: str) -> None:
    if which not in WHICH:
        raise ValueError(f"which must be one of {WHICH}, got {which!r}")


def scene_hp_d(cfg: ChainConfig, which: str) -> Path:
    """The hp_d ``scene_<which>`` was swapped from (from its provenance)."""
    prov = cfg.stage_dir(f"scene_{which}") / "provenance.json"
    if not prov.exists():
        raise FileNotFoundError(f"{prov} missing (run scene_{which} first)")
    return Path(json.loads(prov.read_text())["inputs"]["hp_d"]["path"])


def train_scene(cfg: ChainConfig, which: str) -> Path:
    """Scene the all-star fit trains on: ``nbr_<which>`` if neighbours are configured, else ``scene_<which>``."""
    _check(which)
    return cfg.stage_dir(f"nbr_{which}") if cfg.neighbours is not None else cfg.stage_dir(f"scene_{which}")


# ---------------------------------------------------------------------- init
def run_init(cfg: ChainConfig, which: str, force: bool = False) -> Path:
    from syndiff_pipeline.forward_model.crossfit import photutils_init as PI

    _check(which)
    stage = cfg.stage_dir(f"init_{which}")
    if is_done(stage) and not force:
        print(f"[init_{which}] already done: {stage}")
        return stage
    scene = cfg.stage_dir(f"scene_{which}")
    if not is_done(scene):
        raise FileNotFoundError(f"scene stage not done: {scene}")
    hp_d = scene_hp_d(cfg, which)
    (stage / "DONE").unlink(missing_ok=True)
    PI.build_photutils_init(scene_dir=scene, hp_d=hp_d, out=stage, n_jobs=cfg.condor.request_cpus.get("init", 8))
    write_provenance(stage, cfg, {"scene_bundle": scene / "scene_bundle.npz", "hp_d": hp_d})
    mark_done(stage)
    return stage


# ---------------------------------------------------------------------- neighbours
def run_nbr(cfg: ChainConfig, which: str, force: bool = False) -> Path:
    from . import neighbours as NB

    _check(which)
    nb = cfg.need("neighbours")
    stage = cfg.stage_dir(f"nbr_{which}")
    if is_done(stage) and not force:
        print(f"[nbr_{which}] already done: {stage}")
        return stage
    scene, init = cfg.stage_dir(f"scene_{which}"), cfg.stage_dir(f"init_{which}")
    for d in (scene, init):
        if not is_done(d):
            raise FileNotFoundError(f"stage not done: {d}")
    if force:
        for f in ("scene_bundle.npz", "scene_meta.json", "fit_bundle.npz", "fit_bundle_meta.json", "added_sources.csv"):
            (stage / f).unlink(missing_ok=True)
    (stage / "DONE").unlink(missing_ok=True)
    log = NB.build(scene, stage, full_scene=scene, ledger=nb.ledger, tmax=nb.tmax, gate_override=nb.gate_override,
                   wcs_params=init / "params_init.npz", colour_file=cfg.inputs.colour_file, source=nb.source,
                   gaia_catalog=nb.gaia_catalog)
    (stage / "build.json").write_text(json.dumps(log, indent=2))
    print(f"[nbr_{which}] {log}")
    write_provenance(stage, cfg, {"scene_bundle": scene / "scene_bundle.npz", "ledger": nb.ledger,
                                  "placement_wcs": init / "params_init.npz",
                                  **({"gaia_catalog": nb.gaia_catalog} if nb.gaia_catalog else {}),
                                  "neighbours": {"value": {"source": nb.source, "tmax": nb.tmax,
                                                           "gate_override": nb.gate_override,
                                                           "n_added": log["n_added"]}}})
    mark_done(stage)
    return stage


# ---------------------------------------------------------------------- folds: scenes and tables
def folds_dir(cfg: ChainConfig, which: str) -> Path:
    _check(which)
    return cfg.stage_dir(f"folds_{which}")


def fold_paths(cfg: ChainConfig, which: str, k: int) -> dict[str, Path]:
    d = folds_dir(cfg, which)
    return dict(scene=d / "scenes" / f"fold{k}", init=d / "init" / f"fold{k}", nbr=d / "nbr" / f"fold{k}",
                fit=d / "fits" / f"fold{k}", eval=d / "eval" / f"fold{k}")


def eval_scene(cfg: ChainConfig, which: str) -> Path:
    """Untouched evaluation scene: the all-star training scene (with neighbours when configured)."""
    return train_scene(cfg, which)


def eval_tables(cfg: ChainConfig, which: str, k: int) -> tuple[Path, Path]:
    """(folds table, evaluation contract of fold k) on the evaluation scene's star indexing."""
    d = folds_dir(cfg, which)
    if cfg.neighbours is None:
        return d / "folds.npz", fold_paths(cfg, which, k)["scene"] / "evaluation_contract.npz"
    return d / "eval_tables" / "folds_eval.npz", d / "eval_tables" / f"contract_fold{k}.npz"


def build_fold_scenes(cfg: ChainConfig, which: str) -> Path:
    """Step 1: fold scenes, held pixel masks, contracts, and (with neighbours) the tables padded to the evaluation
    scene. Idempotent per fold (``support_audit.json``)."""
    from syndiff_pipeline.forward_model.crossfit import build_folds as BF, folds as FO

    cf = cfg.crossfit
    src = cfg.stage_dir(f"scene_{which}")
    if not is_done(src):
        raise FileNotFoundError(f"scene stage not done: {src}")
    d = folds_dir(cfg, which)
    for k in range(cf.n_folds):
        sd = d / "scenes" / f"{cfg.field}_fold{k}"
        link = fold_paths(cfg, which, k)["scene"]
        if (sd / "support_audit.json").exists():
            continue
        BF.main(["--scene-dir", str(src), "--name", cfg.field, "--out", str(d), "--seed", str(cf.seed), "--tile",
                 str(cf.tile), "--n-folds", str(cf.n_folds), "--pattern", cf.pattern, "--scene-folds", str(k)])
        f = dict(np.load(d / "folds.npz")); z = dict(np.load(sd / "scene_bundle.npz"))
        S = int(z["stamp"]); kk = np.arange(S * S)
        px = z["cx"][:, None] + (kk % S - S // 2)[None]; py = z["cy"][:, None] + (kk // S - S // 2)[None]
        held = FO.fold_of(f["tile_map"], cf.tile, px, py) == k
        original_valid = z["valid"].copy()
        for key in ["valid", "finite"]:
            z[key] = np.where(held, 0, z[key]).astype(z[key].dtype)
        z["data"] = np.where(held, 0, z["data"]).astype(z["data"].dtype)
        z["noise"] = np.where(held, 1, z["noise"]).astype(z["noise"].dtype)
        np.savez(sd / "scene_bundle.npz", **z)
        y, x = np.mgrid[:2048, :2048]
        np.save(sd / "held_pixel_mask.npy", FO.fold_of(f["tile_map"], cf.tile, x, y) == k)
        core = np.hypot(px - z["x0"][:, None], py - z["y0"][:, None]) <= 3.0
        safe = ((~core) | held).all(1)
        sel = (f["role0"] != 2) & (f["fold"] == k)
        np.savez(sd / "evaluation_contract.npz", source_id=z["source_id"], held_source=sel, core_wholly_held=safe,
                 original_valid=original_valid)
        (sd / "support_audit.json").write_text(json.dumps({
            "field": cfg.field, "fold": k, "n_held_stars": int(sel.sum()), "n_core_wholly_held": int((sel & safe).sum()),
            "masked_stamp_occurrences": int(held.sum()), "source_scene": str(src),
            "source_sha256": hashlib.sha256((src / "scene_bundle.npz").read_bytes()).hexdigest()}, indent=2))
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(sd.resolve())
    if cf.reference_folds is not None:
        new, old = np.load(d / "folds.npz"), np.load(cf.reference_folds)
        for key in ("source_id", "fold", "role0", "tile_map"):
            if not np.array_equal(new[key], old[key]):
                raise AssertionError(f"folds.npz {key} differs from crossfit.reference_folds {cf.reference_folds}")
        print(f"[folds_{which}] folds identical to {cf.reference_folds}")
    if cfg.neighbours is not None:
        from . import neighbours as NB
        from syndiff_pipeline.forward_model import scene_fit as SF
        ev = eval_scene(cfg, which)
        if not is_done(ev):
            raise FileNotFoundError(f"evaluation scene not done: {ev} (run nbr_{which} first)")
        sc0, sce = SF.Scene(src), SF.Scene(ev); n0 = sc0.N; nadd = sce.N - n0
        assert np.array_equal(sce.z["source_id"][:n0], sc0.z["source_id"])
        t = d / "eval_tables"; t.mkdir(parents=True, exist_ok=True)
        fz = dict(np.load(d / "folds.npz")); assert np.array_equal(fz["source_id"], sc0.z["source_id"])
        np.savez(t / "folds_eval.npz", **NB.pad_table(fz, n0, nadd, sce.z["source_id"]))
        for k in range(cf.n_folds):
            ec = dict(np.load(fold_paths(cfg, which, k)["scene"] / "evaluation_contract.npz"))
            assert np.array_equal(ec["source_id"], sc0.z["source_id"])
            np.savez(t / f"contract_fold{k}.npz", **NB.pad_table(ec, n0, nadd, sce.z["source_id"]))
    (d / "SCENES_DONE").write_text("ok\n")
    return d


# ---------------------------------------------------------------------- folds: one fold
def run_fold(cfg: ChainConfig, which: str, k: int, force: bool = False) -> Path:
    """Step 2 for fold ``k``: strict init -> neighbour fold scene -> fit -> held-out evaluation."""
    import pandas as pd
    from astropy.io import fits
    from syndiff_pipeline.forward_model.crossfit import photutils_init as P
    from .fit import fit_command, run_recipe

    fp = fold_paths(cfg, which, k)
    if not (folds_dir(cfg, which) / "SCENES_DONE").exists():
        raise FileNotFoundError(f"fold scenes not built: run folds_{which} first")
    # strict init: held tiles blanked in the hp_d and added to the reject mask
    io = fp["init"]
    if force or not (io / "STRICT_INIT_AUDIT.json").exists():
        io.mkdir(parents=True, exist_ok=True)
        mask = np.load(fp["scene"] / "held_pixel_mask.npy"); path = io / "sanitized_hp_d.fits"
        with fits.open(scene_hp_d(cfg, which)) as hd:
            hd[1].data[mask] = 0; hd[2].data[mask] = 1; hd.writeto(path, overwrite=True)
        original = P.reject_mask

        def reject(meta, hpd_mask, source="scene"):
            m, desc = original(meta, hpd_mask, source)
            return m | mask, desc + " + strict held tile mask"
        P.reject_mask = reject
        try:
            fm = P.build_photutils_init(scene_dir=fp["scene"], hp_d=path, out=io, fold=k,
                                        folds_csv=folds_dir(cfg, which) / "folds.csv",
                                        n_jobs=cfg.condor.request_cpus.get("init", 8))
        finally:
            P.reject_mask = original
        ph = pd.read_parquet(io / "photometry.parquet")
        assert not (ph.held & ph.wcs_fit_used).any()
        (io / "STRICT_INIT_AUDIT.json").write_text(json.dumps({"n_held_in_wcs": int((ph.held & ph.wcs_fit_used).sum()),
            "sanitized_held_pixels": int(mask.sum()), "pool_train_only": fm["crossfit_init"]["star_counts"]}, indent=2))
        path.unlink()
    # neighbour fold scene (training-only WCS)
    scene = fp["scene"]
    if cfg.neighbours is not None:
        from . import neighbours as NB
        nb = cfg.neighbours
        if force or not (fp["nbr"] / "scene_bundle.npz").exists():
            if force:
                for f in ("scene_bundle.npz", "scene_meta.json", "fit_bundle.npz", "fit_bundle_meta.json", "added_sources.csv"):
                    (fp["nbr"] / f).unlink(missing_ok=True)
            log = NB.build(fp["scene"], fp["nbr"], full_scene=cfg.stage_dir(f"scene_{which}"), ledger=nb.ledger,
                           tmax=nb.tmax, gate_override=nb.gate_override, wcs_params=io / "params_init.npz",
                           colour_file=cfg.inputs.colour_file, source=nb.source, gaia_catalog=nb.gaia_catalog)
            (fp["nbr"] / "build.json").write_text(json.dumps(log, indent=2))
        scene = fp["nbr"]
    # fit
    if force or not is_done(fp["fit"]):
        fp["fit"].mkdir(parents=True, exist_ok=True)
        resume = (fp["fit"] / "progress.json").exists() and not force
        cmd = fit_command(cfg, scene, fp["fit"], init_params=io / "params_init.npz", resume=resume)
        write_provenance(fp["fit"], cfg, {"scene_bundle": scene / "scene_bundle.npz", "init_params": io / "params_init.npz",
                                          "colour_file": cfg.inputs.colour_file, "argv": {"cmd": cmd, "resume": resume}})
        run_recipe(cfg, cmd, fp["fit"], tag=f"folds_{which}_{k}")
    evaluate_fold(cfg, which, k)
    mark_done(fp["eval"])
    return fp["eval"]


def evaluate_fold(cfg: ChainConfig, which: str, k: int) -> dict:
    """Held-out score + rasters + residual grids for fold ``k`` (``training_fixes_20261005/code/evaluate.py`` +
    ``raster_all.py``)."""
    from syndiff_pipeline.forward_model.crossfit import raster_oof, score_oof
    from syndiff_pipeline.forward_model.diagnostics import residual_grid as RG

    fp = fold_paths(cfg, which, k); out = fp["eval"]; out.mkdir(parents=True, exist_ok=True)
    source = eval_scene(cfg, which)
    folds, contract = eval_tables(cfg, which, k)
    sel = np.load(contract)
    score, info = score_oof.score(fp["fit"], source, folds, k)
    assert np.array_equal(score["source_id"], sel["source_id"])
    held = sel["held_source"] & sel["core_wholly_held"] & (score["ncore"] >= 15)
    np.savez_compressed(out / "score.npz", **score, held_fixed=held)
    summ = {"fit": str(fp["fit"]), "n_held": int(held.sum()), "bins": {}}
    for lo, hi in T_BINS:
        tb = (score["T"] >= lo) & (score["T"] < hi)
        h, i = held & tb, score["infold"] & tb & (score["ncore"] >= 15) & np.isfinite(score["c2"])
        summ["bins"][f"T{lo}-{hi}"] = {"n_held": int(h.sum()), "n_infold": int(i.sum()),
            "median_c2_held": float(np.median(score["c2"][h])) if h.any() else None,
            "median_c2_infold": float(np.median(score["c2"][i])) if i.any() else None,
            "n_nonpositive_flux_held": int((score["flux"][h] <= 0).sum())}
    (out / "summary.json").write_text(json.dumps(dict(summ, info=info), indent=2, default=str))
    label = f"{cfg.field}_{which}_fold{k}"
    ras = raster_oof.rasterise_fit(fp["fit"], source, k, folds)
    keep = np.isin(ras["source_id"], score["source_id"][held])
    np.savez_compressed(out / "raster_res_model.npz", **{q: v[keep] for q, v in ras.items()})
    RG.main([str(out), label])
    ras = raster_oof.rasterise_fit(fp["fit"], source)          # every trainee, rendered by this fit
    fz = np.load(folds)
    fmap = dict(zip(fz["source_id"].tolist(), fz["fold"].tolist()))
    fd = np.array([fmap.get(int(s), -1) for s in ras["source_id"]])
    for name, keep in (("infold", fd != k), ("all", fd >= 0)):
        dd = out / "insample" / name; dd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dd / "raster_res_model.npz", **{q: v[keep] for q, v in ras.items()})
        RG.main([str(dd), f"{label}_{name}"])
    return summ


# ---------------------------------------------------------------------- folds: summary
def axis_cells(cfg: ChainConfig) -> tuple[str, str]:
    """Residual-grid cell keys ("iy,ix", iy=2 = top) nearest to and farthest from the optical axis. The axis lies
    ~50-60 px outside the top-right corner for CCDs 1/3 and outside the top-left corner for CCDs 2/4 (tess-point;
    memory note tess-optical-axis-pixel-position)."""
    return ("2,2", "0,0") if cfg.scc.ccd in (1, 3) else ("2,0", "0,2")


def cell_width(js: dict, cell: str) -> tuple[float, int]:
    """n-weighted mean width residual (cxx+cyy)/2 [1e-3 px^2] of blue+mid stars, T10-13 (mag rows 1, 2; colour
    columns 0, 1) in one residual-grid cell (``training_fixes_20261005/code/axis_bullseye.py``)."""
    num = den = 0.0
    for jm in (1, 2):
        for jc in (0, 1):
            s = js["cells"][cell][f"{jm},{jc}"]
            if s["cxx"] is None:
                continue
            w = 0.5 * (s["cxx"] + s["cyy"]); num += s["n"] * w; den += s["n"]
    return (num / den if den else float("nan")), int(den)


def summarise(cfg: ChainConfig, which: str) -> dict:
    d = folds_dir(cfg, which)
    axis, far = axis_cells(cfg)
    rows = []
    for k in range(cfg.crossfit.n_folds):
        e = fold_paths(cfg, which, k)["eval"]
        if not is_done(e):
            raise FileNotFoundError(f"fold {k} not evaluated: {e}")
        s = json.loads((e / "summary.json").read_text())
        r = {"fold": k, "n_held": s["n_held"], "bins": s["bins"]}
        for kind, sub in (("held", e), ("all", e / "insample" / "all")):
            js = json.loads(next(sub.glob("resgrid_*.json")).read_text())
            r[f"{kind}_axis"], r[f"{kind}_axis_n"] = cell_width(js, axis)
            r[f"{kind}_far"], _ = cell_width(js, far)
        b = s["bins"]["T8-9"]
        r["gap_T8_9"] = (b["median_c2_held"] / b["median_c2_infold"]) if b["median_c2_held"] and b["median_c2_infold"] else None
        rows.append(r)
    out = {"field": cfg.field, "which": which, "axis_cell": axis, "far_cell": far, "folds": rows}
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    mark_done(d)
    return out


def run_folds(cfg: ChainConfig, which: str, *, fold: int | None = None, summarise_only: bool = False,
              force: bool = False, condor: bool = False) -> Path:
    """``folds_<which>``: scenes (+ submit/run every fold), or one fold (``fold``), or the summary."""
    from .condor import job_env, stage_argv, submit, write_submit

    _check(which)
    d = folds_dir(cfg, which)
    if summarise_only:
        summarise(cfg, which)
        return d
    if fold is not None:
        return run_fold(cfg, which, fold, force=force)
    train = train_scene(cfg, which)
    if not is_done(train):
        raise FileNotFoundError(f"training scene not done: {train}")
    build_fold_scenes(cfg, which)
    write_provenance(d, cfg, {"scene_bundle": cfg.stage_dir(f"scene_{which}") / "scene_bundle.npz",
                              "crossfit": {"value": {"n_folds": cfg.crossfit.n_folds, "seed": cfg.crossfit.seed,
                                                     "tile": cfg.crossfit.tile, "pattern": cfg.crossfit.pattern}}})
    for k in range(cfg.crossfit.n_folds):
        if is_done(fold_paths(cfg, which, k)["eval"]) and not force:
            continue
        argv = stage_argv(cfg, f"folds_{which}", ["--fold", str(k)] + (["--force"] if force else []))
        if condor:
            sub = write_submit(cfg, "fit", argv, tag=f"folds_{which}_{k}")
            print(f"[folds_{which}] fold {k}: {submit(sub)}  ({sub})")
        else:
            # One fresh process per fold, as on Condor: two photutils inits in one process do not reproduce
            # (e2e_final_recipe_20261007 parity, C1 fold 0: 3 of 7227 WCS candidates differ).
            import os
            import subprocess
            import sys
            env = {**os.environ, **job_env(cfg, f"folds_{which}_{k}")}
            subprocess.run([sys.executable, *argv[1:]], check=True, env=env, cwd=cfg.code.forward_model_root)
    if not condor:
        summarise(cfg, which)
    return d


def require_photutils_init(cfg: ChainConfig) -> None:
    if cfg.fit.init != "photutils":
        raise ConfigError("folds stages need fit.init: photutils (strict per-fold inits)")
