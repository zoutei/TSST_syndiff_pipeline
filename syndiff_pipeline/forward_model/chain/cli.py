"""Command line: ``python -m syndiff_pipeline.forward_model.chain <stage> --config F.yaml [--condor] [--force]``.

Stages owned here: ``scene_boot scene_final fit refit wcs mapping gates compare status``.
Stages dispatched to the per-band/kernel half (modules imported lazily; a missing module gives a clear error):

    select -> perband.select        lists -> perband.lists          band_cells -> perband.band_cells
    contrib -> perband.contrib      reduce -> perband.reduce        kernels -> kernels
    hotpants -> hotpants_ref        final -> final                  score -> score

Dispatch contract for those modules: ``run(cfg, **kw)`` where ``kw`` may contain ``force`` (bool), ``condor`` (bool)
and ``args`` (list of extra CLI tokens, e.g. ``contrib`` slice arguments); only the keywords the function accepts are
passed. If it does not accept ``condor`` and ``--condor`` was given, the whole stage is resubmitted to Condor as
``python -m ...chain <stage> --config F.yaml <args>`` (resources from ``condor.request_*``; the key is looked up by stage
name in ``condor.request_cpus``/``request_memory_mb`` and falls back to ``fit``).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

from .config import STAGES, ChainConfig, ConfigError, is_done, load_config

OWN_STAGES = ("scene_boot", "scene_final", "init_boot", "init_final", "nbr_boot", "nbr_final", "fit", "refit",
              "folds_boot", "folds_final", "wcs", "mapping", "gates", "compare", "status")
DISPATCH = {  # CLI stage -> module under forward_model.chain
    "select": "perband.select", "lists": "perband.lists", "band_cells": "perband.band_cells",
    "contrib": "perband.contrib", "reduce": "perband.reduce", "kernels": "kernels",
    "hotpants": "hotpants_ref", "final": "final", "score": "score",
}
ALL_STAGES = OWN_STAGES + tuple(DISPATCH)


class ChainError(RuntimeError):
    pass


def _load_stage_module(stage: str):
    name = f"{__package__}.{DISPATCH[stage]}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name and (name == e.name or name.startswith(e.name + ".")):
            raise ChainError(f"stage {stage!r} needs module {name}, which is not present in this checkout") from e
        raise


def dispatch_external(cfg: ChainConfig, stage: str, *, force: bool, condor: bool, args: list[str]):
    mod = _load_stage_module(stage)
    run = getattr(mod, "run", None)
    if run is None:
        raise ChainError(f"{mod.__name__} has no run(cfg, **kw) entry point")
    params = inspect.signature(run).parameters
    takes_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kw = {}
    for k, v in (("force", force), ("condor", condor), ("args", args)):
        if k in params or takes_kw:
            kw[k] = v
    if condor and "condor" not in kw:
        _generic_condor(cfg, stage, args, force)
        return None
    return run(cfg, **kw)


def _parse_pair(s: str) -> tuple[str, str, str]:
    parts = s.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("--pair expects TEST:REF[:label]")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else f"{parts[0]} vs {parts[1]}"


def _generic_condor(cfg: ChainConfig, stage: str, args: list[str], force: bool = False) -> None:
    from .condor import stage_argv, submit, write_submit
    sub = write_submit(cfg, stage, stage_argv(cfg, stage, list(args) + (["--force"] if force else [])), tag=stage)
    print(f"[{stage}] {submit(sub)}  ({sub})")


def status(cfg: ChainConfig) -> None:
    print(f"{cfg.field}: out_root {cfg.out_root}  config_hash {cfg.config_hash()[:12]}")
    for s in STAGES:
        d = cfg.stage_dir(s)
        print(f"  {s:12s} {'DONE' if is_done(d) else ('started' if d.exists() else '-')}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m syndiff_pipeline.forward_model.chain", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=ALL_STAGES)
    ap.add_argument("--config", required=True, help="chain YAML (configs/<field>.yaml)")
    ap.add_argument("--condor", action="store_true", help="submit the stage to HTCondor instead of running it here")
    ap.add_argument("--force", action="store_true", help="rerun even if the stage has a DONE marker")
    ap.add_argument("--header-wcs", action="store_true", help="mapping: bootstrap mapping on the FFI header WCS (no fitted store)")
    ap.add_argument("--warm", action="store_true", help="refit: warm continuation from the calibration fit (refit/warm)")
    ap.add_argument("--hp-d", default=None, help="scene_boot/scene_final: hp_d image to swap into the scene")
    ap.add_argument("--fold", type=int, default=None, help="folds_boot/folds_final: run only this fold (a Condor job)")
    ap.add_argument("--summarise", action="store_true", help="folds_boot/folds_final: write the summary only")
    ap.add_argument("--fit", action="append", default=[], metavar="NAME=DIR", help="compare: extra fit dir (repeatable)")
    ap.add_argument("--pair", action="append", default=[], type=_parse_pair, metavar="TEST:REF[:label]",
                    help="compare: extra pair (repeatable)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a, rest = ap.parse_known_args(argv)
    if rest and a.stage in OWN_STAGES:
        ap.error(f"unrecognised arguments: {' '.join(rest)}")
    try:
        cfg = load_config(a.config)
        if a.stage != "status":
            cfg.check_code_sha()
    except (ConfigError, OSError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    s = a.stage
    try:
        if s == "status":
            status(cfg)
        elif s in ("scene_boot", "scene_final"):
            if a.condor:
                _generic_condor(cfg, s, ["--hp-d", a.hp_d] if a.hp_d else [], a.force)
            else:
                from .scene import run_scene
                run_scene(cfg, s.split("_")[1], hp_d=a.hp_d, force=a.force)
        elif s in ("init_boot", "init_final", "nbr_boot", "nbr_final"):
            if a.condor:
                _generic_condor(cfg, s, [], a.force)
            else:
                from .crossfit import run_init, run_nbr
                (run_init if s.startswith("init") else run_nbr)(cfg, s.split("_")[1], force=a.force)
        elif s in ("folds_boot", "folds_final"):
            from .crossfit import require_photutils_init, run_folds
            require_photutils_init(cfg)
            run_folds(cfg, s.split("_")[1], fold=a.fold, summarise_only=a.summarise, force=a.force,
                      condor=a.condor and a.fold is None)
        elif s in ("fit", "refit"):
            from .fit import run_fit
            run_fit(cfg, "boot" if s == "fit" else "refit", warm=a.warm, force=a.force, condor=a.condor)
        elif s == "wcs":
            if a.condor:
                _generic_condor(cfg, s, [], a.force)
            else:
                from .wcs_export import run_wcs
                run_wcs(cfg, force=a.force)
        elif s == "mapping":
            from .mapping import run_mapping
            run_mapping(cfg, "header" if a.header_wcs else "fitted", force=a.force, condor=a.condor)
        elif s == "gates":
            if a.condor:
                _generic_condor(cfg, s, [], a.force)
            else:
                from .mapping import run_gates
                run_gates(cfg, force=a.force)
        elif s == "compare":
            if a.condor:
                _generic_condor(cfg, s, [], a.force)
            else:
                from .compare import run_compare
                fits = dict(f.split("=", 1) for f in a.fit)
                run_compare(cfg, fits, a.pair, force=a.force)
        else:
            dispatch_external(cfg, s, force=a.force, condor=a.condor, args=rest)
    except (ChainError, ConfigError, FileNotFoundError) as e:
        print(f"{s}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
