"""Fit stage: ``scene_fit`` with a recipe (e2e ``s3/fit_job.sh``).

``python -m syndiff_pipeline.forward_model.recipe <recipe> -- --scene-dir S --out-dir O --init-params-file P
--colour-file C [extra]`` is run as a subprocess (jax-heavy, long, resumable).

* ``which='boot'``  : scene ``scene_boot/`` -> ``fit/``                (the calibration on the bootstrap image)
* ``which='refit'`` : scene ``scene_final/`` -> ``refit/``              (D14 test: same init + recipe on the final image)
* ``warm=True`` (refit only): continue from the bootstrap fit (``--init-params-file fit/params.npz --init-state-file
  fit/state_stage3.npz --steps-per-stage 0,0,2000``; e2e ``R2_final_warm``) -> ``refit/warm/``.

Resume: if ``<out>/progress.json`` exists the run is resumed (``--resume``); ``force`` discards the DONE marker and
starts fresh. ``<out>/DONE`` is written by scene_fit itself; the chain ``provenance.json`` is written alongside.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .condor import job_env, stage_argv, submit, write_submit
from .config import ChainConfig, is_done, mark_done, write_provenance


# ePSF prior flags that must come EXPLICITLY from the recipe file: scene_fit's code defaults on 648c0bc are the old
# moment-blind fine-neighbour prior (1e8), so a recipe without them would silently fall back to it.
PRIOR_FLAGS = ("lambda-fine-nbr", "lambda-local-poly", "local-poly-window")
# Values the Paper 1 dataset recipe must carry: arm lp7s (prior_bakeoff_20260930), user 2026-09-30.
RECIPE_PRIORS = {"paper1_dataset": {"lambda-fine-nbr": 0.0, "lambda-local-poly": 3.0e8, "local-poly-window": 7.0}}


def check_prior_flags(cfg: ChainConfig) -> dict[str, float]:
    """Refuse to launch unless every ``PRIOR_FLAGS`` key is set in the recipe file itself, is not overridden by
    ``fit.extra_flags``, and (for recipes in ``RECIPE_PRIORS``) has the required value. Returns the resolved values."""
    from syndiff_pipeline.forward_model.recipe import recipe_argv

    argv = recipe_argv(cfg.fit.recipe)
    flags = {argv[i][2:]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")
             and not argv[i + 1].startswith("--")}
    missing = [k for k in PRIOR_FLAGS if k not in flags]
    if missing:
        raise ValueError(f"recipe {cfg.fit.recipe!r} does not set {missing} explicitly; scene_fit would fall back to "
                         "its code defaults (moment-blind fine-neighbour prior)")
    overridden = [k for k in PRIOR_FLAGS if any(f == f"--{k}" or f.startswith(f"--{k}=") for f in cfg.fit.extra_flags)]
    if overridden:
        raise ValueError(f"fit.extra_flags overrides recipe prior flags {overridden}")
    got = {k: float(flags[k]) for k in PRIOR_FLAGS}
    want = RECIPE_PRIORS.get(cfg.fit.recipe)
    if want is not None and got != want:
        raise ValueError(f"recipe {cfg.fit.recipe!r} prior flags {got} != required {want}")
    return got


def fit_dirs(cfg: ChainConfig, which: str, warm: bool = False) -> tuple[Path, Path]:
    """(scene_dir, out_dir)."""
    if which == "boot":
        if warm:
            raise ValueError("warm applies to the refit only")
        return cfg.stage_dir("scene_boot"), cfg.stage_dir("fit")
    if which == "refit":
        out = cfg.stage_dir("refit")
        return cfg.stage_dir("scene_final"), (out / "warm" if warm else out)
    raise ValueError(f"which must be 'boot' or 'refit', got {which!r}")


def fit_command(cfg: ChainConfig, scene_dir: Path, out_dir: Path, *, warm: bool = False,
                resume: bool = False) -> list[str]:
    """argv of the recipe run (without env)."""
    cmd = [sys.executable, "-m", "syndiff_pipeline.forward_model.recipe", cfg.fit.recipe, "--",
           "--scene-dir", str(scene_dir), "--out-dir", str(out_dir)]
    if resume:
        cmd.append("--resume")
    if warm:
        boot = cfg.stage_dir("fit")
        cmd += ["--init-params-file", str(boot / "params.npz"), "--init-state-file", str(boot / "state_stage3.npz"),
                "--steps-per-stage", "0,0,2000"]
    else:
        cmd += ["--init-params-file", str(cfg.need("inputs.init_params"))]
    cmd += ["--colour-file", str(cfg.inputs.colour_file)]
    cmd += list(cfg.fit.extra_flags)
    return cmd


def run_fit(cfg: ChainConfig, which: str = "boot", *, warm: bool = False, force: bool = False,
            condor: bool = False) -> Path:
    scene_dir, out = fit_dirs(cfg, which, warm)
    stage = "fit" if which == "boot" else "refit"
    priors = check_prior_flags(cfg)  # before any launch, Condor or local
    if is_done(out) and not force:
        print(f"[{stage}] already done: {out}")
        return out
    if not is_done(scene_dir):
        raise FileNotFoundError(f"scene stage not done: {scene_dir} (run `{'scene_boot' if which == 'boot' else 'scene_final'}` first)")
    if condor:
        extra = (["--warm"] if warm else []) + (["--force"] if force else [])
        sub = write_submit(cfg, stage, stage_argv(cfg, stage, extra), tag=stage + ("_warm" if warm else ""))
        print(f"[{stage}] {submit(sub)}  ({sub})")
        return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)
    resume = (out / "progress.json").exists() and not force
    cmd = fit_command(cfg, scene_dir, out, warm=warm, resume=resume)
    env = {**os.environ, **job_env(cfg, stage)}
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(cfg.code.forward_model_root)
    write_provenance(out, cfg, {"scene_bundle": scene_dir / "scene_bundle.npz", "scene_meta": scene_dir / "scene_meta.json",
                                "colour_file": cfg.inputs.colour_file,
                                **({} if warm else {"init_params": cfg.need("inputs.init_params")}),
                                "argv": {"cmd": cmd, "resume": resume},
                                "priors": {"value": {"recipe": cfg.fit.recipe, **priors}}})
    print(f"[{stage}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=cfg.code.forward_model_root)
    if not (out / "DONE").exists():
        raise RuntimeError(f"scene_fit exited 0 but wrote no DONE in {out}")
    mark_done(out)  # refresh timestamp / idempotent
    return out
