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
                                "argv": {"cmd": cmd, "resume": resume}})
    print(f"[{stage}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=cfg.code.forward_model_root)
    if not (out / "DONE").exists():
        raise RuntimeError(f"scene_fit exited 0 but wrote no DONE in {out}")
    mark_done(out)  # refresh timestamp / idempotent
    return out
