"""HTCondor submit-file writer shared by every heavy stage (modelled on the e2e ``fit.sub`` / ``m02_mapping.sub``).

The executable is the repo's ``condor_wrapper.sh`` (activates the ``syndiff`` conda env, then ``exec "$@"``), so jobs
need no ``getenv``. The environment is set explicitly: ``PYTHONPATH`` = ``cfg.code.forward_model_root`` (so the chain
runs the code it was configured with, not a stale editable install), ``JAX_PLATFORMS=cpu`` and single-threaded BLAS.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from .config import ChainConfig

WRAPPER_REL = "syndiff_pipeline/common/orchestration/condor_wrapper.sh"

# stage -> key into condor.request_cpus / request_memory_mb
RESOURCE_KEY = {"fit": "fit", "refit": "fit", "mapping": "mapping", "contrib": "f03"}


def _condor_quote(tok: str) -> str:
    """One token in HTCondor's new-syntax ``arguments`` string (itself wrapped in double quotes)."""
    tok = tok.replace('"', '""')
    if any(c in tok for c in " \t'"):
        return "'" + tok.replace("'", "''") + "'"
    return tok


def job_env(cfg: ChainConfig, tag: str, omp_threads: int = 4) -> dict[str, str]:
    root = cfg.code.forward_model_root
    return {
        "PYTHONPATH": str(root),
        "JAX_PLATFORMS": "cpu",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": str(omp_threads),
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONHASHSEED": "0",
        "MPLCONFIGDIR": f"/tmp/chain-mpl-{cfg.field}-{tag}",
        "JAX_COMPILATION_CACHE_DIR": f"/tmp/chain-jax-{cfg.field}-{tag}",
    }


def resources(cfg: ChainConfig, stage: str) -> tuple[int, int]:
    key = RESOURCE_KEY.get(stage, "fit")
    return cfg.condor.request_cpus[key], cfg.condor.request_memory_mb[key]


def submit_text(cfg: ChainConfig, stage: str, argv: Sequence[str], logs_dir: Path, *, tag: str | None = None,
                request_cpus: int | None = None, request_memory_mb: int | None = None,
                omp_threads: int = 4, queue: str = "queue 1") -> str:
    """Text of a vanilla-universe submit file running ``argv`` (e.g. ``["python", "-m", ...]``) under the wrapper."""
    tag = tag or stage
    cpus, mem = resources(cfg, stage)
    cpus = request_cpus or cpus
    mem = request_memory_mb or mem
    env = " ".join(f"{k}={v}" for k, v in job_env(cfg, tag, omp_threads).items())
    cargs = " ".join(_condor_quote(a) for a in argv)
    root = cfg.code.forward_model_root
    return f"""universe = vanilla
executable = {root}/{WRAPPER_REL}
arguments = "{cargs}"
environment = "{env}"
initialdir = {root}
getenv = false
should_transfer_files = NO
request_cpus = {cpus}
request_memory = {mem}
batch_name = {cfg.field}_{tag}
output = {logs_dir}/{tag}.out
error = {logs_dir}/{tag}.err
log = {logs_dir}/{tag}.log
{queue}
"""


def write_submit(cfg: ChainConfig, stage: str, argv: Sequence[str], *, tag: str | None = None, **kw) -> Path:
    """Write ``out_root/condor/<tag>.sub`` (and ensure the log dir). Does not submit."""
    tag = tag or stage
    logs = cfg.out_root / "condor"
    logs.mkdir(parents=True, exist_ok=True)
    p = logs / f"{tag}.sub"
    p.write_text(submit_text(cfg, stage, argv, logs, tag=tag, **kw))
    return p


def submit(sub: Path) -> str:
    """``condor_submit`` the file; returns condor's last output line."""
    out = subprocess.run(["condor_submit", str(sub)], check=True, capture_output=True, text=True).stdout
    return out.strip().splitlines()[-1] if out.strip() else ""


def stage_argv(cfg: ChainConfig, stage: str, extra: Sequence[str] = ()) -> list[str]:
    """Command that re-runs ``stage`` of this chain config in the foreground (what a Condor job executes)."""
    if cfg.config_path is None:
        raise ValueError("config has no file path; Condor jobs need `--config F.yaml`")
    return ["python", "-m", "syndiff_pipeline.forward_model.chain", stage, "--config", str(cfg.config_path), *extra]
