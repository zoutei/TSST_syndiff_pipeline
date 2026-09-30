# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Slim parameter checkpoint history (leaf-only npz) and Colab VM retention."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from . import fit as FIT
from . import runtime as RT

CHECKPOINT_GLOB = "params_s*_step*.npz"
CHECKPOINT_MAIN_RE = re.compile(r"^params_s(\d+)_step(\d+)\.npz$")
CHECKPOINT_PHASE_RE = re.compile(r"^params_s(\d+)_(.+)_step(\d+)\.npz$")
DEFAULT_KEEP_LAST = 30
DEFAULT_MIN_AGE_S = 180.0
DISK_PRESSURE_FREE_BYTES = 500 * 1024 * 1024
DISK_PRESSURE_KEEP_LAST = 10


def _log(msg: str) -> None:
    RT.log(msg)


def _phase_slug(phase_label: str) -> str | None:
    """Return a filename-safe phase tag, or None for the default ``main`` block."""
    if phase_label == "main":
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", phase_label.lower()).strip("_")
    return slug or None


def history_checkpoint_name(*, stage: int, step: int, phase_label: str = "main") -> str:
    """Build a unique slim checkpoint filename (handles stage-2 sub-phases)."""
    phase = _phase_slug(phase_label)
    if phase:
        return f"params_s{stage}_{phase}_step{step:05d}.npz"
    return f"params_s{stage}_step{step:05d}.npz"


def history_checkpoint_path(
    job_dir: Path,
    *,
    stage: int,
    step: int,
    phase_label: str = "main",
) -> Path:
    return Path(job_dir) / "checkpoints" / history_checkpoint_name(
        stage=stage, step=step, phase_label=phase_label,
    )


def save_params_leaves_npz(path: Path, params: dict) -> None:
    """Atomically write the trainable leaves (no decoded ePSF convenience arrays).

    Includes ``FIT.ALL_OPTIONAL_LEAVES`` when present. These rolling checkpoints are a
    legitimate ``--bootstrap-init-params`` source -- the r8 -> r9 chain used exactly
    that -- so writing only the required four would silently reset the chromatic term
    to zero on resume, which is indistinguishable from "it never converged".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately leaf-only (module docstring) -- no epsf_repr tag here. This
    # is safe: EM.convert_legacy_params_raw (invoked by every reader of these
    # leaves, e.g. FIT.load_params_npz) detects representation from the
    # array's own shape (is_subpixel_grid), which is exact and unambiguous
    # (CONTRACT_pixel_integrated_epsf.md), so no separate tag is needed for
    # correctness -- only for a human skimming the archive, which the fuller
    # checkpoints (fit.save_params_npz, fit_bundle) already provide.
    keys = list(FIT.STAGE_LEAVES) + [k for k in FIT.ALL_OPTIONAL_LEAVES if k in params]
    arrays = {k: np.asarray(params[k]) for k in keys}
    tmp = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def append_checkpoint_index(
    job_dir: Path,
    *,
    stage: int,
    step: int,
    phase_label: str,
    rel_path: str,
    elapsed_s: float | None = None,
) -> None:
    index_path = Path(job_dir) / "checkpoints" / "checkpoint_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": int(stage),
        "step": int(step),
        "phase": phase_label,
        "path": rel_path,
    }
    if elapsed_s is not None:
        record["elapsed_s"] = float(elapsed_s)
    with index_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _phase_sort_rank(phase: str | None) -> int:
    if phase is None:
        return 0
    if phase == "soft_epsf_only":
        return 1
    if phase == "joint":
        return 2
    return 3


def parse_checkpoint_filename(name: str) -> tuple[int, int, str | None] | None:
    """Parse ``params_s{S}[_phase]_step{N}.npz``; return ``(stage, step, phase)``."""
    m = CHECKPOINT_MAIN_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), None
    m = CHECKPOINT_PHASE_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(3)), m.group(2)
    return None


def checkpoint_sort_key(path: Path) -> tuple[int, int, int, str]:
    parsed = parse_checkpoint_filename(path.name)
    if parsed is None:
        return (10**9, 10**9, 10**9, path.name)
    stage, step, phase = parsed
    # Phase blocks have independent local step counters.  Chronology is stage,
    # phase, then the phase-local step; comparing step before phase can select
    # an early joint checkpoint over a late soft-ePSF checkpoint.
    return (stage, _phase_sort_rank(phase), step, path.name)


def list_sorted_checkpoint_paths(checkpoints_dir: Path) -> list[Path]:
    """All slim history checkpoints in chronological order."""
    if not checkpoints_dir.is_dir():
        return []
    paths = [p for p in checkpoints_dir.glob(CHECKPOINT_GLOB) if p.is_file()]
    return sorted(paths, key=checkpoint_sort_key)


def _list_history_checkpoints(checkpoints_dir: Path) -> list[Path]:
    return list_sorted_checkpoint_paths(checkpoints_dir)


def prune_colab_checkpoints(
    job_dir: Path,
    *,
    keep_last: int = DEFAULT_KEEP_LAST,
    min_age_s: float = DEFAULT_MIN_AGE_S,
    now: float | None = None,
) -> list[Path]:
    """Drop oldest slim history files on Colab once synced copies are likely local.

    Only deletes files under ``checkpoints/`` matching ``CHECKPOINT_GLOB`` that are
    older than ``min_age_s``. Never touches ``params_latest`` or ``params_stage*``.
    """
    job_dir = Path(job_dir)
    checkpoints_dir = job_dir / "checkpoints"
    paths = _list_history_checkpoints(checkpoints_dir)
    if len(paths) <= keep_last:
        return []

    now = time.time() if now is None else float(now)
    try:
        free = shutil_disk_free(job_dir)
    except OSError:
        free = None
    effective_keep = DISK_PRESSURE_KEEP_LAST if (
        free is not None and free < DISK_PRESSURE_FREE_BYTES
    ) else keep_last

    eligible = [p for p in paths if (now - p.stat().st_mtime) >= min_age_s]
    if len(paths) <= effective_keep:
        return []
    n_drop = len(paths) - effective_keep
    to_drop = eligible[:n_drop]
    dropped: list[Path] = []
    for path in to_drop:
        try:
            path.unlink()
            dropped.append(path)
            _log(f"pruned checkpoint history {path.name}")
        except OSError:
            continue
    if dropped:
        _prune_checkpoint_index(checkpoints_dir, {p.name for p in dropped})
    return dropped


def shutil_disk_free(path: Path) -> int:
    import shutil

    return int(shutil.disk_usage(path).free)


def _prune_checkpoint_index(checkpoints_dir: Path, dropped_names: set[str]) -> None:
    index_path = checkpoints_dir / "checkpoint_index.jsonl"
    if not index_path.is_file() or not dropped_names:
        return
    kept: list[str] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        name = Path(str(rec.get("path", ""))).name
        if name not in dropped_names:
            kept.append(line)
    tmp = index_path.with_name(index_path.name + f".tmp.{os.getpid()}")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    os.replace(tmp, index_path)


def save_history_checkpoint(
    job_dir: Path,
    params: dict,
    *,
    stage: int,
    step: int,
    phase_label: str = "main",
    elapsed_s: float | None = None,
    prune: bool = True,
) -> Path:
    """Write slim history checkpoint + index entry; optionally prune old files."""
    path = history_checkpoint_path(job_dir, stage=stage, step=step, phase_label=phase_label)
    save_params_leaves_npz(path, params)
    rel = str(path.relative_to(job_dir))
    append_checkpoint_index(
        job_dir,
        stage=stage,
        step=step,
        phase_label=phase_label,
        rel_path=rel,
        elapsed_s=elapsed_s,
    )
    if prune:
        prune_colab_checkpoints(job_dir)
    return path
