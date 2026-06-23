"""Workspace config fingerprint lock and immutable diff_config snapshot."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from syndiff_pipeline.difference_imaging.support.paths import (
    DIFF_CONFIG_SNAPSHOT_BASENAME,
)

if TYPE_CHECKING:
    from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
    from syndiff_pipeline.difference_imaging.orchestration.context import (
        PipelineInvocationContext,
    )

log = logging.getLogger(__name__)

DIFF_CONFIG_FINGERPRINT_BASENAME = "diff_config.fingerprint"
_WORKSPACE_SNAPSHOT_MODE = 0o444


class WorkspaceConfigMismatchError(RuntimeError):
    """Raised when a diff config does not match the frozen workspace snapshot."""


def diff_config_fingerprint(cfg: SynDiffConfig) -> str:
    """Stable hash for workspace config lock (matches orchestrator diff stage)."""
    parts = [
        "diff",
        str(cfg.sector),
        str(cfg.camera),
        str(cfg.ccd),
        json.dumps(cfg.pipeline, sort_keys=True, default=str),
        json.dumps(cfg.additional_forced_targets, sort_keys=True, default=str),
        str(cfg.n_jobs),
        str(cfg.pipeline_plots),
        str(getattr(cfg, "workspace_run_id", None) or ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _fingerprint_path(ws_root: str | Path) -> Path:
    return Path(ws_root) / DIFF_CONFIG_FINGERPRINT_BASENAME


def _snapshot_path(ws_root: str | Path) -> Path:
    return Path(ws_root) / DIFF_CONFIG_SNAPSHOT_BASENAME


def _read_stored_fingerprint(ws_root: str | Path) -> str | None:
    path = _fingerprint_path(ws_root)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text if text else None


def assert_workspace_config_lock(ws_root: str | Path, cfg: SynDiffConfig) -> None:
    """
    Require incoming *cfg* to match the frozen workspace snapshot when present.

  First run (no snapshot): no-op. Re-run with matching fingerprint: no-op.
  Mismatch: raise :class:`WorkspaceConfigMismatchError`.
    """
    snap = _snapshot_path(ws_root)
    if not snap.is_file():
        return

    incoming = diff_config_fingerprint(cfg)
    stored = _read_stored_fingerprint(ws_root)
    if stored is None:
        raise WorkspaceConfigMismatchError(
            f"Workspace {ws_root} has {DIFF_CONFIG_SNAPSHOT_BASENAME} but missing "
            f"{DIFF_CONFIG_FINGERPRINT_BASENAME}; cannot verify config compatibility."
        )
    if stored != incoming:
        raise WorkspaceConfigMismatchError(
            f"Workspace {ws_root} was created with a different diff config "
            f"(stored fingerprint {stored!r}, incoming {incoming!r}). "
            f"Use a new workspace_run_id or workspace_inherit.from to start a new tree."
        )


def _cfg_to_dict(cfg: SynDiffConfig) -> dict:
    from dataclasses import asdict

    return asdict(cfg)


def write_immutable_workspace_config_snapshot(
    ctx: PipelineInvocationContext,
    cfg: SynDiffConfig,
) -> None:
    """Write frozen diff config once; skip on re-run; chmod read-only."""
    ws_root = Path(ctx.workspace_root_path())
    snap = _snapshot_path(ws_root)
    fp_path = _fingerprint_path(ws_root)
    incoming = diff_config_fingerprint(cfg)

    if snap.is_file():
        stored = _read_stored_fingerprint(ws_root)
        if stored == incoming:
            log.info(
                "Workspace config snapshot unchanged (%s); skipping rewrite",
                snap,
            )
            return
        raise WorkspaceConfigMismatchError(
            f"Refusing to overwrite {snap} (fingerprint mismatch)."
        )

    os.makedirs(ws_root, exist_ok=True)
    tmp = snap.with_name(f"{snap.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.dump(_cfg_to_dict(cfg), fh, default_flow_style=False, sort_keys=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, snap)

    tmp_fp = fp_path.with_name(f"{fp_path.name}.tmp.{os.getpid()}")
    tmp_fp.write_text(incoming + "\n", encoding="utf-8")
    os.replace(tmp_fp, fp_path)

    try:
        os.chmod(snap, _WORKSPACE_SNAPSHOT_MODE)
        os.chmod(fp_path, _WORKSPACE_SNAPSHOT_MODE)
    except OSError as exc:
        log.warning("Could not chmod workspace snapshot read-only: %s", exc)

    log.info("Wrote immutable workspace diff config snapshot %s", snap)
