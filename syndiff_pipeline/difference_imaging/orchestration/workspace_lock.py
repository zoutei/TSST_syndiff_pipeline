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

# Downstream analysis stages have per-artifact recipe_ids; omitting them from the
# workspace lock lets you append epsf/centroids without a new workspace_run_id.
_WORKSPACE_FINGERPRINT_EXEMPT_STAGE_KINDS = frozenset(
    {"epsf", "centroids", "per_ffi_wcs", "temporal_wcs"}
)

# ── Fingerprint schema versioning ────────────────────────────────────────────
#
# v1 (unversioned, bare 16-hex on disk): hashed every key of every stage dict,
#     so a pure worker-count change invalidated the whole lane.
# v2 (``v2:`` prefixed): execution-resource keys are stripped before hashing, so
#     retuning parallelism no longer forces a new lane.
#
# Existing lanes carry v1 fingerprints. ``assert_workspace_config_lock`` accepts
# a v1 fingerprint when it matches the v1 recomputation (i.e. the recipe is
# provably unchanged) and rewrites it to v2 in place -- lanes self-heal on first
# touch, so no lane is invalidated and no re-run is forced.
FINGERPRINT_SCHEMA_VERSION = 2
_V2_PREFIX = "v2:"

# Keys describing HOW a stage runs, never WHAT it produces. Stripped from the v2
# hash. ``resources`` is the modern nested spelling; the flat names are the
# legacy spelling still present in every already-frozen run config, so both must
# be stripped for a migrated and an unmigrated config to agree.
_STAGE_RESOURCES_KEY = "resources"
_LEGACY_RESOURCE_KEYS = frozenset(
    {
        "hotpants_n_jobs",
        "hotpants_os_n_jobs",
        "template_cache_max_groups",
        "background_estimate_n_jobs",
        "epsf_n_jobs",
        "centroids_n_jobs",
        "per_ffi_wcs_n_jobs",
        "enable_read_cache",
    }
)
# Nested under ``steps.{spatial,temporal,strap}`` of background_temporal_smoothing.
_NESTED_RESOURCE_KEYS = frozenset({"n_jobs"})

# DENYLIST -- these read like resource knobs but change output bytes or which
# artifacts exist. They must NEVER be added above:
#   convolved_templates.use_patch_cache  runs float64 where production runs
#       float32; convolved_templates_patch_cache.py documents that the two "will
#       not match to float32's own machine epsilon".
#   hotpants.use_c_extension             swaps the C extension for the pure-Python
#       implementation -- same math on paper, different numeric path.
#   steps.temporal.tile_size             selects a different function entirely
#       (savgol_smooth_3d vs savgol_smooth_3d_parallel) once n_jobs > 1.
#   max_ffis, rebuild_*, write_*, materialize_fits
#       change which frames are processed / which artifacts exist.


def _pipeline_for_workspace_fingerprint(pipeline: list) -> list:
    """Return pipeline stages that participate in the workspace config lock."""
    return [
        stage
        for stage in pipeline
        if str(stage.get("kind", "")).strip() not in _WORKSPACE_FINGERPRINT_EXEMPT_STAGE_KINDS
    ]


def _strip_stage_resources(stage: dict) -> dict:
    """Drop execution-resource keys from one stage dict (v2 hashing)."""
    if not isinstance(stage, dict):
        return stage
    out = {
        k: v
        for k, v in stage.items()
        if k != _STAGE_RESOURCES_KEY and k not in _LEGACY_RESOURCE_KEYS
    }
    steps = out.get("steps")
    if isinstance(steps, dict):
        out["steps"] = {
            name: (
                {k: v for k, v in step.items() if k not in _NESTED_RESOURCE_KEYS}
                if isinstance(step, dict)
                else step
            )
            for name, step in steps.items()
        }
    return out


class WorkspaceConfigMismatchError(RuntimeError):
    """Raised when a diff config does not match the frozen workspace snapshot."""


def diff_config_fingerprint(cfg: SynDiffConfig, *, version: int = FINGERPRINT_SCHEMA_VERSION) -> str:
    """Stable hash for workspace config lock (matches orchestrator diff stage).

    ``version=1`` reproduces the pre-resource-split hash exactly, so an existing
    lane's stored fingerprint can be validated before being migrated to v2.
    """
    pipeline = _pipeline_for_workspace_fingerprint(cfg.pipeline)
    parts = [
        "diff",
        str(cfg.sector),
        str(cfg.camera),
        str(cfg.ccd),
    ]
    if version == 1:
        parts.append(json.dumps(pipeline, sort_keys=True, default=str))
        parts.append(json.dumps(cfg.additional_forced_targets, sort_keys=True, default=str))
        # pipeline_plots writes only diagnostic PNGs; it is a resource knob in v2.
        parts.append(str(cfg.pipeline_plots))
    else:
        parts.append(
            json.dumps(
                [_strip_stage_resources(s) for s in pipeline], sort_keys=True, default=str
            )
        )
        parts.append(json.dumps(cfg.additional_forced_targets, sort_keys=True, default=str))
    parts.append(str(getattr(cfg, "workspace_run_id", None) or ""))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def versioned_fingerprint(cfg: SynDiffConfig) -> str:
    """The value written to disk today: a ``v2:``-prefixed fingerprint."""
    return f"{_V2_PREFIX}{diff_config_fingerprint(cfg, version=2)}"


def _fingerprint_path(ws_root: str | Path) -> Path:
    """Fingerprint path.
    
    Parameters
    ----------
    ws_root : str | Path
    
    Returns
    -------
    Path"""
    return Path(ws_root) / DIFF_CONFIG_FINGERPRINT_BASENAME


def _snapshot_path(ws_root: str | Path) -> Path:
    """Snapshot path.
    
    Parameters
    ----------
    ws_root : str | Path
    
    Returns
    -------
    Path"""
    return Path(ws_root) / DIFF_CONFIG_SNAPSHOT_BASENAME


def _read_stored_fingerprint(ws_root: str | Path) -> str | None:
    """Read stored fingerprint.
    
    Parameters
    ----------
    ws_root : str | Path
    
    Returns
    -------
    str | None"""
    path = _fingerprint_path(ws_root)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text if text else None


def _write_fingerprint_file(ws_root: str | Path, value: str) -> None:
    """Atomically (re)write the fingerprint file, even though it is mode 444.

    ``os.replace`` acts on the directory entry, so the target's read-only mode
    does not block it.
    """
    fp_path = _fingerprint_path(ws_root)
    tmp_fp = fp_path.with_name(f"{fp_path.name}.tmp.{os.getpid()}")
    tmp_fp.write_text(value + "\n", encoding="utf-8")
    os.replace(tmp_fp, fp_path)
    try:
        os.chmod(fp_path, _WORKSPACE_SNAPSHOT_MODE)
    except OSError as exc:
        log.warning("Could not chmod fingerprint read-only: %s", exc)


def assert_workspace_config_lock(ws_root: str | Path, cfg: SynDiffConfig) -> None:
    """
    Require incoming *cfg* to match the frozen workspace snapshot when present.

    First run (no snapshot): no-op. Re-run with matching fingerprint: no-op.
    Mismatch: raise :class:`WorkspaceConfigMismatchError`.

    A legacy (unversioned) v1 fingerprint is accepted when it matches the v1
    recomputation -- that proves the recipe is unchanged and only the hashing
    rules moved -- and is rewritten to v2 in place, so the lane self-heals on
    first touch and subsequent resource retunes no longer trip the lock.
    """
    snap = _snapshot_path(ws_root)
    if not snap.is_file():
        return

    stored = _read_stored_fingerprint(ws_root)
    if stored is None:
        raise WorkspaceConfigMismatchError(
            f"Workspace {ws_root} has {DIFF_CONFIG_SNAPSHOT_BASENAME} but missing "
            f"{DIFF_CONFIG_FINGERPRINT_BASENAME}; cannot verify config compatibility."
        )

    incoming_v2 = diff_config_fingerprint(cfg, version=2)
    if stored.startswith(_V2_PREFIX):
        if stored[len(_V2_PREFIX) :] != incoming_v2:
            raise WorkspaceConfigMismatchError(
                f"Workspace {ws_root} was created with a different diff config "
                f"(stored fingerprint {stored!r}, incoming {_V2_PREFIX + incoming_v2!r}). "
                f"Use a new output_store_name to start a new lane."
            )
        return

    # Legacy unversioned (v1) fingerprint.
    if stored == diff_config_fingerprint(cfg, version=1):
        _write_fingerprint_file(ws_root, _V2_PREFIX + incoming_v2)
        log.info(
            "Migrated workspace fingerprint %s from v1 %s to v2 %s "
            "(recipe unchanged; execution-resource keys no longer hashed)",
            _fingerprint_path(ws_root),
            stored,
            incoming_v2,
        )
        return

    raise WorkspaceConfigMismatchError(
        f"Workspace {ws_root} was created with a different diff config "
        f"(stored v1 fingerprint {stored!r} does not match this config). "
        f"Use a new output_store_name to start a new lane."
    )


def write_immutable_workspace_config_snapshot(
    ctx: PipelineInvocationContext,
    cfg: SynDiffConfig,
) -> None:
    """Write frozen diff config once; skip on re-run; chmod read-only."""
    from syndiff_pipeline.difference_imaging.orchestration.config import (
        cfg_to_snapshot_dict,
    )

    # Lock artifacts live at the SCC diff lane root (``cfg.output_dir``),
    # as siblings of ``mask_settings.yaml`` -- there is no ``ws/`` tree.
    ws_root = Path(ctx.cfg.output_dir)
    snap = _snapshot_path(ws_root)
    fp_path = _fingerprint_path(ws_root)
    incoming_v2 = diff_config_fingerprint(cfg, version=2)
    incoming = _V2_PREFIX + incoming_v2

    if snap.is_file():
        stored = _read_stored_fingerprint(ws_root)
        # A legacy v1 fingerprint that still matches its v1 recomputation has
        # already been migrated to v2 by assert_workspace_config_lock, which
        # always runs first; accept either spelling so this stays a no-op.
        if stored == incoming or (
            stored is not None
            and not stored.startswith(_V2_PREFIX)
            and stored == diff_config_fingerprint(cfg, version=1)
        ):
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
        yaml.dump(
            cfg_to_snapshot_dict(cfg),
            fh,
            default_flow_style=False,
            sort_keys=False,
        )
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
