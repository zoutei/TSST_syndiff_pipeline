"""Star pipeline stage specifications."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.common.orchestration.spec import StageRunContext, StageSpec
from syndiff_pipeline.difference_imaging.orchestration.site_config import SitePaths
from syndiff_pipeline.star.site_config import (
    find_star_target_row,
    load_star_site_policy,
    load_star_targets,
    resolve_star_config_path,
    resolve_star_run_config,
    write_frozen_star_config,
)

log = logging.getLogger(__name__)


def _frozen_star_config_path(ctx: StageRunContext) -> Path:
    return logs.run_dir(ctx.runs_root, ctx.run_id) / "star_config.yaml"


def _frozen_star_targets_path(ctx: StageRunContext) -> Path:
    star_path = (ctx.meta or {}).get("star_targets_path")
    if star_path:
        return Path(star_path).expanduser().resolve()
    rd = logs.run_dir(ctx.runs_root, ctx.run_id)
    legacy = logs.run_star_targets_path(rd)
    if legacy.is_file():
        return legacy
    return logs.run_targets_path(rd)


def _star_config_fingerprint(ctx: StageRunContext) -> str:
    path = _frozen_star_config_path(ctx)
    if not path.is_file():
        return ""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def _site_dir_from_ctx(ctx: StageRunContext) -> Path:
    source = (ctx.meta or {}).get("source_config_path")
    if source:
        return Path(source).expanduser().resolve().parent
    return SitePaths.from_site_dir(".").site_dir


def _resolve_star_run(ctx: StageRunContext):
    star_config_path = resolve_star_config_path(meta=ctx.meta, runner_cfg=ctx.runner_cfg)
    policy = load_star_site_policy(star_config_path)
    site_dir = star_config_path.parent
    star_targets = load_star_targets(_frozen_star_targets_path(ctx), site_dir=site_dir)
    star_row = find_star_target_row(star_targets, ctx.target_label)
    photometry_config_path = getattr(ctx.runner_cfg, "photometry_config_path", "") or ""
    run_config = resolve_star_run_config(
        policy,
        star_row,
        site_dir=site_dir,
        photometry_config_path=photometry_config_path or None,
    )
    workspace_run_id = (ctx.meta or {}).get("workspace_run_id")
    if workspace_run_id is not None and str(workspace_run_id).strip():
        log.warning(
            "run_meta workspace_run_id=%r is deprecated and ignored; "
            "star outputs land in phot_{photometry_run_id}/host_star/",
            workspace_run_id,
        )
    return policy, star_row, run_config, site_dir


def execute_star_stage(ctx: StageRunContext):
    """Execute star stage for one SCC target."""
    from syndiff_pipeline.star.context import load_event_context
    from syndiff_pipeline.star.runner import run_star_pipeline, star_output_root

    policy, star_row, run_config, site_dir = _resolve_star_run(ctx)
    frozen_path = _frozen_star_config_path(ctx)
    write_frozen_star_config(policy, frozen_path)

    event_ctx = load_event_context(
        site=str(site_dir),
        target_name=ctx.target_label,
        star_run_config=run_config,
        star_target_row=star_row,
    )
    manifest_path = run_star_pipeline(event_ctx, run_config=run_config, validate=True)
    artifacts = [str(manifest_path)]
    host_root = star_output_root(
        event_ctx, photometry_run_id=run_config.photometry_run_id
    )
    if host_root.is_dir():
        artifacts.extend(str(p) for p in sorted(host_root.rglob("lightcurve_*.csv")))
    expected = max(len(artifacts), 1)
    produced = len(artifacts)
    return expected, produced, artifacts


def _verify_star(ctx: StageRunContext) -> bool:
    from syndiff_pipeline.star.context import StarPrerequisiteError, load_event_context
    from syndiff_pipeline.star.runner import resolve_star_host_root, verify_star_batch_manifest

    try:
        _policy, _star_row, run_config, site_dir = _resolve_star_run(ctx)
        event_ctx = load_event_context(
            site=str(site_dir),
            target_name=ctx.target_label,
            star_run_config=run_config,
            star_target_row=_star_row,
        )
        host_root = resolve_star_host_root(
            event_ctx,
            run_config.workspace_run_id,
            photometry_run_id=run_config.photometry_run_id,
        )
        return verify_star_batch_manifest(host_root / "batch_manifest.csv")
    except (StarPrerequisiteError, KeyError, ValueError, FileNotFoundError):
        return False


def _collect_star_artifacts(ctx: StageRunContext) -> tuple[int, int, list[str]]:
    from syndiff_pipeline.star.context import load_event_context
    from syndiff_pipeline.star.runner import resolve_star_host_root

    _policy, star_row, run_config, site_dir = _resolve_star_run(ctx)
    event_ctx = load_event_context(
        site=str(site_dir),
        target_name=ctx.target_label,
        star_run_config=run_config,
        star_target_row=star_row,
    )
    host_root = resolve_star_host_root(
        event_ctx,
        run_config.workspace_run_id,
        photometry_run_id=run_config.photometry_run_id,
    )
    artifacts: list[str] = []
    manifest_path = host_root / "batch_manifest.csv"
    if manifest_path.is_file():
        artifacts.append(str(manifest_path))
    if host_root.is_dir():
        artifacts.extend(str(p) for p in sorted(host_root.rglob("lightcurve_*.csv")))
    expected = max(len(artifacts), 1)
    produced = len(artifacts)
    return expected, produced, artifacts


def _star_condor_resources(cfg):
    from syndiff_pipeline.common.orchestration import condor
    from syndiff_pipeline.star.site_config import StarCondorConfig, load_star_site_policy

    star_path = str(getattr(cfg, "star_config_path", "") or "").strip()
    if star_path:
        policy = load_star_site_policy(star_path)
        c = policy.condor
    else:
        c = StarCondorConfig()
    return condor.CondorResourceRequest(
        request_cpus=c.request_cpus,
        request_memory_mb=c.request_memory,
        requirements=c.requirements,
        rank=c.rank,
    )


def _star_stage_snapshot(ctx: StageRunContext) -> dict:
    return {
        "sector": ctx.target.sector,
        "camera": ctx.target.camera,
        "ccd": ctx.target.ccd,
        "target_name": ctx.target.target_name,
        "stage": "star",
        "pool": "star",
    }


def write_star_manifest(
    manifest_path,
    ctx: StageRunContext,
    artifacts: list[str],
    expected_count: int,
    produced_count: int,
) -> dict:
    from datetime import datetime, timezone

    from syndiff_pipeline.template_creation.orchestration.verify import MANIFEST_SCHEMA_VERSION

    path = Path(manifest_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": "star",
        "expected_count": int(expected_count),
        "produced_count": int(produced_count),
        "artifacts": [str(p) for p in (artifacts or [])],
        "config_fingerprint": _star_config_fingerprint(ctx),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return payload


STAR_STAGE = StageSpec(
    name="star",
    short_name="star",
    deps=(),
    pool="star",
    default_executor="condor",
    execute=execute_star_stage,
    verify_complete=_verify_star,
    collect_artifacts=_collect_star_artifacts,
    config_fingerprint=_star_config_fingerprint,
    condor_resources=_star_condor_resources,
    stage_snapshot=_star_stage_snapshot,
)

STAR_STAGES: tuple[StageSpec, ...] = (STAR_STAGE,)
