"""Difference-imaging stage specifications."""

from __future__ import annotations

import json
import os
from pathlib import Path

from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.common.orchestration.spec import StageRunContext, StageSpec
from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
    collect_diff_workspace_artifacts,
    diff_workspace_complete,
    frozen_diff_config_for_context,
)
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    load_diff_site_policy,
    write_frozen_diff_config,
)


def _diff_site_config_path(ctx: StageRunContext) -> Path:
    for key in ("source_diff_config_path", "diff_config_path"):
        raw = ctx.meta.get(key) or getattr(ctx.runner_cfg, "diff_config_path", "")
        if raw:
            return Path(str(raw)).expanduser().resolve()
    raise ValueError(
        "Diff stage requires source_diff_config_path in run_meta or "
        "diff_config_path on RunnerConfig"
    )


def _event_dir_for_target(ctx: StageRunContext) -> Path:
    return Path(ctx.runner_cfg.workspace_root) / "events" / ctx.target.label()


def _frozen_diff_config_path(ctx: StageRunContext) -> Path:
    return logs.run_dir(ctx.runs_root, ctx.run_id) / "per_target" / ctx.target_label / "diff_config.yaml"


def _diff_config_fingerprint(ctx: StageRunContext) -> str:
    from syndiff_pipeline.difference_imaging.orchestration.workspace_lock import (
        diff_config_fingerprint,
    )

    return diff_config_fingerprint(frozen_diff_config_for_context(ctx))


def execute_diff_stage(ctx: StageRunContext):
    from syndiff_pipeline.difference_imaging.orchestration.execute import run_config_pipeline

    site_path = _diff_site_config_path(ctx)
    frozen_path = _frozen_diff_config_path(ctx)
    cfg = frozen_diff_config_for_context(ctx)
    write_frozen_diff_config(cfg, frozen_path)
    run_config_pipeline(cfg, validate_only=False, diff_log_path=ctx.progress_path)
    event_dir = Path(cfg.output_dir)
    artifacts = collect_diff_workspace_artifacts(cfg, event_dir)
    expected = max(len(artifacts), 1)
    produced = len(artifacts)
    return expected, produced, artifacts


def _verify_diff(ctx: StageRunContext) -> bool:
    cfg = frozen_diff_config_for_context(ctx)
    return diff_workspace_complete(cfg, _event_dir_for_target(ctx))


def _collect_diff_artifacts(ctx: StageRunContext) -> tuple[int, int, list[str]]:
    cfg = frozen_diff_config_for_context(ctx)
    event_dir = _event_dir_for_target(ctx)
    artifacts = collect_diff_workspace_artifacts(cfg, event_dir)
    expected = max(len(artifacts), 1)
    produced = len(artifacts)
    return expected, produced, artifacts


def _diff_condor_resources(cfg):
    from syndiff_pipeline.common.orchestration import condor

    policy = load_diff_site_policy(cfg.diff_config_path)
    c = policy.condor
    return condor.CondorResourceRequest(
        request_cpus=c.request_cpus,
        request_memory_mb=c.request_memory,
        requirements=c.requirements,
        rank=c.rank,
    )


def _diff_stage_snapshot(ctx: StageRunContext) -> dict:
    event_dir = _event_dir_for_target(ctx)
    return {
        "sector": ctx.target.sector,
        "camera": ctx.target.camera,
        "ccd": ctx.target.ccd,
        "target_name": ctx.target.target_name,
        "target_ra": ctx.target.target_ra,
        "target_dec": ctx.target.target_dec,
        "event_dir": str(event_dir),
        "stage": "diff",
        "pool": "diff",
    }


def write_diff_manifest(
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
        "stage": "diff",
        "expected_count": int(expected_count),
        "produced_count": int(produced_count),
        "artifacts": [str(p) for p in (artifacts or [])],
        "config_fingerprint": _diff_config_fingerprint(ctx),
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


DIFF_STAGE = StageSpec(
    name="diff",
    short_name="diff",
    deps=("downsample",),
    pool="diff",
    default_executor="condor",
    execute=execute_diff_stage,
    verify_complete=_verify_diff,
    collect_artifacts=_collect_diff_artifacts,
    config_fingerprint=_diff_config_fingerprint,
    condor_resources=_diff_condor_resources,
    stage_snapshot=_diff_stage_snapshot,
)

DIFF_STAGES: tuple[StageSpec, ...] = (DIFF_STAGE,)
