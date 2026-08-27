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
    """Diff site config path.
    
    Parameters
    ----------
    ctx : StageRunContext
    
    Returns
    -------
    Path"""
    from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
        resolve_diff_site_config_path,
    )

    return resolve_diff_site_config_path(meta=ctx.meta, runner_cfg=ctx.runner_cfg)


def _event_dir_for_target(ctx: StageRunContext) -> Path:
    """Event workspace leaf for one event/SCC pair."""
    from syndiff_pipeline.common.scc_paths import event_scc_leaf

    t = ctx.target
    return event_scc_leaf(
        ctx.runner_cfg.workspace_root,
        t.event_name(),
        t.sector,
        t.camera,
        t.ccd,
    )


def _frozen_diff_config_path(ctx: StageRunContext) -> Path:
    """Frozen diff config path.
    
    Parameters
    ----------
    ctx : StageRunContext
    
    Returns
    -------
    Path"""
    return logs.run_dir(ctx.runs_root, ctx.run_id) / "per_target" / ctx.target_label / "diff_config.yaml"


def _diff_config_fingerprint(ctx: StageRunContext) -> str:
    """Diff config fingerprint.
    
    Parameters
    ----------
    ctx : StageRunContext
    
    Returns
    -------
    str"""
    from syndiff_pipeline.difference_imaging.orchestration.workspace_lock import (
        diff_config_fingerprint,
    )

    return diff_config_fingerprint(frozen_diff_config_for_context(ctx))


# The diff pipeline is split across three Condor stages so only the
# genuinely memory-hungry slice (background_estimate, formerly
# kernel_subtract) needs to bid on the pool's scarce big-RAM boxes; the rest
# can run on any 120GB+ node. cfg.pipeline itself is NEVER filtered -- see
# run_config_pipeline's `kinds` parameter -- so the workspace config lock's
# fingerprint (which always hashes the full, unfiltered pipeline) agrees
# across all three Condor jobs for a target. `syndiff status` still shows
# one "diff" column; progress running-task lines use diff/<substage>.
_DIFF_PREP_KINDS = frozenset({"shared_mask", "kernel_fit", "convolved_templates"})
_BACKGROUND_ESTIMATE_KINDS = frozenset({"background_estimate"})
_DIFF_SPLIT_STAGE_NAMES = ("diff_prep", "background_estimate", "diff")


def _diff_post_kinds(cfg) -> frozenset[str]:
    """Every kind in cfg.pipeline not owned by diff_prep or background_estimate."""
    all_kinds = {
        str(stage.get("kind", "")).strip()
        for stage in cfg.pipeline
        if isinstance(stage, dict)
    }
    return frozenset(all_kinds - _DIFF_PREP_KINDS - _BACKGROUND_ESTIMATE_KINDS)


def _kinds_for_split_stage(cfg, stage_name: str) -> frozenset[str]:
    """Kind subset of cfg.pipeline owned by one of the three split diff stages."""
    if stage_name == "diff_prep":
        return _DIFF_PREP_KINDS
    if stage_name == "background_estimate":
        return _BACKGROUND_ESTIMATE_KINDS
    if stage_name == "diff":
        return _diff_post_kinds(cfg)
    raise ValueError(f"unknown diff split stage {stage_name!r}")


def _execute_diff_split_stage(ctx: StageRunContext, stage_name: str):
    """Shared execute body for diff_prep/background_estimate/diff."""
    from syndiff_pipeline.difference_imaging.orchestration.execute import run_config_pipeline

    frozen_path = _frozen_diff_config_path(ctx)
    cfg = frozen_diff_config_for_context(ctx)
    write_frozen_diff_config(cfg, frozen_path)
    kinds = _kinds_for_split_stage(cfg, stage_name)
    run_config_pipeline(
        cfg,
        validate_only=False,
        diff_log_path=ctx.progress_path,
        force_rerun=ctx.force_rerun,
        kinds=kinds,
    )
    event_dir = Path(cfg.output_dir)
    artifacts = collect_diff_workspace_artifacts(cfg, event_dir)
    expected = max(len(artifacts), 1)
    produced = len(artifacts)
    return expected, produced, artifacts


def execute_diff_prep_stage(ctx: StageRunContext):
    """Run shared_mask/kernel_fit/convolved_templates (120GB-class nodes)."""
    return _execute_diff_split_stage(ctx, "diff_prep")


def execute_background_estimate_stage(ctx: StageRunContext):
    """Run background_estimate, formerly kernel_subtract (500GB-class nodes)."""
    return _execute_diff_split_stage(ctx, "background_estimate")


def execute_diff_stage(ctx: StageRunContext):
    """Run everything after background_estimate: hotpants/epsf/centroids/... ."""
    return _execute_diff_split_stage(ctx, "diff")


def _verify_diff_split_stage(ctx: StageRunContext, stage_name: str) -> bool:
    """Shared verify body for diff_prep/background_estimate/diff."""
    cfg = frozen_diff_config_for_context(ctx)
    kinds = _kinds_for_split_stage(cfg, stage_name)
    return diff_workspace_complete(cfg, _event_dir_for_target(ctx), kinds=kinds)


def _verify_diff_prep(ctx: StageRunContext) -> bool:
    return _verify_diff_split_stage(ctx, "diff_prep")


def _verify_background_estimate(ctx: StageRunContext) -> bool:
    return _verify_diff_split_stage(ctx, "background_estimate")


def _verify_diff(ctx: StageRunContext) -> bool:
    return _verify_diff_split_stage(ctx, "diff")


def _collect_diff_artifacts(ctx: StageRunContext) -> tuple[int, int, list[str]]:
    """Collect diff artifacts.

    Shared across all three split stages: lists everything under the SCC
    lane root, unfiltered by kind. Harmless over-collection for
    diff_prep/background_estimate (their manifest just lists artifacts that
    don't exist yet on disk at that point in the run)."""
    cfg = frozen_diff_config_for_context(ctx)
    event_dir = _event_dir_for_target(ctx)
    artifacts = collect_diff_workspace_artifacts(cfg, event_dir)
    expected = max(len(artifacts), 1)
    produced = len(artifacts)
    return expected, produced, artifacts


def _condor_resources_for_stage(cfg, stage_name: str):
    """Condor resources for one split diff stage, from the per-stage condor: block.

    ``cfg`` here is the ``RunnerConfig`` (see ``launcher.launch_stage``'s
    ``condor_resources(cfg)`` call). Frozen-first: prefer the embedded,
    already-frozen ``cfg.diff.condor_by_stage`` -- read on every Condor
    submit -- over a live re-read of ``cfg.diff_config_path``'s site file.
    This is what makes hand-editing ``diff.condor`` in the frozen
    ``runs/{run_id}/config.yaml`` retune a live run: the supervisor re-reads
    that file every tick, and the new value reaches ``condor.submit_job``
    without needing the live site file to still agree.
    """
    from syndiff_pipeline.common.orchestration import condor

    policy = getattr(cfg, "diff", None)
    if policy is None:
        policy = load_diff_site_policy(cfg.diff_config_path)
    c = policy.condor_by_stage[stage_name]
    return condor.CondorResourceRequest(
        request_cpus=c.request_cpus,
        request_memory_mb=c.request_memory,
        host_stats_min_mem_mb=c.host_stats_min_mem_mb,
        host_stats_max_load15=c.host_stats_max_load15,
    )


def _diff_prep_condor_resources(cfg):
    return _condor_resources_for_stage(cfg, "diff_prep")


def _background_estimate_condor_resources(cfg):
    return _condor_resources_for_stage(cfg, "background_estimate")


def _diff_condor_resources(cfg):
    return _condor_resources_for_stage(cfg, "diff")


def _diff_stage_snapshot_for_stage(ctx: StageRunContext, stage_name: str, pool: str) -> dict:
    event_dir = _event_dir_for_target(ctx)
    return {
        "sector": ctx.target.sector,
        "camera": ctx.target.camera,
        "ccd": ctx.target.ccd,
        "target_name": ctx.target.target_name,
        "target_ra": ctx.target.target_ra,
        "target_dec": ctx.target.target_dec,
        "event_dir": str(event_dir),
        "stage": stage_name,
        "pool": pool,
    }


def _diff_prep_stage_snapshot(ctx: StageRunContext) -> dict:
    return _diff_stage_snapshot_for_stage(ctx, "diff_prep", "diff_prep")


def _background_estimate_stage_snapshot(ctx: StageRunContext) -> dict:
    return _diff_stage_snapshot_for_stage(ctx, "background_estimate", "background_estimate")


def _diff_stage_snapshot(ctx: StageRunContext) -> dict:
    return _diff_stage_snapshot_for_stage(ctx, "diff", "diff")


def write_diff_manifest(
    manifest_path,
    ctx: StageRunContext,
    artifacts: list[str],
    expected_count: int,
    produced_count: int,
    *,
    stage_name: str = "diff",
) -> dict:
    """Write diff manifest.
    
    Parameters
    ----------
    manifest_path
    ctx : StageRunContext
    artifacts : list[str]
    expected_count : int
    produced_count : int
    
    Returns
    -------
    dict"""
    from datetime import datetime, timezone

    from syndiff_pipeline.template_creation.orchestration.verify import MANIFEST_SCHEMA_VERSION

    path = Path(manifest_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": stage_name,
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


DIFF_PREP_STAGE = StageSpec(
    name="diff_prep",
    short_name="diff/diff_prep",
    deps=("downsample",),
    pool="diff_prep",
    default_executor="condor",
    execute=execute_diff_prep_stage,
    verify_complete=_verify_diff_prep,
    collect_artifacts=_collect_diff_artifacts,
    config_fingerprint=_diff_config_fingerprint,
    condor_resources=_diff_prep_condor_resources,
    stage_snapshot=_diff_prep_stage_snapshot,
)

BACKGROUND_ESTIMATE_STAGE = StageSpec(
    name="background_estimate",
    short_name="diff/background_estimate",
    deps=("diff_prep",),
    pool="background_estimate",
    default_executor="condor",
    execute=execute_background_estimate_stage,
    verify_complete=_verify_background_estimate,
    collect_artifacts=_collect_diff_artifacts,
    config_fingerprint=_diff_config_fingerprint,
    condor_resources=_background_estimate_condor_resources,
    stage_snapshot=_background_estimate_stage_snapshot,
)

DIFF_STAGE = StageSpec(
    name="diff",
    short_name="diff",
    deps=("background_estimate",),
    pool="diff",
    default_executor="condor",
    execute=execute_diff_stage,
    verify_complete=_verify_diff,
    collect_artifacts=_collect_diff_artifacts,
    config_fingerprint=_diff_config_fingerprint,
    condor_resources=_diff_condor_resources,
    stage_snapshot=_diff_stage_snapshot,
)

DIFF_STAGES: tuple[StageSpec, ...] = (DIFF_PREP_STAGE, BACKGROUND_ESTIMATE_STAGE, DIFF_STAGE)
