"""Composed SynDiff DAG: template stages plus difference imaging."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from syndiff_pipeline.common.orchestration.spec import PipelineSpec, StageRunContext, StageSpec
from syndiff_pipeline.template_creation.orchestration.stages import resolve_template_context

_PIPELINE: PipelineSpec | None = None


@lru_cache
def get_syndiff_pipeline() -> PipelineSpec:
    from syndiff_pipeline.difference_imaging.orchestration.stages import DIFF_STAGES
    from syndiff_pipeline.template_creation.orchestration.stages import TEMPLATE_STAGES

    return PipelineSpec(
        name="syndiff",
        stages=TEMPLATE_STAGES + DIFF_STAGES,
    )


def _pipeline() -> PipelineSpec:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = get_syndiff_pipeline()
    return _PIPELINE


def get_stage_spec(stage: str) -> StageSpec | None:
    return _pipeline().get(stage)


def build_stage_context(
    *,
    run_id: str,
    runs_root: str,
    target_label: str,
    target,
    runner_cfg,
    stage: str,
    meta: dict | None = None,
    template_resolved=None,
    force_rerun: bool = False,
    progress_path: str | None = None,
) -> StageRunContext:
    ctx = StageRunContext(
        run_id=run_id,
        runs_root=runs_root,
        target_label=target_label,
        target=target,
        runner_cfg=runner_cfg,
        template_resolved=template_resolved,
        meta=dict(meta or {}),
        force_rerun=force_rerun,
        progress_path=progress_path,
    )
    if stage != "diff":
        return resolve_template_context(ctx)
    return ctx


def stage_snapshot(ctx: StageRunContext, stage: str) -> dict:
    spec = _pipeline().require(stage)
    if stage != "diff":
        ctx = resolve_template_context(ctx)
    if spec.stage_snapshot is not None:
        return spec.stage_snapshot(ctx)
    return {"stage": stage}


def config_fingerprint(ctx: StageRunContext, stage: str) -> str:
    spec = _pipeline().require(stage)
    if stage != "diff":
        ctx = resolve_template_context(ctx)
    return spec.config_fingerprint(ctx)


def stage_names() -> tuple[str, ...]:
    return _pipeline().stage_names


def stage_short_names() -> dict[str, str]:
    return _pipeline().stage_short_names()


def resolve_stage_name(name: str) -> str:
    return _pipeline().resolve_stage_name(name)


def __getattr__(name: str) -> Any:
    pipeline = _pipeline()
    if name == "SYNDIFF_PIPELINE":
        return pipeline
    if name == "STAGE_NAMES":
        return pipeline.stage_names
    if name == "STAGE_SHORT_NAMES":
        return pipeline.stage_short_names()
    if name == "STAGE_DEPS":
        return pipeline.stage_deps()
    if name == "STAGE_POOL":
        return pipeline.stage_pools()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
