"""Composed SynDiff DAG: template stages plus difference imaging."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from syndiff_pipeline.common.orchestration.spec import PipelineSpec, StageRunContext, StageSpec

_PIPELINE: PipelineSpec | None = None
_STAGE_SHORT_NAMES: dict[str, str] | None = None

# Display-only short names for CLI progress/notifications. Kept in sync with
# TEMPLATE_STAGES + diff/star without importing heavy stage modules.
_STATIC_STAGE_SHORT_NAMES: dict[str, str] = {
    "diff_prep": "diff/diff_prep",
    "background_estimate": "diff/background_estimate",
    "diff": "diff",
    "photometry": "phot",
    "star": "star",
}

# The diff pipeline runs as three Condor stages (diff_prep -> background_estimate
# -> diff, split so only background_estimate needs big-memory nodes). The status
# grid still shows one "diff" column; syndiff progress running-task lines use
# diff/<substage> so the live Condor job is identifiable.
_DIFF_SPLIT_STAGE_NAMES: tuple[str, ...] = ("diff_prep", "background_estimate", "diff")

# Columns shown by ``syndiff status`` / Discord status grids (bind and star omitted).
STATUS_GRID_STAGES: tuple[str, ...] = (
    "tess_ffi_download",
    "mapping",
    "ps1_download",
    "ps1_process",
    "remap",
    "downsample",
    "diff",
)

# Legacy SQLite stage names from pre-downsample rename runs, plus the two
# split-diff stage names that alias onto the "diff" status-grid column (see
# _DIFF_SPLIT_STAGE_NAMES above -- run_report.py's row aggregation is what
# actually rolls up the 3 real DB rows into one displayed "diff" status).
STATUS_GRID_LEGACY_STAGE_ALIASES: dict[str, str] = {
    "templates": "downsample",
    "tmpl": "downsample",
    "diff_prep": "diff",
    "background_estimate": "diff",
}


@lru_cache
def get_syndiff_pipeline() -> PipelineSpec:
    """Get syndiff pipeline.
    
    Returns
    -------
    PipelineSpec"""
    from syndiff_pipeline.difference_imaging.orchestration.stages import DIFF_STAGES
    from syndiff_pipeline.photometry.orchestration.stages import PHOTOMETRY_STAGES
    from syndiff_pipeline.star.orchestration.stages import STAR_STAGES
    from syndiff_pipeline.template_creation.orchestration.stages import TEMPLATE_STAGES

    return PipelineSpec(
        name="syndiff",
        stages=TEMPLATE_STAGES + DIFF_STAGES + PHOTOMETRY_STAGES + STAR_STAGES,
    )


def _pipeline() -> PipelineSpec:
    """Pipeline.
    
    Returns
    -------
    PipelineSpec"""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = get_syndiff_pipeline()
    return _PIPELINE


def get_stage_spec(stage: str) -> StageSpec | None:
    """Get stage spec.
    
    Parameters
    ----------
    stage : str
    
    Returns
    -------
    StageSpec | None"""
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
    """Build stage context.
    
    Parameters
    ----------
    run_id : str
    runs_root : str
    target_label : str
    target
    runner_cfg
    stage : str
    meta : dict | None, optional, default ``None``
    template_resolved, optional, default ``None``
    force_rerun : bool, optional, default ``False``
    progress_path : str | None, optional, default ``None``
    
    Returns
    -------
    StageRunContext"""
    from syndiff_pipeline.template_creation.orchestration.stages import resolve_template_context

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
    if stage not in ("diff_prep", "background_estimate", "diff", "photometry", "star"):
        return resolve_template_context(ctx)
    return ctx


def stage_snapshot(ctx: StageRunContext, stage: str) -> dict:
    """Stage snapshot.
    
    Parameters
    ----------
    ctx : StageRunContext
    stage : str
    
    Returns
    -------
    dict"""
    from syndiff_pipeline.template_creation.orchestration.stages import resolve_template_context

    spec = _pipeline().require(stage)
    if stage not in ("diff_prep", "background_estimate", "diff", "photometry", "star"):
        ctx = resolve_template_context(ctx)
    if spec.stage_snapshot is not None:
        return spec.stage_snapshot(ctx)
    return {"stage": stage}


def config_fingerprint(ctx: StageRunContext, stage: str) -> str:
    """Config fingerprint.
    
    Parameters
    ----------
    ctx : StageRunContext
    stage : str
    
    Returns
    -------
    str"""
    from syndiff_pipeline.template_creation.orchestration.stages import resolve_template_context

    spec = _pipeline().require(stage)
    if stage not in ("diff_prep", "background_estimate", "diff", "photometry", "star"):
        ctx = resolve_template_context(ctx)
    return spec.config_fingerprint(ctx)


def stage_names() -> tuple[str, ...]:
    """Stage names.
    
    Returns
    -------
    tuple[str, ...]"""
    return _pipeline().stage_names


def status_grid_stages() -> tuple[str, ...]:
    """Stages shown in ``syndiff status`` per-target grids."""
    return STATUS_GRID_STAGES


def canonical_status_grid_stage(stage: str) -> str:
    """Map a SQLite stage name to its status-grid column (if any)."""
    return STATUS_GRID_LEGACY_STAGE_ALIASES.get(stage, stage)


def stage_short_names() -> dict[str, str]:
    """Stage short names without loading diff/star stage modules."""
    global _STAGE_SHORT_NAMES
    if _STAGE_SHORT_NAMES is None:
        from syndiff_pipeline.template_creation.orchestration.stages import TEMPLATE_STAGES

        names = {spec.name: spec.short_name for spec in TEMPLATE_STAGES}
        names.update(_STATIC_STAGE_SHORT_NAMES)
        _STAGE_SHORT_NAMES = names
    return _STAGE_SHORT_NAMES


def resolve_stage_name(name: str) -> str:
    """Resolve stage name.
    
    Parameters
    ----------
    name : str
    
    Returns
    -------
    str"""
    return _pipeline().resolve_stage_name(name)


def __getattr__(name: str) -> Any:
    """Lazy re-exports of composed DAG constants.

    Ignore dunder probes (e.g. ``__path__``) so import machinery does not
    force-load DIFF/STAR stage modules.
    """
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    pipeline = _pipeline()
    if name == "SYNDIFF_PIPELINE":
        return pipeline
    if name == "STAGE_NAMES":
        return pipeline.stage_names
    if name == "STAGE_SHORT_NAMES":
        return stage_short_names()
    if name == "STAGE_DEPS":
        return pipeline.stage_deps()
    if name == "STAGE_POOL":
        return pipeline.stage_pools()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
