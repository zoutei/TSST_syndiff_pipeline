"""Template pipeline stage specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syndiff_pipeline.common.orchestration.spec import StageRunContext, StageSpec

if TYPE_CHECKING:
    from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig


def _ps1_process_effective_deps(stages) -> tuple[str, ...]:
    """Ps1 process effective deps.
    
    Parameters
    ----------
    stages
    
    Returns
    -------
    tuple[str, ...]"""
    if getattr(getattr(stages, "ps1_process", None), "ps1_source", "zarr") == "stream":
        return ("mapping",)
    return ("ps1_download",)


def _condor_resources_for_mapping(cfg):
    """Condor resources for mapping.
    
    Parameters
    ----------
    cfg"""
    from syndiff_pipeline.common.orchestration import condor

    params = cfg.stages.mapping
    return condor.CondorResourceRequest(
        request_cpus=params.condor_request_cpus,
        request_memory_mb=params.condor_request_memory,
        requirements=params.condor_requirements,
        rank=params.condor_rank,
    )


def _condor_resources_for_ps1_process(cfg):
    """Condor resources for ps1 process.
    
    Parameters
    ----------
    cfg"""
    from syndiff_pipeline.common.orchestration import condor

    params = cfg.stages.ps1_process
    return condor.CondorResourceRequest(
        request_cpus=params.condor_request_cpus,
        request_memory_mb=params.condor_request_memory,
        requirements=params.condor_requirements,
        rank=params.condor_rank,
    )


def _downsample_effective_deps(stages) -> tuple[str, ...]:
    """Downsample deps: field mode requires remap; linear omits it."""
    base = ("mapping", "ps1_process")
    ds = getattr(stages, "downsample", None)
    wg = getattr(stages, "wcs_grouping", None)
    mode = str(
        getattr(ds, "geometry_mode", None)
        or getattr(wg, "geometry_mode", None)
        or "linear"
    ).lower()
    if mode == "field":
        return base + ("remap",)
    return base


def _condor_resources_for_remap(cfg):
    """Condor resources for remap."""
    from syndiff_pipeline.common.orchestration import condor

    params = cfg.stages.remap
    return condor.CondorResourceRequest(
        request_cpus=params.condor_request_cpus,
        request_memory_mb=params.condor_request_memory,
        requirements=params.condor_requirements,
        rank=params.condor_rank,
    )


def _condor_resources_for_downsample(cfg):
    """Condor resources for downsample."""
    from syndiff_pipeline.common.orchestration import condor

    params = cfg.stages.downsample
    disk_mb = getattr(params, "condor_request_disk", None)
    request_disk_kb = None if disk_mb is None else int(disk_mb) * 1024
    return condor.CondorResourceRequest(
        request_cpus=params.condor_request_cpus,
        request_memory_mb=params.condor_request_memory,
        request_disk_kb=request_disk_kb,
        requirements=params.condor_requirements,
        rank=params.condor_rank,
    )


def _make_template_stage(
    name: str,
    short_name: str,
    deps: tuple[str, ...],
    *,
    pool: str | None = None,
    default_executor: str = "local",
    effective_deps=None,
    condor_resources=None,
) -> StageSpec:
    """Make template stage.
    
    Parameters
    ----------
    name : str
    short_name : str
    deps : tuple[str, ...]
    pool : str | None, optional, default ``None``
    default_executor : str, optional, default ``'local'``
    effective_deps, optional, default ``None``
    condor_resources, optional, default ``None``
    
    Returns
    -------
    StageSpec"""
    def execute(
        resolved: ResolvedTargetConfig,
        *,
        force_rerun: bool = False,
        progress_path: str | None = None,
    ):
        """Execute.
        
        Parameters
        ----------
        resolved : ResolvedTargetConfig
        force_rerun : bool, optional, default ``False``
        progress_path : str | None, optional, default ``None``"""
        from syndiff_pipeline.template_creation.orchestration import dispatch as dispatch_impl

        return dispatch_impl._execute_template_stage(
            resolved,
            name,
            force_rerun=force_rerun,
            progress_path=progress_path,
        )

    def verify_complete(resolved: ResolvedTargetConfig) -> bool:
        """Verify complete.
        
        Parameters
        ----------
        resolved : ResolvedTargetConfig
        
        Returns
        -------
        bool"""
        from syndiff_pipeline.template_creation.orchestration.verify import (
            stage_complete as _template_stage_complete,
        )

        return _template_stage_complete(resolved, name)

    def collect_artifacts(resolved: ResolvedTargetConfig):
        """Collect artifacts.
        
        Parameters
        ----------
        resolved : ResolvedTargetConfig"""
        from syndiff_pipeline.template_creation.orchestration.verify import (
            collect_stage_artifacts as _collect_template_artifacts,
        )

        return _collect_template_artifacts(resolved, name)

    def config_fingerprint(resolved: ResolvedTargetConfig) -> str:
        """Config fingerprint.
        
        Parameters
        ----------
        resolved : ResolvedTargetConfig
        
        Returns
        -------
        str"""
        from syndiff_pipeline.template_creation.orchestration.verify import (
            config_fingerprint as _template_config_fingerprint,
        )

        return _template_config_fingerprint(resolved, name)

    def stage_snapshot(resolved: ResolvedTargetConfig) -> dict:
        """Stage snapshot.
        
        Parameters
        ----------
        resolved : ResolvedTargetConfig
        
        Returns
        -------
        dict"""
        from syndiff_pipeline.template_creation.orchestration.runner_config import config_snapshot

        snap = config_snapshot(resolved)
        snap["stage"] = name
        snap["pool"] = pool
        return snap

    return StageSpec(
        name=name,
        short_name=short_name,
        deps=deps,
        pool=pool,
        default_executor=default_executor,
        effective_deps=effective_deps,
        execute=execute,
        verify_complete=verify_complete,
        collect_artifacts=collect_artifacts,
        config_fingerprint=config_fingerprint,
        condor_resources=condor_resources,
        stage_snapshot=stage_snapshot,
    )


TEMPLATE_STAGES: tuple[StageSpec, ...] = (
    _make_template_stage("tess_ffi_download", "tess_dl", (), pool="network"),
    _make_template_stage(
        "mapping",
        "map",
        ("tess_ffi_download",),
        pool="mapping",
        default_executor="condor",
        condor_resources=_condor_resources_for_mapping,
    ),
    _make_template_stage("ps1_download", "ps1_dl", ("mapping",), pool="network"),
    _make_template_stage(
        "ps1_process",
        "ps1_pr",
        ("ps1_download",),
        pool="ps1_process",
        default_executor="condor",
        effective_deps=_ps1_process_effective_deps,
        condor_resources=_condor_resources_for_ps1_process,
    ),
    _make_template_stage(
        "remap",
        "rmap",
        ("mapping",),
        pool="remap",
        default_executor="condor",
        condor_resources=_condor_resources_for_remap,
    ),
    _make_template_stage(
        "downsample",
        "down",
        ("mapping", "ps1_process", "remap"),
        pool="downsample",
        effective_deps=_downsample_effective_deps,
        condor_resources=_condor_resources_for_downsample,
    ),
)


def resolve_template_context(ctx: StageRunContext) -> StageRunContext:
    """Resolve template context.
    
    Parameters
    ----------
    ctx : StageRunContext
    
    Returns
    -------
    StageRunContext"""
    from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config

    if ctx.template_resolved is not None:
        return ctx
    resolved = resolve_config(ctx.target, ctx.runner_cfg)
    ctx.template_resolved = resolved
    return ctx
