"""Load photometry site policy and merged per-run configuration."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from syndiff_pipeline.common.orchestration.condor import parse_condor_policy_block
from syndiff_pipeline.common.orchestration.deployment import (
    deployment_path_for_config,
    load_deployment,
    require_deployment_path,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.scc_paths import (
    event_scc_leaf,
    normalize_store_name,
    scc_catalogs_dir,
    scc_ffi_dir,
    scc_templates_dir,
)
from syndiff_pipeline.difference_imaging.orchestration.config import (
    SynDiffConfig,
    absolutize_config,
    normalize_additional_forced_targets,
)

log = logging.getLogger(__name__)


@dataclass
class PhotometryCondorConfig:
    """HTCondor resource request for the photometry stage."""

    request_cpus: int = 16
    request_memory: int = 100_000
    host_stats_min_mem_mb: int = 128_000
    host_stats_max_load15: float = 10.0


@dataclass
class PhotometrySitePolicy:
    """Site-level photometry policy from ``photometry_config.yaml``."""

    deployment_file: str = "deployment.yaml"
    defaults: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    pipeline: list = field(default_factory=list)
    additional_forced_targets: list = field(default_factory=list)
    per_event_force_targets: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    condor: PhotometryCondorConfig = field(default_factory=PhotometryCondorConfig)
    config_path: str = ""


@dataclass
class PhotometryRunConfig:
    """Merged knobs for one target photometry run."""

    photometry_run_id: str | None = None
    n_jobs: int = 16
    pipeline_plots: bool = False
    pipeline_plots_dir: str = "debug_plots"
    pipeline_plot_dpi: int = 150
    max_ffis: int | None = None
    pipeline: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)
    output_store_name: str | None = None
    template_store_name: str | None = None
    remap_store_name: str | None = None
    oversampling_factor: int = 1
    diffs_label: str = "hp_d"
    epsf_label: str | None = None
    additional_forced_targets: list = field(default_factory=list)


def _parse_condor(raw: dict | None) -> PhotometryCondorConfig:
    cpus, mem, min_mem, max_load15 = parse_condor_policy_block(
        raw,
        context="photometry_config.yaml condor",
        default_cpus=16,
        default_memory=100_000,
    )
    return PhotometryCondorConfig(
        request_cpus=cpus,
        request_memory=mem,
        host_stats_min_mem_mb=min_mem,
        host_stats_max_load15=max_load15,
    )


def _parse_per_event_force_targets(raw: Any) -> dict[str, list]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("per_event_force_targets must be a mapping of event label → target list")
    out: dict[str, list] = {}
    for key, val in raw.items():
        label = str(key).strip()
        if not label:
            raise ValueError("per_event_force_targets keys must be non-empty event labels")
        if not isinstance(val, list):
            raise ValueError(
                f"per_event_force_targets[{label!r}] must be a list of target mappings"
            )
        out[label] = copy.deepcopy(val)
    return out


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _target_override(policy: PhotometrySitePolicy, target: Target) -> dict:
    for key in (target.scc_key(), f"{target.sector}/{target.camera}/{target.ccd}"):
        if key in policy.overrides:
            return policy.overrides[key]
    return {}


def _per_event_force_targets_for_target(
    policy: PhotometrySitePolicy, target: Target
) -> list:
    by_label = policy.per_event_force_targets.get(target.label())
    if by_label is not None:
        return list(by_label)
    by_name = policy.per_event_force_targets.get(target.target_name)
    if by_name is not None:
        return list(by_name)
    return []


def load_photometry_site_policy(
    path: str | Path, *, config_path: str | Path | None = None
) -> PhotometrySitePolicy:
    """Load ``photometry_config.yaml`` site policy.

    Parameters
    ----------
    path : str | Path
        Read policy *content* (defaults/paths/pipeline/overrides/etc.) from
        here.
    config_path : str | Path | None, optional
        What to record as the returned policy's ``config_path`` field --
        which ``deployment.yaml``/relative-path resolution keys off (see
        :func:`build_syndiff_config_for_photometry`). Defaults to *path*
        (unchanged pre-existing behaviour) when omitted. Pass this
        explicitly when *path* is a frozen run-directory snapshot, so
        ``config_path`` stays anchored at the live site directory instead of
        a run directory with no ``deployment.yaml`` of its own.
    """
    resolved_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Photometry site config must be a YAML mapping: {resolved_path}")
    pipeline = raw.get("pipeline")
    if pipeline is None or not isinstance(pipeline, list):
        raise ValueError(f"photometry_config.yaml requires a pipeline list: {resolved_path}")
    resolved_config_path = (
        Path(config_path).expanduser().resolve() if config_path is not None else resolved_path
    )
    return PhotometrySitePolicy(
        deployment_file=str(raw.get("deployment_file", "deployment.yaml")).strip()
        or "deployment.yaml",
        defaults=dict(raw.get("defaults") or {}),
        paths=dict(raw.get("paths") or {}),
        pipeline=copy.deepcopy(pipeline),
        additional_forced_targets=copy.deepcopy(raw.get("additional_forced_targets") or []),
        per_event_force_targets=_parse_per_event_force_targets(
            raw.get("per_event_force_targets")
        ),
        overrides=dict(raw.get("overrides") or {}),
        condor=_parse_condor(raw.get("condor")),
        config_path=str(resolved_config_path),
    )


def resolve_photometry_run_config(
    policy: PhotometrySitePolicy,
    target: Target,
    *,
    site_dir: str | Path,
) -> PhotometryRunConfig:
    """Merge policy defaults, SCC overrides, and target-specific extras."""
    site = Path(site_dir).expanduser().resolve()
    override = _target_override(policy, target)
    merged_defaults = _deep_merge_dict(policy.defaults, override.get("defaults", {}))
    merged_paths = _deep_merge_dict(policy.paths, override.get("paths", {}))

    photometry_run_id = merged_defaults.get("photometry_run_id")
    if photometry_run_id is not None:
        photometry_run_id = str(photometry_run_id).strip() or None

    max_ffis_raw = merged_defaults.get("max_ffis")
    max_ffis = int(max_ffis_raw) if max_ffis_raw not in (None, "") else None

    inputs = merged_paths.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise ValueError("photometry paths.inputs must be a mapping")
    diffs_label = str(inputs.get("diffs", "hp_d")).strip() or "hp_d"
    epsf_raw = inputs.get("epsf")
    epsf_label = str(epsf_raw).strip() if epsf_raw not in (None, "") else None

    pipeline = copy.deepcopy(policy.pipeline)
    if override.get("pipeline"):
        pipeline = copy.deepcopy(override["pipeline"])

    per_event = _per_event_force_targets_for_target(policy, target)
    combined_forced = list(policy.additional_forced_targets) + per_event
    additional_forced_targets = normalize_additional_forced_targets(combined_forced)

    return PhotometryRunConfig(
        photometry_run_id=photometry_run_id,
        n_jobs=int(merged_defaults.get("n_jobs", 16)),
        pipeline_plots=bool(merged_defaults.get("pipeline_plots", False)),
        pipeline_plots_dir=str(merged_defaults.get("pipeline_plots_dir", "debug_plots")),
        pipeline_plot_dpi=int(merged_defaults.get("pipeline_plot_dpi", 150) or 150),
        max_ffis=max_ffis,
        pipeline=pipeline,
        paths=dict(merged_paths),
        output_store_name=normalize_store_name(merged_paths.get("output_store_name")),
        template_store_name=normalize_store_name(merged_paths.get("template_store_name")),
        remap_store_name=normalize_store_name(merged_paths.get("remap_store_name")),
        oversampling_factor=max(1, int(merged_defaults.get("oversampling_factor", 1) or 1)),
        diffs_label=diffs_label,
        epsf_label=epsf_label,
        additional_forced_targets=additional_forced_targets,
    )


def resolve_photometry_deployment(policy: PhotometrySitePolicy) -> tuple[dict, Path]:
    """Load ``deployment.yaml`` for a photometry site policy.

    Uses the generic (non-diff-specific) deployment loader directly.
    ``photometry_config.yaml`` happens to share the diff policy's
    ``deployment_file``/``pipeline``/``paths`` shape, but photometry must not
    depend on any diff-specific site_config helper for that reason alone --
    it should stay independent of how the diff side happens to be shaped.

    Returns
    -------
    tuple[dict, Path]
        ``(deployment, deploy_path)`` -- the parsed ``deployment.yaml``
        mapping and the path it was loaded from
        (``{policy.config_path parent}/{policy.deployment_file}``).
    """
    deploy_path = deployment_path_for_config(policy.config_path, policy.deployment_file)
    return load_deployment(policy.config_path, policy.deployment_file), deploy_path


def build_syndiff_config_for_photometry(
    policy: PhotometrySitePolicy,
    target: Target,
    run_config: PhotometryRunConfig,
    *,
    site_dir: str | Path,
) -> SynDiffConfig:
    """Build a :class:`SynDiffConfig` for photometry stage execution."""
    site = Path(site_dir).expanduser().resolve()
    deploy_path = deployment_path_for_config(site / "pipeline.yaml", policy.deployment_file)
    if not deploy_path.is_file():
        deploy_path = deployment_path_for_config(policy.config_path, policy.deployment_file)
    deployment = load_deployment(policy.config_path, policy.deployment_file)
    workspace_root = require_deployment_path(
        deployment, "workspace_root", deployment_path=deploy_path
    )
    data_root = require_deployment_path(deployment, "data_root", deployment_path=deploy_path)
    workspace_root = str(Path(workspace_root).expanduser().resolve())
    data_root_path = Path(data_root).expanduser().resolve()

    event_dir = event_scc_leaf(
        workspace_root,
        target.event_name(),
        target.sector,
        target.camera,
        target.ccd,
    )
    ffi_dir = str(scc_ffi_dir(data_root_path, target.sector, target.camera, target.ccd))
    os_factor = run_config.oversampling_factor
    template_dir = str(
        scc_templates_dir(
            data_root_path,
            target.sector,
            target.camera,
            target.ccd,
            oversampling_factor=os_factor,
            store_name=run_config.template_store_name,
        )
    )
    gaia_catalog = str(
        scc_catalogs_dir(data_root_path, target.sector, target.camera, target.ccd)
        / f"gaia_catalog_s{target.sector:04d}_{target.camera}_{target.ccd}.csv"
    )

    cfg = SynDiffConfig(
        ffi_dir=ffi_dir,
        output_dir=str(event_dir),
        gaia_catalog=gaia_catalog,
        template_dir=template_dir,
        pipeline=copy.deepcopy(run_config.pipeline),
        target_ra=target.target_ra,
        target_dec=target.target_dec,
        target_name=target.target_name,
        sector=target.sector,
        camera=target.camera,
        ccd=target.ccd,
        data_root=str(data_root_path),
        site_config_dir=str(site),
        oversampling_factor=os_factor,
        template_store_name=run_config.template_store_name,
        output_store_name=run_config.output_store_name,
        remap_store_name=run_config.remap_store_name,
        n_jobs=run_config.n_jobs,
        pipeline_plots=run_config.pipeline_plots,
        pipeline_plots_dir=run_config.pipeline_plots_dir,
        pipeline_plot_dpi=run_config.pipeline_plot_dpi,
        max_ffis=run_config.max_ffis,
        additional_forced_targets=copy.deepcopy(run_config.additional_forced_targets),
    )
    return absolutize_config(cfg, site)


def write_frozen_photometry_config(policy: PhotometrySitePolicy, dest: str | Path) -> Path:
    """Write merged site policy YAML to a frozen run path."""
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "deployment_file": policy.deployment_file,
        "defaults": policy.defaults,
        "paths": policy.paths,
        "pipeline": policy.pipeline,
        "additional_forced_targets": policy.additional_forced_targets,
        "per_event_force_targets": policy.per_event_force_targets,
        "overrides": policy.overrides,
        "condor": {
            "request_cpus": policy.condor.request_cpus,
            "request_memory": policy.condor.request_memory,
            "host_stats_min_mem_mb": policy.condor.host_stats_min_mem_mb,
            "host_stats_max_load15": policy.condor.host_stats_max_load15,
        },
    }
    dest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dest_path


def resolve_photometry_config_path(*, meta: dict | None, runner_cfg) -> Path:
    """Resolve the photometry policy *content* path from run metadata.

    The frozen ``runs/{run_id}/photometry_config.yaml`` snapshot -- recorded
    as ``photometry_config_path`` in run_meta / on ``RunnerConfig`` -- is the
    sole source: submit always freezes it (see ``_prepare_run_directory``),
    so a submitted run's frozen config is authoritative (matching the "check
    the frozen copies... when debugging" invariant).

    This is the policy *content* path only. Do not derive a site directory
    from it (e.g. via ``.parent``) for ``deployment.yaml``/relative-path
    resolution -- the frozen copy lives in a run directory with no
    ``deployment.yaml`` of its own. Use :func:`resolve_photometry_site_dir`
    for that, and pass it as ``load_photometry_site_policy(...,
    config_path=...)`` so the returned policy's ``config_path`` field (which
    deployment resolution keys off) stays site-anchored.
    """
    meta = meta or {}
    raw = meta.get("photometry_config_path") or getattr(
        runner_cfg, "photometry_config_path", ""
    )
    if raw:
        return Path(str(raw)).expanduser().resolve()
    raise ValueError(
        "Photometry stage requires photometry_config_path in run_meta or on "
        "RunnerConfig"
    )


def resolve_photometry_site_dir(*, meta: dict | None, runner_cfg) -> Path:
    """Resolve the photometry site's authoring directory from run metadata.

    Always the directory of the live/site ``photometry_config.yaml`` (never
    a frozen run directory), for ``deployment.yaml`` and any other path
    recorded relative to the site config. Falls back to
    :func:`resolve_photometry_config_path`'s result when no
    ``source_photometry_config_path`` is recorded (e.g. an ad hoc run with
    incomplete run_meta) -- matching that function's own pre-fix
    "site wins" behaviour for this edge case.
    """
    meta = meta or {}
    source_raw = str(meta.get("source_photometry_config_path") or "").strip()
    if source_raw:
        return Path(source_raw).expanduser().resolve().parent
    return resolve_photometry_config_path(meta=meta, runner_cfg=runner_cfg).parent
