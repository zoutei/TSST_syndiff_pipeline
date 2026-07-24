"""Photometry pipeline stage specifications."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.common.orchestration.spec import StageRunContext, StageSpec
from syndiff_pipeline.common.scc_paths import event_scc_leaf
from syndiff_pipeline.photometry.orchestration.verify import (
    collect_photometry_artifacts,
    photometry_complete,
    scc_diff_lane_complete,
)
from syndiff_pipeline.photometry.runner import run_photometry_pipeline
from syndiff_pipeline.photometry.site_config import (
    build_syndiff_config_for_photometry,
    load_photometry_site_policy,
    resolve_photometry_config_path,
    resolve_photometry_run_config,
    write_frozen_photometry_config,
)

log = logging.getLogger(__name__)


def _frozen_photometry_config_path(ctx: StageRunContext) -> Path:
    return logs.run_dir(ctx.runs_root, ctx.run_id) / "photometry_config.yaml"


def _photometry_config_fingerprint(ctx: StageRunContext) -> str:
    path = _frozen_photometry_config_path(ctx)
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _resolve_photometry_run(ctx: StageRunContext):
    phot_config_path = resolve_photometry_config_path(meta=ctx.meta, runner_cfg=ctx.runner_cfg)
    policy = load_photometry_site_policy(phot_config_path)
    site_dir = phot_config_path.parent
    run_config = resolve_photometry_run_config(policy, ctx.target, site_dir=site_dir)
    cfg = build_syndiff_config_for_photometry(
        policy, ctx.target, run_config, site_dir=site_dir
    )
    return policy, run_config, cfg, site_dir


def execute_photometry_stage(ctx: StageRunContext):
    """Execute photometry stage for one target."""
    policy, run_config, cfg, site_dir = _resolve_photometry_run(ctx)
    write_frozen_photometry_config(policy, _frozen_photometry_config_path(ctx))
    phot_root = run_photometry_pipeline(
        cfg,
        ctx.target,
        site_dir,
        run_config=run_config,
        policy=policy,
        force_rerun=ctx.force_rerun,
        phot_log_path=ctx.progress_path,
    )
    artifacts = collect_photometry_artifacts(run_config, cfg.output_dir, run_config.photometry_run_id)
    if not artifacts:
        artifacts = [str(phot_root)]
    expected = max(len(artifacts), 1)
    produced = len(artifacts)
    return expected, produced, artifacts


def _verify_photometry(ctx: StageRunContext) -> bool:
    try:
        policy, run_config, cfg, _site_dir = _resolve_photometry_run(ctx)
        if not scc_diff_lane_complete(
            run_config,
            data_root=cfg.data_root,
            sector=int(cfg.sector),
            camera=int(cfg.camera),
            ccd=int(cfg.ccd),
        ):
            return False
        return photometry_complete(
            run_config,
            cfg.output_dir,
            run_config.photometry_run_id,
        )
    except (KeyError, ValueError, FileNotFoundError):
        return False


def _collect_photometry_artifacts(ctx: StageRunContext) -> tuple[int, int, list[str]]:
    _policy, run_config, cfg, _site_dir = _resolve_photometry_run(ctx)
    artifacts = collect_photometry_artifacts(
        run_config, cfg.output_dir, run_config.photometry_run_id
    )
    expected = max(len(artifacts), 1)
    produced = len(artifacts)
    return expected, produced, artifacts


def _photometry_condor_resources(cfg):
    from syndiff_pipeline.common.orchestration import condor
    from syndiff_pipeline.photometry.site_config import (
        PhotometryCondorConfig,
        load_photometry_site_policy,
    )

    phot_path = str(getattr(cfg, "photometry_config_path", "") or "").strip()
    if phot_path:
        policy = load_photometry_site_policy(phot_path)
        c = policy.condor
    else:
        c = PhotometryCondorConfig()
    return condor.CondorResourceRequest(
        request_cpus=c.request_cpus,
        request_memory_mb=c.request_memory,
        host_stats_min_mem_mb=c.host_stats_min_mem_mb,
        host_stats_max_load15=c.host_stats_max_load15,
    )


def _photometry_stage_snapshot(ctx: StageRunContext) -> dict:
    event_dir = event_scc_leaf(
        ctx.runner_cfg.workspace_root,
        ctx.target.event_name(),
        ctx.target.sector,
        ctx.target.camera,
        ctx.target.ccd,
    )
    return {
        "sector": ctx.target.sector,
        "camera": ctx.target.camera,
        "ccd": ctx.target.ccd,
        "target_name": ctx.target.target_name,
        "target_ra": ctx.target.target_ra,
        "target_dec": ctx.target.target_dec,
        "event_dir": str(event_dir),
        "stage": "photometry",
        "pool": "photometry",
    }


def write_photometry_manifest(
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
        "stage": "photometry",
        "expected_count": int(expected_count),
        "produced_count": int(produced_count),
        "artifacts": [str(p) for p in (artifacts or [])],
        "config_fingerprint": _photometry_config_fingerprint(ctx),
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


PHOTOMETRY_STAGE = StageSpec(
    name="photometry",
    short_name="phot",
    deps=(),
    pool="photometry",
    default_executor="condor",
    execute=execute_photometry_stage,
    verify_complete=_verify_photometry,
    collect_artifacts=_collect_photometry_artifacts,
    config_fingerprint=_photometry_config_fingerprint,
    condor_resources=_photometry_condor_resources,
    stage_snapshot=_photometry_stage_snapshot,
)

PHOTOMETRY_STAGES: tuple[StageSpec, ...] = (PHOTOMETRY_STAGE,)
