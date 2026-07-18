"""Event bind stage: WCS grouping handoff into nested event/SCC workspace leaves."""

from __future__ import annotations

import logging
from pathlib import Path

from syndiff_pipeline.common.orchestration.spec import StageRunContext
from syndiff_pipeline.common.scc_paths import event_scc_leaf
from syndiff_pipeline.common.wcs_grouping import (
    EVENT_JOB_FILENAME,
    FRAMES_CSV_BASENAME,
    _event_job_path,
    _frames_csv_path,
)
from syndiff_pipeline.template_creation.orchestration.handoff import run_wcs_grouping
from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config
from syndiff_pipeline.template_creation.orchestration.stages import resolve_template_context

log = logging.getLogger(__name__)

SYNDIFF_FFI_FRAMES_BASENAME = FRAMES_CSV_BASENAME


def event_leaf_dir(ctx: StageRunContext) -> Path:
    """Nested workspace leaf ``events/{event}/s{SSSS}_c{C}_k{K}/``."""
    t = ctx.target
    return event_scc_leaf(
        ctx.runner_cfg.workspace_root,
        t.event_name(),
        t.sector,
        t.camera,
        t.ccd,
    )


def execute_bind_stage(ctx: StageRunContext) -> None:
    """Run event WCS grouping and write handoff JSON + frames CSV."""
    ctx = resolve_template_context(ctx)
    resolved = ctx.template_resolved
    leaf = event_leaf_dir(ctx)
    leaf.mkdir(parents=True, exist_ok=True)
    # Handoff writer uses resolved.event_dir; point it at the nested leaf.
    resolved.event_dir = str(leaf)
    run_wcs_grouping(resolved)
    log.info("Bind handoff written under %s", leaf)


def verify_bind_complete(ctx: StageRunContext) -> bool:
    """True when cluster job JSON and frames CSV exist in the event leaf."""
    leaf = event_leaf_dir(ctx)
    return Path(_event_job_path(leaf)).is_file() and Path(_frames_csv_path(leaf)).is_file()
