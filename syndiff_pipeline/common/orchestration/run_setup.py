"""Post-create run materialization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from syndiff_pipeline.common.orchestration.state import PipelineState


@dataclass(frozen=True)
class PostCreateRunSetupResult:
    stream_skipped: int
    not_selected: int
    superseded: int


def apply_post_create_run_setup(
    state: PipelineState,
    run_id: str,
    targets: Sequence,
    cfg,
    active_stages: Sequence[str],
) -> PostCreateRunSetupResult:
    """Apply skip rules immediately after ``create_run``.

    1. Stream mode → skip ``ps1_download`` (download happens in ``ps1_process``).
    2. Stages outside the selected closure → ``not_selected`` skip.
    3. Upstream artifact checks redundant with downstream work → ``superseded`` skip.
    """
    stream_skipped = 0
    if cfg.stages.ps1_process.ps1_source == "stream":
        stream_skipped = state.apply_ps1_stream_download_skips(run_id, targets, cfg)
    not_selected = state.apply_not_selected_skips(run_id, targets, cfg)
    superseded = state.apply_superseded_skips(run_id, targets)
    return PostCreateRunSetupResult(
        stream_skipped=stream_skipped,
        not_selected=not_selected,
        superseded=superseded,
    )
