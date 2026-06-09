"""Resolve frozen run-local config and targets from a run directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from syndiff_pipeline.template_runner import logs
from syndiff_pipeline.template_runner.runner_config import RunnerConfig, load_runner_config
from syndiff_pipeline.template_runner.targets import Target, load_targets


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    cfg: RunnerConfig
    targets: List[Target]
    meta: dict


def _load_meta(run_directory: Path) -> dict:
    meta_path = logs.run_meta_path(run_directory)
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _validate_frozen_inputs(run_directory: Path) -> None:
    cfg_path = logs.run_config_path(run_directory)
    targets_path = logs.run_targets_path(run_directory)
    if not cfg_path.is_file() or not targets_path.is_file():
        raise SystemExit(
            f"Run directory {run_directory} is missing frozen config.yaml or targets.csv. "
            "Runs created before frozen snapshots require manual copy or a new submit."
        )


def resolve_run_context(
    *,
    run_dir: str | Path | None = None,
    run_id: str | None = None,
    runs_root: str | None = None,
) -> RunContext:
    if run_dir is not None:
        rd = Path(run_dir).expanduser().resolve()
    elif run_id and runs_root:
        rd = logs.run_dir(runs_root, run_id)
    else:
        raise SystemExit(
            "Specify --run-dir, or --config with --run-id."
        )

    if not rd.is_dir():
        raise SystemExit(f"Run directory not found: {rd}")

    _validate_frozen_inputs(rd)
    meta = _load_meta(rd)
    resolved_run_id = meta.get("run_id") or run_id or rd.name

    cfg = load_runner_config(logs.run_config_path(rd))
    targets = load_targets(logs.run_targets_path(rd))
    return RunContext(
        run_id=resolved_run_id,
        run_dir=rd,
        cfg=cfg,
        targets=targets,
        meta=meta,
    )
