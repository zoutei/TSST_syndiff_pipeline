"""Subprocess entry point for a single pipeline stage.

Writes a durable, atomically-updated per-stage status file
(``per_target/<label>/<stage>.status.json``) at start and on exit so the
supervisor daemon can recover the real outcome across its own restarts without
relying on an in-memory ``Popen`` handle.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.template_creation.orchestration import dispatch
from syndiff_pipeline.common.orchestration.run_context import resolve_run_context
from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config
from syndiff_pipeline.template_creation.processing.downsample_progress import progress_path_for_log
from syndiff_pipeline.template_creation.orchestration.verify import collect_stage_artifacts, write_manifest
from syndiff_pipeline.pipeline_spec import build_stage_context
from syndiff_pipeline.difference_imaging.orchestration.stages import (
    DIFF_STAGE,
    write_diff_manifest,
)

log = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, force=True)


def _lookup_target(ctx, target_label: str):
    for target in ctx.targets:
        if target.label() == target_label:
            return target
    available = ", ".join(sorted(t.label() for t in ctx.targets)) or "(none)"
    raise ValueError(
        f"Target label {target_label!r} not found in run targets; available: {available}"
    )


def _write_status(
    runs_root: str,
    run_id: str,
    target_label: str,
    stage: str,
    *,
    launch_token: str,
    state: str,
    started_at: str,
    exit_code: int | None = None,
    finished_at: str | None = None,
) -> None:
    """Atomically (tmp+rename) write the durable per-stage status file."""
    payload = {
        "launch_token": launch_token,
        "pid": os.getpid(),
        "state": state,  # "running" | "exited"
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "updated_at": logs._utc_now_iso(),
    }
    logs.write_json_atomic(
        logs.stage_status_path(runs_root, run_id, target_label, stage),
        payload,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one pipeline stage")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-dir", required=True, help="Path to run directory with frozen config")
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--launch-token", required=True)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run stage even when output artifacts already exist",
    )
    args = parser.parse_args(argv)

    runs_root = str(Path(args.run_dir).resolve().parent)
    started_at = logs._utc_now_iso()
    _write_status(
        runs_root,
        args.run_id,
        args.target_label,
        args.stage,
        launch_token=args.launch_token,
        state="running",
        started_at=started_at,
    )

    resolved = None
    stage_ctx = None
    snap: dict
    try:
        ctx = resolve_run_context(run_dir=args.run_dir, run_id=args.run_id)
        cfg = ctx.cfg
        target = _lookup_target(ctx, args.target_label)
        source_config = (ctx.meta or {}).get("source_config_path") or str(
            logs.run_config_path(ctx.run_dir)
        )
        if args.stage == "diff":
            diff_log_path = logs.target_log_path(
                runs_root, args.run_id, args.target_label, args.stage
            )
            stage_ctx = build_stage_context(
                run_id=args.run_id,
                runs_root=runs_root,
                target_label=args.target_label,
                target=target,
                runner_cfg=cfg,
                stage=args.stage,
                meta=dict(ctx.meta or {}),
                force_rerun=args.force_rerun,
                progress_path=str(diff_log_path),
            )
            snap = DIFF_STAGE.stage_snapshot(stage_ctx) if DIFF_STAGE.stage_snapshot else {}
        else:
            resolved = resolve_config(target, cfg, config_path=source_config)
            snap = dispatch.stage_snapshot(resolved, args.stage)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        log_path = logs.target_log_path(
            runs_root, args.run_id, args.target_label, args.stage
        )
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"Stage setup failed: {exc}\n")
        except OSError:
            pass
        log.error("Stage setup failed: %s", exc)
        _write_status(
            runs_root,
            args.run_id,
            args.target_label,
            args.stage,
            launch_token=args.launch_token,
            state="exited",
            started_at=started_at,
            exit_code=exit_code,
            finished_at=logs._utc_now_iso(),
        )
        return exit_code
    except Exception as exc:
        exit_code = 1
        log_path = logs.target_log_path(
            runs_root, args.run_id, args.target_label, args.stage
        )
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(traceback.format_exc())
        except OSError:
            pass
        log.exception("Stage setup failed: %s", exc)
        _write_status(
            runs_root,
            args.run_id,
            args.target_label,
            args.stage,
            launch_token=args.launch_token,
            state="exited",
            started_at=started_at,
            exit_code=exit_code,
            finished_at=logs._utc_now_iso(),
        )
        return exit_code

    exit_code = 0
    try:
        with logs.stage_log(runs_root, args.run_id, args.target_label, args.stage, snap):
            _configure_logging()
            if args.stage == "diff":
                assert stage_ctx is not None
                manifest = DIFF_STAGE.execute(stage_ctx)
            else:
                assert resolved is not None
                manifest = dispatch.execute_stage(
                    resolved,
                    args.stage,
                    force_rerun=args.force_rerun,
                    progress_path=str(
                        progress_path_for_log(
                            logs.target_log_path(
                                runs_root, args.run_id, args.target_label, args.stage
                            )
                        )
                    )
                    if args.stage == "downsample"
                    else None,
                )
            if manifest is None:
                if args.stage == "diff":
                    assert stage_ctx is not None
                    manifest = DIFF_STAGE.collect_artifacts(stage_ctx)
                else:
                    assert resolved is not None
                    manifest = collect_stage_artifacts(resolved, args.stage)
            if len(manifest) == 4:
                expected_count, produced_count, artifacts, manifest_meta = manifest
            else:
                expected_count, produced_count, artifacts = manifest
                manifest_meta = None
            manifest_paths = (
                logs.stage_manifest_path(
                    runs_root, args.run_id, args.target_label, args.stage
                ),
                logs.stable_stage_manifest_path(
                    runs_root, args.target_label, args.stage
                ),
            )
            if args.stage == "diff":
                assert stage_ctx is not None
                for manifest_dest in manifest_paths:
                    write_diff_manifest(
                        manifest_dest,
                        stage_ctx,
                        artifacts,
                        expected_count,
                        produced_count,
                    )
            else:
                assert resolved is not None
                for manifest_dest in manifest_paths:
                    write_manifest(
                        manifest_dest,
                        resolved,
                        args.stage,
                        artifacts,
                        expected_count,
                        produced_count,
                        meta=manifest_meta,
                    )
    except SystemExit as exc:
        if isinstance(exc.code, int):
            exit_code = exc.code
        elif exc.code is None:
            exit_code = 0
        else:
            exit_code = 1
    except Exception as exc:
        exit_code = 1
        log.exception("Stage failed: %s", exc)

    _write_status(
        runs_root,
        args.run_id,
        args.target_label,
        args.stage,
        launch_token=args.launch_token,
        state="exited",
        started_at=started_at,
        exit_code=exit_code,
        finished_at=logs._utc_now_iso(),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
