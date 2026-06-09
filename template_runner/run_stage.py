"""Subprocess entry point for a single template pipeline stage.

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

from syndiff_pipeline.template_runner import logs, stages
from syndiff_pipeline.template_runner.run_context import resolve_run_context
from syndiff_pipeline.template_runner.runner_config import resolve_config
from syndiff_pipeline.template.downsample_progress import progress_path_for_log
from syndiff_pipeline.template_runner.verify import collect_stage_artifacts, write_manifest

log = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, force=True)


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
    parser = argparse.ArgumentParser(description="Run one template pipeline stage")
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

    ctx = resolve_run_context(run_dir=args.run_dir, run_id=args.run_id)
    cfg = ctx.cfg
    target = next(t for t in ctx.targets if t.label() == args.target_label)
    resolved = resolve_config(target, cfg)
    snap = stages.stage_snapshot(resolved, args.stage)
    runs_root = cfg.runs_dir()

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

    exit_code = 0
    try:
        with logs.stage_log(runs_root, args.run_id, args.target_label, args.stage, snap) as tee:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = tee  # type: ignore[assignment]
            sys.stderr = tee  # type: ignore[assignment]
            try:
                _configure_logging()
                manifest = stages.execute_stage(
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
                    manifest = collect_stage_artifacts(resolved, args.stage)
                expected_count, produced_count, artifacts = manifest
                for manifest_dest in (
                    logs.stage_manifest_path(
                        runs_root, args.run_id, args.target_label, args.stage
                    ),
                    logs.stable_stage_manifest_path(
                        runs_root, args.target_label, args.stage
                    ),
                ):
                    write_manifest(
                        manifest_dest,
                        resolved,
                        args.stage,
                        artifacts,
                        expected_count,
                        produced_count,
                    )
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
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
