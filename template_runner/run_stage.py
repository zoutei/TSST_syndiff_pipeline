"""Subprocess entry point for a single template pipeline stage."""

from __future__ import annotations

import argparse
import logging
import sys

from syndiff_pipeline.template_runner import logs, stages
from syndiff_pipeline.template_runner.run_context import resolve_run_context
from syndiff_pipeline.template_runner.runner_config import resolve_config

log = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _configure_logging() -> None:
    """Attach root logger to current sys.stderr (must run after stdout/stderr redirect)."""
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, force=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one template pipeline stage")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-dir", required=True, help="Path to run directory with frozen config")
    parser.add_argument("--target-label", required=True)
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

    with logs.stage_log(cfg.runs_dir(), args.run_id, args.target_label, args.stage, snap) as tee:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = tee  # type: ignore[assignment]
        sys.stderr = tee  # type: ignore[assignment]
        try:
            _configure_logging()
            stages.execute_stage(resolved, args.stage, force_rerun=args.force_rerun)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, force=True)
        log.exception("Stage failed: %s", exc)
        raise SystemExit(1) from exc
