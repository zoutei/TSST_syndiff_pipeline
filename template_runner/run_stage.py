"""Subprocess entry point for a single template pipeline stage."""

from __future__ import annotations

import argparse
import logging
import sys

from syndiff_pipeline.template_runner import logs, stages
from syndiff_pipeline.template_runner.runner_config import load_runner_config, resolve_config
from syndiff_pipeline.template_runner.targets import load_targets

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one template pipeline stage")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--target-label", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_runner_config(args.config)
    targets = load_targets(args.targets)
    target = next(t for t in targets if t.label() == args.target_label)
    resolved = resolve_config(target, cfg)
    snap = stages.stage_snapshot(resolved, args.stage)

    with logs.stage_log(cfg.runs_dir(), args.run_id, args.target_label, args.stage, snap) as tee:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = tee  # type: ignore[assignment]
        sys.stderr = tee  # type: ignore[assignment]
        try:
            stages.execute_stage(resolved, args.stage)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log.exception("Stage failed: %s", exc)
        raise SystemExit(1) from exc
