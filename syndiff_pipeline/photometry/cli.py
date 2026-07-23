"""``syndiff photometry submit|run`` — event photometry for completed diff lanes."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from syndiff_pipeline.common.orchestration import cli as orch_cli
from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.common.orchestration.deployment import deployment_path_for_config
from syndiff_pipeline.common.orchestration.run_setup import apply_post_create_run_setup
from syndiff_pipeline.common.orchestration.scheduler_control import (
    ensure_daemon_running,
    warn_if_daemon_host_mismatch,
)
from syndiff_pipeline.common.orchestration.state import PipelineState
from syndiff_pipeline.common.orchestration.targets import load_targets
from syndiff_pipeline.common.orchestration.workspace import record_deployment_path
from syndiff_pipeline.difference_imaging.orchestration.site_config import SitePaths
from syndiff_pipeline.photometry.runner import run_photometry_pipeline
from syndiff_pipeline.photometry.site_config import (
    build_syndiff_config_for_photometry,
    load_photometry_site_policy,
    resolve_photometry_run_config,
)
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    load_runner_config,
    write_runner_config,
)

logger = logging.getLogger(__name__)


def _resolve_site_paths(args: argparse.Namespace) -> SitePaths:
    if not args.site:
        raise SystemExit("--site is required")
    return SitePaths.from_site_dir(args.site)


def _resolve_photometry_config_path(args: argparse.Namespace, paths: SitePaths) -> Path:
    if args.photometry_config:
        return Path(args.photometry_config).expanduser().resolve()
    candidate = paths.site_dir / "photometry_config.yaml"
    if candidate.is_file():
        return candidate
    raise SystemExit(
        "photometry config not found; pass --photometry-config or add config/photometry_config.yaml"
    )


def _resolve_targets_path(args: argparse.Namespace, paths: SitePaths) -> Path:
    if args.targets:
        return Path(args.targets).expanduser().resolve()
    raise SystemExit("--targets is required")


def _resolve_pipeline_config(args: argparse.Namespace, paths: SitePaths) -> Path:
    if args.config:
        return Path(args.config).expanduser().resolve()
    return paths.template_config


def _load_photometry_run_bundle(args: argparse.Namespace):
    paths = _resolve_site_paths(args)
    phot_config_path = _resolve_photometry_config_path(args, paths)
    targets_path = _resolve_targets_path(args, paths)
    policy = load_photometry_site_policy(phot_config_path)
    targets = load_targets(targets_path)
    target = next((t for t in targets if t.label() == args.target_name or t.target_name == args.target_name), None)
    if target is None:
        from syndiff_pipeline.common.orchestration.targets import find_target

        target = find_target(targets, args.target_name)
    run_config = resolve_photometry_run_config(policy, target, site_dir=paths.site_dir)
    cfg = build_syndiff_config_for_photometry(
        policy, target, run_config, site_dir=paths.site_dir
    )
    return paths, policy, target, run_config, cfg


def _patch_local_photometry_executor(run_directory: Path) -> None:
    cfg_path = logs.run_config_path(run_directory)
    cfg = load_runner_config(cfg_path)
    cfg.stages.photometry.executor = "local"
    write_runner_config(cfg, cfg_path)


def cmd_submit(args: argparse.Namespace) -> int:
    paths = _resolve_site_paths(args)
    pipeline_config = _resolve_pipeline_config(args, paths)
    phot_config_path = _resolve_photometry_config_path(args, paths)
    targets_path = _resolve_targets_path(args, paths)
    targets = load_targets(targets_path)

    cfg = load_runner_config(pipeline_config)
    run_id = args.run_id or orch_cli._default_run_id()
    runs_root = cfg.runs_dir()
    state = PipelineState(cfg.state_db_path)
    orch_cli._reject_duplicate_run_id(state, run_id)

    run_directory = orch_cli._prepare_run_directory(
        str(pipeline_config),
        run_id,
        runs_root,
        stages=["photometry"],
        detach=True,
        force_rerun=bool(args.force_rerun),
        source_targets=str(targets_path),
        source_photometry_config_path=str(phot_config_path),
    )

    from syndiff_pipeline.common.orchestration.targets import write_normalized_targets

    write_normalized_targets(logs.run_targets_path(run_directory), targets)

    if getattr(args, "local", False):
        _patch_local_photometry_executor(run_directory)

    state.create_run(
        run_id,
        str(logs.run_config_path(run_directory)),
        str(logs.run_targets_path(run_directory)),
        runs_root,
        targets,
        ["photometry"],
        force_rerun=bool(args.force_rerun),
    )
    setup = apply_post_create_run_setup(state, run_id, targets, cfg, ["photometry"])
    if setup.not_selected:
        print(
            f"Marked {setup.not_selected} stage row(s) n/a "
            "(not selected for this run)."
        )

    deploy_path = deployment_path_for_config(str(pipeline_config), cfg.deployment_file)
    record_deployment_path(cfg.workspace_root, deploy_path)
    result = ensure_daemon_running(cfg.workspace_root, deployment_path=deploy_path)
    warn_if_daemon_host_mismatch(cfg.workspace_root)
    daemon_log = logs.daemon_log_path(cfg.workspace_root)

    if cfg.notifications.enabled:
        from syndiff_pipeline.common.orchestration.notifications import send_run_started_notification

        send_run_started_notification(
            state,
            cfg.notifications,
            config_path=str(pipeline_config),
            run_id=run_id,
            run_dir=run_directory,
            target_labels=[t.label() for t in targets if t.enabled],
            stages=["photometry"],
            workspace_root=cfg.workspace_root,
            deployment_file=cfg.deployment_file,
            force_rerun=bool(args.force_rerun),
        )

    print(f"Submitted run_id={run_id} supervisor_pid={result.pid}")
    print(f"  daemon log: {daemon_log}")
    print("Monitor: syndiff progress --run-id", run_id)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    paths, policy, target, run_config, cfg = _load_photometry_run_bundle(args)
    try:
        run_photometry_pipeline(
            cfg,
            target,
            paths.site_dir,
            run_config=run_config,
            policy=policy,
            force_rerun=bool(args.force_rerun),
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syndiff photometry",
        description="Event photometry (astrometry + forced LCs) for completed SCC diff lanes",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--site", required=True, help="Site config directory")
    common.add_argument(
        "--photometry-config",
        default=None,
        help="Photometry site policy YAML (default: <site>/photometry_config.yaml)",
    )
    common.add_argument("--targets", default=None, help="Targets CSV with target_ra/dec")

    submit = sub.add_parser(
        "submit",
        parents=[common],
        help="Submit supervised batch over enabled targets rows",
    )
    submit.add_argument(
        "--config",
        default=None,
        help="Orchestrator policy YAML (default: <site>/pipeline.yaml)",
    )
    submit.add_argument("--run-id", default=None, help="Unique run name")
    submit.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing photometry artifacts for this run",
    )
    submit.add_argument(
        "--local",
        action="store_true",
        help="Run photometry stage locally instead of Condor",
    )

    run = sub.add_parser(
        "run",
        parents=[common],
        help="Foreground single-target debug run",
    )
    run.add_argument(
        "--target-name",
        required=True,
        help="Full event label or target_name from targets CSV",
    )
    run.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run photometry even when outputs exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "submit":
        return cmd_submit(args)
    if args.command == "run":
        return cmd_run(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
