"""``syndiff star submit|run`` — host-star light curves for existing events."""

from __future__ import annotations

import argparse
import logging
import shutil
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
from syndiff_pipeline.common.orchestration.workspace import record_deployment_path
from syndiff_pipeline.difference_imaging.orchestration.site_config import SitePaths
from syndiff_pipeline.star.context import StarPrerequisiteError, load_event_context
from syndiff_pipeline.star.runner import run_star_pipeline
from syndiff_pipeline.star.site_config import (
    find_star_target_row,
    load_star_site_policy,
    load_star_targets,
    resolve_star_run_config,
    star_targets_to_orchestrator_targets,
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


def _resolve_star_config_path(args: argparse.Namespace, paths: SitePaths) -> Path:
    if args.star_config:
        return Path(args.star_config).expanduser().resolve()
    candidate = paths.site_dir / "star_config.yaml"
    if candidate.is_file():
        return candidate
    raise SystemExit(
        "star config not found; pass --star-config or add config/star_config.yaml"
    )


def _resolve_star_targets_path(args: argparse.Namespace, paths: SitePaths) -> Path:
    if args.star_targets:
        return Path(args.star_targets).expanduser().resolve()
    candidate = paths.site_dir / "star_targets_example.csv"
    if candidate.is_file():
        return candidate
    raise SystemExit(
        "star targets not found; pass --star-targets or add config/star_targets_example.csv"
    )


def _resolve_pipeline_config(args: argparse.Namespace, paths: SitePaths) -> Path:
    if args.config:
        return Path(args.config).expanduser().resolve()
    return paths.template_config


def _load_star_run_bundle(args: argparse.Namespace):
    paths = _resolve_site_paths(args)
    star_config_path = _resolve_star_config_path(args, paths)
    star_targets_path = _resolve_star_targets_path(args, paths)
    policy = load_star_site_policy(star_config_path)
    star_rows = load_star_targets(star_targets_path, site_dir=paths.site_dir)
    star_row = find_star_target_row(star_rows, args.target_name)
    run_config = resolve_star_run_config(policy, star_row, site_dir=paths.site_dir)
    if args.workspace_run_id is not None:
        run_config.workspace_run_id = str(args.workspace_run_id).strip() or None
    if args.cutout_size is not None:
        run_config.cutout_size = int(args.cutout_size)
    if args.stamp_size is not None:
        run_config.stamp_size = int(args.stamp_size)
    if args.kernel_margin_px is not None:
        run_config.kernel_margin_px = int(args.kernel_margin_px)
    if args.ps1_source is not None:
        run_config.ps1_source = args.ps1_source
    if args.overwrite:
        run_config.overwrite = True
    if args.debug_plots is not None:
        run_config.debug_plots = bool(args.debug_plots)
    if args.stars_file:
        run_config.stars_file = str(Path(args.stars_file).expanduser().resolve())
    if args.baseline_workspace_run_id:
        run_config.baseline.workspace_run_id = args.baseline_workspace_run_id
    if args.baseline_diffs_label:
        run_config.baseline.diffs = args.baseline_diffs_label
    if args.baseline_convolved_label:
        run_config.baseline.convolved = args.baseline_convolved_label
    if args.baseline_phot_bkg_label:
        run_config.baseline.phot_bkg = args.baseline_phot_bkg_label
    return paths, policy, star_row, run_config


def _patch_local_star_executor(run_directory: Path) -> None:
    cfg_path = logs.run_config_path(run_directory)
    cfg = load_runner_config(cfg_path)
    cfg.stages.star.executor = "local"
    write_runner_config(cfg, cfg_path)


def cmd_submit(args: argparse.Namespace) -> int:
    paths = _resolve_site_paths(args)
    pipeline_config = _resolve_pipeline_config(args, paths)
    star_config_path = _resolve_star_config_path(args, paths)
    star_targets_path = _resolve_star_targets_path(args, paths)

    cfg = load_runner_config(pipeline_config)
    star_rows = load_star_targets(star_targets_path, site_dir=paths.site_dir)
    targets = star_targets_to_orchestrator_targets(star_rows)

    run_id = args.run_id or orch_cli._default_run_id()
    runs_root = cfg.runs_dir()
    state = PipelineState(cfg.state_db_path)
    orch_cli._reject_duplicate_run_id(state, run_id)

    run_directory = orch_cli._prepare_run_directory(
        str(pipeline_config),
        str(star_targets_path),
        run_id,
        runs_root,
        stages=["star"],
        detach=True,
        force_rerun=bool(args.force_rerun),
        source_star_config_path=str(star_config_path),
        workspace_run_id=getattr(args, "workspace_run_id", None),
    )

    from syndiff_pipeline.common.orchestration.targets import write_normalized_targets

    write_normalized_targets(logs.run_targets_path(run_directory), targets)
    star_targets_dest = logs.run_star_targets_path(run_directory)
    shutil.copy2(star_targets_path, star_targets_dest)
    logs.update_run_meta(
        runs_root,
        run_id,
        {
            "source_star_targets_path": str(star_targets_path.resolve()),
            "star_targets_path": str(star_targets_dest.resolve()),
        },
    )

    if getattr(args, "local", False):
        _patch_local_star_executor(run_directory)

    state.create_run(
        run_id,
        str(logs.run_config_path(run_directory)),
        str(logs.run_targets_path(run_directory)),
        runs_root,
        targets,
        ["star"],
        force_rerun=bool(args.force_rerun),
    )
    setup = apply_post_create_run_setup(state, run_id, targets, cfg, ["star"])
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
            stages=["star"],
            workspace_root=cfg.workspace_root,
            deployment_file=cfg.deployment_file,
            force_rerun=bool(args.force_rerun),
        )

    print(f"Submitted run_id={run_id} supervisor_pid={result.pid}")
    print(f"  daemon log: {daemon_log}")
    print("Monitor: syndiff progress --run-id", run_id)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    paths, _policy, star_row, run_config = _load_star_run_bundle(args)

    legacy_targets = getattr(args, "targets", None)
    ctx = load_event_context(
        site=str(paths.site_dir),
        targets_csv=legacy_targets,
        target_name=args.target_name,
        baseline_workspace_run_id=run_config.baseline.workspace_run_id,
        baseline_diffs_label=run_config.baseline.diffs,
        baseline_convolved_label=run_config.baseline.convolved,
        baseline_phot_bkg_label=run_config.baseline.phot_bkg,
        star_run_config=run_config,
        star_target_row=star_row,
    )

    try:
        run_star_pipeline(ctx, run_config=run_config, validate=True)
    except StarPrerequisiteError as exc:
        logger.error("%s", exc)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syndiff star",
        description="Host-star light curves for an existing syndiff event",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--site", required=True, help="Site config directory")
    common.add_argument(
        "--star-config",
        default=None,
        help="Star site policy YAML (default: <site>/star_config.yaml)",
    )
    common.add_argument(
        "--star-targets",
        default=None,
        help="Star targets CSV (default: <site>/star_targets_example.csv)",
    )
    common.add_argument(
        "--workspace-run-id",
        default=None,
        help="Suffix for star outputs under events/{label}/star_{id}/",
    )

    submit = sub.add_parser(
        "submit",
        parents=[common],
        help="Submit supervised batch over enabled star_targets rows",
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
        help="Ignore existing star artifacts for this run",
    )
    submit.add_argument(
        "--local",
        action="store_true",
        help="Run star stage locally instead of Condor",
    )

    run = sub.add_parser(
        "run",
        parents=[common],
        help="Foreground single-SCC debug run",
    )
    run.add_argument(
        "--target-name",
        required=True,
        help="SCC key (20/3/2) or full event label from star_targets.csv",
    )
    run.add_argument(
        "--targets",
        default=None,
        help="Legacy transient targets CSV (optional when using star_targets)",
    )
    run.add_argument(
        "--stars-file",
        default=None,
        help="Override stars CSV from star_targets row",
    )
    run.add_argument(
        "--baseline-workspace-run-id",
        default=None,
        help="Override baseline workspace suffix (default from star config)",
    )
    run.add_argument("--baseline-diffs-label", default=None)
    run.add_argument("--baseline-convolved-label", default=None)
    run.add_argument("--baseline-phot-bkg-label", default=None)
    run.add_argument("--cutout-size", type=int, default=None)
    run.add_argument("--stamp-size", type=int, default=None)
    run.add_argument("--kernel-margin-px", type=int, default=None)
    run.add_argument(
        "--ps1-source",
        default=None,
        choices=("zarr_local_only", "zarr_download", "stream", "zarr", "download"),
        help="PS1 skycell ingest mode",
    )
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--debug-plots",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Write per-host debug plots",
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
