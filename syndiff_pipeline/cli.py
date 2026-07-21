"""Unified ``syndiff`` CLI entry point (noun/verb structure)."""

from __future__ import annotations

import argparse
import logging
import sys

from syndiff_pipeline.common.orchestration import cli as orch_cli
from syndiff_pipeline.common.orchestration.cli import PRESET_NAMES, preset_stages  # noqa: F401

EXECUTION_VERBS = frozenset({"submit", "run"})


def _add_shared_execution_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--site",
        default=None,
        help="Config directory with pipeline.yaml, diff_config.yaml, and deployment.yaml",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Orchestrator policy YAML (default: <site>/pipeline.yaml when --site is set)",
    )
    p.add_argument(
        "--deployment",
        default=None,
        help="Path to deployment.yaml (optional override)",
    )
    p.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stage override (replaces the preset stage list)",
    )
    p.add_argument("--run-id", default=None, help="Unique run name (must not already exist)")
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing artifacts for selected stages (new run only)",
    )
    p.add_argument(
        "--skip-artifact-verify",
        action="store_true",
        help=(
            "Skip supervisor pre-flight artifact scanning; trust upstream "
            "external stages as complete (unsafe)"
        ),
    )
    p.add_argument(
        "--workspace-run-id",
        default=None,
        help="Debug workspace suffix (writes to ws_{id}/ instead of ws/)",
    )


def build_execution_parser(preset: str, verb: str) -> argparse.ArgumentParser:
    """Build execution parser.
    
    Parameters
    ----------
    preset : str
    verb : str
    
    Returns
    -------
    argparse.ArgumentParser"""
    if preset not in PRESET_NAMES:
        raise ValueError(f"Unknown preset: {preset!r}")
    if verb not in EXECUTION_VERBS:
        raise ValueError(f"Unknown execution verb: {verb!r}")

    p = argparse.ArgumentParser(
        prog=f"syndiff {preset} {verb}",
        description=f"SynDiff pipeline ({preset} stage preset, {verb})",
    )
    _add_shared_execution_args(p)

    if preset == "template":
        p.add_argument(
            "--scc",
            default=None,
            help="SCC CSV path (sector,camera,ccd[,enabled])",
        )
        p.add_argument("--sector", type=int, default=None, help="Single SCC sector")
        p.add_argument("--camera", type=int, default=None, help="Single SCC camera")
        p.add_argument("--ccd", type=int, default=None, help="Single SCC CCD")
        p.add_argument(
            "--local",
            action="store_true",
            help=argparse.SUPPRESS,
        )
    elif preset == "diff":
        p.add_argument(
            "--targets",
            default=None,
            help="Targets CSV path (event photometry); optional for SCC-only diff",
        )
        p.add_argument(
            "--scc",
            default=None,
            help="SCC CSV path (sector,camera,ccd[,enabled]) for field-mode diff",
        )
        p.add_argument("--sector", type=int, default=None, help="Single SCC sector")
        p.add_argument("--camera", type=int, default=None, help="Single SCC camera")
        p.add_argument("--ccd", type=int, default=None, help="Single SCC CCD")
        p.add_argument(
            "--target-name",
            default=None,
            help="Run a single target by label (diff foreground debugging)",
        )
        p.add_argument(
            "--validate-only",
            action="store_true",
            help="Validate diff config/stages without executing (diff foreground run only)",
        )
        p.add_argument(
            "--local",
            action="store_true",
            help="Executor override: run diff stage locally (submit only)",
        )
    return p


def _resolve_execution_config(args: argparse.Namespace) -> None:
    """Resolve execution config.
    
    Parameters
    ----------
    args : argparse.Namespace"""
    if args.config:
        return
    if args.site:
        from syndiff_pipeline.difference_imaging.orchestration.site_config import SitePaths

        paths = SitePaths.from_site_dir(args.site)
        args.config = str(paths.template_config)
        return
    raise SystemExit("--config or --site is required for submit/run")


def _finalize_execution_args(preset: str, args: argparse.Namespace) -> argparse.Namespace:
    """Finalize execution args.
    
    Parameters
    ----------
    preset : str
    args : argparse.Namespace
    
    Returns
    -------
    argparse.Namespace"""
    _resolve_execution_config(args)
    args.preset = preset
    return args


def parse_execution_argv(argv: list[str]) -> tuple[str, str, argparse.Namespace]:
    """Parse ``[preset, verb, ...flags]`` into preset name, verb, and namespace."""
    if len(argv) < 2:
        raise SystemExit("usage: syndiff <template|diff> <submit|run> ...")
    preset = argv[0]
    verb = argv[1]
    if preset not in PRESET_NAMES:
        raise SystemExit(f"Unknown preset {preset!r}; expected one of: {', '.join(sorted(PRESET_NAMES))}")
    if verb not in EXECUTION_VERBS:
        raise SystemExit(f"Unknown verb {verb!r}; expected submit or run")
    parser = build_execution_parser(preset, verb)
    args = parser.parse_args(argv[2:])
    return preset, verb, _finalize_execution_args(preset, args)


def _cmd_diff_foreground_run(args: argparse.Namespace) -> int:
    """Foreground diff pipeline for one target (no supervisor / state DB)."""
    from syndiff_pipeline.common.orchestration.targets import find_target, load_targets
    from syndiff_pipeline.difference_imaging.orchestration.cli import run_pipeline
    from syndiff_pipeline.difference_imaging.orchestration.site_config import (
        SitePaths,
        freeze_target_diff_config,
    )

    if not args.target_name:
        raise SystemExit("--target-name is required for diff foreground run")
    if not args.targets:
        raise SystemExit("--targets is required for diff foreground run with --target-name")
    if args.site:
        paths = SitePaths.from_site_dir(args.site)
        diff_config = str(paths.diff_config)
        deployment = str(paths.deployment) if paths.deployment.is_file() else None
    elif args.config:
        diff_config = args.config
        deployment = args.deployment
    else:
        raise SystemExit("--site or --config (diff site config) is required")

    targets = load_targets(args.targets)
    target = find_target(targets, args.target_name)
    cfg = freeze_target_diff_config(
        diff_config,
        target,
        deployment_path=deployment,
    )
    if getattr(args, "workspace_run_id", None):
        cfg.workspace_run_id = str(args.workspace_run_id).strip()
        cfg.pipeline_plots = True
    run_pipeline(
        cfg,
        validate_only=bool(args.validate_only),
        force_rerun=bool(getattr(args, "force_rerun", False)),
    )
    return 0


def _dispatch_execution(preset: str, argv: list[str]) -> int:
    """Dispatch execution.
    
    Parameters
    ----------
    preset : str
    argv : list[str]
    
    Returns
    -------
    int"""
    if not argv:
        raise SystemExit(f"usage: syndiff {preset} submit|run ...")
    verb = argv[0]
    if verb in ("-h", "--help"):
        build_execution_parser(preset, "submit" if preset else "submit").print_help()
        return 0
    if verb not in EXECUTION_VERBS:
        raise SystemExit(f"syndiff {preset} requires submit or run (got {verb!r})")

    parser = build_execution_parser(preset, verb)
    args = _finalize_execution_args(preset, parser.parse_args(argv[1:]))

    if preset == "diff" and verb == "run":
        if args.target_name:
            return _cmd_diff_foreground_run(args)
        has_scc_scope = bool(getattr(args, "scc", None)) or any(
            getattr(args, name, None) is not None for name in ("sector", "camera", "ccd")
        )
        if has_scc_scope:
            # Supervised field-mode / SCC-only run (same path as template run).
            return orch_cli.cmd_run(args)
        raise SystemExit(
            "syndiff diff run requires --target-name for foreground debugging, "
            "or --scc/--sector/--camera/--ccd for supervised SCC-only diff. "
            "For supervised multi-target event diff, use: syndiff diff submit ..."
        )

    if verb == "submit":
        return orch_cli.cmd_submit(args)
    return orch_cli.cmd_run(args)


def main(argv: list[str] | None = None) -> int:
    """Main.
    
    Parameters
    ----------
    argv : list[str] | None, optional, default ``None``
    
    Returns
    -------
    int"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: syndiff <noun> <verb> ...\n\n"
            "Execution presets:\n"
            "  syndiff template submit|run --site SITE --scc SCCS.csv\n"
            "  syndiff diff submit|run --site SITE --targets TARGETS.csv\n"
            "  syndiff diff submit|run --site SITE --scc SCCS.csv\n\n"
            "Host-star light curves:\n"
            "  syndiff star submit --site SITE --star-targets STAR_TARGETS.csv\n"
            "  syndiff star run --site SITE --star-targets STAR_TARGETS.csv "
            "--target-name 20/3/2\n\n"
            "Monitoring & control:\n"
            "  syndiff status|progress|runs|active|show|logs|tail|retry|pause|resume|kill\n"
            "  syndiff verify|reconcile-manifests|daemon|notify\n\n"
            "Run: syndiff <command> --help"
        )
        return 0

    noun = argv[0]
    if noun == "all":
        raise SystemExit(
            "The 'all' preset was removed. Use 'syndiff template submit|run ...' "
            "and 'syndiff diff submit|run ...' separately."
        )
    if noun == "star":
        from syndiff_pipeline.star.cli import main as star_main

        return star_main(argv[1:])

    if noun in PRESET_NAMES:
        return _dispatch_execution(noun, argv[1:])

    return orch_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
