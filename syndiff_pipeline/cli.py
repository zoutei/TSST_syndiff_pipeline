"""Unified ``syndiff`` CLI entry point (noun/verb structure)."""

from __future__ import annotations

import argparse
import logging
import sys

# Keep the console-script import cheap.  The orchestration CLI imports YAML,
# SQLite/state, daemon, and scheduler-control support; none of that is needed
# until a command is actually dispatched.  In particular, this matters for
# ``syndiff progress`` and ``syndiff --help``.
COMBINED_PRESET = "combined"
PRESET_NAMES = frozenset({"template", "diff", COMBINED_PRESET})


def _orchestration_cli():
    """Load the orchestration command module only when it is needed."""
    from syndiff_pipeline.common.orchestration import cli as orch_cli

    return orch_cli


def __getattr__(name: str):
    """Lazily preserve the historical orchestration-CLI re-exports."""
    if name == "preset_stages":
        return _orchestration_cli().preset_stages
    raise AttributeError(name)

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
        help=(
            "Unified orchestrator policy YAML -- always a site pipeline.yaml-style "
            "file (default: <site>/pipeline.yaml when --site is set), never a bare "
            "diff_config.yaml. For 'diff run --target-name' (foreground debug run: "
            "no run id, no state DB, no supervisor) this is the same pipeline.yaml; "
            "its embedded 'diff:' policy is used when present, else the legacy "
            "'diff_config:'/'diff_site_config:' pointer it names."
        ),
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


def build_combined_submit_parser() -> argparse.ArgumentParser:
    """Build ``syndiff submit``: one SCC-scoped template-to-diff run."""
    p = argparse.ArgumentParser(
        prog="syndiff submit",
        description="SynDiff combined SCC template-to-diff submission",
    )
    _add_shared_execution_args(p)
    p.add_argument(
        "--scc",
        required=True,
        help="SCC CSV (sector,camera,ccd[,enabled])",
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
    """Foreground diff pipeline for one target.

    Debug entry point: no run id, no state DB, no supervisor -- it resolves
    one target's diff config and calls ``run_pipeline`` directly in this
    process.
    """
    from pathlib import Path

    from syndiff_pipeline.common.orchestration.deployment import (
        deployment_path_for_config,
        load_deployment_file,
    )
    from syndiff_pipeline.common.orchestration.targets import find_target, load_targets
    from syndiff_pipeline.difference_imaging.orchestration.cli import run_pipeline
    from syndiff_pipeline.difference_imaging.orchestration.site_config import (
        SitePaths,
        freeze_target_diff_config,
        resolve_diff_config,
    )
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        load_runner_config,
    )

    if not args.target_name:
        raise SystemExit("--target-name is required for diff foreground run")
    if not args.targets:
        raise SystemExit("--targets is required for diff foreground run with --target-name")
    if args.config:
        config_path = args.config
    elif args.site:
        config_path = str(SitePaths.from_site_dir(args.site).template_config)
    else:
        raise SystemExit("--site or --config (unified pipeline.yaml) is required")

    targets = load_targets(args.targets)
    target = find_target(targets, args.target_name)

    runner_cfg = load_runner_config(config_path)
    deploy_override = getattr(args, "deployment", None)

    if runner_cfg.diff is not None:
        # Unified (schema v2) policy from pipeline.yaml's embedded ``diff:``.
        # diff.deployment_file is not a thing (see parse_unified_diff_policy) --
        # the site-level deployment_file always governs here.
        deploy_path = (
            Path(deploy_override).expanduser().resolve()
            if deploy_override
            else deployment_path_for_config(config_path, runner_cfg.deployment_file)
        )
        deployment = load_deployment_file(deploy_path)
        cfg = resolve_diff_config(
            target,
            runner_cfg.diff,
            deployment,
            deployment_path=deploy_path,
            site_config_dir=Path(runner_cfg.diff.source_dir),
        )
    elif runner_cfg.diff_config_path:
        # Legacy (schema v1): pipeline.yaml's 'diff_config:'/'diff_site_config:'
        # pointer names a standalone diff_config.yaml -- kept for sites not yet
        # migrated to an embedded 'diff:' block; a later wave removes it. Let
        # freeze_target_diff_config resolve its own deployment path (honoring
        # that file's own deployment_file key) unless the caller overrode it.
        cfg = freeze_target_diff_config(
            runner_cfg.diff_config_path,
            target,
            deployment_path=deploy_override,
        )
    else:
        raise SystemExit(
            f"{config_path} has no diff policy: no embedded 'diff:' block and no "
            "'diff_config'/'diff_site_config' pointer to a diff_config.yaml."
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
            return _orchestration_cli().cmd_run(args)
        raise SystemExit(
            "syndiff diff run requires --target-name for foreground debugging, "
            "or --scc/--sector/--camera/--ccd for supervised SCC-only diff. "
            "For supervised multi-target event diff, use: syndiff diff submit ..."
        )

    if verb == "submit":
        return _orchestration_cli().cmd_submit(args)
    return _orchestration_cli().cmd_run(args)


def _dispatch_combined_submit(argv: list[str]) -> int:
    """Submit one event-targeted run spanning templates through diff."""
    parser = build_combined_submit_parser()
    args = parser.parse_args(argv)
    args = _finalize_execution_args(COMBINED_PRESET, args)
    return _orchestration_cli().cmd_submit(args)


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
            "Combined template-to-diff:\n"
            "  syndiff submit --config PIPELINE.yaml --scc SCCS.csv\n\n"
            "Host-star light curves:\n"
            "  syndiff star submit --site SITE --star-targets STAR_TARGETS.csv\n"
            "  syndiff star run --site SITE --star-targets STAR_TARGETS.csv "
            "--target-name 20/3/2\n\n"
            "Monitoring & control:\n"
            "  syndiff status|progress|active|cluster|runs|show|logs|tail|retry|pause|resume|kill\n"
            "  syndiff verify|reconcile-manifests|daemon|notify\n\n"
            "Mask export (SCC diff lane):\n"
            "  syndiff mask export --scc s0022/c3/k3 --ffi tess2020050192921\n\n"
            "Run: syndiff <command> --help"
        )
        return 0

    noun = argv[0]
    if noun == "all":
        raise SystemExit(
            "The 'all' preset was removed. Use 'syndiff template submit|run ...' "
            "and 'syndiff diff submit|run ...' separately."
        )
    if noun == "submit":
        return _dispatch_combined_submit(argv[1:])
    if noun == "star":
        from syndiff_pipeline.star.cli import main as star_main

        return star_main(argv[1:])

    if noun == "photometry":
        from syndiff_pipeline.photometry.cli import main as photometry_main

        return photometry_main(argv[1:])

    if noun == "mask":
        from syndiff_pipeline.difference_imaging.masking.cli import main as mask_main

        return mask_main(argv[1:])

    if noun in PRESET_NAMES:
        return _dispatch_execution(noun, argv[1:])

    return _orchestration_cli().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
