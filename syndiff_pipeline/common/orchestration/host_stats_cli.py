"""CLI and Discord formatting for cluster host sampler JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from syndiff_pipeline.common.orchestration.host_stats import (
    default_stats_dir,
    discover_samples,
    evaluate_host,
    expected_hosts,
    format_machine_exclusions,
    plan_host_selection,
)

PRESETS = {
    "500gb": {"min_mem_mb": 300_000, "max_load15": 10.0},
    "128gb": {"min_mem_mb": 128_000, "max_load15": 10.0},
}

# Typical HTCondor slot Memory values on the science cluster (MB).
_COMMON_CONDOR_MEMORY_MB = (128_000, 256_000, 500_000, 512_000, 515_000)

_TEMPLATE_STAGES = frozenset({"mapping", "ps1_process", "remap", "downsample"})
_BRANCH_STAGES = frozenset({"diff", "star", "photometry"})


def format_mem_gb(mb: int) -> str:
    """Format megabytes as decimal GB for display (one decimal place)."""
    gb = mb / 1000
    return f"{gb:.1f}GB"


def format_condor_slot_gb(mem_total_mb: int) -> str:
    """Format total RAM like ``condor_status -af Memory`` (128GB / 515GB buckets)."""
    if mem_total_mb <= 0:
        return "?"
    best = min(_COMMON_CONDOR_MEMORY_MB, key=lambda m: abs(m - mem_total_mb))
    if abs(best - mem_total_mb) <= 25_000:
        return f"{best // 1000}GB"
    return format_mem_gb(mem_total_mb)


def format_reasons_for_display(
    reasons: tuple[str, ...],
    sample,
) -> tuple[str, ...]:
    out: list[str] = []
    for reason in reasons:
        if reason.startswith("low mem ") and sample is not None:
            out.append(f"low mem {format_mem_gb(sample.mem_available_mb)}")
        else:
            out.append(reason)
    return tuple(out)


@dataclass(frozen=True)
class HostVerdict:
    hostname: str
    reasons: tuple[str, ...]

    @property
    def label(self) -> str:
        if not self.reasons:
            return "OK"
        return "EXCLUDE (" + ", ".join(self.reasons) + ")"


def build_verdicts(
    samples: dict,
    *,
    min_mem_mb: int,
    max_load15: float,
    max_age_s: int,
) -> list[tuple[str, HostVerdict]]:
    verdicts: list[tuple[str, HostVerdict]] = []
    for host in expected_hosts():
        sample = samples.get(host)
        reasons = evaluate_host(
            sample,
            min_mem_mb=min_mem_mb,
            max_load15=max_load15,
            max_age_s=max_age_s,
        )
        verdicts.append(
            (
                host,
                HostVerdict(
                    hostname=host,
                    reasons=format_reasons_for_display(reasons, sample),
                ),
            )
        )
    return verdicts


@dataclass(frozen=True)
class _ClusterTableRow:
    host: str
    slot: str
    avail: str
    load15: str
    age: str
    verdict: str | None = None


def _cluster_table_row(
    host: str,
    sample,
    verdict: HostVerdict | None,
) -> _ClusterTableRow:
    if sample is None:
        slot = "?"
        avail = "?"
        load15 = "?"
        age = "?"
    else:
        slot = format_condor_slot_gb(sample.mem_total_mb)
        avail = format_mem_gb(sample.mem_available_mb)
        load15 = f"{sample.load15:.2f}"
        age = f"{sample.age_s}s"
    return _ClusterTableRow(
        host=host,
        slot=slot,
        avail=avail,
        load15=load15,
        age=age,
        verdict=verdict.label if verdict is not None else None,
    )


def _format_fixed_width_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Sequence[str],
) -> str:
    """Render a fixed-width table with per-column width from headers + data."""
    if len(headers) != len(aligns):
        raise ValueError("headers and aligns must have the same length")
    widths = [len(header) for header in headers]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("row width does not match headers")
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def _format_cells(cells: Sequence[str]) -> str:
        parts: list[str] = []
        for cell, width, align in zip(cells, widths, aligns):
            if align == "l":
                parts.append(f"{cell:<{width}}")
            else:
                parts.append(f"{cell:>{width}}")
        return " ".join(parts)

    lines = [_format_cells(headers)]
    lines.append(" ".join("-" * width for width in widths))
    lines.extend(_format_cells(row) for row in rows)
    return "\n".join(lines)


def format_cluster_table(
    samples: dict,
    verdicts: Sequence[tuple[str, HostVerdict]] | None = None,
    *,
    include_verdict: bool = False,
) -> str:
    """Render the host status table (optionally with VERDICT column)."""
    if verdicts is None:
        verdicts = [(host, HostVerdict(host, ())) for host in expected_hosts()]

    table_rows = [
        _cluster_table_row(host, samples.get(host), verdict if include_verdict else None)
        for host, verdict in verdicts
    ]
    headers = ["HOST", "SLOT", "AVAIL", "LOAD15", "AGE"]
    aligns = ["l", "r", "r", "r", "r"]
    if include_verdict:
        headers.append("VERDICT")
        aligns.append("l")

    rows = [
        (
            (row.host, row.slot, row.avail, row.load15, row.age, row.verdict)
            if include_verdict
            else (row.host, row.slot, row.avail, row.load15, row.age)
        )
        for row in table_rows
    ]
    return _format_fixed_width_table(headers, rows, aligns=aligns)


def render_cluster_table_text(
    *,
    stats_dir: Path | None = None,
    include_verdict: bool = False,
    min_mem_mb: int | None = None,
    max_load15: float | None = None,
    max_age_s: int = 300,
) -> str:
    """Build the cluster table body (for CLI stdout or Discord code fence)."""
    stats_dir = stats_dir or default_stats_dir()
    samples = discover_samples(stats_dir)
    verdicts = None
    if include_verdict:
        min_mem = min_mem_mb if min_mem_mb is not None else PRESETS["500gb"]["min_mem_mb"]
        max_load = max_load15 if max_load15 is not None else PRESETS["500gb"]["max_load15"]
        verdicts = build_verdicts(
            samples,
            min_mem_mb=min_mem,
            max_load15=max_load,
            max_age_s=max_age_s,
        )
    return format_cluster_table(samples, verdicts, include_verdict=include_verdict)


def resolve_thresholds_from_site(site_dir: str | Path, stage: str) -> tuple[int, float]:
    """Load host_stats thresholds from a site config stage block."""
    from syndiff_pipeline.difference_imaging.orchestration.site_config import SitePaths

    site = SitePaths.from_site_dir(site_dir)
    stage_key = stage.strip().lower()
    if stage_key in _TEMPLATE_STAGES:
        from syndiff_pipeline.template_creation.orchestration.runner_config import (
            load_runner_config,
        )

        cfg = load_runner_config(str(site.template_config))
        params = getattr(cfg.stages, stage_key)
        return int(params.host_stats_min_mem_mb), float(params.host_stats_max_load15)
    if stage_key == "diff":
        from syndiff_pipeline.difference_imaging.orchestration.site_config import (
            load_diff_site_policy,
        )

        policy = load_diff_site_policy(site.diff_config)
        return int(policy.condor.host_stats_min_mem_mb), float(policy.condor.host_stats_max_load15)
    if stage_key == "star":
        from syndiff_pipeline.star.site_config import load_star_site_policy

        star_path = site.site_dir / "star_config.yaml"
        if not star_path.is_file():
            raise FileNotFoundError(f"star_config.yaml not found under {site.site_dir}")
        policy = load_star_site_policy(star_path)
        return int(policy.condor.host_stats_min_mem_mb), float(policy.condor.host_stats_max_load15)
    if stage_key == "photometry":
        phot_path = site.site_dir / "photometry_config.yaml"
        if not phot_path.is_file():
            raise FileNotFoundError(
                f"photometry_config.yaml not found under {site.site_dir}"
            )
        from syndiff_pipeline.photometry.site_config import load_photometry_site_policy

        policy = load_photometry_site_policy(phot_path)
        return int(policy.condor.host_stats_min_mem_mb), float(policy.condor.host_stats_max_load15)
    raise ValueError(
        f"Unknown --stage {stage!r}; expected one of: "
        f"{', '.join(sorted(_TEMPLATE_STAGES | _BRANCH_STAGES))}"
    )


def resolve_thresholds(args: argparse.Namespace) -> tuple[int, float]:
    min_mem_mb = args.min_mem_mb
    max_load15 = args.max_load15
    if getattr(args, "site", None) and getattr(args, "stage", None):
        site_min, site_max = resolve_thresholds_from_site(args.site, args.stage)
        if min_mem_mb is None:
            min_mem_mb = site_min
        if max_load15 is None:
            max_load15 = site_max
    if args.preset:
        preset = PRESETS[args.preset]
        if min_mem_mb is None:
            min_mem_mb = preset["min_mem_mb"]
        if max_load15 is None:
            max_load15 = preset["max_load15"]
    if min_mem_mb is None:
        min_mem_mb = PRESETS["500gb"]["min_mem_mb"]
    if max_load15 is None:
        max_load15 = PRESETS["500gb"]["max_load15"]
    return min_mem_mb, max_load15


def format_bad_machines(hosts: list[str]) -> str:
    return json.dumps({"hosts": sorted(hosts)}, indent=2) + "\n"


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Show science-cluster execute-host memory and load from sampler JSON. "
            "Default is a compact status table (no VERDICT column). "
            "Use --check for placement preview with pass/fail per host."
        ),
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=None,
        help=f"Per-host JSON directory (default: {default_stats_dir()})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Evaluate placement thresholds; add VERDICT column and exclusion summary",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Threshold preset for --check: 500gb (ps1_process) or 128gb (mapping/remap)",
    )
    parser.add_argument(
        "--min-mem-mb",
        type=int,
        help="With --check: min MemAvailable (MB) from sampler JSON",
    )
    parser.add_argument(
        "--max-load15",
        type=float,
        help="With --check: exclude hosts with load15 at or above this value",
    )
    parser.add_argument(
        "--max-age-s",
        type=int,
        default=300,
        help="With --check: max heartbeat age in seconds (default: 300)",
    )
    parser.add_argument(
        "--site",
        default=None,
        help="Site config directory; use with --stage and --check for stage thresholds",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Stage for --site thresholds (mapping, ps1_process, diff, star, ...)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "requirements", "bad-machines", "hosts"),
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--include-ok",
        action="store_true",
        help="For --format hosts only, list every host instead of just excluded "
        "ones (no effect on requirements/bad-machines, which are exclusion "
        "lists by definition)",
    )
    parser.add_argument(
        "--no-stats-dir-line",
        action="store_true",
        help="Omit stats_dir header line (for machine consumers)",
    )
    return parser


def main(argv: list[str] | None = None, *, default_check: bool = False) -> int:
    parser = build_parser()
    if default_check:
        parser.set_defaults(check=True)
    args = parser.parse_args(argv)
    stats_dir = args.stats_dir or default_stats_dir()

    check_mode = bool(args.check) or args.format != "table"
    min_mem_mb, max_load15 = resolve_thresholds(args) if check_mode else (0, 0.0)

    samples = discover_samples(stats_dir)
    verdicts = (
        build_verdicts(
            samples,
            min_mem_mb=min_mem_mb,
            max_load15=max_load15,
            max_age_s=args.max_age_s,
        )
        if check_mode
        else None
    )

    selection = (
        plan_host_selection(
            stats_dir=stats_dir,
            min_mem_mb=min_mem_mb,
            max_load15=max_load15,
            max_age_s=args.max_age_s,
        )
        if check_mode
        else None
    )
    excluded = sorted(selection.excluded) if selection else []
    ok_hosts = [host.hostname for host in selection.eligible] if selection else []

    if args.format == "table":
        if not args.no_stats_dir_line:
            print(f"stats_dir: {stats_dir}")
            if not stats_dir.is_dir():
                print(
                    f"WARNING: stats directory does not exist: {stats_dir}",
                    file=sys.stderr,
                )
        print(
            format_cluster_table(
                samples,
                verdicts,
                include_verdict=check_mode,
            )
        )
        if check_mode:
            print()
            print(
                f"Thresholds: min_mem={format_mem_gb(min_mem_mb)}, "
                f"max_load15={max_load15}, max_age_s={args.max_age_s}"
            )
            print(f"Excluded: {len(excluded)}  OK: {len(ok_hosts)}")
            if excluded:
                print()
                print("requirements:")
                print(format_machine_exclusions(excluded))
        return 0

    if args.format == "hosts":
        # "Include OK hosts" makes sense here -- it's a genuine "list everyone"
        # listing, not an exclusion structure.
        hosts_out = expected_hosts() if args.include_ok else excluded
        for host in hosts_out:
            print(host)
        return 0

    # requirements/bad-machines are exclusion structures by definition -- both
    # feed real Condor policy (format_bad_machines' schema matches what
    # condor.py's read_bad_machines() consumes), so "including OK hosts"
    # would silently turn "hosts to avoid" into "every host", which is not
    # what the flag name suggests and would be a dangerous misuse if piped
    # into an actual exclusion file. Warn instead of silently doing that.
    if args.include_ok:
        print(
            f"WARNING: --include-ok has no effect on --format {args.format} "
            "(an exclusion list can't sensibly include OK hosts); ignoring.",
            file=sys.stderr,
        )
    if args.format == "requirements":
        print(format_machine_exclusions(excluded))
    elif args.format == "bad-machines":
        print(format_bad_machines(excluded))
    return 0
