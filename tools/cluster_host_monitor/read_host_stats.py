#!/usr/bin/env python3
"""Read cluster host sampler JSON files and recommend Condor exclusions."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import os

DEFAULT_STATS_DIR = Path(
    os.environ.get("HOST_STATS_DIR", "/home/kshukawa/.syndiff/host_stats")
)

PRESETS = {
    "500gb": {"min_mem_mb": 520_000, "max_load1": 10.0},
    "128gb": {"min_mem_mb": 140_000, "max_load1": 10.0},
}


@dataclass(frozen=True)
class HostSample:
    hostname: str
    login_hostname: str | None
    timestamp: int
    mem_available_mb: int
    mem_total_mb: int
    load1: float
    load5: float
    load15: float
    path: Path

    @property
    def age_s(self) -> int:
        return max(0, int(time.time()) - self.timestamp)


@dataclass(frozen=True)
class HostVerdict:
    sample: HostSample | None
    reasons: tuple[str, ...]

    @property
    def exclude(self) -> bool:
        return bool(self.reasons)

    @property
    def label(self) -> str:
        if not self.reasons:
            return "OK"
        return "EXCLUDE (" + ", ".join(self.reasons) + ")"


def normalize_condor_host(host: str) -> str:
    text = str(host or "").strip()
    if "@" in text:
        text = text.rsplit("@", 1)[-1]
    match = re.fullmatch(r"science(\d+)(?:\.stsci\.edu)?", text)
    if match:
        return f"plscience{match.group(1)}.stsci.edu"
    return text


def load_sample(path: Path) -> HostSample:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hostname = normalize_condor_host(str(payload.get("hostname", path.stem)))
    login_hostname = payload.get("login_hostname")
    return HostSample(
        hostname=hostname,
        login_hostname=str(login_hostname) if login_hostname else None,
        timestamp=int(payload["timestamp"]),
        mem_available_mb=int(payload["mem_available_mb"]),
        mem_total_mb=int(payload["mem_total_mb"]),
        load1=float(payload["load1"]),
        load5=float(payload["load5"]),
        load15=float(payload["load15"]),
        path=path,
    )


def discover_samples(stats_dir: Path) -> dict[str, HostSample]:
    samples: dict[str, HostSample] = {}
    if not stats_dir.is_dir():
        return samples
    for path in sorted(stats_dir.glob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        try:
            sample = load_sample(path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        samples[sample.hostname] = sample
    return samples


def evaluate_sample(
    sample: HostSample | None,
    *,
    min_mem_mb: int,
    max_load1: float,
    max_age_s: int,
) -> HostVerdict:
    if sample is None:
        return HostVerdict(sample=None, reasons=("missing",))
    reasons: list[str] = []
    if sample.age_s > max_age_s:
        reasons.append(f"stale {sample.age_s}s")
    if sample.mem_available_mb < min_mem_mb:
        reasons.append(f"low mem {sample.mem_available_mb}MB")
    if sample.load1 > max_load1:
        reasons.append(f"high load {sample.load1:.2f}")
    return HostVerdict(sample=sample, reasons=tuple(reasons))


def expected_hosts() -> list[str]:
    return [f"plscience{n}.stsci.edu" for n in range(1, 16)]


def format_table(verdicts: list[tuple[str, HostVerdict]]) -> str:
    lines = [
        f"{'HOST':<28} {'MEM_AVAIL':>10} {'LOAD1':>7} {'AGE':>8}  VERDICT",
        f"{'-' * 28} {'-' * 10} {'-' * 7} {'-' * 8}  {'-' * 7}",
    ]
    for host, verdict in verdicts:
        sample = verdict.sample
        if sample is None:
            mem = "?"
            load1 = "?"
            age = "?"
        else:
            mem = f"{sample.mem_available_mb}MB"
            load1 = f"{sample.load1:.2f}"
            age = f"{sample.age_s}s"
        lines.append(f"{host:<28} {mem:>10} {load1:>7} {age:>8}  {verdict.label}")
    return "\n".join(lines)


def format_requirements(hosts: list[str]) -> str:
    if not hosts:
        return "# no exclusions"
    return " && ".join(f'Machine != "{host}"' for host in sorted(hosts))


def format_bad_machines(hosts: list[str]) -> str:
    return json.dumps({"hosts": sorted(hosts)}, indent=2) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read host sampler JSON heartbeats and recommend Condor host exclusions."
        )
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=DEFAULT_STATS_DIR,
        help=f"Directory with per-host JSON files (default: {DEFAULT_STATS_DIR})",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Threshold preset: 500gb (ps1_process) or 128gb (mapping/downsample)",
    )
    parser.add_argument(
        "--min-mem-mb",
        type=int,
        help="Exclude hosts with MemAvailable below this many MB",
    )
    parser.add_argument(
        "--max-load1",
        type=float,
        help="Exclude hosts with 1-minute load above this value",
    )
    parser.add_argument(
        "--max-age-s",
        type=int,
        default=300,
        help="Exclude hosts whose heartbeat is older than this (default: 300)",
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
        help="For non-table formats, include OK hosts instead of only exclusions",
    )
    return parser


def resolve_thresholds(args: argparse.Namespace) -> tuple[int, float]:
    min_mem_mb = args.min_mem_mb
    max_load1 = args.max_load1
    if args.preset:
        preset = PRESETS[args.preset]
        if min_mem_mb is None:
            min_mem_mb = preset["min_mem_mb"]
        if max_load1 is None:
            max_load1 = preset["max_load1"]
    if min_mem_mb is None:
        min_mem_mb = PRESETS["500gb"]["min_mem_mb"]
    if max_load1 is None:
        max_load1 = PRESETS["500gb"]["max_load1"]
    return min_mem_mb, max_load1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    min_mem_mb, max_load1 = resolve_thresholds(args)

    samples = discover_samples(args.stats_dir)
    verdicts: list[tuple[str, HostVerdict]] = []
    for host in expected_hosts():
        sample = samples.get(host)
        verdict = evaluate_sample(
            sample,
            min_mem_mb=min_mem_mb,
            max_load1=max_load1,
            max_age_s=args.max_age_s,
        )
        verdicts.append((host, verdict))

    excluded = [host for host, verdict in verdicts if verdict.exclude]
    ok_hosts = [host for host, verdict in verdicts if not verdict.exclude]

    if args.format == "table":
        print(f"stats_dir: {args.stats_dir}")
        if not args.stats_dir.is_dir():
            print(
                f"WARNING: stats directory does not exist on this machine: {args.stats_dir}",
                file=sys.stderr,
            )
        print(format_table(verdicts))
        print()
        print(
            f"Thresholds: min_mem_mb={min_mem_mb}, max_load1={max_load1}, "
            f"max_age_s={args.max_age_s}"
        )
        print(f"Excluded: {len(excluded)}  OK: {len(ok_hosts)}")
        if excluded:
            print()
            print("requirements:")
            print(format_requirements(excluded))
        return 0

    hosts_out = expected_hosts() if args.include_ok else excluded
    if args.format == "requirements":
        print(format_requirements(hosts_out if args.include_ok else excluded))
    elif args.format == "bad-machines":
        print(format_bad_machines(hosts_out if args.include_ok else excluded))
    elif args.format == "hosts":
        for host in hosts_out if args.include_ok else excluded:
            print(host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
