#!/usr/bin/env python3
"""Fetch TESSreduce comparison light curves for SynDiff pipeline targets.

Reads a targets CSV (same format as ``syndiff submit/run``), runs tessreduce per
enabled row, and writes viewer-compatible CSVs to a flat ``tessreduce_root``
directory (default: ``/astro/armin/koji/tessreduce_data``).

Output naming: ``{idx:04d}_SN{target_name}_s{sector}_tessreduce.csv``
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import tessreduce as tr

from syndiff_pipeline.common.orchestration.targets import Target, load_targets

DEFAULT_OUTPUT_DIR = Path("/astro/armin/koji/tessreduce_data")
TESSREDUCE_INDEX_RE = re.compile(r"^(\d{4})_SN.+_s\d+_tessreduce\.csv$", re.IGNORECASE)


def save_tessreduce_csv(tess, out_path: Path) -> None:
    """Save light curve CSV; tessreduce save_lc writes without .csv extension."""
    csv_base = str(out_path.with_suffix(""))
    tess.save_lc(csv_base)
    bare = Path(csv_base)
    if bare.exists() and bare != out_path:
        bare.rename(out_path)
    elif not out_path.exists():
        raise FileNotFoundError(f"Expected light-curve CSV at {out_path}")


def viewer_stem(target_name: str, sector: int) -> str:
    return f"SN{target_name.strip()}_s{sector}_tessreduce.csv"


def next_index(output_dir: Path, start_index: int | None) -> int:
    if start_index is not None:
        return start_index
    if not output_dir.is_dir():
        return 1
    indices = [
        int(m.group(1))
        for path in output_dir.glob("*_tessreduce.csv")
        if (m := TESSREDUCE_INDEX_RE.match(path.name))
    ]
    return (max(indices) + 1) if indices else 1


def find_existing_csv(output_dir: Path, target_name: str, sector: int) -> Path | None:
    if not output_dir.is_dir():
        return None
    needle = viewer_stem(target_name, sector).lower()
    for path in output_dir.glob("*_tessreduce.csv"):
        if path.name.lower().endswith(needle):
            return path
    return None


def run_tessreduce_for_target(
    target: Target,
    out_path: Path,
    *,
    num_cores: int,
) -> None:
    tess = tr.tessreduce(
        ra=target.target_ra,
        dec=target.target_dec,
        sector=target.sector,
        plot=False,
        diagnostic_plot=False,
        verbose=1,
        num_cores=num_cores,
        calibrate=False,
    )
    save_tessreduce_csv(tess, out_path)


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run tessreduce for SynDiff targets and write comparison CSVs "
            "for syndiff-review."
        )
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=repo_root / "config" / "targets_test.csv",
        help="SynDiff targets CSV (default: config/targets_test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Flat tessreduce CSV directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose viewer-matching CSV already exists",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=-1,
        help="joblib cores passed to tessreduce (default: -1 = all)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Leading 4-digit file prefix (default: max existing + 1)",
    )
    args = parser.parse_args(argv)

    targets = load_targets(args.targets)
    if not targets:
        print("fetch_tessreduce_for_targets: no enabled targets", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = next_index(output_dir, args.start_index)

    counts: dict[str, int] = {}
    for target in targets:
        existing = find_existing_csv(output_dir, target.target_name, target.sector)
        if args.skip_existing and existing is not None:
            counts["skipped"] = counts.get("skipped", 0) + 1
            print(f"[SKIP]  {target.label()} -> {existing.name} (exists)")
            continue

        out_path = output_dir / f"{idx:04d}_{viewer_stem(target.target_name, target.sector)}"
        print(f"[RUN]   {target.label()} -> {out_path.name}")
        try:
            run_tessreduce_for_target(target, out_path, num_cores=args.num_cores)
        except Exception as exc:  # noqa: BLE001
            counts["error"] = counts.get("error", 0) + 1
            print(f"[ERR]   {target.label()}: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue

        counts["ok"] = counts.get("ok", 0) + 1
        print(f"[OK]    {target.label()} -> {out_path}")
        idx += 1

    ok = counts.get("ok", 0)
    skipped = counts.get("skipped", 0)
    errors = counts.get("error", 0)
    print(
        f"fetch_tessreduce_for_targets: ok {ok}, skipped {skipped}, errors {errors}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
