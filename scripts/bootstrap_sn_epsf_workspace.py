#!/usr/bin/env python3
"""Bootstrap ws_multi_hp_temp_calib_epsf from existing workspace trees.

Symlinks missing multi_hp / ePSF artifacts into the target workspace so the
unified diff_config_sn_multi_hp_epsf pipeline can skip already-completed work.

Does not bootstrap hp_d from canonical ws/ (single-kernel diffs != multi_hp).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from syndiff_pipeline.common.orchestration.targets import Target, load_targets
from syndiff_pipeline.difference_imaging.support.paths import workspace_tree_name

TARGET_RUN_ID = "multi_hp_temp_calib_epsf"

SOURCE_RUN_IDS = (
    "multi_hp_temp_calib_epsf",
    "multi_hp_temp_calib",
    "multi_hp_nosmooth",
)

MULTI_HP_LABELS = (
    "kernel_fit",
    "tmpl_conv",
    "ks_d",
    "ks_b",
    "ks_b_s",
    "hp_d",
    "hp_m",
    "hp_b",
    "hp_c",
)

EPSF_LABELS = (
    "epsf_r1",
    "centroids_r1",
    "lc_gepsf_on_hp_diffs",
)

ROOT_ARTIFACTS = (
    "shared_mask.fits.fz",
    "shared_mask.fits.gz",
    "shared_mask.fits",
    "gaia_catalog_pipeline.csv",
    "hotpants_substamp_stars.csv",
    "targets.reg",
)

HP_D_LABEL = "hp_d"


def _hp_d_has_fits(ws_dir: Path) -> bool:
    hp_d = ws_dir / HP_D_LABEL
    if not hp_d.is_dir():
        return False
    for pat in ("*.fits.fz", "*.fits.gz", "*.fits"):
        if any(hp_d.glob(pat)):
            return True
    return False


def _resolve_root_artifact(event_dir: Path, ws_dir: Path, basename: str) -> Path | None:
    for candidate in (ws_dir / basename, event_dir / basename):
        if candidate.is_file():
            return candidate
    return None


def _ensure_symlink(child: Path, src: Path, *, dry_run: bool) -> bool:
    if child.exists() or child.is_symlink():
        return False
    print(f"  link {child.name} <- {src}")
    if dry_run:
        return True
    child.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, start=child.parent)
    os.symlink(rel, child)
    return True


def _source_workspaces(event_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for run_id in SOURCE_RUN_IDS:
        tree = workspace_tree_name(run_id)
        ws = event_dir / tree
        if ws.is_dir():
            out.append((run_id, ws))
    return out


def _pick_hp_d_source(sources: list[tuple[str, Path]]) -> Path | None:
    for _run_id, ws in sources:
        if _run_id == "multi_hp_temp_calib_epsf":
            continue
        if _hp_d_has_fits(ws):
            return ws
    return None


def bootstrap_event(event_dir: Path, *, dry_run: bool = False) -> int:
    """Return number of symlinks created."""
    event_dir = event_dir.resolve()
    child_ws = event_dir / workspace_tree_name(TARGET_RUN_ID)
    child_ws.mkdir(parents=True, exist_ok=True)

    sources = _source_workspaces(event_dir)
    if not sources:
        print(f"{event_dir.name}: no source workspaces found")
        return 0

    hp_d_source = _pick_hp_d_source(sources)
    created = 0

    def link_if_missing(label: str, src_ws: Path) -> None:
        nonlocal created
        dst = child_ws / label
        if dst.exists() or dst.is_symlink():
            return
        src = src_ws / label
        if not src.exists():
            return
        if _ensure_symlink(dst, src, dry_run=dry_run):
            created += 1

    for label in MULTI_HP_LABELS + EPSF_LABELS:
        if child_ws.joinpath(label).exists() or child_ws.joinpath(label).is_symlink():
            continue
        if label == HP_D_LABEL:
            if hp_d_source is not None:
                link_if_missing(label, hp_d_source)
            continue
        for _run_id, src_ws in sources:
            if src_ws.resolve() == child_ws.resolve():
                continue
            src = src_ws / label
            if src.exists():
                link_if_missing(label, src_ws)
                break

    for basename in ROOT_ARTIFACTS:
        dst = child_ws / basename
        if dst.exists() or dst.is_symlink():
            continue
        src_path: Path | None = None
        for _run_id, src_ws in sources:
            candidate = src_ws / basename
            if candidate.is_file():
                src_path = candidate
                break
        if src_path is None:
            src_path = _resolve_root_artifact(event_dir, child_ws, basename)
        if src_path is not None and _ensure_symlink(dst, src_path, dry_run=dry_run):
            created += 1

    print(f"{event_dir.name}: {created} symlink(s) created")
    return created


def _target_event_dir(events_root: Path, target: Target) -> Path:
    return events_root / "events" / target.label()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events-root",
        type=Path,
        required=True,
        help="Workspace root containing events/{label}/",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        required=True,
        help="Targets CSV (normalized header)",
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help="Optional single event label (e.g. s0023_c1_k3_2020ftl)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    events_root = args.events_root.expanduser().resolve()
    targets = load_targets(args.targets)
    if args.target_name:
        label = args.target_name.strip()
        targets = [t for t in targets if t.label() == label]
        if not targets:
            raise SystemExit(f"target not found in CSV: {label!r}")

    total = 0
    for target in targets:
        if not target.enabled:
            continue
        event_dir = _target_event_dir(events_root, target)
        if not event_dir.is_dir():
            print(f"{target.label()}: missing event dir {event_dir}", file=sys.stderr)
            continue
        total += bootstrap_event(event_dir, dry_run=args.dry_run)

    print(f"Done: {total} symlink(s) across {len(targets)} target(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
