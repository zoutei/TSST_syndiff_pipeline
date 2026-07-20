#!/usr/bin/env python3
"""Remove legacy syndiff template FITS when a preferred compressed sibling exists.

Prefers ``.fits.fz`` over ``.fits.gz``. Removes plain ``.fits`` (and gzip when
fpack exists) for the same syndiff_template offset key.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from syndiff_pipeline.common.fits_variants import (
    FITS_FPACK_EXT,
    FITS_GZIP_EXT,
    FITS_PLAIN_EXT,
    storage_suffix_rank,
)
from syndiff_pipeline.common.orchestration.deployment import load_deployment
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    parse_syndiff_template_filename,
)


def _offset_key(parsed) -> tuple[int, int, int, float, float]:
    return (
        parsed.sector,
        parsed.camera,
        parsed.ccd,
        round(float(parsed.dx), 6),
        round(float(parsed.dy), 6),
    )


def _find_stale_fits(template_dir: Path) -> list[Path]:
    """Return lower-preference paths when a better storage variant exists."""
    by_offset: dict[tuple, list[Path]] = defaultdict(list)
    for entry in sorted(template_dir.iterdir()):
        if not entry.is_file():
            continue
        parsed = parse_syndiff_template_filename(str(entry))
        if parsed is None:
            continue
        by_offset[_offset_key(parsed)].append(entry)

    to_remove: list[Path] = []
    for paths in by_offset.values():
        best = min(paths, key=lambda p: storage_suffix_rank(p.name))
        best_rank = storage_suffix_rank(best.name)
        if best_rank >= storage_suffix_rank(f"x{FITS_PLAIN_EXT}"):
            continue
        for p in paths:
            if p == best:
                continue
            rank = storage_suffix_rank(p.name)
            # Drop plain always when compressed exists; drop gz when fz exists.
            if rank > best_rank:
                to_remove.append(p)
    return sorted(to_remove)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete legacy syndiff_template_*.fits / *.fits.gz when a preferred "
            ".fits.fz (or .fits.gz) sibling exists under shifted_downsampled/."
        )
    )
    parser.add_argument(
        "--deployment",
        default="deployment.yaml",
        help="Deployment YAML with data_root (default: deployment.yaml)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Override template root (default: {data_root}/shifted_downsampled)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be deleted without removing them",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete files (required unless --dry-run)",
    )
    args = parser.parse_args(argv)

    if args.root:
        scan_root = Path(args.root).expanduser().resolve()
    else:
        dep = load_deployment(args.deployment)
        data_root = Path(dep["data_root"]).expanduser().resolve()
        scan_root = data_root / "shifted_downsampled"

    if not scan_root.is_dir():
        print(f"cleanup_duplicate_templates: root not found: {scan_root}", file=sys.stderr)
        return 1

    if not args.dry_run and not args.force:
        print(
            "cleanup_duplicate_templates: pass --force to delete files, or --dry-run to preview",
            file=sys.stderr,
        )
        return 2

    scanned_dirs = 0
    removed = 0
    for template_dir in sorted(p for p in scan_root.rglob("sector*_camera*_ccd*") if p.is_dir()):
        scanned_dirs += 1
        for path in _find_stale_fits(template_dir):
            print(path)
            removed += 1
            if not args.dry_run:
                path.unlink(missing_ok=True)

    print(
        f"cleanup_duplicate_templates: scanned {scanned_dirs} dir(s), "
        f"{'would remove' if args.dry_run else 'removed'} {removed} file(s) "
        f"(prefer {FITS_FPACK_EXT} > {FITS_GZIP_EXT} > {FITS_PLAIN_EXT})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
