#!/usr/bin/env python3
"""Migrate SynDiff FITS products from ``.fits.gz`` to CFITSIO fpack ``.fits.fz``.

Scopes:
  tess_ffi   — data_root/tess_ffi and SCC …/ffi/ leaves
  data_root  — templates/, mapping/ regmaps, master_pixels2skycells
  workspace  — events/*/ws/** workspace FITS trees

Example::

    mamba activate syndiff
    python scripts/fpack_fits_migration.py --scope tess_ffi --dry-run --limit 5
    python scripts/fpack_fits_migration.py --scope data_root --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from astropy.io import fits

from syndiff_pipeline.common.fits_io import fpack_plain_fits
from syndiff_pipeline.common.fits_variants import fits_fpack_path, fits_logical_path

log = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("/astro/armin/koji/syndiff/data")
DEFAULT_WORKSPACE_ROOT = Path("/astro/armin/koji/syndiff/workspace")
DEFAULT_LOG_DIR = Path("/astro/armin/koji/syndiff/logs/ffi_fpack")

SCOPES = ("tess_ffi", "data_root", "workspace")


@dataclass
class RunCounts:
    total: int = 0
    ok: int = 0
    skip: int = 0
    fail: int = 0
    bytes_before: int = 0
    bytes_after: int = 0


@dataclass
class RunState:
    scope: str
    dry_run: bool
    log_dir: Path
    counts: RunCounts = field(default_factory=RunCounts)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_mono: float = field(default_factory=time.monotonic)
    failure_log_path: Path | None = None


def _setup_logging(log_dir: Path, scope: str, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"fpack_{scope}.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO if verbose else logging.WARNING)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return log_path


def _collect_gz_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            log.warning("skip missing root: %s", root)
            continue
        for path in sorted(root.rglob("*.fits.gz")):
            if path.is_file():
                out.append(path)
    return out


def _roots_for_scope(
    scope: str, *, data_root: Path, workspace_root: Path
) -> list[Path]:
    if scope == "tess_ffi":
        roots = [data_root / "tess_ffi"]
        # SCC-nested ffi leaves: sNNNN/cC/kK/ffi
        for ffi in data_root.glob("s*/c*/k*/ffi"):
            if ffi.is_dir():
                roots.append(ffi)
        return roots
    if scope == "data_root":
        return [
            data_root / "shifted_downsampled",
            data_root / "skycell_pixel_mapping",
            data_root / "templates",
            data_root / "mapping",
        ]
    if scope == "workspace":
        return [workspace_root / "events"]
    raise ValueError(f"unknown scope: {scope}")


def _gunzip_to_plain(gz_path: Path, plain_path: Path) -> None:
    part = plain_path.with_suffix(plain_path.suffix + ".part")
    try:
        with gzip.open(gz_path, "rb") as f_in, open(part, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.replace(part, plain_path)
    except BaseException:
        if part.is_file():
            try:
                part.unlink()
            except OSError:
                pass
        raise


def _convert_one(gz_path: Path, *, dry_run: bool) -> tuple[str, Path | None]:
    """Return (status, fz_path). status in ok|skip|fail."""
    fz_path = Path(fits_fpack_path(gz_path))
    if fz_path.is_file():
        return "skip", fz_path
    if dry_run:
        return "ok", fz_path

    plain = Path(fits_logical_path(gz_path))
    # Work in a temp dir beside the target to stay on the same filesystem.
    tmpdir = Path(
        tempfile.mkdtemp(prefix=".fpack_mig.", dir=str(gz_path.parent))
    )
    try:
        tmp_plain = tmpdir / plain.name
        _gunzip_to_plain(gz_path, tmp_plain)
        # Verify readable before fpack
        with fits.open(tmp_plain, memmap=False) as hdul:
            _ = len(hdul)
        produced = fpack_plain_fits(tmp_plain, delete_plain=True)
        if produced.resolve() != fz_path.resolve():
            if fz_path.is_file():
                fz_path.unlink()
            shutil.move(str(produced), str(fz_path))
        with fits.open(fz_path, memmap=False) as hdul:
            _ = len(hdul)
        gz_path.unlink()
        return "ok", fz_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        required=True,
        choices=SCOPES,
        help="Which tree to migrate",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Science data root",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
        help="Workspace root (events/)",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Max files (0=all)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    data_root = args.data_root.expanduser().resolve()
    workspace_root = args.workspace_root.expanduser().resolve()
    log_path = _setup_logging(args.log_dir.expanduser().resolve(), args.scope, args.verbose)
    state = RunState(scope=args.scope, dry_run=args.dry_run, log_dir=args.log_dir)
    state.failure_log_path = state.log_dir / f"fpack_{args.scope}_failures.log"

    roots = _roots_for_scope(
        args.scope, data_root=data_root, workspace_root=workspace_root
    )
    files = _collect_gz_files(roots)
    if args.limit > 0:
        files = files[: args.limit]
    state.counts.total = len(files)
    log.info(
        "scope=%s files=%d dry_run=%s log=%s",
        args.scope,
        len(files),
        args.dry_run,
        log_path,
    )

    for gz_path in files:
        try:
            before = gz_path.stat().st_size
            status, fz = _convert_one(gz_path, dry_run=args.dry_run)
            if status == "skip":
                state.counts.skip += 1
                log.info("skip (fz exists): %s", gz_path)
            elif status == "ok":
                state.counts.ok += 1
                state.counts.bytes_before += before
                if fz is not None and fz.is_file() and not args.dry_run:
                    state.counts.bytes_after += fz.stat().st_size
                log.info("%s -> %s", gz_path, fz)
            else:
                state.counts.fail += 1
        except Exception as exc:
            state.counts.fail += 1
            log.exception("fail %s: %s", gz_path, exc)
            if state.failure_log_path is not None:
                with state.failure_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"{gz_path}\t{exc}\n")

    log.info(
        "done scope=%s ok=%d skip=%d fail=%d elapsed=%.1fs",
        args.scope,
        state.counts.ok,
        state.counts.skip,
        state.counts.fail,
        time.monotonic() - state.started_mono,
    )
    return 1 if state.counts.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
