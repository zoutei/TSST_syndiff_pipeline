"""Report-only garbage-collection analysis for provenance bookkeeping (PR6).

Walks fingerprint-addressed trees and compares against ``provenance.db`` without
deleting anything. Operators review the report before any destructive cleanup.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from syndiff_pipeline.common.scc_paths import (
    DIFF_SUBDIR,
    ps1_combined_zarr_path,
    ps1_convolved_zarr_path,
    provenance_db_path,
)

log = logging.getLogger(__name__)

__all__ = ["GcReport", "gc_report"]


@dataclass
class GcReport:
    db_artifacts: int = 0
    db_missing_files: list[str] = field(default_factory=list)
    orphan_fingerprint_dirs: list[str] = field(default_factory=list)
    diff_recipe_dirs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "db_artifacts": self.db_artifacts,
            "db_missing_files": self.db_missing_files,
            "orphan_fingerprint_dirs": self.orphan_fingerprint_dirs,
            "diff_recipe_dirs": self.diff_recipe_dirs,
            "errors": self.errors,
        }


def _fingerprint_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for proj in root.iterdir():
        if not proj.is_dir():
            continue
        for cell in proj.iterdir():
            if not cell.is_dir():
                continue
            for fp_dir in cell.iterdir():
                if fp_dir.is_dir() and not fp_dir.name.startswith("_tmp_"):
                    yield fp_dir


def _db_rows(db_path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    if not db_path.is_file():
        return set(), []
    fps: set[str] = set()
    locations: list[tuple[str, str]] = []
    try:
        conn = sqlite3.connect(str(db_path))
        for row in conn.execute("SELECT fingerprint, location FROM artifacts"):
            fps.add(str(row[0]))
            if row[1]:
                locations.append((str(row[0]), str(row[1])))
        conn.close()
    except Exception as exc:
        log.debug("gc: db read failed", exc_info=True)
        raise RuntimeError(f"db read failed: {exc}") from exc
    return fps, locations


def gc_report(data_root: str | Path) -> GcReport:
    """Build a report-only GC summary for one deployment ``data_root``."""
    data_root = Path(data_root).expanduser()
    report = GcReport()

    try:
        db_fps, locations = _db_rows(provenance_db_path(data_root))
    except RuntimeError as exc:
        report.errors.append(str(exc))
        db_fps, locations = set(), []

    report.db_artifacts = len(db_fps)
    for fp, location in locations:
        loc_path = Path(location).expanduser()
        if not loc_path.is_file() and not loc_path.is_dir():
            report.db_missing_files.append(f"{fp}:{location}")

    known_fps = set(db_fps)
    for store_root in (ps1_combined_zarr_path(data_root), ps1_convolved_zarr_path(data_root)):
        for fp_dir in _fingerprint_dirs(store_root):
            if fp_dir.name not in known_fps:
                report.orphan_fingerprint_dirs.append(str(fp_dir))

    for scc in data_root.glob("s*/c*/k*"):
        diff_root = scc / DIFF_SUBDIR
        if not diff_root.is_dir():
            continue
        for stage_dir in diff_root.iterdir():
            if not stage_dir.is_dir():
                continue
            for recipe_dir in stage_dir.iterdir():
                if recipe_dir.is_dir():
                    report.diff_recipe_dirs.append(str(recipe_dir))

    return report
