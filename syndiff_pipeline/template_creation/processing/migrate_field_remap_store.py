"""Migrate legacy colocated field remap artifacts from templates/ to remap/.

Non-destructive migration: copies artifacts from the legacy templates store
into the dedicated remap store, verifies each destination file, and leaves the
source files in place. Safe to re-run (idempotent).

Legacy monolithic ``exact_cache/`` (L4b-lite pollution) is copied to
``exact_cache_legacy_polluted/`` and must not be used as clean L4a; rebuild
``exact_cache_l4a/`` via field_remap.
"""

from __future__ import annotations

import filecmp
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from syndiff_pipeline.common.scc_paths import scc_remap_dir, scc_templates_dir
from syndiff_pipeline.template_creation.processing.field_remap import (
    EXACT_CACHE_LEGACY_DIRNAME,
    EXACT_CACHE_LEGACY_POLLUTED_DIRNAME,
    REMAP_MANIFEST_NAME,
    REMAP_SCHEMA_VERSION,
)

log = logging.getLogger(__name__)

MIGRATION_NOTE = (
    "Legacy remap artifacts were copied from templates/ to remap/; "
    "source files were left in place for safety. "
    "Legacy exact_cache/ was archived under exact_cache_legacy_polluted/ "
    "and is not valid L4a; rebuild exact_cache_l4a/ via field_remap."
)

_REMAP_FILES = (
    "shift_schedule.npz",
    "shift_schedule.json",
    "template_group_shifts.parquet",
    "template_groups.json",
)


def _verify_file_copy(src: Path, dst: Path) -> None:
    if not dst.is_file():
        raise RuntimeError(f"copy missing at destination: {dst}")
    if not filecmp.cmp(src, dst, shallow=False):
        raise RuntimeError(f"copy verification failed: {src} -> {dst}")


def _copy_file_if_missing(src: Path, dst: Path) -> str:
    if not src.is_file():
        return "missing_at_source"
    if dst.is_file():
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    _verify_file_copy(src, dst)
    return "copied"


def _copy_tree_if_missing(src_dir: Path, dst_dir: Path) -> str:
    if not src_dir.is_dir():
        return "missing_at_source"
    if not any(src_dir.iterdir()):
        return "missing_at_source"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied_any = False
    for src_path in sorted(src_dir.rglob("*")):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        if dst_path.is_file():
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        _verify_file_copy(src_path, dst_path)
        copied_any = True
    if copied_any:
        return "copied"
    return "skipped"


def _write_migration_manifest(dest: Path, *, oversampling_factor: int) -> bool:
    manifest_path = dest / REMAP_MANIFEST_NAME
    if manifest_path.is_file():
        return False
    payload = {
        "schema_version": REMAP_SCHEMA_VERSION,
        "geometry_mode": "field",
        "oversampling_factor": int(oversampling_factor),
        "migrated": True,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return True


def migrate_scc_remap_artifacts(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
) -> dict[str, Any]:
    """Copy legacy L2–L4 remap artifacts from templates/ into remap/ for one SCC.

    Source: ``{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/``
    Dest:   ``{data_root}/s{SSSS}/c{C}/k{K}/remap/oversampling_{N}/``

    Copies (when present at source and missing at dest):

    - ``shift_schedule.npz``, ``shift_schedule.json``
    - ``template_group_shifts.parquet``, ``template_groups.json``
    - ``exact_cache/`` → ``exact_cache_legacy_polluted/`` (polluted; not L4a)

    Does **not** touch ``contribs/``, ``template_manifest.json``, or
    ``field_mode_assembly.json``. Uses copy-then-verify and leaves sources in
    place. Idempotent.
    """
    data_root = Path(data_root)
    source = scc_templates_dir(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    dest = scc_remap_dir(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []
    missing_at_source: list[str] = []

    for name in _REMAP_FILES:
        status = _copy_file_if_missing(source / name, dest / name)
        if status == "copied":
            copied.append(name)
        elif status == "skipped":
            skipped.append(name)
        else:
            missing_at_source.append(name)

    cache_status = _copy_tree_if_missing(
        source / EXACT_CACHE_LEGACY_DIRNAME,
        dest / EXACT_CACHE_LEGACY_POLLUTED_DIRNAME,
    )
    if cache_status == "copied":
        copied.append(f"{EXACT_CACHE_LEGACY_POLLUTED_DIRNAME}/")
    elif cache_status == "skipped":
        skipped.append(f"{EXACT_CACHE_LEGACY_POLLUTED_DIRNAME}/")
    else:
        missing_at_source.append(f"{EXACT_CACHE_LEGACY_DIRNAME}/")

    manifest_written = _write_migration_manifest(
        dest, oversampling_factor=oversampling_factor
    )

    if copied:
        log.info(
            "Migrated remap artifacts for s%04d_c%d_k%d os=%d: copied %s",
            sector,
            camera,
            ccd,
            oversampling_factor,
            ", ".join(copied),
        )

    return {
        "source": str(source),
        "dest": str(dest),
        "copied": copied,
        "skipped": skipped,
        "missing_at_source": missing_at_source,
        "manifest_written": manifest_written,
        "note": MIGRATION_NOTE,
    }
