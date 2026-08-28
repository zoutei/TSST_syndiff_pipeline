"""
reindex.py
==========
Offline rebuild of ``provenance.db`` from on-disk content (§5 invariant:
"the DB is a derived, rebuildable index"; §16 bootstrap; §18 testing:
"reindex == live index").

Two kinds of tree are walked:

1. **Self-describing shared stores** -- ``ps1_skycells_zarr/ps1_combined.zarr``
   and ``ps1_skycells_zarr/ps1_convolved.zarr``, each laid out
   ``{proj}/{skycell}/{fp}/`` with a ``_provenance.json`` written by
   :func:`syndiff_pipeline.common.provenance.publish.publish_dir`. When the
   sidecar is present and internally consistent (its own fingerprint
   recomputes to the directory name), the artifact is ingested verbatim,
   recipe and all.
2. **Legacy per-SCC trees** that predate this package and carry no sidecar:
   ``convolved.zarr`` (checkpoint-only, presence == complete),
   ``remap[/_{NAME}]/oversampling_{N}/remap_manifest.json``,
   ``templates[/_{NAME}]/oversampling_{N}/``, ``mapping/oversampling_{N}/``,
   ``diff[/_{NAME}]/{label}/``. These have
   no recoverable recipe, so per decision #8 they are ingested under
   ``{kind}_legacy_unverified`` with a synthetic, deterministic (idempotent)
   fingerprint tied to their on-disk path -- visible in queries, but never
   satisfying a ``scc_stage_complete`` check against a freshly-computed
   fingerprint (which always names a *_legacy_unverified-free kind). Named
   lanes include ``store_name`` in the spatial key so they do not collide
   with the default lane.

This module does filesystem walks by design (that is its whole job) -- it is
explicitly **offline only** and must never be imported on the scheduler hot
path (§10/§18: the fault-injection test proves the hot path never calls a
directory-walking probe; reindex is a separate, deliberate, operator-invoked
rebuild).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from syndiff_pipeline.common.provenance import model
from syndiff_pipeline.common.provenance.fingerprint import fingerprint as make_fingerprint
from syndiff_pipeline.common.provenance.fingerprint import recipe_id as make_recipe_id
from syndiff_pipeline.common.provenance.store import ProvenanceStore

log = logging.getLogger(__name__)

__all__ = [
    "ReindexResult",
    "REINDEX_CLEAR_PER_FFI_WARNING",
    "collect_reindex_clear_warnings",
    "reindex_shared_store",
    "reindex_scc_tree",
    "discover_scc_dirs",
    "reindex_data_root",
]

REINDEX_CLEAR_PER_FFI_WARNING = (
    "FULL REINDEX CLEARS provenance.db: per-FFI diff provenance (background, "
    "diff images, ePSF, masks) is spool-ingested only and will NOT be rebuilt "
    "from on-disk FITS. Drain bookkeeping/spool/ (supervisor ingest) before "
    "clearing, then re-emit diff runs for any lost rows."
)

_OVERSAMPLING_RE = re.compile(r"^oversampling_(\d+)$")
_SCC_DIR_RE = re.compile(r"^s(\d{4})/c(\d+)/k(\d+)$")

_LEGACY_CODE_VERSION = 0
_LEGACY_NOTE = "legacy artifact discovered at reindex; recipe not recoverable"


@dataclass
class ReindexResult:
    shared_store_artifacts: int = 0
    shared_store_legacy: int = 0
    scc_legacy_artifacts: int = 0
    sccs_scanned: int = 0
    errors: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def total_ingested(self) -> int:
        return self.shared_store_artifacts + self.shared_store_legacy + self.scc_legacy_artifacts


def _legacy_artifact(
    store: ProvenanceStore,
    kind: str,
    spatial_key: dict,
    location: str,
    *,
    disk_key: Optional[str] = None,
) -> str:
    """Ingest one legacy-unverified checkpoint artifact; returns its fingerprint.

    ``disk_key`` disambiguates multiple untracked directories that would
    otherwise collapse onto the same synthetic key (e.g. two stale recipe
    variants at the same skycell) -- it is folded into the recipe params so
    distinct on-disk content gets distinct rows, while still being
    deterministic/idempotent across repeated reindex runs.
    """
    legacy_kind = model.legacy_unverified_kind(kind)
    params = {"note": _LEGACY_NOTE}
    if disk_key is not None:
        params["disk_key"] = str(disk_key)
    rid = make_recipe_id(legacy_kind, params, _LEGACY_CODE_VERSION)
    fp = make_fingerprint(legacy_kind, spatial_key, rid, [])
    store.upsert_recipe(rid, legacy_kind, params, _LEGACY_CODE_VERSION)
    store.upsert_artifact(fp, legacy_kind, spatial_key, rid, location, "complete")
    return fp


def reindex_shared_store(
    store: ProvenanceStore,
    store_root: str | Path,
    kind: str,
) -> tuple[int, int]:
    """
    Walk ``store_root/{proj}/{skycell}/{fp}/_provenance.json`` and ingest.

    Returns ``(n_verified, n_legacy)``.
    """
    store_root = Path(store_root)
    if not store_root.is_dir():
        return (0, 0)

    n_verified = 0
    n_legacy = 0

    for proj_dir in sorted(p for p in store_root.iterdir() if p.is_dir()):
        for skycell_dir in sorted(p for p in proj_dir.iterdir() if p.is_dir()):
            for fp_dir in sorted(p for p in skycell_dir.iterdir() if p.is_dir()):
                disk_fp = fp_dir.name
                sidecar = fp_dir / "_provenance.json"
                spatial_key = {"projection": proj_dir.name, "skycell": skycell_dir.name}
                if sidecar.is_file():
                    try:
                        record = json.loads(sidecar.read_text(encoding="utf-8"))
                        recomputed = make_fingerprint(
                            record["kind"],
                            record["spatial_key"],
                            record["recipe_id"],
                            record.get("inputs", ()),
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("reindex: unreadable sidecar %s: %s", sidecar, exc)
                        record = None
                        recomputed = None
                else:
                    record = None
                    recomputed = None

                if (
                    record is not None
                    and recomputed == disk_fp
                    and record.get("kind") == kind
                ):
                    recipe_params = record.get("recipe_params")
                    if recipe_params is not None:
                        store.upsert_recipe(
                            record["recipe_id"],
                            record["kind"],
                            recipe_params,
                            int(record.get("code_version", 0)),
                            git_sha=record.get("git_sha"),
                            created_at=record.get("created_at"),
                        )
                    store.upsert_artifact(
                        disk_fp,
                        record["kind"],
                        record["spatial_key"],
                        record["recipe_id"],
                        record.get("location", str(fp_dir)),
                        record.get("state", "complete"),
                        bytes_=record.get("bytes"),
                        wall_time_s=record.get("wall_time_s"),
                        produced_by=record.get("produced_by"),
                        created_at=record.get("created_at"),
                    )
                    store.add_edges(disk_fp, record.get("inputs", ()))
                    n_verified += 1
                else:
                    _legacy_artifact(store, kind, spatial_key, str(fp_dir), disk_key=disk_fp)
                    n_legacy += 1

    return (n_verified, n_legacy)


def discover_scc_dirs(data_root: str | Path) -> list[tuple[int, int, int, Path]]:
    """Shallow (three-level) glob for ``s{SSSS}/c{C}/k{K}`` dirs under *data_root*.

    Not the O(cells) scan the plan retires -- this walks one directory listing
    per level to find which SCCs exist at all, which is unavoidable for an
    offline full rebuild and is never called on the scheduler hot path.
    """
    data_root = Path(data_root)
    out: list[tuple[int, int, int, Path]] = []
    if not data_root.is_dir():
        return out
    for s_dir in sorted(data_root.glob("s[0-9][0-9][0-9][0-9]")):
        try:
            sector = int(s_dir.name[1:])
        except ValueError:
            continue
        for c_dir in sorted(s_dir.glob("c[0-9]*")):
            try:
                camera = int(c_dir.name[1:])
            except ValueError:
                continue
            for k_dir in sorted(c_dir.glob("k[0-9]*")):
                try:
                    ccd = int(k_dir.name[1:])
                except ValueError:
                    continue
                out.append((sector, camera, ccd, k_dir))
    return out


def _oversampling_children(parent: Path) -> list[tuple[int, Path]]:
    if not parent.is_dir():
        return []
    out = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        m = _OVERSAMPLING_RE.match(child.name)
        if m:
            out.append((int(m.group(1)), child))
    return out


def _diff_kind_from_workspace_label(workspace_label: str) -> str:
    """Best-effort kind guess for a legacy SCC diff recipe directory."""
    label = str(workspace_label).strip().lower()
    if "shared_mask" in label or label in {"mask", "sharedmask"}:
        return "shared_mask"
    if (
        "bkg" in label
        or label.endswith("_bg")
        or label.endswith("_b")
        or label.endswith("_b_s")
        or label in {"hp_b", "ks_b", "ks_b_s"}
    ):
        return "diff_background"
    if "epsf" in label or label.startswith("gepsf"):
        return "epsf"
    return "diff_image"


def collect_reindex_clear_warnings(data_root: str | Path) -> list[str]:
    """Return operator warnings to emit before a full (non-incremental) reindex."""
    from syndiff_pipeline.common.scc_paths import provenance_spool_dir

    warnings = [REINDEX_CLEAR_PER_FFI_WARNING]
    spool_dir = provenance_spool_dir(data_root)
    if spool_dir.is_dir():
        live = sorted(spool_dir.glob("*.jsonl"))
        if live:
            warnings.append(
                f"Undrained spool files under {spool_dir} ({len(live)} file(s)); "
                "drain into provenance.db before clearing or spool-only per-FFI "
                "rows will be lost."
            )
    return warnings


def _diff_label_dir_has_content(label_dir: Path) -> bool:
    if not label_dir.is_dir():
        return False
    return any(
        child.is_file() and not child.name.startswith("_tmp_") for child in label_dir.iterdir()
    )


def _store_lane_roots(scc_dir: Path, base: str) -> list[tuple[str | None, Path]]:
    """Return ``[(store_name, root)]`` for ``base/`` and ``base_{NAME}/`` siblings."""
    out: list[tuple[str | None, Path]] = []
    default = scc_dir / base
    if default.is_dir():
        out.append((None, default))
    prefix = f"{base}_"
    if not scc_dir.is_dir():
        return out
    for child in sorted(scc_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        name = child.name[len(prefix) :]
        if name:
            out.append((name, child))
    return out


def reindex_scc_tree(store: ProvenanceStore, scc_dir: str | Path, s: int, c: int, k: int) -> int:
    """
    Legacy-marker sweep of one SCC directory: ``convolved.zarr`` presence,
    ``remap[/_{NAME}]/oversampling_{N}/remap_manifest.json``,
    ``templates[/_{NAME}]/oversampling_{N}/``, ``mapping/oversampling_{N}/``,
    ``diff[/_{NAME}]/{label}/``.

    Returns the count of legacy_unverified artifacts ingested.
    """
    from syndiff_pipeline.common.scc_paths import (
        CONVOLVED_ZARR_BASENAME,
        DIFF_SUBDIR,
        EVENTS_SUBDIR,
        MAPPING_SUBDIR,
        REMAP_SUBDIR,
        TEMPLATES_SUBDIR,
    )

    scc_dir = Path(scc_dir)
    n = 0

    convolved = scc_dir / CONVOLVED_ZARR_BASENAME
    if convolved.exists():
        _legacy_artifact(store, "scc_assembly", {"s": s, "c": c, "k": k}, str(convolved))
        n += 1

    for os_val, mapping_dir in _oversampling_children(scc_dir / MAPPING_SUBDIR):
        if any(mapping_dir.iterdir()) if mapping_dir.is_dir() else False:
            _legacy_artifact(
                store, "mapping", {"s": s, "c": c, "k": k, "os": os_val}, str(mapping_dir)
            )
            n += 1

    for store_name, remap_root in _store_lane_roots(scc_dir, REMAP_SUBDIR):
        for os_val, remap_dir in _oversampling_children(remap_root):
            manifest = remap_dir / "remap_manifest.json"
            if not manifest.is_file():
                continue
            spatial = {"s": s, "c": c, "k": k, "os": os_val}
            if store_name is not None:
                spatial["store_name"] = store_name
            _legacy_artifact(store, "remap_store", spatial, str(remap_dir))
            n += 1

    for store_name, templates_root in _store_lane_roots(scc_dir, TEMPLATES_SUBDIR):
        for os_val, templates_dir in _oversampling_children(templates_root):
            has_content = templates_dir.is_dir() and any(templates_dir.iterdir())
            if not has_content:
                continue
            spatial = {"s": s, "c": c, "k": k, "os": os_val}
            if store_name is not None:
                spatial["store_name"] = store_name
            _legacy_artifact(store, "downsample", spatial, str(templates_dir))
            n += 1

    for store_name, diff_root in _store_lane_roots(scc_dir, DIFF_SUBDIR):
        if not diff_root.is_dir():
            continue
        for label_dir in sorted(p for p in diff_root.iterdir() if p.is_dir()):
            if label_dir.name == EVENTS_SUBDIR or label_dir.name.startswith("_tmp_"):
                continue
            if not _diff_label_dir_has_content(label_dir):
                continue
            workspace_label = label_dir.name
            kind = _diff_kind_from_workspace_label(workspace_label)
            if kind == "shared_mask":
                spatial = {"s": s, "c": c, "k": k}
            else:
                spatial = {
                    "s": s,
                    "c": c,
                    "k": k,
                    "workspace_label": workspace_label,
                }
            if store_name is not None:
                spatial["store_name"] = store_name
            _legacy_artifact(
                store,
                kind,
                spatial,
                str(label_dir),
                disk_key=workspace_label,
            )
            n += 1

    return n


def reindex_data_root(
    data_root: str | Path,
    store: ProvenanceStore,
    *,
    scc_dirs: Optional[Iterable[tuple[int, int, int, Path]]] = None,
    clear_first: bool = False,
) -> ReindexResult:
    """
    Full offline rebuild: shared stores + every discovered SCC's legacy sweep.

    Parameters
    ----------
    data_root : path
        Pipeline data root (contains ``ps1_skycells_zarr/`` and ``s*/c*/k*/``).
    store : ProvenanceStore
        Destination index (typically freshly created / ``clear_first=True``
        for a full rebuild; left alone for an incremental top-up).
    scc_dirs : iterable of (sector, camera, ccd, path), optional
        Override SCC discovery (tests build a synthetic tree without a full
        ``discover_scc_dirs`` glob). Defaults to ``discover_scc_dirs(data_root)``.
    clear_first : bool
        Wipe the store before rebuilding (true "reindex from scratch").
    """
    from syndiff_pipeline.common.scc_paths import (
        ps1_combined_zarr_path,
        ps1_convolved_zarr_path,
    )

    if clear_first:
        for msg in collect_reindex_clear_warnings(data_root):
            log.warning("reindex: %s", msg)
        store.clear()

    result = ReindexResult()

    combined_root = ps1_combined_zarr_path(data_root)
    n_ok, n_legacy = reindex_shared_store(store, combined_root, "combined_skycell")
    result.shared_store_artifacts += n_ok
    result.shared_store_legacy += n_legacy

    convolved_root = ps1_convolved_zarr_path(data_root)
    n_ok, n_legacy = reindex_shared_store(store, convolved_root, "convolved_skycell")
    result.shared_store_artifacts += n_ok
    result.shared_store_legacy += n_legacy

    sccs = list(scc_dirs) if scc_dirs is not None else discover_scc_dirs(data_root)
    for sector, camera, ccd, scc_dir in sccs:
        try:
            result.scc_legacy_artifacts += reindex_scc_tree(store, scc_dir, sector, camera, ccd)
        except Exception as exc:  # noqa: BLE001
            msg = f"reindex: failed on SCC {sector}/{camera}/{ccd} ({scc_dir}): {exc}"
            log.error(msg, exc_info=True)
            result.errors.append(msg)
        result.sccs_scanned += 1

    return result
