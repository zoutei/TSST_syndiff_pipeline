"""
SCC and event workspace path conventions for the storage-first layout.

SCC-scoped template artifacts live under ``{data_root}/s{SSSS}/c{C}/k{K}/``.
Event-scoped diff/star workspaces live under
``{workspace_root}/events/{event_name}/s{SSSS}_c{C}_k{K}/``.

Path helpers return :class:`~pathlib.Path` objects and do **not** create
directories unless a helper explicitly documents ``mkdir`` behavior.
"""

from __future__ import annotations

import re
from pathlib import Path

EVENTS_SUBDIR = "events"

FFI_SUBDIR = "ffi"
CATALOGS_SUBDIR = "catalogs"
MAPPING_SUBDIR = "mapping"
REMAP_SUBDIR = "remap"
TEMPLATES_SUBDIR = "templates"
DEBUG_PLOTS_SUBDIR = "debug_plots"
LEGACY_SUBDIR = "legacy"
BOOKKEEPING_SUBDIR = "bookkeeping"
DIFF_SUBDIR = "diff"

# Named store lanes: templates_{NAME}/, remap_{NAME}/. Empty/None → base subdir.
_STORE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

CONVOLVED_ZARR_BASENAME = "convolved.zarr"
CONVOLVED_REMOVED_STARS_CSV_BASENAME = "convolved_removed_stars.csv"
FFI_LIST_PARQUET_BASENAME = "ffi_list.parquet"
FFI_LIST_CSV_BASENAME = "ffi_list.csv"

# Shared PS1 raw-band Zarr store (ps1_download + syndiff star).
PS1_SKYCELLS_ZARR_DIRNAME = "ps1_skycells_zarr"
PS1_SKYCELLS_ZARR_BASENAME = "ps1_skycells.zarr"

# Sky-keyed shared combined/convolved PS1 stores (provenance plan §7, decision
# #14): both live alongside ps1_skycells.zarr under the same
# ps1_skycells_zarr/ directory. Phase 1/2 writers/readers land in later PRs;
# these are path constants only.
PS1_COMBINED_ZARR_BASENAME = "ps1_combined.zarr"
PS1_CONVOLVED_ZARR_BASENAME = "ps1_convolved.zarr"

# Provenance bookkeeping tree (provenance plan §7/§8): data_root-scoped, not
# SCC-scoped -- ``{data_root}/bookkeeping/provenance.db`` + a per-host/pid
# spool of sidecar JSONL files the supervisor drains.
PROVENANCE_DB_BASENAME = "provenance.db"
PROVENANCE_SPOOL_SUBDIR = "spool"

__all__ = [
    "BOOKKEEPING_SUBDIR",
    "DEBUG_PLOTS_SUBDIR",
    "DIFF_SUBDIR",
    "CATALOGS_SUBDIR",
    "CONVOLVED_REMOVED_STARS_CSV_BASENAME",
    "CONVOLVED_ZARR_BASENAME",
    "EVENTS_SUBDIR",
    "FFI_SUBDIR",
    "LEGACY_SUBDIR",
    "MAPPING_SUBDIR",
    "REMAP_SUBDIR",
    "PS1_SKYCELLS_ZARR_BASENAME",
    "PS1_SKYCELLS_ZARR_DIRNAME",
    "TEMPLATES_SUBDIR",
    "FFI_LIST_CSV_BASENAME",
    "FFI_LIST_PARQUET_BASENAME",
    "event_root",
    "event_scc_leaf",
    "normalize_store_name",
    "oversampling_dirname",
    "mapping_sector_tree_root",
    "mapping_write_dir",
    "ps1_combined_zarr_path",
    "ps1_convolved_zarr_path",
    "ps1_skycells_zarr_dir",
    "ps1_skycells_zarr_lock_path",
    "ps1_skycells_zarr_path",
    "provenance_db_path",
    "provenance_spool_dir",
    "scc_bookkeeping_dir",
    "scc_bookkeeping_stage_dir",
    "scc_catalogs_dir",
    "scc_convolved_removed_stars_csv",
    "scc_convolved_zarr",
    "scc_debug_plots_dir",
    "scc_diff_pipeline_plots_dir",
    "scc_diff_dir",
    "scc_diff_label_dir",
    "scc_diff_workspace_dir",
    "scc_diff_event_dir",
    "scc_diff_bookkeeping_dir",
    "legacy_scc_diff_bookkeeping_dir",
    "resolve_scc_diff_bookkeeping_dir",
    "scc_ffi_dir",
    "scc_label",
    "scc_legacy_dir",
    "scc_mapping_dir",
    "scc_mapping_master_pixels2skycells",
    "scc_mapping_master_skycells_csv",
    "scc_remap_dir",
    "scc_root",
    "scc_templates_dir",
    "scc_ffi_list_csv",
    "scc_ffi_list_parquet",
    "store_subdir",
]


def scc_label(sector: int, camera: int, ccd: int) -> str:
    """Return the canonical SCC filesystem label ``s{SSSS}_c{C}_k{K}``."""
    return f"s{int(sector):04d}_c{int(camera)}_k{int(ccd)}"


def oversampling_dirname(oversampling_factor: int) -> str:
    """Directory name for one oversampling factor, e.g. ``oversampling_2``."""
    return f"oversampling_{int(oversampling_factor)}"


def normalize_store_name(store_name: str | None) -> str | None:
    """Return a validated store lane name, or ``None`` for the default lane.

    Empty / whitespace-only names become ``None``. Non-empty names must match
    ``^[A-Za-z0-9][A-Za-z0-9_-]*$`` (no path separators).
    """
    if store_name is None:
        return None
    name = str(store_name).strip()
    if not name:
        return None
    if not _STORE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid store_name {store_name!r}: must match "
            r"^[A-Za-z0-9][A-Za-z0-9_-]*$ (no path separators)"
        )
    return name


def store_subdir(base: str, store_name: str | None) -> str:
    """Return ``base`` or ``base_{name}`` for a named store lane."""
    name = normalize_store_name(store_name)
    return base if name is None else f"{base}_{name}"


def mapping_sector_tree_root(output_path: str | Path, oversampling_factor: int) -> Path:
    """Resolve the oversampling leaf directory for mapping outputs.

    Pipeline dispatch passes ``scc_mapping_dir`` (already ``…/mapping/oversampling_N``).
    Legacy callers pass the parent ``…/mapping`` directory.
    """
    root = Path(output_path).expanduser().resolve()
    os_n = int(oversampling_factor)
    if os_n <= 1:
        return root
    os_dir = oversampling_dirname(os_n)
    if root.name == os_dir:
        return root
    return root / os_dir


def mapping_write_dir(
    output_path: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
) -> Path:
    """Directory where mapping FITS and CSV files are written.

    Pipeline runs (``output_path`` already ``…/mapping/oversampling_N``) use a flat
    layout matching :func:`scc_mapping_dir` — same as existing OS1 data.

    Legacy standalone pancakes callers keep the ``sector_*/camera_*/ccd_*`` tree.
    """
    root = Path(output_path).expanduser().resolve()
    os_n = int(oversampling_factor)
    os_dir = oversampling_dirname(os_n)
    if os_n >= 1 and root.name == os_dir:
        return root
    base = mapping_sector_tree_root(output_path, oversampling_factor)
    return (
        base
        / f"sector_{int(sector):04d}"
        / f"camera_{int(camera)}"
        / f"ccd_{int(ccd)}"
    )


def scc_root(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Root directory for one SCC under ``data_root``: ``s{SSSS}/c{C}/k{K}/``."""
    return (
        Path(data_root).expanduser()
        / f"s{int(sector):04d}"
        / f"c{int(camera)}"
        / f"k{int(ccd)}"
    )


def scc_ffi_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Directory for downloaded FFIs shared across oversampling factors."""
    return scc_root(data_root, sector, camera, ccd) / FFI_SUBDIR


def scc_catalogs_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Directory for SCC-scoped source catalogs."""
    return scc_root(data_root, sector, camera, ccd) / CATALOGS_SUBDIR


def scc_convolved_zarr(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Path to the shared PS1 convolved Zarr store for one SCC."""
    return scc_root(data_root, sector, camera, ccd) / CONVOLVED_ZARR_BASENAME


def scc_convolved_removed_stars_csv(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Path to removed-star bookkeeping for the convolved store."""
    return (
        scc_root(data_root, sector, camera, ccd)
        / CONVOLVED_REMOVED_STARS_CSV_BASENAME
    )


def scc_ffi_list_parquet(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Path to the shared per-SCC FFI header inventory (Parquet)."""
    return scc_root(data_root, sector, camera, ccd) / FFI_LIST_PARQUET_BASENAME


def scc_ffi_list_csv(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Path to the slim CSV twin of the shared FFI list."""
    return scc_root(data_root, sector, camera, ccd) / FFI_LIST_CSV_BASENAME


def scc_mapping_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int,
) -> Path:
    """Directory for skycell pixel mapping at one oversampling factor."""
    return (
        scc_root(data_root, sector, camera, ccd)
        / MAPPING_SUBDIR
        / oversampling_dirname(oversampling_factor)
    )


def scc_remap_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int,
    store_name: str | None = None,
) -> Path:
    """Directory for field remap artifacts at one oversampling factor.

    Default lane: ``remap/oversampling_{N}/``. Named lane:
    ``remap_{store_name}/oversampling_{N}/``.
    """
    return (
        scc_root(data_root, sector, camera, ccd)
        / store_subdir(REMAP_SUBDIR, store_name)
        / oversampling_dirname(oversampling_factor)
    )


def scc_templates_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int,
    store_name: str | None = None,
) -> Path:
    """Directory for full-chip template products at one oversampling factor.

    Default lane: ``templates/oversampling_{N}/``. Named lane:
    ``templates_{store_name}/oversampling_{N}/``.
    """
    return (
        scc_root(data_root, sector, camera, ccd)
        / store_subdir(TEMPLATES_SUBDIR, store_name)
        / oversampling_dirname(oversampling_factor)
    )


def scc_debug_plots_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """SCC-scoped template-pipeline diagnostics directory (``debug_plots/``)."""
    return scc_root(data_root, sector, camera, ccd) / DEBUG_PLOTS_SUBDIR


def scc_diff_pipeline_plots_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    category: str,
    *,
    store_name: str | None = None,
) -> Path:
    """Diff-lane diagnostic subdirectory under ``diff/debug_plots/{category}/``.

    Examples: ``masks``, ``background``, ePSF labels (``epsf_r1``). Named lanes
    use ``diff_{store_name}/debug_plots/{category}/``. Template-pipeline plots
    stay under :func:`scc_debug_plots_dir`.
    """
    cat = str(category).strip()
    if not cat:
        raise ValueError("category must be non-empty")
    return (
        scc_diff_dir(data_root, sector, camera, ccd, store_name=store_name)
        / DEBUG_PLOTS_SUBDIR
        / cat
    )


def scc_legacy_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Directory for archived pre-cutover SCC artifacts (never read live)."""
    return scc_root(data_root, sector, camera, ccd) / LEGACY_SUBDIR


def scc_bookkeeping_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Directory for SCC template-pipeline stage bookkeeping."""
    return scc_root(data_root, sector, camera, ccd) / BOOKKEEPING_SUBDIR


def scc_bookkeeping_stage_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    stage: str,
) -> Path:
    """Directory for one template stage's bookkeeping under an SCC."""
    return scc_bookkeeping_dir(data_root, sector, camera, ccd) / str(stage).strip()


def scc_diff_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    store_name: str | None = None,
) -> Path:
    """SCC diff lane root: ``diff/`` or ``diff_{store_name}/``."""
    return scc_root(data_root, sector, camera, ccd) / store_subdir(DIFF_SUBDIR, store_name)


def scc_diff_label_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    store_name: str | None,
    label: str,
) -> Path:
    """Per-label directory under a diff lane, e.g. ``diff_{lane}/hp_d/``."""
    return (
        scc_diff_dir(data_root, sector, camera, ccd, store_name=store_name)
        / str(label).strip()
    )


def scc_diff_workspace_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    store_name: str | None,
    workspace_label: str,
) -> Path:
    """Alias for :func:`scc_diff_label_dir` (workspace label = lane subdirectory)."""
    return scc_diff_label_dir(
        data_root,
        sector,
        camera,
        ccd,
        store_name=store_name,
        label=workspace_label,
    )


def scc_diff_event_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    store_name: str | None,
    event_name: str,
) -> Path:
    """Event-scoped photometry subtree under a diff lane."""
    return (
        scc_diff_dir(data_root, sector, camera, ccd, store_name=store_name)
        / EVENTS_SUBDIR
        / str(event_name).strip()
    )


def scc_diff_bookkeeping_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
    template_store_name: str | None = None,
) -> Path:
    """Bookkeeping dir for one template lane: ``bookkeeping/diff[_NAME]/oversampling_{N}/``.

    Mirrors the ``templates``/``remap`` lane layout (:func:`scc_templates_dir`)
    so diff handoff bookkeeping partitions by ``(template_store_name,
    oversampling_factor)`` instead of a single flat directory. Always the
    *write* path — canonical bookkeeping location for diff handoff artifacts.
    """
    return (
        scc_bookkeeping_dir(data_root, sector, camera, ccd)
        / store_subdir(DIFF_SUBDIR, template_store_name)
        / oversampling_dirname(oversampling_factor)
    )


def legacy_scc_diff_bookkeeping_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Pre-lane flat bookkeeping dir: ``bookkeeping/diff/`` (no oversampling leaf)."""
    return scc_bookkeeping_dir(data_root, sector, camera, ccd) / DIFF_SUBDIR


def resolve_scc_diff_bookkeeping_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
    template_store_name: str | None = None,
) -> Path:
    """Return the canonical bookkeeping dir for one diff template lane."""
    return scc_diff_bookkeeping_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
        template_store_name=template_store_name,
    )


def scc_mapping_master_skycells_csv(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int,
) -> Path:
    """Path to mapping master skycells CSV under the SCC oversampling leaf."""
    suffix = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    return (
        scc_mapping_dir(data_root, sector, camera, ccd, oversampling_factor=oversampling_factor)
        / f"tess_s{int(sector):04d}_{int(camera)}_{int(ccd)}_master_skycells_list{suffix}.csv"
    )


def scc_mapping_master_pixels2skycells(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int,
) -> Path:
    """Path to mapping master pixels2skycells FITS under the SCC oversampling leaf."""
    suffix = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    return (
        scc_mapping_dir(data_root, sector, camera, ccd, oversampling_factor=oversampling_factor)
        / f"tess_s{int(sector):04d}_{int(camera)}_{int(ccd)}_master_pixels2skycells{suffix}.fits.fz"
    )


def ps1_skycells_zarr_dir(data_root: str | Path) -> Path:
    """Directory that holds the shared PS1 skycells Zarr store."""
    return Path(data_root).expanduser() / PS1_SKYCELLS_ZARR_DIRNAME


def ps1_skycells_zarr_path(data_root: str | Path) -> Path:
    """Canonical shared PS1 raw-band Zarr store under ``data_root``."""
    return ps1_skycells_zarr_dir(data_root) / PS1_SKYCELLS_ZARR_BASENAME


def ps1_skycells_zarr_lock_path(data_root: str | Path) -> Path:
    """Lock file path alongside the shared PS1 skycells Zarr store."""
    zarr_path = ps1_skycells_zarr_path(data_root)
    return zarr_path.parent / f"{zarr_path.name}.lock"


def event_root(workspace_root: str | Path, event_name: str) -> Path:
    """Root directory for one event under ``workspace_root``."""
    return Path(workspace_root).expanduser() / EVENTS_SUBDIR / str(event_name).strip()


def event_scc_leaf(
    workspace_root: str | Path,
    event_name: str,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Event workspace leaf for one SCC: ``events/{event}/s{SSSS}_c{C}_k{K}/``."""
    return event_root(workspace_root, event_name) / scc_label(sector, camera, ccd)


def ps1_combined_zarr_path(data_root: str | Path) -> Path:
    """Sky-keyed shared combined-skycell Zarr store (provenance plan §7/§14, decision #14)."""
    return ps1_skycells_zarr_dir(data_root) / PS1_COMBINED_ZARR_BASENAME


def ps1_convolved_zarr_path(data_root: str | Path) -> Path:
    """Sky-keyed shared convolved-skycell Zarr store (provenance plan §7/§14, decision #14)."""
    return ps1_skycells_zarr_dir(data_root) / PS1_CONVOLVED_ZARR_BASENAME


def provenance_db_path(data_root: str | Path) -> Path:
    """Path to the rebuildable provenance index ``{data_root}/bookkeeping/provenance.db``."""
    return Path(data_root).expanduser() / BOOKKEEPING_SUBDIR / PROVENANCE_DB_BASENAME


def provenance_spool_dir(data_root: str | Path) -> Path:
    """Directory of per-host/pid sidecar JSONL spool files the supervisor drains."""
    return Path(data_root).expanduser() / BOOKKEEPING_SUBDIR / PROVENANCE_SPOOL_SUBDIR
