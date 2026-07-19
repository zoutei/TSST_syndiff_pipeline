"""
SCC and event workspace path conventions for the storage-first layout.

SCC-scoped template artifacts live under ``{data_root}/s{SSSS}/c{C}/k{K}/``.
Event-scoped diff/star workspaces live under
``{workspace_root}/events/{event_name}/s{SSSS}_c{C}_k{K}/``.

Path helpers return :class:`~pathlib.Path` objects and do **not** create
directories unless a helper explicitly documents ``mkdir`` behavior.
"""

from __future__ import annotations

from pathlib import Path

EVENTS_SUBDIR = "events"

FFI_SUBDIR = "ffi"
CATALOGS_SUBDIR = "catalogs"
MAPPING_SUBDIR = "mapping"
TEMPLATES_SUBDIR = "templates"
LEGACY_SUBDIR = "legacy"
BOOKKEEPING_SUBDIR = "bookkeeping"

CONVOLVED_ZARR_BASENAME = "convolved.zarr"
CONVOLVED_REMOVED_STARS_CSV_BASENAME = "convolved_removed_stars.csv"
WCS_CACHE_PARQUET_BASENAME = "wcs_cache.parquet"
WCS_CACHE_CSV_BASENAME = "wcs_cache.csv"

# Shared PS1 raw-band Zarr store (ps1_download + syndiff star).
PS1_SKYCELLS_ZARR_DIRNAME = "ps1_skycells_zarr"
PS1_SKYCELLS_ZARR_BASENAME = "ps1_skycells.zarr"

__all__ = [
    "BOOKKEEPING_SUBDIR",
    "CATALOGS_SUBDIR",
    "CONVOLVED_REMOVED_STARS_CSV_BASENAME",
    "CONVOLVED_ZARR_BASENAME",
    "EVENTS_SUBDIR",
    "FFI_SUBDIR",
    "LEGACY_SUBDIR",
    "MAPPING_SUBDIR",
    "PS1_SKYCELLS_ZARR_BASENAME",
    "PS1_SKYCELLS_ZARR_DIRNAME",
    "TEMPLATES_SUBDIR",
    "WCS_CACHE_CSV_BASENAME",
    "WCS_CACHE_PARQUET_BASENAME",
    "event_root",
    "event_scc_leaf",
    "oversampling_dirname",
    "ps1_skycells_zarr_dir",
    "ps1_skycells_zarr_lock_path",
    "ps1_skycells_zarr_path",
    "scc_bookkeeping_dir",
    "scc_bookkeeping_stage_dir",
    "scc_catalogs_dir",
    "scc_convolved_removed_stars_csv",
    "scc_convolved_zarr",
    "scc_ffi_dir",
    "scc_label",
    "scc_legacy_dir",
    "scc_mapping_dir",
    "scc_mapping_master_pixels2skycells",
    "scc_mapping_master_skycells_csv",
    "scc_root",
    "scc_templates_dir",
    "scc_wcs_cache_csv",
    "scc_wcs_cache_parquet",
]


def scc_label(sector: int, camera: int, ccd: int) -> str:
    """Return the canonical SCC filesystem label ``s{SSSS}_c{C}_k{K}``."""
    return f"s{int(sector):04d}_c{int(camera)}_k{int(ccd)}"


def oversampling_dirname(oversampling_factor: int) -> str:
    """Directory name for one oversampling factor, e.g. ``oversampling_2``."""
    return f"oversampling_{int(oversampling_factor)}"


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


def scc_wcs_cache_parquet(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Path to the shared per-SCC WCS header cache (Parquet)."""
    return scc_root(data_root, sector, camera, ccd) / WCS_CACHE_PARQUET_BASENAME


def scc_wcs_cache_csv(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Path to the CSV twin of the shared WCS header cache."""
    return scc_root(data_root, sector, camera, ccd) / WCS_CACHE_CSV_BASENAME


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


def scc_templates_dir(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int,
) -> Path:
    """Directory for full-chip template products at one oversampling factor."""
    return (
        scc_root(data_root, sector, camera, ccd)
        / TEMPLATES_SUBDIR
        / oversampling_dirname(oversampling_factor)
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
