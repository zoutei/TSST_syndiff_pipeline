"""Bootstrap SCC-primary diff bookkeeping from template pipeline artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.common.scc_paths import (
    normalize_store_name,
    resolve_scc_diff_bookkeeping_dir,
    scc_diff_bookkeeping_dir,
    scc_diff_dir,
    scc_ffi_list_parquet,
    scc_remap_dir,
    scc_templates_dir,
)
from syndiff_pipeline.template_creation.processing.field_remap import (
    GROUP_ID_PER_FRAME_NPY,
)

DIFF_JOB_BASENAME = "diff_job.json"
FRAMES_CSV_BASENAME = "frames.csv"
FIELD_MODE_ASSEMBLY_BASENAME = "field_mode_assembly.json"
LINEAR_MODE_ASSEMBLY_BASENAME = "linear_mode_assembly.json"


@dataclass(frozen=True)
class SccDiffBootstrapResult:
    """Artifacts written by :func:`bootstrap_scc_diff`."""

    mapping_grid: MappingGrid
    crop_bounds: dict[str, Any]
    frames_df: pd.DataFrame
    diff_store_root: Path
    bookkeeping_dir: Path
    diff_job_path: Path
    frames_csv_path: Path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def bootstrap_scc_diff(
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    template_store_name: str | None,
    output_store_name: str | None,
    remap_store_name: str | None,
    oversampling_factor: int = 1,
    event_name: str | None = None,
) -> SccDiffBootstrapResult:
    """
    Assemble ``bookkeeping/diff/frames.csv`` and ``diff_job.json`` for field-mode diff.

    Replaces event ``bind`` handoff for SCC-primary workflows.
    """
    from syndiff_pipeline.common.download import list_local_ffis, manifest_basename_from_local
    from syndiff_pipeline.common.scc_paths import scc_ffi_dir
    from syndiff_pipeline.common.wcs_header_cache import load_ffi_list

    data_root = Path(data_root).expanduser()
    os_factor = max(1, int(oversampling_factor))

    template_store = scc_templates_dir(
        data_root, sector, camera, ccd,
        oversampling_factor=os_factor,
        store_name=template_store_name,
    )
    mapping_grid = _load_mapping_grid_from_template_store(template_store)

    remap_store = scc_remap_dir(
        data_root, sector, camera, ccd,
        oversampling_factor=os_factor,
        store_name=remap_store_name,
    )
    _validate_template_remap_provenance(template_store, remap_store, mapping_grid)
    gid_path = remap_store / GROUP_ID_PER_FRAME_NPY
    if not gid_path.is_file():
        raise FileNotFoundError(f"Missing remap artifact: {gid_path}")
    group_ids = np.load(gid_path)

    ffi_dir = scc_ffi_dir(data_root, sector, camera, ccd)
    paths = sorted(list_local_ffis(str(ffi_dir), sector, camera, ccd))
    if len(paths) != len(group_ids):
        raise ValueError(
            f"FFI count {len(paths)} != group_id_per_frame length {len(group_ids)}"
        )

    ffi_list_path = scc_ffi_list_parquet(data_root, sector, camera, ccd)
    ffi_list_df = load_ffi_list(ffi_list_path) if ffi_list_path.is_file() else None

    rows: list[dict[str, Any]] = []
    for i, p in enumerate(paths):
        logical = manifest_basename_from_local(p)
        row: dict[str, Any] = {
            "path": str(p),
            "ffi_basename": logical,
            "group_id": int(group_ids[i]),
        }
        if ffi_list_df is not None and logical in ffi_list_df.index:
            src = ffi_list_df.loc[logical]
            for col in ("DATE-OBS", "BTJD", "wcs_ok"):
                if col in src.index:
                    row[col] = src[col]
        rows.append(row)
    frames_df = pd.DataFrame(rows)

    crop_bounds = mapping_grid.science_ffi_bounds()
    from syndiff_pipeline.common.coordinate_preflight import validate_coordinate_contract

    validate_coordinate_contract(mapping_grid, crop_bounds)
    bookkeeping_dir = scc_diff_bookkeeping_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=os_factor,
        template_store_name=template_store_name,
    )
    bookkeeping_dir.mkdir(parents=True, exist_ok=True)
    frames_csv_path = bookkeeping_dir / FRAMES_CSV_BASENAME
    frames_df.to_csv(frames_csv_path, index=False)

    diff_job = {
        "schema_version": 2,
        "sector": int(sector),
        "camera": int(camera),
        "ccd": int(ccd),
        "geometry_mode": "field",
        "mapping_grid": mapping_grid.to_mapping_dict(),
        "crop_bounds": crop_bounds,
        "template_store_name": template_store_name,
        "output_store_name": output_store_name,
        "remap_store_name": remap_store_name,
        "oversampling_factor": os_factor,
        "event_name": event_name,
    }
    diff_job_path = bookkeeping_dir / DIFF_JOB_BASENAME
    _atomic_write_json(diff_job_path, diff_job)

    diff_store_root = scc_diff_dir(
        data_root, sector, camera, ccd, store_name=output_store_name
    )

    return SccDiffBootstrapResult(
        mapping_grid=mapping_grid,
        crop_bounds=crop_bounds,
        frames_df=frames_df,
        diff_store_root=diff_store_root,
        bookkeeping_dir=bookkeeping_dir,
        diff_job_path=diff_job_path,
        frames_csv_path=frames_csv_path,
    )


def _template_store_geometry_mode(template_store: Path) -> str | None:
    """``"field"`` / ``"linear"`` / ``None`` (neither sidecar present yet)."""
    if (template_store / LINEAR_MODE_ASSEMBLY_BASENAME).is_file():
        return "linear"
    if (template_store / FIELD_MODE_ASSEMBLY_BASENAME).is_file():
        return "field"
    return None


def _load_mapping_grid_from_template_store(template_store: Path) -> MappingGrid:
    from syndiff_pipeline.common.mapping_grid import MappingGridError

    sidecar = template_store / FIELD_MODE_ASSEMBLY_BASENAME
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing field mode sidecar: {sidecar}")
    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    schema = int(doc.get("schema_version", 0))
    if schema >= 3:
        return MappingGrid.from_sidecar(doc)
    raise MappingGridError(
        f"template store {template_store} is v1/v2 field mode "
        f"(field_mode_assembly schema_version={schema}; need >=3 with mapping_grid). "
        "Rebuild mapping with MAPGRID=3 and field downsample before scc_bootstrap."
    )


def _validate_template_remap_provenance(
    template_store: Path, remap_store: Path, mapping_grid: MappingGrid
) -> None:
    """Fail closed when the L5 and remap temporal handoffs differ."""
    sidecar = template_store / FIELD_MODE_ASSEMBLY_BASENAME
    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    saved = doc.get("mapping_grid") or {}
    if str(saved.get("geometry_fingerprint")) != str(mapping_grid.geometry_fingerprint):
        raise ValueError("template sidecar geometry does not match MappingGrid")
    if int(getattr(mapping_grid, "mapgrid_version", 0)) != 3:
        raise ValueError("scc_bootstrap requires MAPGRID=3 geometry")
    if int(getattr(mapping_grid, "mapgrid_version", 0)) == 3:
        if str(doc.get("science_pad_policy", "")) != "neutral_invalid":
            raise ValueError(
                "MAPGRID=3 template sidecar must declare science_pad_policy=neutral_invalid"
            )
        expected_support = {
            "x_min": int(mapping_grid.template_xmin),
            "x_max": int(mapping_grid.template_xmax),
            "y_min": int(mapping_grid.template_ymin),
            "y_max": int(mapping_grid.template_ymax),
        }
        if doc.get("template_support_bounds_ffi") != expected_support:
            raise ValueError("template sidecar template_support_bounds_ffi does not match MappingGrid")
    remap_manifest = remap_store / "remap_manifest.json"
    if not remap_manifest.is_file():
        # Older non-temporal field fixtures may not have a remap manifest;
        # temporal stores must always carry one for frame-contract checks.
        if str(doc.get("geometry_mode", "field")) == "temporal_wcs":
            raise FileNotFoundError(f"Missing remap provenance manifest: {remap_manifest}")
        return
    remap = json.loads(remap_manifest.read_text(encoding="utf-8"))
    provenance = dict(doc.get("geometry_provenance") or {})
    for key in ("temporal_wcs_fingerprint", "temporal_wcs_frame_contract_fingerprint"):
        expected = remap.get(key)
        if expected is not None and str(provenance.get(key)) != str(expected):
            raise ValueError(
                f"template/remap provenance mismatch for {key}: "
                f"template={provenance.get(key)!r}, remap={expected!r}"
            )


def bootstrap_scc_diff_linear(
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    template_store_name: str | None,
    output_store_name: str | None,
    remap_store_name: str | None,
    oversampling_factor: int = 1,
    event_name: str | None = None,
) -> SccDiffBootstrapResult:
    """Assemble ``bookkeeping/diff/frames.csv`` and ``diff_job.json`` for linear-mode diff.

    Mirrors :func:`bootstrap_scc_diff` but sources per-frame ``group_id`` from
    the SCC point-drift table (``point_drift_table.csv``) instead of a
    remap-produced ``group_id_per_frame.npy`` -- linear mode skips the remap
    *stage* entirely (``SKIP_REASON_LINEAR_GEOMETRY``), so that file never
    gets written.
    """
    from syndiff_pipeline.common.download import list_local_ffis, manifest_basename_from_local
    from syndiff_pipeline.common.scc_paths import scc_ffi_dir
    from syndiff_pipeline.common.wcs_header_cache import load_ffi_list

    data_root = Path(data_root).expanduser()
    os_factor = max(1, int(oversampling_factor))

    template_store = scc_templates_dir(
        data_root, sector, camera, ccd,
        oversampling_factor=os_factor,
        store_name=template_store_name,
    )
    sidecar = template_store / LINEAR_MODE_ASSEMBLY_BASENAME
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing linear mode sidecar: {sidecar}")
    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    if int(doc.get("schema_version", 0)) < 3 or "mapping_grid" not in doc:
        raise ValueError(
            f"linear mode sidecar {sidecar} missing mapping_grid "
            "(schema_version >= 3 required); rebuild linear downsample."
        )
    mapping_grid = MappingGrid.from_mapping_dict(doc)

    point_drift_store = scc_remap_dir(
        data_root, sector, camera, ccd,
        oversampling_factor=os_factor, store_name="linear",
    )
    point_drift_table_path = point_drift_store / "point_drift_table.csv"
    if not point_drift_table_path.is_file():
        raise FileNotFoundError(f"Missing point-drift table: {point_drift_table_path}")
    drift_table = pd.read_csv(point_drift_table_path)
    if "group_id" not in drift_table.columns:
        raise ValueError(f"{point_drift_table_path} missing group_id column")
    name_col = "filename" if "filename" in drift_table.columns else (
        "ffi_basename" if "ffi_basename" in drift_table.columns else None
    )
    if name_col is None:
        raise ValueError(
            f"{point_drift_table_path} missing a per-frame FFI name column "
            "(expected filename or ffi_basename)"
        )
    if "group_dx" not in drift_table.columns or "group_dy" not in drift_table.columns:
        raise ValueError(f"{point_drift_table_path} missing group_dx/group_dy columns")
    group_id_by_name = dict(zip(drift_table[name_col].astype(str), drift_table["group_id"]))
    group_dx_by_name = dict(zip(drift_table[name_col].astype(str), drift_table["group_dx"]))
    group_dy_by_name = dict(zip(drift_table[name_col].astype(str), drift_table["group_dy"]))

    ffi_dir = scc_ffi_dir(data_root, sector, camera, ccd)
    paths = sorted(list_local_ffis(str(ffi_dir), sector, camera, ccd))
    ffi_list_path = scc_ffi_list_parquet(data_root, sector, camera, ccd)
    ffi_list_df = load_ffi_list(ffi_list_path) if ffi_list_path.is_file() else None

    rows: list[dict[str, Any]] = []
    for p in paths:
        logical = manifest_basename_from_local(p)
        if logical not in group_id_by_name:
            continue
        row: dict[str, Any] = {
            "path": str(p),
            "ffi_basename": logical,
            "group_id": int(group_id_by_name[logical]),
            "group_dx": float(group_dx_by_name[logical]),
            "group_dy": float(group_dy_by_name[logical]),
        }
        if ffi_list_df is not None and logical in ffi_list_df.index:
            src = ffi_list_df.loc[logical]
            for col in ("DATE-OBS", "BTJD", "wcs_ok"):
                if col in src.index:
                    row[col] = src[col]
        rows.append(row)
    frames_df = pd.DataFrame(rows)
    if frames_df.empty:
        raise ValueError(
            f"No FFIs under {ffi_dir} matched point-drift table entries in {point_drift_table_path}"
        )

    crop_bounds = mapping_grid.science_ffi_bounds()
    from syndiff_pipeline.common.coordinate_preflight import validate_coordinate_contract

    validate_coordinate_contract(mapping_grid, crop_bounds)
    bookkeeping_dir = scc_diff_bookkeeping_dir(
        data_root, sector, camera, ccd,
        oversampling_factor=os_factor, template_store_name=template_store_name,
    )
    bookkeeping_dir.mkdir(parents=True, exist_ok=True)
    frames_csv_path = bookkeeping_dir / FRAMES_CSV_BASENAME
    frames_df.to_csv(frames_csv_path, index=False)

    diff_job = {
        "schema_version": 2,
        "sector": int(sector), "camera": int(camera), "ccd": int(ccd),
        "geometry_mode": "linear",
        "mapping_grid": mapping_grid.to_mapping_dict(),
        "crop_bounds": crop_bounds,
        "template_store_name": template_store_name,
        "output_store_name": output_store_name,
        "remap_store_name": remap_store_name,
        "oversampling_factor": os_factor,
        "event_name": event_name,
    }
    diff_job_path = bookkeeping_dir / DIFF_JOB_BASENAME
    _atomic_write_json(diff_job_path, diff_job)

    diff_store_root = scc_diff_dir(data_root, sector, camera, ccd, store_name=output_store_name)

    return SccDiffBootstrapResult(
        mapping_grid=mapping_grid,
        crop_bounds=crop_bounds,
        frames_df=frames_df,
        diff_store_root=diff_store_root,
        bookkeeping_dir=bookkeeping_dir,
        diff_job_path=diff_job_path,
        frames_csv_path=frames_csv_path,
    )


def _resolve_reference_ffi_path(
    data_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    frames_df: pd.DataFrame,
) -> str:
    from syndiff_pipeline.common.scc_paths import scc_bookkeeping_stage_dir

    meta_path = scc_bookkeeping_stage_dir(data_root, sector, camera, ccd, "mapping") / "run_meta.json"
    if meta_path.is_file():
        doc = json.loads(meta_path.read_text(encoding="utf-8"))
        ref = str(doc.get("reference_ffi_path") or "").strip()
        if ref and Path(ref).is_file():
            return ref
    if not frames_df.empty and "path" in frames_df.columns:
        first = str(frames_df.iloc[0]["path"])
        if Path(first).is_file():
            return first
    raise FileNotFoundError(
        f"Could not resolve reference FFI for s{sector:04d}/c{camera}/k{ccd}"
    )


def _inherit_remap_store_name(
    data_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    template_store_name: str | None,
    oversampling_factor: int,
) -> str | None:
    """Read ``remap_store_name`` off the template sidecar when caller left it unset."""
    template_store = scc_templates_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
        store_name=template_store_name,
    )
    sidecar = template_store / FIELD_MODE_ASSEMBLY_BASENAME
    if not sidecar.is_file():
        return None
    try:
        doc = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return normalize_store_name(doc.get("remap_store_name"))


def ensure_scc_diff_handoff(
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    template_store_name: str | None = None,
    output_store_name: str | None = None,
    remap_store_name: str | None = None,
    oversampling_factor: int = 1,
    event_name: str | None = None,
) -> SccDiffBootstrapResult:
    """Load or create SCC-primary diff bookkeeping for field-mode differencing.

    Reuses an existing ``diff_job.json`` only when it matches the requested
    lane identity (``oversampling_factor``, ``template_store_name``,
    ``remap_store_name``, ``schema_version >= 2``); otherwise rebuilds it in
    place. When ``remap_store_name`` is not given explicitly, it is inherited
    from the template store's ``field_mode_assembly.json`` sidecar.
    """
    data_root = Path(data_root).expanduser()
    os_factor = max(1, int(oversampling_factor))
    template_store_name = normalize_store_name(template_store_name)
    remap_store_name = normalize_store_name(remap_store_name)
    if remap_store_name is None:
        remap_store_name = _inherit_remap_store_name(
            data_root,
            sector,
            camera,
            ccd,
            template_store_name=template_store_name,
            oversampling_factor=os_factor,
        )

    bookkeeping_dir = resolve_scc_diff_bookkeeping_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=os_factor,
        template_store_name=template_store_name,
    )
    job_path = bookkeeping_dir / DIFF_JOB_BASENAME
    frames_path = bookkeeping_dir / FRAMES_CSV_BASENAME
    if job_path.is_file() and frames_path.is_file():
        doc = json.loads(job_path.read_text(encoding="utf-8"))
        identity_ok = (
            int(doc.get("schema_version", 0)) >= 2
            and int(doc.get("oversampling_factor", 1)) == os_factor
            and normalize_store_name(doc.get("template_store_name")) == template_store_name
            and normalize_store_name(doc.get("remap_store_name")) == remap_store_name
        )
        if identity_ok:
            grid = MappingGrid.from_mapping_dict(doc["mapping_grid"])
            frames_df = pd.read_csv(frames_path)
            crop_bounds = doc.get("crop_bounds") or grid.science_ffi_bounds()
            return SccDiffBootstrapResult(
                mapping_grid=grid,
                crop_bounds=crop_bounds,
                frames_df=frames_df,
                diff_store_root=scc_diff_dir(
                    data_root, sector, camera, ccd, store_name=output_store_name
                ),
                bookkeeping_dir=bookkeeping_dir,
                diff_job_path=job_path,
                frames_csv_path=frames_path,
            )
    template_store = scc_templates_dir(
        data_root, sector, camera, ccd,
        oversampling_factor=os_factor, store_name=template_store_name,
    )
    builder = (
        bootstrap_scc_diff_linear
        if _template_store_geometry_mode(template_store) == "linear"
        else bootstrap_scc_diff
    )
    return builder(
        data_root=data_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
        template_store_name=template_store_name,
        output_store_name=output_store_name,
        remap_store_name=remap_store_name,
        oversampling_factor=os_factor,
        event_name=event_name,
    )


def load_scc_diff_handoff_for_config(cfg) -> tuple[pd.DataFrame, dict, str, float, MappingGrid]:
    """Return frames manifest, crop bounds, reference FFI, offset threshold, and grid."""
    from syndiff_pipeline.common.scc_paths import normalize_store_name

    if not getattr(cfg, "data_root", None):
        raise RuntimeError("SCC diff handoff requires data_root on SynDiffConfig")

    os_factor = max(1, int(getattr(cfg, "oversampling_factor", 1) or 1))
    result = ensure_scc_diff_handoff(
        data_root=cfg.data_root,
        sector=int(cfg.sector),
        camera=int(cfg.camera),
        ccd=int(cfg.ccd),
        template_store_name=normalize_store_name(
            getattr(cfg, "template_store_name", None)
        ),
        output_store_name=normalize_store_name(
            getattr(cfg, "output_store_name", None)
        ),
        remap_store_name=normalize_store_name(getattr(cfg, "remap_store_name", None)),
        oversampling_factor=os_factor,
        event_name=str(getattr(cfg, "target_name", "") or "") or None,
    )
    ref_ffi = getattr(cfg, "ref_ffi_path", None) or None
    if ref_ffi and Path(str(ref_ffi)).is_file():
        ref_path = str(Path(ref_ffi).resolve())
    else:
        ref_path = _resolve_reference_ffi_path(
            Path(cfg.data_root),
            int(cfg.sector),
            int(cfg.camera),
            int(cfg.ccd),
            result.frames_df,
        )
    return result.frames_df, result.crop_bounds, ref_path, 0.01, result.mapping_grid
