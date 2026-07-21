"""Template-stage provenance checkpoints (plan §11).

Each template stage that produces a graph node emits a coarse checkpoint
artifact on success (dual-write alongside manifests). Input fingerprints follow
the kind registry (§6): ``ffi_set``←N×``ffi``, ``mapping``←``ffi_set``, etc.
Filesystem-backed inputs (FFI files, convolved cells) are best-effort: when
unresolvable, emit and ``expected_*`` both omit those edges so they stay in
lockstep (scheduler hit path never walks when nothing is on disk).

All :func:`emit_*_checkpoint` helpers are best-effort and **never raise**.
:func:`expected_*_fingerprint` helpers share the same input-resolution logic
as emit (may touch ``ffi_list`` / mapping CSV / convolved sidecars when present).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        ResolvedTargetConfig,
    )

log = logging.getLogger(__name__)

FFI_KIND = "ffi"
FFI_SET_KIND = "ffi_set"
MAPPING_KIND = "mapping"
REMAP_STORE_KIND = "remap_store"
DOWNSAMPLE_KIND = "downsample"
SCC_ASSEMBLY_KIND = "scc_assembly"

CHECKPOINT_STAGE_FINGERPRINTS: dict[str, str] = {
    "tess_ffi_download": "expected_ffi_set_fingerprint",
    "mapping": "expected_mapping_fingerprint",
    "remap": "expected_remap_store_fingerprint",
    "downsample": "expected_downsample_fingerprint",
    "ps1_process": "expected_scc_assembly_fingerprint",
}

CHECKPOINT_STAGES = frozenset(CHECKPOINT_STAGE_FINGERPRINTS)

__all__ = [
    "FFI_KIND",
    "FFI_SET_KIND",
    "MAPPING_KIND",
    "REMAP_STORE_KIND",
    "DOWNSAMPLE_KIND",
    "SCC_ASSEMBLY_KIND",
    "CHECKPOINT_STAGE_FINGERPRINTS",
    "CHECKPOINT_STAGES",
    "checkpoint_stage_indexed",
    "expected_ffi_set_fingerprint",
    "emit_ffi_set_checkpoint",
    "expected_mapping_fingerprint",
    "emit_mapping_checkpoint",
    "expected_remap_store_fingerprint",
    "emit_remap_store_checkpoint",
    "expected_downsample_fingerprint",
    "emit_downsample_checkpoint",
    "expected_scc_assembly_fingerprint",
    "emit_scc_assembly_checkpoint",
]


def checkpoint_stage_indexed(resolved: "ResolvedTargetConfig", stage: str) -> bool:
    """True when the stage's expected fingerprint is present in provenance.db.

    Pure config recompute + one indexed query; no filesystem scans. Returns
    ``False`` on miss or when the store is unavailable.
    """
    fingerprint_fn_name = CHECKPOINT_STAGE_FINGERPRINTS.get(stage)
    if fingerprint_fn_name is None:
        return False
    try:
        from syndiff_pipeline.common.provenance.store import ProvenanceStore
        from syndiff_pipeline.common.scc_paths import provenance_db_path

        expected_fp_fn = globals()[fingerprint_fn_name]
        expected_fp = expected_fp_fn(resolved)
        store = ProvenanceStore(
            str(provenance_db_path(resolved.data_root)), read_only=True
        )
        return store.scc_stage_complete([expected_fp])
    except Exception:
        log.debug(
            "checkpoint_stage_indexed unavailable for %s", stage, exc_info=True
        )
        return False


def _scc_spatial_key_dict(
    resolved: "ResolvedTargetConfig",
    *,
    os_factor: int | None = None,
    store_name: str | None = None,
) -> Dict[str, Any]:
    """Build an SCC spatial-key dict via ``provenance.model.SccKey``."""
    from syndiff_pipeline.common.provenance.model import SccKey

    t = resolved.target
    key = SccKey(
        s=int(t.sector),
        c=int(t.camera),
        k=int(t.ccd),
        os=os_factor,
        store_name=store_name,
    )
    return key.to_dict()


def _ffi_set_spatial_key_dict(resolved: "ResolvedTargetConfig") -> Dict[str, Any]:
    return _scc_spatial_key_dict(resolved)


def _mapping_spatial_key_dict(resolved: "ResolvedTargetConfig") -> Dict[str, Any]:
    oversampling = int(resolved.stages.mapping.oversampling_factor)
    return _scc_spatial_key_dict(resolved, os_factor=oversampling)


def _remap_store_spatial_key_dict(resolved: "ResolvedTargetConfig") -> Dict[str, Any]:
    oversampling = int(resolved.stages.mapping.oversampling_factor)
    store_name = resolved.stages.remap.store_name
    return _scc_spatial_key_dict(
        resolved, os_factor=oversampling, store_name=store_name
    )


def _downsample_spatial_key_dict(resolved: "ResolvedTargetConfig") -> Dict[str, Any]:
    oversampling = int(resolved.stages.downsample.oversampling_factor)
    store_name = resolved.stages.downsample.output_store_name
    return _scc_spatial_key_dict(
        resolved, os_factor=oversampling, store_name=store_name
    )


def _scc_assembly_spatial_key_dict(resolved: "ResolvedTargetConfig") -> Dict[str, Any]:
    """``scc_assembly`` spatial key (mapping-stage oversampling factor)."""
    oversampling = int(resolved.stages.mapping.oversampling_factor)
    return _scc_spatial_key_dict(resolved, os_factor=oversampling)


def _merkle_fingerprint(
    kind: str,
    spatial_key: Dict[str, Any],
    params: dict,
    input_fingerprints: Sequence[str],
) -> str:
    from syndiff_pipeline.common.provenance.fingerprint import (
        RECIPE_SCHEMA_VERSION,
        fingerprint as merkle_fingerprint,
        recipe_id as compute_recipe_id,
    )

    rid = compute_recipe_id(kind, params, RECIPE_SCHEMA_VERSION)
    return merkle_fingerprint(kind, spatial_key, rid, list(input_fingerprints))


def _expected_checkpoint_fingerprint(
    kind: str,
    resolved: "ResolvedTargetConfig",
    recipe_params_fn: Callable[..., dict],
    spatial_key_fn: Callable[["ResolvedTargetConfig"], Dict[str, Any]],
    input_fingerprints: Sequence[str] = (),
) -> str:
    params = recipe_params_fn(resolved)
    spatial_key = spatial_key_fn(resolved)
    return _merkle_fingerprint(kind, spatial_key, params, input_fingerprints)


# ── Input-edge resolution (emit ↔ expected lockstep) ─────────────────────────


def _ffi_paths_from_list(resolved: "ResolvedTargetConfig") -> List[str]:
    """Best-effort FFI paths from ``ffi_list.parquet`` + nested FFI dir."""
    from syndiff_pipeline.common.download import nested_ffi_dir
    from syndiff_pipeline.common.fits_variants import (
        strip_fits_storage_suffix,
        try_resolve_fits_variant,
    )
    from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet
    from syndiff_pipeline.common.wcs_header_cache import load_ffi_list

    t = resolved.target
    parquet = scc_ffi_list_parquet(resolved.data_root, t.sector, t.camera, t.ccd)
    df = load_ffi_list(parquet)
    if df is None or len(df) == 0:
        return []
    ffi_leaf = nested_ffi_dir(
        int(t.sector),
        int(t.camera),
        int(t.ccd),
        root=str(Path(resolved.data_root) / "tess_ffi"),
    )
    paths: List[str] = []
    for logical in df.index.astype(str):
        candidate = Path(ffi_leaf) / str(logical)
        resolved_path = try_resolve_fits_variant(candidate)
        if resolved_path is None:
            # Index may be a bare stem; try logical .fits under the leaf.
            stem = strip_fits_storage_suffix(str(logical))
            resolved_path = try_resolve_fits_variant(Path(ffi_leaf) / f"{stem}.fits")
        if resolved_path is not None:
            paths.append(str(resolved_path))
    return paths


def _ffi_input_fingerprint_for_path(
    resolved: "ResolvedTargetConfig", ffi_path: str
) -> Optional[str]:
    """Fingerprint one ``ffi`` node (mirrors ``provenance_glue.ffi_input_fingerprint``)."""
    try:
        from syndiff_pipeline.difference_imaging.orchestration.provenance_glue import (
            ffi_input_fingerprint,
        )

        t = resolved.target
        return ffi_input_fingerprint(int(t.sector), int(t.camera), int(t.ccd), ffi_path)
    except Exception:
        log.debug("ffi input fingerprint failed for %s", ffi_path, exc_info=True)
        return None


def _resolve_ffi_input_fingerprints(resolved: "ResolvedTargetConfig") -> List[str]:
    """N×``ffi`` fingerprints for ``ffi_set``, or ``[]`` when unresolvable."""
    try:
        paths = _ffi_paths_from_list(resolved)
        if not paths:
            return []
        fps: List[str] = []
        for p in paths:
            fp = _ffi_input_fingerprint_for_path(resolved, p)
            if fp is None:
                return []
            fps.append(fp)
        return sorted(fps)
    except Exception:
        log.debug("ffi_set input resolution failed", exc_info=True)
        return []


def _resolve_convolved_skycell_fingerprints(
    resolved: "ResolvedTargetConfig",
) -> List[str]:
    """N×``convolved_skycell`` fps from mapping CSV + shared store sidecars.

    Returns ``[]`` when the CSV/store is missing, ambiguous, or any walk fails
    (scheduler hit path must fail open to empty edges, not raise).
    """
    try:
        from syndiff_pipeline.common.scc_paths import (
            ps1_convolved_zarr_path,
            scc_mapping_master_skycells_csv,
        )
        from syndiff_pipeline.template_creation.processing.csv_utils import load_csv_data
        from syndiff_pipeline.template_creation.processing.ps1_process import (
            create_master_task_list,
        )
        from syndiff_pipeline.template_creation.processing.csv_utils import (
            get_projections_from_csv,
        )

        t = resolved.target
        os_factor = int(resolved.stages.mapping.oversampling_factor)
        csv_path = scc_mapping_master_skycells_csv(
            resolved.data_root, t.sector, t.camera, t.ccd, oversampling_factor=os_factor
        )
        if not Path(csv_path).is_file():
            return []
        df = load_csv_data(str(csv_path))
        projections = get_projections_from_csv(str(csv_path))
        limit = getattr(resolved.stages.ps1_process, "projections_limit", None)
        if limit:
            projections = projections[: int(limit)]
        required: list[tuple[str, str]] = []
        for projection in projections:
            _, task_list = create_master_task_list(df, projection)
            for skycell_id, proj, _row_id in task_list:
                name = skycell_id[0] if isinstance(skycell_id, (tuple, list)) else skycell_id
                required.append((str(proj), str(name)))
        if not required:
            return []

        root = Path(ps1_convolved_zarr_path(resolved.data_root))
        fps: List[str] = []
        seen: set[str] = set()
        for proj, cell in required:
            cell_dir = root / proj / cell
            if not cell_dir.is_dir():
                return []
            children = [
                p
                for p in cell_dir.iterdir()
                if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
            ]
            if len(children) != 1:
                return []
            sidecar = children[0] / "_provenance.json"
            if not sidecar.is_file():
                # Directory name is the fingerprint when sidecar absent
                fp = children[0].name
            else:
                doc = json.loads(sidecar.read_text(encoding="utf-8"))
                fp = str(doc.get("fingerprint") or children[0].name)
            if fp not in seen:
                seen.add(fp)
                fps.append(fp)
        return sorted(fps)
    except Exception:
        log.debug("convolved_skycell input resolution failed", exc_info=True)
        return []


def _ffi_set_inputs(resolved: "ResolvedTargetConfig") -> List[str]:
    return _resolve_ffi_input_fingerprints(resolved)


def _mapping_inputs(resolved: "ResolvedTargetConfig") -> List[str]:
    return [expected_ffi_set_fingerprint(resolved)]


def _remap_store_inputs(resolved: "ResolvedTargetConfig") -> List[str]:
    return [
        expected_ffi_set_fingerprint(resolved),
        expected_mapping_fingerprint(resolved),
    ]


def _scc_assembly_inputs(resolved: "ResolvedTargetConfig") -> List[str]:
    inputs = [expected_mapping_fingerprint(resolved)]
    inputs.extend(_resolve_convolved_skycell_fingerprints(resolved))
    return inputs


def _downsample_inputs(resolved: "ResolvedTargetConfig") -> List[str]:
    return [
        expected_scc_assembly_fingerprint(resolved),
        expected_remap_store_fingerprint(resolved),
    ]


def _emit_checkpoint_record(
    resolved: "ResolvedTargetConfig",
    *,
    kind: str,
    recipe_params_fn: Callable[..., dict],
    spatial_key_fn: Callable[["ResolvedTargetConfig"], Dict[str, Any]],
    location_fn: Callable[["ResolvedTargetConfig"], str],
    expected_fp_fn: Callable[["ResolvedTargetConfig"], str],
    input_fps_fn: Callable[["ResolvedTargetConfig"], List[str]],
) -> None:
    from syndiff_pipeline.common.provenance.fingerprint import (
        RECIPE_SCHEMA_VERSION,
        recipe_id as compute_recipe_id,
    )
    from syndiff_pipeline.common.provenance.publish import (
        append_spool_record,
        build_record,
    )
    from syndiff_pipeline.common.scc_paths import provenance_spool_dir

    params = recipe_params_fn(resolved)
    rid = compute_recipe_id(kind, params, RECIPE_SCHEMA_VERSION)
    spatial_key = spatial_key_fn(resolved)
    input_fps = list(input_fps_fn(resolved))
    fp = expected_fp_fn(resolved)
    location = location_fn(resolved)

    record = build_record(
        fp,
        kind,
        spatial_key,
        rid,
        RECIPE_SCHEMA_VERSION,
        input_fps,
        location,
        recipe_params=params,
        state="complete",
    )
    append_spool_record(provenance_spool_dir(resolved.data_root), record)


def _best_effort_emit(
    resolved: "ResolvedTargetConfig",
    emit_fn: Callable[["ResolvedTargetConfig"], None],
    *,
    kind: str,
) -> None:
    try:
        emit_fn(resolved)
    except Exception:
        log.exception(
            "%s checkpoint emit failed (non-fatal; manifest still authoritative)",
            kind,
        )


def _ffi_set_location(resolved: "ResolvedTargetConfig") -> str:
    from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet

    t = resolved.target
    return str(scc_ffi_list_parquet(resolved.data_root, t.sector, t.camera, t.ccd))


def _mapping_location(resolved: "ResolvedTargetConfig") -> str:
    return str(resolved.mapping_root)


def _remap_store_location(resolved: "ResolvedTargetConfig") -> str:
    return str(resolved.remap_output_base)


def _downsample_location(resolved: "ResolvedTargetConfig") -> str:
    return str(resolved.template_output_base)


def _scc_assembly_location(resolved: "ResolvedTargetConfig") -> str:
    from syndiff_pipeline.template_creation.orchestration.verify import (
        resolve_ps1_process_checkpoint_location,
    )

    return str(resolve_ps1_process_checkpoint_location(resolved))


def expected_ffi_set_fingerprint(resolved: "ResolvedTargetConfig") -> str:
    from syndiff_pipeline.common.provenance.model import ffi_set_recipe_params

    return _expected_checkpoint_fingerprint(
        FFI_SET_KIND,
        resolved,
        lambda _resolved: ffi_set_recipe_params(),
        _ffi_set_spatial_key_dict,
        _ffi_set_inputs(resolved),
    )


def _emit_individual_ffi_nodes(resolved: "ResolvedTargetConfig") -> None:
    """Best-effort spool records for each ``ffi`` leaf under this SCC."""
    from syndiff_pipeline.common.provenance.fingerprint import (
        RECIPE_SCHEMA_VERSION,
        recipe_id as compute_recipe_id,
    )
    from syndiff_pipeline.common.provenance.publish import (
        append_spool_record,
        build_record,
    )
    from syndiff_pipeline.common.scc_paths import provenance_spool_dir
    from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
    from syndiff_pipeline.difference_imaging.support.ffi_naming import (
        tess_product_id_from_ffi_path,
    )

    t = resolved.target
    spool_dir = provenance_spool_dir(resolved.data_root)
    for ffi_path in _ffi_paths_from_list(resolved):
        fp = _ffi_input_fingerprint_for_path(resolved, ffi_path)
        product_id = tess_product_id_from_ffi_path(ffi_path)
        if not fp or not product_id:
            continue
        version = pg.ffi_input_version(ffi_path)
        rid = compute_recipe_id(FFI_KIND, version, RECIPE_SCHEMA_VERSION)
        spatial_key = pg.ffi_spatial_key(int(t.sector), int(t.camera), int(t.ccd), product_id)
        record = build_record(
            fp,
            FFI_KIND,
            spatial_key,
            rid,
            RECIPE_SCHEMA_VERSION,
            [],
            str(ffi_path),
            recipe_params=version,
            state="complete",
        )
        append_spool_record(spool_dir, record)


def _emit_ffi_set_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    from syndiff_pipeline.common.provenance.model import ffi_set_recipe_params

    try:
        _emit_individual_ffi_nodes(resolved)
    except Exception:
        log.debug("individual ffi node emit failed (non-fatal)", exc_info=True)

    _emit_checkpoint_record(
        resolved,
        kind=FFI_SET_KIND,
        recipe_params_fn=lambda _resolved: ffi_set_recipe_params(),
        spatial_key_fn=_ffi_set_spatial_key_dict,
        location_fn=_ffi_set_location,
        expected_fp_fn=expected_ffi_set_fingerprint,
        input_fps_fn=_ffi_set_inputs,
    )


def emit_ffi_set_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    _best_effort_emit(resolved, _emit_ffi_set_checkpoint, kind=FFI_SET_KIND)


def expected_mapping_fingerprint(resolved: "ResolvedTargetConfig") -> str:
    from syndiff_pipeline.common.provenance.model import mapping_recipe_params

    return _expected_checkpoint_fingerprint(
        MAPPING_KIND,
        resolved,
        mapping_recipe_params,
        _mapping_spatial_key_dict,
        _mapping_inputs(resolved),
    )


def _emit_mapping_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    from syndiff_pipeline.common.provenance.model import mapping_recipe_params

    _emit_checkpoint_record(
        resolved,
        kind=MAPPING_KIND,
        recipe_params_fn=mapping_recipe_params,
        spatial_key_fn=_mapping_spatial_key_dict,
        location_fn=_mapping_location,
        expected_fp_fn=expected_mapping_fingerprint,
        input_fps_fn=_mapping_inputs,
    )


def emit_mapping_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    _best_effort_emit(resolved, _emit_mapping_checkpoint, kind=MAPPING_KIND)


def expected_remap_store_fingerprint(resolved: "ResolvedTargetConfig") -> str:
    from syndiff_pipeline.common.provenance.model import remap_store_recipe_params

    return _expected_checkpoint_fingerprint(
        REMAP_STORE_KIND,
        resolved,
        remap_store_recipe_params,
        _remap_store_spatial_key_dict,
        _remap_store_inputs(resolved),
    )


def _emit_remap_store_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    from syndiff_pipeline.common.provenance.model import remap_store_recipe_params

    _emit_checkpoint_record(
        resolved,
        kind=REMAP_STORE_KIND,
        recipe_params_fn=remap_store_recipe_params,
        spatial_key_fn=_remap_store_spatial_key_dict,
        location_fn=_remap_store_location,
        expected_fp_fn=expected_remap_store_fingerprint,
        input_fps_fn=_remap_store_inputs,
    )


def emit_remap_store_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    _best_effort_emit(resolved, _emit_remap_store_checkpoint, kind=REMAP_STORE_KIND)


def expected_downsample_fingerprint(resolved: "ResolvedTargetConfig") -> str:
    from syndiff_pipeline.common.provenance.model import downsample_recipe_params

    return _expected_checkpoint_fingerprint(
        DOWNSAMPLE_KIND,
        resolved,
        downsample_recipe_params,
        _downsample_spatial_key_dict,
        _downsample_inputs(resolved),
    )


def _emit_downsample_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    from syndiff_pipeline.common.provenance.model import downsample_recipe_params

    _emit_checkpoint_record(
        resolved,
        kind=DOWNSAMPLE_KIND,
        recipe_params_fn=downsample_recipe_params,
        spatial_key_fn=_downsample_spatial_key_dict,
        location_fn=_downsample_location,
        expected_fp_fn=expected_downsample_fingerprint,
        input_fps_fn=_downsample_inputs,
    )


def emit_downsample_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    _best_effort_emit(resolved, _emit_downsample_checkpoint, kind=DOWNSAMPLE_KIND)


def expected_scc_assembly_fingerprint(resolved: "ResolvedTargetConfig") -> str:
    from syndiff_pipeline.common.provenance.model import scc_assembly_recipe_params

    return _expected_checkpoint_fingerprint(
        SCC_ASSEMBLY_KIND,
        resolved,
        scc_assembly_recipe_params,
        _scc_assembly_spatial_key_dict,
        _scc_assembly_inputs(resolved),
    )


def _emit_scc_assembly_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    from syndiff_pipeline.common.provenance.model import scc_assembly_recipe_params

    _emit_checkpoint_record(
        resolved,
        kind=SCC_ASSEMBLY_KIND,
        recipe_params_fn=scc_assembly_recipe_params,
        spatial_key_fn=_scc_assembly_spatial_key_dict,
        location_fn=_scc_assembly_location,
        expected_fp_fn=expected_scc_assembly_fingerprint,
        input_fps_fn=_scc_assembly_inputs,
    )


def emit_scc_assembly_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    _best_effort_emit(resolved, _emit_scc_assembly_checkpoint, kind=SCC_ASSEMBLY_KIND)
