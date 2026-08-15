"""Workspace-aware diff completion checks for orchestrator verification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.pipeline_entries import split_pipeline
from syndiff_pipeline.difference_imaging.orchestration.site_config import freeze_target_diff_config
from syndiff_pipeline.difference_imaging.orchestration.validate import _outputs_for_stage
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    is_pipeline_fits_filename,
    resolve_pipeline_artifact_path,
)
from syndiff_pipeline.common.scc_paths import (
    normalize_store_name,
    resolve_scc_diff_bookkeeping_dir,
    scc_diff_dir,
    scc_diff_label_dir,
)
from syndiff_pipeline.difference_imaging.support.manifest import manifest_path_from_output_dir
from syndiff_pipeline.difference_imaging.support.paths import (
    SHARED_MASK_FITS_BASENAME,
    workspace_root,
)

# Guarded: provenance_glue itself never raises on import (it guards its own
# common.provenance import internally), but this import is wrapped again here
# per the task contract ("diff_verify works if the package is absent") in
# case provenance_glue.py is missing/broken in a partial checkout.
try:
    from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as _prov
except Exception:  # pragma: no cover
    _prov = None

if TYPE_CHECKING:
    from syndiff_pipeline.common.orchestration.spec import StageRunContext
    from syndiff_pipeline.template_creation.orchestration.runner_config import RunnerConfig

log = logging.getLogger(__name__)

# YAML pipeline "kind" -> provenance kind, for the per-FFI stages this module
# can derive an exact required-fingerprint set for (registry §6). Stages not
# listed here (forced_photometry, centroids, subtract, sat_template,
# kernel_fit, convolved_templates, astrometry) always fall open to the legacy
# marker check.
_INDEXED_STAGE_KIND = {
    "hotpants": "diff_image",
    "kernel_subtract": "diff_image",
    "epsf": "epsf",
    "shared_mask": "shared_mask",
    "background": "diff_background",
}


def resolve_diff_site_config_path(
    *,
    meta: dict | None = None,
    runner_cfg: RunnerConfig | None = None,
) -> Path:
    """Match execute-time config resolution: run_meta overrides runner default."""
    for key in ("source_diff_config_path", "diff_config_path"):
        raw = (meta or {}).get(key) or getattr(runner_cfg, "diff_config_path", "")
        if raw:
            return Path(str(raw)).expanduser().resolve()
    raise ValueError(
        "Diff verification requires source_diff_config_path in run_meta or "
        "diff_config_path on RunnerConfig"
    )


def apply_workspace_run_id_override(
    cfg: SynDiffConfig,
    meta: dict | None,
) -> SynDiffConfig:
    """Apply orchestrator run meta override on top of frozen site defaults."""
    if not meta:
        return cfg
    override = meta.get("workspace_run_id")
    if override is not None and str(override).strip():
        cfg.workspace_run_id = str(override).strip()
    return cfg


def frozen_diff_config_for_verify(
    site_config_path: str | Path,
    target: Target,
    *,
    meta: dict | None = None,
) -> SynDiffConfig:
    """Load site policy + deployment and apply optional run-level workspace override."""
    cfg = freeze_target_diff_config(site_config_path, target)
    return apply_workspace_run_id_override(cfg, meta)


def frozen_diff_config_for_context(ctx: StageRunContext) -> SynDiffConfig:
    """Frozen diff config for context.
    
    Parameters
    ----------
    ctx : StageRunContext
    
    Returns
    -------
    SynDiffConfig"""
    return frozen_diff_config_for_verify(
        resolve_diff_site_config_path(meta=ctx.meta, runner_cfg=ctx.runner_cfg),
        ctx.target,
        meta=ctx.meta,
    )


def load_diff_frames_for_verify(
    cfg: SynDiffConfig, event_dir: str | Path
) -> "pd.DataFrame":
    """
    Frame manifest for indexed diff verify from SCC ``bookkeeping/diff/frames.csv``.
    """
    import pandas as pd

    del event_dir
    data_root = getattr(cfg, "data_root", "") or ""
    if not data_root:
        raise RuntimeError(
            "load_diff_frames_for_verify requires deployment data_root on diff config"
        )
    from syndiff_pipeline.common.scc_paths import (
        normalize_store_name,
        resolve_scc_diff_bookkeeping_dir,
    )
    from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
        DIFF_JOB_BASENAME,
        FRAMES_CSV_BASENAME,
    )

    bk_dir = resolve_scc_diff_bookkeeping_dir(
        data_root,
        int(cfg.sector),
        int(cfg.camera),
        int(cfg.ccd),
        oversampling_factor=max(1, int(getattr(cfg, "oversampling_factor", 1) or 1)),
        template_store_name=normalize_store_name(
            getattr(cfg, "template_store_name", None)
        ),
    )
    frames_path = bk_dir / FRAMES_CSV_BASENAME
    job_path = bk_dir / DIFF_JOB_BASENAME
    if frames_path.is_file() and job_path.is_file():
        return pd.read_csv(frames_path)
    raise FileNotFoundError(
        f"SCC diff handoff missing under {bk_dir!r} "
        f"(need {FRAMES_CSV_BASENAME!r} and {DIFF_JOB_BASENAME!r})"
    )


def _diff_frame_manifest_available(cfg: SynDiffConfig, event_dir: str | Path) -> bool:
    """True when SCC bookkeeping frames + diff job exist."""
    del event_dir
    data_root = getattr(cfg, "data_root", "") or ""
    if not data_root:
        return False
    try:
        from syndiff_pipeline.common.scc_paths import (
            normalize_store_name,
            resolve_scc_diff_bookkeeping_dir,
        )
        from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
            DIFF_JOB_BASENAME,
            FRAMES_CSV_BASENAME,
        )

        bk_dir = resolve_scc_diff_bookkeeping_dir(
            data_root,
            int(cfg.sector),
            int(cfg.camera),
            int(cfg.ccd),
            oversampling_factor=max(1, int(getattr(cfg, "oversampling_factor", 1) or 1)),
            template_store_name=normalize_store_name(
                getattr(cfg, "template_store_name", None)
            ),
        )
        return (bk_dir / FRAMES_CSV_BASENAME).is_file() and (
            bk_dir / DIFF_JOB_BASENAME
        ).is_file()
    except Exception:
        log.debug("_diff_frame_manifest_available: SCC check failed", exc_info=True)
        return False


def diff_workspace_root(cfg: SynDiffConfig, event_dir: str | Path) -> Path:
    """Diff workspace root.
    
    Parameters
    ----------
    cfg : SynDiffConfig
    event_dir : str | Path
    
    Returns
    -------
    Path"""
    return Path(
        workspace_root(str(event_dir), run_id=getattr(cfg, "workspace_run_id", None))
    )


_NON_SCC_DIFF_STAGE_KINDS = frozenset(
    {"photometry", "astrometry", "forced_photometry"}
)


def _scc_lane_root(cfg: SynDiffConfig) -> Path | None:
    data_root = getattr(cfg, "data_root", "") or ""
    if not data_root:
        return None
    return scc_diff_dir(
        data_root,
        int(cfg.sector),
        int(cfg.camera),
        int(cfg.ccd),
        store_name=normalize_store_name(getattr(cfg, "output_store_name", None)),
    )


def _last_scc_executable_stage(cfg: SynDiffConfig) -> dict | None:
    """Last diff pipeline stage that writes SCC lane artifacts."""
    _, _, stages = split_pipeline(cfg.pipeline)
    for _idx, stage in reversed(stages):
        kind = str(stage.get("kind") or "").strip()
        if kind not in _NON_SCC_DIFF_STAGE_KINDS:
            return stage
    return None


def _scc_label_dir_has_fits(label_dir: Path) -> bool:
    if not label_dir.is_dir():
        return False
    return any(
        is_pipeline_fits_filename(p.name) for p in label_dir.rglob("*") if p.is_file()
    )


def _scc_final_stage_complete(cfg: SynDiffConfig, lane_root: Path) -> bool:
    stage = _last_scc_executable_stage(cfg)
    if stage is None:
        return False

    kind = stage.get("kind")

    if kind == "shared_mask":
        return (
            resolve_pipeline_artifact_path(str(lane_root), SHARED_MASK_FITS_BASENAME)
            is not None
        )

    if kind == "kernel_subtract":
        o = stage.get("output") or {}
        diffs = str(o.get("diffs", "")).strip()
        return bool(diffs) and _scc_label_dir_has_fits(lane_root / diffs)

    if kind == "background":
        label = str(stage.get("output", "")).strip()
        if not label:
            return False
        out_dir = lane_root / label
        if not out_dir.is_dir():
            return False
        if (out_dir / "stack.npz").is_file() or (out_dir / "stack.npy").is_file():
            return True
        return _scc_label_dir_has_fits(out_dir)

    if kind in ("subtract", "sat_template"):
        outputs = _outputs_for_stage(stage)
        return bool(outputs) and all(
            (lane_root / lab).is_dir() and any((lane_root / lab).rglob("*"))
            for lab in outputs
        )

    if kind == "epsf":
        label = str(stage["output"]).strip()
        epsf_dir = lane_root / label
        if not epsf_dir.is_dir():
            return False
        return any(epsf_dir.rglob("group_epsf_*.npy"))

    if kind == "centroids":
        label = str(stage["output"]).strip()
        centroids_dir = lane_root / label
        if not centroids_dir.is_dir():
            return False
        from syndiff_pipeline.difference_imaging.stages.centroids import (
            CENTROIDS_INDEX_BASENAME,
            PHOTRESULTS_ECSV_SUFFIX,
        )

        index_path = centroids_dir / CENTROIDS_INDEX_BASENAME
        if index_path.is_file():
            import json

            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
            return bool(index)
        return any(centroids_dir.glob(f"*{PHOTRESULTS_ECSV_SUFFIX}"))

    if kind == "per_ffi_wcs":
        label = str(stage["output"]).strip()
        wcs_dir = lane_root / label
        return (wcs_dir / "per_ffi_coeffs.csv").is_file()

    if kind == "kernel_fit":
        from syndiff_pipeline.difference_imaging.stages.kernel import (
            KERNEL_FIT_META_BASENAME,
            KERNEL_R2_NPZ_BASENAME,
        )

        label = str(stage.get("output", "")).strip()
        if not label:
            return False
        d = lane_root / label
        return (d / KERNEL_FIT_META_BASENAME).is_file() and (
            d / KERNEL_R2_NPZ_BASENAME
        ).is_file()

    if kind == "convolved_templates":
        label = str(stage.get("output", "")).strip()
        return bool(label) and (lane_root / label / "convolved_templates.csv").is_file()

    if kind == "hotpants":
        o = stage.get("output") or {}
        diffs = str(o.get("diffs", "")).strip()
        return bool(diffs) and _scc_label_dir_has_fits(lane_root / diffs)

    return False


def scc_diff_lane_complete(cfg: SynDiffConfig) -> bool:
    """True when SCC bookkeeping and final diff pipeline outputs exist on the lane."""
    lane_root = _scc_lane_root(cfg)
    if lane_root is None or not lane_root.is_dir():
        return False
    from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
        DIFF_JOB_BASENAME,
        FRAMES_CSV_BASENAME,
    )

    bk_dir = resolve_scc_diff_bookkeeping_dir(
        cfg.data_root,
        int(cfg.sector),
        int(cfg.camera),
        int(cfg.ccd),
        oversampling_factor=max(1, int(getattr(cfg, "oversampling_factor", 1) or 1)),
        template_store_name=normalize_store_name(getattr(cfg, "template_store_name", None)),
    )
    if not (bk_dir / DIFF_JOB_BASENAME).is_file():
        return False
    if not (bk_dir / FRAMES_CSV_BASENAME).is_file():
        return False
    return _scc_final_stage_complete(cfg, lane_root)


def _label_for_indexed_stage(stage: dict, kind: str) -> Optional[str]:
    """Workspace label a per-FFI provenance kind is keyed under, for one stage dict."""
    if kind == "shared_mask":
        return "shared_mask"
    if kind == "diff_image":
        o = stage.get("output") or {}
        label = str(o.get("diffs", "")).strip()
        return label or None
    if kind in ("epsf", "diff_background"):
        label = str(stage.get("output", "")).strip()
        return label or None
    return None


def _params_for_indexed_stage(stage: dict):
    """Reparse one stage dict into its strict param dataclass (registry §6 recipe source)."""
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
        parse_background,
        parse_epsf,
        parse_hotpants,
        parse_kernel_subtract,
        parse_shared_mask,
    )

    kind = stage.get("kind")
    try:
        if kind == "hotpants":
            return parse_hotpants(stage, 0)
        if kind == "kernel_subtract":
            return parse_kernel_subtract(stage, 0)
        if kind == "epsf":
            return parse_epsf(stage, 0)
        if kind == "shared_mask":
            return parse_shared_mask(stage, 0)
        if kind == "background":
            return parse_background(stage, 0)
    except Exception:
        log.debug("diff_stage_complete_indexed: stage param reparse failed", exc_info=True)
        return None
    return None


def _upstream_diff_image_stage(cfg: SynDiffConfig, stage: dict | None = None) -> Optional[dict]:
    """Hotpants/kernel_subtract stage that produced the diffs feeding *stage*."""
    _, _, stages = split_pipeline(cfg.pipeline)
    diffs_label = None
    if stage is not None:
        inp = stage.get("inputs") or {}
        raw = inp.get("diffs")
        if raw is not None and str(raw).strip():
            diffs_label = str(raw).strip()
    if diffs_label:
        for _, st in stages:
            k = st.get("kind")
            if k not in ("hotpants", "kernel_subtract"):
                continue
            o = st.get("output") or {}
            if str(o.get("diffs", "")).strip() == diffs_label:
                return st
    found = None
    for _, st in stages:
        if st.get("kind") in ("hotpants", "kernel_subtract"):
            found = st
    return found


def _diff_image_fingerprint_for_product(
    cfg: SynDiffConfig,
    frames_df,
    product_id: str,
    diff_stage: dict,
    *,
    downsample_fp: Optional[str] = None,
) -> Optional[str]:
    """Reconstruct one ``diff_image`` fingerprint (real edges, no ``loc:``)."""
    params = _params_for_indexed_stage(diff_stage)
    if params is None:
        return None
    label = _label_for_indexed_stage(diff_stage, "diff_image")
    if not label:
        return None
    ffi_dir = getattr(cfg, "ffi_dir", "") or ""
    ffi_path = _prov.resolve_ffi_path_for_product_id(
        frames_df, product_id, ffi_dir=ffi_dir or None
    )
    if not ffi_path:
        return None
    if downsample_fp is None:
        downsample_fp = _prov.resolve_downsample_fingerprint_from_cfg(cfg)
    if downsample_fp is None:
        return None
    sector, camera, ccd = int(cfg.sector), int(cfg.camera), int(cfg.ccd)
    inputs = _prov.diff_image_input_fingerprints(
        sector=sector,
        camera=camera,
        ccd=ccd,
        ffi_path=ffi_path,
        downsample_fp=downsample_fp,
    )
    if inputs is None:
        return None
    return _prov.diff_kind_fingerprint(
        "diff_image",
        sector=sector,
        camera=camera,
        ccd=ccd,
        product_id=product_id,
        label=label,
        params=params,
        input_fingerprints=inputs,
    )


def _indexed_input_fingerprints(
    *,
    cfg: SynDiffConfig,
    stage: dict,
    kind: str,
    frames_df,
    product_id: str,
    downsample_fp: Optional[str] = None,
) -> Optional[list[str]]:
    """Input-fingerprint vector matching emit for one indexed per-FFI kind."""
    sector, camera, ccd = int(cfg.sector), int(cfg.camera), int(cfg.ccd)
    ffi_dir = getattr(cfg, "ffi_dir", "") or ""

    if kind == "diff_background":
        from syndiff_pipeline.difference_imaging.stages.background.pipeline import (
            BackgroundParams,
        )

        params = _params_for_indexed_stage(stage)
        if isinstance(params, BackgroundParams):
            diff_stage = _upstream_diff_image_stage(cfg, stage)
            if diff_stage is None:
                return None
            diff_image_fp = _diff_image_fingerprint_for_product(
                cfg,
                frames_df,
                product_id,
                diff_stage,
                downsample_fp=downsample_fp,
            )
            return _prov.required_input_fingerprints(diff_image_fp)
        ffi_path = _prov.resolve_ffi_path_for_product_id(
            frames_df, product_id, ffi_dir=ffi_dir or None
        )
        if not ffi_path:
            return None
        ffi_fp = _prov.ffi_input_fingerprint(sector, camera, ccd, ffi_path)
        return _prov.diff_background_input_fingerprints(ffi_fp)

    if kind == "epsf":
        diff_stage = _upstream_diff_image_stage(cfg, stage)
        if diff_stage is None:
            return None
        diff_image_fp = _diff_image_fingerprint_for_product(
            cfg,
            frames_df,
            product_id,
            diff_stage,
            downsample_fp=downsample_fp,
        )
        return _prov.epsf_input_fingerprints(diff_image_fp)

    return None


def diff_stage_complete_indexed(
    cfg: SynDiffConfig, event_dir: str | Path, stage: dict
) -> Optional[bool]:
    """
    Indexed per-FFI completeness for one diff *stage* (PR-D1 primary path).

    Computes the stage's required per-FFI fingerprints from the frozen
    ``ws/diff_config.yaml``-resolved params (the same recipe
    ``emit_diff_artifact`` used at publish time) and the frame manifest's
    required-product-id set, then answers via
    ``ProvenanceStore.scc_stage_complete``.

    Returns ``True``/``False`` when the store can answer authoritatively, or
    ``None`` when it cannot — provenance package absent, ``cfg.data_root``
    unset, unsupported stage kind, empty required set, unreadable
    ``frames.csv``, or any store error. ``None`` means "fall open"; callers
    must fall back to the legacy marker check on ``None``, never treat it as
    False.
    """
    if _prov is None or not getattr(_prov, "PROVENANCE_AVAILABLE", False):
        return None
    if stage is None:
        return None
    kind = _INDEXED_STAGE_KIND.get(stage.get("kind"))
    if kind is None:
        return None
    data_root = getattr(cfg, "data_root", "") or ""
    if not data_root:
        return None
    label = _label_for_indexed_stage(stage, kind)
    if not label:
        return None
    params = _params_for_indexed_stage(stage)
    if params is None:
        return None

    try:
        frames_df = load_diff_frames_for_verify(cfg, event_dir)
    except Exception:
        log.debug("diff_stage_complete_indexed: frame manifest load failed", exc_info=True)
        return None

    pids = _prov.required_product_ids(frames_df)
    if not pids:
        return None

    sector, camera, ccd = int(cfg.sector), int(cfg.camera), int(cfg.ccd)
    downsample_fp = None
    if kind == "diff_image":
        downsample_fp = _prov.resolve_downsample_fingerprint_from_cfg(cfg)
        if downsample_fp is None:
            return None
    elif kind == "epsf":
        downsample_fp = _prov.resolve_downsample_fingerprint_from_cfg(cfg)
        if downsample_fp is None:
            return None
    elif kind == "diff_background":
        from syndiff_pipeline.difference_imaging.stages.background.pipeline import (
            BackgroundParams,
        )

        if isinstance(_params_for_indexed_stage(stage), BackgroundParams):
            downsample_fp = _prov.resolve_downsample_fingerprint_from_cfg(cfg)
            if downsample_fp is None:
                return None

    ffi_dir = getattr(cfg, "ffi_dir", "") or ""
    required_fps: list[str] = []
    for pid in pids:
        if kind == "shared_mask":
            fp = _prov.diff_kind_fingerprint_shared_mask(sector, camera, ccd, params, label=label)
        elif kind == "diff_image":
            ffi_path = _prov.resolve_ffi_path_for_product_id(
                frames_df, pid, ffi_dir=ffi_dir or None
            )
            if not ffi_path:
                return None
            inputs = _prov.diff_image_input_fingerprints(
                sector=sector,
                camera=camera,
                ccd=ccd,
                ffi_path=ffi_path,
                downsample_fp=downsample_fp,
            )
            if inputs is None:
                return None
            fp = _prov.diff_kind_fingerprint(
                kind,
                sector=sector,
                camera=camera,
                ccd=ccd,
                product_id=pid,
                label=label,
                params=params,
                input_fingerprints=inputs,
            )
        elif kind in ("diff_background", "epsf"):
            inputs = _indexed_input_fingerprints(
                cfg=cfg,
                stage=stage,
                kind=kind,
                frames_df=frames_df,
                product_id=pid,
                downsample_fp=downsample_fp,
            )
            if inputs is None:
                return None
            fp = _prov.diff_kind_fingerprint(
                kind,
                sector=sector,
                camera=camera,
                ccd=ccd,
                product_id=pid,
                label=label,
                params=params,
                input_fingerprints=inputs,
            )
        else:
            fp = _prov.diff_kind_fingerprint(
                kind,
                sector=sector,
                camera=camera,
                ccd=ccd,
                product_id=pid,
                label=label,
                params=params,
            )
        if fp is None:
            return None
        required_fps.append(fp)
        if kind == "shared_mask":
            # shared_mask has one node per SCC, not per FFI; one fingerprint suffices.
            break

    store = _prov.open_store(data_root)
    if store is None:
        return None
    try:
        return bool(store.scc_stage_complete(required_fps))
    except Exception:
        log.debug("diff_stage_complete_indexed: store query failed", exc_info=True)
        return None
    finally:
        _prov.close_store(store)


def diff_workspace_complete(cfg: SynDiffConfig, event_dir: str | Path) -> bool:
    """True when SCC diff lane bookkeeping and final pipeline outputs exist."""
    event_dir = Path(event_dir)
    if not _diff_frame_manifest_available(cfg, event_dir):
        return False

    data_root = getattr(cfg, "data_root", "") or ""
    if not data_root:
        return False

    try:
        indexed = diff_stage_complete_indexed(cfg, event_dir, _last_scc_executable_stage(cfg))
    except Exception:
        log.debug("diff_stage_complete_indexed raised; falling open", exc_info=True)
        indexed = None
    if indexed is not None:
        return indexed
    return scc_diff_lane_complete(cfg)


def collect_diff_workspace_artifacts(cfg: SynDiffConfig, event_dir: str | Path) -> list[str]:
    """List SCC diff lane artifact paths for diff manifest collection."""
    event_dir = Path(event_dir)
    artifacts: list[str] = []
    manifest_csv = manifest_path_from_output_dir(str(event_dir), None)
    if Path(manifest_csv).is_file():
        artifacts.append(manifest_csv)

    lane_root = _scc_lane_root(cfg)
    if lane_root is None or not lane_root.is_dir():
        return artifacts

    for path in sorted(lane_root.rglob("*")):
        if path.is_file():
            artifacts.append(str(path.resolve()))
    return artifacts


def verify_scc_diff(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    template_store_name: str | None = None,
    oversampling_factor: int = 1,
) -> list[str]:
    """
    Verify SCC-primary diff bookkeeping and template sidecar for field-mode diff.

    Returns a list of human-readable error strings (empty when OK).
    """
    import json

    from syndiff_pipeline.common.scc_paths import (
        resolve_scc_diff_bookkeeping_dir,
        scc_templates_dir,
    )
    from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
        DIFF_JOB_BASENAME,
        FIELD_MODE_ASSEMBLY_BASENAME,
        FRAMES_CSV_BASENAME,
    )

    errors: list[str] = []
    data_root = Path(data_root).expanduser()
    bk = resolve_scc_diff_bookkeeping_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=max(1, int(oversampling_factor)),
        template_store_name=template_store_name,
    )
    job_path = bk / DIFF_JOB_BASENAME
    frames_path = bk / FRAMES_CSV_BASENAME
    if not job_path.is_file():
        errors.append(f"missing {job_path}")
    if not frames_path.is_file():
        errors.append(f"missing {frames_path}")
    if job_path.is_file():
        try:
            doc = json.loads(job_path.read_text(encoding="utf-8"))
            if int(doc.get("schema_version", 0)) < 2:
                errors.append(f"{job_path}: schema_version < 2")
            if not doc.get("mapping_grid"):
                errors.append(f"{job_path}: missing mapping_grid")
        except Exception as exc:
            errors.append(f"{job_path}: invalid JSON ({exc})")
    tmpl = scc_templates_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=max(1, int(oversampling_factor)),
        store_name=template_store_name,
    )
    sidecar = tmpl / FIELD_MODE_ASSEMBLY_BASENAME
    if not sidecar.is_file():
        errors.append(f"missing template sidecar {sidecar}")
    elif sidecar.is_file():
        try:
            doc = json.loads(sidecar.read_text(encoding="utf-8"))
            if int(doc.get("schema_version", 0)) < 3:
                errors.append(f"{sidecar}: requires schema v3")
        except Exception as exc:
            errors.append(f"{sidecar}: invalid JSON ({exc})")
    return errors
