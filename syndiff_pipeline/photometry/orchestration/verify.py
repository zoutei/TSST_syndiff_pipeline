"""Photometry completion checks for orchestrator verification."""

from __future__ import annotations

import logging
from pathlib import Path

from syndiff_pipeline.common.scc_paths import (
    normalize_store_name,
    resolve_scc_diff_bookkeeping_dir,
    scc_diff_label_dir,
)
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    DIFF_JOB_BASENAME,
    FRAMES_CSV_BASENAME,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    is_pipeline_fits_filename,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    TARGETS_DS9_REGION_BASENAME,
    photometry_root,
)
from syndiff_pipeline.photometry.site_config import (
    PhotometryRunConfig,
    PhotometrySitePolicy,
    resolve_photometry_run_config,
)

log = logging.getLogger(__name__)


def scc_diff_lane_complete(
    run_config: PhotometryRunConfig,
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> bool:
    """True when SCC bookkeeping and required diff/ePSF lane dirs exist."""
    data_root = Path(data_root).expanduser()
    store_name = normalize_store_name(run_config.output_store_name)
    bk_dir = resolve_scc_diff_bookkeeping_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=run_config.oversampling_factor,
        template_store_name=run_config.template_store_name,
    )
    if not (bk_dir / DIFF_JOB_BASENAME).is_file():
        return False
    if not (bk_dir / FRAMES_CSV_BASENAME).is_file():
        return False

    diffs_dir = scc_diff_label_dir(
        data_root,
        sector,
        camera,
        ccd,
        store_name=store_name,
        label=run_config.diffs_label,
    )
    if not diffs_dir.is_dir():
        return False
    if not any(
        is_pipeline_fits_filename(p.name)
        for p in diffs_dir.rglob("*")
        if p.is_file()
    ):
        return False

    epsf_label = run_config.epsf_label
    if epsf_label:
        epsf_dir = scc_diff_label_dir(
            data_root,
            sector,
            camera,
            ccd,
            store_name=store_name,
            label=epsf_label,
        )
        if not epsf_dir.is_dir():
            return False
        index = epsf_dir / "gridded_epsf_index.json"
        if not index.is_file():
            return False
    return True


def _forced_photometry_stage(pipeline: list) -> dict | None:
    for stage in pipeline:
        if isinstance(stage, dict) and stage.get("kind") == "forced_photometry":
            return stage
    return None


def photometry_complete(
    run_config: PhotometryRunConfig,
    event_dir: str | Path,
    photometry_run_id: str | None = None,
) -> bool:
    """True when photometry LC CSVs exist under ``phot_{run_id}/``."""
    from syndiff_pipeline.difference_imaging.stages.photometry import lightcurve_csv_basename

    event_dir = Path(event_dir)
    run_id = photometry_run_id if photometry_run_id is not None else run_config.photometry_run_id
    phot_root = Path(photometry_root(str(event_dir), run_id))
    if not phot_root.is_dir():
        return False

    stage = _forced_photometry_stage(run_config.pipeline)
    if stage is None:
        # astrometry-only photometry configs are complete when astrometry JSON exists
        astro = phot_root / "astrometry_result.json"
        return astro.is_file()

    label = str(stage.get("output", "")).strip()
    if not label:
        return False
    phot_out = phot_root / label
    if not phot_out.is_dir():
        return False

    methods = stage.get("methods") or []
    if not methods:
        return False
    extras = run_config.additional_forced_targets or []
    for entry in methods:
        if not isinstance(entry, dict):
            return False
        name = str(entry.get("name", "")).strip()
        if not name:
            return False
        primary_csv = lightcurve_csv_basename(name)
        if not (phot_out / primary_csv).is_file():
            return False
        for pt in extras:
            if not isinstance(pt, dict):
                continue
            extra_name = str(pt.get("name", "")).strip()
            if not extra_name:
                continue
            extra_csv = lightcurve_csv_basename(name, extra_name)
            if not (phot_out / extra_csv).is_file():
                return False
    return True


def verify_photometry_prerequisites(
    policy: PhotometrySitePolicy,
    target,
    *,
    site_dir: str | Path,
) -> tuple[bool, str]:
    """Check SCC lane + target coordinates before photometry execution."""
    run_config = resolve_photometry_run_config(policy, target, site_dir=site_dir)
    import numpy as np

    if not (
        target.target_ra is not None
        and target.target_dec is not None
        and np.isfinite(float(target.target_ra))
        and np.isfinite(float(target.target_dec))
    ):
        return False, "target requires finite target_ra and target_dec"

    from syndiff_pipeline.difference_imaging.orchestration.site_config import (
        load_deployment_for_diff_config,
    )

    _, deploy_path = load_deployment_for_diff_config(policy.config_path)
    deployment = __import__(
        "syndiff_pipeline.common.orchestration.deployment", fromlist=["load_deployment"]
    ).load_deployment(policy.config_path, policy.deployment_file)
    data_root = deployment.get("data_root")
    if not data_root:
        return False, "deployment missing data_root"
    if not scc_diff_lane_complete(
        run_config,
        data_root=data_root,
        sector=int(target.sector),
        camera=int(target.camera),
        ccd=int(target.ccd),
    ):
        return False, "SCC diff lane incomplete for configured photometry inputs"
    return True, ""


def collect_photometry_artifacts(
    run_config: PhotometryRunConfig,
    event_dir: str | Path,
    photometry_run_id: str | None = None,
) -> list[str]:
    """Collect photometry output paths for manifest writing."""
    from syndiff_pipeline.difference_imaging.stages.photometry import lightcurve_csv_basename

    event_dir = Path(event_dir)
    run_id = photometry_run_id if photometry_run_id is not None else run_config.photometry_run_id
    phot_root = Path(photometry_root(str(event_dir), run_id))
    artifacts: list[str] = []
    targets_reg = phot_root / TARGETS_DS9_REGION_BASENAME
    if targets_reg.is_file():
        artifacts.append(str(targets_reg.resolve()))
    astro = phot_root / "astrometry_result.json"
    if astro.is_file():
        artifacts.append(str(astro.resolve()))

    stage = _forced_photometry_stage(run_config.pipeline)
    if stage is None:
        return artifacts

    label = str(stage.get("output", "")).strip()
    phot_out = phot_root / label
    if not phot_out.is_dir():
        return artifacts
    methods = stage.get("methods") or []
    extras = run_config.additional_forced_targets or []
    for entry in methods:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        primary = phot_out / lightcurve_csv_basename(name)
        if primary.is_file():
            artifacts.append(str(primary.resolve()))
        for pt in extras:
            if not isinstance(pt, dict):
                continue
            extra_name = str(pt.get("name", "")).strip()
            if not extra_name:
                continue
            extra = phot_out / lightcurve_csv_basename(name, extra_name)
            if extra.is_file():
                artifacts.append(str(extra.resolve()))
    return artifacts
