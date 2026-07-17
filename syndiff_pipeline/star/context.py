"""Event context and prerequisite validation for ``syndiff star run``."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from syndiff_pipeline.common.orchestration.deployment import (
    load_deployment_file,
    require_deployment_path,
)
from syndiff_pipeline.common.orchestration.event_ws_symlinks import workspace_tree_path
from syndiff_pipeline.common.orchestration.targets import Target, find_target, load_targets
from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    SitePaths,
    _gaia_catalog_path,
    freeze_target_diff_config,
    load_diff_site_policy,
    resolve_event_template_dir,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import frame_kernels_dir
from syndiff_pipeline.difference_imaging.support.paths import (
    DEFAULT_MANIFEST_BASENAME,
    STATIC_MASK_FITS_BASENAME,
    normalize_workspace_run_id,
    workspace_dir,
)
from syndiff_pipeline.star.site_config import StarRunConfig, StarTargetRow
from syndiff_pipeline.star.identifiers import ResolvedHost

_DEFAULT_MAPPING_ROOT_NAME = "skycell_pixel_mapping"


class StarPrerequisiteError(Exception):
    """Raised when one or more star-run prerequisites are missing."""


@dataclass
class StarEventContext:
    target: Target
    event_dir: str
    workspace_root: str
    data_root: str
    cluster_job_path: str
    cluster_job: dict
    crop_bounds: dict
    mapping_dir: str
    mapping_csv: str
    master_mapping_fits: str
    gaia_catalog_path: str
    templates_dir: str
    reference_ffi_path: str
    sector: int
    camera: int
    ccd: int
    baseline_workspace_dir: str
    baseline_diffs_label: str
    baseline_convolved_dir: str
    baseline_phot_bkg_dir: str
    baseline_phot_bkg_label: str
    baseline_kernels_dir: str


def _resolve_hotpants_output_labels(
    *,
    site_dir: Path,
    diffs_label: str | None,
) -> tuple[str, str | None, str | None]:
    policy = load_diff_site_policy(site_dir / "diff_config.yaml")
    hotpants_stages = [
        stage for stage in policy.pipeline if stage.get("kind") == "hotpants"
    ]
    if not hotpants_stages:
        raise ValueError("diff_config.yaml has no hotpants stage")

    if diffs_label is None:
        stage = hotpants_stages[-1]
    else:
        matches = [
            stage
            for stage in hotpants_stages
            if str((stage.get("output") or {}).get("diffs", "")).strip() == diffs_label
        ]
        if not matches:
            labels = [
                str((stage.get("output") or {}).get("diffs", "")).strip()
                for stage in hotpants_stages
            ]
            raise ValueError(
                f"No hotpants stage with output.diffs={diffs_label!r}; "
                f"available labels: {labels}"
            )
        stage = matches[-1]

    output = stage.get("output") or {}
    resolved_diffs = str(output.get("diffs", "")).strip()
    if not resolved_diffs:
        raise ValueError("hotpants stage output.diffs is empty in diff_config.yaml")
    convolved = str(output.get("convolved", "")).strip() or None
    bkg = str(output.get("bkg", "")).strip() or None
    return resolved_diffs, convolved, bkg


def _load_yaml_pipeline(config_path: Path) -> list[dict]:
    if not config_path.is_file():
        return []
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    pipeline = raw.get("pipeline")
    if not isinstance(pipeline, list):
        return []
    return [stage for stage in pipeline if isinstance(stage, dict)]


def _photutils_bkg_from_pipeline(
    stages: list[dict],
    *,
    diffs_label: str,
) -> str | None:
    """Resolve photutils background label from diff pipeline stages."""
    kernel_subtract_label: str | None = None
    for stage in stages:
        kind = str(stage.get("kind") or "").strip()
        if kind == "hotpants":
            output = stage.get("output") or {}
            stage_diffs = str(output.get("diffs", "")).strip()
            if stage_diffs != diffs_label:
                continue
            inp_bkg = str((stage.get("inputs") or {}).get("bkg") or "").strip()
            if inp_bkg:
                return inp_bkg
        if kind == "kernel_subtract":
            phot_bkg = str((stage.get("output") or {}).get("phot_bkg") or "").strip()
            if phot_bkg:
                kernel_subtract_label = phot_bkg
    return kernel_subtract_label


def _resolve_photutils_bkg_label(
    *,
    site_dir: Path,
    baseline_workspace_dir: str,
    diffs_label: str,
) -> str:
    """
    Workspace label for per-frame photutils background (e.g. ``ks_b`` / ``ks_b_s``).

    Star stamps subtract this map from raw science. Hotpants ``hp_b`` is not used.
    """
    ws_dir = Path(baseline_workspace_dir)
    config_paths = [
        ws_dir / "diff_config.yaml",
        site_dir / "diff_config.yaml",
    ]
    candidates: list[str] = []
    for config_path in config_paths:
        label = _photutils_bkg_from_pipeline(
            _load_yaml_pipeline(config_path),
            diffs_label=diffs_label,
        )
        if label:
            candidates.append(label)

    for fallback in ("ks_b_s", "ks_b"):
        if fallback not in candidates:
            candidates.append(fallback)

    for label in candidates:
        if (ws_dir / label).is_dir():
            return label

    tried = ", ".join(repr(label) for label in candidates)
    raise ValueError(
        f"No photutils background workspace found under {ws_dir} "
        f"(tried {tried}). Run kernel_subtract (ks_b) and/or inherit ks_b_s "
        "before star run."
    )


def _mapping_paths(data_root: str, target: Target) -> tuple[str, str, str]:
    mapping_dir = Path(data_root) / _DEFAULT_MAPPING_ROOT_NAME
    rel = f"sector_{target.sector:04d}/camera_{target.camera}/ccd_{target.ccd}"
    stem = f"tess_s{target.sector:04d}_{target.camera}_{target.ccd}"
    mapping_csv = str(mapping_dir / rel / f"{stem}_master_skycells_list.csv")
    master_mapping_fits = str(
        mapping_dir / rel / f"{stem}_master_pixels2skycells.fits.gz"
    )
    return str(mapping_dir), mapping_csv, master_mapping_fits


def _coords_missing(target: Target) -> bool:
    return abs(target.target_ra) < 1e-9 and abs(target.target_dec) < 1e-9


def _target_coords_from_event_diff_configs(event_dir: Path) -> tuple[float, float] | None:
    candidates = [event_dir / "ws" / "diff_config.yaml"]
    candidates.extend(sorted(event_dir.glob("ws_*/diff_config.yaml")))
    for path in candidates:
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        ra = data.get("target_ra")
        dec = data.get("target_dec")
        if ra is not None and dec is not None:
            return float(ra), float(dec)
    return None


def _enrich_star_target_coords(
    target: Target,
    *,
    targets_csv: str | None,
    target_name: str,
    workspace_root: str,
) -> Target:
    """Fill placeholder (0, 0) coords from transient targets or frozen event configs."""
    if not _coords_missing(target):
        return target
    if targets_csv:
        try:
            main = find_target(load_targets(targets_csv), target_name)
            if not _coords_missing(main):
                return replace(
                    target,
                    target_ra=main.target_ra,
                    target_dec=main.target_dec,
                )
        except (KeyError, ValueError):
            pass
    event_dir = Path(workspace_root).expanduser() / "events" / target.label()
    coords = _target_coords_from_event_diff_configs(event_dir)
    if coords is not None:
        ra, dec = coords
        return replace(target, target_ra=ra, target_dec=dec)
    return target


def load_event_context(
    *,
    site: str,
    targets_csv: str | None = None,
    target_name: str,
    baseline_workspace_run_id: str = "none",
    baseline_diffs_label: str | None = None,
    baseline_convolved_label: str | None = None,
    baseline_phot_bkg_label: str | None = None,
    star_run_config: StarRunConfig | None = None,
    star_target_row: StarTargetRow | None = None,
) -> StarEventContext:
    """Resolve site/target inputs into a :class:`StarEventContext`."""
    paths = SitePaths.from_site_dir(site)

    if star_target_row is not None:
        target = star_target_row.target
    else:
        if not targets_csv:
            raise ValueError("targets_csv is required when star_target_row is not provided")
        targets = load_targets(targets_csv)
        target = find_target(targets, target_name)

    if star_run_config is not None:
        baseline_workspace_run_id = star_run_config.baseline.workspace_run_id
        baseline_diffs_label = star_run_config.baseline.diffs
        baseline_convolved_label = star_run_config.baseline.convolved
        baseline_phot_bkg_label = star_run_config.baseline.phot_bkg

    deployment = load_deployment_file(paths.deployment)
    workspace_root = require_deployment_path(
        deployment, "workspace_root", deployment_path=paths.deployment
    )
    data_root = require_deployment_path(
        deployment, "data_root", deployment_path=paths.deployment
    )

    target = _enrich_star_target_coords(
        target,
        targets_csv=targets_csv,
        target_name=target_name,
        workspace_root=workspace_root,
    )

    cfg = freeze_target_diff_config(
        paths.diff_config,
        target,
        deployment_path=paths.deployment if paths.deployment.is_file() else None,
    )

    event_dir = str(Path(cfg.output_dir).expanduser().resolve())
    cluster_job_path = str(Path(event_dir) / wcs_grouping.CLUSTER_TEMPLATE_JOB_FILENAME)
    with Path(cluster_job_path).open(encoding="utf-8") as fh:
        cluster_job = json.load(fh)
    crop_bounds = wcs_grouping.resolve_diff_crop_bounds(cfg, event_dir)

    reference_ffi_path = wcs_grouping.load_reference_ffi_path(event_dir)
    if not reference_ffi_path:
        raise FileNotFoundError(
            f"Missing reference_ffi_path in {cluster_job_path}"
        )

    try:
        templates_dir = resolve_event_template_dir(event_dir)
    except FileNotFoundError:
        templates_dir = str(cfg.template_dir)

    gaia_catalog_path = str(cfg.gaia_catalog)
    if not gaia_catalog_path:
        gaia_catalog_path = str(
            _gaia_catalog_path(
                target,
                data_root=Path(data_root),
                event_dir=Path(event_dir),
                catalog_root="catalogs",
            )
        )

    mapping_dir, mapping_csv, master_mapping_fits = _mapping_paths(data_root, target)

    baseline_run_id = normalize_workspace_run_id(baseline_workspace_run_id)
    baseline_workspace_dir = str(
        workspace_tree_path(event_dir, run_id=baseline_run_id)
    )
    diffs_label, convolved_label, _hp_bkg_label = _resolve_hotpants_output_labels(
        site_dir=paths.site_dir,
        diffs_label=baseline_diffs_label,
    )
    if baseline_convolved_label:
        convolved_label = baseline_convolved_label
    if baseline_phot_bkg_label:
        phot_bkg_label = baseline_phot_bkg_label
    else:
        phot_bkg_label = _resolve_photutils_bkg_label(
            site_dir=paths.site_dir,
            baseline_workspace_dir=baseline_workspace_dir,
            diffs_label=diffs_label,
        )
    baseline_diffs_dir = workspace_dir(
        event_dir, diffs_label, run_id=baseline_run_id
    )
    baseline_convolved_dir = (
        workspace_dir(event_dir, convolved_label, run_id=baseline_run_id)
        if convolved_label
        else ""
    )
    baseline_phot_bkg_dir = workspace_dir(
        event_dir, phot_bkg_label, run_id=baseline_run_id
    )
    baseline_kernels_dir = frame_kernels_dir(baseline_diffs_dir)

    return StarEventContext(
        target=target,
        event_dir=event_dir,
        workspace_root=str(Path(workspace_root).expanduser().resolve()),
        data_root=str(Path(data_root).expanduser().resolve()),
        cluster_job_path=cluster_job_path,
        cluster_job=cluster_job,
        crop_bounds=crop_bounds,
        mapping_dir=mapping_dir,
        mapping_csv=mapping_csv,
        master_mapping_fits=master_mapping_fits,
        gaia_catalog_path=gaia_catalog_path,
        templates_dir=templates_dir,
        reference_ffi_path=str(reference_ffi_path),
        sector=target.sector,
        camera=target.camera,
        ccd=target.ccd,
        baseline_workspace_dir=baseline_workspace_dir,
        baseline_diffs_label=diffs_label,
        baseline_convolved_dir=baseline_convolved_dir,
        baseline_phot_bkg_dir=baseline_phot_bkg_dir,
        baseline_phot_bkg_label=phot_bkg_label,
        baseline_kernels_dir=baseline_kernels_dir,
    )


def full_ffi_to_crop_local(
    ctx: StarEventContext,
    x_ref: float,
    y_ref: float,
) -> tuple[float, float]:
    """Convert full-FFI pixel coordinates to crop-local pixels."""
    x_min = int(ctx.crop_bounds["x_min"])
    y_min = int(ctx.crop_bounds["y_min"])
    return float(x_ref - x_min), float(y_ref - y_min)


def resolve_host_full_ffi_xy(
    ctx: StarEventContext,
    host: ResolvedHost,
) -> tuple[float, float]:
    """Project host RA/Dec to full-FFI pixel coordinates via the reference FFI WCS."""
    from astropy.wcs import WCS

    with wcs_grouping.open_fits_memmap(ctx.reference_ffi_path) as hdul:
        wcs = WCS(hdul[1].header)
    return wcs_grouping.world_ra_dec_to_pixel(wcs, host.ra, host.dec)


def _dir_has_glob(path: str, pattern: str) -> bool:
    root = Path(path)
    if not root.is_dir():
        return False
    return any(root.glob(pattern))


def validate_star_prerequisites(ctx: StarEventContext) -> None:
    """Fail fast with an actionable checklist when prerequisites are missing."""
    missing: list[str] = []

    cluster_path = Path(ctx.cluster_job_path)
    if not cluster_path.is_file():
        missing.append(
            f"cluster_template_job.json missing at {cluster_path}; "
            "run wcs_grouping for this event"
        )

    manifest_path = Path(ctx.event_dir) / DEFAULT_MANIFEST_BASENAME
    if not manifest_path.is_file():
        missing.append(
            f"syndiff_ffi_frames.csv missing at {manifest_path}; "
            "run wcs_grouping for this event"
        )

    templates_path = Path(ctx.templates_dir)
    if not _dir_has_glob(str(templates_path), "syndiff_template_*"):
        missing.append(
            f"no syndiff_template_* files under {templates_path}; "
            "run template downsample for this event"
        )

    diffs_dir = Path(ctx.baseline_workspace_dir) / ctx.baseline_diffs_label
    if not _dir_has_glob(str(diffs_dir), "*.fits*"):
        missing.append(
            f"no baseline diff FITS under {diffs_dir}; "
            f"complete the baseline hotpants stage ({ctx.baseline_diffs_label}) "
            "for this event"
        )

    conv_dir = Path(ctx.baseline_convolved_dir)
    if not ctx.baseline_convolved_dir or not _dir_has_glob(str(conv_dir), "*.fits*"):
        missing.append(
            f"no convolved-template FITS under {conv_dir or '(unset)'}; "
            "re-run the baseline hotpants stage with write_convolved: true"
        )

    phot_bkg_dir = Path(ctx.baseline_phot_bkg_dir)
    if not ctx.baseline_phot_bkg_dir or not _dir_has_glob(str(phot_bkg_dir), "*.fits*"):
        missing.append(
            f"no photutils background FITS under {phot_bkg_dir or '(unset)'}; "
            f"ensure baseline.phot_bkg={ctx.baseline_phot_bkg_label!r} exists "
            "(run kernel_subtract ks_b and/or inherit ks_b_s in the baseline workspace)"
        )

    kernels_dir = Path(ctx.baseline_kernels_dir)
    if not _dir_has_glob(str(kernels_dir), "*_kernel.npz"):
        missing.append(
            f"no *_kernel.npz files under {kernels_dir}; "
            "re-run the baseline hotpants stage with write_kernel_solutions: true"
        )

    static_mask = Path(ctx.baseline_workspace_dir) / STATIC_MASK_FITS_BASENAME
    if not static_mask.is_file():
        missing.append(
            f"static mask ({STATIC_MASK_FITS_BASENAME}) missing at {static_mask}; "
            "run the shared_mask diff stage for this event"
        )

    mapping_csv = Path(ctx.mapping_csv)
    if not mapping_csv.is_file():
        missing.append(
            f"mapping CSV missing at {mapping_csv}; "
            "run the mapping stage for this sector/camera/ccd"
        )

    master_mapping = Path(ctx.master_mapping_fits)
    if not master_mapping.is_file():
        missing.append(
            f"master_pixels2skycells FITS missing at {master_mapping}; "
            "run the mapping stage for this sector/camera/ccd"
        )

    gaia_catalog = Path(ctx.gaia_catalog_path)
    if not gaia_catalog.is_file():
        missing.append(
            f"Gaia catalog CSV missing at {gaia_catalog}; "
            "ensure the SCC Gaia catalog exists under data_root/catalogs"
        )

    if missing:
        lines = "\n".join(f"  - {item}" for item in missing)
        raise StarPrerequisiteError(
            "Star run prerequisites missing:\n" + lines
        )
