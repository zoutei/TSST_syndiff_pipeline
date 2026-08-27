"""Event context and prerequisite validation for ``syndiff star run``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from syndiff_pipeline.common.fits_variants import try_resolve_fits_variant
from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.common.scc_paths import (
    normalize_store_name,
    resolve_scc_diff_bookkeeping_dir,
    scc_diff_dir,
    scc_diff_label_dir,
)
from syndiff_pipeline.common.orchestration.deployment import (
    load_deployment_file,
    require_deployment_path,
)
from syndiff_pipeline.common.orchestration.event_ws_symlinks import workspace_tree_path
from syndiff_pipeline.common.orchestration.targets import Target, find_target, load_targets
from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    DIFF_JOB_BASENAME,
    FRAMES_CSV_BASENAME,
    _resolve_reference_ffi_path,
)
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    DiffSitePolicy,
    SitePaths,
    _gaia_catalog_path,
    freeze_target_diff_config,
    load_diff_site_policy,
    resolve_scc_template_dir,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import frame_kernels_dir
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    PIPELINE_FITS_EXT,
    resolve_pipeline_artifact_path,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    DEFAULT_MANIFEST_BASENAME,
    SHARED_MASK_FITS_BASENAME,
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
    baseline_diffs_dir: str
    baseline_convolved_dir: str
    baseline_phot_bkg_dir: str
    baseline_phot_bkg_label: str
    baseline_kernels_dir: str
    oversampling_factor: int = 1
    mapping_grid: MappingGrid | None = None
    output_store_name: str | None = None


def _resolve_hotpants_output_labels(
    *,
    site_dir: Path,
    diffs_label: str | None,
    diff_policy: DiffSitePolicy | None = None,
) -> tuple[str, str | None, str | None]:
    """Resolve hotpants ``(diffs, convolved, bkg)`` output labels.

    Prefers *diff_policy* -- the same :class:`DiffSitePolicy` the site's
    ``diff_config:`` pointer (or embedded v2 ``diff:`` block) resolves to for
    ``syndiff diff submit`` -- so star and diff agree on one recipe. Falls
    back to a hardcoded ``{site_dir}/diff_config.yaml`` read when no policy is
    supplied (legacy path; a later wave removes it).
    """
    if diff_policy is None:
        diff_policy = load_diff_site_policy(site_dir / "diff_config.yaml")
    hotpants_stages = [
        stage for stage in diff_policy.pipeline if stage.get("kind") == "hotpants"
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
    background_estimate_label: str | None = None
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
        if kind == "background_estimate":
            phot_bkg = str((stage.get("output") or {}).get("phot_bkg") or "").strip()
            if phot_bkg:
                background_estimate_label = phot_bkg
    return background_estimate_label


def _label_dir_has_fits(
    *,
    label: str,
    baseline_workspace_dir: str,
    data_root: str | None = None,
    target: Target | None = None,
    output_store_name: str | None = None,
) -> bool:
    """True when *label* exists under the flat SCC diff lane."""
    del baseline_workspace_dir  # SCC-only; no event workspace fallback
    if data_root is None or target is None or not label:
        return False
    lane_dir = scc_diff_label_dir(
        data_root,
        target.sector,
        target.camera,
        target.ccd,
        store_name=output_store_name,
        label=label,
    )
    if not lane_dir.is_dir():
        return False
    return any(
        p.is_file() and p.name.endswith((".fits", ".fits.fz", ".fits.gz"))
        for p in lane_dir.rglob("*")
    )


def _resolve_photutils_bkg_label(
    *,
    site_dir: Path,
    baseline_workspace_dir: str,
    diffs_label: str,
    data_root: str | None = None,
    target: Target | None = None,
    output_store_name: str | None = None,
    diff_policy: DiffSitePolicy | None = None,
) -> str:
    """
    Workspace label for per-frame photutils background (e.g. ``ks_b`` / ``ks_b_s``).

    Star stamps subtract this map from raw science. Hotpants ``hp_b`` is not used.
    Prefers SCC ``diff_{lane}/`` when present.

    The immutable per-lane snapshot ``{baseline_workspace_dir}/diff_config.yaml``
    is always read from disk -- it is a workspace lock file, not the site's
    live/pointer-resolved recipe. The *site*-level recipe comes from
    *diff_policy* when supplied (the same policy ``syndiff diff submit``
    resolves via ``diff_config:``/embedded ``diff:``), falling back to a
    hardcoded ``{site_dir}/diff_config.yaml`` read when no policy is given
    (legacy path; a later wave removes it).
    """
    ws_dir = Path(baseline_workspace_dir)
    pipelines: list[list[dict]] = [_load_yaml_pipeline(ws_dir / "diff_config.yaml")]
    if diff_policy is not None:
        pipelines.append(list(diff_policy.pipeline))
    else:
        pipelines.append(_load_yaml_pipeline(site_dir / "diff_config.yaml"))
    candidates: list[str] = []
    for pipeline in pipelines:
        label = _photutils_bkg_from_pipeline(
            pipeline,
            diffs_label=diffs_label,
        )
        if label:
            candidates.append(label)

    for fallback in ("ks_b_s", "ks_b"):
        if fallback not in candidates:
            candidates.append(fallback)

    for label in candidates:
        if _label_dir_has_fits(
            label=label,
            baseline_workspace_dir=baseline_workspace_dir,
            data_root=data_root,
            target=target,
            output_store_name=output_store_name,
        ):
            return label

    tried = ", ".join(repr(label) for label in candidates)
    raise ValueError(
        f"No photutils background workspace found under SCC lane "
        f"(tried {tried}). Run background_estimate (ks_b) and/or inherit ks_b_s "
        "before star run."
    )


def _mapping_paths(
    data_root: str,
    target: Target,
    *,
    oversampling_factor: int = 1,
) -> tuple[str, str, str]:
    from syndiff_pipeline.common.scc_paths import (
        scc_mapping_dir,
        scc_mapping_master_pixels2skycells,
        scc_mapping_master_skycells_csv,
    )

    os_factor = max(1, int(oversampling_factor))
    mapping_dir = scc_mapping_dir(
        data_root,
        target.sector,
        target.camera,
        target.ccd,
        oversampling_factor=os_factor,
    )
    mapping_csv = str(
        scc_mapping_master_skycells_csv(
            data_root,
            target.sector,
            target.camera,
            target.ccd,
            oversampling_factor=os_factor,
        )
    )
    master_mapping_fits = str(
        scc_mapping_master_pixels2skycells(
            data_root,
            target.sector,
            target.camera,
            target.ccd,
            oversampling_factor=os_factor,
        )
    )
    resolved_master = try_resolve_fits_variant(master_mapping_fits)
    if resolved_master is not None:
        master_mapping_fits = str(resolved_master)
    # Fall back to legacy flat mapping tree when SCC nest is absent (older data_roots).
    if not Path(mapping_csv).is_file():
        legacy_dir = Path(data_root) / _DEFAULT_MAPPING_ROOT_NAME
        rel = f"sector_{target.sector:04d}/camera_{target.camera}/ccd_{target.ccd}"
        stem = f"tess_s{target.sector:04d}_{target.camera}_{target.ccd}"
        suffix = f"_os{os_factor}" if os_factor > 1 else ""
        legacy_csv = legacy_dir / rel / f"{stem}_master_skycells_list{suffix}.csv"
        legacy_fits = legacy_dir / rel / f"{stem}_master_pixels2skycells{suffix}{PIPELINE_FITS_EXT}"
        resolved_legacy = try_resolve_fits_variant(legacy_fits)
        if legacy_csv.is_file():
            return (
                str(legacy_dir),
                str(legacy_csv),
                str(resolved_legacy) if resolved_legacy is not None else str(legacy_fits),
            )
    return str(mapping_dir), mapping_csv, master_mapping_fits


def _load_scc_diff_job(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
    template_store_name: str | None = None,
) -> dict | None:
    job_path = (
        resolve_scc_diff_bookkeeping_dir(
            data_root,
            sector,
            camera,
            ccd,
            oversampling_factor=max(1, int(oversampling_factor)),
            template_store_name=normalize_store_name(template_store_name),
        )
        / "diff_job.json"
    )
    if not job_path.is_file():
        return None
    try:
        doc = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(doc.get("schema_version", 0)) < 2:
        return None
    return doc


def _resolve_baseline_label_dir(
    *,
    data_root: str,
    target: Target,
    event_dir: str,
    label: str,
    baseline_run_id: str,
    output_store_name: str | None,
) -> str:
    """Return flat SCC ``diff_{lane}/{label}/`` directory for one pipeline label."""
    del event_dir, baseline_run_id  # SCC-only reads
    if not label:
        return ""
    lane_dir = scc_diff_label_dir(
        data_root,
        target.sector,
        target.camera,
        target.ccd,
        store_name=output_store_name,
        label=label,
    )
    return str(lane_dir.resolve())


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
    diff_policy: DiffSitePolicy | None = None,
) -> StarEventContext:
    """Resolve site/target inputs into a :class:`StarEventContext`.

    *diff_policy*, when supplied, is the site's resolved diff recipe -- the
    same :class:`DiffSitePolicy` ``syndiff diff submit`` uses (via
    ``diff_config:``/``diff_site_config:``/``diff_config_path`` or an
    embedded v2 ``diff:`` block) -- and is threaded into hotpants output
    label and photutils-background label resolution so star and diff always
    agree on one recipe. When omitted, both fall back to reading
    ``{site}/diff_config.yaml`` directly (legacy path; a later wave removes
    it).
    """
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

    cfg = freeze_target_diff_config(
        paths.diff_config,
        target,
        deployment_path=paths.deployment if paths.deployment.is_file() else None,
    )

    event_dir = str(Path(cfg.output_dir).expanduser().resolve())

    os_factor = 1
    template_store_name = None
    if star_run_config is not None:
        os_factor = max(1, int(star_run_config.oversampling_factor or 1))
        template_store_name = star_run_config.template_store_name

    scc_bookkeeping_dir = resolve_scc_diff_bookkeeping_dir(
        data_root,
        target.sector,
        target.camera,
        target.ccd,
        oversampling_factor=os_factor,
        template_store_name=normalize_store_name(template_store_name),
    )

    diff_job = _load_scc_diff_job(
        data_root,
        target.sector,
        target.camera,
        target.ccd,
        oversampling_factor=os_factor,
        template_store_name=template_store_name,
    )
    mapping_grid = None
    output_store_name = None
    if diff_job is not None:
        # Modern SCC-bootstrapped lane: bookkeeping/diff_{lane}/oversampling_N/
        # holds both diff_job.json (already loaded above) and frames.csv.
        cluster_job_path = str(scc_bookkeeping_dir / DIFF_JOB_BASENAME)
        mapping_grid = MappingGrid.from_mapping_dict(diff_job["mapping_grid"])
        crop_bounds = diff_job.get("crop_bounds") or mapping_grid.science_ffi_bounds()
        output_store_name = diff_job.get("output_store_name")
        cluster_job = diff_job
        frames_csv_path = scc_bookkeeping_dir / FRAMES_CSV_BASENAME
        frames_df = (
            pd.read_csv(frames_csv_path) if frames_csv_path.is_file() else pd.DataFrame()
        )
        reference_ffi_path = _resolve_reference_ffi_path(
            Path(data_root),
            target.sector,
            target.camera,
            target.ccd,
            frames_df,
        )
    else:
        # Legacy event-dir handoff (pre-SCC-bootstrap lane); run_wcs_grouping has
        # zero callers repo-wide, so this branch is a dead-writer fallback kept
        # only for older data_roots that already have the files on disk.
        cluster_job_path = str(Path(event_dir) / wcs_grouping.EVENT_JOB_FILENAME)
        with Path(wcs_grouping._event_job_path(event_dir)).open(encoding="utf-8") as fh:
            cluster_job = json.load(fh)
        # crop_mode/crop_box_size were removed (wave A-3): the diff crop is
        # always the cluster job's full science bounds, never overridable
        # per-config. Was ``wcs_grouping.resolve_diff_crop_bounds(cfg, event_dir)``,
        # which (with those knobs gone) always fell through to this same call.
        crop_bounds = wcs_grouping.load_crop_bounds(event_dir)
        reference_ffi_path = wcs_grouping.load_reference_ffi_path(event_dir)
        if not reference_ffi_path:
            raise FileNotFoundError(
                f"Missing reference_ffi_path in {cluster_job_path}"
            )

    templates_dir = str(cfg.template_dir) if cfg.template_dir else ""
    if star_run_config is not None:
        output_store_name = output_store_name or getattr(
            star_run_config, "output_store_name", None
        )
    expected_templates = resolve_scc_template_dir(
        data_root,
        target,
        oversampling_factor=os_factor,
        store_name=template_store_name,
    )
    if not templates_dir or not Path(templates_dir).is_dir():
        templates_dir = str(expected_templates)
    elif (
        star_run_config is not None
        and Path(templates_dir).resolve() != expected_templates.resolve()
        and expected_templates.is_dir()
    ):
        import logging

        logging.getLogger(__name__).warning(
            "star oversampling_factor=%s expects templates at %s but baseline "
            "template_dir is %s; using star oversampling_factor store",
            os_factor,
            expected_templates,
            templates_dir,
        )
        templates_dir = str(expected_templates)

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

    mapping_dir, mapping_csv, master_mapping_fits = _mapping_paths(
        data_root, target, oversampling_factor=os_factor
    )

    baseline_run_id = normalize_workspace_run_id(baseline_workspace_run_id)
    baseline_workspace_dir = str(
        workspace_tree_path(event_dir, run_id=baseline_run_id)
    )
    diffs_label, convolved_label, _hp_bkg_label = _resolve_hotpants_output_labels(
        site_dir=paths.site_dir,
        diffs_label=baseline_diffs_label,
        diff_policy=diff_policy,
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
            data_root=data_root,
            target=target,
            output_store_name=output_store_name,
            diff_policy=diff_policy,
        )
    resolve_kw = dict(
        data_root=data_root,
        target=target,
        event_dir=event_dir,
        baseline_run_id=baseline_run_id,
        output_store_name=output_store_name,
    )
    baseline_diffs_dir = _resolve_baseline_label_dir(label=diffs_label, **resolve_kw)
    baseline_convolved_dir = (
        _resolve_baseline_label_dir(label=convolved_label, **resolve_kw)
        if convolved_label
        else ""
    )
    baseline_phot_bkg_dir = _resolve_baseline_label_dir(
        label=phot_bkg_label, **resolve_kw
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
        baseline_diffs_dir=baseline_diffs_dir,
        baseline_convolved_dir=baseline_convolved_dir,
        baseline_phot_bkg_dir=baseline_phot_bkg_dir,
        baseline_phot_bkg_label=phot_bkg_label,
        baseline_kernels_dir=baseline_kernels_dir,
        oversampling_factor=os_factor,
        mapping_grid=mapping_grid,
        output_store_name=output_store_name,
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


def resolve_star_manifest_path(ctx: StarEventContext) -> Path:
    """Absolute path to the frame manifest CSV (``frames.csv``) for *ctx*'s lane.

    Sits beside ``ctx.cluster_job_path`` — the SCC bookkeeping directory
    (``bookkeeping/diff_{lane}/oversampling_N/``) for modern SCC-bootstrapped
    lanes, or the legacy event-dir handoff directory for older data_roots
    without SCC bootstrap (see the two branches in ``load_event_context``).
    Every manifest reader and the prerequisite gate must go through this one
    function so they can never disagree about which lane they're reading.
    """
    return Path(ctx.cluster_job_path).parent / DEFAULT_MANIFEST_BASENAME


def _dir_has_glob(path: str, pattern: str) -> bool:
    root = Path(path)
    if not root.is_dir():
        return False
    return any(root.glob(pattern))


def validate_star_prerequisites(ctx: StarEventContext) -> None:
    """Fail fast with an actionable checklist when prerequisites are missing."""
    missing: list[str] = []

    # cluster_job_path/DEFAULT_MANIFEST_BASENAME sit side by side, sourced from
    # SCC bookkeeping (bookkeeping/diff_{lane}/oversampling_N/) for modern lanes,
    # or the legacy event-dir handoff for older data_roots (see load_event_context).
    cluster_path = Path(ctx.cluster_job_path)
    if not cluster_path.is_file():
        missing.append(
            f"{cluster_path.name} missing at {cluster_path}; "
            "run the SCC diff bootstrap (bootstrap_scc_diff/ensure_scc_diff_handoff) "
            "for this sector/camera/ccd"
        )

    manifest_path = resolve_star_manifest_path(ctx)
    if not manifest_path.is_file():
        missing.append(
            f"frames.csv missing at {manifest_path}; "
            "run the SCC diff bootstrap (bootstrap_scc_diff/ensure_scc_diff_handoff) "
            "for this sector/camera/ccd"
        )

    templates_path = Path(ctx.templates_dir)
    field_mode = str(ctx.cluster_job.get("geometry_mode") or "linear").lower() == "field"
    if field_mode:
        # Field mode: the "templates_dir" is the SCC field store; require its
        # manifest + the per-event group-shift schedule instead of linear FITS.
        if not (templates_path / "template_manifest.json").is_file():
            missing.append(
                f"no field template_manifest.json under {templates_path}; "
                "run template downsample (geometry_mode: field) for this event"
            )
        if not (Path(ctx.event_dir) / "template_group_shifts.parquet").is_file():
            missing.append(
                f"template_group_shifts.parquet missing under {ctx.event_dir}; "
                "run field template downsample for this event"
            )
    elif not _dir_has_glob(str(templates_path), "syndiff_template_*"):
        missing.append(
            f"no syndiff_template_* files under {templates_path}; "
            "run template downsample for this event"
        )

    diffs_dir = Path(ctx.baseline_diffs_dir)
    if not ctx.baseline_diffs_dir or not _dir_has_glob(str(diffs_dir), "*.fits*"):
        missing.append(
            f"no baseline diff FITS under {diffs_dir or '(unset)'}; "
            f"complete the baseline hotpants stage ({ctx.baseline_diffs_label}) "
            "in the SCC diff lane or event workspace"
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
            "(run background_estimate ks_b and/or inherit ks_b_s in the SCC lane "
            "or baseline workspace)"
        )

    kernels_dir = Path(ctx.baseline_kernels_dir)
    if not _dir_has_glob(str(kernels_dir), "*_kernel.npz"):
        missing.append(
            f"no *_kernel.npz files under {kernels_dir}; "
            "re-run the baseline hotpants stage with write_kernel_solutions: true"
        )

    lane_root = scc_diff_dir(
        ctx.data_root,
        ctx.sector,
        ctx.camera,
        ctx.ccd,
        store_name=ctx.output_store_name,
    )
    shared_mask = resolve_pipeline_artifact_path(
        str(lane_root), SHARED_MASK_FITS_BASENAME
    )
    if shared_mask is None:
        missing.append(
            f"shared_mask missing under SCC lane {lane_root} "
            f"(expected {SHARED_MASK_FITS_BASENAME}); "
            "run the shared_mask diff stage for this SCC"
        )

    mapping_csv = Path(ctx.mapping_csv)
    if not mapping_csv.is_file():
        missing.append(
            f"mapping CSV missing at {mapping_csv}; "
            "run the mapping stage for this sector/camera/ccd"
        )

    master_mapping = try_resolve_fits_variant(ctx.master_mapping_fits)
    if master_mapping is None:
        missing.append(
            f"master_pixels2skycells FITS missing at {ctx.master_mapping_fits}; "
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
