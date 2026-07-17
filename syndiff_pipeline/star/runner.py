"""Core star pipeline execution (shared by CLI and orchestration stage)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from astropy.io import fits

from syndiff_pipeline.difference_imaging.support.manifest import row_ffi_product_id_series
from syndiff_pipeline.difference_imaging.support.paths import DEFAULT_MANIFEST_BASENAME
from syndiff_pipeline.star.context import (
    StarEventContext,
    StarPrerequisiteError,
    full_ffi_to_crop_local,
    resolve_host_full_ffi_xy,
    validate_star_prerequisites,
)
from syndiff_pipeline.star.diff_runner import (
    compute_star_only_stamp_for_frame,
    write_star_diff_stamp,
)
from syndiff_pipeline.star.epsf_runner import ensure_star_epsf_catalog
from syndiff_pipeline.star.hosts import load_star_hosts_file
from syndiff_pipeline.star.identifiers import (
    ResolvedHost,
    resolve_host,
    write_host_gaia_row_csv,
    write_identifier_json,
)
from syndiff_pipeline.star.plots import write_lightcurve_debug_png
from syndiff_pipeline.star.site_config import (
    StarRunConfig,
    epsf_workspace_from_method,
    required_epsf_workspaces,
)
from syndiff_pipeline.star.star_segments import isolate_and_write_mini_templates
from syndiff_pipeline.star.windowed_photometry import run_windowed_forced_photometry

logger = logging.getLogger(__name__)

HOST_STAR_SUBDIR = "host_star"


def star_output_root(ctx: StarEventContext) -> Path:
    """Return ``{baseline_ws}/host_star`` for star-branch outputs."""
    return Path(ctx.baseline_workspace_dir) / HOST_STAR_SUBDIR


def legacy_star_output_root(
    ctx: StarEventContext, workspace_run_id: str | None
) -> Path | None:
    """Return pre-``host_star`` sibling path ``events/{label}/star[_id]/`` if any.

    Used only for backward-compatible verify of runs that wrote under the old
    layout. New runs always write via :func:`star_output_root`.
    """
    from syndiff_pipeline.difference_imaging.support.paths import (
        normalize_workspace_run_id,
    )

    event = Path(ctx.event_dir)
    run_id = normalize_workspace_run_id(workspace_run_id)
    if run_id:
        return event / f"star_{run_id}"
    return event / "star"


def resolve_star_host_root(
    ctx: StarEventContext, workspace_run_id: str | None = None
) -> Path:
    """Prefer ``host_star/``; fall back to legacy sibling when only that exists."""
    host_root = star_output_root(ctx)
    if (host_root / "batch_manifest.csv").is_file():
        return host_root
    legacy = legacy_star_output_root(ctx, workspace_run_id)
    if legacy is not None and (legacy / "batch_manifest.csv").is_file():
        return legacy
    return host_root


def _mini_template_fits_paths(mini_paths: list[str]) -> dict[tuple[float, float], str]:
    out: dict[tuple[float, float], str] = {}
    for path in mini_paths:
        with fits.open(path, memmap=True) as hdul:
            hdr = hdul[0].header
            dx = float(hdr["DX_SHIFT"])
            dy = float(hdr["DY_SHIFT"])
        out[(dx, dy)] = path
    return out


def _blend_flag(iso_result: dict) -> bool:
    skycells = iso_result.get("skycells") or {}
    return any(bool(info.get("blend_flag")) for info in skycells.values())


def _load_manifest(ctx: StarEventContext) -> pd.DataFrame:
    manifest_path = Path(ctx.event_dir) / DEFAULT_MANIFEST_BASENAME
    return pd.read_csv(manifest_path)


def _load_product_ids(ctx: StarEventContext, *, max_ffis: int | None = None) -> list[str]:
    manifest = _load_manifest(ctx)
    pids = row_ffi_product_id_series(manifest)
    product_ids = [str(pid) for pid in pids if pid]
    if max_ffis is not None and max_ffis > 0:
        return product_ids[:max_ffis]
    return product_ids


def _load_product_id_btjd_map(ctx: StarEventContext) -> dict[str, float]:
    manifest = _load_manifest(ctx)
    if "btjd" not in manifest.columns:
        return {}
    pids = row_ffi_product_id_series(manifest)
    return {
        str(pid): float(btjd)
        for pid, btjd in zip(pids, manifest["btjd"])
        if pid
    }


def _manifest_row(
    *,
    host: ResolvedHost | None,
    status: str,
    blend_flag: bool,
    frames_processed: int,
    frames_failed: int,
    lightcurve_paths: list[str],
    request_label: str | None = None,
    error: str | None = None,
) -> dict:
    return {
        "gaia_source_id": int(host.gaia_source_id) if host is not None else "",
        "tic_id": host.tic_id if host is not None else "",
        "label": (host.label if host is not None else request_label) or "",
        "status": status,
        "blend_flag": bool(blend_flag),
        "frames_processed": int(frames_processed),
        "frames_failed": int(frames_failed),
        "lightcurve_paths": ";".join(lightcurve_paths),
        "error": error or "",
    }


def _build_prf_method(ctx: StarEventContext, x_ref: float, y_ref: float) -> dict:
    from syndiff_pipeline.difference_imaging.stages.photometry import (
        resolve_tess_prf_localdatadir,
    )

    try:
        from PRF import TESS_PRF
    except ImportError as exc:
        raise ImportError(
            "The PRF package is required for prf photometry. "
            "Install with: pip install PRF"
        ) from exc

    localdatadir = resolve_tess_prf_localdatadir(ctx)
    prf = TESS_PRF(
        ctx.camera,
        ctx.ccd,
        ctx.sector,
        float(x_ref),
        float(y_ref),
        localdatadir=localdatadir,
    )
    return {
        "name": "prf",
        "type": "prf",
        "epsf_model": prf,
        "psf_size": 11,
        "phot_bkg_poly_order": 3,
    }


def _resolve_photometry_methods(
    ctx: StarEventContext,
    *,
    methods: list[dict],
    x_ref: float,
    y_ref: float,
    epsf_catalogs: dict[str, object] | None = None,
) -> list[dict]:
    resolved: list[dict] = []
    catalogs = epsf_catalogs or {}
    for method in methods:
        if str(method.get("type", "")).strip().lower() in ("psf", "prf"):
            psf_type = str(method.get("psf_type", "prf")).strip().lower()
            if psf_type == "prf":
                prf_method = _build_prf_method(ctx, x_ref, y_ref)
                prf_method["name"] = str(method.get("name", "prf"))
                resolved.append(prf_method)
                continue
            if psf_type == "epsf":
                label = epsf_workspace_from_method(method) or method.get("epsf_workspace")
                if not label:
                    raise ValueError(
                        f"method {method.get('name')!r} requires inputs.epsf"
                    )
                catalog = catalogs.get(str(label))
                if catalog is None:
                    raise ValueError(
                        f"method {method.get('name')!r} references inputs.epsf={label!r} "
                        "but no gridded ePSF catalog was loaded for that label"
                    )
                gepsf_method = dict(method)
                gepsf_method["gridded_catalog"] = catalog
                resolved.append(gepsf_method)
                continue
        resolved.append(dict(method))
    return resolved


def run_star_pipeline(
    ctx: StarEventContext,
    *,
    run_config: StarRunConfig,
    stars_file: str | None = None,
    validate: bool = True,
) -> Path:
    """Run mini-templates, stamps, and photometry for all hosts in *stars_file*."""
    if validate:
        validate_star_prerequisites(ctx)

    hosts_path = stars_file or run_config.stars_file
    if not hosts_path:
        raise ValueError("stars_file is required")

    requests = load_star_hosts_file(hosts_path)
    star_root = star_output_root(ctx)
    star_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    product_ids = _load_product_ids(ctx, max_ffis=run_config.max_ffis)
    btjd_by_product_id = _load_product_id_btjd_map(ctx)

    epsf_catalogs: dict[str, object] = {}
    for label in required_epsf_workspaces(run_config.photometry_methods):
        build_cfg = run_config.epsf if run_config.epsf is not None and run_config.epsf.output == label else None
        epsf_catalogs[label] = ensure_star_epsf_catalog(
            ctx,
            label,
            build_cfg=build_cfg,
            diffs_label=build_cfg.diffs if build_cfg is not None else None,
            overwrite=run_config.overwrite,
            max_ffis=run_config.max_ffis,
        )

    for request in requests:
        host: ResolvedHost | None = None
        try:
            host = resolve_host(request, gaia_catalog_path=ctx.gaia_catalog_path)
        except Exception as exc:
            logger.error("Failed to resolve host row %s: %s", request, exc)
            manifest_rows.append(
                _manifest_row(
                    host=None,
                    status="error",
                    blend_flag=False,
                    frames_processed=0,
                    frames_failed=0,
                    lightcurve_paths=[],
                    request_label=request.label,
                    error=str(exc),
                )
            )
            continue

        host_dir = star_root / str(host.gaia_source_id)
        host_dir.mkdir(parents=True, exist_ok=True)
        write_identifier_json(host, str(host_dir / "identifier.json"))
        write_host_gaia_row_csv(host, str(host_dir / "host_gaia_row.csv"))

        iso_result = isolate_and_write_mini_templates(
            ctx,
            host,
            cutout_size=run_config.cutout_size,
            ps1_source=run_config.ps1_source,
            ps1_zarr_path=run_config.ps1_zarr_path,
            output_dir=str(host_dir / "mini_templates"),
            write_debug_plots=run_config.debug_plots,
        )

        if iso_result.get("already_removed"):
            logger.info(
                "Skipping gaia_source_id=%s: host already removed in ps1_removed_stars.csv",
                host.gaia_source_id,
            )
            manifest_rows.append(
                _manifest_row(
                    host=host,
                    status="skipped_already_removed",
                    blend_flag=False,
                    frames_processed=0,
                    frames_failed=0,
                    lightcurve_paths=[],
                )
            )
            continue

        mini_paths = iso_result.get("mini_template_paths") or []
        iso_error = iso_result.get("error")
        if iso_error or not mini_paths:
            reason = iso_error or "no_valid_segments"
            logger.warning(
                "Skipping gaia_source_id=%s: no usable segment (%s)",
                host.gaia_source_id,
                reason,
            )
            manifest_rows.append(
                _manifest_row(
                    host=host,
                    status="skipped_no_segment",
                    blend_flag=_blend_flag(iso_result),
                    frames_processed=0,
                    frames_failed=0,
                    lightcurve_paths=[],
                    error=str(reason),
                )
            )
            continue

        mini_template_fits_paths = _mini_template_fits_paths(mini_paths)
        # Field mode: group_id -> mini-template path (built in star_segments).
        field_group_to_template = iso_result.get("field_group_to_template")
        if field_group_to_template:
            field_group_to_template = {
                int(k): v for k, v in field_group_to_template.items()
            }
        blend_flag = _blend_flag(iso_result)

        x_ref, y_ref = resolve_host_full_ffi_xy(ctx, host)
        host_local_xy = full_ffi_to_crop_local(ctx, x_ref, y_ref)

        stamp_dir = host_dir / "diff_stamps"
        stamp_dir.mkdir(parents=True, exist_ok=True)

        stamp_paths: list[str] = []
        stamp_time_values: list[float] = []
        frames_ok = 0
        frames_failed = 0
        frame_errors: list[str] = []

        for product_id in product_ids:
            stamp_path = stamp_dir / f"{product_id}.fits.gz"
            if stamp_path.is_file() and not run_config.overwrite:
                stamp_paths.append(str(stamp_path))
                stamp_time_values.append(btjd_by_product_id.get(product_id, float("nan")))
                frames_ok += 1
                continue
            try:
                stamp, metadata = compute_star_only_stamp_for_frame(
                    ctx=ctx,
                    product_id=product_id,
                    host_local_xy=host_local_xy,
                    mini_template_fits_paths=mini_template_fits_paths,
                    stamp_size=run_config.stamp_size,
                    kernel_margin_px=run_config.kernel_margin_px,
                    field_group_to_template=field_group_to_template or None,
                )
                write_star_diff_stamp(
                    str(stamp_path),
                    stamp,
                    window_origin=(metadata["window_x0"], metadata["window_y0"]),
                    host_local_xy=host_local_xy,
                )
                stamp_paths.append(str(stamp_path))
                stamp_time_values.append(btjd_by_product_id.get(product_id, float("nan")))
                frames_ok += 1
            except Exception as exc:
                logger.warning(
                    "Frame %s failed for gaia_source_id=%s: %s",
                    product_id,
                    host.gaia_source_id,
                    exc,
                )
                frames_failed += 1
                frame_errors.append(f"{product_id}: {exc}")

        methods = _resolve_photometry_methods(
            ctx,
            methods=run_config.photometry_methods,
            x_ref=x_ref,
            y_ref=y_ref,
            epsf_catalogs=epsf_catalogs,
        )

        lightcurve_paths: list[str] = []
        if stamp_paths:
            lc_results = run_windowed_forced_photometry(
                stamp_paths,
                host=host,
                methods=methods,
                output_dir=str(host_dir),
                time_values=stamp_time_values,
            )
            for method_name in lc_results:
                lightcurve_paths.append(
                    str(host_dir / f"lightcurve_{method_name}_gaia_{host.gaia_source_id}.csv")
                )
            if run_config.debug_plots:
                lc_plot_path = (
                    host_dir / "plots" / f"lightcurve_debug_gaia_{host.gaia_source_id}.png"
                )
                try:
                    write_lightcurve_debug_png(
                        lc_plot_path,
                        lightcurves=lc_results,
                        host=host,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed writing light curve debug plot for gaia_source_id=%s: %s",
                        host.gaia_source_id,
                        exc,
                    )

        status = "ok" if frames_ok > 0 else "error"
        manifest_rows.append(
            _manifest_row(
                host=host,
                status=status,
                blend_flag=blend_flag,
                frames_processed=frames_ok,
                frames_failed=frames_failed,
                lightcurve_paths=lightcurve_paths,
                error="; ".join(frame_errors) if frame_errors else None,
            )
        )

    manifest_path = star_root / "batch_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def verify_star_batch_manifest(manifest_path: Path) -> bool:
    """Return True when batch_manifest exists and every host row has status=ok."""
    if not manifest_path.is_file():
        return False
    df = pd.read_csv(manifest_path)
    if df.empty:
        return False
    if "status" not in df.columns:
        return False
    return bool((df["status"] == "ok").all())
