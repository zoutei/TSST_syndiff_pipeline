"""Photometry pipeline runner: astrometry + forced photometry on SCC diff products."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.scc_paths import (
    normalize_store_name,
    scc_diff_dir,
    scc_diff_label_dir,
)
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    AperturePhotometryMethodParams,
    PsfPhotometryMethodParams,
    parse_forced_photometry,
)
from syndiff_pipeline.difference_imaging.stages import photometry
from syndiff_pipeline.difference_imaging.support.ds9_regions import write_targets_ds9_regions
from syndiff_pipeline.difference_imaging.support.manifest import (
    limit_diff_paths,
    ordered_diff_paths_for_scc,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    photometry_root,
    pipeline_plots_root,
)
from syndiff_pipeline.photometry.site_config import (
    PhotometryRunConfig,
    PhotometrySitePolicy,
    build_syndiff_config_for_photometry,
    load_photometry_site_policy,
    resolve_photometry_run_config,
)

log = logging.getLogger(__name__)


def _finite_ra_dec(target: Target) -> bool:
    return (
        target.target_ra is not None
        and target.target_dec is not None
        and np.isfinite(float(target.target_ra))
        and np.isfinite(float(target.target_dec))
    )


def _load_template_handoff(
    cfg: SynDiffConfig, event_dir: str
) -> tuple[pd.DataFrame, dict, str]:
    del event_dir
    if not getattr(cfg, "data_root", None):
        raise RuntimeError(
            "SCC-only photometry requires data_root for template handoff "
            "(bookkeeping/diff/)"
        )
    from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
        load_scc_diff_handoff_for_config,
    )

    wcs_table, crop_bounds, ref_ffi_path, _offset, _grid = (
        load_scc_diff_handoff_for_config(cfg)
    )
    return wcs_table, crop_bounds, ref_ffi_path


def _pipeline_plots_root(cfg: SynDiffConfig, phot_root: str) -> str:
    sub = getattr(cfg, "pipeline_plots_dir", None)
    return pipeline_plots_root(phot_root, sub, run_id=None)


def _forced_photometry_lightcurve_plot_path(
    plot_dir: str,
    label_out: str,
    method_name: str,
    target_name: Optional[str],
) -> str:
    safe_method = re.sub(r"[^0-9A-Za-z._-]+", "_", method_name)
    if target_name:
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", target_name)
        return os.path.join(plot_dir, f"lightcurve_{label_out}_{safe_method}_{safe}.png")
    return os.path.join(plot_dir, f"lightcurve_{label_out}_{safe_method}.png")


def _warn_if_forced_target_outside_crop(
    target_x: float,
    target_y: float,
    crop_bounds: dict,
    phot_cutout_size: int,
    *,
    ra: float,
    dec: float,
    tag: str,
) -> None:
    sh = crop_bounds.get("shape")
    if not sh or len(sh) != 2:
        return
    ny, nx = int(sh[0]), int(sh[1])
    half = phot_cutout_size // 2
    margin = half + 2
    if (
        target_x < -margin
        or target_x > nx - 1 + margin
        or target_y < -margin
        or target_y > ny - 1 + margin
    ):
        log.warning(
            "forced_photometry: position %r (ra=%s dec=%s) crop-local (%.2f, %.2f) "
            "is outside the crop [0,%d) x [0,%d) with margin %d; expect weak/NaN cutouts.",
            tag,
            ra,
            dec,
            target_x,
            target_y,
            nx,
            ny,
            margin,
        )


def _derive_tile_centers(crop_bounds: dict, tile_nx: int, tile_ny: int) -> list:
    from syndiff_pipeline.difference_imaging.stages.epsf import _make_tile_grid

    ny, nx = crop_bounds["shape"]
    tiles = _make_tile_grid(ny, nx, tile_ny, tile_nx)
    return [(c0 + ts / 2, r0 + ts / 2) for (r0, c0, ts) in tiles]


def _run_forced_photometry(
    *,
    cfg: SynDiffConfig,
    run_config: PhotometryRunConfig,
    stage: dict,
    stage_idx: int,
    phot_root: str,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    ref_ffi_path: str,
    diff_log_path: str | None,
    force_rerun: bool,
) -> None:
    phot_params = parse_forced_photometry(stage, stage_idx)
    label_out = str(stage["output"]).strip()
    phot_out = os.path.join(phot_root, label_out)
    os.makedirs(phot_out, exist_ok=True)

    store_name = normalize_store_name(run_config.output_store_name)
    paths_for_phot = ordered_diff_paths_for_scc(
        wcs_table,
        cfg.data_root,
        int(cfg.sector),
        int(cfg.camera),
        int(cfg.ccd),
        run_config.diffs_label,
        store_name=store_name,
    )
    if cfg.max_ffis is not None:
        paths_for_phot, phot_rows = limit_diff_paths(paths_for_phot, cfg.max_ffis)
    else:
        phot_rows = list(range(len(paths_for_phot)))
    wcs_for_phot = wcs_table.iloc[phot_rows].reset_index(drop=True)
    ref_idx = wcs_grouping.ref_manifest_row_index(wcs_table, ref_ffi_path)
    if ref_idx is not None and ref_idx in phot_rows:
        ref_idx = phot_rows.index(ref_idx)
    elif ref_idx is not None:
        ref_idx = None
        log.warning(
            "forced_photometry: reference FFI not in max_ffis subset; "
            "phot_snap='ref' may use (0,0) offsets."
        )

    tile_centers = _derive_tile_centers(
        crop_bounds, phot_params.tile_nx, phot_params.tile_ny
    )

    epsf_by_workspace: dict[str, np.ndarray] = {}
    gridded_epsf_by_workspace: dict = {}
    stage_epsf_ws = run_config.epsf_label
    inp = stage.get("inputs") or {}
    if inp.get("epsf"):
        stage_epsf_ws = str(inp["epsf"]).strip()

    def _load_epsf_workspace(ws_lab: str) -> None:
        if ws_lab in epsf_by_workspace or ws_lab in gridded_epsf_by_workspace:
            return
        epsf_dir = scc_diff_label_dir(
            cfg.data_root,
            int(cfg.sector),
            int(cfg.camera),
            int(cfg.ccd),
            store_name=store_name,
            label=ws_lab,
        )
        from syndiff_pipeline.difference_imaging.stages import gridded_epsf

        catalog = gridded_epsf.catalog_from_workspace(str(epsf_dir))
        if catalog is not None:
            gridded_epsf_by_workspace[ws_lab] = catalog
            return

        def _method_epsf_label(m: PsfPhotometryMethodParams) -> Optional[str]:
            return m.epsf_workspace or stage_epsf_ws

        needs_epsf = any(
            isinstance(m, PsfPhotometryMethodParams)
            and m.psf_type == "epsf"
            and _method_epsf_label(m) == ws_lab
            for m in phot_params.methods
        )
        if needs_epsf:
            raise ValueError(
                f"forced_photometry: ePSF workspace {ws_lab!r} has no "
                "gridded_epsf_index.json. Rebuild the ePSF stage."
            )

    if stage_epsf_ws:
        _load_epsf_workspace(str(stage_epsf_ws).strip())
    for method in phot_params.methods:
        if isinstance(method, PsfPhotometryMethodParams) and method.epsf_workspace:
            _load_epsf_workspace(method.epsf_workspace)

    extras = list(cfg.additional_forced_targets or [])
    science = (float(cfg.target_ra), float(cfg.target_dec))
    primary_xy = photometry.per_frame_target_crop_xy(
        wcs_table,
        float(cfg.target_ra),
        float(cfg.target_dec),
        crop_bounds,
        manifest_science_ra_dec=science,
    )
    primary_xy = primary_xy[phot_rows]

    target_specs: list[tuple] = [
        (
            primary_xy,
            None,
            "primary",
            {
                "position_mode": "sky",
                "ra": float(cfg.target_ra),
                "dec": float(cfg.target_dec),
            },
        ),
    ]
    for j, pt in enumerate(extras):
        extra_xy = photometry.resolve_forced_target_xy(
            pt,
            primary_xy,
            wcs_for_phot,
            crop_bounds,
            manifest_science_ra_dec=science,
        )
        target_specs.append(
            (
                extra_xy,
                str(pt["name"]),
                f"extra[{j}]",
                pt,
            )
        )

    for target_xy, _lc_name, tag, pt in target_specs:
        mx = float(np.nanmedian(target_xy[:, 0]))
        my = float(np.nanmedian(target_xy[:, 1]))
        mode = pt.get("position_mode", "sky")
        if mode == "sky":
            ra_log = float(pt["ra"])
            dec_log = float(pt["dec"])
        else:
            ra_log = float("nan")
            dec_log = float("nan")

        psf_sizes = [
            m.phot_cutout_size
            for m in phot_params.methods
            if hasattr(m, "phot_cutout_size")
        ]
        warn_cutout = int(max(psf_sizes)) if psf_sizes else 15
        if np.isfinite(mx) and np.isfinite(my):
            _warn_if_forced_target_outside_crop(
                mx,
                my,
                crop_bounds,
                warn_cutout,
                ra=ra_log,
                dec=dec_log,
                tag=tag,
            )

    def _plot_path(method_name: str, extra_name: Optional[str]) -> str:
        pdir = _pipeline_plots_root(cfg, phot_root)
        os.makedirs(pdir, exist_ok=True)
        return _forced_photometry_lightcurve_plot_path(
            pdir, label_out, method_name, extra_name
        )

    shared_mask_for_phot = None
    if any(
        isinstance(m, AperturePhotometryMethodParams)
        and getattr(m, "mask_sky_with_shared_mask", False)
        for m in phot_params.methods
    ):
        from syndiff_pipeline.difference_imaging.support.ffi_naming import (
            resolve_pipeline_artifact_path,
        )
        from syndiff_pipeline.difference_imaging.support.paths import (
            SHARED_MASK_FITS_BASENAME,
        )

        lane_root = scc_diff_dir(
            cfg.data_root,
            int(cfg.sector),
            int(cfg.camera),
            int(cfg.ccd),
            store_name=store_name,
        )
        sm_path = resolve_pipeline_artifact_path(str(lane_root), SHARED_MASK_FITS_BASENAME)
        if sm_path and os.path.isfile(sm_path):
            from astropy.io import fits

            shared_mask_for_phot = np.asarray(fits.getdata(sm_path), dtype=np.int16)

    diffs_dir = scc_diff_label_dir(
        cfg.data_root,
        int(cfg.sector),
        int(cfg.camera),
        int(cfg.ccd),
        store_name=store_name,
        label=run_config.diffs_label,
    )

    photometry.run_forced_photometry_stage(
        diff_paths=paths_for_phot,
        target_specs=target_specs,
        phot_stage=phot_params,
        epsf_by_workspace=epsf_by_workspace,
        gridded_epsf_by_workspace=gridded_epsf_by_workspace,
        stage_epsf_workspace=stage_epsf_ws,
        tile_centers=tile_centers,
        wcs_table=wcs_for_phot,
        crop_bounds=crop_bounds,
        cfg=cfg,
        output_dir=phot_out,
        ref_frame_index=ref_idx,
        plot_title_suffix=label_out,
        output_label=label_out,
        diffs_input=run_config.diffs_label,
        diff_log_path=diff_log_path,
        plot_path_fn=_plot_path,
        diffs_dir=str(diffs_dir),
        shared_mask=shared_mask_for_phot,
    )

    if getattr(cfg, "pipeline_plots", False):
        from syndiff_pipeline.difference_imaging.support.plot import (
            write_lightcurve_diagnostics_from_workspace,
        )

        dpi = int(getattr(cfg, "pipeline_plot_dpi", 150) or 150)
        write_lightcurve_diagnostics_from_workspace(
            phot_out,
            _pipeline_plots_root(cfg, phot_root),
            lc_label=label_out,
            dpi=dpi,
        )


def run_photometry_pipeline(
    cfg: SynDiffConfig,
    target: Target,
    site_dir: str | Path,
    *,
    run_config: PhotometryRunConfig | None = None,
    policy: PhotometrySitePolicy | None = None,
    force_rerun: bool = False,
    phot_log_path: str | None = None,
) -> Path:
    """Run astrometry + forced photometry for one event target."""
    site = Path(site_dir).expanduser().resolve()
    if policy is None:
        raise ValueError("run_photometry_pipeline requires policy")
    if run_config is None:
        run_config = resolve_photometry_run_config(policy, target, site_dir=site)
    if not _finite_ra_dec(target):
        raise ValueError("photometry requires finite target_ra and target_dec on the target")

    event_dir = str(cfg.output_dir)
    phot_root = photometry_root(event_dir, run_config.photometry_run_id)
    os.makedirs(phot_root, exist_ok=True)

    wcs_table, crop_bounds, ref_ffi_path = _load_template_handoff(cfg, event_dir)

    from syndiff_pipeline.difference_imaging.stages.astrometry import (
        load_astrometry_coords,
        run_astrometry_stage,
    )

    coords = load_astrometry_coords(phot_root)
    if coords is not None:
        cfg.target_ra, cfg.target_dec = coords

    for idx, stage in enumerate(run_config.pipeline):
        if not isinstance(stage, dict):
            continue
        kind = stage.get("kind")
        log.info("=" * 70)
        log.info("Photometry stage: %s", kind)
        if kind == "astrometry":
            run_astrometry_stage(cfg, stage, phot_root, force_rerun=force_rerun)
            coords = load_astrometry_coords(phot_root)
            if coords is not None:
                cfg.target_ra, cfg.target_dec = coords
        elif kind == "forced_photometry":
            _run_forced_photometry(
                cfg=cfg,
                run_config=run_config,
                stage=stage,
                stage_idx=idx,
                phot_root=phot_root,
                wcs_table=wcs_table,
                crop_bounds=crop_bounds,
                ref_ffi_path=ref_ffi_path,
                diff_log_path=phot_log_path,
                force_rerun=force_rerun,
            )
        else:
            raise RuntimeError(f"Unhandled photometry pipeline kind {kind!r}")

    if (
        cfg.target_ra is not None
        and cfg.target_dec is not None
        and np.isfinite(cfg.target_ra)
        and np.isfinite(cfg.target_dec)
    ):
        write_targets_ds9_regions(
            phot_root,
            target_ra=float(cfg.target_ra),
            target_dec=float(cfg.target_dec),
            target_name=str(getattr(cfg, "target_name", "") or Path(event_dir).name),
            sector=int(cfg.sector),
            camera=int(cfg.camera),
            ccd=int(cfg.ccd),
            additional_forced_targets=getattr(cfg, "additional_forced_targets", None) or [],
            wcs_table=wcs_table,
            crop_bounds=crop_bounds,
            ref_ffi_path=ref_ffi_path,
        )

    return Path(phot_root)


def run_photometry_delegator(
    cfg: SynDiffConfig,
    stage: dict,
    site_dir: str | Path,
    *,
    force_rerun: bool = False,
) -> None:
    """Execute ``kind: photometry`` from a diff_config pipeline delegator stage."""
    config_ref = stage.get("config")
    if not config_ref:
        raise RuntimeError("photometry delegator stage requires config: path to photometry yaml")
    phot_path = Path(str(config_ref)).expanduser()
    if not phot_path.is_absolute():
        phot_path = (Path(site_dir).expanduser().resolve() / phot_path).resolve()
    policy = load_photometry_site_policy(phot_path)
    target = Target(
        sector=int(cfg.sector),
        camera=int(cfg.camera),
        ccd=int(cfg.ccd),
        target_ra=float(cfg.target_ra),
        target_dec=float(cfg.target_dec),
        target_name=str(cfg.target_name),
    )
    run_config = resolve_photometry_run_config(policy, target, site_dir=site_dir)
    run_photometry_pipeline(
        cfg,
        target,
        site_dir,
        run_config=run_config,
        policy=policy,
        force_rerun=force_rerun,
    )
