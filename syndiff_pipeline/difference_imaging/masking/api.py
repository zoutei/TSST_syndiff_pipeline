"""Public generate_* entry points for shared / TNS / asteroid masks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.masking.asteroids import ensure_asteroid_products
from syndiff_pipeline.difference_imaging.masking.catalog import MaskCatalog
from syndiff_pipeline.difference_imaging.masking.settings import (
    MaskSettings,
    apply_stage_overrides,
    default_asteroid_intervals_dir,
    default_tns_public_csv,
    resolve_mask_settings,
    write_mask_settings,
)
from syndiff_pipeline.difference_imaging.masking.shared import build_static_mask
from syndiff_pipeline.difference_imaging.masking.tns import (
    ensure_tns_public_csv,
    load_or_build_transient_fixed,
)

log = logging.getLogger(__name__)


def generate_shared_mask_catalog(
    *,
    ref_image: np.ndarray,
    gaia_df: pd.DataFrame,
    crop_bounds: dict,
    ws_root: str | Path,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    straps_csv: str | None = None,
    ref_ffi_path: str | None = None,
    bsc_catalog_path: str | None = None,
    nx: int | None = None,
    ny: int | None = None,
    x_left_dead: int = 44,
    x_right_dead: int = 44,
    y_edge_strip: int = 30,
    template_path: str | None = None,
    template_count_crop: np.ndarray | None = None,
    settings: MaskSettings | None = None,
    stage_mask_settings: str | None = None,
    site_dir: str | Path | None = None,
    gaia_mag_bright: float | None = None,
    strapsize: int | None = None,
    ps1_min_hit_count: int | None = None,
    wcs_table: pd.DataFrame | None = None,
    write_plots_dir: str | Path | None = None,
    mask_params: object | None = None,
) -> MaskCatalog:
    """
    Resolve settings, build static FITS (+ TNS), load/generate asteroids, return catalog.
    """
    ws_root = Path(ws_root)
    ws_root.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings, _ = resolve_mask_settings(
            stage_mask_settings=stage_mask_settings,
            site_dir=site_dir,
            ws_root=ws_root,
        )
    settings = apply_stage_overrides(
        settings,
        gaia_mag_bright=gaia_mag_bright,
        strapsize=strapsize,
        ps1_min_hit_count=ps1_min_hit_count,
    )
    write_mask_settings(settings, ws_root / "mask_settings.yaml")

    tns_table = None
    if settings.tns.enabled and settings.shared.style == "empirical":
        public = (
            Path(settings.tns.public_csv)
            if settings.tns.public_csv
            else default_tns_public_csv(data_root)
        )
        try:
            ensure_tns_public_csv(
                sector, public, url=settings.tns.download_url or None
            )
            tns_table = load_or_build_transient_fixed(
                ws_root=ws_root,
                sector=sector,
                camera=camera,
                ccd=ccd,
                public_csv=public,
                scale=settings.shared.scale,
            )
        except Exception as exc:
            log.warning("TNS enabled but failed (%s); continuing without bit 64", exc)
            tns_table = None

    static = build_static_mask(
        ref_image=ref_image,
        gaia_df=gaia_df,
        crop_bounds=crop_bounds,
        settings=settings,
        straps_csv=straps_csv,
        ref_ffi_path=ref_ffi_path,
        bsc_catalog_path=bsc_catalog_path,
        nx=nx,
        ny=ny,
        x_left_dead=x_left_dead,
        x_right_dead=x_right_dead,
        y_edge_strip=y_edge_strip,
        template_path=template_path,
        template_count_crop=template_count_crop,
        output_dir=ws_root,
        tns_table=tns_table if settings.tns.include_in_static_fits else None,
        sck=(int(sector), int(camera), int(ccd)),
        data_root=str(data_root) if data_root else None,
        mask_params=mask_params,
    )

    asteroid_iv = None
    asteroid_tm = None
    if settings.asteroids.enabled and settings.shared.style == "empirical":
        intervals_dir = settings.asteroids.intervals_dir or str(
            default_asteroid_intervals_dir(data_root, sector, camera, ccd)
        )
        asteroid_iv, asteroid_tm = ensure_asteroid_products(
            data_root=data_root,
            sector=sector,
            camera=camera,
            ccd=ccd,
            intervals_dir=intervals_dir,
            vmag_lim=settings.asteroids.vmag_lim,
            wcs_table=wcs_table,
            enabled=True,
            orbit_times_path=settings.asteroids.orbit_times_path,
            orbit_times_url=settings.asteroids.orbit_times_url or None,
            run_discover=bool(settings.asteroids.run_discover),
        )
        if asteroid_iv is None:
            log.warning("Asteroids enabled but no intervals available; omit bit 128")

    catalog = MaskCatalog.from_arrays(
        static,
        tns_table=tns_table,
        asteroid_intervals_ffi=asteroid_iv,
        asteroid_times=asteroid_tm,
        crop_bounds=crop_bounds,
    )

    if write_plots_dir is not None:
        from syndiff_pipeline.difference_imaging.masking.plots import write_mask_debug_plots

        write_mask_debug_plots(catalog, write_plots_dir)

    return catalog
