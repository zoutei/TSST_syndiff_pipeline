"""Hybrid empirical / TESSreduce static shared-mask builders."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.fits_io import write_primary_hdu_fits
from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME
from syndiff_pipeline.difference_imaging.masking import bits
from syndiff_pipeline.difference_imaging.masking.detector import (
    detector_edge_mask,
    ps1_coverage_mask,
    resolve_straps_csv,
)
from syndiff_pipeline.difference_imaging.masking.geometry import (
    big_sat_empirical,
    gaia_circle_mask,
    load_geometry,
)
from syndiff_pipeline.difference_imaging.masking.settings import MaskSettings
from syndiff_pipeline.difference_imaging.masking.faint_star_squares import faint_star_squares
from syndiff_pipeline.difference_imaging.masking.tessreduce_squares import Big_sat, Strap_mask

log = logging.getLogger(__name__)


def _resolve_ps1_count_crop(
    *,
    ref_image: np.ndarray,
    crop_bounds: dict,
    template_path: str | None,
    template_count_crop: np.ndarray | None,
) -> np.ndarray | None:
    """Load or accept a preassembled COUNT crop for PS1 coverage masking."""
    if template_count_crop is not None:
        return np.asarray(template_count_crop)
    if not template_path:
        return None
    from syndiff_pipeline.common.template_coverage import load_template_count_cropped

    return load_template_count_cropped(template_path, crop_bounds)


def _write_shared_mask_fits(
    mask: np.ndarray,
    output_dir: str | Path,
    *,
    sck: tuple | None = None,
    data_root: str | None = None,
    mask_params: Any = None,
) -> str:
    """Write fpacked shared_mask and best-effort provenance emit."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(str(output_dir), SHARED_MASK_FITS_BASENAME)
    hdu = fits.PrimaryHDU(np.asarray(mask, dtype=np.int16))
    write_primary_hdu_fits(out_path, hdu)
    log.info(
        "Shared mask written to %s  (masked pixels: %d / %d)",
        out_path,
        int((mask > 0).sum()),
        mask.size,
    )
    if sck is not None and mask_params is not None:
        try:
            from syndiff_pipeline.difference_imaging.orchestration import provenance_glue

            provenance_glue.emit_shared_mask_artifact(
                sector=sck[0],
                camera=sck[1],
                ccd=sck[2],
                params=mask_params,
                location=out_path,
                data_root=data_root,
            )
        except Exception:
            log.debug("provenance emit (shared_mask) failed", exc_info=True)
    return out_path


def Cat_mask(
    data_image: np.ndarray,
    gaia_df: pd.DataFrame,
    straps_csv: str,
    maglim: float = 13.0,
    scale: float = 1.0,
    strapsize: int = 6,
    col_offset: int = 0,
    bsc_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    TESSreduce full bitmask (bits 1, 2, 4).

    Bit layout:
      bit 1 — catalog sources (faint_star_squares / historical gaia_auto_mask)
      bit 2 — very bright star crosses (Big_sat, mag < 7; Gaia + BSC)
      bit 4 — TESS straps
    """
    gaia_sub = gaia_df[gaia_df["mag"] < maglim].copy()

    mg = faint_star_squares(gaia_sub, data_image, scale)
    bit1 = (mg["all"] > 0).astype(int)

    sat_table = gaia_sub
    if bsc_df is not None and len(bsc_df) > 0:
        bsc_sat = bsc_df.copy()
        bsc_sat["mag"] = bsc_sat["vmag"]
        sat_table = pd.concat(
            [gaia_sub, bsc_sat[["x", "y", "mag"]]],
            ignore_index=True,
        )
    sat_list = Big_sat(sat_table, data_image, scale)
    if len(sat_list) > 0:
        bit2 = (np.nansum(sat_list, axis=0) > 0).astype(int) * 2
    else:
        bit2 = np.zeros_like(data_image, dtype=int)

    if strapsize > 0:
        bit4 = Strap_mask(data_image, col_offset, straps_csv, size=strapsize).astype(int) * 4
    else:
        bit4 = np.zeros_like(data_image, dtype=int)

    return bit1 | bit2 | bit4


def make_shared_mask(
    ref_image: np.ndarray,
    gaia_df: pd.DataFrame,
    crop_bounds: dict,
    straps_csv: str,
    maglim: float = 13.0,
    strapsize: int = 6,
    output_dir: str = None,
    *,
    ref_ffi_path: str | None = None,
    bsc_catalog_path: str | None = None,
    nx: int | None = None,
    ny: int | None = None,
    x_left_dead: int = 44,
    x_right_dead: int = 44,
    y_edge_strip: int = 30,
    template_path: str | None = None,
    template_count_crop: np.ndarray | None = None,
    ps1_min_hit_count: int = 5000,
    scale: float = 1.0,
    sck: tuple | None = None,
    data_root: str | None = None,
    mask_params: Any = None,
) -> np.ndarray:
    """
    TESSreduce shared bitmask (bits 1/2/4/8/16 only; no 32/64/128).

    Rollback path when ``style: tessreduce``.
    """
    bsc_in_crop = None
    if ref_ffi_path is not None:
        from syndiff_pipeline.common.bsc_catalog import (
            load_bright_star_catalog,
            project_bsc_to_crop,
        )

        bsc_full = load_bright_star_catalog(bsc_catalog_path)
        bsc_in_crop = project_bsc_to_crop(bsc_full, ref_ffi_path, crop_bounds)
        if len(bsc_in_crop):
            log.info("  BSC: %d stars in crop for saturation crosses", len(bsc_in_crop))

    mask = Cat_mask(
        data_image=ref_image,
        gaia_df=gaia_df,
        straps_csv=straps_csv,
        maglim=maglim,
        scale=scale,
        strapsize=strapsize,
        col_offset=crop_bounds["x_min"],
        bsc_df=bsc_in_crop,
    )

    if nx is not None and ny is not None:
        edge = detector_edge_mask(
            ref_image.shape,
            crop_bounds,
            nx=int(nx),
            ny=int(ny),
            x_left_dead=int(x_left_dead),
            x_right_dead=int(x_right_dead),
            y_edge_strip=int(y_edge_strip),
        )
        mask = mask | (edge.astype(np.int16) * bits.EDGE)

    if (template_count_crop is not None or template_path) and int(ps1_min_hit_count) > 0:
        count_crop = _resolve_ps1_count_crop(
            ref_image=ref_image,
            crop_bounds=crop_bounds,
            template_path=template_path,
            template_count_crop=template_count_crop,
        )
        if count_crop is not None:
            if count_crop.shape != ref_image.shape:
                raise ValueError(
                    f"Template COUNT crop shape {count_crop.shape} != ref_image "
                    f"{ref_image.shape} for {template_path!r}"
                )
            no_ps1 = ps1_coverage_mask(count_crop, min_hit_count=ps1_min_hit_count)
            mask = mask | (no_ps1.astype(np.int16) * bits.PS1)
            log.info(
                "  PS1 coverage: %d pixels with COUNT < %d",
                int(no_ps1.sum()),
                int(ps1_min_hit_count),
            )

    if output_dir:
        _write_shared_mask_fits(
            mask,
            output_dir,
            sck=sck,
            data_root=data_root,
            mask_params=mask_params,
        )

    return mask.astype(np.int16)


def _project_bsc(
    ref_ffi_path: str | None,
    crop_bounds: dict,
    bsc_catalog_path: str | None,
) -> pd.DataFrame | None:
    if ref_ffi_path is None:
        return None
    try:
        from syndiff_pipeline.common.bsc_catalog import (
            load_bright_star_catalog,
            project_bsc_to_crop,
        )

        bsc_full = load_bright_star_catalog(bsc_catalog_path)
        bsc_in_crop = project_bsc_to_crop(bsc_full, ref_ffi_path, crop_bounds)
        if len(bsc_in_crop):
            log.info("  BSC: %d stars in crop for saturation crosses", len(bsc_in_crop))
        return bsc_in_crop
    except Exception as exc:
        log.warning("BSC load/project skipped: %s", exc)
        return None


def build_static_mask(
    ref_image: np.ndarray,
    gaia_df: pd.DataFrame,
    crop_bounds: dict,
    *,
    settings: MaskSettings | None = None,
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
    output_dir: str | Path | None = None,
    tns_table: pd.DataFrame | None = None,
    sck: tuple | None = None,
    data_root: str | None = None,
    mask_params: Any = None,
) -> np.ndarray:
    """
    Build static shared bitmask.

    Default ``style: empirical``:
      T < bright_maglim: empirical circles (T≥9 → bit 1) + crosses (T<9 → 1|2);
      BSC ∪ Gaia for crosses.
      bright_maglim ≤ T < faint_maglim: faint_star_squares → bit 32.
      straps → 4; edges → 8; PS1 → 16; optional TNS → 64.

    ``style: tessreduce``: legacy Cat_mask path (bits 1/2/4/8/16 only).
    """
    settings = settings or MaskSettings()
    shared = settings.shared
    straps = resolve_straps_csv(straps_csv)

    if "mag" not in gaia_df.columns:
        if "tess_mag" in gaia_df.columns:
            gaia_df = gaia_df.copy()
            gaia_df["mag"] = gaia_df["tess_mag"]
        else:
            raise KeyError("gaia_df must have 'mag' or 'tess_mag'")

    if shared.style == "tessreduce":
        return make_shared_mask(
            ref_image=ref_image,
            gaia_df=gaia_df,
            crop_bounds=crop_bounds,
            straps_csv=straps,
            maglim=shared.bright_maglim,
            strapsize=shared.strapsize if shared.include_straps else 0,
            output_dir=str(output_dir) if output_dir else None,
            ref_ffi_path=ref_ffi_path,
            bsc_catalog_path=bsc_catalog_path,
            nx=nx,
            ny=ny,
            x_left_dead=x_left_dead,
            x_right_dead=x_right_dead,
            y_edge_strip=y_edge_strip,
            template_path=template_path,
            template_count_crop=template_count_crop,
            ps1_min_hit_count=shared.ps1_min_hit_count,
            scale=shared.scale,
            sck=sck,
            data_root=data_root,
            mask_params=mask_params,
        )

    geo = load_geometry(settings.geometry_file)
    scale = float(shared.scale)
    bright_lim = float(shared.bright_maglim)
    faint_lim = float(shared.faint_maglim)

    bsc_in_crop = _project_bsc(ref_ffi_path, crop_bounds, bsc_catalog_path)

    mag = gaia_df["mag"].to_numpy(float)
    bright = gaia_df[mag < bright_lim].copy()
    faint = gaia_df[(mag >= bright_lim) & (mag < faint_lim)].copy()

    # Bit 1: empirical circles on bright with mag >= 9
    mg = gaia_circle_mask(bright, ref_image, scale=scale, mag_min=9.0, geometry=geo)
    bit1 = (mg["all"] > 0).astype(np.int16) * bits.BRIGHT_CAT

    # Bits 1|2: crosses on bright ∪ BSC with mag < 9
    sat_table = bright
    if bsc_in_crop is not None and len(bsc_in_crop) > 0:
        bsc_sat = bsc_in_crop.copy()
        bsc_sat["mag"] = bsc_sat["vmag"]
        sat_table = pd.concat(
            [bright, bsc_sat[["x", "y", "mag"]]],
            ignore_index=True,
        )
    sat_mask = big_sat_empirical(sat_table, ref_image, scale=scale, mag_max=9.0)
    bit2 = (sat_mask > 0).astype(np.int16) * bits.SAT_CROSS
    # Cross body also sets bit 1 (plan: crosses → bits 1|2)
    bit1 = bit1 | ((sat_mask > 0).astype(np.int16) * bits.BRIGHT_CAT)

    # Bit 32: faint_star_squares on faint catalog stars
    if len(faint):
        sq = faint_star_squares(faint, ref_image, scale=scale)
        bit32 = (sq["all"] > 0).astype(np.int16) * bits.FAINT_CAT
    else:
        bit32 = np.zeros(ref_image.shape, dtype=np.int16)

    mask = bit1 | bit2 | bit32

    if shared.include_straps and shared.strapsize > 0:
        bit4 = (
            Strap_mask(
                ref_image,
                int(crop_bounds.get("x_min", 0)),
                straps,
                size=int(shared.strapsize),
            ).astype(np.int16)
            * bits.STRAP
        )
        mask = mask | bit4

    if shared.include_edges and nx is not None and ny is not None:
        edge = detector_edge_mask(
            ref_image.shape,
            crop_bounds,
            nx=int(nx),
            ny=int(ny),
            x_left_dead=int(x_left_dead),
            x_right_dead=int(x_right_dead),
            y_edge_strip=int(y_edge_strip),
        )
        mask = mask | (edge.astype(np.int16) * bits.EDGE)

    if (template_count_crop is not None or template_path) and int(shared.ps1_min_hit_count) > 0:
        count_crop = _resolve_ps1_count_crop(
            ref_image=ref_image,
            crop_bounds=crop_bounds,
            template_path=template_path,
            template_count_crop=template_count_crop,
        )
        if count_crop is not None:
            if count_crop.shape != ref_image.shape:
                raise ValueError(
                    f"Template COUNT crop shape {count_crop.shape} != ref_image "
                    f"{ref_image.shape} for {template_path!r}"
                )
            no_ps1 = ps1_coverage_mask(
                count_crop, min_hit_count=int(shared.ps1_min_hit_count)
            )
            mask = mask | (no_ps1.astype(np.int16) * bits.PS1)
            log.info(
                "  PS1 coverage: %d pixels with COUNT < %d",
                int(no_ps1.sum()),
                int(shared.ps1_min_hit_count),
            )

    if (
        tns_table is not None
        and len(tns_table)
        and settings.tns.include_in_static_fits
    ):
        from syndiff_pipeline.difference_imaging.masking.tns import paint_tns_bit

        mask = paint_tns_bit(mask, tns_table, crop_bounds)

    mask = mask.astype(np.int16)
    if output_dir is not None:
        _write_shared_mask_fits(
            mask,
            output_dir,
            sck=sck,
            data_root=data_root,
            mask_params=mask_params,
        )
    return mask
