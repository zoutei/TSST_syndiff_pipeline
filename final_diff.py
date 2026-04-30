"""
final_diff.py
=============
``diff_final`` pipeline stage — final difference images:
    1. (FFI − bkg_final) is differenced against the PS1 template (already
       done in round-2 hotpants; the round-2 diff is the input here).
    2. For each frame, build an updated saturated-star template by
       deconvolving the high-resolution template from the hotpants kernel
       Gaussian and reconvolving with the smoothed ePSF in Fourier space.
    3. Subtract this refined sat template from the round-2 diff.
    4. NaN masked pixels.

FFT deconvolution / reconvolution follows a standard Wiener-style pattern
applied to the high-resolution saturated-star model.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom as nd_zoom
from scipy.fft import rfft2, irfft2, rfftfreq

from . import frame_manifest
from .sat_template import get_tile_epsf_at_position, _block_sum_downsample

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Fourier deconvolution / reconvolution ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _make_gaussian_ft(shape: tuple, sigma_os: float) -> np.ndarray:
    """
    Compute the DFT of a 2D Gaussian with standard deviation sigma_os (in
    pixels of the oversampled grid) for an array of given shape.

    Returns the real-valued frequency-domain array compatible with rfft2.
    """
    ny, nx = shape
    fy = np.fft.fftfreq(ny)[:, np.newaxis]         # (ny, 1)
    fx = rfftfreq(nx)[np.newaxis, :]                # (1, nx//2+1)
    ft_gauss = np.exp(-2.0 * np.pi ** 2 * sigma_os ** 2 * (fx ** 2 + fy ** 2))
    return ft_gauss.astype(np.complex128)


def deconvolve_reconvolve_template(sat_template_hr: np.ndarray,
                                    gaussian_sigma_native: float,
                                    epsf_tile_2d: np.ndarray,
                                    epsf_os: int,
                                    high_res_os: int) -> np.ndarray:
    """
    In Fourier space:
      (1) deconvolve the high-res sat template from the hotpants kernel Gaussian,
      (2) reconvolve with the smoothed empirical ePSF.

    This replaces the Gaussian-smeared sat stars with ePSF-convolved stars,
    producing a more accurate model of what the saturated stars look like in
    the difference image.

    Parameters
    ----------
    sat_template_hr       : 2D ndarray (ny_native, nx_native) — high-res template
                            already block-sum-downsampled to native resolution.
                            Note: the Fourier operations are performed in
                            native-pixel space because the hi-res template has
                            already been downsampled.  The `gaussian_sigma_native`
                            and zoom operations use native-pixel coordinates.
    gaussian_sigma_native : float — std dev of the hotpants convolution kernel
                            in native pixels (≈ sci_fwhm / 2.355).
    epsf_tile_2d          : 2D ndarray (over_size, over_size) — smoothed ePSF
                            at cfg.epsf_oversample resolution (default 2×).
    epsf_os               : int — ePSF oversampling factor (cfg.epsf_oversample).
    high_res_os           : int — oversampling used when building sat_template_hr
                            (cfg.high_res_os).  Used only to compute the correct
                            Gaussian sigma in the oversampled grid.

    Returns
    -------
    2D ndarray (ny_native, nx_native) — refined sat template in native pixels.
    """
    ny, nx = sat_template_hr.shape

    # Gaussian sigma in the native-pixel grid for the Fourier operations
    sigma_native = gaussian_sigma_native   # already in native pixels

    # FT of the template
    ft_tmpl = rfft2(sat_template_hr)

    # FT of the hotpants Gaussian kernel (in native pixels)
    ft_gauss = _make_gaussian_ft((ny, nx), sigma_native)

    # Clip small values to avoid division by zero
    ft_gauss_safe = np.where(np.abs(ft_gauss) > 1e-6, ft_gauss, 1e-6)

    # Deconvolve
    ft_deconv = ft_tmpl / ft_gauss_safe

    # Build ePSF in native-pixel space by block-sum-downsampling the oversampled stamp
    # after zooming to match the native pixel grid
    over_size = epsf_tile_2d.shape[0]

    # Zoom ePSF from epsf_os → 1x (native) using scipy.ndimage.zoom
    zoom_factor = 1.0 / epsf_os
    epsf_native = nd_zoom(epsf_tile_2d, zoom_factor, order=1)
    norm = epsf_native.sum()
    if norm > 0:
        epsf_native /= norm

    # Embed ePSF_native into an array the size of the template for FFT convolution
    epsf_embed = np.zeros((ny, nx), dtype=np.float64)
    hh = epsf_native.shape[0] // 2
    hw = epsf_native.shape[1] // 2
    epsf_embed[:epsf_native.shape[0], :epsf_native.shape[1]] = epsf_native
    # Roll to centre the PSF at (0,0) for FFT convolution convention
    epsf_embed = np.roll(np.roll(epsf_embed, -hh, axis=0), -hw, axis=1)

    ft_epsf = rfft2(epsf_embed)

    # Reconvolve: deconvolved template × ePSF
    ft_final = ft_deconv * ft_epsf
    refined  = irfft2(ft_final, s=(ny, nx)).real

    return refined


# ═══════════════════════════════════════════════════════════════════════════════
# ── Per-frame final diff ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def make_final_diff_frame(hotpants_diff_r2: np.ndarray,
                           sat_template_final: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    """
    diff_final = hotpants_diff_r2 − sat_template_final,
    with masked pixels set to NaN.

    Parameters
    ----------
    hotpants_diff_r2   : 2D ndarray (crop-local round-2 difference image)
    sat_template_final : 2D ndarray (same shape) — refined sat model
    mask               : 2D int ndarray (bitmask; >0 = bad pixel)

    Returns
    -------
    2D ndarray — final difference image (NaN at masked positions)
    """
    diff_final = hotpants_diff_r2 - sat_template_final
    mask_bool = (mask > 0)
    diff_final[mask_bool] = np.nan
    return diff_final


def final_diff_loop(hotpants_results_r2: list,
                    bkg_final: np.ndarray,
                    sat_template_hr_map: dict,
                    epsf_r2_smooth: np.ndarray,
                    tile_centers: list,
                    wcs_table,
                    mask: np.ndarray,
                    crop_bounds: dict,
                    cfg,
                    output_dir: str,
                    ffi_stems_epsf: list | None = None,
                    output_images_dir: str | None = None) -> list:
    """
    Produce final difference images for every frame.

    Per frame:
      1. Load round-2 diff FITS.
      2. Subtract bkg_final[i] if the diff was not already background-subtracted.
      3. Get the group's high-res sat template.
      4. Get the per-tile ePSF at the frame's group (use group-median ePSF).
      5. deconvolve_reconvolve_template → refined sat template (native res).
      6. make_final_diff_frame → diff_final.
      7. Save to output_dir/diff_final/{stem}.fits.

    Parameters
    ----------
    hotpants_results_r2 : list of dicts from hotpants_loop (round 2)
    bkg_final           : ndarray (n_frames, ny, nx)
    sat_template_hr_map : dict {group_id: 2D ndarray}
    epsf_r2_smooth      : ndarray (n_frames, n_tiles, over_size²)
    ffi_stems_epsf      : optional list of str, length n_frames — axis-0 ID for
                          ``epsf_r2_smooth`` (from ``epsf_r2_smooth.npz``). If
                          set, group ePSF medians use stem-aligned indices.
    tile_centers        : list of (cx, cy)
    wcs_table           : pd.DataFrame with group_id column
    mask                : 2D int ndarray
    crop_bounds         : dict
    cfg                 : SynDiffConfig
    output_dir          : str
    output_images_dir   : str, optional
        If set, FITS are written here. Otherwise ``{output_dir}/diff_final``.

    Returns
    -------
    list of str — paths to saved diff_final FITS files (None for failed frames)
    """
    out_subdir = (
        output_images_dir
        if output_images_dir
        else os.path.join(output_dir, "diff_final")
    )
    os.makedirs(out_subdir, exist_ok=True)

    over_size = 2 * cfg.psf_size + 1
    gaussian_sigma_native = cfg.sci_fwhm / 2.355   # convert FWHM → sigma

    paths = []
    n_frames = len(hotpants_results_r2)

    for i, result in enumerate(hotpants_results_r2):
        if not result.get("success") or result.get("diff") is None:
            paths.append(None)
            continue

        stem     = result.get("stem", f"frame_{i:04d}")
        group_id = int(result.get("group_id", 0))

        diff_r2  = result["diff"].astype(np.float64)

        # Subtract final background (if any)
        if bkg_final is not None and i < len(bkg_final):
            diff_r2 = diff_r2 - bkg_final[i].astype(np.float64)

        # High-res sat template for this group
        sat_hr = sat_template_hr_map.get(group_id)
        if sat_hr is None:
            sat_hr = np.zeros_like(diff_r2)
            log.warning(f"  Frame {i}: no high-res sat template for group {group_id}.")

        # Group-median ePSF over frames in this template group (stem-aligned)
        use_stems = (
            ffi_stems_epsf is not None
            and len(ffi_stems_epsf) == len(epsf_r2_smooth)
        )
        if use_stems:
            group_frames = frame_manifest.epsf_row_indices_for_group(
                wcs_table, ffi_stems_epsf, group_id,
            )
            if len(group_frames) > 0:
                group_epsf = np.nanmedian(
                    epsf_r2_smooth[group_frames], axis=0,
                )
            else:
                group_epsf = (
                    epsf_r2_smooth[i]
                    if i < len(epsf_r2_smooth)
                    else epsf_r2_smooth[0]
                )
        elif len(wcs_table) == len(epsf_r2_smooth):
            group_ids_arr = wcs_table["group_id"].values
            group_frames = np.where(group_ids_arr == group_id)[0]
            if len(group_frames) > 0:
                group_epsf = np.nanmedian(
                    epsf_r2_smooth[group_frames], axis=0,
                )
            else:
                group_epsf = (
                    epsf_r2_smooth[i]
                    if i < len(epsf_r2_smooth)
                    else epsf_r2_smooth[0]
                )
        else:
            log.warning(
                "final_diff: WCS table length %d != ePSF n_frames %d; "
                "using frame-local ePSF for frame %s.",
                len(wcs_table),
                len(epsf_r2_smooth),
                stem,
            )
            group_epsf = (
                epsf_r2_smooth[i]
                if i < len(epsf_r2_smooth)
                else epsf_r2_smooth[0]
            )

        # Pick ePSF at image centre for the deconvolution
        ny, nx = crop_bounds["shape"]
        cx, cy = nx / 2, ny / 2
        epsf_2d = get_tile_epsf_at_position(group_epsf, tile_centers, cx, cy, over_size)

        try:
            sat_tmpl_final = deconvolve_reconvolve_template(
                sat_template_hr=sat_hr,
                gaussian_sigma_native=gaussian_sigma_native,
                epsf_tile_2d=epsf_2d,
                epsf_os=cfg.epsf_oversample,
                high_res_os=cfg.high_res_os,
            )
        except Exception as exc:
            log.warning(f"  Frame {i} deconv/reconv failed ({exc}); using zero sat template.")
            sat_tmpl_final = np.zeros_like(diff_r2)

        diff_final = make_final_diff_frame(diff_r2, sat_tmpl_final, mask)

        out_path = os.path.join(out_subdir, f"{stem}.fits")
        fits.writeto(out_path, diff_final.astype(np.float32), overwrite=True)
        paths.append(out_path)

        if (i + 1) % 20 == 0 or i == n_frames - 1:
            log.info(f"  final_diff: {i + 1}/{n_frames} frames done")

    n_ok = sum(1 for p in paths if p is not None)
    log.info(f"Final diff: {n_ok}/{n_frames} frames saved to {out_subdir}/")
    return paths
