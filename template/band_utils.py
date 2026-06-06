"""
Simple band combination utilities.

Function-oriented approach for PS1 r,i,z,y band combination.
"""

import logging
import multiprocessing

import numpy as np
import sep
from astropy.io import fits

logger = logging.getLogger(__name__)


def compute_tess_mag(
    g: np.ndarray,
    bp: np.ndarray,
    rp: np.ndarray,
) -> np.ndarray:
    """Compute TESS-equivalent magnitude from Gaia photometry.

    Uses the polynomial correction when BP and RP are both finite:
        T = G − 0.00522555(BP−RP)^3 + 0.0891337(BP−RP)^2 − 0.633923(BP−RP) + 0.0324473

    Falls back to a simple offset when either colour term is missing:
        T = G − 0.430

    Args:
        g:  Gaia G-band magnitudes (array-like, may contain NaN)
        bp: Gaia BP-band magnitudes (array-like, NaN when unavailable)
        rp: Gaia RP-band magnitudes (array-like, NaN when unavailable)

    Returns:
        TESS magnitude array of the same shape as the inputs.
    """
    g = np.asarray(g, dtype=np.float64)
    bp = np.asarray(bp, dtype=np.float64)
    rp = np.asarray(rp, dtype=np.float64)
    color = bp - rp
    full = (
        g
        - 0.00522555 * color ** 3
        + 0.0891337  * color ** 2
        - 0.633923   * color
        + 0.0324473
    )
    fallback = g - 0.430
    return np.where(np.isfinite(color), full, fallback)


def extract_header_values(header_string: str) -> tuple[float, float, float]:
    """Extract BOFFSET, BSOFTEN, and EXPTIME from FITS header string.

    Args:
        header_string: FITS header as string

    Returns:
        Tuple of (boffset, bsoften, exptime)
    """
    try:
        header = fits.Header.fromstring(header_string)
        boffset = float(header["BOFFSET"])
        bsoften = float(header["BSOFTEN"])
        exptime = float(header["EXPTIME"])
        return boffset, bsoften, exptime
    except Exception as e:
        logger.warning(f"[Band] Failed to parse header, using defaults: {e}")
        return 1000.0, 1000.0, 1.0


def apply_flux_conversion(data: np.ndarray, boffset: float, bsoften: float, exptime: float, std: bool = False) -> np.ndarray:
    """Apply PS1 flux conversion from log scale.

    Args:
        data: Raw data array
        boffset: BOFFSET header value
        bsoften: BSOFTEN header value
        exptime: EXPTIME header value

    Returns:
        Converted flux data
    """
    a = 2.5 / np.log(10)
    x = data / a
    flux = boffset + bsoften * 2 * np.sinh(x)
    val = flux if not std else np.sqrt(flux)
    return val / exptime


def _process_single_band(band_data, weight, header_str=None):
    """Worker function to process one band. For parallel execution."""
    band_data = band_data.astype(np.float32)

    # Apply flux conversion if header is provided
    if header_str:
        boffset, bsoften, exptime = extract_header_values(header_str)
        band_data = apply_flux_conversion(band_data, boffset, bsoften, exptime)
    # If no header, use default flux conversion values
    else:
        logger.warning("[Band] No header data available for a band, using default flux conversion.")
        band_data = apply_flux_conversion(band_data)  # Uses defaults

    # Return the weighted contribution
    return band_data * weight


def combine_rizy_bands_parallel(bands_data: dict[str, np.ndarray], weights: list[float] = None, apply_flux_conv: bool = True, headers_data: dict[str, str] = None) -> np.ndarray:
    """
    Combine r,i,z,y bands into a single image in parallel using 4 processes.
    """
    if weights is None:
        weights = [0.238, 0.344, 0.283, 0.135]  # r, i, z, y

    bands = ["r", "i", "z", "y"]

    tasks = []
    for i, band in enumerate(bands):
        if band not in bands_data:
            logger.warning(f"[Band] Missing band {band}, skipping")
            continue

        current_band_data = bands_data[band]
        current_weight = weights[i]
        header_str = None

        if apply_flux_conv and headers_data and band in headers_data:
            header_str = headers_data[band]
            logger.debug(f"Queuing band {band} for processing with its header.")
        elif apply_flux_conv:
            logger.warning(f"Band {band}: no header data available, will use defaults.")

        tasks.append((current_band_data, current_weight, header_str))

    if not tasks:
        raise ValueError("No valid bands found in data")

    with multiprocessing.Pool(processes=4) as pool:
        processed_bands = pool.starmap(_process_single_band, tasks)

    combined = np.sum(processed_bands, axis=0)

    logger.debug(f"[Band] Combined {len(processed_bands)} bands, range: [{combined.min():.3f}, {combined.max():.3f}]")
    return combined


def combine_rizy_bands(bands_data: dict[str, np.ndarray], weights: list[float] = None, apply_flux_conv: bool = True, headers_data: dict[str, str] = None, bands_weights: dict[str, float] = None, headers_weight_data: dict[str, str] = None) -> np.ndarray:
    """Combine r,i,z,y bands into single image.

    Args:
        bands_data: Dictionary mapping band names to arrays
        weights: Weights for [r, i, z, y]. Defaults to optimized values.
        apply_flux_conv: Whether to apply flux conversion
        headers_data: Dictionary mapping band names to FITS header strings

    Returns:
        combined_image array
    """
    if weights is None:
        weights = [0.238, 0.344, 0.283, 0.135]  # r, i, z, y

    bands = ["r", "i", "z", "y"]

    # Get first available band for shape reference
    first_band = None
    for band in bands:
        if band in bands_data:
            first_band = band
            break

    if first_band is None:
        raise ValueError("No valid bands found in data")

    combined = np.zeros_like(bands_data[first_band], dtype=np.float32)
    combined_uncert = np.zeros_like(combined, dtype=np.float32)

    # Process and combine each band
    for i, band in enumerate(bands):
        if band not in bands_data:
            logger.warning(f"[Band] Missing band {band}, skipping")
            continue

        band_data = bands_data[band].astype(np.float32)

        if apply_flux_conv:
            # Extract header values for this specific band
            if headers_data and band in headers_data:
                boffset, bsoften, exptime = extract_header_values(headers_data[band])
                logger.debug(f"[Band] Band {band}: using BOFFSET={boffset}, BSOFTEN={bsoften}, EXPTIME={exptime}")
                band_data = apply_flux_conversion(band_data, boffset, bsoften, exptime)
            else:
                logger.warning(f"[Band] Band {band}: no header data available")

        # Add weighted contribution
        combined += band_data * weights[i]

        if bands_weights and band in bands_weights:
            band_weight = bands_weights[band].astype(np.float32)

            if headers_weight_data and band in headers_weight_data:
                boffset_wt, bsoften_wt, exptime_wt = extract_header_values(headers_weight_data[band])
                logger.debug(f"[Band] Band {band} weights: using BOFFSET={boffset_wt}, BSOFTEN={bsoften_wt}, EXPTIME={exptime_wt}")
                band_weight = apply_flux_conversion(band_weight, boffset_wt, bsoften_wt, exptime_wt, std=True)
            else:
                logger.warning(f"[Band] Band {band} weights: no header data available")

            combined_uncert += (band_weight**2) * (weights[i] ** 2)

    combined_uncert = np.sqrt(combined_uncert)
    logger.debug(f"[Band] Combined {len(bands_data)} bands, range: [{combined.min():.3f}, {combined.max():.3f}]")
    return combined, combined_uncert


def combine_masks(masks_data: dict[str, np.ndarray]) -> np.ndarray:
    """
    Combine multiple mask bands using a vectorized bitwise OR.

    Args:
        masks_data: Dictionary mapping band names to mask arrays.

    Returns:
        Combined mask array as uint16, or None if no masks are provided.
    """
    bands = ["r", "i", "z", "y"]

    # 1. Create a list of all mask arrays that exist in the input dict.
    # This is a single pass over the data.
    valid_masks = [masks_data[b] for b in bands if b in masks_data]

    # 2. Handle the case where no valid masks were found.
    if not valid_masks:
        logger.warning("[Band] No masks available to combine.")
        return None

    # 3. Use np.bitwise_or.reduce to combine all arrays in the list at once.
    # This operation is highly optimized and runs in C code.
    # We cast to uint16 once on the result.
    combined = np.bitwise_or.reduce(valid_masks).astype(np.uint16)

    # Note: For a bitmask, counting non-zero elements is a more accurate
    # way to find the number of affected pixels than using .sum().
    masked_pixel_count = np.count_nonzero(combined)
    logger.debug(f"[Band] Combined {len(valid_masks)} masks, {masked_pixel_count} masked pixels")

    return combined


def process_skycell_bands(bands_data: dict[str, np.ndarray], masks_data: dict[str, np.ndarray] = None, weights_data: dict[str, np.ndarray] = None, headers_data: dict[str, str] = None, headers_weight_data: dict[str, str] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process single skycell: combine bands and masks with variance propagation.

    Args:
        bands_data: Dictionary of band arrays
        masks_data: Dictionary of mask arrays (optional)
        weights_data: Dictionary of variance arrays (optional)
        headers_data: Dictionary of FITS header strings (optional)

    Returns:
        Tuple of (combined_image, combined_mask_uint16, combined_uncert)
    """
    # Combine bands with proper flux conversion using headers and variance maps
    combined_image, combined_uncert = combine_rizy_bands(bands_data, headers_data=headers_data, bands_weights=weights_data, headers_weight_data=headers_weight_data)

    # Combine masks if provided
    combined_mask = None
    if masks_data:
        combined_mask = combine_masks(masks_data)

    # Create dummy mask if none provided
    if combined_mask is None:
        combined_mask = np.zeros_like(combined_image, dtype=np.uint16)

    return combined_image, combined_mask, combined_uncert


def remove_background(
    data: np.ndarray,
    uncert: np.ndarray = None,
    sigma: float = 2.5,
    sigma_mask: float = 50,
    mask: np.ndarray = None,
    remove_saturated_stars: bool = False,
    gaia_catalog_pixels=None,
    bright_star_mag_threshold: float = 13.0,
) -> tuple[np.ndarray, list[dict]]:
    """Remove background from image using SEP, with optional catalog-based segment removal.

    Args:
        data: Input image array (modified in-place).
        uncert: Uncertainty map for SEP extraction.
        sigma: SEP detection threshold.
        sigma_mask: Bright-star masking threshold multiplier.
        mask: Optional PS1 bit-mask array (uint16).
        remove_saturated_stars: If True, run both segment-removal passes.
        gaia_catalog_pixels: Optional DataFrame (already projected to pixel
            coordinates and filtered to the skycell footprint) with columns:
            pixel_x, pixel_y, tess_mag, ra, dec, phot_g_mean_mag,
            phot_bp_mean_mag, phot_rp_mean_mag, and optionally source_id.
            Produced by project_gaia_to_skycell().
        bright_star_mag_threshold: T-mag cutoff for primary catalog pass.

    Returns:
        Tuple of (processed image, removed_stars_list).

    removed_stars_list record schema (one dict per entry):
        source_id        – Gaia DR3 source_id (int, or -1 when unknown)
        ra, dec          – sky coordinates (nan for quality_flag_no_star)
        pixel_x, pixel_y – skycell pixel position (nan for quality_flag_no_star)
        tess_mag         – TESS magnitude (nan for quality_flag_no_star)
        phot_g/bp/rp_mean_mag
        seg_centroid_x/y – SEP centroid (filled only for quality_flag_no_star)
        seg_flux         – SEP flux (filled only for quality_flag_no_star)
        segment_id       – SEP segment ID of the removed segment
        removal_reason   – one of:
            "catalog_bright_star"  primary: T < threshold, caused the removal
            "catalog_neighbor"     primary: T >= threshold, same segment
            "quality_flag_star"    secondary: Gaia star in a flag-removed seg
            "quality_flag_no_star" secondary: no Gaia star found in seg (rare)
    """
    removed_stars_list: list[dict] = []
    try:
        ndimage = None
        if remove_saturated_stars:
            from scipy import ndimage

        mask_bright_stars = data > np.nanmedian(uncert) * sigma_mask
        if remove_saturated_stars:
            mask_bright_stars = ndimage.binary_closing(mask_bright_stars, structure=np.ones((20, 20)))

        data_s = data.astype(data.dtype.newbyteorder("="))
        uncert_s = uncert.astype(uncert.dtype.newbyteorder("="))
        sep.set_extract_pixstack(10000000)
        objects, segmap = sep.extract(data_s, sigma, err=uncert_s, mask=mask_bright_stars, segmentation_map=True)
        data[np.logical_and(segmap == 0, ~mask_bright_stars)] = 0

        if not remove_saturated_stars:
            return data, removed_stars_list

        H, W = data.shape
        has_id = segmap > 0

        # Build the base distance-transform once; reused by both passes.
        # For every pixel, filled_seg_map_base holds the nearest segment ID.
        _, indices = ndimage.distance_transform_edt(~has_id, return_indices=True)
        filled_seg_map_base = segmap[indices[0], indices[1]]

        has_catalog = (
            gaia_catalog_pixels is not None
            and len(gaia_catalog_pixels) > 0
        )

        catalog_seg_ids: set = set()
        px_arr: np.ndarray | None = None
        py_arr: np.ndarray | None = None
        cat_df = None

        # ------------------------------------------------------------------
        # PRIMARY PASS: catalog-based removal
        # ------------------------------------------------------------------
        if has_catalog:
            # Extend segment assignment into bright-star-masked pixels so
            # that catalog stars sitting under the bright-star mask still get
            # assigned to the nearest SEP segment.
            filled_seg_map_cat = np.where(
                has_id | mask_bright_stars, filled_seg_map_base, 0
            )

            px_arr = np.clip(
                np.round(gaia_catalog_pixels["pixel_x"].values).astype(int), 0, W - 1
            )
            py_arr = np.clip(
                np.round(gaia_catalog_pixels["pixel_y"].values).astype(int), 0, H - 1
            )
            seg_ids_cat = filled_seg_map_cat[py_arr, px_arr]

            cat_df = gaia_catalog_pixels.copy()
            cat_df["seg_id_cat"] = seg_ids_cat

            bright_mask = (
                (seg_ids_cat > 0)
                & (cat_df["tess_mag"].values < bright_star_mag_threshold)
            )
            catalog_seg_ids = set(seg_ids_cat[bright_mask].tolist())

            if catalog_seg_ids:
                data[np.isin(filled_seg_map_cat, list(catalog_seg_ids))] = 0
                logger.info(
                    f"[Band] Catalog-based removal: {len(catalog_seg_ids)} segments "
                    f"zeroed (T < {bright_star_mag_threshold})"
                )

                in_removed = cat_df[cat_df["seg_id_cat"].isin(catalog_seg_ids)]
                for row in in_removed.itertuples(index=False):
                    reason = (
                        "catalog_bright_star"
                        if row.tess_mag < bright_star_mag_threshold
                        else "catalog_neighbor"
                    )
                    removed_stars_list.append(_make_star_record(row, int(row.seg_id_cat), reason))
            else:
                logger.info(
                    "[Band] No catalog-based removals "
                    "(no in-footprint Gaia star below magnitude threshold)"
                )

        # ------------------------------------------------------------------
        # SECONDARY PASS: quality-flag (sat + starcore) removal
        # ------------------------------------------------------------------
        if mask is None:
            logger.warning("[Band] Saturated-star removal requested but no mask was provided.")
        else:
            try:
                mask_sat = ((mask & 0x0020) != 0) & ((mask & 0x1000) != 0)

                # Extend segment assignment into sat pixels as well.
                flag_total_mask = has_id | mask_bright_stars | mask_sat
                filled_seg_map_flag = np.where(flag_total_mask, filled_seg_map_base, 0)

                overlap_ids = np.unique(filled_seg_map_flag[mask_sat])
                overlap_ids = overlap_ids[overlap_ids > 0]
                flag_seg_ids = set(overlap_ids.tolist()) - catalog_seg_ids

                if flag_seg_ids:
                    data[np.isin(filled_seg_map_flag, list(flag_seg_ids))] = 0
                    logger.info(
                        f"[Band] Quality-flag removal: {len(flag_seg_ids)} additional "
                        f"segments zeroed (sat+starcore bits)"
                    )

                    # Re-assign catalog stars using the flag-extended map so
                    # stars sitting under sat pixels get correct segment IDs.
                    if has_catalog:
                        seg_ids_flag = filled_seg_map_flag[py_arr, px_arr]
                        cat_df["seg_id_flag"] = seg_ids_flag

                    for seg_id in flag_seg_ids:
                        if has_catalog:
                            stars_df = cat_df[cat_df["seg_id_flag"] == seg_id]
                        else:
                            stars_df = None

                        if stars_df is not None and len(stars_df) > 0:
                            for row in stars_df.itertuples(index=False):
                                removed_stars_list.append(
                                    _make_star_record(row, seg_id, "quality_flag_star")
                                )
                        else:
                            # No Gaia star in this segment — emit a synthetic record
                            # anchored to the SEP object centroid.
                            try:
                                obj = objects[seg_id - 1]
                                removed_stars_list.append({
                                    "source_id": -1,
                                    "ra": float("nan"),
                                    "dec": float("nan"),
                                    "pixel_x": float("nan"),
                                    "pixel_y": float("nan"),
                                    "tess_mag": float("nan"),
                                    "phot_g_mean_mag": float("nan"),
                                    "phot_bp_mean_mag": float("nan"),
                                    "phot_rp_mean_mag": float("nan"),
                                    "seg_centroid_x": float(obj["x"]),
                                    "seg_centroid_y": float(obj["y"]),
                                    "seg_flux": float(obj["flux"]),
                                    "segment_id": int(seg_id),
                                    "removal_reason": "quality_flag_no_star",
                                })
                            except Exception as obj_err:
                                logger.warning(
                                    f"[Band] Could not read SEP object for seg_id={seg_id}: {obj_err}"
                                )
                                removed_stars_list.append({
                                    "source_id": -1,
                                    "ra": float("nan"),
                                    "dec": float("nan"),
                                    "pixel_x": float("nan"),
                                    "pixel_y": float("nan"),
                                    "tess_mag": float("nan"),
                                    "phot_g_mean_mag": float("nan"),
                                    "phot_bp_mean_mag": float("nan"),
                                    "phot_rp_mean_mag": float("nan"),
                                    "seg_centroid_x": float("nan"),
                                    "seg_centroid_y": float("nan"),
                                    "seg_flux": float("nan"),
                                    "segment_id": int(seg_id),
                                    "removal_reason": "quality_flag_no_star",
                                })
                else:
                    logger.info("[Band] No additional quality-flag segments to remove")
            except Exception as e:
                logger.warning(
                    f"[Band] Quality-flag removal failed; continuing with catalog-only results: {e}"
                )

    except Exception as e:
        logging.error(f"[Band] SEP extraction failed: {e}")
        return data, removed_stars_list

    return data, removed_stars_list


def _make_star_record(row, seg_id: int, reason: str) -> dict:
    """Build a unified removed-star record from a catalog row namedtuple."""

    def _safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    def _source_id_for_record(val):
        if val is None:
            return -1
        try:
            if isinstance(val, float) and np.isnan(val):
                return -1
        except TypeError:
            pass
        try:
            return int(val)
        except (TypeError, ValueError):
            return -1

    return {
        "source_id": _source_id_for_record(getattr(row, "source_id", None)),
        "ra": _safe_float(getattr(row, "ra", float("nan"))),
        "dec": _safe_float(getattr(row, "dec", float("nan"))),
        "pixel_x": _safe_float(getattr(row, "pixel_x", float("nan"))),
        "pixel_y": _safe_float(getattr(row, "pixel_y", float("nan"))),
        "tess_mag": _safe_float(getattr(row, "tess_mag", float("nan"))),
        "phot_g_mean_mag": _safe_float(getattr(row, "phot_g_mean_mag", float("nan"))),
        "phot_bp_mean_mag": _safe_float(getattr(row, "phot_bp_mean_mag", float("nan"))),
        "phot_rp_mean_mag": _safe_float(getattr(row, "phot_rp_mean_mag", float("nan"))),
        "seg_centroid_x": float("nan"),
        "seg_centroid_y": float("nan"),
        "seg_flux": float("nan"),
        "segment_id": seg_id,
        "removal_reason": reason,
    }
