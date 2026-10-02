"""
Simple band combination utilities.

Function-oriented approach for PS1 r,i,z,y band combination.
"""

import logging
import multiprocessing
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sep
from astropy.io import fits

logger = logging.getLogger(__name__)

REMOVAL_CONVENTION_LEGACY = "segment_v0"
REMOVAL_CONVENTION_FOOTPRINT = "footprint_v1"
REMOVAL_CONVENTION_STARMODEL = "starmodel_v1"
# Production convention. footprint_v1 zeroes a bright star's whole 8-connected
# footprint and with it every faint source chained to the halo: a hole in the
# template at each removed star (dev_runs/removal_hole_20261001). starmodel_v1
# is the candidate fix; it has not passed acceptance yet (same README).
REMOVAL_CONVENTION = REMOVAL_CONVENTION_FOOTPRINT

# PS1 skycells overlap their neighbours by this many pixels, so no star
# footprint that matters can extend further than this.
_CELL_OVERLAP_PX = 480
# PS1 NaN-masks the outermost row/column, which belongs to no segment.
_EDGE_LOOKUP_INSET = 10  # = EDGE_EXCLUSION: max inward steps past the masked edge
# An off-cell star's component must peak within this many px of the facing edge.
_EDGE_BAND_PX = 13
# Extra reach (px) beyond R(T) when selecting off-cell catalogue stars.
_SELECT_MARGIN_PX = 10


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


def process_skycell_bands(bands_data: dict[str, np.ndarray], masks_data: dict[str, np.ndarray] = None, weights_data: dict[str, np.ndarray] = None, headers_data: dict[str, str] = None, headers_weight_data: dict[str, str] = None, band_weights: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process single skycell: combine bands and masks with variance propagation.

    Args:
        bands_data: Dictionary of band arrays
        masks_data: Dictionary of mask arrays (optional)
        weights_data: Dictionary of variance arrays (optional)
        headers_data: Dictionary of FITS header strings (optional)
        band_weights: Band combination weights ``{"r", "i", "z", "y"}`` from the
            combined recipe. ``None`` uses the production defaults. Callers that
            publish to the combined store must pass the recipe's weights: the
            recipe (and so the fingerprint) records them, and before this
            argument existed they were never applied.

    Returns:
        Tuple of (combined_image, combined_mask_uint16, combined_uncert)
    """
    weights = None if band_weights is None else [float(band_weights[b]) for b in ("r", "i", "z", "y")]
    # Combine bands with proper flux conversion using headers and variance maps
    combined_image, combined_uncert = combine_rizy_bands(bands_data, weights=weights, headers_data=headers_data, bands_weights=weights_data, headers_weight_data=headers_weight_data)

    # Combine masks if provided
    combined_mask = None
    if masks_data:
        combined_mask = combine_masks(masks_data)

    # Create dummy mask if none provided
    if combined_mask is None:
        combined_mask = np.zeros_like(combined_image, dtype=np.uint16)

    return combined_image, combined_mask, combined_uncert


@dataclass
class SepBackgroundResult:
    """SEP extraction outputs used for background suppression and star removal."""

    objects: np.ndarray
    segmap: np.ndarray
    mask_bright_stars: np.ndarray


def build_sep_background_segmentation(
    data: np.ndarray,
    uncert: np.ndarray,
    *,
    sigma: float = 2.5,
    sigma_mask: float = 50,
    close_bright_mask: bool = False,
) -> SepBackgroundResult:
    """Run SEP source extraction with a bright-star mask.

    Args:
        data: Input image array.
        uncert: Uncertainty map passed to SEP as ``err``.
        sigma: SEP detection threshold.
        sigma_mask: Bright-star masking threshold multiplier.
        close_bright_mask: If True, morphologically close the bright-star mask.

    Returns:
        SepBackgroundResult with SEP objects, segmap, and bright-star mask.
    """
    mask_bright_stars = data > np.nanmedian(uncert) * sigma_mask
    if close_bright_mask:
        from scipy import ndimage

        mask_bright_stars = ndimage.binary_closing(
            mask_bright_stars, structure=np.ones((20, 20))
        )

    data_s = data.astype(data.dtype.newbyteorder("="))
    uncert_s = uncert.astype(uncert.dtype.newbyteorder("="))
    sep.set_extract_pixstack(10000000)
    objects, segmap = sep.extract(
        data_s, sigma, err=uncert_s, mask=mask_bright_stars, segmentation_map=True
    )
    return SepBackgroundResult(
        objects=objects,
        segmap=segmap,
        mask_bright_stars=mask_bright_stars,
    )


def filled_segment_map(segmap: np.ndarray) -> np.ndarray:
    """Fill zero-valued segmap pixels with the nearest segment id.

    Args:
        segmap: SEP segmentation map.

    Returns:
        Segmentation map with every pixel assigned to the nearest segment id.
    """
    from scipy import ndimage

    has_id = segmap > 0
    _, indices = ndimage.distance_transform_edt(~has_id, return_indices=True)
    return segmap[indices[0], indices[1]]


def catalog_segment_assignments(
    gaia_catalog_pixels: pd.DataFrame,
    filled_seg_map: np.ndarray,
    mask_bright_stars: np.ndarray,
    *,
    segmap: np.ndarray,
) -> pd.DataFrame:
    """Assign each catalog row to a segment id via the filled segmap.

    Args:
        gaia_catalog_pixels: Gaia rows projected to skycell pixel coordinates.
        filled_seg_map: Output of :func:`filled_segment_map`.
        mask_bright_stars: Bright-star mask used during SEP extraction.
        segmap: Raw SEP segmentation map (defines original segment pixels).

    Returns:
        Copy of ``gaia_catalog_pixels`` with a ``seg_id_cat`` column.
    """
    h, w = segmap.shape
    has_id = segmap > 0
    filled_seg_map_cat = np.where(has_id | mask_bright_stars, filled_seg_map, 0)

    px_arr = np.clip(
        np.round(gaia_catalog_pixels["pixel_x"].values).astype(int), 0, w - 1
    )
    py_arr = np.clip(
        np.round(gaia_catalog_pixels["pixel_y"].values).astype(int), 0, h - 1
    )
    seg_ids_cat = filled_seg_map_cat[py_arr, px_arr]

    cat_df = gaia_catalog_pixels.copy()
    cat_df["seg_id_cat"] = seg_ids_cat
    return cat_df


def _remove_background_segment_v0(
    data: np.ndarray,
    uncert: np.ndarray = None,
    sigma: float = 2.5,
    sigma_mask: float = 50,
    mask: np.ndarray = None,
    remove_saturated_stars: bool = True,
    gaia_catalog_pixels=None,
    bright_star_mag_threshold: float = 13.0,
) -> tuple[np.ndarray, list[dict]]:
    """Legacy (``segment_v0``) background/star removal, kept bit-identical.

    Removes only the single SEP segment under each in-cell catalogue star.

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
        sep_result = build_sep_background_segmentation(
            data,
            uncert,
            sigma=sigma,
            sigma_mask=sigma_mask,
            close_bright_mask=remove_saturated_stars,
        )
        objects = sep_result.objects
        segmap = sep_result.segmap
        mask_bright_stars = sep_result.mask_bright_stars

        data[np.logical_and(segmap == 0, ~mask_bright_stars)] = 0

        if not remove_saturated_stars:
            return data, removed_stars_list

        has_id = segmap > 0
        filled_seg_map_base = filled_segment_map(segmap)

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
            filled_seg_map_cat = np.where(
                has_id | mask_bright_stars, filled_seg_map_base, 0
            )

            cat_df = catalog_segment_assignments(
                gaia_catalog_pixels,
                filled_seg_map_base,
                mask_bright_stars,
                segmap=segmap,
            )
            seg_ids_cat = cat_df["seg_id_cat"].values
            px_arr = np.clip(
                np.round(gaia_catalog_pixels["pixel_x"].values).astype(int),
                0,
                data.shape[1] - 1,
            )
            py_arr = np.clip(
                np.round(gaia_catalog_pixels["pixel_y"].values).astype(int),
                0,
                data.shape[0] - 1,
            )

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
                # Contract is uint16 (0x1000 = bit 12 needs >= 9 bits), but
                # upcast defensively: a narrower on-disk/cached mask dtype
                # (e.g. stale uint8) would otherwise raise under numpy's
                # strict same-dtype bitwise-AND casting.
                mask_wide = np.asarray(mask).astype(np.int64, copy=False)
                mask_sat = ((mask_wide & 0x0020) != 0) & ((mask_wide & 0x1000) != 0)

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
        """Safe float.
        
        Parameters
        ----------
        val"""
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    def _source_id_for_record(val):
        """Source id for record.
        
        Parameters
        ----------
        val"""
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


# ----------------------------------------------------------------------------
# footprint_v1 removal convention
# ----------------------------------------------------------------------------


def star_footprint_radius(tess_mag):
    """Radius (PS1 px) within which a star's halo is removed.

    ``R(T) = min(480, 160 * 10**(-0.2 * min(T - 10, 0)))``: a constant 160 px
    for T >= 10, growing for brighter stars, capped at the 480 px cell overlap.
    Measured p99 footprint radius of T 8-13 stars is <= 138 px.
    """
    t = np.asarray(tess_mag, dtype=np.float64)
    r = np.minimum(
        float(_CELL_OVERLAP_PX),
        160.0 * 10.0 ** (-0.2 * np.minimum(t - 10.0, 0.0)),
    )
    if r.ndim == 0:
        return float(r)
    return r


def select_catalog_for_cell(
    gaia_catalog: pd.DataFrame,
    wcs,
    cell_shape: tuple,
    *,
    bright_star_mag_threshold: float = 13.0,
) -> pd.DataFrame:
    """Project a catalogue onto a cell and keep in-cell plus nearby bright stars.

    Keeps every star inside the cell (any magnitude) and T < threshold stars
    centred outside the cell but within ``R(T) + 10`` px of it, since their
    halos reach into the cell.  Adds ``pixel_x``, ``pixel_y``, ``tess_mag``
    and ``in_cell``.
    """
    if gaia_catalog is None or len(gaia_catalog) == 0:
        return pd.DataFrame()
    from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel

    h, w = cell_shape
    ra = gaia_catalog["ra"].to_numpy(dtype=np.float64)
    dec = gaia_catalog["dec"].to_numpy(dtype=np.float64)
    px, py = world_ra_dec_to_pixel(wcs, ra, dec)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    n = len(gaia_catalog)
    nan = np.full(n, np.nan)
    g = gaia_catalog["phot_g_mean_mag"].to_numpy(dtype=np.float64)
    bp = (
        gaia_catalog["phot_bp_mean_mag"].to_numpy(dtype=np.float64)
        if "phot_bp_mean_mag" in gaia_catalog.columns else nan
    )
    rp = (
        gaia_catalog["phot_rp_mean_mag"].to_numpy(dtype=np.float64)
        if "phot_rp_mean_mag" in gaia_catalog.columns else nan
    )
    tmag = compute_tess_mag(g, bp, rp)

    finite = np.isfinite(px) & np.isfinite(py)
    in_cell = finite & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    dx = np.maximum(np.maximum(-px, px - (w - 1)), 0.0)
    dy = np.maximum(np.maximum(-py, py - (h - 1)), 0.0)
    dist = np.hypot(dx, dy)
    with np.errstate(invalid="ignore"):
        bright = tmag < bright_star_mag_threshold
        near = bright & (dist <= star_footprint_radius(tmag) + _SELECT_MARGIN_PX)
    keep = in_cell | (finite & near)

    result = gaia_catalog[keep].copy().reset_index(drop=True)
    result["pixel_x"] = px[keep]
    result["pixel_y"] = py[keep]
    result["tess_mag"] = tmag[keep]
    result["in_cell"] = in_cell[keep]
    return result


def _facing_sides(px: float, py: float, h: int, w: int) -> list[str]:
    """Cell sides a star outside the cell faces (light enters through them)."""
    sides = []
    if px < 0:
        sides.append("left")
    if px >= w:
        sides.append("right")
    if py < 0:
        sides.append("bottom")
    if py >= h:
        sides.append("top")
    return sides


def _offcell_lookup(x: float, y: float, valid: np.ndarray):
    """Nearest in-cell ``valid`` pixel to an off-cell star, or ``None``.

    ``valid`` = finite pixel that belongs to a footprint component.  Clips the
    star position onto the cell, then steps inward along the normal(s) of the
    edge(s) it was clipped to, up to ``_EDGE_LOOKUP_INSET`` steps.  Stepping
    past non-component pixels matters because PS1 NaN-masks the outer row and
    SEP leaves a ~2 px dead zone next to NaNs.
    """
    h, w = valid.shape
    # Step inward from any edge within the inset, not only for off-cell stars:
    # an in-cell star centred on the PS1-masked border has no valid own pixel.
    sx = 1 if x < _EDGE_LOOKUP_INSET else (-1 if x > w - 1 - _EDGE_LOOKUP_INSET else 0)
    sy = 1 if y < _EDGE_LOOKUP_INSET else (-1 if y > h - 1 - _EDGE_LOOKUP_INSET else 0)
    lx = int(np.clip(round(x), 0, w - 1))
    ly = int(np.clip(round(y), 0, h - 1))
    for _ in range(_EDGE_LOOKUP_INSET + 1):
        if not (0 <= lx < w and 0 <= ly < h):
            return None
        if valid[ly, lx]:
            return lx, ly
        lx += sx
        ly += sy
    return None


def _nearest_valid(x: float, y: float, valid: np.ndarray, rmax: float):
    """Nearest ``valid`` pixel to ``(x, y)`` within ``rmax`` px, or ``None``.

    For in-cell stars whose centre has no footprint: a saturated core PS1 masks
    as NaN (a T = 7 star's core in 2486.097 is NaN out to ~16 px), or a centre on
    the masked border."""
    h, w = valid.shape
    r = int(np.ceil(rmax))
    xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
    y0, y1, x0, x1 = max(0, yi - r), min(h, yi + r + 1), max(0, xi - r), min(w, xi + r + 1)
    win = valid[y0:y1, x0:x1]
    if not win.any():
        return None
    from scipy import ndimage

    dist, (iy, ix) = ndimage.distance_transform_edt(~win, return_indices=True)
    cy, cx = yi - y0, xi - x0
    if dist[cy, cx] > rmax:
        return None
    return int(ix[cy, cx]) + x0, int(iy[cy, cx]) + y0


def _light_falls_inward(img: np.ndarray, comp: np.ndarray, look: tuple[int, int], step: tuple[int, int]) -> bool:
    """True when the component's light decreases going into the cell from the
    lookup pixel along the inward normal ``step``: the signature of a halo whose
    star is outside. An unrelated source near the edge brightens inward.

    Mean over a 9-px-wide strip perpendicular to the normal, first 5 px vs
    px 11-15 inward; only finite component pixels count.
    """
    h, w = img.shape
    lx, ly = look
    sx, sy = step
    px, py = -sy, sx  # perpendicular direction
    vals = []
    for d in range(16):
        acc = []
        for t in range(-4, 5):
            xx, yy = lx + sx * d + px * t, ly + sy * d + py * t
            if 0 <= xx < w and 0 <= yy < h and comp[yy, xx] and np.isfinite(img[yy, xx]):
                acc.append(float(img[yy, xx]))
        vals.append(np.mean(acc) if acc else np.nan)
    near, far = np.nanmean(vals[:5]), np.nanmean(vals[11:])
    if not np.isfinite(near):
        return False
    return (not np.isfinite(far)) or near > far


def _row_record(cat: pd.DataFrame, i: int, label: int, reason: str) -> dict:
    """Removed-star record for catalogue row ``i`` (positional)."""
    row = next(cat.iloc[[i]].itertuples(index=False))
    return _make_star_record(row, label, reason)


def _remove_background_footprint_v1(
    data: np.ndarray,
    uncert,
    sigma: float,
    sigma_mask: float,
    mask,
    remove_saturated_stars: bool,
    gaia_catalog_pixels,
    bright_star_mag_threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    """``footprint_v1`` removal: zero whole 8-connected star footprints.

    A footprint is a connected component of (SEP segments | bright-core mask |
    saturated pixels), so a halo SEP split into several segments is removed
    as one.  Off-cell bright stars are removed too when their halo enters
    the cell (see :func:`remove_background`).
    """
    from scipy import ndimage

    removed: list[dict] = []
    try:
        sep_result = build_sep_background_segmentation(
            data, uncert, sigma=sigma, sigma_mask=sigma_mask,
            close_bright_mask=remove_saturated_stars,
        )
        segmap = sep_result.segmap
        bright = sep_result.mask_bright_stars

        finite_in = np.isfinite(data)  # PS1-masked (NaN) pixels, before zeroing
        data[np.logical_and(segmap == 0, ~bright)] = 0
        if not remove_saturated_stars:
            return data, removed

        h, w = data.shape
        if mask is None:
            mask_sat = np.zeros(data.shape, dtype=bool)
        else:
            mask_wide = np.asarray(mask).astype(np.int64, copy=False)
            mask_sat = ((mask_wide & 0x0020) != 0) & ((mask_wide & 0x1000) != 0)

        footprint = (segmap > 0) | bright | mask_sat
        labels, _ = ndimage.label(footprint, structure=np.ones((3, 3), dtype=int))
        slices = ndimage.find_objects(labels)

        has_catalog = gaia_catalog_pixels is not None and len(gaia_catalog_pixels) > 0
        cat = None
        removed_labels: set[int] = set()
        bright_recorded: set[int] = set()  # positional row indices
        inside = np.zeros(0, dtype=bool)
        cpx = cpy = ctm = np.zeros(0)

        def _label_at(i: int) -> int:
            return int(labels[int(np.clip(round(cpy[i]), 0, h - 1)),
                              int(np.clip(round(cpx[i]), 0, w - 1))])

        if has_catalog:
            cat = gaia_catalog_pixels.reset_index(drop=True)
            cpx = cat["pixel_x"].to_numpy(dtype=np.float64)
            cpy = cat["pixel_y"].to_numpy(dtype=np.float64)
            ctm = cat["tess_mag"].to_numpy(dtype=np.float64)
            finite = np.isfinite(cpx) & np.isfinite(cpy)
            inside = finite & (cpx >= 0) & (cpx < w) & (cpy >= 0) & (cpy < h)
            in_cell = cat["in_cell"].to_numpy(dtype=bool) if "in_cell" in cat.columns else inside
            with np.errstate(invalid="ignore"):
                candidates = np.flatnonzero(finite & (ctm < bright_star_mag_threshold))

            # Decide off-cell acceptance first, on the pre-removal image, so
            # zeroing for one star cannot change another's peak test.
            accepted: dict[int, set[int]] = {}
            peak_cache: dict[int, tuple[int, int]] = {}
            valid_lookup = finite_in & (labels > 0)
            for i in candidates:
                if in_cell[i]:
                    xi = int(np.clip(round(cpx[i]), 0, w - 1))
                    yi = int(np.clip(round(cpy[i]), 0, h - 1))
                    own = int(labels[yi, xi])
                    labs = {own} if own > 0 else set()
                    if own == 0 or not finite_in[yi, xi]:
                        # Centre without a usable footprint: NaN-masked
                        # saturated core, the masked border, or SEP's dead zone
                        # next to it (a border pixel's label may be only its
                        # saturated-pixel island, not the halo). Also take the
                        # nearest finite footprint pixel.
                        near_px = _nearest_valid(
                            cpx[i], cpy[i], valid_lookup, min(float(star_footprint_radius(ctm[i])), 100.0)
                        )
                        if near_px is not None:
                            labs.add(int(labels[near_px[1], near_px[0]]))
                    accepted[int(i)] = labs
                    continue
                x, y, t = cpx[i], cpy[i], ctm[i]
                look = _offcell_lookup(x, y, valid_lookup)
                if look is None:
                    continue
                lx, ly = look
                lab = int(labels[ly, lx])
                if lab <= 0:
                    continue
                if np.hypot(lx - x, ly - y) > float(star_footprint_radius(t)):
                    continue  # guard (a): lookup too far from the star
                # guard (b): the light at the lookup is the star's halo entering
                # through the facing edge -- either the component's brightest
                # finite pixel lies within _EDGE_BAND_PX of the lookup along the
                # inward normal (measured from the first valid pixel, since
                # PS1's NaN border is several px wide), or the component's light
                # falls going inward from the lookup. An unrelated source near
                # the edge peaks inside and brightens inward.
                if lab not in peak_cache:
                    sl = slices[lab - 1]
                    sub = np.where(
                        (labels[sl] == lab) & finite_in[sl],
                        np.asarray(data[sl], dtype=np.float64), -np.inf,
                    )
                    k = np.unravel_index(int(np.argmax(sub)), sub.shape)
                    peak_cache[lab] = (k[1] + sl[1].start, k[0] + sl[0].start)
                qx, qy = peak_cache[lab]
                step = (1 if x < 0 else (-1 if x >= w else 0), 1 if y < 0 else (-1 if y >= h else 0))
                depth = max(abs(qx - lx) if step[0] else 0, abs(qy - ly) if step[1] else 0)
                if depth <= _EDGE_BAND_PX or _light_falls_inward(data, labels == lab, (lx, ly), step):
                    accepted[int(i)] = {lab}

            for i in candidates:
                labs = sorted(v for v in accepted.get(int(i), set()) if v > 0)
                x, y = cpx[i], cpy[i]
                if not labs:
                    continue

                # Zero each component ∩ disc(480 px) around the star centre.
                for lab in labs:
                    sl = slices[lab - 1]
                    y0 = max(sl[0].start, int(np.floor(y - _CELL_OVERLAP_PX)))
                    y1 = min(sl[0].stop, int(np.ceil(y + _CELL_OVERLAP_PX)) + 1)
                    x0 = max(sl[1].start, int(np.floor(x - _CELL_OVERLAP_PX)))
                    x1 = min(sl[1].stop, int(np.ceil(x + _CELL_OVERLAP_PX)) + 1)
                    if y1 > y0 and x1 > x0:
                        yy, xx = np.ogrid[y0:y1, x0:x1]
                        disc = (yy - y) ** 2 + (xx - x) ** 2 <= float(_CELL_OVERLAP_PX) ** 2
                        sub = data[y0:y1, x0:x1]
                        sub[(labels[y0:y1, x0:x1] == lab) & disc] = 0
                    removed_labels.add(lab)
                bright_recorded.add(int(i))
                removed.append(_row_record(cat, int(i), labs[0], "catalog_bright_star"))

            if removed_labels:
                logger.info(
                    f"[Band] footprint_v1 catalog removal: {len(removed_labels)} "
                    f"footprints zeroed (T < {bright_star_mag_threshold})"
                )
                for i in np.flatnonzero(inside):
                    if int(i) in bright_recorded:
                        continue
                    lab = _label_at(i)
                    if lab in removed_labels:
                        removed.append(_row_record(cat, int(i), lab, "catalog_neighbor"))
            else:
                logger.info("[Band] footprint_v1: no catalog-based removals")

        # Saturation pass, on whole footprints.
        if mask is None:
            logger.warning("[Band] Saturated-star removal requested but no mask was provided.")
        else:
            try:
                sat_labels = [
                    int(v) for v in np.unique(labels[mask_sat])
                    if v > 0 and int(v) not in removed_labels
                ]
                for lab in sat_labels:
                    sl = slices[lab - 1]
                    comp = labels[sl] == lab
                    stars = [int(i) for i in np.flatnonzero(inside) if _label_at(i) == lab]
                    if stars:
                        for i in stars:
                            removed.append(_row_record(cat, i, lab, "quality_flag_star"))
                    else:
                        fv = np.nan_to_num(np.asarray(data[sl], dtype=np.float64))[comp]
                        ys, xs = np.nonzero(comp)
                        total = float(fv.sum())
                        if total > 0:
                            cx = float((xs * fv).sum() / total) + sl[1].start
                            cy = float((ys * fv).sum() / total) + sl[0].start
                        else:
                            cx = float(xs.mean()) + sl[1].start
                            cy = float(ys.mean()) + sl[0].start
                        removed.append({
                            "source_id": -1,
                            "ra": float("nan"),
                            "dec": float("nan"),
                            "pixel_x": float("nan"),
                            "pixel_y": float("nan"),
                            "tess_mag": float("nan"),
                            "phot_g_mean_mag": float("nan"),
                            "phot_bp_mean_mag": float("nan"),
                            "phot_rp_mean_mag": float("nan"),
                            "seg_centroid_x": cx,
                            "seg_centroid_y": cy,
                            "seg_flux": total,
                            "segment_id": lab,
                            "removal_reason": "quality_flag_no_star",
                        })
                    data[sl][comp] = 0
                if sat_labels:
                    logger.info(
                        f"[Band] Quality-flag removal: {len(sat_labels)} additional "
                        f"footprints zeroed (sat+starcore bits)"
                    )
            except Exception as e:
                logger.warning(
                    f"[Band] Quality-flag removal failed; continuing with catalog-only results: {e}"
                )
    except Exception as e:
        logging.error(f"[Band] SEP extraction failed: {e}")
        return data, removed
    return data, removed


def _radial_star_model(img: np.ndarray, x: float, y: float, rmax: int, bin_px: int) -> tuple:
    """Azimuthal running-median profile of ``img`` around ``(x, y)`` out to ``rmax``.

    Returns ``(window, model, profile)``: the window slices, the model image
    over it (0 outside ``rmax``) and the binned profile. The median over each
    ``bin_px``-wide ring ignores NaN and the neighbours that cover a minority of
    the ring, so it is the star's own halo; the outermost ring's level is
    subtracted (no sky pedestal is removed) and the profile is clipped at 0.
    """
    h, w = img.shape
    y0, y1 = max(0, int(np.floor(y - rmax))), min(h, int(np.ceil(y + rmax)) + 1)
    x0, x1 = max(0, int(np.floor(x - rmax))), min(w, int(np.ceil(x + rmax)) + 1)
    nb = int(rmax) // int(bin_px) + 1
    if y1 <= y0 or x1 <= x0:
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0)), np.zeros(nb)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    b = np.minimum((rr // bin_px).astype(np.int64), nb)
    v = np.asarray(img[y0:y1, x0:x1], dtype=np.float64)
    ok = np.isfinite(v) & (b < nb)
    bs, vs = b[ok], v[ok]
    order = np.argsort(bs, kind="stable")
    bs, vs = bs[order], vs[order]
    cuts = np.searchsorted(bs, np.arange(nb + 1))
    prof = np.full(nb, np.nan)
    for k in range(nb):
        seg = vs[cuts[k]:cuts[k + 1]]
        if seg.size >= 8:
            prof[k] = np.median(seg)
    good = np.isfinite(prof)
    if not good.any():
        return (slice(y0, y1), slice(x0, x1)), np.zeros((y1 - y0, x1 - x0)), np.zeros(nb)
    prof = np.interp(np.arange(nb), np.flatnonzero(good), prof[good])
    outer = prof[max(0, nb - max(1, 30 // int(bin_px))):].mean()
    prof = np.clip(prof - outer, 0.0, None)
    model = np.where(b < nb, prof[np.minimum(b, nb - 1)], 0.0)
    return (slice(y0, y1), slice(x0, x1)), model, prof


def _remove_background_starmodel_v1(
    data: np.ndarray,
    uncert,
    sigma: float,
    sigma_mask: float,
    mask,
    remove_saturated_stars: bool,
    gaia_catalog_pixels,
    bright_star_mag_threshold: float,
    *,
    core_nsigma: float = 20.0,
    fill_core: bool = False,
    zero_halo_fragments: bool | str = False,
) -> tuple[np.ndarray, list[dict]]:
    """``starmodel_v1`` removal: subtract each bright star, keep its neighbours.

    ``footprint_v1`` zeroed a bright star's whole 8-connected footprint, which
    also deleted every faint source chained to its halo, so the template held a
    hole at each removed star (dev_runs/removal_hole_20261001). Here:

    1. For every catalogue star with ``T < bright_star_mag_threshold`` whose
       centre lies within ``R(T) + 10`` px of the cell (in or off the cell), a
       radial running-median model of its halo (out to 480 px) is subtracted,
       brightest first, each from the residual of the previous ones.
    2. The residual is background-zeroed exactly as everywhere else (SEP
       segments on the residual; pixels outside segments -> 0), so faint
       sources on the halo stay.
    3. The core, where the model exceeds ``core_nsigma`` times the local noise
       (the radial model cannot follow the PSF structure there; saturated and
       NaN pixels inside are included), is set to 0 (default), or with
       ``fill_core=True`` to the local faint-source level (mean of the zeroed
       residual over the ring core..core+200 px). Filling overshot the core by
       +0.04..+0.06 on all three test sets (the ring next to the core still
       holds residual halo), so it is off.
    4. Saturation-flagged footprints with no bright catalogue star are zeroed as
       in ``footprint_v1`` (``quality_flag_*`` records).
    """
    from scipy import ndimage

    removed: list[dict] = []
    img = np.asarray(data, dtype=np.float32)
    if not remove_saturated_stars:
        return _remove_background_footprint_v1(
            img, uncert, sigma, sigma_mask, mask, False, None, bright_star_mag_threshold)
    h, w = img.shape
    resid = img.astype(np.float64)
    cores: list[tuple] = []
    cat = None
    if gaia_catalog_pixels is not None and len(gaia_catalog_pixels) > 0:
        cat = gaia_catalog_pixels.reset_index(drop=True)
        tm = cat["tess_mag"].to_numpy(dtype=np.float64)
        px = cat["pixel_x"].to_numpy(dtype=np.float64)
        py = cat["pixel_y"].to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore"):
            cand = np.flatnonzero(np.isfinite(px) & np.isfinite(py) & (tm < bright_star_mag_threshold))
        unc = np.asarray(uncert, dtype=np.float64) if uncert is not None else None
        for i in cand[np.argsort(tm[cand], kind="stable")]:
            x, y = px[i], py[i]
            reach = float(star_footprint_radius(tm[i])) + _SELECT_MARGIN_PX
            if not (-reach <= x < w + reach and -reach <= y < h + reach):
                continue
            win, model, prof = _radial_star_model(resid, x, y, _CELL_OVERLAP_PX, 3)
            if model.size == 0:
                continue
            resid[win] -= model
            if unc is not None:
                sub = unc[win]
                noise = float(np.nanmedian(sub[np.isfinite(sub) & (sub > 0)])) if np.any(np.isfinite(sub) & (sub > 0)) else np.nan
            else:
                noise = np.nan
            if not np.isfinite(noise) or noise <= 0:
                noise = 1.4826 * float(np.nanmedian(np.abs(resid[win] - np.nanmedian(resid[win]))))
            above = np.flatnonzero(prof > core_nsigma * noise)
            r_core = (int(above.max()) + 1) * 3 if above.size else 0
            cores.append((int(i), x, y, r_core))
            removed.append(_row_record(cat, int(i), -1, "catalog_bright_star"))
    resid = resid.astype(np.float32)
    # Background zeroing on the residual, exactly as everywhere else.
    sep_result = build_sep_background_segmentation(
        resid, uncert, sigma=sigma, sigma_mask=sigma_mask, close_bright_mask=True)
    segmap, bright = sep_result.segmap, sep_result.mask_bright_stars
    out = resid.copy()
    out[np.logical_and(segmap == 0, ~bright)] = 0
    finite_in = np.isfinite(img)

    core_mask = np.zeros((h, w), dtype=bool)
    core_windows = []
    for i, x, y, r_core in cores:
        if r_core <= 0:
            continue
        rr_max = r_core + 200
        y0, y1 = max(0, int(np.floor(y - rr_max))), min(h, int(np.ceil(y + rr_max)) + 1)
        x0, x1 = max(0, int(np.floor(x - rr_max))), min(w, int(np.ceil(x + rr_max)) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        rr = np.hypot(xx - x, yy - y)
        core = rr < r_core
        if core.any():
            core_mask[y0:y1, x0:x1] |= core
            core_windows.append((y0, y1, x0, x1, core, rr >= r_core))

    # Saturation pass (footprint_v1 rule) for saturated footprints that are
    # not a modelled bright star's core: uncatalogued or T >= threshold stars.
    if mask is not None:
        mw = np.asarray(mask).astype(np.int64, copy=False)
        mask_sat = ((mw & 0x0020) != 0) & ((mw & 0x1000) != 0)
        labels, _ = ndimage.label((segmap > 0) | bright | mask_sat, structure=np.ones((3, 3), dtype=int))
        sat_labels = set(np.unique(labels[mask_sat]).tolist()) - {0}
        sat_labels -= set(np.unique(labels[core_mask]).tolist())
        if sat_labels:
            slices = ndimage.find_objects(labels)
            for lab in sorted(sat_labels):
                sl = slices[lab - 1]
                comp = labels[sl] == lab
                ys, xs = np.nonzero(comp)
                fv = np.nan_to_num(np.asarray(out[sl], dtype=np.float64))[comp]
                total = float(fv.sum())
                cx = float((xs * fv).sum() / total) + sl[1].start if total > 0 else float(xs.mean()) + sl[1].start
                cy = float((ys * fv).sum() / total) + sl[0].start if total > 0 else float(ys.mean()) + sl[0].start
                removed.append({
                    "source_id": -1, "ra": float("nan"), "dec": float("nan"),
                    "pixel_x": float("nan"), "pixel_y": float("nan"), "tess_mag": float("nan"),
                    "phot_g_mean_mag": float("nan"), "phot_bp_mean_mag": float("nan"),
                    "phot_rp_mean_mag": float("nan"), "seg_centroid_x": cx, "seg_centroid_y": cy,
                    "seg_flux": total, "segment_id": int(lab), "removal_reason": "quality_flag_no_star",
                })
                out[sl][comp] = 0

    # Halo fragments left by the radial model (diffraction spikes, asymmetric
    # wings): a residual segment within R(T) of a modelled star whose brightest
    # pixel lies on its edge facing the star (within 3 px of its nearest pixel),
    # and which holds no catalogue source, is the star's light -> 0. A real
    # source peaks inside its own segment and is kept.
    if zero_halo_fragments and cores:
        seg_slices = ndimage.find_objects(segmap)
        src_segs: set[int] = set()
        if cat is not None:
            bright_ids = {i for i, *_ in cores}
            keep_rows = [j for j in range(len(cat)) if j not in bright_ids]
            sx = cat["pixel_x"].to_numpy(dtype=np.float64)[keep_rows]
            sy = cat["pixel_y"].to_numpy(dtype=np.float64)[keep_rows]
            ok = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
            src_segs = set(np.unique(segmap[sy[ok].round().astype(int).clip(0, h - 1),
                                            sx[ok].round().astype(int).clip(0, w - 1)]).tolist()) - {0}
        for i, x, y, _r in cores:
            reach = float(star_footprint_radius(tm[i]))
            y0, y1 = max(0, int(np.floor(y - reach))), min(h, int(np.ceil(y + reach)) + 1)
            x0, x1 = max(0, int(np.floor(x - reach))), min(w, int(np.ceil(x + reach)) + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            for s in np.unique(segmap[y0:y1, x0:x1]):
                if s == 0 or s in src_segs or seg_slices[s - 1] is None:
                    continue
                sl = seg_slices[s - 1]
                m = segmap[sl] == s
                ys, xs = np.nonzero(m)
                ys, xs = ys + sl[0].start, xs + sl[1].start
                d = np.hypot(xs - x, ys - y)
                if d.min() > reach:
                    continue
                if zero_halo_fragments == "uncatalogued":
                    out[ys, xs] = 0  # no catalogue source in it: treated as the star's residual light
                    continue
                vals = np.nan_to_num(resid[ys, xs])
                if d[int(np.argmax(vals))] < d.min() + 3:
                    out[ys, xs] = 0

    # Core: local faint-source level (or 0), measured outside every core.
    for y0, y1, x0, x1, core, ring in core_windows:
        sub = out[y0:y1, x0:x1]
        ring = ring & finite_in[y0:y1, x0:x1] & ~core_mask[y0:y1, x0:x1]
        level = float(np.nanmean(sub[ring])) if (fill_core and ring.any()) else 0.0
        sub[core] = level
    return out, removed


def remove_background(
    data: np.ndarray,
    uncert: np.ndarray = None,
    sigma: float = 2.5,
    sigma_mask: float = 50,
    mask: np.ndarray = None,
    remove_saturated_stars: bool = True,
    gaia_catalog_pixels=None,
    bright_star_mag_threshold: float = 13.0,
    convention: str = REMOVAL_CONVENTION,
) -> tuple[np.ndarray, list[dict]]:
    """Remove background and bright-star light from a skycell image.

    ``convention="segment_v0"`` is the legacy single-segment removal;
    ``"footprint_v1"`` (default) removes whole connected footprints, including
    those of bright stars centred just outside the cell.  Use
    :func:`select_catalog_for_cell` to build ``gaia_catalog_pixels`` for the
    new convention.  See :func:`_remove_background_segment_v0` for the record
    schema (``segment_id`` is the component label under ``footprint_v1``).
    """
    if convention == REMOVAL_CONVENTION_LEGACY:
        return _remove_background_segment_v0(
            data, uncert, sigma, sigma_mask, mask, remove_saturated_stars,
            gaia_catalog_pixels, bright_star_mag_threshold,
        )
    if convention == REMOVAL_CONVENTION_STARMODEL:
        return _remove_background_starmodel_v1(
            data, uncert, sigma, sigma_mask, mask, remove_saturated_stars,
            gaia_catalog_pixels, bright_star_mag_threshold,
        )
    if convention == REMOVAL_CONVENTION_FOOTPRINT:
        return _remove_background_footprint_v1(
            data, uncert, sigma, sigma_mask, mask, remove_saturated_stars,
            gaia_catalog_pixels, bright_star_mag_threshold,
        )
    raise ValueError(f"Unknown removal convention: {convention!r}")
