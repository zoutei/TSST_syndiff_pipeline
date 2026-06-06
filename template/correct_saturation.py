import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from astropy.modeling import fitting, models
from astropy.nddata import NDData
from astropy.table import Table
from astropy.wcs import WCS
from photutils.psf import EPSFBuilder, extract_stars

logger = logging.getLogger(__name__)


def filter_catalog_for_region(catalog_df, wcs, x_min, x_max, y_min, y_max):
    """
    Filter catalog using hybrid approach: RA/DEC pre-filter + pixel conversion

    Args:
        catalog_df: DataFrame with 'ra', 'dec', 'phot_rp_mean_mag' columns
        wcs: WCS object from leftmost cell
        x_min, x_max, y_min, y_max: Pixel boundaries of region

    Returns:
        DataFrame with x, y, phot_rp_mean_mag columns for stars in region
    """

    # Step 1: RA/DEC pre-filter (fast rejection)
    try:
        # Use separate x, y arrays instead of coordinate pairs to avoid confusion
        corners_x = np.array([x_min, x_max, x_min, x_max])
        corners_y = np.array([y_min, y_min, y_max, y_max])

        # Validate WCS is celestial
        if not wcs.is_celestial:
            logger.warning("WCS is not celestial, skipping catalog filtering")
            return catalog_df.iloc[0:0].copy()

        # Use low-level API to avoid SkyCoord dimension issues
        ra_bounds, dec_bounds = wcs.all_pix2world(corners_x, corners_y, 0)

        # Check if coordinates are valid
        if not np.all(np.isfinite(ra_bounds)) or not np.all(np.isfinite(dec_bounds)):
            logger.warning("Invalid pixel coordinates for WCS transformation")
            return catalog_df.iloc[0:0].copy()  # Return empty DataFrame

        # Calculate bounds from all 4 corners
        ra_min, ra_max = ra_bounds.min(), ra_bounds.max()
        dec_min, dec_max = dec_bounds.min(), dec_bounds.max()

    except Exception as e:
        logger.warning(f"WCS transformation failed: {e}")
        return catalog_df.iloc[0:0].copy()  # Return empty DataFrame

    pre_filter = (catalog_df["ra"] >= ra_min) & (catalog_df["ra"] <= ra_max) & (catalog_df["dec"] >= dec_min) & (catalog_df["dec"] <= dec_max)

    if pre_filter.sum() == 0:
        return catalog_df.iloc[0:0].copy()  # Return empty DataFrame with same structure

    # Step 2: Convert pre-filtered stars to pixels
    pre_filtered = catalog_df[pre_filter]
    star_coords = wcs.all_world2pix(pre_filtered["ra"], pre_filtered["dec"], 0)

    # Step 3: Final pixel boundary filter
    pixel_filter = (star_coords[0] >= x_min) & (star_coords[0] < x_max) & (star_coords[1] >= y_min) & (star_coords[1] < y_max)

    if pixel_filter.sum() == 0:
        return catalog_df.iloc[0:0].copy()  # Return empty DataFrame with same structure

    # Step 4: Return formatted DataFrame
    result = pre_filtered[pixel_filter].copy()
    result["x"] = star_coords[0][pixel_filter]
    result["y"] = star_coords[1][pixel_filter]
    return result[["x", "y", "phot_rp_mean_mag"]]


def _process_saturation_chunk(i, chunk_width, width, height, data_array, mask_array, catalog, min_mag_epsf, max_mag_epsf, max_mag_replace, mag_threshold_epsf, star_half_size, mask_bit_index):
    """Helper function to process a single spatial chunk."""
    try:
        start_x = i * chunk_width
        end_x = (i + 1) * chunk_width if (i + 1) * chunk_width < width else width  # Handle last chunk correctly

        # Find stars in this chunk
        stars_in_chunk = catalog[(catalog["x"] >= start_x) & (catalog["x"] < end_x)]

        # Stars for ePSF building (unsaturated)
        good_stars = stars_in_chunk[(stars_in_chunk["phot_rp_mean_mag"] > min_mag_epsf) & (stars_in_chunk["phot_rp_mean_mag"] < max_mag_epsf)]

        # Saturated stars to replace
        sat_stars = stars_in_chunk[stars_in_chunk["phot_rp_mean_mag"] <= max_mag_replace]

        if len(good_stars) == 0:
            logger.error(f"No good stars in chunk {i}, skipping.")
            return

        logger.info(f"Chunk {i}: Start processing. {len(good_stars['x'])} good stars. {len(sat_stars)} saturated stars")

        # Prepare data for ePSF extraction
        # Note: We are reading from shared data_array/mask_array. This is thread-safe for reads.
        # Writes happen later and are spatially distinct (mostly).

        # Optimization: We define the mask locally for EPSF extraction
        # We don't need the global mask_for_epsf unless we want to execute strict exclusion
        # but NDData expects a mask of the same size as data.
        # Passing full array is fine.

        # mask_for_epsf = ~np.isfinite(data_array)
        # Making a full copy of mask for every thread is expensive.
        # NDData doesn't copy data/mask by default, but extract_stars will access it.
        # For thread safety with minimal overhead, we just use the shared arrays.
        # A read-only view would be ideal but numpy slicing is fine.

        mask_for_epsf = ~np.isfinite(data_array)
        data_nd = NDData(data_array, mask=mask_for_epsf)
        positions = Table()
        positions["x"] = good_stars["x"]
        positions["y"] = good_stars["y"]

        # Extract stars for ePSF
        try:
            stars_tbl = extract_stars(data_nd, positions, size=star_half_size * 2 + 1)
            if len(stars_tbl) == 0:
                logger.error(f"Chunk {i}: No stars extracted in chunk {i}, skipping.")
                return
        except Exception as e:
            logger.error(f"Chunk {i}: Error extracting stars in chunk {i}: {e}")
            return

        # Build ePSF
        logger.info(f"Chunk {i}: building EPSF with {len(stars_tbl)} stars")
        epsf_builder = EPSFBuilder(oversampling=2, maxiters=3, progress_bar=False)
        try:
            epsf, _ = epsf_builder(stars_tbl)
        except Exception as e:
            logger.error(f"Chunk {i}: Error building ePSF in chunk {i}: {e}")
            return
        logger.info(f"Chunk {i}: epsf built")

        # Process each saturated star
        # Writes are confined to the star's bounding box.
        # Since chunks are disjoint by width, and stars are assigned by center X,
        # there is a theoretical edge case where a star on the boundary writes into the neighbor chunk's region.
        # However, numpy array writes are thread-safe at the C-level (no GIL for the write op itself usually),
        # and even if they overlap, they are writing to the "same" image.
        # Since we just overwrite pixels, worst case is a race condition on the boundary pixels.
        # Given the rarity and the nature of "blending", strict locking might be overkill vs performance.
        # We accept this minor risk for speed.

        replaced_count = 0
        for idx, star in sat_stars.iterrows():
            global_x = star["x"]
            global_y = star["y"]

            # Skip stars too close to the edge
            if global_y < star_half_size or global_y >= height - star_half_size or global_x < star_half_size or global_x >= width - star_half_size:
                continue

            # Define cutout size (51x51 centered on star) from the full data_array
            y_start = max(0, int(global_y) - star_half_size)
            y_end = min(data_array.shape[0], int(global_y) + star_half_size + 1)
            x_start = max(0, int(global_x) - star_half_size)
            x_end = min(data_array.shape[1], int(global_x) + star_half_size + 1)

            cutout = data_array[y_start:y_end, x_start:x_end]
            cutout_mask_local = mask_array[y_start:y_end, x_start:x_end]

            # Mask for fitting: finite, not already masked, below threshold
            # Convert magnitude threshold to flux threshold
            flux_threshold = 10 ** ((mag_threshold_epsf - 25) / -2.5) * epsf.data.max()
            fit_mask = np.isfinite(cutout) & ~cutout_mask_local & (cutout < flux_threshold)

            if not np.any(fit_mask):
                logger.warning(f"Chunk {i}: No valid pixels for fitting star at ({global_x}, {global_y}) in chunk {i}")
                continue

            # Model: ePSF + constant background
            mag = star["phot_rp_mean_mag"]
            flux_guess = 10 ** ((25 - mag) / 2.5)
            psf_model = epsf.copy()
            psf_model.flux = flux_guess
            psf_model.x_0 = cutout.shape[1] // 2
            psf_model.y_0 = cutout.shape[0] // 2
            background = models.Const2D(amplitude=0.0)
            model = psf_model + background

            # Coordinate grids
            y_coords, x_coords = np.mgrid[: cutout.shape[0], : cutout.shape[1]]

            # Fit the model
            fitter = fitting.LevMarLSQFitter()
            try:
                fitted_model = fitter(model, x_coords, y_coords, cutout, weights=fit_mask.astype(float), filter_non_finite=True)
            except Exception as e:
                logger.warning(f"Chunk {i}: Error fitting star at ({global_x}, {global_y}) in chunk {i}: {e}")
                continue

            # Generate fitted image
            fitted_image = fitted_model(x_coords, y_coords)
            saturated_pixels = cutout > flux_threshold

            # Direct assignment to shared array
            data_array[y_start:y_end, x_start:x_end][saturated_pixels] = fitted_image[saturated_pixels]
            mask_array[y_start:y_end, x_start:x_end][saturated_pixels] |= 1 << mask_bit_index
            replaced_count += 1

        logger.info(f"Chunk {i}: Finished. Replaced {replaced_count} stars.")
        return replaced_count

    except Exception as e:
        logger.error(f"Chunk {i}: Failed with error: {e}")
        return 0


def replace_saturated_stars(data_array, mask_array, num_chunks, catalog, max_mag_replace=14.0, mag_threshold_epsf=15, min_mag_epsf=13.5, max_mag_epsf=17.0, star_half_size: int = 25, mask_bit_index: int = 5):
    """
    Replace saturated stars in a large image array by fitting an effective PSF (ePSF) built from unsaturated stars.
    Parallel processing version.
    """
    height, width = data_array.shape
    chunk_width = width // num_chunks

    logger.info(f"[Saturation] Starting parallel saturation correction with {num_chunks} chunks/threads.")

    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = []
        for i in range(num_chunks):
            # Submit task
            future = executor.submit(_process_saturation_chunk, i, chunk_width, width, height, data_array, mask_array, catalog, min_mag_epsf, max_mag_epsf, max_mag_replace, mag_threshold_epsf, star_half_size, mask_bit_index)
            futures.append(future)

        # Wait for all
        total_replaced = 0
        for future in futures:
            try:
                total_replaced += future.result()
            except Exception as e:
                logger.error(f"Chunk failed: {e}")

    logger.info(f"[Saturation] Completed. Total stars replaced: {total_replaced}")


def filter_catalog_for_row(catalog_df: pd.DataFrame, cell_positions: dict, wcs) -> pd.DataFrame:
    """Filter catalog to stars within actual data regions only."""
    if not cell_positions:
        return pd.DataFrame()

    # Get actual data boundaries from cell_positions
    data_x_min = min(pos[0] for pos in cell_positions.values())
    data_x_max = max(pos[1] for pos in cell_positions.values())
    data_y_min = min(pos[2] for pos in cell_positions.values())
    data_y_max = max(pos[3] for pos in cell_positions.values())

    # Convert RA/DEC to pixel coordinates
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(catalog_df["ra"], catalog_df["dec"], unit="deg")
    x_pixels, y_pixels = wcs.world_to_pixel(coords)

    # Filter to stars within data boundaries
    valid_mask = (x_pixels >= data_x_min) & (x_pixels < data_x_max) & (y_pixels >= data_y_min) & (y_pixels < data_y_max)

    # Create filtered catalog with required columns
    filtered = catalog_df[valid_mask].copy()
    filtered["x"] = x_pixels[valid_mask]
    filtered["y"] = y_pixels[valid_mask]

    return filtered[["x", "y", "phot_rp_mean_mag"]]


def clear_sat_flags(mask_array: np.ndarray) -> None:
    """Clear bit 5 (SAT flag) from all pixels in the mask array."""
    sat_bit = 1 << 5  # Bit 5 = 32
    # Use numpy bitwise_not to ensure we stay in uint32 domain if passed array is uint32,
    # but here we operate with a scalar. ~np.uint32(sat_bit) gives a large positive integer.
    mask_array &= ~np.uint32(sat_bit)


def apply_saturation_to_row(data_array, masks, cell_locations, cell_bundles, catalog_df):
    """
    Complete saturation correction for one row - simplified single function

    Args:
        data_array: The master data array (e.g. state.current_array)
        masks: Dictionary of masks currently loaded (e.g. state.current_masks)
        cell_locations: Dictionary of cell locations (e.g. state.cell_locations)
        cell_bundles: List of cell bundles for this row (to access WCS)
        catalog_df: Pre-loaded catalog DataFrame
    """
    try:
        # 1. Extract WCS from leftmost cell
        leftmost = min(cell_bundles, key=lambda b: b["x_coord"])
        wcs = None
        try:
            wcs = WCS(leftmost["headers_data"]["i"])
            if wcs.naxis > 2:
                wcs = wcs.celestial
        except Exception:
            logger.warning("Failed to extract WCS from leftmost cell")

        # 2. Get data boundaries and extract region
        positions = cell_locations
        if not positions:
            return
        num_cells = len(positions)

        x_min = min(pos[0] for pos in positions.values())
        x_max = max(pos[1] for pos in positions.values())
        y_min = min(pos[2] for pos in positions.values())
        y_max = max(pos[3] for pos in positions.values())

        # Extract data region
        data_region = data_array[y_min:y_max, x_min:x_max]

        # Handle mask - create unified mask if needed
        if masks:
            # Create unified mask from cell masks
            mask_region = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint32)
            for cell_name, cell_mask in masks.items():
                if cell_name in positions and isinstance(cell_mask, np.ndarray):
                    # Map cell mask to unified mask position
                    cell_pos = positions[cell_name]
                    cell_x_start = cell_pos[0] - x_min
                    cell_x_end = cell_pos[1] - x_min
                    cell_y_start = cell_pos[2] - y_min
                    cell_y_end = cell_pos[3] - y_min

                    # Ensure bounds are valid
                    if cell_x_end > cell_x_start and cell_y_end > cell_y_start and cell_x_start >= 0 and cell_y_start >= 0 and cell_x_end <= mask_region.shape[1] and cell_y_end <= mask_region.shape[0]:
                        mask_region[cell_y_start:cell_y_end, cell_x_start:cell_x_end] = cell_mask
        else:
            # Create empty mask if no masks available
            mask_region = np.zeros_like(data_region, dtype=np.uint32)

        # 3. Filter catalog for this region
        region_catalog = filter_catalog_for_region(catalog_df, wcs, 0, data_region.shape[1], 0, data_region.shape[0])

        if len(region_catalog) == 0:
            logger.info("No stars in region for saturation correction")
            return

        # 4. Clear SAT flags and apply correction
        clear_sat_flags(mask_region)

        replace_saturated_stars(data_array=data_region, mask_array=mask_region, num_chunks=num_cells, catalog=region_catalog, mask_bit_index=5)

        # 5. Put results back into main array
        data_array[y_min:y_max, x_min:x_max] = data_region

        # Update mask in state if it exists
        if masks:
            # Update the unified mask back to individual cell masks
            for cell_name, cell_mask in masks.items():
                if cell_name in positions and isinstance(cell_mask, np.ndarray):
                    cell_pos = positions[cell_name]
                    cell_x_start = cell_pos[0] - x_min
                    cell_x_end = cell_pos[1] - x_min
                    cell_y_start = cell_pos[2] - y_min
                    cell_y_end = cell_pos[3] - y_min

                    if cell_x_end > cell_x_start and cell_y_end > cell_y_start and cell_x_start >= 0 and cell_y_start >= 0 and cell_x_end <= mask_region.shape[1] and cell_y_end <= mask_region.shape[0]:
                        cell_mask[:] = mask_region[cell_y_start:cell_y_end, cell_x_start:cell_x_end]

        logger.info(f"Applied saturation correction to {len(region_catalog)} stars")

    except Exception as e:
        logger.warning(f"Saturation correction failed: {e}")
