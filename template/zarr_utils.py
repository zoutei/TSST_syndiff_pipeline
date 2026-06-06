"""
Simple zarr utilities for PS1 data loading and saving.

Function-oriented approach for maximum simplicity.
Refactored to accept open Zarr store objects and use sequential loading internally.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_skycell_bands_masks_and_headers(zarr_store, projection: str, skycell: str) -> tuple[dict, dict, dict, dict]:
    """Load band data, masks, weights (variance maps), and headers for a single skycell sequentially.

    Args:
        zarr_store: An open Zarr store object.
        projection: PS1 projection ID
        skycell: Skycell ID

    Returns:
        Tuple of (bands_data, masks_data, weights_data, headers_data, headers_weight_data) dictionaries
    """
    if skycell.startswith("skycell."):
        skycell_key = skycell
    else:
        skycell_key = f"skycell.{projection}.{skycell.split('.')[-1]}"

    try:
        skycell_group = zarr_store[projection][skycell_key]
    except KeyError:
        logger.warning(f"Skycell group not found: {projection}/{skycell_key}")
        return {}, {}, {}, {}, {}

    bands_data = {}
    masks_data = {}
    weights_data = {}
    headers_data = {}
    headers_weight_data = {}

    # Load bands, masks, weights (variance), and headers sequentially
    for band in ["r", "i", "z", "y"]:
        band_data, mask_data, weight_data, header_data = None, None, None, None

        if band in skycell_group:
            band_array = skycell_group[band]
            band_data = np.array(band_array)
            if hasattr(band_array, "attrs") and "header" in band_array.attrs:
                header_data = band_array.attrs["header"]

        mask_key = f"{band}_mask"
        if mask_key in skycell_group:
            mask_data = np.array(skycell_group[mask_key])

        weight_key = f"{band}_wt"
        if weight_key in skycell_group:
            weight_array = skycell_group[weight_key]
            weight_data = np.array(weight_array)
            if hasattr(weight_array, "attrs") and "header" in weight_array.attrs:
                header_weight_data = weight_array.attrs["header"]

        if band_data is not None:
            bands_data[band] = band_data
        if mask_data is not None:
            masks_data[band] = mask_data
        if weight_data is not None:
            weights_data[band] = weight_data
        if header_data is not None:
            headers_data[band] = header_data
        if header_weight_data is not None:
            headers_weight_data[band] = header_weight_data

    return bands_data, masks_data, weights_data, headers_data, headers_weight_data


def save_convolved_results(output_store, projection: str, row_id: int, results_data: dict, results_masks: dict, results_weights: dict = None) -> None:
    """Save convolved results, masks, and weights for a row.

    Args:
        output_store: An open, writable Zarr store object.
        projection: PS1 projection ID
        row_id: Row identifier
        results_data: Dictionary mapping skycell names to convolved data arrays
        results_masks: Dictionary mapping skycell names to mask arrays
        results_weights: Dictionary mapping skycell names to weight arrays (optional)
    """

    # Compressor and chunking policy
    compressor = {"name": "zstd", "configuration": {"level": 3}}

    for skycell_name, convolved_data in results_data.items():
        # Ensure data is numpy array
        if not isinstance(convolved_data, np.ndarray):
            convolved_data = np.array(convolved_data)

        # Choose chunks conservatively based on shape
        chunks = (min(1024, convolved_data.shape[0]), min(1024, convolved_data.shape[1]))

        array_name = f"{skycell_name}_data"
        # Remove existing array if present
        if array_name in output_store:
            del output_store[array_name]

        # Create array with float data and NaN fill
        output_store.create_array(name=array_name, data=convolved_data, chunks=chunks, compressors=[compressor], fill_value=np.nan)

    for skycell_name, mask_data in results_masks.items():
        # Ensure mask is numpy array
        if not isinstance(mask_data, np.ndarray):
            mask_data = np.array(mask_data)

        # Convert to uint16 if needed (store masks as compact uint16)
        if mask_data.dtype != np.uint16:
            mask_data = mask_data.astype(np.uint16)

        chunks = (min(1024, mask_data.shape[0]), min(1024, mask_data.shape[1]))

        array_name = f"{skycell_name}_mask"
        if array_name in output_store:
            del output_store[array_name]

        output_store.create_array(name=array_name, data=mask_data, chunks=chunks, compressors=[compressor], fill_value=0)

    # Save weights if provided
    if results_weights:
        for skycell_name, weight_data in results_weights.items():
            # Ensure weight is numpy array
            if not isinstance(weight_data, np.ndarray):
                weight_data = np.array(weight_data)

            # Convert to float32 if needed
            if weight_data.dtype != np.float32:
                weight_data = weight_data.astype(np.float32)

            chunks = (min(1024, weight_data.shape[0]), min(1024, weight_data.shape[1]))

            array_name = f"{skycell_name}_weight"
            if array_name in output_store:
                del output_store[array_name]

            output_store.create_array(name=array_name, data=weight_data, chunks=chunks, compressors=[compressor], fill_value=0.0)

    logger.info(f"Saved {len(results_data)} convolved cells, {len(results_masks)} masks{' and ' + str(len(results_weights)) + ' weights' if results_weights else ''} for proj {projection}, row {row_id}")
