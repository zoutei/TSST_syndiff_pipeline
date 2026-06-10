"""
Zarr to Combined FITS Extractor

This script extracts combined r,i,z,y band data from a Zarr store and converts it back to a single FITS file.
Useful for accessing combined skycells from the unified Zarr store.

Features:
- Extract combined r,i,z,y bands for a skycell
- Apply flux conversion using stored headers
- Reconstruct FITS headers from stored metadata
- Save as standard FITS files
- Command-line interface

Usage:
    python zarr_to_combined_fits.py skycell.2556.080 --output "output.fits"
    python zarr_to_combined_fits.py skycell.2556.080 --batch --output-dir "extracted_fits"
"""

import argparse
import logging
import os

# Add parent directory to path for imports
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from astropy.io import fits

sys.path.append(str(Path(__file__).parent.parent))

from band_utils import combine_masks, combine_rizy_bands
from zarr_utils import load_skycell_bands_masks_and_headers

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_projection_from_name(skycell_name: str) -> Optional[str]:
    """Extract projection ID from skycell name."""
    try:
        return skycell_name.split(".")[1]
    except IndexError:
        logging.warning(f"Could not parse projection from skycell name: {skycell_name}")
        return None


def extract_combined_from_zarr(zarr_path: Path, skycell_name: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    """
    Extract and combine all bands for a skycell from the Zarr store.

    Returns:
        tuple: (combined_data_array, header_string) or (None, None) if not found
    """
    projection_id = get_projection_from_name(skycell_name)
    if not projection_id:
        return None, None

    try:
        # Open zarr store
        root = zarr.open(zarr_path, mode="r")

        # Load all bands, masks, and headers using the utility function
        bands_data, masks_data, weights_data, headers_data, headers_weight_data = load_skycell_bands_masks_and_headers(root, projection_id, skycell_name)

        if not bands_data:
            logging.error(f"No bands found for skycell {skycell_name}")
            return None, None

        # Combine the bands using the utility function
        combined_data, combined_weights = combine_rizy_bands(bands_data, headers_data=headers_data, bands_weights=weights_data, headers_weight_data=headers_weight_data)
        combined_mask = combine_masks(masks_data)

        # Use the header from the 'r' band as representative, or first available
        header_string = None
        for band in ["r", "i", "z", "y"]:
            if band in headers_data:
                header_string = headers_data[band]
                break

        logging.info(f"Successfully extracted and combined bands for {skycell_name}")
        logging.info(f"Combined data shape: {combined_data.shape}, dtype: {combined_data.dtype}")

        return combined_data, combined_weights, combined_mask, header_string

    except Exception as e:
        logging.error(f"Error extracting combined data: {e}")
        return None, None, None, None


def reconstruct_fits_header(header_string: Optional[str], band: str, is_mask: bool = False) -> fits.Header:
    """
    Reconstruct a FITS header from the stored header string.
    """
    if header_string:
        try:
            # Parse the header string back into a FITS header
            header = fits.Header.fromstring(header_string)
            # remove BSOFTEN and BOFFSET
            for key in ["BOFFSET", "BSOFTEN"]:
                if key in header:
                    del header[key]
        except Exception as e:
            logging.warning(f"Could not parse stored header: {e}")
            header = fits.Header()
    else:
        header = fits.Header()

    return header


def save_as_fits(data: np.ndarray, header: fits.Header, output_path: Path, weights: np.ndarray = None, mask: np.ndarray = None) -> bool:
    """
    Save data as a FITS file.
    """
    try:
        # Create HDU and add data to extensions weights and mask
        hdul_list = []
        hdu = fits.PrimaryHDU(header=header)
        hdu1 = fits.ImageHDU(data, name="SCI", header=header)
        hdul_list.append(hdu)
        hdul_list.append(hdu1)
        if weights is not None:
            hdu2 = fits.ImageHDU(weights, name="WEIGHT", header=header)
            hdul_list.append(hdu2)
        if mask is not None:
            hdu3 = fits.ImageHDU(mask, name="MASK", header=header)
            hdul_list.append(hdu3)

        hdul = fits.HDUList(hdul_list)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        hdul.writeto(output_path, overwrite=True)

        os.system(f"fpack -F -Y {output_path}")

        logging.info(f"Successfully saved FITS file: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving FITS file: {e}")
        return False


def extract_combined_skycell_to_fits(zarr_path: Path, skycell_name: str, output_path: Optional[Path] = None) -> bool:
    """
    Complete pipeline to extract combined data from Zarr and save as FITS.
    """
    # Generate output filename if not provided
    if output_path is None:
        output_path = Path(f"{skycell_name}_combined.fits")

    # Extract combined data from Zarr
    combined_data, combined_weights, combined_mask, header_string = extract_combined_from_zarr(zarr_path, skycell_name)

    if combined_data is None:
        return False

    # Reconstruct FITS header
    header = reconstruct_fits_header(header_string, "combined")

    # Save as FITS
    return save_as_fits(combined_data, header, output_path, combined_weights, combined_mask)


def batch_extract_combined_skycell(zarr_path: Path, skycell_name: str, output_dir: Path) -> bool:
    """
    Extract combined image for a given skycell.
    """
    output_dir = output_dir / skycell_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract combined image
    combined_path = output_dir / f"{skycell_name}_combined.fits"
    success = extract_combined_skycell_to_fits(zarr_path, skycell_name, output_path=combined_path)

    return success


def list_available_data(zarr_path: Path) -> dict[str, list[str]]:
    """
    List all available projections and skycells in the Zarr store.
    """
    try:
        root = zarr.open(zarr_path, mode="r")

        available_data = {}

        # Handle both old and new structures
        if "projections" in root:
            # Old structure: projections/[projection_id]/[skycell_name]
            projections_group = root["projections"]
            for projection_id in projections_group.group_keys():
                skycells = list(projections_group[projection_id].group_keys())
                available_data[projection_id] = skycells
        else:
            # New structure: [projection_id]/[skycell_name]
            for projection_id in root.group_keys():
                skycells = list(root[projection_id].group_keys())
                available_data[projection_id] = skycells

        return available_data

    except Exception as e:
        logging.error(f"Error listing available data: {e}")
        return {}


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Extract combined FITS files from Zarr store")

    parser.add_argument("--zarr-path", type=str, default="data/ps1_skycells_zarr/ps1_skycells.zarr", help="Path to the Zarr store")

    parser.add_argument("skycell", type=str, help="Skycell name (e.g., skycell.0306.067)")

    parser.add_argument("--output", type=str, help="Output FITS file path. If not specified, auto-generates filename")

    parser.add_argument("--batch", action="store_true", help="Extract combined image for the specified skycell")

    parser.add_argument("--list", action="store_true", help="List available projections and skycells")

    parser.add_argument("--output-dir", type=str, default="extracted_fits", help="Output directory for batch extraction")

    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)

    if not zarr_path.exists():
        logging.error(f"Zarr store not found: {zarr_path}")
        return

    # List available data
    if args.list:
        available_data = list_available_data(zarr_path)
        print("\nAvailable data in Zarr store:")
        print("=" * 50)
        for projection_id, skycells in available_data.items():
            print(f"\nProjection: {projection_id}")
            print(f"Skycells ({len(skycells)}):")
            for i, skycell in enumerate(sorted(skycells)):
                print(f"  {i + 1:3d}. {skycell}")
        return

    # Batch extraction
    if args.batch:
        output_dir = Path(args.output_dir)
        logging.info(f"Batch extracting combined image for skycell: {args.skycell}")
        success = batch_extract_combined_skycell(zarr_path, args.skycell, output_dir)

        if success:
            print(f"✓ Successfully extracted combined image for {args.skycell}")
        else:
            print(f"✗ Failed to extract combined image for {args.skycell}")

    else:
        # Single extraction
        output_path = Path(args.output) if args.output else None

        success = extract_combined_skycell_to_fits(zarr_path=zarr_path, skycell_name=args.skycell, output_path=output_path)

        if success:
            print(f"✓ Successfully extracted combined image for {args.skycell}")
        else:
            print(f"✗ Failed to extract combined image for {args.skycell}")


if __name__ == "__main__":
    main()
