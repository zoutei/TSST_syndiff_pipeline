"""
Zarr to FITS Extractor

This script extracts data from a Zarr store and converts it back to FITS files.
Useful for accessing individual skycells/bands from the unified Zarr store.

Features:
- Extract specific skycell/band combinations
- Reconstruct FITS headers from stored metadata
- Save as standard FITS files
- Batch extraction capabilities
- Command-line interface

Usage:
    python zarr_to_fits.py --skycell "skycell.2556.080" --band "r" --output "output.fits"
    python zarr_to_fits.py --skycell "skycell.2556.080" --band "r" --mask --output "output_mask.fits"
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from astropy.io import fits

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_projection_from_name(skycell_name: str) -> Optional[str]:
    """Extract projection ID from skycell name."""
    try:
        return skycell_name.split(".")[1]
    except IndexError:
        logging.warning(f"Could not parse projection from skycell name: {skycell_name}")
        return None


def extract_array_from_zarr(zarr_path: Path, skycell_name: str, band: str, is_mask: bool = False) -> tuple[Optional[np.ndarray], Optional[str]]:
    """
    Extract a specific array from the Zarr store.

    Returns:
        tuple: (data_array, header_string) or (None, None) if not found
    """
    projection_id = get_projection_from_name(skycell_name)
    if not projection_id:
        return None, None

    try:
        # Open zarr store
        root = zarr.open(zarr_path, mode="r")

        # Handle both old and new structures
        if "projections" in root:
            # Old structure: projections/[projection_id]/[skycell_name]
            if projection_id not in root["projections"]:
                logging.error(f"Projection {projection_id} not found in Zarr store")
                return None, None

            if skycell_name not in root["projections"][projection_id]:
                logging.error(f"Skycell {skycell_name} not found in projection {projection_id}")
                return None, None

            skycell_group = root["projections"][projection_id][skycell_name]
        else:
            # New structure: [projection_id]/[skycell_name]
            if projection_id not in root:
                logging.error(f"Projection {projection_id} not found in Zarr store")
                return None, None

            if skycell_name not in root[projection_id]:
                logging.error(f"Skycell {skycell_name} not found in projection {projection_id}")
                return None, None

            skycell_group = root[projection_id][skycell_name]

        # Determine array name
        array_name = f"{band}_mask" if is_mask else band

        if array_name not in skycell_group:
            logging.error(f"Array {array_name} not found in skycell {skycell_name}")
            return None, None

        # Extract data
        data_array = skycell_group[array_name][:]

        # Extract header if available - now look at array level first, then fallback to skycell level
        header_string = None
        if hasattr(skycell_group[array_name], "attrs") and "header" in skycell_group[array_name].attrs:
            # Preferred: header stored at array level (new format)
            header_string = skycell_group[array_name].attrs["header"]

        logging.info(f"Successfully extracted {array_name} from {skycell_name}")
        logging.info(f"Data shape: {data_array.shape}, dtype: {data_array.dtype}")

        return data_array, header_string

    except Exception as e:
        logging.error(f"Error extracting data: {e}")
        return None, None


def reconstruct_fits_header(header_string: Optional[str], band: str, is_mask: bool = False) -> fits.Header:
    """
    Reconstruct a FITS header from the stored header string.
    """
    if header_string:
        try:
            # Parse the header string back into a FITS header
            header = fits.Header.fromstring(header_string)
        except Exception as e:
            logging.warning(f"Could not parse stored header: {e}")
            header = fits.Header()
    else:
        header = fits.Header()

    return header


def save_as_fits(data: np.ndarray, header: fits.Header, output_path: Path) -> bool:
    """
    Save data as a FITS file.
    """
    try:
        # Create HDU
        hdu = fits.PrimaryHDU(data=data, header=header)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        hdu.writeto(output_path, overwrite=True)

        logging.info(f"Successfully saved FITS file: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving FITS file: {e}")
        return False


def extract_skycell_to_fits(zarr_path: Path, skycell_name: str, band: str, is_mask: bool = False, output_path: Optional[Path] = None) -> bool:
    """
    Complete pipeline to extract data from Zarr and save as FITS.
    """
    # Generate output filename if not provided
    if output_path is None:
        suffix = "_mask" if is_mask else ""
        output_path = Path(f"{skycell_name}_{band}{suffix}.fits")

    # Extract data from Zarr
    data, header_string = extract_array_from_zarr(zarr_path, skycell_name, band, is_mask)

    if data is None:
        return False

    # Reconstruct FITS header
    header = reconstruct_fits_header(header_string, band, is_mask)

    # Save as FITS
    return save_as_fits(data, header, output_path)


def batch_extract_skycell(zarr_path: Path, skycell_name: str, output_dir: Path) -> dict[str, bool]:
    """
    Extract all bands and masks for a given skycell.
    """
    output_dir = output_dir / skycell_name
    output_dir.mkdir(parents=True, exist_ok=True)

    bands = ["r", "i", "z", "y"]
    results = {}

    for band in bands:
        # Extract image
        img_path = output_dir / f"{skycell_name}_{band}.fits"
        img_success = extract_skycell_to_fits(zarr_path, skycell_name, band, is_mask=False, output_path=img_path)
        results[f"{band}_image"] = img_success

        # Extract mask
        mask_path = output_dir / f"{skycell_name}_{band}_mask.fits"
        mask_success = extract_skycell_to_fits(zarr_path, skycell_name, band, is_mask=True, output_path=mask_path)
        results[f"{band}_mask"] = mask_success

    return results


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
    parser = argparse.ArgumentParser(description="Extract FITS files from Zarr store")

    parser.add_argument("--zarr-path", type=str, default="data/ps1_skycells_zarr/ps1_skycells.zarr", help="Path to the Zarr store")

    parser.add_argument("--skycell", type=str, help="Skycell name (e.g., skycell.0306.067)")

    parser.add_argument("--band", type=str, choices=["r", "i", "z", "y"], help="Band to extract (r, i, z, y). If not specified, extracts all bands")

    parser.add_argument("--mask", action="store_true", help="Extract mask instead of image data")

    parser.add_argument("--output", type=str, help="Output FITS file path. If not specified, auto-generates filename")

    parser.add_argument("--batch", action="store_true", help="Extract all bands and masks for the specified skycell")

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

    # Check if skycell is required for non-list operations
    if not args.skycell:
        logging.error("Must specify --skycell for extraction operations")
        return

    # Batch extraction
    if args.batch:
        output_dir = Path(args.output_dir)
        logging.info(f"Batch extracting all bands for skycell: {args.skycell}")
        results = batch_extract_skycell(zarr_path, args.skycell, output_dir)

        print(f"\nBatch extraction results for {args.skycell}:")
        print("-" * 50)
        for array_name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {array_name}")

        success_count = sum(results.values())
        total_count = len(results)
        print(f"\nSummary: {success_count}/{total_count} arrays extracted successfully")

    else:
        # Single extraction
        if not args.band:
            logging.error("Must specify --band for single extraction or use --batch")
            return

        output_path = Path(args.output) if args.output else None

        success = extract_skycell_to_fits(zarr_path=zarr_path, skycell_name=args.skycell, band=args.band, is_mask=args.mask, output_path=output_path)

        if success:
            print(f"✓ Successfully extracted {args.band}{'_mask' if args.mask else ''} for {args.skycell}")
        else:
            print(f"✗ Failed to extract {args.band}{'_mask' if args.mask else ''} for {args.skycell}")


if __name__ == "__main__":
    main()
