"""
Zarr Inventory Script

This script analyzes a Zarr store created by download_and_store_zarr.py
and creates a comprehensive inventory of all stored data, including corruption detection.

Features:
- Detects complete vs incomplete skycells
- Identifies corrupted arrays
- Finds orphaned temporary arrays
- Exports detailed CSV reports
- Provides summary statistics
- Optional cleanup of corrupted/incomplete data

Usage:
    python zarr_inventory.py                    # Basic inventory
    python zarr_inventory.py --cleanup-temp     # Remove orphaned temporary arrays
    python zarr_inventory.py --cleanup-corrupt  # Remove corrupted arrays
    python zarr_inventory.py --cleanup-incomplete  # Remove incomplete skycells
    python zarr_inventory.py --cleanup-all      # Remove all problematic data
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_skycell_info_from_name(skycell_name: str) -> dict[str, Any]:
    """Extract metadata from skycell name."""
    try:
        parts = skycell_name.split(".")
        if len(parts) >= 3:
            return {
                "projection_id": parts[1],
                "cell_id": parts[2],
                "sector": None,  # Would need additional parsing if encoded in name
                "camera": None,
                "ccd": None,
            }
        else:
            return {"projection_id": None, "cell_id": None, "sector": None, "camera": None, "ccd": None}
    except Exception:
        return {"projection_id": None, "cell_id": None, "sector": None, "camera": None, "ccd": None}


def check_array_health(array: zarr.Array) -> dict[str, Any]:
    """Check if an array is healthy and accessible."""
    health_info = {"exists": True, "readable": False, "shape": None, "dtype": None, "size_bytes": 0, "has_valid_data": False, "corruption_notes": []}

    try:
        # Basic properties
        health_info["shape"] = array.shape
        health_info["dtype"] = str(array.dtype)
        health_info["size_bytes"] = array.nbytes

        # Test readability with a small sample
        if len(array.shape) == 2 and array.shape[0] > 0 and array.shape[1] > 0:
            # Read a small corner to test accessibility
            sample = array[0 : min(10, array.shape[0]), 0 : min(10, array.shape[1])]
            health_info["readable"] = True

            # Check for valid data (not all NaN or all zeros)
            if not np.all(np.isnan(sample)) and not np.all(sample == 0):
                health_info["has_valid_data"] = True
            elif np.all(np.isnan(sample)):
                health_info["corruption_notes"].append("All NaN values")
            elif np.all(sample == 0):
                health_info["corruption_notes"].append("All zero values")

        else:
            health_info["corruption_notes"].append(f"Invalid shape: {array.shape}")

    except Exception as e:
        health_info["readable"] = False
        health_info["corruption_notes"].append(f"Read error: {str(e)}")

    return health_info


def analyze_skycell_group(group: zarr.Group, skycell_name: str, projection_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Analyze a single skycell group."""
    expected_arrays = ["r", "r_mask", "r_wt", "i", "i_mask", "i_wt", "z", "z_mask", "z_wt", "y", "y_mask", "y_wt"]

    # Get basic info
    skycell_info = get_skycell_info_from_name(skycell_name)
    skycell_info.update({"skycell_name": skycell_name, "projection_id": projection_id})

    # Find all arrays in the group
    actual_arrays = list(group.array_keys())
    temp_arrays = [name for name in actual_arrays if "_temp_" in name]
    regular_arrays = [name for name in actual_arrays if "_temp_" not in name]

    # Check completeness
    missing_arrays = set(expected_arrays) - set(regular_arrays)
    extra_arrays = set(regular_arrays) - set(expected_arrays)

    skycell_info.update(
        {"is_complete": len(missing_arrays) == 0, "arrays_present": regular_arrays, "arrays_missing": list(missing_arrays), "arrays_extra": list(extra_arrays), "temp_arrays": temp_arrays, "arrays_present_count": len(regular_arrays), "arrays_missing_count": len(missing_arrays), "temp_arrays_count": len(temp_arrays)}
    )

    # Check each array's health
    array_details = []
    corrupted_arrays = []
    total_size_bytes = 0

    for array_name in regular_arrays:
        try:
            array = group[array_name]
            health = check_array_health(array)

            array_detail = {"skycell_name": skycell_name, "projection_id": projection_id, "array_name": array_name, **health}
            array_details.append(array_detail)

            if not health["readable"] or health["corruption_notes"]:
                corrupted_arrays.append(array_name)

            total_size_bytes += health["size_bytes"]

        except Exception as e:
            array_detail = {"skycell_name": skycell_name, "projection_id": projection_id, "array_name": array_name, "exists": True, "readable": False, "shape": None, "dtype": None, "size_bytes": 0, "has_valid_data": False, "corruption_notes": [f"Access error: {str(e)}"]}
            array_details.append(array_detail)
            corrupted_arrays.append(array_name)

    # Check for orphaned temp arrays
    for temp_array_name in temp_arrays:
        try:
            array = group[temp_array_name]
            health = check_array_health(array)

            array_detail = {"skycell_name": skycell_name, "projection_id": projection_id, "array_name": temp_array_name, **health, "corruption_notes": health["corruption_notes"] + ["Orphaned temporary array"]}
            array_details.append(array_detail)
            total_size_bytes += health["size_bytes"]

        except Exception as e:
            array_detail = {"skycell_name": skycell_name, "projection_id": projection_id, "array_name": temp_array_name, "exists": True, "readable": False, "shape": None, "dtype": None, "size_bytes": 0, "has_valid_data": False, "corruption_notes": [f"Orphaned temp array access error: {str(e)}"]}
            array_details.append(array_detail)

    # Check for header
    has_header = "header" in group.attrs if hasattr(group, "attrs") else False
    header_size = len(str(group.attrs.get("header", ""))) if has_header else 0

    # Determine corruption status
    is_corrupted = len(corrupted_arrays) > 0 or len(temp_arrays) > 0
    corruption_types = []

    if missing_arrays:
        corruption_types.append(f"Missing arrays: {list(missing_arrays)}")
    if corrupted_arrays:
        corruption_types.append(f"Corrupted arrays: {corrupted_arrays}")
    if temp_arrays:
        corruption_types.append(f"Orphaned temp arrays: {temp_arrays}")
    if extra_arrays:
        corruption_types.append(f"Extra arrays: {list(extra_arrays)}")

    skycell_info.update(
        {
            "is_corrupted": is_corrupted,
            "corruption_type": "; ".join(corruption_types) if corruption_types else None,
            "arrays_corrupted": corrupted_arrays,
            "arrays_corrupted_count": len(corrupted_arrays),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "has_header": has_header,
            "header_size": header_size,
        }
    )

    return skycell_info, array_details


def inventory_zarr_store(zarr_path: Path) -> tuple[list[dict], list[dict]]:
    """Create complete inventory of zarr store."""
    logging.info(f"Starting inventory of Zarr store: {zarr_path}")

    if not zarr_path.exists():
        logging.error(f"Zarr store not found: {zarr_path}")
        return [], []

    # Open zarr store
    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception as e:
        logging.error(f"Failed to open Zarr store: {e}")
        return [], []

    skycell_summaries = []
    array_details = []

    # Iterate through all projection groups
    for top_level_key in root.group_keys():
        if top_level_key == "projections":
            # Handle old structure: projections/[projection_id]/[skycell_name]
            projections_group = root[top_level_key]
            for projection_id in projections_group.group_keys():
                logging.info(f"Processing projection: {projection_id}")
                projection_group = projections_group[projection_id]

                # Iterate through all skycell groups in this projection
                skycell_count = 0
                for skycell_name in projection_group.group_keys():
                    try:
                        skycell_group = projection_group[skycell_name]
                        skycell_summary, skycell_arrays = analyze_skycell_group(skycell_group, skycell_name, projection_id)

                        skycell_summaries.append(skycell_summary)
                        array_details.extend(skycell_arrays)
                        skycell_count += 1

                        if skycell_count % 100 == 0:
                            logging.info(f"Processed {skycell_count} skycells in {projection_id}")

                    except Exception as e:
                        logging.error(f"Error processing skycell {skycell_name}: {e}")
                        # Add error entry
                        error_summary = {
                            "skycell_name": skycell_name,
                            "projection_id": projection_id,
                            "is_complete": False,
                            "is_corrupted": True,
                            "corruption_type": f"Processing error: {str(e)}",
                            "arrays_present_count": 0,
                            "arrays_missing_count": 12,
                            "arrays_corrupted_count": 0,
                            "temp_arrays_count": 0,
                            "total_size_mb": 0,
                        }
                        skycell_summaries.append(error_summary)

                logging.info(f"Completed projection {projection_id}: {skycell_count} skycells")
        else:
            # Handle new structure: [projection_id]/[skycell_name]
            projection_id = top_level_key
            logging.info(f"Processing projection: {projection_id}")
            projection_group = root[projection_id]

            # Iterate through all skycell groups in this projection
            skycell_count = 0
            for skycell_name in projection_group.group_keys():
                try:
                    skycell_group = projection_group[skycell_name]
                    skycell_summary, skycell_arrays = analyze_skycell_group(skycell_group, skycell_name, projection_id)

                    skycell_summaries.append(skycell_summary)
                    array_details.extend(skycell_arrays)
                    skycell_count += 1

                    if skycell_count % 100 == 0:
                        logging.info(f"Processed {skycell_count} skycells in {projection_id}")

                except Exception as e:
                    logging.error(f"Error processing skycell {skycell_name}: {e}")
                    # Add error entry
                    error_summary = {"skycell_name": skycell_name, "projection_id": projection_id, "is_complete": False, "is_corrupted": True, "corruption_type": f"Processing error: {str(e)}", "arrays_present_count": 0, "arrays_missing_count": 12, "arrays_corrupted_count": 0, "temp_arrays_count": 0, "total_size_mb": 0}
                    skycell_summaries.append(error_summary)

            logging.info(f"Completed projection {projection_id}: {skycell_count} skycells")

    logging.info(f"Inventory complete: {len(skycell_summaries)} skycells, {len(array_details)} arrays")
    return skycell_summaries, array_details


def print_summary_stats(skycell_summaries: list[dict], array_details: list[dict]):
    """Print summary statistics."""
    if not skycell_summaries:
        logging.info("No data to summarize")
        return

    total_skycells = len(skycell_summaries)
    complete_skycells = sum(1 for s in skycell_summaries if s.get("is_complete", False))
    corrupted_skycells = sum(1 for s in skycell_summaries if s.get("is_corrupted", False))

    total_size_mb = sum(s.get("total_size_mb", 0) for s in skycell_summaries)
    total_arrays = len(array_details)
    readable_arrays = sum(1 for a in array_details if a.get("readable", False))

    print("\n" + "=" * 60)
    print("ZARR STORE INVENTORY SUMMARY")
    print("=" * 60)
    print(f"Total Skycells: {total_skycells}")
    print(f"Complete Skycells: {complete_skycells} ({complete_skycells / total_skycells * 100:.1f}%)")
    print(f"Corrupted Skycells: {corrupted_skycells} ({corrupted_skycells / total_skycells * 100:.1f}%)")
    print(f"Incomplete Skycells: {total_skycells - complete_skycells - corrupted_skycells}")
    print(f"\nTotal Arrays: {total_arrays}")
    if total_arrays > 0:
        print(f"Readable Arrays: {readable_arrays} ({readable_arrays / total_arrays * 100:.1f}%)")
        print(f"Corrupted Arrays: {total_arrays - readable_arrays}")
    else:
        print("Readable Arrays: 0 (N/A)")
        print("Corrupted Arrays: 0")
    print(f"\nTotal Storage: {total_size_mb:.1f} MB ({total_size_mb / 1024:.1f} GB)")
    print("=" * 60)


def cleanup_temp_arrays(zarr_path: Path, skycell_summaries: list[dict], array_details: list[dict]) -> dict[str, int]:
    """Remove orphaned temporary arrays from the Zarr store."""
    logging.info("Starting cleanup of orphaned temporary arrays...")

    temp_arrays_removed = 0
    storage_freed_mb = 0

    try:
        root = zarr.open(zarr_path, mode="a")  # Open in append mode for modifications

        # Find all temp arrays
        temp_arrays = [a for a in array_details if "_temp_" in a.get("array_name", "")]

        for temp_array in temp_arrays:
            try:
                skycell_name = temp_array["skycell_name"]
                projection_id = temp_array["projection_id"]
                array_name = temp_array["array_name"]

                # Navigate to the array location
                if "projections" in root:
                    # Old structure
                    array_group = root["projections"][projection_id][skycell_name]
                else:
                    # New structure
                    array_group = root[projection_id][skycell_name]

                if array_name in array_group:
                    # Get size before deletion
                    size_mb = temp_array.get("size_bytes", 0) / (1024 * 1024)

                    # Delete the temporary array
                    del array_group[array_name]
                    temp_arrays_removed += 1
                    storage_freed_mb += size_mb

                    logging.info(f"Removed temp array: {projection_id}/{skycell_name}/{array_name}")

            except Exception as e:
                logging.warning(f"Failed to remove temp array {temp_array.get('array_name', 'unknown')}: {e}")

    except Exception as e:
        logging.error(f"Error during temp array cleanup: {e}")

    results = {"temp_arrays_removed": temp_arrays_removed, "storage_freed_mb": storage_freed_mb}

    logging.info(f"Cleanup complete: Removed {temp_arrays_removed} temp arrays, freed {storage_freed_mb:.1f} MB")
    return results


def cleanup_corrupted_arrays(zarr_path: Path, skycell_summaries: list[dict], array_details: list[dict]) -> dict[str, int]:
    """Remove corrupted arrays from the Zarr store."""
    logging.info("Starting cleanup of corrupted arrays...")

    corrupted_arrays_removed = 0
    storage_freed_mb = 0

    try:
        root = zarr.open(zarr_path, mode="a")

        # Find all corrupted arrays (not readable or have corruption notes)
        corrupted_arrays = [a for a in array_details if not a.get("readable", True) or a.get("corruption_notes", [])]

        for corrupted_array in corrupted_arrays:
            try:
                skycell_name = corrupted_array["skycell_name"]
                projection_id = corrupted_array["projection_id"]
                array_name = corrupted_array["array_name"]

                # Skip temp arrays (handled separately)
                if "_temp_" in array_name:
                    continue

                # Navigate to the array location
                if "projections" in root:
                    array_group = root["projections"][projection_id][skycell_name]
                else:
                    array_group = root[projection_id][skycell_name]

                if array_name in array_group:
                    # Get size before deletion
                    size_mb = corrupted_array.get("size_bytes", 0) / (1024 * 1024)

                    # Delete the corrupted array
                    del array_group[array_name]
                    corrupted_arrays_removed += 1
                    storage_freed_mb += size_mb

                    logging.info(f"Removed corrupted array: {projection_id}/{skycell_name}/{array_name}")

            except Exception as e:
                logging.warning(f"Failed to remove corrupted array {corrupted_array.get('array_name', 'unknown')}: {e}")

    except Exception as e:
        logging.error(f"Error during corrupted array cleanup: {e}")

    results = {"corrupted_arrays_removed": corrupted_arrays_removed, "storage_freed_mb": storage_freed_mb}

    logging.info(f"Cleanup complete: Removed {corrupted_arrays_removed} corrupted arrays, freed {storage_freed_mb:.1f} MB")
    return results


def cleanup_incomplete_skycells(zarr_path: Path, skycell_summaries: list[dict], array_details: list[dict]) -> dict[str, int]:
    """Remove incomplete skycells from the Zarr store."""
    logging.info("Starting cleanup of incomplete skycells...")

    skycells_removed = 0
    storage_freed_mb = 0

    try:
        root = zarr.open(zarr_path, mode="a")

        # Find all incomplete skycells
        incomplete_skycells = [s for s in skycell_summaries if not s.get("is_complete", False) and not s.get("is_corrupted", False)]

        for incomplete_skycell in incomplete_skycells:
            try:
                skycell_name = incomplete_skycell["skycell_name"]
                projection_id = incomplete_skycell["projection_id"]

                # Navigate to the skycell location
                if "projections" in root:
                    projection_group = root["projections"][projection_id]
                else:
                    projection_group = root[projection_id]

                if skycell_name in projection_group:
                    # Get size before deletion
                    size_mb = incomplete_skycell.get("total_size_mb", 0)

                    # Delete the entire skycell group
                    del projection_group[skycell_name]
                    skycells_removed += 1
                    storage_freed_mb += size_mb

                    logging.info(f"Removed incomplete skycell: {projection_id}/{skycell_name}")

            except Exception as e:
                logging.warning(f"Failed to remove incomplete skycell {incomplete_skycell.get('skycell_name', 'unknown')}: {e}")

    except Exception as e:
        logging.error(f"Error during incomplete skycell cleanup: {e}")

    results = {"skycells_removed": skycells_removed, "storage_freed_mb": storage_freed_mb}

    logging.info(f"Cleanup complete: Removed {skycells_removed} incomplete skycells, freed {storage_freed_mb:.1f} MB")
    return results


def perform_cleanup(zarr_path: Path, skycell_summaries: list[dict], array_details: list[dict], cleanup_temp: bool = False, cleanup_corrupt: bool = False, cleanup_incomplete: bool = False) -> dict[str, Any]:
    """Perform the requested cleanup operations."""
    if not any([cleanup_temp, cleanup_corrupt, cleanup_incomplete]):
        return {}

    print("\n" + "=" * 60)
    print("CLEANUP OPERATIONS")
    print("=" * 60)

    total_results = {}

    if cleanup_temp:
        temp_results = cleanup_temp_arrays(zarr_path, skycell_summaries, array_details)
        total_results.update(temp_results)

    if cleanup_corrupt:
        corrupt_results = cleanup_corrupted_arrays(zarr_path, skycell_summaries, array_details)
        total_results.update(corrupt_results)

    if cleanup_incomplete:
        incomplete_results = cleanup_incomplete_skycells(zarr_path, skycell_summaries, array_details)
        total_results.update(incomplete_results)

    # Calculate totals
    total_freed_mb = sum(v for k, v in total_results.items() if k.endswith("_freed_mb"))

    print("\nCLEANUP SUMMARY:")
    if cleanup_temp:
        print(f"- Temporary arrays removed: {total_results.get('temp_arrays_removed', 0)}")
    if cleanup_corrupt:
        print(f"- Corrupted arrays removed: {total_results.get('corrupted_arrays_removed', 0)}")
    if cleanup_incomplete:
        print(f"- Incomplete skycells removed: {total_results.get('skycells_removed', 0)}")
    print(f"- Total storage freed: {total_freed_mb:.1f} MB ({total_freed_mb / 1024:.1f} GB)")
    print("=" * 60)

    return total_results


def main():
    """Main function to run the inventory with optional cleanup."""
    parser = argparse.ArgumentParser(description="Analyze and optionally clean up Zarr store")

    parser.add_argument("--zarr-path", type=str, default="data/ps1_skycells_zarr/ps1_skycells.zarr", help="Path to the Zarr store")

    parser.add_argument("--output-dir", type=str, default="data/zarr_inventory", help="Directory to save inventory reports")

    parser.add_argument("--cleanup-temp", action="store_true", help="Remove orphaned temporary arrays")

    parser.add_argument("--cleanup-corrupt", action="store_true", help="Remove corrupted arrays")

    parser.add_argument("--cleanup-incomplete", action="store_true", help="Remove incomplete skycells")

    parser.add_argument("--cleanup-all", action="store_true", help="Remove all problematic data (temp, corrupt, incomplete)")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without actually deleting")

    args = parser.parse_args()

    # Set cleanup flags
    if args.cleanup_all:
        cleanup_temp = cleanup_corrupt = cleanup_incomplete = True
    else:
        cleanup_temp = args.cleanup_temp
        cleanup_corrupt = args.cleanup_corrupt
        cleanup_incomplete = args.cleanup_incomplete

    # Configuration
    zarr_path = Path(args.zarr_path)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inventory
    skycell_summaries, array_details = inventory_zarr_store(zarr_path)

    if not skycell_summaries:
        logging.error("No data found. Exiting.")
        return

    # Convert to DataFrames
    skycells_df = pd.DataFrame(skycell_summaries)
    arrays_df = pd.DataFrame(array_details)

    # Generate timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save CSV files
    skycells_csv = output_dir / f"skycell_inventory_{timestamp}.csv"
    arrays_csv = output_dir / f"array_inventory_{timestamp}.csv"

    skycells_df.to_csv(skycells_csv, index=False)
    arrays_df.to_csv(arrays_csv, index=False)

    logging.info(f"Saved skycell inventory to: {skycells_csv}")
    logging.info(f"Saved array inventory to: {arrays_csv}")

    # Print summary statistics
    print_summary_stats(skycell_summaries, array_details)

    # Perform cleanup if requested
    any_cleanup = cleanup_temp or cleanup_corrupt or cleanup_incomplete
    if any_cleanup:
        if args.dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN MODE - NO ACTUAL CLEANUP PERFORMED")
            print("=" * 60)

            # Show what would be cleaned
            temp_count = len([a for a in array_details if "_temp_" in a.get("array_name", "")])
            corrupt_count = len([a for a in array_details if not a.get("readable", True) or a.get("corruption_notes", [])])
            incomplete_count = len([s for s in skycell_summaries if not s.get("is_complete", False) and not s.get("is_corrupted", False)])

            print("Would remove:")
            if cleanup_temp:
                print(f"- {temp_count} temporary arrays")
            if cleanup_corrupt:
                print(f"- {corrupt_count} corrupted arrays")
            if cleanup_incomplete:
                print(f"- {incomplete_count} incomplete skycells")
            print("\nRe-run without --dry-run to perform actual cleanup.")
        else:
            # Perform actual cleanup
            cleanup_results = perform_cleanup(zarr_path, skycell_summaries, array_details, cleanup_temp, cleanup_corrupt, cleanup_incomplete)

            # Re-run inventory after cleanup to see the results
            if cleanup_results:
                logging.info("Re-running inventory after cleanup...")
                post_cleanup_summaries, post_cleanup_arrays = inventory_zarr_store(zarr_path)
                print("\nPOST-CLEANUP INVENTORY:")
                print_summary_stats(post_cleanup_summaries, post_cleanup_arrays)

    # Save latest versions without timestamp
    skycells_df.to_csv(output_dir / "skycell_inventory_latest.csv", index=False)
    arrays_df.to_csv(output_dir / "array_inventory_latest.csv", index=False)

    logging.info("Inventory completed successfully!")


if __name__ == "__main__":
    main()
