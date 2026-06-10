"""
Zarr to FITS Extractor for convolved/combined results

This script extracts arrays named like `skycell.2627.091_data` or
`skycell.2627.091_mask` from Zarr stores created by the convolved/combined
pipeline. The Zarr files live in a directory such as
`data/convolved_results_sliding/` and are named
`sector_0020_camera_3_ccd_3.zarr`.

Behavior changes from the original script:
- `--zarr-path` is a directory containing .zarr files (default
    `data/convolved_results_sliding`). The script constructs the .zarr filename
    from `--sector`, `--camera`, and `--ccd`.
- No `--which` option. Arrays expected are `{skycell}_data` and `{skycell}_mask`.
- Batch extraction feature removed; single extraction only.

CLI inputs: sector, camera, ccd, skycell, mask, output, list, out-dir, zarr-path
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from astropy.io import fits

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def default_zarr_dir() -> Path:
    """Default directory that contains the .zarr files."""
    return Path("data/convolved_results")


def candidate_name_for_skycell(skycell: str, is_mask: bool) -> str:
    """Return the expected array name for the skycell.

    The dataset layout uses `{skycell}_data` for images and
    `{skycell}_mask` for masks.
    """
    return f"{skycell}_mask" if is_mask else f"{skycell}_data"


def extract_array_from_zarr_flat(zarr_path: Path, skycell: str, is_mask: bool = False) -> tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """Open flat-layout Zarr and extract `{skycell}_data` or `{skycell}_mask`.

    Returns (data, header_string, chosen_name) or (None, None, None) if not found.
    """
    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception as e:
        logging.error(f"Could not open Zarr store {zarr_path}: {e}")
        return None, None, None

    name = candidate_name_for_skycell(skycell, is_mask)
    if name not in root:
        logging.error(f"Array {name} not found in Zarr store {zarr_path}")
        return None, None, None

    try:
        arr = root[name][:]
    except Exception as e:
        logging.error(f"Failed to read array {name}: {e}")
        return None, None, None

    header_string = None
    if hasattr(root[name], "attrs") and "header" in root[name].attrs:
        header_string = root[name].attrs["header"]

    logging.info(f"Found array '{name}' in Zarr store: shape={arr.shape}, dtype={arr.dtype}")
    return arr, header_string, name


def reconstruct_fits_header(header_string: Optional[str]) -> fits.Header:
    if not header_string:
        return fits.Header()

    # header_string may be bytes; convert to str
    try:
        if isinstance(header_string, (bytes, bytearray)):
            header_string = header_string.decode("utf-8", errors="ignore")
    except Exception:
        pass

    try:
        header = fits.Header.fromstring(header_string)
    except Exception as e:
        logging.warning(f"Could not parse stored header: {e}")
        header = fits.Header()

    return header


def save_as_fits(data: np.ndarray, header: fits.Header, output_path: Path) -> bool:
    try:
        hdu = fits.PrimaryHDU(data=data, header=header)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hdu.writeto(output_path, overwrite=True)
        logging.info(f"Wrote FITS: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to write FITS {output_path}: {e}")
        return False


def list_available_data(zarr_path: Path) -> dict[str, list[str]]:
    """Scan root arrays and return mapping of skycell -> available variants"""
    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception as e:
        logging.error(f"Could not open Zarr store {zarr_path}: {e}")
        return {}

    pattern = re.compile(r"(skycell\.\d+\.\d+)_([a-zA-Z0-9_]+)")
    mapping: dict[str, list[str]] = {}

    for key in root.array_keys():
        m = pattern.match(key)
        if m:
            sc = m.group(1)
            variant = m.group(2)
            mapping.setdefault(sc, []).append(variant)

    # sort variants for nicer output
    for sc in mapping:
        mapping[sc] = sorted(list(set(mapping[sc])))

    return mapping


def extract_to_fits(zarr_path: Path, skycell: str, is_mask: bool, output_path: Optional[Path]) -> bool:
    data, header_string, chosen = extract_array_from_zarr_flat(zarr_path, skycell, is_mask=is_mask)
    if data is None:
        return False

    header = reconstruct_fits_header(header_string)

    if output_path is None:
        output_path = Path(f"{skycell}_{chosen}.fits")

    return save_as_fits(data, header, output_path)


# batch_extract removed by request - single extraction only


def main():
    parser = argparse.ArgumentParser(description="Extract FITS from flat-layout convolved/combined Zarr stores")

    # positional form: python zarr_to_convolved_fits.py 20 3 3 2556.080
    parser.add_argument("sector", nargs="?", type=int, default=20, help="Sector number (default: 20)")
    parser.add_argument("camera", nargs="?", type=int, default=3, help="Camera id (default: 3)")
    parser.add_argument("ccd", nargs="?", type=int, default=3, help="CCD id (default: 3)")
    parser.add_argument("skycell", nargs="?", type=str, default=None, help="Skycell (e.g., 2556.080 or skycell.2556.080)")

    parser.add_argument("--zarr-path", type=str, default=None, help="Directory containing .zarr files (overrides sector/camera/ccd)")
    parser.add_argument("--mask", action="store_true", help="Extract mask instead of image data")
    parser.add_argument("--output", type=str, help="Output FITS file path (for single extraction)")
    parser.add_argument("--list", action="store_true", help="List available skycells and variants in the Zarr store")
    parser.add_argument("--output-dir", type=str, default="extracted_fits", help="Output directory for extraction results")

    args = parser.parse_args()

    # Build the .zarr path from the directory and sector/camera/ccd
    zarr_dir = Path(args.zarr_path) if args.zarr_path else default_zarr_dir()
    # construct filename like sector_0020_camera_3_ccd_3.zarr
    zarr_filename = f"sector_{args.sector:04d}_camera_{args.camera}_ccd_{args.ccd}.zarr"
    zpath = zarr_dir / zarr_filename

    if not zpath.exists():
        logging.error(f"Zarr store not found: {zpath}")
        return

    if args.list:
        available = list_available_data(zpath)
        print("\nAvailable skycells in Zarr store:")
        print("=" * 60)
        for sc, variants in sorted(available.items()):
            print(f"{sc}: {', '.join(variants)}")
        return

    if not args.skycell and not args.list:
        logging.error("Must specify skycell (positional) for extraction operations, or use --list")
        return

    # Batch feature removed; proceed to single extraction

    # Single extraction
    if not args.list:
        # normalize skycell name
        sc = args.skycell
        if not sc.startswith("skycell."):
            sc = f"skycell.{sc}"

        out_path = Path(args.output) if args.output else None
        success = extract_to_fits(zpath, sc, args.mask, out_path)

        kind = "mask" if args.mask else "data"
        if success:
            print(f"✓ Successfully extracted {kind} for {sc}")
        else:
            print(f"✗ Failed to extract {kind} for {sc}")


if __name__ == "__main__":
    main()
