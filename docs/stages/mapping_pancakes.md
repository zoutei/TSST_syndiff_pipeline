> **Package integration**: `syndiff-template` stage `mapping` · module `template/pancakes.py` · legacy script `pancakes_v2.py`  
> **Orchestration docs**: [template pipeline guide](../template_pipeline.md)

# PanCAKES v2.0: TESS-PS1 Pixel Mapping Pipeline

## Overview

This is a high-performance Python pipeline designed to process TESS (Transiting Exoplanet Survey Satellite) Full Frame Images (FFIs). Its primary function is to generate precise pixel-level mappings between the TESS FFIs and the PanSTARRS1 (PS1) SkyCell data.

The pipeline takes a TESS FITS file and a catalog of PS1 SkyCell World Coordinate System (WCS) data as input, and produces FITS and CSV files that map every TESS pixel to a corresponding PS1 SkyCell and vice versa. The code is optimized for speed using parallel processing, Numba-accelerated functions, and a custom-modified mocpy library with a Rust backend.

## Dependencies

The following Python libraries are required:

- numpy
- pandas
- astropy
- numba
- shapely
- tqdm
- mocpy (modified version)

### Important note on mocpy

This project requires a modified version of the `mocpy` library. The standard `mocpy` from pip does not include the MOC.filter_points_in_polygons function used for high-performance point-in-polygon filtering. The necessary version is available at <https://github.com/zoutei/mocpy_syndiff/> — follow that repository's instructions to build and install the modified package.

## How it works

The pipeline runs in a series of optimized steps to ensure accuracy and speed.

### 1. Initialization and data loading

- Load the primary TESS FFI from the provided FITS file (`tess_file`). Extract image data, WCS information, and essential header metadata (Sector, Camera, CCD).
- Load the PS1 SkyCell WCS database from a CSV file (`skycell_wcs_csv`) containing WCS parameters for many skycells.

### 2. Finding relevant skycells

To avoid processing every skycell, the pipeline first identifies those that spatially overlap the TESS image. This uses mocpy to create a Multi-Order Coverage (MOC) map from the corner coordinates of the TESS image (with a configurable buffer). The TESS MOC is used to filter skycell centers, producing a reduced list of relevant skycells.

### 3. Master TESS-to-Skycell mapping

Goal: create a 1D array where each element corresponds to a TESS pixel and its value is the index of the skycell it falls into.

- Coordinate calculation: compute RA/Dec for all TESS pixels.
- Polygon filtering (mocpy optimization): use the custom `MOC.filter_points_in_polygons` function to test which TESS pixels fall inside each skycell polygon. This function is backed by a Rust implementation for high-performance, parallelized point-in-polygon tests.
- Resolving overlaps: when skycell footprints overlap, a TESS pixel may be assigned to multiple skycells. The pipeline computes distances from the pixel to candidate skycell centers and assigns the pixel to the closest one (accelerated by the Numba-jitted `create_closest_center_array_numba`).

The final outputs from this step are a master mapping array (`tess_pix_skycell_mapping`) and a `selected_skycells` DataFrame containing only skycells that are used.

### 4. Parallel skycell processing

With the master map available, the pipeline builds an inverse map for each skycell in parallel (via `ProcessPoolExecutor`). For each skycell `process_single_skycell` does:

- Retrieve the list of TESS pixel indices assigned to that skycell.
- Compute each TESS pixel's on-sky footprint (small polygon).
- Project those footprints onto the PS1 skycell 2D pixel grid.
- Use the Numba-jitted `find_pixels_in_rectangles` to efficiently find PS1 pixels covered by each TESS pixel footprint.
- Produce a 2D array matching the PS1 skycell dimensions where each cell stores the index of the covering TESS pixel (or -1 for no coverage).
- Save this array as a compressed FITS file.

### 5. Padding calculation

Workers also check if TESS data reaches the edges of a skycell; if so, a "padding" skycell may be required for downstream tasks (e.g., difference imaging). The `analyze_single_skycell_padding` function determines which sides/corners need padding. For normal cases it picks adjacent skycells within the same projection group. For complex projection edges it computes the geometric area needed and uses shapely to find the best-fitting skycell from another projection group. Padding information from all workers is saved into the master skycell CSV.

## Output files

Output is organized under `output_path/sector_XXXX/camera_X/ccd_X/` and includes:

- Master Skycell List (CSV)
  - Filename: `tess_sXXXX_Y_Z_master_skycells_list.csv`
  - Description: CSV with WCS data, pixel counts, and padding information (e.g. `pad_skycell_top`, `pad_skycell_right`).

- Master TESS-to-Skycell Map (FITS)
  - Filename: `tess_sXXXX_Y_Z_master_pixels2skycells.fits.gz`
  - Description: compressed FITS with two main extensions:
    - Primary image (HDU 1): 2D image with dimensions matching the TESS FFI; each pixel stores an integer index corresponding to a skycell (table maps index → skycell name). A value of `-1` indicates no mapping.
    - Table (HDU 2): binary table mapping image indices to full skycell names (e.g. `index 0 -> skycell.2004.012`).

- Individual Skycell Maps (FITS)
  - Filename pattern: `tess_sXXXX_Y_Z_skycell.PROJ.CELL.fits.gz`
  - Description: compressed FITS per skycell. Image is a 2D array matching PS1 skycell dimensions; each pixel contains the 1D index of the covering TESS pixel or `-1` for none. FITS headers include combined TESS and PS1 WCS info.

## Usage

Run the script from the command line. The only required argument is the path to the TESS FITS file.

### Syntax

```bash
python pancakes_v2.py <tess_file_path> [OPTIONS]
```

### Example

```bash
python pancakes_v2.py ./data/tess_ffis/tess2019140104529-s0012-1-3-0144-s_ffic.fits \
        --output_path ./output --max_workers 16
```

### Arguments

- `tess_file` (required): Path to the TESS FITS file.
- `--skycell_wcs_csv`: Path to the skycell WCS CSV file. (Default: `./data/SkyCells/skycell_wcs.csv`)
- `--output_path`: Directory to save output files. (Default: `./data/skycell_pixel_mapping`)
- `--pad_distance`: Distance in pixels from an edge to check for padding. (Default: `480`)
- `--edge_exclusion`: Pixels to exclude from the very edge during padding checks. (Default: `10`)
- `--tess_buffer`: Buffer around the TESS image (in pixels) for finding relevant skycells. (Default: `150`)
- `--n_threads`: Number of threads for the mocpy filtering step. (Default: `8`)
- `--max_workers`: Maximum number of parallel processes for generating individual skycell maps. (Default: auto-detected)
- `--overwrite / --no-overwrite`: Flag to control overwriting existing files. (Default: `--overwrite`)
