> **Legacy workflow**: monolithic `pipeline.py` + individual scripts in the [`syndiff`](../../../syndiff/) repo.  
> **Recommended for production**: [`syndiff-template`](../template_pipeline.md) in this package (multi-target scheduler, WCS grouping, Condor).  
> **Algorithm deep-dives**: [stages index](README.md)

> **Note (integrated package)**: In `syndiff-pipeline`, PS1 download writes one shared store `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr` (not per-SCC Zarr paths). Convolved output is `{data_root}/convolved_results/sector_{SSSS}_camera_{C}_ccd_{K}.zarr`. See the [template pipeline guide](../template_pipeline.md) for current layouts.

# Syndiff Pipeline (standalone scripts)

Complete end-to-end pipeline for creating TESS Full Frame Image Template with PanSTARRS1 (PS1) data. The pipeline automatically extracts sector, camera, and CCD information from the TESS FITS file and runs the full processing workflow.

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Command Line Options](#command-line-options)
- [Output Structure](#output-structure)
- [Examples](#examples)
- [Running Individual Pipeline Components](#running-individual-pipeline-components)
- [Notes](#notes)

## Requirements

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 128GB+ recommended
- **Storage**: 1TB+ free space for PS1 data downloads
- **CPU**: Multi-core processor

### Python Dependencies

Install the required packages using pip:

```bash
pip install numpy astropy pandas zarr sep scipy dask dask-image numba shapely tqdm filelock
```

#### Core Dependencies

- **numpy**: Array operations and mathematical functions
- **astropy**: FITS file handling, WCS operations, and astronomical coordinate transformations
- **pandas**: Data manipulation and CSV file handling
- **zarr**: Efficient array storage and retrieval for large datasets
- **sep**: Source Extractor Python for astronomical object detection and background removal
- **scipy**: Signal processing and convolution operations
- **dask**: Parallel computing and distributed processing
- **dask-image**: Image processing with Dask arrays
- **numba**: JIT compilation for performance-critical functions
- **shapely**: Geometric operations for sky cell processing
- **tqdm**: Progress bars for long-running operations
- **filelock**: Thread-safe file operations

### Special Dependencies

#### Custom MOCPy Installation

The pipeline uses a custom modified version of MOCPy with enhanced performance for astronomical region processing:

```bash
./install_mocpy.sh
```

#### Alternative Installation Method

1. **Install Rust** (required for building):

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rust-lang.org | sh
   source ~/.cargo/env
   ```

2. **Install maturin** (Python-Rust build tool):

   ```bash
   pip install maturin
   ```

3. **Build and install custom MOCPy**:

   ```bash
   cd mocpy_syndiff
   maturin develop --release
   cd ..
   ```

## Quick Start

Simply provide the TESS FITS file - the pipeline will automatically determine sector, camera, and CCD, and use the default skycell catalog:

```bash
# Basic usage with default skycell catalog (data/SkyCells/skycell_wcs.csv)
python pipeline.py /path/to/tess-ffi.fits

# With custom skycell catalog
python pipeline.py /path/to/tess-ffi.fits /path/to/custom-catalog.csv

# With verbose output
python pipeline.py /path/to/tess-ffi.fits -v

# Override data directory and processing parameters
python pipeline.py /path/to/tess-ffi.fits \
    --data-root /custom/data/path \
    --cores 16 \
    --jobs 100 \
    --overwrite
```

## Pipeline Steps

The pipeline automatically runs four main steps:

1. **Pancakes v2** - Generate TESS↔PS1 mapping files and skycell list
2. **Download** - Download PS1 skycells and store in efficient Zarr format  
3. **Process PS1** - Combine PS1 bands using modern sliding window pipeline
4. **Downsample** - Multi-offset downsample to TESS grid

## Command Line Options

### Required Arguments

- `tess_fits` - Path to TESS FFI FITS file (sector/camera/CCD auto-extracted)

### Optional Arguments

- `skycell_wcs_csv` - Path to skycell WCS catalog CSV for Pancakes (default: `data/SkyCells/skycell_wcs.csv`)
- `--data-root` - Root directory for data storage (default: `data`)
- `--cores` - Number of CPU cores to use (default: 8)
- `--jobs` - Number of parallel download jobs (default: 60)
- `--overwrite` - Overwrite existing files
- `--verbose` - Increase verbosity (`-v` for INFO, `-vv` for DEBUG)
- `--multi-offset-array` - Comma-separated dx,dy pairs for downsampling (default: `0.0,0.0`)
- `--ignore-mask-bits` - Comma-separated mask bits to ignore (default: `12`)

## Output Structure

The pipeline creates the following directory structure under `data_root`:

```bash
data/
├── mapping_output/
│   └── sector_XXXX/
│       └── camera_X/
│           └── ccd_X/
│               ├── tess_sXXXX_X_X_master_skycells_list.csv
│               └── TESS_sXXXX_X_X_skycell.*.fits.gz
├── ps1_skycells_zarr/
│   └── sector_XXXX_camera_X_ccd_X.zarr/
└── convolved_results/
    └── sector_XXXX/
        └── camera_X/
            └── ccd_X/
                ├── convolved_images.zarr
                └── cell_metadata.json
```

## Examples

### Process a single TESS image with default settings

```bash
python pipeline.py tess2020123456-s0020-1-3-0000-s_ffic.fits
```

### High-performance processing

```bash
python pipeline.py tess2020123456-s0020-1-3-0000-s_ffic.fits \
    --cores 32 \
    --jobs 200 \
    --verbose
```

### Multi-offset downsampling

```bash
python pipeline.py tess2020123456-s0020-1-3-0000-s_ffic.fits \
    --multi-offset-array "0.0,0.0,0.5,0.0,0.0,0.5,0.5,0.5"
```

### Using custom skycell catalog

```bash
python pipeline.py tess2020123456-s0020-1-3-0000-s_ffic.fits /path/to/custom-catalog.csv
```

## Running Individual Pipeline Components

While the main pipeline runs all steps automatically, you can also run individual components for debugging, development, or partial processing:

### 1. Pancakes v2 - TESS↔PS1 Mapping

Generate TESS to PS1 skycell mappings and create the master skycells list.

```bash
python pancakes_v2.py /path/to/tess-ffi.fits
```

**Key Options:**

- `--skycell_wcs_csv` - Path to skycell WCS catalog (default: `./data/SkyCells/skycell_wcs.csv`)
- `--output_path` - Output directory for mapping files (default: `./data/skycell_pixel_mapping`)
- `--max_workers` - Number of parallel workers for processing
- `--overwrite` - Overwrite existing output files

📖 **More Info:** See [PanCAKES mapping deep-dive](mapping_pancakes.md) for detailed documentation.

### 2. Download PS1 Data

Download PS1 skycell data and store in efficient Zarr format.

```bash
python download_and_store_zarr.py 20 3 3
```

**Required Arguments:**

- `sector` - TESS sector number
- `camera` - TESS camera number (1-4)
- `ccd` - TESS CCD number (1-4)

**Key Options:**

- `--num-workers` - Number of parallel download workers (default: 32)
- `--zarr-output-dir` - Directory for Zarr output (default: `data/ps1_skycells_zarr`)
- `--use-local-files` - Use locally saved FITS files instead of downloading
- `--overwrite` - Overwrite existing Zarr arrays

### 3. Process PS1 - Modern Sliding Window Pipeline

Combine PS1 bands and convolve using the modern sliding window approach.

```bash
python process_ps1.py 20 3 3
```

**Required Arguments:**

- `sector` - TESS sector number
- `camera` - TESS camera number (1-4)
- `ccd` - TESS CCD number (1-4)

**Key Options:**

- `--data-root` - Root data directory (default: `data`)
- `--limit` - Limit number of projections for testing
- `--psf-sigma` - PSF sigma for convolution (default: 40.0)

📖 **More Info:** See [PS1 process technical reference](ps1_process_technical.md) for comprehensive documentation of the sliding window architecture.

### 4. Multi-Offset Downsampling

Generate multiple downsampled images with different pixel offsets.

```bash
python multi_offset_downsampling.py 20 3 3
```

**Optional Arguments:**

- `sector` - TESS sector number (default: 20)
- `camera` - Camera number (default: 3)
- `ccd` - CCD number (default: 3)

**Key Options:**

- `--data-root` - Root data directory
- `--convolved-dir` - Convolved results directory override
- `--output-base` - Base output directory override

### Component Dependencies

The pipeline components have the following dependencies:

```mermaid
Pancakes v2 → Download PS1 → Process PS1 → Multi-Offset Downsampling
```

Each step uses outputs from the previous step, so they must be run in order when using individual components.

## Notes

- **Automatic Metadata Extraction**: Sector, camera, and CCD are automatically extracted from the TESS FITS filename and header
- **Default Skycell Catalog**: The pipeline uses `data/SkyCells/skycell_wcs.csv` by default - ensure this file exists or provide a custom catalog
- **Resumable Processing**: Each step checks for existing outputs and can resume if interrupted
- **Memory Efficient**: Uses Zarr format for efficient storage and streaming processing
- **Parallel Processing**: Optimized for multi-core systems with configurable parallelism
- **Error Handling**: Comprehensive error checking and informative logging
