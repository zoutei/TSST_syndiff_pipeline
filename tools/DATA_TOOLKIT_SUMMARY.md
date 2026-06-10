# Zarr Data Processing Toolkit - Summary

This document summarizes the comprehensive Zarr data processing toolkit that has been created, including inventory, extraction, and download capabilities with both command-line and programmatic interfaces.

## Overview

The toolkit consists of three main scripts that work together to provide a complete solution for managing PS1 skycell data in Zarr format:

1. **zarr_inventory.py** - Analyze and cleanup Zarr stores
2. **zarr_to_fits.py** - Extract data from Zarr back to FITS
3. **download_and_store_zarr.py** - Download and store PS1 data (enhanced with local file support and callable function)

## Scripts Details

### 1. zarr_inventory.py

**Purpose**: Analyze Zarr store health and perform cleanup operations

**Command-line Usage**:

```bash
python zarr_inventory.py                           # Basic analysis
python zarr_inventory.py --cleanup-temp --dry-run  # Preview temp cleanup
python zarr_inventory.py --cleanup-all             # Clean everything
```

**Key Features**:

- Detects corruption, incomplete data, temporary arrays
- Supports dry-run mode for safe preview
- Provides detailed statistics and cleanup options
- Can remove specific types of problematic data

### 2. zarr_to_fits.py

**Purpose**: Extract data from Zarr store and convert back to FITS files

**Command-line Usage**:

```bash
# Extract single skycell
python zarr_to_fits.py --skycell skycell.012345.006789.stk.v3.skycell.unconv --band g

# Batch extract all skycells for a band
python zarr_to_fits.py --band r --batch

# List available data
python zarr_to_fits.py --list
```

**Key Features**:

- Single and batch extraction modes
- Reconstructs proper FITS headers
- Handles both old and new Zarr structure formats
- Supports listing available data

### 3. download_and_store_zarr.py (Enhanced)

**Purpose**: Download PS1 data with local file fallback and callable function interface

**Command-line Usage**:

```bash
# Basic download
python download_and_store_zarr.py

# Use local files when available
python download_and_store_zarr.py --use-local-files

# Custom configuration
python download_and_store_zarr.py --sector 21 --camera 2 --ccd 1 --num-workers 16 --use-local-files
```

**Programmatic Usage**:

```python
from download_and_store_zarr import download_and_store_ps1_data

# Basic usage
result = download_and_store_ps1_data()

# Custom parameters
result = download_and_store_ps1_data(
    sector=21, 
    camera=2, 
    ccd=1, 
    use_local_files=True,
    local_data_path="data/my_ps1_data"
)

# Check result
if result['status'] == 'completed':
    print(f"Success! Zarr saved to: {result['zarr_path']}")
```

**Key Features**:

- **Local file support**: Checks for local FITS files before downloading
- **Callable function**: Can be imported and used from other Python scripts
- **Return values**: Provides structured results with status and metadata
- **Error handling**: Graceful handling of missing files and errors
- **Resumable**: Can continue interrupted downloads
- **Thread-safe**: Uses file locking for concurrent operations

## New Capabilities Added

### Local File Support

- The download script now checks for locally available FITS files before attempting downloads
- Configurable local data path (default: `data/ps1_skycells`)
- Graceful fallback to download mode if local files are not found

### Callable Function Interface

- `download_and_store_ps1_data()` function can be imported and called from other scripts
- Returns structured results with processing status and metadata
- Enables programmatic batch processing and integration workflows

### Enhanced Error Handling

- Structured return values with status codes ('completed', 'error', 'interrupted')
- Detailed error messages for troubleshooting
- Graceful handling of missing configuration files

## Example Workflows

### 1. Data Management Workflow

```bash
# 1. Analyze current Zarr store
python zarr_inventory.py

# 2. Clean up any issues found
python zarr_inventory.py --cleanup-corrupt --cleanup-incomplete

# 3. Extract specific data for analysis
python zarr_to_fits.py --skycell TARGET_SKYCELL --band g --output extracted_data/
```

### 2. Batch Processing Workflow

```python
# Python script for batch processing multiple configurations
from download_and_store_zarr import download_and_store_ps1_data

configurations = [
    {'sector': 20, 'camera': 1, 'ccd': 1},
    {'sector': 20, 'camera': 1, 'ccd': 2},
    {'sector': 21, 'camera': 1, 'ccd': 1},
]

for config in configurations:
    result = download_and_store_ps1_data(**config)
    print(f"Config {config}: {result['status']}")
```

### 3. Local Data Processing

```bash
# Use local files when available, fall back to download
python download_and_store_zarr.py --use-local-files --local-data-path "data/ps1_skycells"
```

## Benefits

1. **Comprehensive Coverage**: Complete toolkit for Zarr data lifecycle management
2. **Flexible Interfaces**: Both command-line and programmatic access
3. **Performance Optimized**: Local file support reduces unnecessary downloads
4. **Robust Error Handling**: Graceful handling of various failure modes
5. **Developer Friendly**: Easy integration into larger workflows and pipelines
6. **Data Integrity**: Built-in validation and cleanup capabilities

## Files Created/Modified

- `zarr_inventory.py` - New inventory and cleanup script
- `zarr_to_fits.py` - New extraction script  
- `download_and_store_zarr.py` - Enhanced with local files and callable function
- `example_usage.py` - Example demonstrating programmatic usage

This toolkit provides a complete solution for PS1 data management with Zarr, from initial download through analysis and extraction, with both command-line convenience and programmatic flexibility.
