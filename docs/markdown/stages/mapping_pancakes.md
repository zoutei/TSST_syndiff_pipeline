> **Package integration**: `syndiff` stage `mapping` · module `template_creation/processing/pancakes.py` (legacy script `pancakes_v2.py`)  
> **Orchestration docs**: [template pipeline guide](../template_pipeline.md)

# PanCAKES: TESS–PS1 Pixel Mapping Pipeline

## Overview

PanCAKES generates precise pixel-level mappings between a TESS Full Frame Image (FFI) and PanSTARRS1 (PS1) skycells. It consumes **only the WCS of the reference FFI** chosen by `wcs_grouping` (image pixel values are never used) plus a static catalog of PS1 skycell WCS parameters, and produces FITS/CSV artifacts that map every TESS pixel to a PS1 skycell and every PS1 pixel back to a TESS pixel. It is optimized with parallel processing, Numba-accelerated kernels, and a custom-modified `mocpy` with a Rust backend.

In the orchestrated pipeline, the `mapping` stage also **downloads the Gaia DR3 catalog** for the FFI footprint before running PanCAKES itself (see [Gaia catalog download](#gaia-catalog-download) below).

## Dependencies

numpy, pandas, astropy, numba, shapely, tqdm, and a **modified mocpy**: the standard pip `mocpy` lacks `MOC.filter_points_in_polygons`, which this pipeline requires for high-performance point-in-polygon filtering. Use <https://github.com/zoutei/mocpy_syndiff/> and follow its build instructions.

## How it works

Main driver: `process_tess_image_optimized()`.

### 1. Initialization and data loading

- `load_tess_image(tess_file)` opens the reference FFI, builds `WCS(hdul[1].header)` (falls back to HDU 0), and reads the data shape, sector, camera, and CCD. The CLI also accepts a `cluster_template_job.json` path and resolves it to `reference_ffi_path`.
- The PS1 skycell WCS database is read from CSV (`skycell_wcs_csv`, default `./data/SkyCells/skycell_wcs.csv`). All PS1-side geometry (centers, corners, grid dimensions) comes from this static catalog and is **independent of the TESS WCS**.
- `prepare_mapping_csv_workspace()` deletes any stale `.partial.csv` (and the final CSV when overwriting).

### 2. Finding relevant skycells (`find_relevant_skycells`)

The TESS footprint is an **8-point buffered polygon** in pixel space (`tess_buffer`, default 150 px) projected to sky with the TESS WCS (RA shifted by `moc_ra_shift_degrees` to avoid the 0/360 seam), converted to a MOC (`max_depth=21`). Skycell **centers** from the CSV are then filtered with `MOC.contains_lonlat`, reducing tens of thousands of skycells to the ~1000 overlapping the FFI.

### 3. Master TESS-to-skycell mapping (`process_tess_to_skycell_mapping`)

Goal: an array where each element is a TESS pixel (or sub-pixel when oversampling) and its value is the index of the skycell that covers it.

- Skycell sky polygons are built from PS1 WCS with **asymmetric edge buffers** (`calculate_radec_corners_shift`; `edge_buffer_large=410`, `edge_buffer_small=70`, `buffer=200` depending on the cell's `(x, y)` position in the 0–9 projection grid).
- All TESS pixel centers (~4.4M at native resolution) are projected to RA/Dec via `tess_wcs.all_pix2world`.
- The Rust-backed `MOC.filter_points_in_polygons` (parallel, `n_threads`) tests which TESS pixels fall inside each skycell polygon.
- **Overlap resolution**: when a pixel is claimed by several skycells, it is assigned to the skycell whose CRVAL center projects closest **in TESS pixel space** (not angular distance) — `create_closest_center_array_numba`.
- Skycell indices are densified to 0…N−1, and per-skycell inverse lists (`pixel_indices`) are built with `create_skycell_pixel_lists_numba`.

Outputs: `tess_pix_skycell_mapping` (master map) and the `selected_skycells` DataFrame. This step takes ~30 s at native resolution.

### 4. Parallel per-skycell registration

With the master map available, an inverse map is built for each skycell in parallel (`ProcessPoolExecutor`, one skycell per task). For each skycell, `process_skycell_pixel_mapping`:

- Retrieves the TESS pixel indices assigned to that skycell.
- Builds each TESS pixel's footprint as a 4-corner square of half-width `0.5 / oversampling_factor` TESS px, projected to sky (TESS WCS) and then into PS1 pixel coordinates (PS1 WCS).
- Uses Numba `find_pixels_in_rectangles` (bounding-box loop + ray-casting `point_in_polygon`) to find the PS1 pixels covered by each footprint.
- Fills a 2D array matching the PS1 skycell dimensions where each PS1 pixel stores the **1D ravelled index of the covering TESS (sub-)pixel** or `-1` (`populate_array_numba`).
- Writes the array as a FITS file, then gzips it (`os.system("gzip -f …")`).

This loop dominates the runtime (see [Timing](#timing-and-bottlenecks)).

### 5. Padding calculation

Workers also check whether TESS coverage reaches the edges of a skycell; if so, a "padding" skycell is recorded for downstream use. `analyze_single_skycell_padding` picks grid neighbors within the same projection (`skycell.{proj}.0{y±1}{x±1}`) for normal cases; at projection boundaries it uses shapely (`find_best_padding_skycell`) to find the best-fitting cell from another projection. Padding columns from all workers are merged into the master CSV, which is written as `.partial.csv` early and atomically renamed at the end (`finalize_master_skycells_csv`).

## Oversampling

`MappingStageParams.oversampling_factor` (default **1**) subdivides each TESS pixel into N×N sub-cells; coordinate count and master-map size scale by N². When N > 1, outputs gain an `_os{N}` filename suffix and live under an `oversampling_{N}/` subdirectory, and `downsample` must be run with the matching factor (it decodes linear indices with `os_width = t_x * N`).

## Output files

Layout: `{output_path}/[oversampling_{N}/]sector_{SSSS}/camera_{C}/ccd_{D}/`

- **Master skycell list (CSV)** — `tess_s{SSSS}_{C}_{D}_master_skycells_list[_osN].csv`  
  One row per selected skycell: `NAME`, full PS1 WCS columns (`NAXIS1/2`, `CRVAL*`, `CRPIX*`, `PC*`, `CDELT*`, …), sky corners, grid position (`projection`, `cell`, `y`, `x`), `pixel_indices` + `pixel_indices_num_pix`, and padding columns (`pad_skycell_top/right/…`, `special_padding_needed`, …). **This CSV is the only artifact the orchestrator's verify step checks.**

- **Master TESS-to-skycell map (FITS)** — `tess_s{SSSS}_{C}_{D}_master_pixels2skycells[_osN].fits.fz`  
  - HDU 0: empty primary.
  - HDU 1: 2D `int16` image, FFI-shaped (or oversampled); each pixel = skycell index 0…N−1 or `-1`. Header carries the full TESS FFI header plus `TESS_FFI`, `DATE-MOD`, `SOFTWARE`, `CREATOR`, optional `OVERSAMP`. Downsample reads the TESS WCS from this header.
  - HDU 2: binary table with columns `SKYCELL` (name) and `SKYCIND` (index matching the image values).

- **Per-skycell registration maps (FITS)** — `tess_s{SSSS}_{C}_{D}_skycell.{PROJ}.{CELL}[_osN].fits.fz`  
  - HDU 0: primary header (`SECTOR`, `CAMERA`, `CCD`, `SKYCELL`, …).
  - HDU 1: `ImageHDU` with `EXTNAME='TESS_PIXEL_MAP'`, PS1-skycell-shaped; each PS1 pixel = 1D ravelled TESS index or `-1`, stored as scaled `int32` (`BZERO=32768`). Header includes the full PS1 WCS.

## Gaia catalog download

Before PanCAKES, the orchestrator (`dispatch.py`, unless `skip_download_catalog: true`) calls `download_gaia_catalog_for_tess_file()`:

1. Same reference FFI WCS; padded 4-corner sky polygon (`pixel_padding=50`).
2. Async ADQL TAP query on `gaiadr3.gaia_source` with `phot_rp_mean_mag < limit`, with retries.
3. Output: `{catalog_dir}/gaia_catalog_s{SSSS}_{C}_{D}.csv` with columns `source_id`, `ra/dec` (+errors), `parallax` (+error), `pm`, `pmra/pmdec` (+errors), `phot_g/bp/rp_mean_mag`.
4. Backend: **flathub** (Flatiron) bbox prefetch + exact padded polygon filter by default; falls back to ESA TAP ADQL polygon on failure. Override with `gaia_backend="tap"` or `"flathub"` in `download_gaia_catalog()`.
5. Skipped if the file already exists (unless `force_download`).

The catalog is consumed by `ps1_process` (saturation handling / star removal) and the diff stage, not by the mapping math itself. Download takes ~2–3 minutes on a cold cache.

## Timing and bottlenecks

Reference run (sector 20, camera 3, CCD 1; 1067 skycells; native resolution):

| Phase | Duration |
|-------|----------|
| Gaia catalog download | ~2.5 min |
| Load + MOC prefilter + master map + save | ~28 s |
| Per-skycell registration loop | **~10.4 min** (0.5–8 s/skycell) |
| **Total mapping stage** | **~13.5 min** |

Within each skycell worker the cost is dominated by `find_pixels_in_rectangles` (∝ TESS pixels in the cell × PS1 bbox area per footprint), `all_pix2world` on the footprint corners, and the per-file `gzip` subprocess.

### WCS dependence and recomputation

Everything on the TESS side (footprint MOC, master map, per-skycell footprints, Gaia polygon, output headers) depends on the **reference FFI WCS**; all PS1-side geometry is static. If the reference WCS changes, the current code **always recomputes all artifacts** — there is no per-skycell skip/diff, checksums, or incremental mode; `prepare_mapping_csv_workspace` deletes the prior CSV at start, and verify only checks the final CSV.

Note, however, that **sub-pixel WCS drift across a sector does not require re-running mapping at all**: `downsample` absorbs small offsets by rolling PS1 data an integer number of PS1 pixels per skycell (`compute_ps1_shift_for_skycell`). Mapping only needs re-running when the reference frame itself changes materially (≳ a TESS pixel scale of geometry change).

## Usage

```bash
python -m syndiff_pipeline.template_creation.processing.pancakes <tess_file_or_cluster_job_json> [OPTIONS]
```

Key arguments:

- `tess_file` (required): TESS FITS path **or** `cluster_template_job.json` (resolved to `reference_ffi_path`).
- `--skycell_wcs_csv` (default `./data/SkyCells/skycell_wcs.csv`)
- `--output_path` (default `./data/skycell_pixel_mapping`)
- `--pad_distance` (480), `--edge_exclusion` (10), `--tess_buffer` (150)
- `--n_threads` (8, mocpy Rust filter), `--max_workers` (default `min(32, n_skycells, cpu_count+4)`)
- `--overwrite / --no-overwrite`

---

## Orchestrator integration

In the supervised pipeline, PanCAKES runs as the `mapping` stage via `syndiff template submit` (SCC-scoped; there is no combined `syndiff all` preset). It first resolves the SCC's reference FFI via the SCC-scoped chooser in `scc_reference_ffi.py` (median-CRVAL anchor + Earth/Moon-angle cuts), then runs PanCAKES (Condor pool `mapping`: 16 CPUs / 100 GB by default). Outputs land under `{data_root}/s{SSSS}/c{C}/k{K}/mapping/oversampling_{N}/…`; the scheduler verifies only `tess_s{sector}_{camera}_{ccd}_master_skycells_list[_os{N}].csv` before advancing to `ps1_download`. See the [template pipeline guide](../template_pipeline.md).
