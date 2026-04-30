# SynDiff Pipeline

TESS Full Frame Image (FFI) difference imaging pipeline for transient detection and forced photometry.

---

## Table of Contents

- [Overview](#overview)
- [Current Code Status](#current-code-status)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
  - [Config-Driven Pipeline](#config-driven-pipeline)
- [Configuration Reference](#configuration-reference)
  - [Global Keys](#global-keys)
  - [Pipeline Stage Kinds](#pipeline-stage-kinds)
- [Config-Driven Pipeline Stages in Detail](#config-driven-pipeline-stages-in-detail)
  - [wcs_grouping](#wcs_grouping)
  - [shared_mask](#shared_mask)
  - [hotpants](#hotpants)
  - [epsf](#epsf)
  - [sat_template](#sat_template)
  - [subtract](#subtract)
  - [background_rough](#background_rough)
  - [background_adaptive](#background_adaptive)
  - [background_estimate](#background_estimate)
  - [diff_final](#diff_final)
  - [forced_photometry](#forced_photometry)
- [Workspace and Output Layout](#workspace-and-output-layout)
- [CLI Usage](#cli-usage)
- [Example Recipes](#example-recipes)
  - [Recipe A — Simple PRF Photometry](#recipe-a--simple-prf-photometry)
  - [Recipe B — ePSF + Saturated Star Removal + Background Chain](#recipe-b--epsf--saturated-star-removal--background-chain)
  - [Recipe C — Two-Pass Hotpants](#recipe-c--two-pass-hotpants)
- [External Prerequisites (Before Running SynDiff)](#external-prerequisites-before-running-syndiff)
- [Module Reference](#module-reference)
- [Known Limitations and Work in Progress](#known-limitations-and-work-in-progress)

---

## Overview

SynDiff performs difference imaging on TESS FFIs against PS1 (Pan-STARRS1) templates to produce clean difference images and extract forced PSF photometry light curves at a target coordinate. The pipeline handles:

1. **FFI Download** — bulk download of calibrated TESS FFIs from MAST (tesscurl or astroquery).
2. **WCS Drift Analysis** — extract per-frame WCS, compute pixel drift of the science target, and group frames by template offset.
3. **Shared Masking** — build a bitmask from Gaia catalog (bright stars, saturation crosses, TESS straps).
4. **Image Differencing** — Hotpants-based (pyhotpants) kernel-matching subtraction of FFI crops against PS1 templates.
5. **Empirical PSF Fitting** — tiled ePSF fitting using TGLC across difference images.
6. **Saturated Star Templates** — model images of removed saturated stars from the ePSF.
7. **Background Estimation** — rough background via inpainting/smoothing, then adaptive temporal median filtering using TESSVectors.
8. **Final Difference Images** — Fourier deconvolution/reconvolution of the saturated star model with the ePSF.
9. **Forced Photometry** — PSF-fitted flux extraction at the target position (ePSF or official TESS PRF).

---

## Current Code Status

> **This project has not been released.** All modules are under active development.

### Fully Implemented and Functional


| Module                   | Status    | Notes                                                                                                                                                                                                                                                                                                                                                |
| ------------------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `download.py`            | **Ready** | Two download backends (tesscurl + astroquery). CLI entry point works. Handles resume/skip of existing files.                                                                                                                                                                                                                                         |
| `config.py`              | **Ready** | `SynDiffConfig` dataclass with YAML I/O, CLI argument parsing, path resolution relative to config file, `template_dir` auto-discovery, required `pipeline` list.                                                                                                               |
| `wcs_grouping.py`        | **Ready** | WCS extraction, pixel drift computation, template group assignment, crop bounds (explicit or quadrant mode), `cluster_template_job.json` generation, Gaia catalog WCS projection (`ensure_gaia_crop_xy`), diagnostic drift plot.                                                                                                                     |
| `masking.py`             | **Ready** | Shared bitmask (catalog stars, bright-star crosses, TESS straps). Reference star selection for Hotpants substamps with isolation and separation filters. Strap QE correction (vendored from TESSreduce).                                                                                                                                             |
| `hotpants_runner.py`     | **Ready** | Per-frame Hotpants differencing with joblib **loky** (process pool) when `n_jobs` > 1 — pyhotpants holds the GIL in C, so threads would not parallelize. Flat `diff_rN/` layout and workspace layout (`ws/<label>/`). Split outputs: diffs, convolved, optional polynomial bkg. Template discovery from `syndiff_template_*.fits` filenames or `group_*/ps1_template.fits`. Second-pass Hotpants with `sci_bkg_stack` subtraction. |
| `epsf_fitting.py`        | **Ready** | Tiled ePSF fitting via TGLC `get_psf`/`fit_psf`. Handles `CustomSource` setup, tile grid, median mask column correction, stack bundle save/load. Per-frame diagnostic logging.                                                                                                                                                                       |
| `temporal_smooth.py`     | **Ready** | ePSF temporal smoothing (uniform filter or no-filter passthrough). Group ePSF median computation. Adaptive background smoothing via `AdaptiveBackground` (TESSVectors Earth/Moon angles). Final background combination (round 1 + round 2).                                                                                                          |
| `sat_template.py`        | **Ready** | Native-resolution and high-resolution saturated star template construction. Sub-pixel ePSF stamp placement. Block-sum downsampling. Per-group save/load.                                                                                                                                                                                             |
| `background.py`          | **Ready** | Per-frame rough background (`estimate_frame_background`: shared-masked diff + `Smooth_bkg`, then add Hotpants polynomial bkg). **Per-frame joblib loky** when `n_jobs` > 1 (via `background_loop(..., n_jobs=cfg.n_jobs)`). Vendored TESSreduce `Smooth_bkg` and `inpaint_biharmonic`.                                                                                                                                                                                |
| `adaptive_background.py` | **Ready** | `adaptive_medfilt_3d` adaptive median on the background cube; uses ``cfg.n_jobs`` for **coarse-scale / window-size** joblib tasks (not one task per FFI). Earth/Moon angles, gap-aware segments, block-reduce. Vendored from TESSreduce dev branch.                                                                                                                   |
| `final_diff.py`          | **Ready** | Fourier deconvolution/reconvolution (Wiener-style) of high-res sat template from Hotpants Gaussian kernel to ePSF. Per-frame final difference image production.                                                                                                                                                                                      |
| `photometry.py`          | **Ready** | Forced PSF photometry with `create_psf` (vendored TESSreduce). `EpsfLocator` for ePSF. Per-epoch joblib **loky** when `n_jobs` > 1. Brightest-frame position fit. Light curve CSV + diagnostic plot.                                                                            |
| `frame_manifest.py`      | **Ready** | Per-FFI manifest CSV management. Hotpants/ePSF status columns (round IDs and labeled workspaces). Ordered diff path lookup for photometry.                                                                                                                                                                                                    |
| `paths.py`               | **Ready** | Workspace directory convention (`{output_dir}/ws/{label}/`). Manifest path resolution.                                                                                                                                                                                                                                                               |
| `pipeline_context.py`    | **Ready** | `PipelineInvocationContext` dataclass (resolved paths for config-driven execution).                                                                                                                                                                                                                                                                  |
| `pipeline_validate.py`   | **Ready** | Config-driven pipeline validation: checks stage `kind` membership, required keys per stage, input label dependency graph (every input label must be produced by an earlier stage or listed in `pipeline_external_workspace_labels`).                                                                                                                 |
| `pipeline_execute.py`    | **Ready** | Config-driven orchestrator (`run_config_pipeline`). Executes all 9 stage kinds in YAML order. Handles state bootstrap when `wcs_grouping` is omitted (resume from existing manifest).                                                                                                                                                                |
| `run_pipeline.py`        | **Ready** | Main CLI entry point; requires non-empty `pipeline:` and delegates to `pipeline_execute`.                                                                                                                                                                                                           |
| `plot_pipeline.py`       | **Ready** | Background removal animated GIF (round-1 smooth bkg cube).                                                                                                                                                                                                                                                                                           |


### External Dependencies (Not Part of This Package)


| Dependency          | Required By                                 | Status                                                                                                                           |
| ------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `pyhotpants`        | `hotpants_runner.py`                        | **Must be on `sys.path`**. The runner auto-discovers it in sibling directories. Not pip-installable; clone separately.           |
| `tglc` (TGLC)       | `epsf_fitting.py`                           | **Must be importable**. Uses `tglc.effective_psf.get_psf` / `fit_psf` and `tglc.ffi.Source`. Clone TGLC and add to `PYTHONPATH`. |
| `PRF` package       | `photometry.py` (only when `psf_type: prf`) | **Optional**. `pip install PRF` for official TESS PRF. Not needed if using `psf_type: epsf`.                                     |
| `TSST_Syndiff_Core` | Template creation                           | **Separate pipeline** for PS1 template generation. Run before SynDiff to produce per-group PS1 template FITS and Gaia catalog.   |


### Not Implemented (Out of Scope for This Package)

- **Source extraction and astrometry** (external TSST stages before WCS grouping) — SynDiff reads WCS from FFI FITS headers directly. If mission WCS is insufficient, astrometry must be improved externally before running `wcs_grouping`.
- **PS1 template creation** (TSST_Syndiff_Core pancakes through multi-offset downsampling) — handled outside this package.

---

## Dependencies

Python packages (all available via conda-forge or pip):

```
numpy
pandas
scipy
astropy
matplotlib
joblib
pyyaml
scikit-image      # skimage.restoration.inpaint for background estimation
```

Optional:

```
tqdm              # download progress bars
astroquery        # alternative FFI download via MAST (--via-mast)
PRF               # official TESS PRF for psf_type: prf
```

External (clone/install separately):

```
pyhotpants        # Hotpants image subtraction (must be on sys.path)
tglc              # TGLC ePSF fitting (must be importable)
```

---

## Installation

```bash
mamba activate syndiff

# Clone and ensure pyhotpants is accessible
# (auto-discovered if placed as a sibling directory)

# Run from the SynDiff repo root:
python -m syndiff_pipeline.run_pipeline --config your_config.yaml
```

---

## Pipeline Architecture

### Config-Driven Pipeline

The YAML must define a non-empty `pipeline:` list. Stages run in list order. Each stage declares its `kind`, `inputs` (referencing labels produced by earlier stages), and `output` (a label that names the workspace directory under `{output_dir}/ws/`).

Global artifacts (`cluster_template_job.json`, `shared_mask.fits`, `ref_stars.csv`, `tile_centers.json`) remain directly under `output_dir`.

The frame manifest (`syndiff_ffi_frames.csv`) tracks per-FFI WCS drift, template group assignment, and per-stage success/failure status.

**Validation** (`--validate-only`):

- Every stage `kind` must be one of the 9 recognized kinds.
- Every `inputs` label must be produced by an earlier stage (or listed in `pipeline_external_workspace_labels`).
- Stage-specific required keys are checked (e.g., `hotpants` must have `output.diffs` and `output.convolved`; `forced_photometry` must have exactly one of `psf: prf` or `inputs.epsf`).

**Checkpointing:** each invocation re-runs all listed stages. To validate or run against existing workspaces, use `pipeline_external_workspace_labels` and trim the `pipeline:` list accordingly.

---

## Configuration Reference

### Global Keys


| Key                                         | Type    | Default         | Description                                                                                                                                   |
| ------------------------------------------- | ------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `ffi_dir`                                   | str     | `""`            | Root `tess_ffi` directory. FFIs read from `{ffi_dir}/s{sector:04d}/cam{camera}_ccd{ccd}/`.                                                    |
| `output_dir`                                | str     | `""`            | Root directory for all pipeline outputs.                                                                                                      |
| `manifest`                                  | str     | `""`            | Optional absolute path to frame manifest CSV. Default: `syndiff_ffi_frames.csv` under `output_dir`.                                           |
| `gaia_catalog`                              | str     | `""`            | Path to crop-local Gaia CSV with `x`, `y` and photometry columns. Required for masking and ePSF.                                              |
| `template_dir`                              | str     | `""`            | Directory containing `group_<id>/ps1_template.fits` or `syndiff_template_*.fits`. Auto-discovered.                                            |
| `template_paths`                            | dict    | `{}`            | Explicit mapping `{group_id: path}`. Filled from `template_dir` if empty.                                                                     |
| `removed_stars_csv`                         | str     | `""`            | Optional CSV of removed saturated stars (used for sat templates when Gaia catalog lacks crop-local x/y).                                      |
| `median_mask_path`                          | str     | `""`            | Path to TGLC `median_mask.fits` (column correction for ePSF fitting).                                                                         |
| `straps_csv`                                | str     | `""`            | CSV listing TESS detector strap columns.                                                                                                      |
| `target_ra`                                 | float   | `null`          | RA (deg, J2000) of the science target. Required for photometry and `max_ffis`.                                                                |
| `target_dec`                                | float   | `null`          | Dec (deg, J2000) of the science target.                                                                                                       |
| `sector`                                    | int     | `20`            | TESS sector number.                                                                                                                           |
| `camera`                                    | int     | `3`             | TESS camera number (1–4).                                                                                                                     |
| `ccd`                                       | int     | `3`             | TESS CCD number (1–4).                                                                                                                        |
| `crop_quadrant`                             | str     | `"tr"`          | `tl` / `tr` / `bl` / `br` / `full`. Used when no explicit `x_min`/`x_max`/`y_min`/`y_max` are set.                                            |
| `x_min`, `x_max`, `y_min`, `y_max`          | int     | `null`          | Explicit crop bounds in FFI pixels. If any is set, explicit mode; unset edges use usable area.                                                |
| `x_left_dead`                               | int     | `44`            | Dead columns on the left edge of the FFI.                                                                                                     |
| `x_right_dead`                              | int     | `44`            | Dead columns on the right edge.                                                                                                               |
| `y_edge_strip`                              | int     | `30`            | Dead rows on the top of the FFI.                                                                                                              |
| `offset_threshold`                          | float   | `0.01`          | Maximum pixel offset (TESS pixels) before a new template group is created.                                                                    |
| `ref_ffi_path`                              | str     | `null`          | Optional explicit reference FFI path. If null, derived from the first valid-WCS frame.                                                        |
| `sci_fwhm`                                  | float   | `1.0`           | Science image FWHM in native pixels. Drives Hotpants kernel/substamp widths.                                                                  |
| `hp_`*                                      | various |                 | Hotpants parameters (kernel order, background order, stamp grid, Gaussian params, thresholds). See `config.py` for all `hp_` prefixed fields. |
| `gaia_mag_bright`                           | float   | `13.0`          | Mask all Gaia stars brighter than this magnitude.                                                                                             |
| `ref_mag_min` / `ref_mag_max`               | float   | `13.5` / `14.5` | Magnitude range for Hotpants reference stars.                                                                                                 |
| `ref_isolation_mag`                         | float   | `13.5`          | Reject reference star if any star brighter than this is within `ref_isolation_px`.                                                            |
| `ref_isolation_px`                          | int     | `8`             | Isolation radius for reference star selection.                                                                                                |
| `ref_separation_px`                         | int     | `10`            | Minimum pairwise separation between selected reference stars.                                                                                 |
| `strapsize`                                 | int     | `6`             | Width of the strap mask kernel.                                                                                                               |
| `tile_nx` / `tile_ny`                       | int     | `4` / `4`       | Number of tiles along x/y for ePSF fitting.                                                                                                   |
| `epsf_oversample`                           | int     | `2`             | ePSF oversampling factor.                                                                                                                     |
| `psf_size`                                  | int     | `11`            | Half-size of ePSF stamp in native pixels. `over_size = 2 * psf_size + 1`.                                                                     |
| `high_res_os`                               | int     | `9`             | Oversampling for the Fourier deconv/reconv step in `final_diff.py`.                                                                           |
| `temporal_smooth_window`                    | int     | `11`            | Window size (frames) for ePSF temporal smoothing.                                                                                             |
| `epsf_temporal_smooth`                      | bool    | `true`          | Whether to apply temporal filtering to ePSF stacks.                                                                                           |
| `bkg_vector_path`                           | str     | `null`          | Local directory with TESSVectors CSV. If null, downloaded from HEASARC.                                                                       |
| `bkg_adaptive_w_min` / `bkg_adaptive_w_max` | int     | `3` / `51`      | Temporal window bounds for adaptive background median filter.                                                                                 |
| `bkg_adaptive_block_size`                   | int     | `5`             | Spatial block size for adaptive background smoother.                                                                                          |
| `bkg_r1_recombine_hotpants`                 | bool    | `false`         | If true, recombine Hotpants bkg into diff before `Smooth_bkg` in round-1 background.                                                          |
| `psf_type`                                  | str     | `"epsf"`        | `epsf` (fitted ePSF) or `prf` (official TESS PRF via `PRF` package).                                                                          |
| `phot_cutout_size`                          | int     | `15`            | Side length of the photometry cutout stamp.                                                                                                   |
| `phot_bkg_poly_order`                       | int     | `3`             | Polynomial order for local background surface in photometry.                                                                                  |
| `phot_snap`                                 | str     | `"brightest"`   | Position-fit strategy: `brightest` / `ref` / `fixed`.                                                                                         |
| `pipeline_plots`                            | bool    | `false`         | Write diagnostic figures (WCS drift, background GIF, light curve plot). Light-curve titles append the `forced_photometry` stage `output` label. |
| `pipeline_plot_dpi`                         | int     | `150`           | DPI for diagnostic PNGs.                                                                                                                      |
| `pipeline_external_workspace_labels`        | list    | `null`          | Pre-existing workspace labels to include in the dependency graph during validation.                                                           |
| `n_jobs`                                    | int     | `8`             | Joblib **loky** workers for **Hotpants**, **forced photometry** (per epoch), **rough background** (`background_loop`, per frame), plus **adaptive** smoother **sub-tasks** (not one worker per FFI in `adaptive_medfilt_3d`). ePSF over frames stays serial. Lower if RAM is tight. |
| `max_ffis`                                  | int     | `null`          | Cap on number of FFIs to process (requires `target_ra`/`target_dec`; skips bad WCS).                                                          |
| `pipeline`                                  | list    | `[]`            | **Required (non-empty to run):** ordered list of stage dicts (`kind` + fields).                                                                |


### Pipeline Stage Kinds

The recognized stage kinds for the `pipeline:` list include:

| Kind                   | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `wcs_grouping`         | WCS extraction, drift grouping, crop bounds, reference FFI   |
| `shared_mask`          | Bitmask (Gaia catalog + straps) and Hotpants reference stars |
| `hotpants`             | Image differencing (FFI vs PS1 template)                     |
| `epsf`                 | Tiled empirical PSF fitting on difference images             |
| `sat_template`         | Saturated star model construction from ePSF                  |
| `subtract`             | Pixel-level subtraction (science − template workspace)       |
| `background_rough`     | Per-frame rough background stack (`rough_bkg_r*.npz` + `.npy`, optional per-frame FITS) |
| `background_adaptive`  | Temporal adaptive smoothing from a rough stack → `bkg_smooth.npz` + `.npy` (optional per-frame FITS) |
| `background_estimate`  | Convenience: rough + adaptive in one stage (`mode: rough_then_adaptive`) |
| `diff_final`           | Final diff via Fourier deconv/reconv of sat template         |
| `forced_photometry`    | PSF-fitted flux extraction at target coordinates             |


---

## Config-Driven Pipeline Stages in Detail

### wcs_grouping

No pipeline-specific fields. Reads from global config only.

**Produces**: `syndiff_ffi_frames.csv` (frame manifest), `cluster_template_job.json`, crop bounds, reference FFI path. Optional: `wcs_drift_template_debug.png` if `pipeline_plots: true`.

**Global config used**: `ffi_dir`, `sector`, `camera`, `ccd`, `target_ra`, `target_dec`, `offset_threshold`, `crop_quadrant` or explicit crop bounds, `max_ffis`, `ref_ffi_path`.

### shared_mask

No pipeline-specific fields. Requires `wcs_grouping` to have run first.

**Produces**: `shared_mask.fits`, `ref_stars.csv`, `gaia_catalog_pipeline.csv` under `output_dir`.

**Global config used**: `gaia_catalog`, `straps_csv`, `gaia_mag_bright`, `strapsize`, `ref_mag_min`, `ref_mag_max`, `ref_isolation_mag`, `ref_isolation_px`, `ref_separation_px`.

### hotpants

```yaml
- kind: hotpants
  science: ffi              # required, only "ffi" is implemented
  inputs:                   # optional, for second-pass Hotpants
    bkg: <label>            # prior Hotpants output.bkg workspace
    convolved: <label>      # prior Hotpants output.convolved (logged but not yet consumed)
  output:
    diffs: <label>          # required — workspace for difference images
    convolved: <label>      # required — workspace for convolved products
    bkg: <label>            # optional — workspace for polynomial background FITS
```

**Key behavior**:

- `output.bkg` is optional. If omitted, the Hotpants polynomial background is used internally for the diff but not saved to disk.
- `inputs.bkg` + `inputs.convolved` enable a second-pass Hotpants (FFI − saved_bkg).
- `inputs.convolved` is accepted but logged as a warning ("ignored in this version").

### epsf

```yaml
- kind: epsf
  inputs:
    diffs: <label>          # required — diff workspace to fit ePSFs on
  temporal_smooth: true     # optional, overrides global epsf_temporal_smooth for this stage
  output: <label>           # required — workspace for ePSF products
```

**Produces**: `epsf_stack_r1.npz`, `epsf_r1_smooth.npz`, `group_epsf/group_epsf_*.npy` under `ws/<output_label>/`.

### sat_template

```yaml
- kind: sat_template
  inputs:
    diffs: <label>          # required — diff workspace (used for alignment context)
    epsf: <label>           # required — ePSF workspace to load group ePSFs from
  output: <label>           # required — workspace for sat template FITS
```

**Produces**: `sat_tmpl_native_r1/group_*.fits` and `sat_tmpl_hr_r1/group_*.fits` under `ws/<output_label>/`.

### subtract

```yaml
- kind: subtract
  inputs:
    science: <label>        # required — workspace with per-frame FITS to subtract from
    template: <label>       # required — workspace with template to subtract (or bkg_smooth.npy)
  output: <label>           # required — workspace for result FITS
```

**Behavior**: For each frame, computes `science − template`. If the template workspace contains `bkg_smooth.npz` (array key `stack`) or `bkg_smooth.npy`, uses the stack (indexed by frame order); otherwise looks for per-frame FITS.

### background_rough

```yaml
- kind: background_rough
  inputs:
    diffs: <label>          # required — diff workspace (Hotpants)
    bkg: <label>            # required — Hotpants polynomial bkg workspace
  output: <label>           # required — workspace for `rough_bkg_rN.npz` / `.npy`
  round_id: 1               # optional, default 1 — basename `rough_bkg_r{round_id}`
  write_per_frame_fits: true  # optional, default true — write `{stem}_rough_bkg.fits` per epoch (set false to skip)
```

**Produces**: `rough_bkg_r{round_id}.npz` (single array `stack`) and `rough_bkg_r{round_id}.npy` under `ws/<output_label>/`, plus optional `{stem}_rough_bkg.fits` when `write_per_frame_fits` is true. Drops Hotpants FITS arrays from memory after this stage when run alone (next stage is a new process or later in the same run after `background_adaptive`).

### background_adaptive

```yaml
- kind: background_adaptive
  inputs:
    rough: <label>          # required — workspace containing `rough_bkg_rN.npz` or `.npy`
    diffs: <label>          # required — same diff workspace as used for rough (stem order / BTJD only; FITS not loaded)
  output: <label>           # required — workspace for `bkg_smooth.npz` / `.npy`
  round_id: 1               # optional — must match the rough stack file
  write_per_frame_fits: true  # optional, default true — `{stem}_bkg_smooth.fits`
```

**Produces**: `bkg_smooth.npz` (array `stack`) and `bkg_smooth.npy`, plus optional `{stem}_bkg_smooth.fits`. Axis 0 of the rough stack must match the FFI ordering from `_hotpants_results_from_dirs` for the same `max_ffis` / manifest (validated at run time). When both `.npz` and `.npy` exist from an older run, adaptive prefers loading `.npz` first.

### background_estimate

```yaml
- kind: background_estimate
  inputs:
    diffs: <label>          # required — diff workspace
    bkg: <label>            # required — Hotpants polynomial bkg workspace
  mode: rough_then_adaptive # required, only this mode is implemented
  output: <label>           # required — workspace for `rough_bkg_r1` + `bkg_smooth` stacks and optional per-frame FITS
  round_id: 1               # optional — passed to the rough step
  write_per_frame_fits: true  # optional, default true — rough and smoothed per-frame FITS
```

**Produces**: Same as running `background_rough` then `background_adaptive` into the **same** output workspace: `rough_bkg_r1.npz` / `.npy`, `bkg_smooth.npz` / `.npy`, and the same optional `{stem}_rough_bkg.fits` / `{stem}_bkg_smooth.fits` behavior. Hotpants diff/bkg arrays are stripped from RAM before the adaptive step to reduce peak memory.

### diff_final

```yaml
- kind: diff_final
  inputs:
    diffs: <label>          # required — round-2 diff workspace
    bkg_final: <label>      # required — workspace containing bkg_final.npy
    sat_hr: <label>         # required — high-res sat template workspace
    epsf: <label>           # required — ePSF workspace
  output: <label>           # required — workspace for final diff FITS
```

**Produces**: Final difference FITS under `ws/<output_label>/`.

### forced_photometry

```yaml
- kind: forced_photometry
  inputs:
    diffs: <label>          # required — diff workspace to extract photometry from
    epsf: <label>           # required if using ePSF (mutually exclusive with psf: prf)
  psf: prf                  # set this OR inputs.epsf, not both
  output: <label>           # required — workspace directory (lightcurve.csv written inside)
```

**Validation rule**: Exactly one of `psf: prf` or `inputs.epsf: <label>` must be set.

**Produces**: `lightcurve.csv` and optionally `lightcurve_control.png` (if `pipeline_plots: true`) under `ws/<output_label>/`.

---

## Workspace and Output Layout

### Config-driven pipeline

```
output_dir/
├── syndiff_ffi_frames.csv          # frame manifest
├── cluster_template_job.json       # WCS groups, crop bounds, reference FFI
├── shared_mask.fits                # bitmask
├── ref_stars.csv                   # Hotpants reference stars
├── gaia_catalog_pipeline.csv       # Gaia catalog with tess_mag
├── tile_centers.json               # ePSF tile centers
├── wcs_drift_template_debug.png    # (if pipeline_plots)
│
└── ws/                             # labeled workspaces
    ├── hp_d/                       # Hotpants diff FITS
    │   ├── tess2020..._ffic.fits
    │   └── ...
    ├── hp_c/                       # Hotpants convolved FITS
    ├── hp_b/                       # Hotpants polynomial bkg FITS (optional)
    ├── ep/                         # ePSF products
    │   ├── epsf_stack_r1.npz
    │   ├── epsf_r1_smooth.npz
    │   └── group_epsf/
    ├── sat/                        # sat template FITS
    │   ├── sat_tmpl_native_r1/
    │   └── sat_tmpl_hr_r1/
    ├── bkg_rough/                  # optional — split background: rough stack only
    │   ├── rough_bkg_r1.npz
    │   └── rough_bkg_r1.npy
    ├── bkg_smooth/                 # adaptive temporal background (or legacy single-stage outputs)
    │   ├── rough_bkg_r1.npz        # written by background_rough or background_estimate
    │   ├── rough_bkg_r1.npy
    │   ├── bkg_smooth.npz
    │   └── bkg_smooth.npy
    ├── diffs_minus_bkg/            # subtracted diffs
    └── lc_prf_on_diffs/            # photometry workspace
        ├── lightcurve.csv
        └── lightcurve_control.png
```

### Flat `output_dir` layout (historical / external tools)

Some tools still expect Hotpants round directories and Numpy stacks directly under `output_dir` (e.g. `diff_r1/`, `bkg_smooth_r1.npy`, `diff_final/`). The default orchestrator writes labeled workspaces under `ws/`; use or migrate paths as needed when mixing with older trees.

---

## CLI Usage

### Run the full pipeline (config-driven)

```bash
python -m syndiff_pipeline.run_pipeline --config recipe_a_prf.yaml
```

### Validate without running

```bash
python -m syndiff_pipeline.run_pipeline --config recipe_a_prf.yaml --validate-only
```

### Download FFIs before running

```bash
python -m syndiff_pipeline.run_pipeline --config config.yaml --download
```

### Download FFIs standalone

```bash
python -m syndiff_pipeline.download --sector 20 --camera 3 --ccd 3
python -m syndiff_pipeline.download --sector 20 --camera 3 --ccd 3 --via-mast
```

### CLI overrides

The following config values can be overridden on the command line:

```
--sector N          --camera N          --ccd N
--output-dir PATH   --ffi-dir PATH      --n-jobs N
--max-ffis N        --psf-type {epsf,prf}
--pipeline-plots / --no-pipeline-plots
--verbose           --download
```

---

## Example Recipes

### Recipe A — Simple PRF Photometry

Hotpants with saved polynomial bkg → PRF photometry on raw diffs → **background_rough** + **background_adaptive** (split lowers peak RAM vs. one-shot ``background_estimate``) → subtract background → PRF photometry on cleaned diffs.

```yaml
ffi_dir: "data/tess_ffi"
output_dir: "output/run_a"
gaia_catalog: "data/catalogs/gaia.csv"
template_dir: "output/templates"
target_ra: 210.219333
target_dec: 81.846589
sector: 20
camera: 3
ccd: 3

pipeline:
  - kind: wcs_grouping
  - kind: shared_mask
  - kind: hotpants
    science: ffi
    output:
      diffs: hp_d
      convolved: hp_c
      bkg: hp_b
  - kind: forced_photometry
    inputs:
      diffs: hp_d
    psf: prf
    output: lc_prf_on_diffs
  - kind: background_rough
    inputs:
      diffs: hp_d
      bkg: hp_b
    output: bkg_rough
  - kind: background_adaptive
    inputs:
      rough: bkg_rough
      diffs: hp_d
    output: bkg_smooth
  - kind: subtract
    inputs:
      science: hp_d
      template: bkg_smooth
    output: diffs_minus_bkg
  - kind: forced_photometry
    inputs:
      diffs: diffs_minus_bkg
    psf: prf
    output: lc_prf_after_bkg
```

(Alternatively, a single `background_estimate` stage with `mode: rough_then_adaptive` still works as a convenience wrapper.)

### Recipe B — ePSF + Saturated Star Removal + Background Chain

Full chain: Hotpants → ePSF → saturated star template → subtract sat → ePSF photometry → background estimation → subtract background → ePSF photometry.

```yaml
ffi_dir: "data/tess_ffi"
output_dir: "output/run_b"
gaia_catalog: "data/catalogs/gaia.csv"
template_dir: "output/templates"
target_ra: 210.219333
target_dec: 81.846589
sector: 20
camera: 3
ccd: 3
epsf_temporal_smooth: true

pipeline:
  - kind: wcs_grouping
  - kind: shared_mask
  - kind: hotpants
    science: ffi
    output:
      diffs: hp_d
      convolved: hp_c
      bkg: hp_b
  - kind: epsf
    inputs:
      diffs: hp_d
    temporal_smooth: true
    output: ep
  - kind: sat_template
    inputs:
      diffs: hp_d
      epsf: ep
    output: sat
  - kind: subtract
    inputs:
      science: hp_d
      template: sat
    output: d_sat
  - kind: forced_photometry
    inputs:
      diffs: d_sat
      epsf: ep
    output: lc_epsf_after_sat
  - kind: background_estimate
    inputs:
      diffs: d_sat
      bkg: hp_b
    mode: rough_then_adaptive
    output: bkg_smooth
  - kind: subtract
    inputs:
      science: d_sat
      template: bkg_smooth
    output: d_after_bkg
  - kind: forced_photometry
    inputs:
      diffs: d_after_bkg
      epsf: ep
    output: lc_epsf_final
```

### Recipe C — Two-Pass Hotpants

First Hotpants pass → full ePSF/sat/bkg chain → second Hotpants pass (FFI − saved bkg, reusing convolved) → repeat chain on new diffs.

```yaml
ffi_dir: "data/tess_ffi"
output_dir: "output/run_c"
gaia_catalog: "data/catalogs/gaia.csv"
template_dir: "output/templates"
target_ra: 210.219333
target_dec: 81.846589
sector: 20
camera: 3
ccd: 3
epsf_temporal_smooth: true

pipeline:
  - kind: wcs_grouping
  - kind: shared_mask

  # ── First pass ──
  - kind: hotpants
    science: ffi
    output:
      diffs: hp1_d
      convolved: hp1_c
      bkg: hp1_b
  - kind: epsf
    inputs: { diffs: hp1_d }
    temporal_smooth: true
    output: ep1
  - kind: sat_template
    inputs: { diffs: hp1_d, epsf: ep1 }
    output: sat1
  - kind: subtract
    inputs: { science: hp1_d, template: sat1 }
    output: d1_sat
  - kind: forced_photometry
    inputs: { diffs: d1_sat, epsf: ep1 }
    output: lc1_epsf
  - kind: background_estimate
    inputs: { diffs: d1_sat, bkg: hp1_b }
    mode: rough_then_adaptive
    output: bkg1
  - kind: subtract
    inputs: { science: d1_sat, template: bkg1 }
    output: d1_fin
  - kind: forced_photometry
    inputs: { diffs: d1_fin, epsf: ep1 }
    output: lc2_epsf

  # ── Second pass (background-subtracted FFI) ──
  - kind: hotpants
    science: ffi
    inputs:
      bkg: hp1_b
      convolved: hp1_c
    output:
      diffs: hp2_d
      convolved: hp2_c
      bkg: hp2_b
  - kind: epsf
    inputs: { diffs: hp2_d }
    temporal_smooth: true
    output: ep2
  - kind: sat_template
    inputs: { diffs: hp2_d, epsf: ep2 }
    output: sat2
  - kind: subtract
    inputs: { science: hp2_d, template: sat2 }
    output: d2_sat
  - kind: forced_photometry
    inputs: { diffs: d2_sat, epsf: ep2 }
    output: lc3_epsf
  - kind: background_estimate
    inputs: { diffs: d2_sat, bkg: hp2_b }
    mode: rough_then_adaptive
    output: bkg2
  - kind: subtract
    inputs: { science: d2_sat, template: bkg2 }
    output: d2_fin
  - kind: forced_photometry
    inputs: { diffs: d2_fin, epsf: ep2 }
    output: lc4_epsf
```

---

## External Prerequisites (Before Running SynDiff)

SynDiff assumes the following inputs exist before the pipeline runs:

1. **TESS FFIs** — calibrated FFI FITS files under `{ffi_dir}/s{sector:04d}/cam{camera}_ccd{ccd}/`. Use `python -m syndiff_pipeline.download` or provide pre-downloaded files.
2. **PS1 Templates** — per-group template FITS on the TESS pixel grid, produced by `TSST_Syndiff_Core`. Place under `template_dir` as either:
  - `group_<id>/ps1_template.fits` (or `template.fits`)
  - Flat `syndiff_template_sNNNN_C_C_dxX.XXX_dyY.YYY.fits` naming convention
3. **Gaia Catalog** — a CSV with crop-local `x`, `y` pixel coordinates and photometry columns (`tess_mag` or `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag`). Produced by the template creation pipeline. If only `ra`/`dec` are present, `wcs_grouping.ensure_gaia_crop_xy` will project coordinates using the reference FFI WCS.
4. **Optional: `median_mask.fits`** — TGLC column correction file for ePSF fitting. If absent, uniform correction (1.0) is used.
5. **Optional: `straps_csv`** — CSV of TESS strap column positions for masking. If absent, strap masking is disabled.
6. **Optional: `bkg_vector_path`** — local directory with TESSVectors CSV files. If absent, downloaded from HEASARC during background smoothing.

---

## Module Reference


| File                     | Purpose                                                                                               | Lines |
| ------------------------ | ----------------------------------------------------------------------------------------------------- | ----- |
| `__init__.py`            | Package docstring listing all modules                                                                 | 22    |
| `config.py`              | `SynDiffConfig` dataclass, YAML I/O, CLI argument parsing                                             | 522   |
| `download.py`            | TESS FFI download (tesscurl + astroquery backends)                                                    | 503   |
| `wcs_grouping.py`        | WCS extraction, drift computation, template grouping, crop bounds, Gaia projection                    | ~1000 |
| `masking.py`             | Shared bitmask (Gaia, bright stars, straps), reference star selection, strap QE correction            | 505   |
| `hotpants_runner.py`     | Hotpants differencing loop, template discovery, workspace layout                                      | 618   |
| `epsf_fitting.py`        | Tiled ePSF fitting via TGLC, stack bundle I/O, `CustomSource`, column correction                      | ~627  |
| `temporal_smooth.py`     | ePSF temporal smoothing, group ePSF computation, adaptive background smoothing, final bkg combination | ~372  |
| `adaptive_background.py` | Adaptive median filter (`adaptive_medfilt_3d`), TESSVectors fetch, `AdaptiveBackground` class         | ~516  |
| `sat_template.py`        | Saturated star template construction (native + high-res), sub-pixel ePSF placement                    | ~332  |
| `background.py`          | Rough background estimation (`Smooth_bkg`, inpainting), frame-level stacking                          | 225   |
| `final_diff.py`          | Fourier deconvolution/reconvolution, final difference image production                                | 314   |
| `photometry.py`          | Forced PSF photometry (`create_psf`), `EpsfLocator`, light curve CSV + diagnostic plot                | 619   |
| `frame_manifest.py`      | Frame manifest CSV management, hotpants/ePSF status tracking, diff path lookup                        | 313   |
| `paths.py`               | Workspace dir and manifest path conventions                                                           | 39    |
| `pipeline_context.py`    | `PipelineInvocationContext` (resolved paths for config-driven mode)                                   | 27    |
| `pipeline_validate.py`   | Config-driven pipeline validation (stage kinds, required keys, dependency graph)                      | 188   |
| `pipeline_execute.py`    | Config-driven orchestrator (`run_config_pipeline`)                                                    | 715   |
| `run_pipeline.py`        | CLI entry point; requires `pipeline:` and calls `run_config_pipeline`                                  | ~100  |
| `plot_pipeline.py`       | Background removal animated GIF diagnostic                                                            | 155   |


---

## Known Limitations and Work in Progress

1. `**hotpants inputs.convolved` not consumed** — the second-pass Hotpants accepts `inputs.convolved` in the config and validates it, but logs a warning that it is ignored in this version. Convolved products are always written fresh to `output.convolved`.
2. **Only `science: ffi` for Hotpants** — the `science` field on the `hotpants` stage must be `"ffi"`. Using a labeled workspace as science input is not yet implemented (validation rejects it).
3. **`background_rough` + `background_adaptive` vs `background_estimate`** — use the split kinds to run rough and adaptive in separate processes or to resume adaptive only (`recipe_a_prf_from_background.yaml`). `background_estimate` with `mode: rough_then_adaptive` remains a one-shot convenience that writes both artifacts into one workspace and strips Hotpants arrays before adaptive.
4. **No per-stage checkpointing** — each run re-executes every stage in the `pipeline:` list. To skip work, trim the list and use `pipeline_external_workspace_labels` to declare pre-existing workspaces.
5. **Source extraction / astrometry not included** — SynDiff reads WCS from FFI FITS headers directly. If mission WCS quality is insufficient for drift grouping or PS1 alignment, external astrometric correction is required before `wcs_grouping`.
6. **PS1 template creation is separate** — the `TSST_Syndiff_Core` package handles pancakes, PS1 download, processing, and multi-offset downsampling. This must be run independently before SynDiff.
7. `**diff_final` stage** — currently requires all four input labels (`diffs`, `bkg_final`, `sat_hr`, `epsf`). There is no simplified variant that skips the Fourier deconvolution step.
8. **Sequential-only execution** — stages run sequentially within a single process. There is no distributed or DAG-based execution scheduler.

