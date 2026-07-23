> **Package integration**: diff sub-stage `centroids` · module `difference_imaging/stages/centroids.py`  
> **Related docs**: [gridded ePSF](gridded_epsf.md) · [diff pipeline internals](diff_pipeline.md) · [linear centroids campaign](../linear_centroids_pipeline.md) · [forced photometry](forced_photometry.md)

# Gaia centroids on difference images (`centroids`)

Runs **multi-star PSF photometry** on every difference image using photutils `PSFPhotometry` with the per-frame `GriddedPSFModel` from an upstream `epsf` stage. Positions and fluxes are written per FFI for astrometric calibration — notably the [linear centroids campaign](../linear_centroids_pipeline.md) that feeds a future temporally varying WCS (phase 2).

This stage measures **many Gaia stars per frame**; target forced photometry is a separate `forced_photometry` sub-stage.

---

## Dependencies

```text
… → hotpants (or kernel_subtract) → epsf → centroids
```

| Input | Source |
|-------|--------|
| Difference FITS | `inputs.diffs` (e.g. `hp_d`) |
| Gridded ePSF | `inputs.epsf` workspace with `gridded_epsf_index.json` |
| Gaia catalog | `gaia_catalog_pipeline.csv` from `shared_mask` |
| Frame WCS | `ffi_list.parquet` + `MappingGrid.science_ffi_bounds()` |

**Hard dependency:** without a gridded ePSF npz for a frame, that frame is skipped (`centroids: no gridded ePSF for {stem}`).

Example YAML:

```yaml
- kind: centroids
  inputs:
    diffs: hp_d
    epsf: epsf_r1
  output: centroids_r1
```

---

## Gaia star selection

Two magnitude cuts (defaults match `starpositioningscript.py`):

| Cut | Default | Applied |
|-----|---------|---------|
| Bright end | `mag_max_rp` = 12.95 | Parent-process pre-filter (same as ePSF) |
| Faint end | `mag_min_rp` = 7.5 | Per-frame after `gaia_science_xy_for_frame` |

Requires `ra`, `dec` in the Gaia table. Star **x/y** are computed per FFI from full-FFI WCS — not taken from the static Gaia CSV crop columns.

**No shared mask** is passed to `PSFPhotometry` (unlike ePSF fitting, which rejects stars on mask bits). Centroids rely on magnitude cuts and the grouper only.

---

## Algorithm (per frame)

1. Load diff image FITS (crop-local science array).
2. Load `GriddedPSFModel` from `{ffi_stem}_gridded_epsf.npz` via `GriddedEpsfCatalog`.
3. Project Gaia to science pixels for this FFI (`gaia_science_xy_for_frame`).
4. Apply faint-end magnitude cut (`_filter_gaia_for_centroids`).
5. Build `init_params` with `x_init`, `y_init` plus Gaia metadata columns (`source_id`, `ra`, `dec`, `phot_*`, `tess_mag`, …).
6. `PSFPhotometry(gridded_model, fit_shape, aperture_radius, SourceGrouper(min_separation=…))` — **no** local background estimator.
7. Write results table to ECSV; update `centroids_index.json`.

Parallelism: joblib `loky` over frames (`centroids_n_jobs` or `defaults.n_jobs`).

---

## Outputs

SCC-primary path when configured:

`{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/{output}/`

| Artifact | Description |
|----------|-------------|
| `{ffi_stem}_photresults.ecsv` | Per-frame multi-star table: Gaia metadata + fitted `x_fit`, `y_fit`, fluxes, flags |
| `centroids_index.json` | `ffi_stem` → photresults path |
| `centroids.progress.json` | Frame progress; mirrored as `diff.centroids.progress.json` beside `diff.log` |

Resume: valid non-empty ECSV on disk (or provenance-complete `centroids` artifact) is skipped unless `force_rerun: true`.

### Debug helper

`write_frame_residual_fits()` runs one frame and writes a full-frame residual FITS via `psf_phot.make_residual_image()` — intended for interactive debugging, not production pipeline output.

---

## Key YAML parameters (`CentroidsParams`)

| Key | Default | Notes |
|-----|---------|-------|
| `mag_max_rp` | 12.95 | Upper RP magnitude |
| `mag_min_rp` | 7.5 | Lower RP magnitude |
| `fit_shape` | 11 | PSFPhotometry fit region |
| `aperture_radius` | 4 | Aperture for initial fluxes |
| `psf_grouper_min_separation` | 10 | `SourceGrouper` separation (pixels) |
| `centroids_n_jobs` | — | Frame parallelism override |

---

## Linear centroids campaign

Phase 1b of [linear centroids](../linear_centroids_pipeline.md) runs kernel-fit diff → Hotpants `hp_d` → `epsf` → `centroids` on the `diff_linear/` lane. Centroid ECSV files under `centroids_r1/` are the planned input to phase 2 (temporally varying WCS from measured star positions).

Typical submit:

```bash
syndiff diff submit \
  --config config/linear_centroids/pipeline.yaml \
  --scc config/scc_my_lanes.csv \
  --stages diff
```

Lane table (under `{data_root}/s{SSSS}/c{C}/k{K}/diff_linear/`):

| Label | Contents |
|-------|----------|
| `hp_d/` | Difference images for photometry |
| `epsf_r1/` | `{ffi_stem}_gridded_epsf.npz` |
| `centroids_r1/` | `{ffi_stem}_photresults.ecsv` |

---

## Relation to `forced_photometry`

| | `centroids` | `forced_photometry` (`psf_type: epsf`) |
|--|-------------|----------------------------------------|
| Targets | Many Gaia stars | Primary + `additional_forced_targets` |
| Init positions | Gaia per frame | Target RA/Dec → pixels |
| Output | Per-frame ECSV | `lightcurve_*.csv` |
| Grouper | `SourceGrouper` enabled | `grouper=None` (isolated fit) |

Both load the same per-frame `GriddedPSFModel`; see [gridded ePSF](gridded_epsf.md) and [forced photometry](forced_photometry.md).

---

## Key source files

| File | Role |
|------|------|
| [`centroids.py`](../../../syndiff_pipeline/difference_imaging/stages/centroids.py) | Frame loop, PSFPhotometry, index I/O |
| [`centroids_progress.py`](../../../syndiff_pipeline/difference_imaging/stages/centroids_progress.py) | Progress sidecars |
| [`gridded_epsf.py`](../../../syndiff_pipeline/difference_imaging/stages/gridded_epsf.py) | `GriddedEpsfCatalog`, fingerprint helpers |
| [`execute.py`](../../../syndiff_pipeline/difference_imaging/orchestration/execute.py) | Sub-stage dispatch |
