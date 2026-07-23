> **Package integration**: diff sub-stage `epsf` · core module `difference_imaging/stages/gridded_epsf.py` · orchestrated by `stages/epsf.py`  
> **Related docs**: [diff pipeline internals](diff_pipeline.md) · [centroids](centroids.md) · [forced photometry](forced_photometry.md) · [static masking](../masking.md) · [linear centroids campaign](../linear_centroids_pipeline.md)

# Gridded empirical PSF (`epsf` / `gridded_epsf`)

Builds a **per-frame** spatially varying PSF model on difference images using **photutils** (`EPSFBuilder` + `GriddedPSFModel`). This is **not** TGLC/TESSreduce ePSF — those appear only in `forced_photometry` with `fitter: tessreduce`.

The YAML stage kind is `epsf`; `execute.py` calls `epsf.fit_epsf_all_frames()`, which delegates to `gridded_epsf.fit_gridded_epsf_all_frames()`.

---

## Role in the diff pipeline

Typical placement (see [`config/pipeline_epsf_gepsf.yaml`](../../../config/pipeline_epsf_gepsf.yaml)):

```text
shared_mask → hotpants → epsf → centroids → forced_photometry
```

`epsf` reads difference FITS from an upstream `hotpants` or `kernel_subtract` label (`inputs.diffs`). Outputs feed:

- **`centroids`** — multi-star PSF photometry for astrometry / linear-centroids campaigns
- **`forced_photometry`** with `psf_type: epsf` — requires `gridded_epsf_index.json` under `inputs.epsf`
- **`star`** — host-star workflow can consume gridded models when configured for gepsf inputs

Legacy tile-stack bundles (`epsf_stack_r*.npz`, `epsf_r*_smooth.npz`, `group_epsf/group_epsf_{gid}.npy`) are still written for **`sat_template`** only; forced photometry does **not** use the smooth-stack fallback.

---

## Algorithm (per difference image)

1. **Gaia pre-filter** — `phot_rp_mean_mag < mag_max_rp` (default 12.95); expects `ra`/`dec` in the catalog.
2. **Per-frame positions** — `gaia_science_xy_for_frame()` projects stars using **per-FFI full-FFI WCS** from `ffi_list.parquet`, rebased to the science crop via `MappingGrid.science_ffi_bounds()` (not diff FITS headers).
3. **Tile grid** — image split into `tile_ny × tile_nx` sections (default **5×5**). Section bounds match `starpositioningscript.py` layout (`step_x = nx // tile_nx`, half-open intervals).
4. **Per section** — Gaia stars in section (with edge margin `extract_size/2 + 2`) → mask filter → `extract_stars` + `EPSFBuilder` → oversampled stamp.
5. **Fallback** — failed or star-poor sections receive the mean of successful section stamps.
6. **Border crop** — optional symmetric trim `epsf_stamp_border_crop` (default 8) before stacking.
7. **Model** — stamps stacked into `GriddedPSFModel` with `grid_xypos` metadata.

**Masking:** when a `MaskCatalog` is available, each FFI uses `epsf_reject_mask(mask_at(btjd))` — bits **1|2|32** (catalog stars) are **ignored**; straps, edges, PS1, TNS, asteroids reject stars. Static `shared_mask.fits.fz` is a fallback when no catalog is wired.

---

## Primary outputs

Under the stage `output` label (e.g. `epsf_r1/`), SCC-primary when `data_root` is set:

`{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/{output}/`

| Artifact | Description |
|----------|-------------|
| `{ffi_stem}_gridded_epsf.npz` | Per-frame archive: `data` (grid cube), `grid_xypos`, `oversampling` |
| `gridded_epsf_index.json` | Map `ffi_stem` → npz path (required by downstream ePSF photometry) |
| `epsf.progress.json` | Frame progress sidecar; mirrored as `diff.epsf.progress.json` beside `diff.log` |
| `epsf_stack_r{N}.npz` | Legacy flat stack `(n_frames, n_tiles, n_pix)` + `ffi_stem` axis (from `epsf.py`) |

Per-FFI stems follow `tess{digits}-s{SSSS}-{C}-{K}_{label}` convention when using SCC lane storage (`support/ffi_naming.py`).

### NPZ layout

```python
# load_gridded_psf_model(path) → photutils GriddedPSFModel
data          # (n_tiles, ny_stamp, nx_stamp) float64 cube
grid_xypos    # (n_tiles, 2) tile centers in crop-local pixels
oversampling  # int, from epsf_oversample (default 2)
```

`GriddedEpsfCatalog` (`catalog_from_workspace`) provides `load_model(ffi_stem)` for photometry stages.

---

## Key YAML parameters (`EpsfParams`)

| Key | Default | Notes |
|-----|---------|-------|
| `tile_nx`, `tile_ny` | 5 | Section grid |
| `epsf_oversample` | 2 | EPSFBuilder oversampling |
| `psf_size` | 3 | Half-size of model stamp |
| `extract_size` | — | Star cutout size (defaults to `psf_size` derivation) |
| `min_stars_per_tile` | 5 | Minimum Gaia stars per section |
| `mag_max_rp` | 12.95 | Bright-end cut (`null` → 12.95) |
| `epsf_maxiters` | 15 | EPSFBuilder iterations |
| `epsf_recentering_maxiters` | 20 | Recentering iterations |
| `epsf_smoothing_kernel` | `quadratic` | Builder smoothing |
| `epsf_builder_fit_shape` | 5 | Builder fit shape |
| `epsf_recentering_boxsize` | 3 | Recentering box |
| `epsf_star_box_radius` | 7 | Geometric mask filter around each star |
| `epsf_use_section_mask` | true | Pass section mask into `NDData` |
| `epsf_stamp_border_crop` | 8 | Symmetric stamp trim |
| `epsf_n_jobs` | — | Frame parallelism override (else `defaults.n_jobs`) |

Stage wiring example:

```yaml
- kind: epsf
  inputs:
    diffs: hp_d
  output: epsf_r1
```

---

## Parallelism and resume

- Frame loop uses joblib `loky` with `_init_gridded_epsf_worker` (Gaia table, mask catalog, provenance fingerprints pickled once per worker).
- `skip_existing` (default): valid `{ffi_stem}_gridded_epsf.npz` on disk or provenance-complete `epsf` artifact → skip frame.
- `force_rerun: true` on the stage disables skip (recompute every frame).
- BLAS threads capped per worker (`OMP_NUM_THREADS`, etc.) to avoid oversubscription.

---

## Group-level medians (optional)

`compute_group_epsf_gridded()` medians per-frame cubes by WCS `group_id`, writing `group_epsf/group_epsf_{gid}.npz` — used by legacy `sat_template` grouping, not by `forced_photometry` gepsf mode.

---

## Consumers

| Consumer | Requirement |
|----------|-------------|
| `centroids` | `inputs.epsf` label; loads models via `GriddedEpsfCatalog` |
| `forced_photometry` (`psf_type: epsf`) | `gridded_epsf_index.json` under `inputs.epsf` |
| `sat_template` | Legacy stacks / `group_epsf_*` (not the per-frame npz index) |

See [forced photometry § ePSF photutils](forced_photometry.md#3-epsf-photutils-default-epsf).

---

## Key source files

| File | Role |
|------|------|
| [`gridded_epsf.py`](../../../syndiff_pipeline/difference_imaging/stages/gridded_epsf.py) | Fitting, NPZ I/O, index, frame loop |
| [`epsf.py`](../../../syndiff_pipeline/difference_imaging/stages/epsf.py) | Stage entry `fit_epsf_all_frames`, legacy stack bundles |
| [`epsf_progress.py`](../../../syndiff_pipeline/difference_imaging/stages/epsf_progress.py) | Progress JSON sidecars |
| [`centroids.py`](../../../syndiff_pipeline/difference_imaging/stages/centroids.py) | Downstream PSF photometry |
| [`photometry.py`](../../../syndiff_pipeline/difference_imaging/stages/photometry.py) | Target forced photometry with gepsf |
