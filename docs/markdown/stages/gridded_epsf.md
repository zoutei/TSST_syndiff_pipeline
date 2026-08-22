> **Package integration**: diff sub-stage `epsf` · core module `difference_imaging/stages/gridded_epsf.py` · orchestrated by `stages/epsf.py`  
> **Related docs**: [diff pipeline internals](diff_pipeline.md) · [centroids](centroids.md) · [forced photometry](forced_photometry.md) · [static masking](../masking.md) · [linear centroids campaign](../linear_centroids_pipeline.md)

# Gridded empirical PSF (`epsf` / `gridded_epsf`)

Builds a spatially varying PSF model on difference images using **photutils** (`EPSFBuilder` + `GriddedPSFModel`). This is **not** TGLC/TESSreduce ePSF — those appear only in `forced_photometry` with `fitter: tessreduce`.

The YAML stage kind is `epsf`; `execute.py` and `star/epsf_runner.py` both call `epsf.fit_epsf_all_frames()`, which is the single dispatch point on `EpsfParams.epsf_mode`:

- **`orbit_binned` (default)** — fits only a handful of **anchor** models per orbit from batches of representative FFIs; every other frame resolves to a BTJD-interpolated blend of its two bracketing anchors. Delegates to `gridded_epsf_orbit.fit_gridded_epsf_orbit_binned()`. See [§ Orbit-binned mode](#orbit-binned-mode-default) below.
- **`per_frame`** — the original one-model-per-FFI path, delegates to `gridded_epsf.fit_gridded_epsf_all_frames()` unchanged.

Both modes return the same `(epsf_stack, tile_centers, ffi_stems, epsf_ok)` contract, so every downstream consumer (`centroids`, `forced_photometry`, `sat_template`'s legacy stack) works identically regardless of mode.

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
| `epsf_mode` | `orbit_binned` | `orbit_binned` or `per_frame` — see [§ Orbit-binned mode](#orbit-binned-mode-default) |
| `epsf_per_orbit` | 5 | Anchor models per orbit (orbit-binned only) |
| `epsf_frames_per_anchor` | 20 | Frames pooled into each anchor's fit |
| `epsf_stack_before_fit` | true | `true` = mean-combine window before fitting (fast, default); `false` = pool per-frame star extractions into one `EPSFBuilder` call (keeps per-frame recentering, more expensive) |
| `epsf_anchor_edge_fraction` | 0.12 | Fraction of orbit duration treated as the dense edge zone |
| `epsf_anchor_edge_boost` | 3.0 | Relative anchor density in the edge zone vs. interior |
| `epsf_anchor_window_max_expand` | 80 | Cap on window-expansion radius before falling back to best-available frame count |
| `epsf_quality_bitmask` | 583 | `DQUALITY` bits that disqualify a frame from anchor building (default: attitude tweak\|safe mode\|coarse point\|manual exclude\|Earth/Moon in FFI — see table below) |
| `epsf_debug_plots` | true | Write per-orbit anchor/window diagnostic plots (orbit-binned only) |
| `epsf_mag_source` | `phot_rp_mean_mag` | `phot_rp_mean_mag` or `tess_mag` — see [§ Star-selection parity with dev/forward_epsf_wcs](#star-selection-parity-with-devforward_epsf_wcs) |
| `epsf_isolation_min_sep_px` | null (disabled) | Minimum pixel separation from any `tess_mag < epsf_isolation_neighbor_mag_max` neighbor |
| `epsf_isolation_neighbor_mag_max` | 13.0 | TESS-mag threshold for isolation-check neighbors |

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

## Orbit-binned mode (default)

Per-frame fitting (`epsf_mode: per_frame`) is the dominant cost of the `epsf`
stage: one `EPSFBuilder` fit per FFI. Orbit-binned mode instead fits only
`epsf_per_orbit` **anchor** models per orbit, each from a batch of
`epsf_frames_per_anchor` representative FFIs, and has every other frame in
that orbit resolve to an interpolated blend of its two bracketing anchors —
via the *same* `gridded_epsf_index.json` (`ffi_stem → npz path`) contract
`per_frame` already uses (many stems already CAN point at one file; this is
the mechanism that makes "N anchors cover the whole orbit" work with zero
consumer changes).

**Orbit segmentation** reuses `shift_schedule._split_orbit_segments_from_csv`
(the same MIT `TESS_orbit_times.csv` partitioner `temporal_wcs.py::
_orbit_bounds` wraps), sourced from `ffi_list`'s cached `DATE-OBS` — so orbit
indices agree with `temporal_wcs` without re-opening any FITS file.

**Anchor placement** uses an edge-weighted density (denser near orbit
start/end, where drift changes fastest): a symmetric density weight over
normalized orbit phase is integrated to a CDF and inverted at
`epsf_per_orbit` evenly spaced quantiles. This is the density idea
`TemporalWcsParams.edge_densify_knots`/`edge_fraction` name but never
consume (`_make_knots` ignores them) — orbit-binned ePSF actually implements
it, independently, in `gridded_epsf_orbit.anchor_target_phases()`.

**Anchor frame selection**: each target phase maps to its nearest actual
FFI (fixing that anchor's identity/npz stem); a contiguous window then
grows outward from that position, excluding frames whose `DQUALITY &
epsf_quality_bitmask != 0`, until `epsf_frames_per_anchor` good frames are
collected or `epsf_anchor_window_max_expand` is hit (logged, falls back to
best-available count). Default `DQUALITY` bits excluded (583 = 1+2+64+512):

| Bit | Value | Meaning |
|-----|-------|---------|
| 1 | 1 | Attitude tweak |
| 2 | 2 | Safe mode |
| 7 | 64 | Manual exclude |
| 10 | 512 | Earth/Moon in FFI (stray light) |

Fully config-overridable via `epsf_quality_bitmask`.

**Anchor fitting** — `epsf_stack_before_fit: true` (default, stacked mode):
NaN-aware mean-combine the window's diff images into one synthetic frame,
using the **union** (logical OR) of every window frame's reject mask (not
one representative frame's — avoids leaking transient artifacts unique to
other frames into the average unmasked), then reuse
`build_gridded_psf_for_frame()` unchanged. Gaia positions use the anchor's
own assigned FFI's WCS. **Caveat**: pre-averaging in pixel space forfeits
`EPSFBuilder`'s per-frame recentering — residual sub-pixel pointing jitter
across the window can smear the resulting model (known limitation, see
`syndiff-pipeline-map` invariant #2: drift is only measured/corrected at the
target position). `epsf_stack_before_fit: false` (pooled mode) instead pools
each frame's own star extractions into one `EPSFBuilder` call per tile via
`gridded_epsf.fit_epsf_section_multi()` (keeps per-frame recentering; more
expensive).

**Index materialization**: an anchor frame's own npz path is stored under
its own `ffi_stem` (via the same `resolve_diff_write_path(kind="epsf", ...)`
contract `per_frame` uses — no synthetic orbit/anchor-index filename); a
non-anchor frame between two anchors gets a cheap BTJD-weighted blended npz
(no re-fitting — just a weighted array combination) written under its own
stem; a frame before the first / after the last anchor in its orbit clamps
directly to that anchor's npz. The legacy `epsf_stack_r{N}.npz` bundle is
still assembled from these (real + blended) per-frame stacks, so
`sat_template` keeps working unmodified.

**Provenance**: an anchor's `epsf` fingerprint depends on every diff image
in its own fit window (not just its own frame); an interpolated frame's
fingerprint depends on its two neighboring anchors' own `epsf` fingerprints
(not its own diff image, which the artifact does not actually depend on in
isolation) — see `gridded_epsf_orbit.anchor_epsf_fingerprint()` /
`interpolated_epsf_fingerprint()`. The bulk indexed-verify fast path
(`diff_verify.py`) falls open (`None` → legacy marker check) for
`epsf_mode: orbit_binned` rather than replay this scheme, since doing so
would require re-deriving orbit segmentation/anchor placement per product
id.

**`max_ffis` debug crops**: `epsf_frames_per_anchor`/`epsf_per_orbit` are
automatically relaxed per orbit (capped at that orbit's actual frame count,
with a logged warning) when a truncated smoke-test run doesn't have enough
frames — see `CLAUDE.md`'s `max_ffis` testing convention.

**Debug plot** (`epsf_debug_plots: true`, default): one
`debug_plots/epsf_orbit_{NN}_anchor_selection.png` per orbit under the
stage's output dir — FFI BTJD rug (quality-excluded frames marked), each
anchor's target phase vs. its assigned FFI, each anchor's selected window
bracketed, and the interpolation blend weight for non-anchor frames.
Best-effort (a missing/broken matplotlib never invalidates science output).

**Star branch**: `star_config.yaml`'s `epsf:` block (`StarEpsfConfig`) takes
the same `epsf_mode`/anchor/quality-bitmask keys, defaulting to
`orbit_binned` like the diff stage.

---

## Star-selection parity with dev/forward_epsf_wcs

`dev/forward_epsf_wcs` is a separate, non-production GPU-based per-epoch
WCS+ePSF fitter (not in the diff DAG — see `CLAUDE.md`); its star-selection
criteria can optionally be ported into the production `epsf` stage, applied
within the existing tile-grid `EPSFBuilder` pipeline (not that fitter's own
irregular-stamp/packed-support architecture):

- **`epsf_mag_source: tess_mag`** — `mag_min_rp`/`mag_max_rp` are
  reinterpreted as bounds on a per-star TESS magnitude derived from Gaia
  G/BP/RP (`epsf.tess_mag_from_gaia_phot`, the TGLC polynomial) instead of
  raw `phot_rp_mean_mag`. Default remains `phot_rp_mean_mag` (unchanged
  legacy behavior).
- **`epsf_isolation_min_sep_px`** — when set, adds a global (whole-frame,
  not per-tile) isolation filter: a candidate star is dropped unless its
  nearest Gaia neighbor brighter than `epsf_isolation_neighbor_mag_max`
  (TESS mag, default 13.0) is at least this many pixels away. Direct port
  of `dev/forward_epsf_wcs.isolated_forced_phot.select_isolated_stars`'s
  rule (`gridded_epsf.apply_epsf_isolation_filter`) — the candidate window
  and the neighbor pool are drawn from the *same* full, unfiltered Gaia
  catalog (a star between `mag_max_rp` and `neighbor_mag_max` isn't itself a
  candidate but still counts as a contaminating neighbor), evaluated after
  per-frame `x`/`y` projection, before the tile-section loop. `None`
  (default) disables isolation filtering entirely.
- forward_epsf_wcs's own defaults: `--tess-mag 7,11`, `--min-sep-px 6.0`,
  neighbor threshold 13.0, `--epsf-grid 2x2` over a small central `--region`
  crop. Production `epsf` always tiles the *whole* science array (no
  `--region`-equivalent restriction exists), so matching "2x2 over the
  middle" is a straight `tile_nx`/`tile_ny` bump (e.g. to 4x4) with no other
  change needed.

When `epsf_isolation_min_sep_px` is set, `prepare_gaia_for_gridded_epsf`
deliberately **skips** its magnitude-window prefilter (deferring it to the
per-frame isolation pass) — narrowing to the mag window before computing
isolation would silently drop the fainter neighbors the isolation check
needs to see.

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
| [`gridded_epsf.py`](../../../syndiff_pipeline/difference_imaging/stages/gridded_epsf.py) | Per-frame fitting, NPZ I/O, index, frame loop (`per_frame` mode) |
| [`gridded_epsf_orbit.py`](../../../syndiff_pipeline/difference_imaging/stages/gridded_epsf_orbit.py) | Orbit-binned anchor placement/selection/fitting/index materialization/debug plot (`orbit_binned` mode) |
| [`epsf.py`](../../../syndiff_pipeline/difference_imaging/stages/epsf.py) | Stage entry `fit_epsf_all_frames` — dispatches on `epsf_mode`, legacy stack bundles |
| [`ffi_quality.py`](../../../syndiff_pipeline/common/ffi_quality.py) | `DQUALITY` extraction from cached `ffi_list` headers |
| [`epsf_progress.py`](../../../syndiff_pipeline/difference_imaging/stages/epsf_progress.py) | Progress JSON sidecars |
| [`centroids.py`](../../../syndiff_pipeline/difference_imaging/stages/centroids.py) | Downstream PSF photometry |
| [`photometry.py`](../../../syndiff_pipeline/difference_imaging/stages/photometry.py) | Target forced photometry with gepsf |
