> **Package integration**: `syndiff photometry` · `syndiff_pipeline/photometry/site_config.py`  
> **Related docs**: [photometry quick start](../photometry.md) · [photometry pipeline](photometry_pipeline.md) · [forced photometry](forced_photometry.md) · [config README](../../../config/README.md)

# Photometry configuration reference

Site policy file: **`photometry_config.yaml`** (or any path passed as `--photometry-config`). Examples: [`config/photometry_config_*.yaml`](../../../config/).

## Top-level keys

| Key | Required | Role |
|-----|----------|------|
| `deployment_file` | no (default `deployment.yaml`) | Relative to the config file’s directory |
| `defaults` | no | Multi-stage knobs merged with per-SCC `overrides` |
| `paths` | no | Lane / input labels (`inputs.diffs`, `inputs.epsf`, store names) |
| `pipeline` | **yes** | Ordered list of `{kind: astrometry\|forced_photometry, …}` |
| `additional_forced_targets` | no | Extra forced positions applied to every event |
| `per_event_force_targets` | no | Map of event label → extra forced-target list |
| `overrides` | no | Per-SCC (`sector/camera/ccd`) overrides of `defaults` / `paths` / `pipeline` |
| `condor` | no | `request_cpus`, `request_memory`, `host_stats_min_mem_mb`, `host_stats_max_load15` (legacy `requirements` / `rank` rejected) |

## `defaults`

| Key | Default | Notes |
|-----|---------|-------|
| `photometry_run_id` | `null` | Namespaces `phot_{id}/` under the event SCC leaf |
| `n_jobs` | `16` | Frame-parallel workers |
| `pipeline_plots` | `false` | Write LC debug PNGs |
| `pipeline_plots_dir` | `debug_plots` | Subdir under the photometry root |
| `pipeline_plot_dpi` | `150` | |
| `max_ffis` | `null` | Cap frames for smoke tests |

## `paths`

| Key | Role |
|-----|------|
| `inputs.diffs` | Diff label under `diff_{lane}/` (default `hp_d`) |
| `inputs.epsf` | Optional ePSF label; required when any method uses `psf_type: epsf` |
| `output_store_name` / `template_store_name` / `remap_store_name` | Named SCC lanes (same conventions as diff) |
| Oversampling | Taken from merged defaults / paths as resolved into `PhotometryRunConfig.oversampling_factor` |

## `pipeline` kinds

### `astrometry`

Optional first stage. Params mirror the former diff-side astrometry block (`sigma_mag_limit`, `clip_n_sigma`, survey credentials). Writes `astrometry_result.json` and updates the transient RA/Dec used by forced photometry.

### `forced_photometry`

| Key | Role |
|-----|------|
| `inputs.diffs` / `inputs.epsf` | May override site `paths.inputs` for this stage |
| `output` | Light-curve subdirectory label under `phot_{run_id}/` |
| `methods` | List of aperture / PSF method dicts — see [forced_photometry.md](forced_photometry.md) |
| `position_source` | `native_wcs` (default) or `temporal_wcs` — see below. Stage-level default; each method may override it. |
| `temporal_wcs_version` | Required when `position_source: temporal_wcs` (stage-level or per-method); must match the diff_config `temporal_wcs` stage's `version` |

Every `psf_type: epsf` method needs a resolvable gridded ePSF catalog (`gridded_epsf_index.json`) under the named ePSF label on the SCC diff lane.

#### `position_source`

By default every `sky`-mode forced target (the primary target, and any
`additional_forced_targets`/`per_event_force_targets` entry with
`position_mode: sky`) is placed on each frame by projecting its RA/Dec
through that frame's own native archive FITS WCS (`native_wcs`).

`position_source: temporal_wcs` instead resolves the position through the
self-calibrated per-orbit WCS model built by the diff-side `temporal_wcs`
stage (Gaia-matched centroids, Chebyshev spatial + B-spline temporal
correction) — generally more accurate than the archive WCS, since it's fit
directly from stars observed in the same diff images. It requires that
stage to have already completed for the SCC (`{data_root}/s{S}/c{C}/k{K}/wcs/{temporal_wcs_version}/manifest.json`
must exist); `scripts/submit_photometry_when_ready.py` checks for this
automatically when a config sets `position_source: temporal_wcs` (stage-level
or per-method). Frames not covered by the temporal_wcs fit (a small number
per sector — orbit edges / rejected fits) fall back to `native_wcs`
automatically for those frames only. `offset`/`fixed`-mode extra targets are
unaffected either way.

`position_source`/`temporal_wcs_version` may also be set on an individual
`methods[]` entry, overriding the stage default for just that method. This
is how one stage compares several position sources (e.g. a free-fit
diagnostic plus `native_wcs` and `temporal_wcs` forced methods) on the same
target without duplicating `astrometry`/`inputs`: every `psf_type: epsf`
method using the default `fitter: photutils`, regardless of its own
`position_source`, is still batched through one shared read of each diff
FITS (`run_forced_photometry_gridded_multi_method`) — the number of such
methods in a stage does not multiply I/O.

When `pipeline_plots: true` and any method resolves to `position_source:
temporal_wcs`, a debug plot `temporal_wcs_offset_{output}.png` (or
`temporal_wcs_offset_{output}_{version}.png` when methods in the stage use
more than one `temporal_wcs_version`) is written under the pipeline-plots
dir: BTJD vs dx (left y-axis) and dy (right y-axis, `twinx`) on one panel,
plus two separate histograms — one for dx, one for dy, each on their own
axes — so their scales don't compress each other. Both WCS solutions are
read from the SCC's cached `ffi_list.parquet` (header already extracted at
download time), so this never opens FFI FITS files; frames with
`wcs_ok=False` in that cache, and frames the temporal_wcs fit itself
doesn't cover, are excluded entirely rather than plotted with a degenerate
placeholder WCS. Built by `write_temporal_wcs_offset_debug_plot` in
`stages/photometry.py`.

## Forced-target extras

`additional_forced_targets` and `per_event_force_targets[event_label]` are concatenated, then normalized (`normalize_additional_forced_targets`). Each entry typically has `name`, `position_mode` (`sky` or pixel offsets), and either `ra`/`dec` or `dx`/`dy` relative to the primary target.

## Frozen run copy

`syndiff photometry submit` freezes the photometry policy under `runs/{run_id}/photometry_config.yaml`. Debug with that file plus `run_meta.json`, not only the live site YAML.

## Orchestrator wiring

`pipeline.yaml` still owns `stages.photometry` executor / Condor pool settings. Photometry science knobs stay in `photometry_config.yaml` — same split as `diff_config.yaml` vs `pipeline.yaml` for the diff stage.
