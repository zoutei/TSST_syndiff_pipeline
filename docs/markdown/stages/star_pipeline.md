# Star pipeline (technical)

Host-star light curves are the **third branch** of SynDiff: template → diff → **star**. Star never re-runs Hotpants; it consumes per-frame diff artifacts from an already-completed transient workspace.

**Quick start:** [star_lightcurves.md](../star_lightcurves.md)  
**Configuration:** [star_config.md](star_config.md)

## Pipeline position

```text
tess_ffi_download → … → downsample → diff  (transient/supernova)
                                      ↓ verify on disk
                                    star   (host TIC/Gaia stars)
```

Star runs are configured separately from transient `targets.csv` via [`star_config.yaml`](../../../config/star_config.yaml) and [`star_targets.csv`](../../../config/star_targets_example.csv).

## Package layout

| Module | Role |
|--------|------|
| [`cli.py`](../../../syndiff_pipeline/star/cli.py) | `syndiff star submit|run` argument parsing and run setup |
| [`runner.py`](../../../syndiff_pipeline/star/runner.py) | Per-host loop: resolve → mini-templates → stamps → photometry → `batch_manifest.csv` |
| [`diff_runner.py`](../../../syndiff_pipeline/star/diff_runner.py) | Per-frame stamp: `FFI − (conv_temp − S_conv) − phot_bkg` using persisted kernels |
| [`star_segments.py`](../../../syndiff_pipeline/star/star_segments.py) | PS1 skycell lookup, SEP isolation, blend flag, mini-template orchestration |
| [`mini_downsample.py`](../../../syndiff_pipeline/star/mini_downsample.py) | Gaussian convolution of star-only cutout + sparse downsampling to TESS grid |
| [`windowed_photometry.py`](../../../syndiff_pipeline/star/windowed_photometry.py) | Forced aperture / PRF photometry on small diff stamps |
| [`hosts.py`](../../../syndiff_pipeline/star/hosts.py) | Parse `star_hosts/*.csv` (`tic_id` / `gaia_source_id`) |
| [`context.py`](../../../syndiff_pipeline/star/context.py) | Event context, baseline label resolution, prerequisite validation |
| [`site_config.py`](../../../syndiff_pipeline/star/site_config.py) | Load/merge `star_config.yaml` + `star_targets.csv` |
| [`identifiers.py`](../../../syndiff_pipeline/star/identifiers.py) | TIC/Gaia resolution, `identifier.json` |
| [`ps1_cache.py`](../../../syndiff_pipeline/star/ps1_cache.py) | `ps1_source` modes: zarr read, cache-on-miss, MAST stream |
| [`epsf_runner.py`](../../../syndiff_pipeline/star/epsf_runner.py) | Build/reuse gridded ePSFs on baseline diffs for gepsf stamp photometry |
| [`orchestration/stages.py`](../../../syndiff_pipeline/star/orchestration/stages.py) | HTCondor `star` stage spec (pool `star`; site default 8 CPU / 100 GB) |

## Per-host workflow

Before the host loop, each `psf_type: epsf` method resolves its required
baseline catalog from `inputs.epsf`. An enabled `epsf` block can build a
missing catalog; its `output` must match the referenced label. Without a build
block, the catalog must already exist.

For each host in the row's `stars_file`:

1. **Resolve** TIC or Gaia → sky position; write `identifier.json` and `host_gaia_row.csv`.
2. **Skip check** — if `gaia_source_id` is already in `ps1_removed_stars.csv`, status `skipped_already_removed`.
3. **PS1 segment** — find owning skycell via `master_pixels2skycells`, load r/i/z/y bands (`ps1_source` mode), SEP isolation, blend flag.
4. **Mini-template** — convolve star-only segment (Gaussian σ=60 PS1 px), downsample per template offset group from `cluster_template_job.json`.
5. **Diff stamp** — per FFI frame, reuse Hotpants kernel from `{baseline.diffs}_kernels/` (e.g. `hp_d_kernels`).
6. **Photometry** — forced photometry on stamps (`ap3`, `prf`, … from config).

Hosts with no usable SEP segment get status `skipped_no_segment`. Resolution failures get status `error`.

## Stamp formula

```text
stamp = FFI - (conv_temp - S_conv) - phot_bkg
```

| Term | Source | Notes |
|------|--------|-------|
| `FFI` | raw science from manifest | crop-local window around host |
| `conv_temp` | `baseline.convolved` (e.g. `hp_c`) | per-frame convolved template window |
| `S_conv` | mini-template convolved with same kernel | star-only component |
| `phot_bkg` | `baseline.phot_bkg` (e.g. `ks_b_s`) | photutils background from `kernel_subtract` / `background` |

**Not used:** `hp_b` (Hotpants internal background), `hp_d` diff image itself.

Kernels are read from `{baseline.diffs}_kernels/` (e.g. `hp_d_kernels/{product_id}_kernel.npz`). Convolution uses a margin-expanded window (`kernel_margin_px`, default 470) so kernel edge artifacts fall outside the final stamp (`stamp_size`, default 24).

## Coordinate frames

- Host position: full-FFI pixels via reference FFI WCS (`resolve_host_full_ffi_xy`).
- Stamp windows: **crop-local** pixels (same as diff stage).
- Mini-template downsample: registration indices are **full-FFI**; `XMIN`/`YMIN` in mini-template FITS are crop-local.

## Prerequisites

`validate_star_prerequisites` in `context.py` checks the baseline workspace before any host work:

| Artifact | Location | Produced by |
|----------|----------|-------------|
| `cluster_template_job.json` | `events/{label}/` | `wcs_grouping` |
| `syndiff_ffi_frames.csv` | `events/{label}/` | `wcs_grouping` |
| `syndiff_template_*` | `templates_dir` (symlink or `shifted_downsampled`) | `downsample` |
| Baseline diffs (`hp_d`, …) | `ws/` or `ws_{run_id}/` | `hotpants` |
| Convolved templates (`hp_c`) | same workspace | `hotpants` with `write_convolved: true` |
| Photutils background (`ks_b_s` / `ks_b`) | same workspace | `kernel_subtract` + optional `background` |
| Kernel solutions | `{diffs}_kernels/*_kernel.npz` | `hotpants` with `write_kernel_solutions: true` |
| `shared_mask.fits.gz` | workspace root | `shared_mask` |
| `{inputs.epsf}/gridded_epsf_index.json` + per-frame NPZ | baseline workspace; optionally built by matching `epsf.output` | star `epsf_runner` or prior diff `epsf` stage |
| Mapping CSV + master FITS | `data_root/skycell_pixel_mapping/…` | `mapping` |
| Gaia catalog CSV | `data_root/catalogs/…` | `mapping` |

Baseline workspace path: `events/{label}/ws/` when `baseline.workspace_run_id: none`, else `events/{label}/ws_{run_id}/`.

### Kernel backfill

Workspaces that completed Hotpants before `write_kernel_solutions` was enabled have `hp_d` and `hp_c` but no `hp_d_kernels/`. Run a one-time backfill with [`diff_config_star_full_backfill.yaml`](../../../config/diff_config_star_full_backfill.yaml):

- `workspace_inherit` from the existing multi-kernel workspace (`ks_b_s`, `shared_mask`, …)
- Single `hotpants` stage with `write_convolved: true` and `write_kernel_solutions: true`
- Default `workspace_run_id: star_full_lc` → outputs under `ws_star_full_lc/`

Point `star_config` / `star_targets` `baseline.workspace_run_id` at that suffix.

Site and example diff configs now set `write_kernel_solutions: true` on every active `hotpants` stage so new runs do not need backfill.

## PS1 ingest (`ps1_source`)

Implemented in `ps1_cache.py`; same Zarr layout as `ps1_download`:

```text
{data_root}/ps1_skycells.zarr/{projection}/{skycell}/{band, band_mask, band_wt}
```

The schema is shared, but the default paths are not: template `ps1_download`
writes `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`. To reuse that store,
set top-level `ps1_zarr_path` in `star_config.yaml` to its full path.

| Mode | Behavior |
|------|----------|
| `zarr_local_only` | Read shared store only; fail on miss |
| `zarr_download` | Download and cache on miss (default) |
| `stream` | MAST every time; no zarr write |

Optional `ps1_zarr_path` in `star_config.yaml` overrides the default store location. Legacy CLI values: `zarr` → `zarr_download`, `download` → `stream`.

For batch star runs after a normal `ps1_download` stage, point `ps1_zarr_path`
at that stage's store and prefer `zarr_local_only`. Without the override,
pre-populate the star-specific default store or use `zarr_download`. For
sector-wide astrometry campaigns that used `ps1_source: stream` in template
processing (see [`pipeline_multi_kernel_s20_astrometry.yaml`](../../../config/pipeline_multi_kernel_s20_astrometry.yaml)),
star can use `stream` or a pre-populated Zarr store.

## Outputs per Gaia host

Root: `{baseline_ws}/host_star/` (e.g. `ws_star_full_lc/host_star/`). Legacy sibling trees `events/{label}/star/` / `star_{id}/` remain readable for verify.

```text
{gaia_source_id}/
  identifier.json              # resolved TIC/Gaia, RA/Dec, resolution_method
  host_gaia_row.csv            # one-row Gaia-catalog-style record
  mini_templates/
    star_template_{id}_s{S}_{C}_{K}[_x…_y…]_dx{D}_dy{D}.fits.gz
  diff_stamps/
    {product_id}.fits.gz       # crop-local stamp; headers XMIN, YMIN, HOSTX, HOSTY
  lightcurve_{method}_gaia_{id}.csv
  plots/                       # when debug_plots: true
    ps1_segment_{skycell}.png
    mini_template_downsampled_dx{D}_dy{D}.png
    lightcurve_debug_gaia_{id}.png
batch_manifest.csv             # per-host status summary
```

### `batch_manifest.csv` columns

`gaia_source_id`, `tic_id`, `label`, `status`, `blend_flag`, `frames_processed`, `frames_failed`, `lightcurve_paths`, `error`

| `status` | Meaning |
|----------|---------|
| `ok` | At least one stamp succeeded; light curves written |
| `error` | Resolution failed or zero frames succeeded |
| `skipped_already_removed` | Host in `ps1_removed_stars.csv` |
| `skipped_no_segment` | No SEP segment or PS1 coverage |

Orchestrator verify requires every row `status=ok`.

## Photometry

Methods from `star_config.yaml` `photometry.methods`. Default: `ap3`
(aperture) + `prf` (TESS_PRF at host full-FFI position; requires `PRF`
package). `psf_type: epsf` performs gepsf fitting on each star stamp using the
per-frame catalog named by required `inputs.epsf`. Add a matching `epsf` block
only when star must build that catalog.

**Aperture CSV columns:** `btjd`, `flux`, `flux_wo_sky`, `sky`, `eflux`, `filename`, `group_id`, `xmin`, `ymin`, `host_x`, `host_y`

**PSF/PRF CSV columns:** `btjd`, `flux`, `eflux`, `filename`, `group_id`, `xmin`, `ymin`, `host_x`, `host_y`

## Related configs

| File | Role |
|------|------|
| [`star_config.yaml`](../../../config/star_config.yaml) | Site star policy |
| [`star_config_epsf_gepsf.yaml`](../../../config/star_config_epsf_gepsf.yaml) | Gridded-ePSF verification policy |
| [`star_targets_example.csv`](../../../config/star_targets_example.csv) | Example SCC registry |
| [`diff_config_star_full_backfill.yaml`](../../../config/diff_config_star_full_backfill.yaml) | One-time Hotpants kernel + convolved backfill |
| [`diff_config_multi_kernel.yaml`](../../../config/diff_config_multi_kernel.yaml) | Production multi-kernel diff (`hp_d`, `ks_b_s`, kernels); committed config does not write `hp_c` |
| [`pipeline_multi_kernel_s20_astrometry.yaml`](../../../config/pipeline_multi_kernel_s20_astrometry.yaml) | Sector-20 astrometry template+diff orchestrator (`ps1_source: stream`) |
| [`pipeline_epsf_gepsf.yaml`](../../../config/pipeline_epsf_gepsf.yaml) | 2020ut ePSF/gepsf diff-only orchestrator (transient LC recipe; star uses separate `star_config`) |

## Orchestration

- **Foreground:** `syndiff star run --target-name 20/3/2`
- **Batch:** `syndiff star submit --run-id star_lc_2026q1` (Condor or `--local`)
- Stage name: `star`, pool: `star`, deps: none (prerequisites verified on disk in `execute`)
- Monitor: `syndiff progress --run-id …`

Frozen run inputs: `runs/{run_id}/star_config.yaml`, `targets.csv` (from `star_targets`), `config.yaml` (from `pipeline.yaml`). Per-target logs under `per_target/{label}/star.*`.
