# Template stage deep-dive documentation

These documents describe the **science algorithms** behind each template pipeline stage. For running the multi-target scheduler, configuration, and HTCondor, see the [template pipeline guide](../template_pipeline.md).

## Documents

| Document | Stage | Legacy script | Package module |
|----------|-------|---------------|----------------|
| [Field (distortion-aware) templates](../field_geometry.md) | `remap` + `downsample` (**default** `geometry_mode: field`) | — | `field_remap.py`, `field_downsample.py`, … |
| [Oversampled templates + Hotpants stamp modes](../oversampled_templates.md) | `mapping` / `templates` / `diff` / `star` | — | `template_coverage.py`, `hotpants.py`, `kernel.py`, `star/*` |
| [Standalone pipeline overview](standalone_pipeline_overview.md) | All four core steps | `pipeline.py` | — |
| [TESS FFI download](tess_ffi_download.md) | `tess_ffi_download` | — | `common/download.py` |
| [WCS grouping](wcs_grouping.md) | `wcs_grouping` | — | `template_creation/orchestration/handoff.py` + `common/wcs_grouping.py` |
| [PanCAKES mapping](mapping_pancakes.md) | `mapping` | `pancakes_v2.py` | `template_creation/processing/pancakes.py` |
| [PS1 process (technical)](ps1_process_technical.md) | `ps1_process` | `process_ps1.py` | `template_creation/processing/ps1_process.py` |
| [Multi-offset downsample](downsample_technical.md) | `templates` (linear `geometry_mode`) | `multi_offset_downsampling.py` | `template_creation/processing/downsample.py` |
| [Diff pipeline internals](diff_pipeline.md) | `diff` | — | `difference_imaging/orchestration/execute.py` + `difference_imaging/stages/` |
| [Forced photometry](forced_photometry.md) | `forced_photometry` (diff sub-stage) | — | `difference_imaging/stages/photometry.py` |
| [Host-star light curves](../star_lightcurves.md) | `star` | — | `syndiff_pipeline/star/cli.py` |
| [Star pipeline (technical)](star_pipeline.md) | `star` | — | `syndiff_pipeline/star/` |
| [Star configuration](star_config.md) | — | — | `config/star_config.yaml`, `star_targets.csv` |
| [Unified background stage](background.md) | `background` (diff sub-stage) | — | `difference_imaging/stages/background/` |

## PS1 download (no separate deep-dive)

The **`ps1_download`** stage wraps `download_and_store_zarr.py` logic in `template_creation/processing/ps1_download.py`. See [standalone pipeline overview — Download PS1 Data](standalone_pipeline_overview.md#2-download-ps1-data) for CLI options and the shared Zarr layout.

Key points:

- One shared store per `data_root`: `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`
- File lock serializes concurrent writers across SCCs
- Reads skycell names from the mapping stage CSV

## TESS FFI download (runner-only)

There is no legacy `syndiff/` script — **`tess_ffi_download`** fetches sector FFI FITS via `common/download.py` before WCS grouping.

## WCS grouping (runner-only)

There is no legacy `syndiff/` script — **`wcs_grouping`** was added for the SynDiff template runner. See the [WCS grouping deep-dive](wcs_grouping.md) for drift measurement, smoothing, reference-FFI selection, template groups, crop bounds, and output schemas. Downsample reads crop bounds and offset list from `cluster_template_job.json`.

## Diff imaging

**`diff`** runs the config-driven internal pipeline from [`config/diff_config.yaml`](../../config/diff_config.yaml). See the [diff pipeline internals](diff_pipeline.md) for all sub-stage kinds (shared_mask, hotpants, kernel_fit, convolved_templates, kernel_subtract, epsf, sat_template, subtract, background, forced_photometry), workspace naming, template resolution, and kernel persistence. Forced-photometry modes and parameters: [forced_photometry.md](forced_photometry.md). For oversampled templates (`F>1`) and Hotpants `stamp_mode` / `region_*`, see [oversampled templates](../oversampled_templates.md). Orchestration, SCC overrides, and Condor settings are in the [template pipeline guide](../template_pipeline.md) and [`config/README.md`](../../config/README.md).

## Host-star light curves

**`star`** is a third pipeline branch after template + diff. It reads per-frame Hotpants kernels (`hp_d_kernels`), convolved templates (`hp_c`), and photutils background (`ks_b_s` / `ks_b`) from an existing transient workspace — it does **not** re-run Hotpants.

- Quick start: [Host-star light curves](../star_lightcurves.md)
- Algorithms: [Star pipeline (technical)](star_pipeline.md)
- Config schema: [Star configuration](star_config.md)
- Kernel backfill for older workspaces: [`config/diff_config_star_full_backfill.yaml`](../../config/diff_config_star_full_backfill.yaml)

**Sub-stage deep-dives:**

| Document | Stage |
|----------|-------|
| [Unified background stage](background.md) | `background` — spatial / temporal Savitzky–Golay / strap on `ks_b` → `ks_b_s` |
| [Forced photometry](forced_photometry.md) | `forced_photometry` — aperture / PRF / ePSF photutils / ePSF tessreduce |

## Typical data flow

```text
tess_ffi_download          →  FFI FITS on disk ({data_root}/tess_ffi/)
wcs_grouping               →  {workspace_root}/events/{target_label}/cluster_template_job.json
mapping (PanCAKES)         →  data_root/skycell_pixel_mapping/sector_*/camera_*/ccd_*/tess_s*_master_skycells_list.csv
ps1_download               →  data_root/ps1_skycells_zarr/ps1_skycells.zarr
ps1_process                →  data_root/convolved_results/sector_*_camera_*_ccd_*.zarr
downsample                 →  data_root/shifted_downsampled/.../syndiff_template_*.fits.fz
diff                       →  {workspace_root}/events/{target_label}/ws/{workspace_label}/
star (after diff verify)   →  {workspace_root}/events/{target_label}/star[_run_id]/{gaia_source_id}/
```

## Provenance

These files are copies of the step READMEs from the sibling [`syndiff`](../../../syndiff/) repository, imported into `syndiff-pipeline` for the open-source release. When updating algorithm documentation, edit both locations or consolidate here and treat `syndiff/` as the development sandbox.
