# Template stage deep-dive documentation

These documents describe the **science algorithms** behind each template pipeline stage. For running the multi-target scheduler, configuration, and HTCondor, see the [template pipeline guide](../template_pipeline.md).

## Documents

| Document | Stage | Legacy script | Package module |
|----------|-------|---------------|----------------|
| [Standalone pipeline overview](standalone_pipeline_overview.md) | All four core steps | `pipeline.py` | — |
| — | `tess_ffi_download` | — | `common/download.py` |
| — | `wcs_grouping` | — | `template_creation/orchestration/handoff.py` + `common/wcs_grouping.py` |
| [PanCAKES mapping](mapping_pancakes.md) | `mapping` | `pancakes_v2.py` | `template_creation/processing/pancakes.py` |
| [PS1 process (technical)](ps1_process_technical.md) | `ps1_process` | `process_ps1.py` | `template_creation/processing/ps1_process.py` |
| [Multi-offset downsample](downsample_technical.md) | `downsample` | `multi_offset_downsampling.py` | `template_creation/processing/downsample.py` |
| — | `diff` | — | `difference_imaging/orchestration/execute.py` (see [`config/diff_config.yaml`](../../config/diff_config.yaml)) |

## PS1 download (no separate deep-dive)

The **`ps1_download`** stage wraps `download_and_store_zarr.py` logic in `template_creation/processing/ps1_download.py`. See [standalone pipeline overview — Download PS1 Data](standalone_pipeline_overview.md#2-download-ps1-data) for CLI options and the shared Zarr layout.

Key points:

- One shared store per `data_root`: `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`
- File lock serializes concurrent writers across SCCs
- Reads skycell names from the mapping stage CSV

## TESS FFI download (runner-only)

There is no legacy `syndiff/` script — **`tess_ffi_download`** fetches sector FFI FITS via `common/download.py` before WCS grouping.

## WCS grouping (runner-only)

There is no legacy `syndiff/` script — **`wcs_grouping`** was added for the SynDiff template runner. It uses `syndiff_pipeline.common.wcs_grouping` via `template_creation/orchestration/handoff.py` to:

1. Select FFIs where the transient has valid WCS
2. Smooth pixel drift and assign template offset groups
3. Write `cluster_template_job.json`, `wcs_drift_template_debug.png`, and `syndiff_ffi_frames.csv` under `{workspace_root}/events/{target_label}/`

Downsample reads crop bounds and offset list from `cluster_template_job.json`.

## Diff imaging (orchestration docs)

There is no standalone algorithm deep-dive — **`diff`** runs the Hotpants → photometry pipeline from [`config/diff_config.yaml`](../../config/diff_config.yaml). Stage lists, SCC overrides, and Condor settings are documented in the [template pipeline guide](../template_pipeline.md) and [`config/README.md`](../../config/README.md).

## Typical data flow

```text
tess_ffi_download          →  FFI FITS on disk ({data_root}/tess_ffi/)
wcs_grouping               →  {workspace_root}/events/{target_label}/cluster_template_job.json
mapping (PanCAKES)         →  data_root/skycell_pixel_mapping/sector_*/camera_*/ccd_*/tess_s*_master_skycells_list.csv
ps1_download               →  data_root/ps1_skycells_zarr/ps1_skycells.zarr
ps1_process                →  data_root/convolved_results/sector_*_camera_*_ccd_*.zarr
downsample                 →  data_root/shifted_downsampled/.../syndiff_template_*.fits.gz
diff                       →  {workspace_root}/events/{target_label}/ws/{workspace_label}/
```

## Provenance

These files are copies of the step READMEs from the sibling [`syndiff`](../../../syndiff/) repository, imported into `syndiff-pipeline` for the open-source release. When updating algorithm documentation, edit both locations or consolidate here and treat `syndiff/` as the development sandbox.
