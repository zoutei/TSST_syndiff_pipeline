# Template stage deep-dive documentation

These documents describe the **science algorithms** behind each pipeline stage. For running the scheduler, configuration, and HTCondor, see the [template pipeline guide](../template_pipeline.md) and [`syndiff` CLI](../syndiff_cli.md).

## Documents

| Document | Stage | Package module |
|----------|-------|----------------|
| [Field (distortion-aware) templates](../field_geometry.md) | `remap` + `downsample` (default `geometry_mode: field`) | `field_remap.py`, `field_downsample.py`, … |
| [Oversampled templates + Hotpants stamp modes](../oversampled_templates.md) | `mapping` / `downsample` / `diff` / `star` | `hotpants.py`, `kernel.py`, … |
| [TESS FFI download](tess_ffi_download.md) | `tess_ffi_download` | `common/download.py` |
| [WCS grouping](wcs_grouping.md) | Linear-mode drift algorithms (config `stages.wcs_grouping`; not a scheduler stage) | `common/wcs_grouping.py` |
| [PanCAKES mapping](mapping_pancakes.md) | `mapping` | `template_creation/processing/pancakes.py` |
| [PS1 download](ps1_download.md) | `ps1_download` | `template_creation/processing/ps1_download.py` |
| [PS1 process (technical)](ps1_process_technical.md) | `ps1_process` | `template_creation/processing/ps1_process.py` |
| [Multi-offset downsample](downsample_technical.md) | `downsample` (product path `templates/`) | `field_downsample.py`, `linear_downsample.py` |
| [Diff pipeline internals](diff_pipeline.md) | `diff` | `difference_imaging/orchestration/execute.py` + `stages/` + `masking/` |
| [Multi-kernel diff path](multi_kernel_diff.md) | `kernel_fit` → `convolved_templates` → `kernel_subtract` | `kernel_*.py` |
| [Gridded ePSF](gridded_epsf.md) | `epsf` (gridded models) | `gridded_epsf.py` |
| [Centroids](centroids.md) | `centroids` | `centroids.py` |
| [Forced photometry](forced_photometry.md) | `forced_photometry` (photometry pipeline kind) | `difference_imaging/stages/photometry.py` |
| [Static masking](../masking.md) | `shared_mask` | `difference_imaging/masking/` |
| [Event photometry](../photometry.md) | `photometry` | `photometry/cli.py` |
| [Photometry pipeline (technical)](photometry_pipeline.md) | `photometry` | `photometry/runner.py` |
| [Photometry configuration](photometry_config.md) | — | `photometry_config.yaml` |
| [Host-star light curves](../star_lightcurves.md) | `star` | `star/cli.py` |
| [Star pipeline (technical)](star_pipeline.md) | `star` | `syndiff_pipeline/star/` |
| [Star configuration](star_config.md) | — | `star_config.yaml` |
| [Unified background stage](background.md) | `background` | `difference_imaging/stages/background/` |
| [Standalone pipeline overview](standalone_pipeline_overview.md) | **LEGACY** | — |

## Typical data flow

```text
tess_ffi_download          →  FFI FITS ({data_root}/s{SSSS}/c{C}/k{K}/ffi/)
mapping (PanCAKES)         →  {data_root}/s{SSSS}/c{C}/k{K}/mapping/oversampling_{N}/
remap (field mode)         →  {data_root}/s{SSSS}/c{C}/k{K}/remap/oversampling_{N}/
ps1_download               →  {data_root}/ps1_skycells_zarr/ps1_skycells.zarr
ps1_process                →  {data_root}/s{SSSS}/c{C}/k{K}/convolved.zarr
downsample                 →  {data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/
diff (scc_bootstrap)       →  {data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/ + bookkeeping/diff/
photometry (after diff)    →  {workspace_root}/events/{event}/s…/phot_{run_id}/
star (after diff verify)   →  host_star/ under baseline workspace or photometry tree
```

## WCS drift (field vs linear)

Field mode measures drift during **`remap`** (see [field_geometry.md](../field_geometry.md)). Linear mode uses point-drift grouping documented in [WCS grouping](wcs_grouping.md) and `linear_downsample.py`. Diff field mode uses **`scc_bootstrap`** inside `diff` execute — not a separate scheduler stage.

## Diff imaging

**`diff`** runs the config-driven internal pipeline from `pipeline.yaml`'s embedded `diff:` block (schema v2 — see [`config/pipeline.yaml`](../../../config/pipeline.yaml) and [config_schema_v2.md](../config_schema_v2.md)) or a legacy standalone [`config/diff_config.yaml`](../../../config/diff_config.yaml) (schema v1 default: `shared_mask` + `hotpants`). See [diff pipeline internals](diff_pipeline.md). Multi-kernel path: [multi_kernel_diff.md](multi_kernel_diff.md). Masks: [masking.md](../masking.md). Oversampling: [oversampled templates](../oversampled_templates.md).

## Event photometry and host stars

- Photometry: [photometry.md](../photometry.md), [photometry_pipeline.md](photometry_pipeline.md), [photometry_config.md](photometry_config.md)
- Star: [star_lightcurves.md](../star_lightcurves.md), [star_pipeline.md](star_pipeline.md), [star_config.md](star_config.md)
