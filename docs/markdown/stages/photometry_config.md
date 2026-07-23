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
| `condor` | no | `request_cpus`, `request_memory`, `requirements`, `rank` |

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

Every `psf_type: epsf` method needs a resolvable gridded ePSF catalog (`gridded_epsf_index.json`) under the named ePSF label on the SCC diff lane.

## Forced-target extras

`additional_forced_targets` and `per_event_force_targets[event_label]` are concatenated, then normalized (`normalize_additional_forced_targets`). Each entry typically has `name`, `position_mode` (`sky` or pixel offsets), and either `ra`/`dec` or `dx`/`dy` relative to the primary target.

## Frozen run copy

`syndiff photometry submit` freezes the photometry policy under `runs/{run_id}/photometry_config.yaml`. Debug with that file plus `run_meta.json`, not only the live site YAML.

## Orchestrator wiring

`pipeline.yaml` still owns `stages.photometry` executor / Condor pool settings. Photometry science knobs stay in `photometry_config.yaml` — same split as `diff_config.yaml` vs `pipeline.yaml` for the diff stage.
