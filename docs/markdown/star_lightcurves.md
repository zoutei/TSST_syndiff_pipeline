# Host-star light curves (`syndiff star`)

Produce forced-photometry light curves for TIC/Gaia host stars in an **already-existing** syndiff event. Star consumes transient diff outputs; it does **not** re-run Hotpants (and therefore has no Hotpants `stamp_mode`).

Configuration is separate from the transient pipeline: see [star_config.md](stages/star_config.md) and the technical deep-dive [star_pipeline.md](stages/star_pipeline.md). When the baseline templates were built at `oversampling_factor F>1`, set the same `defaults.oversampling_factor` in `star_config.yaml` — details in [oversampled templates §10](oversampled_templates.md#10-star-branch).

## Quick start

**Foreground (one SCC):**

```bash
syndiff star run \
  --site config \
  --star-config config/star_config.yaml \
  --star-targets config/star_targets_example.csv \
  --target-name 20/3/2
```

**Batch (all enabled rows in `star_targets.csv`):**

```bash
syndiff star submit \
  --site config \
  --star-config config/star_config.yaml \
  --star-targets config/star_targets_example.csv \
  --run-id star_lc_2026q1
```

Use `--local` on submit for local executor instead of Condor. Monitor with `syndiff progress --run-id star_lc_2026q1`.

## Prerequisites

Transient template + diff must have completed for the baseline workspace referenced in `star_targets` / `star_config`. Star validates these on disk before processing hosts:

| Artifact | Typical location | Stage |
|----------|------------------|-------|
| `bookkeeping/diff/frames.csv` | `{data_root}/s{SSSS}/c{C}/k{K}/` | `diff` (`scc_bootstrap`) |
| `bookkeeping/diff/diff_job.json` | `{data_root}/s{SSSS}/c{C}/k{K}/` | `diff` (`scc_bootstrap`) |
| Field template store | `templates/oversampling_{N}/` or `templates_{lane}/` | `downsample` |
| Convolved template | `hp_c` | `hotpants` (`write_convolved: true`) |
| Kernel solutions | `hp_d_kernels/*_kernel.npz` | `hotpants` (`write_kernel_solutions: true`) |
| Photutils background | `ks_b_s` or `ks_b` | `kernel_subtract` + `background` |
| Shared mask | `shared_mask.fits.fz` | `shared_mask` |
| Mapping + Gaia catalog | `data_root/s{SSSS}/c{C}/k{K}/mapping/oversampling_{N}/…`, `.../catalogs/…` | `mapping` |

Star subtracts **`phot_bkg`** (e.g. `ks_b_s`), not Hotpants `hp_b`. Baseline workspace: `ws/` when `baseline.workspace_run_id: none`, else `ws_{run_id}/`.

### Kernel backfill

Older workspaces may have `hp_d` without `hp_d_kernels/`. Run a one-time Hotpants backfill:

```bash
syndiff diff run \
  --config config/diff_config_star_full_backfill.yaml \
  --deployment config/deployment.yaml \
  --targets config/targets_example.csv \
  --target-name 20/3/2
```

`--config` is the diff policy only when `--site` is omitted; with `--site`,
foreground diff always reads `<site>/diff_config.yaml`. Then set
`baseline.workspace_run_id: star_full_lc` (or your `workspace_run_id`) in
`star_config` / `star_targets`. See [star_pipeline.md](stages/star_pipeline.md)
and [`config/diff_config_star_full_backfill.yaml`](../../config/diff_config_star_full_backfill.yaml).

## Config files

| File | Purpose |
|------|---------|
| `config/star_config.yaml` | Defaults, baseline labels, photometry methods, SCC overrides |
| `config/star_config_epsf_gepsf.yaml` | Example/verification policy for baseline gridded-ePSF fitting and gepsf stamp photometry |
| `config/star_targets_example.csv` | Example SCC registry + per-row baseline overrides |
| `config/star_targets_full.csv` | Production SCC registry |
| `config/star_hosts/*.csv` | Host lists (`tic_id` / `gaia_source_id`) |

Merge order: **star_targets row > overrides > defaults**.

Related transient orchestrators (produce the baseline diff workspace star reads):

| File | Purpose |
|------|---------|
| `config/diff_config_multi_kernel.yaml` | Multi-kernel diff (`hp_d`, `hp_c`, `ks_b_s`, kernels) |
| `config/pipeline_multi_kernel_s20_astrometry.yaml` | Sector-20 astrometry template+diff (`ps1_source: stream`) |
| `config/pipeline_epsf_gepsf.yaml` | 2020ut ePSF/gepsf diff-only orchestrator |

## `ps1_source`

Controls how star loads PS1 skycells for mini-template isolation:

| Mode | When to use |
|------|-------------|
| `zarr_download` | Default; download on cache miss to `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr` |
| `zarr_local_only` | Batch runs after `ps1_download` (or with a pre-populated shared store) |
| `stream` | Always fetch from MAST (no zarr write); matches sector-20 stream template runs |

Star and template `ps1_download` share
`{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`. Optional top-level
`ps1_zarr_path` in `star_config.yaml` overrides that path for unusual
deployments.

## Photometry

Methods come from `star_config.yaml` `photometry.methods` (default: `ap3` +
`prf`). PRF uses TESS_PRF at the host's full-FFI pixel position (requires the
`PRF` package).

For spatially varying empirical-PSF photometry, add a method with `type: psf`,
`psf_type: epsf`, and required `inputs.epsf: <label>`. If the baseline
workspace does not already contain that catalog, add an enabled `epsf` block
whose `output` matches the label; `epsf.inputs.diffs` optionally chooses the
source baseline diffs. Star then fits the matching per-frame model on each
diff stamp. See `config/star_config_epsf_gepsf.yaml`.

CLI flags on `syndiff star run` override merged config: `--stars-file`, `--baseline-workspace-run-id`, `--baseline-diffs-label`, `--baseline-convolved-label`, `--baseline-phot-bkg-label`, `--cutout-size`, `--stamp-size`, `--ps1-source`, `--overwrite`, `--debug-plots`.

## Outputs

Under `{baseline_ws}/host_star/` (e.g. `events/{event_name}/s{SSSS}_c{C}_k{K}/ws_star_full_lc/host_star/` when `baseline.workspace_run_id: star_full_lc`, or `events/{event_name}/s{SSSS}_c{C}_k{K}/ws/host_star/` when baseline is `none`):

```text
{gaia_source_id}/
  identifier.json
  host_gaia_row.csv
  mini_templates/
    star_template_{id}_s{S}_{C}_{K}[_x…_y…]_dx{D}_dy{D}.fits.fz
  diff_stamps/
    {product_id}.fits.fz
  lightcurve_ap3_gaia_{id}.csv
  lightcurve_prf_gaia_{id}.csv
  plots/                         # when debug_plots: true
batch_manifest.csv               # per-host status (ok | error | skipped_*)
```

Legacy sibling trees `events/{label}/star/` and `star_{id}/` are still accepted by verify when `host_star/` is absent.

## Stamp formula

```text
stamp = FFI - (conv_temp - S_conv) - phot_bkg
```

- `conv_temp` from `hp_c` (or `baseline.convolved`)
- `S_conv` = mini star-template convolved with the per-frame Hotpants kernel from `hp_d_kernels/`
- `phot_bkg` from `ks_b_s` (or `ks_b`)

See [star_pipeline.md](stages/star_pipeline.md) for coordinate frames, module layout, and orchestration details.
