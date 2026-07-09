# Star configuration reference

Star uses three config surfaces (parallel to diff's `diff_config.yaml` + `targets.csv`):

| File | Role |
|------|------|
| [`star_config.yaml`](../../config/star_config.yaml) | Site policy: defaults, baseline labels, photometry methods, SCC overrides |
| [`star_targets.csv`](../../config/star_targets_example.csv) | One row per SCC/event to process |
| [`star_hosts/*.csv`](../../config/star_hosts/) | Host list per event (`tic_id` / `gaia_source_id`) |

[`pipeline.yaml`](../../config/pipeline.yaml) references `star_config` and `stages.star.executor` for batch runs.

## Merge precedence

**`star_targets` row > `star_config.overrides` > `star_config.defaults`**

CLI flags on `syndiff star run` override merged config for that foreground run.

## `star_config.yaml`

```yaml
deployment_file: deployment.yaml
# ps1_zarr_path: /path/to/custom/ps1_skycells.zarr   # optional override

defaults:
  cutout_size: 96
  stamp_size: 24
  kernel_margin_px: 470
  ps1_source: zarr_download   # zarr_local_only | zarr_download | stream
  debug_plots: true
  workspace_run_id: null      # null → events/{label}/star/
  max_ffis: null              # truncate manifest for debug runs
  overwrite: false

baseline:
  workspace_run_id: none      # ws/ vs ws_{id}/
  diffs: hp_d                 # locates {diffs}_kernels/
  convolved: hp_c
  phot_bkg: ks_b_s            # subtract from raw FFI (NOT hp_b)

photometry:
  methods:
    - name: ap3
      type: aperture
      tar_ap: 3
      sky_in: 5
      sky_out: 9
    - name: prf
      type: psf
      psf_type: prf

overrides:
  "20/3/2":
    baseline:
      workspace_run_id: star_full_lc
      phot_bkg: ks_b_s
```

### `defaults.*`

| Key | Default | Purpose |
|-----|---------|---------|
| `cutout_size` | 96 | Mini-template ROI side length (full-FFI pixels) |
| `stamp_size` | 24 | Final diff-stamp window side (crop-local pixels) |
| `kernel_margin_px` | 470 | Convolution margin around stamp (Hotpants kernel extent) |
| `ps1_source` | `zarr_download` | PS1 skycell ingest mode (see below) |
| `debug_plots` | `true` | Write segment / downsample / LC debug PNGs |
| `workspace_run_id` | `null` | Star output suffix: `star/` vs `star_{id}/` |
| `max_ffis` | `null` | Limit frames processed (debug) |
| `overwrite` | `false` | Recompute existing stamps |

### `baseline.*` labels

| Key | Example | Purpose |
|-----|---------|---------|
| `workspace_run_id` | `star_full_lc` | Baseline diff under `ws_{id}/` (`none` → `ws/`) |
| `diffs` | `hp_d` | Hotpants diffs label; kernels at `hp_d_kernels/` |
| `convolved` | `hp_c` | Per-frame convolved template windows |
| `phot_bkg` | `ks_b_s` or `ks_b` | Photutils background subtracted in stamp |

- `ks_b` — raw photutils map from `kernel_subtract`
- `ks_b_s` — temporally smoothed (`background` stage); typical choice for star stamps
- `hp_b` — Hotpants internal background; **not used by star**

If `phot_bkg` is unset, star falls back to reading `inputs.bkg` from the workspace/site `diff_config.yaml`, then probes `ks_b_s` / `ks_b` on disk.

### `photometry.methods`

Same shape as diff `forced_photometry` methods. Each entry needs unique `name` (`[a-z0-9_]+`) and `type`:

| `type` | Required fields | Output CSV |
|--------|-----------------|------------|
| `aperture` | `tar_ap`, `sky_in`, `sky_out` | `lightcurve_{name}_gaia_{id}.csv` |
| `psf` + `psf_type: prf` | (built at runtime from TESS_PRF) | same |

PRF photometry requires the `PRF` package and TESS PRF data (same as diff stage).

### `ps1_source`

| Value | Network on miss | Zarr write |
|-------|-----------------|------------|
| `zarr_local_only` | No | No |
| `zarr_download` | Yes | Yes |
| `stream` | Always | No |

Legacy CLI values: `zarr` → `zarr_download`, `download` → `stream`.

Optional top-level `ps1_zarr_path` overrides `{data_root}/ps1_skycells.zarr`.

## `star_targets.csv`

```csv
sector,camera,ccd,target_name,stars_file,baseline_workspace_run_id,baseline_diffs,baseline_convolved,phot_bkg,enabled
20,3,2,s20_astrometry,star_hosts/s20_c3_k2_example.csv,star_full_lc,hp_d,hp_c,ks_b_s,true
```

- `stars_file` resolves relative to the site directory.
- Row columns override policy defaults for that SCC only.
- `target_name` becomes the event label suffix (`s20_astrometry` → `s0020_c03_k02_s20_astrometry`).
- Separate from transient [`targets_example.csv`](../../config/targets_example.csv).
- [`star_targets_full.csv`](../../config/star_targets_full.csv) — production registry for larger campaigns.

## `star_hosts/*.csv`

```csv
tic_id,gaia_source_id,label
142748283,,
,12345678901234567,host_a
```

Exactly one of `tic_id` or `gaia_source_id` per row. Optional `label` for logging only.

**TIC resolution:** MAST TIC query → Gaia DR3 id from TIC row, or nearest Gaia match within 1″ of TIC position. **Gaia resolution:** local SCC catalog first, then remote Gaia DR3 if missing.

## Frozen run inputs

On `syndiff star submit`, the orchestrator copies into `{workspace_root}/runs/{run_id}/`:

- `config.yaml` — frozen `pipeline.yaml`
- `targets.csv` — frozen `star_targets.csv`
- `star_config.yaml` — frozen site policy

Per-target stage logs and manifests live under `per_target/{label}/star.*`.

## Baseline workspace pairing

Typical multi-kernel transient run (`diff_config_multi_kernel.yaml`):

```text
ws_multi_hp_temp_calib/     # kernel_subtract → ks_b, background → ks_b_s
  hp_d/                     # baseline.diffs
  hp_c/                     # baseline.convolved (write_convolved: true)
  hp_d_kernels/             # write_kernel_solutions: true
  ks_b_s/                   # baseline.phot_bkg
```

After kernel backfill (`diff_config_star_full_backfill.yaml`):

```text
ws_star_full_lc/            # baseline.workspace_run_id: star_full_lc
  hp_d/, hp_c/, hp_d_kernels/, ks_b_s/
```

Set matching `baseline_workspace_run_id` in `star_targets` or `star_config.overrides`.
