# Star configuration reference

Star uses three config surfaces (parallel to diff's `diff_config.yaml` + `targets.csv`):

| File | Role |
|------|------|
| [`star_config.yaml`](../../../config/star_config.yaml) | Site policy: defaults, baseline labels, photometry methods, SCC overrides |
| [`star_targets.csv`](../../../config/star_targets_example.csv) | One row per SCC/event to process |
| [`star_hosts/*.csv`](../../../config/star_hosts/) | Host list per event (`tic_id` / `gaia_source_id`) |

[`pipeline.yaml`](../../../config/pipeline.yaml) references `star_config` and `stages.star.executor` for batch runs.

## Merge precedence

**`star_targets` row > `star_config.overrides` > `star_config.defaults`**

CLI flags on `syndiff star run` override merged config for that foreground run.

## `star_config.yaml`

```yaml
deployment_file: deployment.yaml
# ps1_zarr_path: /path/to/custom/ps1_skycells.zarr   # rare override of shared store

defaults:
  cutout_size: 96
  stamp_size: 24
  kernel_margin_px: 470
  ps1_source: zarr_download   # zarr_local_only | zarr_download | stream
  debug_plots: true
  max_ffis: null              # truncate manifest for debug runs
  overwrite: false

baseline:
  workspace_run_id: none      # ws/ vs ws_{id}/; star writes to {baseline_ws}/host_star/
  diffs: hp_d                 # locates {diffs}_kernels/
  convolved: hp_c
  phot_bkg: ks_b_s            # subtract from raw FFI (NOT hp_b)

# Optional: build/reuse gridded ePSFs on baseline diffs for gepsf photometry.
epsf:
  enabled: true
  inputs:
    diffs: hp_d              # optional; defaults to baseline.diffs
  output: epsf_r1
  tile_nx: 2
  tile_ny: 2
  epsf_oversample: 4
  psf_size: 11
  extract_size: 11
  min_stars_per_tile: 5
  mag_max_rp: 12.95
  epsf_maxiters: 15
  epsf_recentering_maxiters: 20
  epsf_n_jobs: 8

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
    - name: gepsf
      type: psf
      psf_type: epsf
      inputs:
        epsf: epsf_r1        # required; must match epsf.output when building
      fit_shape: 11
      aperture_radius: 2
      psf_grouper_min_separation: 10

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
| `max_ffis` | `null` | Limit frames processed (debug) |
| `overwrite` | `false` | Recompute existing stamps |

Deprecated: `defaults.workspace_run_id` (formerly named sibling `star_{id}/` trees). Ignored for writes; still used only to locate legacy manifests during verify.

### `baseline.*` labels

| Key | Example | Purpose |
|-----|---------|---------|
| `workspace_run_id` | `star_full_lc` | Baseline diff under `ws_{id}/` (`none` → `ws/`); star outputs go in `{baseline_ws}/host_star/` |
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
| `psf` + `psf_type: epsf` | `inputs.epsf`; optional `fit_shape`, `aperture_radius`, `psf_grouper_min_separation` | same |

PRF photometry requires the `PRF` package and TESS PRF data (same as diff stage).
Gepsf photometry loads a per-frame `GriddedPSFModel` catalog from
`{baseline_workspace}/{photometry.inputs.epsf}`. To build a missing catalog,
add an enabled `epsf` block whose `output` matches that label; `epsf.inputs.diffs`
optionally selects the source baseline difference workspace and defaults to
`baseline.diffs`. The fit uses the SCC Gaia catalog and baseline shared mask.
If no `epsf` block is present, the referenced catalog must already exist.

### `ps1_source`

| Value | Network on miss | Zarr write |
|-------|-----------------|------------|
| `zarr_local_only` | No | No |
| `zarr_download` | Yes | Yes |
| `stream` | Always | No |

Legacy CLI values: `zarr` → `zarr_download`, `download` → `stream`.

Optional top-level `ps1_zarr_path` overrides the shared default
`{data_root}/ps1_skycells_zarr/ps1_skycells.zarr` (same store as template
`ps1_download`). Use only for unusual deployments.

## `star_targets.csv`

```csv
sector,camera,ccd,target_name,stars_file,baseline_workspace_run_id,baseline_diffs,baseline_convolved,phot_bkg,enabled
20,3,2,s20_astrometry,star_hosts/s20_c3_k2_example.csv,star_full_lc,hp_d,hp_c,ks_b_s,true
```

- `stars_file` resolves relative to the site directory.
- Row columns override policy defaults for that SCC only.
- `target_name` becomes the event label suffix (`s20_astrometry` → `s0020_c03_k02_s20_astrometry`).
- Separate from transient [`targets_example.csv`](../../../config/targets_example.csv).
- [`star_targets_full.csv`](../../../config/star_targets_full.csv) — production registry for larger campaigns.

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

## HTCondor resources

The `condor` block controls the independent `star` stage claim. The committed
site policy requests 8 CPUs and 100000 MB of memory. Keep `requirements`
consistent with `request_memory`; `--local` on `syndiff star submit` bypasses
Condor for a smoke test.

## Baseline workspace pairing

The committed `diff_config_multi_kernel.yaml` writes `hp_d`, kernels, and
backgrounds but has `write_convolved: false`; it does not by itself satisfy
star's `hp_c` prerequisite:

```text
ws_multi_hp_temp_calib/     # kernel_subtract → ks_b, background → ks_b_s
  hp_d/                     # baseline.diffs
  hp_d_kernels/             # write_kernel_solutions: true
  ks_b_s/                   # baseline.phot_bkg
```

After kernel/convolved backfill (`diff_config_star_full_backfill.yaml`):

```text
ws_star_full_lc/            # baseline.workspace_run_id: star_full_lc
  hp_d/, hp_c/, hp_d_kernels/, ks_b_s/
  host_star/                # syndiff star outputs (nested in baseline ws)
    batch_manifest.csv
    {gaia_source_id}/...
```

Set matching `baseline_workspace_run_id` in `star_targets` or `star_config.overrides`.
