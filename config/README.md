# SynDiff configuration

This directory is the **config root** passed to `syndiff --site config`.

## Files

| File | Role |
|------|------|
| `pipeline.yaml` | Orchestrator policy: stage DAG params, resource pools, scheduler, notifications |
| `diff_config.yaml` | Site diff policy: `pipeline:` stage list, defaults (`n_jobs`), SCC overrides, Condor |
| `mask_settings.example.yaml` | Copy to site `mask_settings.yaml` for empirical/TNS/asteroid mask policy (`difference_imaging/masking`) |
| `diff_config_multi_kernel.yaml` | Multi-kernel diff (`hp_d`, `ks_b_s`, per-frame kernels; `write_convolved: false`) |
| `diff_config_star_full_backfill.yaml` | One-time Hotpants backfill (`write_convolved` + `write_kernel_solutions`) for star |
| `star_config.yaml` | Site star policy: baseline labels, photometry, `ps1_source`, SCC overrides |
| `star_config_epsf_gepsf.yaml` | Star gridded-ePSF build/reuse and gepsf photometry example |
| `star_targets_example.csv` | Example star SCC registry for `syndiff star submit` |
| `star_targets_full.csv` | Production star SCC registry |
| `star_hosts/` | Per-event host star CSVs referenced from `star_targets` |
| `pipeline_multi_kernel_s20_astrometry.yaml` | Sector-20 astrometry template+diff orchestrator (`ps1_source: stream`) |
| `pipeline_epsf_gepsf.yaml` | 2020ut ePSF/gepsf diff-only orchestrator |
| `deployment.yaml` | Gitignored: `workspace_root`, `data_root`, credentials (copy from `deployment.yaml.example`) |
| `targets_example.csv` | Example targets list for `--targets` |

## Foreground diff (two entry points)

| Path | When to use | Command |
|------|-------------|---------|
| **Site policy** | Normal debugging with live `diff_config.yaml` | `syndiff diff run --site config --targets targets_example.csv --target-name 2020ut` |
| **Alternate diff policy** | Foreground run with a selected diff YAML | `syndiff diff run --config diff_config_star_full_backfill.yaml --deployment deployment.yaml --targets targets_example.csv --target-name 20/3/2` |
| **Materialized YAML** | Frozen per-target config with absolute paths | `python -m syndiff_pipeline.difference_imaging.orchestration.cli --config example/diff_config_a_prf.yaml` |

With `--site`, foreground diff always reads `<site>/diff_config.yaml`; the
`diff_config:` selected by `pipeline.yaml` is used by supervised submit. Omit
`--site` and pass `--config` + `--deployment` to choose another foreground
diff policy. Materialized examples live under `example/diff_config_*.yaml`;
legacy names are in `example/legacy/recipe_*.yaml` (read-only reference).

## Supervised pipeline

```bash
cp config/deployment.yaml.example config/deployment.yaml   # first time
syndiff all submit --site config --targets config/targets_example.csv --run-id my_run
```

**Host-star light curves** (after transient diff artifacts exist on disk):

```bash
syndiff star submit --site config \
  --star-config config/star_config.yaml \
  --star-targets config/star_targets_example.csv \
  --run-id star_lc_run
```

See [docs/markdown/star_lightcurves.md](../docs/markdown/star_lightcurves.md) for prerequisites (`hp_c`, `hp_d_kernels`, `ks_b_s`), kernel backfill, and outputs.

### Related orchestrator configs

| File | Role |
|------|------|
| `pipeline_multi_kernel_s20_astrometry.yaml` | Sector-20 astrometry: `ps1_source: stream` in `ps1_process`, multi-kernel diff |
| `pipeline_epsf_gepsf.yaml` | 2020ut gridded ePSF + gepsf forced photometry (diff-only submit) |

These produce the transient baseline workspace that `syndiff star` reads. Pair with `diff_config_multi_kernel.yaml` or site `diff_config.yaml` as noted in each file's header comments.

## Runtime frozen configs

On submit, the orchestrator copies policy into the workspace:

- `{workspace_root}/runs/{run_id}/config.yaml` — frozen orchestrator
- `{workspace_root}/runs/{run_id}/targets.csv` — frozen targets (`targets.csv` or `star_targets.csv`)
- `{workspace_root}/runs/{run_id}/star_config.yaml` — frozen star policy (star submit only)
- `{workspace_root}/runs/{run_id}/per_target/{label}/diff_config.yaml` — frozen per-target diff

See [docs/markdown/storage_layout.md](../docs/markdown/storage_layout.md).

## `forced_photometry` methods

The `forced_photometry` stage uses a **`methods`** list. Each entry has a unique
`name` (slug `[a-z0-9_]+`) and `type` (`psf` or `aperture`). The stage writes one
CSV per method per target under `ws/<output>/`.

```yaml
- kind: forced_photometry
  inputs:
    diffs: hp_d
  output: lc_photometry
  methods:
    - name: prf
      type: psf
      psf_type: prf
    - name: ap3
      type: aperture
      tar_ap: 3
      sky_in: 5
      sky_out: 9
```

**CSV names:** primary → `lightcurve_{name}.csv`; extra targets from
`additional_forced_targets` → `lightcurve_{name}_{target}.csv`.

**PSF columns:** `btjd`, `flux`, `eflux`, `filename`, `group_id`.

**Aperture columns:** same metadata plus `flux` (raw sum with sky), `flux_wo_sky`
(sky-subtracted, primary science column), `sky`, and `eflux` (uncertainty on
`flux_wo_sky`). Defaults match TESSreduce `diff_lc`: `tar_ap=3`, `sky_in=5`,
`sky_out=9`.

Top-level `psf_type` is no longer supported; migrate existing configs to a
`methods` entry with `type: psf`.

## `background` (spatial → temporal → strap)

Inserted after `kernel_subtract` in [`diff_config_single_kernel.yaml`](diff_config_single_kernel.yaml) and [`diff_config_multi_kernel.yaml`](diff_config_multi_kernel.yaml). Temporally smooths the full-crop photutils background cube (`ks_b`) with Savitzky–Golay when `steps.temporal.enabled: true`; writes `ks_b_s`. Pair with a `subtract` stage for resubtracted diffs.

```yaml
- kind: background
  inputs:
    bkg_in: ks_b
  output: ks_b_s
  steps:
    spatial: { enabled: false }
    temporal: { enabled: true, method: savgol, tile_size: 256 }
    strap: { enabled: false }
- kind: subtract
  inputs:
    expression: "ks_d + ks_b - ks_b_s"
  output: ks_d_s
```

- **Single-kernel:** `forced_photometry` uses `inputs.diffs: ks_d_s`.
- **Multi-kernel:** `hotpants` uses `inputs.bkg: ks_b_s`; photometry stays on `hp_d`.
- **Resume:** [`diff_config_multi_kernel_resume.yaml`](diff_config_multi_kernel_resume.yaml) inherits `ks_b` + `ks_d`, runs background smooth, then Hotpants.

Full algorithm, naming (`ks_` vs `hp_`), Savitzky–Golay details, meta artifacts, and performance notes: [docs/markdown/stages/background.md](../docs/markdown/stages/background.md).

## Hotpants: per-frame kernels for `syndiff star`

Every active `hotpants` stage in site and example diff configs sets **`write_kernel_solutions: true`**, which writes one `{product_id}_kernel.npz` per FFI under `ws/hp_d_kernels/`. The star pipeline reads these (plus `hp_c` convolved templates and `ks_b_s`/`ks_b` photometry background) and does **not** re-run Hotpants.

For a workspace that already has `hp_d` but no kernels (e.g. `multi_hp_temp_calib` before backfill), run a one-time Hotpants-only backfill with [`diff_config_star_full_backfill.yaml`](diff_config_star_full_backfill.yaml) (`write_convolved: true` + `write_kernel_solutions: true`). See [docs/markdown/stages/star_pipeline.md](../docs/markdown/stages/star_pipeline.md).
