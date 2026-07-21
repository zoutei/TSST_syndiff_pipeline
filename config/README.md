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
| `targets_example.csv` | Example event targets CSV for `syndiff diff --targets` |
| `scc_example.csv` | Example SCC-only CSV (`sector,camera,ccd[,enabled]`) for `syndiff template --scc` |

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

# Template DAG: SCC-only input, no event coordinates
syndiff template submit --site config --scc config/scc_example.csv --run-id my_template_run

# Field-mode diff (SCC-only), once templates exist
syndiff diff submit --site config \
  --config config/diff_config_single_kernel.yaml \
  --scc config/scc_example.csv --run-id my_diff_run

# Event photometry (optional)
syndiff diff submit --site config --targets config/targets_example.csv --run-id my_event_diff
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
    epsf: epsf_r1          # required if any method uses psf_type: epsf
  output: lc_photometry
  methods:
    - name: ap3
      type: aperture
      tar_ap: 3
      sky_in: 5
      sky_out: 9
      subtract_sky: true                 # false → primary LC uses raw flux
      mask_sky_with_shared_mask: false   # true → exclude shared_mask catalog/cross bits from sky annulus

    - name: prf
      type: psf
      psf_type: prf
      phot_cutout_size: 15
      phot_bkg_poly_order: 3   # null → flux-only (no poly surface); 0 → constant bkg
      phot_snap: brightest     # brightest | ref | fixed

    - name: epsf
      type: psf
      psf_type: epsf           # photutils GriddedPSFModel (default fitter: photutils)
      fit_shape: 11
      aperture_radius: 2.0
      # psf_grouper_min_separation: null  # default; set a float only for multi-init grouping

    - name: epsf_bkg
      type: psf
      psf_type: epsf
      fitter: tessreduce       # TESSreduce create_psf BFGS on that frame's gridded stamp
      phot_bkg_poly_order: 0
      phot_cutout_size: 15
      phot_snap: brightest
```

| Mode | YAML | Fitter |
|------|------|--------|
| Aperture | `type: aperture` | Square sum ± sky annulus |
| PRF | `type: psf`, `psf_type: prf` | Official TESS PRF + TESSreduce `create_psf` |
| ePSF photutils | `psf_type: epsf` (default / `fitter: photutils`) | Per-frame `GriddedPSFModel` |
| ePSF tessreduce | `psf_type: epsf`, `fitter: tessreduce` | Same BFGS as PRF; stamp from that frame's gridded ePSF |

**ePSF requirement:** `psf_type: epsf` needs a modern gridded catalog
(`gridded_epsf_index.json` under the ePSF workspace). Missing index raises a
clear error; the legacy tile-smooth stack is not used for forced photometry.

**CSV names:** primary → `lightcurve_{name}.csv`; extra targets from
`additional_forced_targets` → `lightcurve_{name}_{target}.csv`.

**PSF columns:** `btjd`, `flux`, `eflux`, `filename`, `group_id` (photutils also
writes `x_fit`, `y_fit`).

**Aperture columns:** same metadata plus `flux` (raw sum), `flux_wo_sky`
(sky-subtracted), `sky`, and `eflux`. With `subtract_sky: true` (default), ZP and
plots use `flux_wo_sky`; with `false`, they use raw `flux`. Defaults match
TESSreduce `diff_lc`: `tar_ap=3`, `sky_in=5`, `sky_out=9`. Star masking for the
sky annulus uses existing `shared_mask.fits.fz` (bits 1|2 = Gaia catalog + bright
crosses), not a per-method mag cut.

Top-level `psf_type` is no longer supported; migrate existing configs to a
`methods` entry with `type: psf`.

Allowed `fitter` values for `psf_type: epsf`: `photutils` (default) | `tessreduce`.
`fitter` is forbidden on `psf_type: prf`.

Full parameter tables, fitting steps, outputs, and dual-method examples:
[docs/markdown/stages/forced_photometry.md](../docs/markdown/stages/forced_photometry.md).

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

### Oversampling (`F`) and stamp modes

Template `oversampling_factor` lives in `pipeline.yaml` (`stages.mapping` /
`stages.templates`). Diff Hotpants accepts optional `oversample`,
`stamp_mode` (`grid` \| `connected_regions`), `use_c_extension`, and
`region_*` on the `kind: hotpants` stage. Star uses
`defaults.oversampling_factor` in `star_config.yaml` (must match template
`F`). Full parameter tables and recipes:
[docs/markdown/oversampled_templates.md](../docs/markdown/oversampled_templates.md).
