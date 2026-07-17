> **Package integration**: `syndiff` stage `diff` · package `syndiff_pipeline/difference_imaging/` · configured by `config/diff_config*.yaml`  
> **Related docs**: [template pipeline guide](../template_pipeline.md) · [background stage](background.md)

# Difference-Imaging (`diff`) Stage — Internal Pipeline Reference

The orchestrator sees a single stage `diff` (`orchestration/stages.py`, `deps=("downsample",)`, Condor pool `diff`). Internally it runs an **ordered YAML pipeline of sub-stages** (`orchestration/execute.py: run_config_pipeline()`), validated against `STAGE_KINDS` in `orchestration/validate.py`:

`astrometry`, `shared_mask`, `hotpants`, `kernel_fit`, `convolved_templates`, `kernel_subtract`, `epsf`, `centroids`, `sat_template`, `subtract`, `background`, `forced_photometry`

Preamble entries (no `kind`, must precede the first stage): `external_workspaces`, `workspace_inherit`.

Required handoff from the template pipeline (all under `events/{label}/`): `cluster_template_job.json`, `syndiff_ffi_frames.csv`, and the `ws/templates` symlink to the downsampled `syndiff_template_*.fits.gz` files. **Exception:** an astrometry-only pipeline (`pipeline: [{kind: astrometry}]`) skips template handoff, DS9 regions at startup, and the master FITS mirror.

---

## 0. Config ownership (`diff_config` vs `mask_settings`)

Site authoring is intentionally split:

| File | Owns |
|------|------|
| `diff_config.yaml` `defaults:` / top-level | Multi-stage knobs: crop, `n_jobs`, `max_ffis`, `workspace_run_id`, `pipeline_plots*`, `master_fits_mirror`, `additional_forced_targets` |
| `diff_config.yaml` `pipeline:` `kind:` blocks | Stage-only knobs. **Omit keys that match dataclass defaults** in `stage_params.py` |
| `mask_settings.yaml` (optional sibling) | Mask geometry/policy (style, maglims, strap/edge/PS1 *policy*, TNS, asteroids). **Not** embedded in `diff_config` |
| `deployment.yaml` | `workspace_root`, `data_root`, credentials |

`- kind: shared_mask` does **not** require a `mask_settings:` path. Resolve order is stage path → `{ws}/mask_settings.yaml` → site `mask_settings.yaml` → packaged defaults. An existing `{ws}/mask_settings.yaml` wins over later site edits until removed/replaced. See [masking.md](../masking.md).

Frozen `{ws}/diff_config.yaml` is a **slim snapshot** (`cfg_to_snapshot_dict`): empties, SynDiffConfig defaults, and bundled `straps_csv`/`bsc_catalog` paths are omitted; pipeline stages drop keys equal to their param-dataclass defaults. Legacy full dumps still load via `load_config`.

Bundled straps / BSC: leave `straps_csv` / `bsc_catalog` unset (empty = packaged resource at use time). Do not expect absolute bundled paths in the frozen snapshot.

---

## 1. Workspace layout and naming

- Event root: `{workspace_root}/events/{target_label}/`; pipeline tree `events/{label}/ws/` (or `ws_{workspace_run_id}/`).
- Per-sub-stage dirs: `ws/{label}/` where `label` comes from the stage's `output:` key (e.g. `hp_d`, `hp_c`, `hp_b`, `ep`, `lc_prf_on_diffs`).
- Per-FFI FITS: `{tess_product_id}_{label}.fits.gz` (e.g. `tess2020019142923_hp_d.fits.gz`); see `support/ffi_naming.py`.
- Root artifacts in `ws/`: `shared_mask.fits.gz`, `hotpants_substamp_stars.csv`, `gaia_catalog_pipeline.csv`, `targets.reg`, `diff_config.yaml` (frozen copy), `tile_centers.json`.
- Astrometry artifacts in the active workspace root (`ws/` or `ws_{workspace_run_id}/`): `astrometry_result.json`, optional `debug_plots/astrometry_mix.png` when `pipeline_plots: true`.
- Meta workspace paired with a diffs label (`hp_d` → `hp_m`): `kernel_reconstruction.npz`, `phot_calib.csv`, `hotpants.progress.json`.
- Optional flat mirror of all workspace FITS: `ws/master/` (symlinks, `master_fits_mirror: true`).

## 2. Sub-stages

### `astrometry` (`stages/astrometry.py`)

First sub-stage in the default `config/diff_config.yaml`. Resolves the transient position from ATLAS, ZTF (IRSA), Gaia alert, and TNS data using the notebook default algorithm (`survey_ivw` inverse-variance mix). Writes refined `ra_deg` / `dec_deg` to `astrometry_result.json` under the active workspace and updates `cfg.target_ra` / `cfg.target_dec` for downstream forced photometry and DS9 regions.

Targets CSV may omit `target_ra` / `target_dec` (event-name-only rows); astrometry uses TNS/Fink as the search seed. When coordinates are present they are used as the seed but still overwritten by the refined mix.

Optional YAML params: `sigma_mag_limit` (default 0.15), `clip_n_sigma` (default 3.0), `atlas_credentials_file`, `irsa_credentials_file`. Credentials may also come from env (`TNS_API_KEY`, `ATLAS_CREDENTIALS_FILE`, `IRSA_CREDENTIALS_FILE`).

**Smoke test (astrometry only, no template handoff):**

```bash
mamba activate syndiff
syndiff diff run \
  --config config/diff_config_astrometry_only.yaml \
  --deployment config/deployment.yaml \
  --targets config/targets_example.csv \
  --target-name s0100_c1_k2_2026gvk \
  --workspace-run-id astrometry_smoke
```

**Full diff integration (astrometry + shared_mask + hotpants + forced_photometry):**

```bash
syndiff diff run --site config/ \
  --targets config/targets_example.csv \
  --target-name s0020_c3_k3_2020ut \
  --workspace-run-id astrometry_integration_test
```

Outputs land in `events/{label}/ws_{workspace_run_id}/` (not production `ws/`).

### `shared_mask` (`stages/masking.py` → `syndiff_pipeline.difference_imaging.masking`)

Builds the shared bitmask and selects isolated Hotpants reference stars.
**Default style is empirical** (see [masking.md](../masking.md)): T&lt;13 empirical
circles/crosses (bits 1/2; Gaia ∪ BSC for crosses), 13≤T&lt;18 `faint_star_squares`
(bit 32), straps/edges/PS1 (4/8/16), optional TNS (64) and per-cadence asteroids
(128 via `MaskCatalog`). Rollback: site `mask_settings.yaml` with
`shared.style: tessreduce` (bits 1/2/4/8/16 only).

**Config:** mask policy comes from `mask_settings.yaml` (sibling of
`diff_config.yaml`, or packaged defaults). Stage YAML only needs
`- kind: shared_mask` plus optional Hotpants ref-star selection keys
(`ref_mag_*`, `ref_isolation_*`, `ref_separation_px`) or an explicit
`mask_settings:` path override. Do not put maglims/strapsize/PS1 thresholds on
the stage (legacy keys still work if explicit; prefer the mask file).

Writes `shared_mask.fits.gz`, `hotpants_substamp_stars.csv`,
`gaia_catalog_pipeline.csv`, frozen `mask_settings.yaml`. See [masking.md](../masking.md).

### `hotpants` (`stages/hotpants.py`)

Per-FFI Hotpants: cropped science frame vs the PS1 template of that frame's WCS group. Runs in-memory through pyhotpants `Hotpants.run_pipeline()`; FITS written afterward.

Outputs (per YAML `output:` block): diffs `ws/{diffs}/tess{pid}_{diffs}.fits.gz` (PRIMARY + NOISE + MASK), optional convolved model, Hotpants background, and stamps. Production default (`config/diff_config.yaml`): `write_convolved: false`, `write_bkg: true`, `write_stamps: false`. Also appends Hotpants status columns to `syndiff_ffi_frames.csv`.

When `write_kernel_solutions: true`, per-frame kernel vectors are persisted as `{diffs_label}_kernels/{product_id}_kernel.npz` (e.g. `hp_d_kernels/tess2020019142923_kernel.npz`) alongside the diffs workspace. Default is off; see §5.

### `kernel_fit` (`stages/kernel_fit.py`)

Fits **one target-level kernel** on the minimum-background FFI: Hotpants pass 1 (`hp_bgo=3`) → photutils background removal → Hotpants pass 2 (`hp_bgo=0`), extracting the kernel solution. Writes `ws/{output}/kernel_r2.npz` (`kernel_solution`, `kernel_image`, `basis`, substamps) and `kernel_fit_meta.json`.

### `convolved_templates` (`stages/convolved_templates.py`)

Convolves each unique WCS-group template with the fixed `kernel_r2.npz` solution (`convolve_template_with_kernel_solution()`). Writes `convolved_template_dx{X.XXX}_dy{Y.YYY}.fits.gz` plus a `convolved_templates.csv` manifest (`group_id`, `group_dx`, `group_dy`, `template_path`, `convolved_path`).

### `kernel_subtract` (`stages/kernel_subtract.py`)

Algebraic per-frame diff without re-running Hotpants: `diff_raw = ffi_crop − convolved_template(group_dx, group_dy)`, plus a photutils background estimate on the diff (written separately if `output.phot_bkg` is set; the diff itself is *not* background-subtracted). Parallel via joblib.

### `background` (`stages/background/pipeline.py`)

Unified background cube (spatial photutils, temporal Savitzky–Golay, strap correction). See [background.md](background.md). Writes `stack.npz`/`stack.npy` and optional per-frame FITS.

### `subtract` (`support/subtract.py` + `execute.py`)

Per-frame linear combination of workspace planes (or the virtual cropped `ffi` label), e.g. `expression: "ks_d + ks_b - ks_b_s"` → `ws/ks_d_s/`.

### `epsf` (`stages/epsf.py`, `stages/gridded_epsf.py`)

Per-frame gridded empirical PSF fitting on difference images with **photutils** (`EPSFBuilder` + `GriddedPSFModel`), not TGLC. Each frame is tiled into `tile_ny × tile_nx` sections (e.g. 2×2 or 3×3); Gaia stars are pre-filtered to `phot_rp_mean_mag < mag_max_rp` (default 12.95) with crop-local `x`/`y`. Star extraction uses per-FFI `mask_at` and **ignores bits 1|2|32** (bright catalog / saturation crosses / faint squares) so Gaia ePSF stars are kept; any other set bit (4/8/16/64/128) rejects. See [masking.md](../masking.md).

Primary outputs under `ws/{output}/`:

| Artifact | Description |
|----------|-------------|
| `{ffi_stem}_gridded_epsf.npz` | Per-frame `GriddedPSFModel` archive: `data` (grid cube), `grid_xypos`, `oversampling` |
| `gridded_epsf_index.json` | `ffi_stem` → npz path mapping |
| `epsf.progress.json` | Frame progress sidecar (also mirrored as `diff.epsf.progress.json` beside `diff.log`) |

Legacy tile-stack bundles (for `sat_template` and tile-interpolated photometry) are still written: `epsf_stack_r1.npz`, `epsf_r1_smooth.npz`, `group_epsf/group_epsf_{gid}.npy`, and `group_epsf/group_epsf_{gid}.npz` (median gridded cube per WCS group). `tile_centers.json` is saved in `ws/` root (shared with `sat_template` and legacy ePSF photometry).

Key YAML params: `tile_nx`, `tile_ny`, `epsf_oversample`, `psf_size`, `extract_size`, `min_stars_per_tile`, `mag_max_rp`, `epsf_maxiters`, `epsf_recentering_maxiters`, `epsf_n_jobs`.

### `centroids` (`stages/centroids.py`)

Gaia-star PSF photometry on difference images using per-frame `GriddedPSFModel` from an `epsf` workspace (`photutils.PSFPhotometry` with the same brightness pre-filter as ePSF fitting). Inputs: `diffs` (difference-image label), `epsf` (gridded ePSF workspace label).

Outputs under `ws/{output}/`:

| Artifact | Description |
|----------|-------------|
| `{ffi_stem}_photresults.ecsv` | Per-frame multi-star PSF photometry table (Gaia metadata + fitted positions/fluxes) |
| `centroids_index.json` | `ffi_stem` → photresults path mapping |
| `centroids.progress.json` | Frame progress sidecar (also mirrored as `diff.centroids.progress.json` beside `diff.log`) |
| `centroids.progress.json` | Frame progress sidecar (also mirrored as `diff.centroids.progress.json` beside `diff.log`) |

Key YAML params: `mag_max_rp`, `fit_shape`, `aperture_radius`, `psf_grouper_min_separation`, `centroids_n_jobs`.

### `sat_template` (`stages/sat_template.py`) — see §6

Builds per-group model images of bright stars as flux-scaled ePSF stamps: `ws/{output}/sat_tmpl_native_r1/group_{gid}.fits.gz` (2× oversampling path) and `sat_tmpl_hr_r1/group_{gid}.fits.gz` (9×).

### `forced_photometry` (`stages/photometry.py`)

Forced PSF and/or aperture photometry at the primary target and `additional_forced_targets`. PSF methods (`type: psf`) support:

- `psf_type: prf` — official TESS PRF (`PRF.TESS_PRF`)
- `psf_type: epsf` with `inputs.epsf` pointing at a gridded ePSF workspace — per-frame `GriddedPSFModel` via `photutils.PSFPhotometry` (e.g. method name `gepsf` in `config/diff_config_2020ut_epsf_gepsf.yaml`)
- `psf_type: epsf` without gridded index — legacy tile-interpolated group ePSF from `epsf_r1_smooth.npz`

Writes `ws/{output}/lightcurve_{method}.csv` (and `lightcurve_{method}_{extra_name}.csv`), plus diagnostic plots when `pipeline_plots: true`. If `ws/{diffs_m}/phot_calib.csv` exists, kernel-sum zero-point calibration columns (`kernel_ref`, calibrated `flux`, `tmag`, `flux_jy`) are added.

## 3. Production pipeline orders

| Config | Order |
|--------|-------|
| `config/diff_config.yaml` (default) | `astrometry` → `shared_mask` → `hotpants` → `forced_photometry` |
| `config/diff_config_astrometry_only.yaml` | `astrometry` (smoke test; no template handoff) |
| `config/diff_config_single_kernel.yaml` | `shared_mask` → `kernel_fit` → `convolved_templates` → `kernel_subtract` → `background` → `subtract` → `forced_photometry` |
| `config/diff_config_multi_kernel.yaml` | same prefix → `background` → `hotpants` (round 2, `hp_bgo=0`) → `forced_photometry` |
| `config/diff_config_multi_kernel_resume.yaml` | `workspace_inherit` → `background` → `hotpants` → `forced_photometry` |
| `config/pipeline_epsf_gepsf.yaml` / `config/diff_config_2020ut_epsf_gepsf.yaml` | `workspace_inherit` (from `multi_hp_temp_calib`: `hp_d`, `hp_m`, shared mask, Gaia catalog, substamp stars) → `epsf` (3×3 grid) → `centroids` → `forced_photometry` (`gepsf`, `psf_type: epsf`) on `hp_d` |

## 4. Template resolution

Template filename pattern (parsed by `parse_syndiff_template_filename()` in `stages/hotpants.py`):

```
syndiff_template_s{sector}_{camera}_{ccd}[_x{x0}-{x1}_y{y0}-{y1}][_osN]_dx{dx}_dy{dy}.fits[.gz]
```

Per-frame selection: the manifest row gives `(group_id, group_dx, group_dy)`; `find_template_by_offset()` matches the filename `dx`/`dy` against the manifest offsets with tolerance `max(1e-5, 0.01 × offset_threshold)`; `cfg.template_paths[group_id]` then holds the absolute path. Discovery prefers flat `syndiff_template_*` files under `template_dir` (normally the `ws/templates` symlink); fallback is `group_{id}/ps1_template.fits`.

This is the hook for swapping in modified templates: point `template_dir` (or the `ws/templates` symlink) at a directory of alternative templates with the **same filenames**, and every sub-stage picks them up.

## 5. Kernel persistence — what is and is not saved

| Location | What | Per-frame? |
|----------|------|------------|
| `ws/{diffs_m}/kernel_reconstruction.npz` | Hotpants **basis** stack + config scalars | No (one per hotpants pass) |
| `ws/{diffs_m}/phot_calib.csv` | `kernel_sum`, `tess_zp` per FFI | Yes (scalars only) |
| in-memory (default) | Full per-frame `kernel_solution` vector | Yes — **discarded** after each frame |
| `ws/{diffs_label}_kernels/{product_id}_kernel.npz` | Per-frame `kernel_solution` + Hotpants config scalars | Yes — only when `write_kernel_solutions: true` |
| `ws/{kernel_fit}/kernel_r2.npz` | Target-level kernel from `kernel_fit` | One per target |

**Per-frame Hotpants kernel vectors are not written by default** (see docstring at the top of `stages/hotpants.py`). Set `write_kernel_solutions: true` on a `hotpants` stage to persist them under `{diffs_label}_kernels/`. Consequences:

- "Convolve a new/modified template with the already-derived kernel" is only possible on the **kernel_fit path**: reuse `kernel_r2.npz`, re-run `convolved_templates` (with `skip_existing: false` or cleared outputs) → `kernel_subtract` → `forced_photometry`.
- On the default hotpants-only path, changing a template requires re-running `hotpants` (new per-frame fits).

## 6. `sat_template` — current behavior and known gaps

Intended purpose: model images of the **PS1-removed** (saturated) stars, per WCS group, for later subtraction. Actual behavior has two gaps that matter when planning star-subtraction work:

1. **Star selection**: `execute.py: _load_removed_stars_in_crop()` returns the full Gaia catalog whenever `gaia_df` already carries crop-local `x`/`y` columns — which is always true after `shared_mask`. So in production runs `removed_stars_csv` (i.e. `events/{label}/ps1_removed_stars.csv` produced by downsample) is **ignored**, and the "sat" template contains *all* Gaia stars in the crop, not just removed ones.
2. **Subtraction wiring**: outputs are per-group `group_{gid}.fits.gz`, but the `subtract` stage consumes per-frame `tess{pid}_{label}.fits.gz` planes. No code bridges the two; the example config `config/example/diff_config_b_epsf_sat_bkg.yaml` that pairs them is stale (it also references a removed `background_estimate` kind). `load_group_templates()` is unused outside the module.

There is currently **no mechanism to subtract a single, chosen star from a template** via the main diff pipeline — see [host-star light curves](../star_lightcurves.md) and [star pipeline](star_pipeline.md) for the `syndiff star` workflow that builds per-host mini templates and star-only diff stamps from persisted per-frame Hotpants artifacts.

### Downstream: host-star light curves

The **`star`** stage reads baseline diff workspaces (not `hp_d` images directly):

- `convolved` (e.g. `hp_c`) — per-frame convolved template windows
- `{diffs}_kernels` — per-frame Hotpants kernels
- `phot_bkg` (e.g. `ks_b_s` or `ks_b`) — photutils background subtracted in star stamps (**not** `hp_b`)

Produce `ks_b` via `kernel_subtract`; smooth to `ks_b_s` with the `background` stage. Star config sets `baseline.phot_bkg` explicitly.

## 7. Config schema highlights

Top-level keys in `diff_config.yaml`: `deployment_file`, `defaults` (merged into `SynDiffConfig`: `n_jobs`, `crop_mode`, `crop_box_size`, `pipeline_plots`, `master_fits_mirror`, `workspace_run_id`, …), `pipeline` (ordered stage list; unknown keys per stage fail validation via `*_ALLOWED` frozensets in `orchestration/stage_params.py`), `additional_forced_targets`, `per_event_force_targets`, `overrides` (keyed `"sector/camera/ccd"`), `condor`.

A frozen per-target copy of the effective config is written to `runs/.../per_target/{label}/diff_config.yaml` at orchestrator launch (`site_config.freeze_target_diff_config()`).

## 8. Recipes: reusing prior work

| Goal | What to reuse | What to re-run |
|------|---------------|----------------|
| New photometry on existing diffs | `ws/hp_d/*.fits.gz` | `forced_photometry` only |
| Modified templates, kernel-fit path | `kernel_r2.npz` + `kernel_fit_meta.json` | `convolved_templates` → `kernel_subtract` → (`background`/`subtract`) → `forced_photometry` |
| Modified templates, hotpants path | shared mask, substamp stars | full `hotpants` (per-frame kernels re-fit) |
| Continue a multi-kernel run | inherit labels via `workspace_inherit` + `workspace_run_id` | remaining stages in the new run id |

## 9. Key files

| Concern | Path |
|---------|------|
| Pipeline driver | `difference_imaging/orchestration/execute.py` |
| Stage kinds / validation | `difference_imaging/orchestration/validate.py`, `stage_params.py` |
| YAML → config | `difference_imaging/orchestration/config.py`, `site_config.py` |
| Template matching | `difference_imaging/support/template_resolution.py` |
| Kernel fit / convolve / subtract | `difference_imaging/stages/kernel_fit.py`, `convolved_templates.py`, `kernel_subtract.py`, `kernel.py` |
| Sat templates | `difference_imaging/stages/sat_template.py` |
| Photometry / ePSF / centroids | `difference_imaging/stages/photometry.py`, `epsf.py`, `gridded_epsf.py`, `centroids.py`, `epsf_progress.py` |
| Paths / naming | `difference_imaging/support/paths.py`, `ffi_naming.py` |
