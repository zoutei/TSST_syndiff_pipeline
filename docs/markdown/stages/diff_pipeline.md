> **Package integration**: `syndiff` stage `diff` · package `syndiff_pipeline/difference_imaging/` · configured by `config/diff_config*.yaml`  
> **Related docs**: [template pipeline guide](../template_pipeline.md) · [background stage](background.md)

# Difference-Imaging (`diff`) Stage — Internal Pipeline Reference

The orchestrator sees a single stage `diff` (`orchestration/stages.py`, `deps=("downsample",)`, Condor pool `diff`). Internally it runs an **ordered YAML pipeline of sub-stages** (`orchestration/execute.py: run_config_pipeline()`), validated against `STAGE_KINDS` in `orchestration/validate.py`:

`shared_mask`, `hotpants`, `kernel_fit`, `convolved_templates`, `kernel_subtract`, `epsf`, `sat_template`, `subtract`, `background`, `forced_photometry`

Preamble entries (no `kind`, must precede the first stage): `external_workspaces`, `workspace_inherit`.

Required handoff from the template pipeline (all under `events/{label}/`): `cluster_template_job.json`, `syndiff_ffi_frames.csv`, and the `ws/templates` symlink to the downsampled `syndiff_template_*.fits.gz` files.

---

## 1. Workspace layout and naming

- Event root: `{workspace_root}/events/{target_label}/`; pipeline tree `events/{label}/ws/` (or `ws_{workspace_run_id}/`).
- Per-sub-stage dirs: `ws/{label}/` where `label` comes from the stage's `output:` key (e.g. `hp_d`, `hp_c`, `hp_b`, `ep`, `lc_prf_on_diffs`).
- Per-FFI FITS: `{tess_product_id}_{label}.fits.gz` (e.g. `tess2020019142923_hp_d.fits.gz`); see `support/ffi_naming.py`.
- Root artifacts in `ws/`: `shared_mask.fits.gz`, `hotpants_substamp_stars.csv`, `gaia_catalog_pipeline.csv`, `targets.reg`, `diff_config.yaml` (frozen copy), `tile_centers.json`.
- Meta workspace paired with a diffs label (`hp_d` → `hp_m`): `kernel_reconstruction.npz`, `phot_calib.csv`, `hotpants.progress.json`.
- Optional flat mirror of all workspace FITS: `ws/master/` (symlinks, `master_fits_mirror: true`).

## 2. Sub-stages

### `shared_mask` (`stages/masking.py`)

Builds the shared bitmask (Gaia magnitude bins, bright-star crosses, BSC, TESS straps, optional PS1 coverage from the reference template when `ps1_min_hit_count > 0`) and selects isolated Hotpants reference stars (mag 13.5–14.5 default). Writes `shared_mask.fits.gz`, `hotpants_substamp_stars.csv`, `gaia_catalog_pipeline.csv`.

### `hotpants` (`stages/hotpants.py`)

Per-FFI Hotpants: cropped science frame vs the PS1 template of that frame's WCS group. Runs in-memory through pyhotpants `Hotpants.run_pipeline()`; FITS written afterward.

Outputs (per YAML `output:` block): diffs `ws/{diffs}/tess{pid}_{diffs}.fits.gz` (PRIMARY + NOISE + MASK), optional convolved model, Hotpants background, and stamps. Production default (`config/diff_config.yaml`): `write_convolved: false`, `write_bkg: true`, `write_stamps: false`. Also appends Hotpants status columns to `syndiff_ffi_frames.csv`.

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

### `epsf` (`stages/epsf.py`)

Tiled empirical PSF fitting on difference images; per-template-group median ePSF. Writes `epsf_stack_r1.npz`, `epsf_r1_smooth.npz`, `group_epsf/group_epsf_{gid}.npy`, and `tile_centers.json` (in `ws/` root; shared with `sat_template` and photometry).

### `sat_template` (`stages/sat_template.py`) — see §5

Builds per-group model images of bright stars as flux-scaled ePSF stamps: `ws/{output}/sat_tmpl_native_r1/group_{gid}.fits.gz` (2× oversampling path) and `sat_tmpl_hr_r1/group_{gid}.fits.gz` (9×).

### `forced_photometry` (`stages/photometry.py`)

Forced PSF (`prf` official TESS PRF, or `epsf` from the epsf stage) and/or aperture photometry at the primary target and `additional_forced_targets`. Writes `ws/{output}/lightcurve_{method}.csv` (and `lightcurve_{method}_{extra_name}.csv`), plus diagnostic plots when `pipeline_plots: true`. If `ws/{diffs_m}/phot_calib.csv` exists, kernel-sum zero-point calibration columns (`kernel_ref`, calibrated `flux`, `tmag`, `flux_jy`) are added.

## 3. Production pipeline orders

| Config | Order |
|--------|-------|
| `config/diff_config.yaml` (default) | `shared_mask` → `hotpants` → `forced_photometry` |
| `config/diff_config_single_kernel.yaml` | `shared_mask` → `kernel_fit` → `convolved_templates` → `kernel_subtract` → `background` → `subtract` → `forced_photometry` |
| `config/diff_config_multi_kernel.yaml` | same prefix → `background` → `hotpants` (round 2, `hp_bgo=0`) → `forced_photometry` |
| `config/diff_config_multi_kernel_resume.yaml` | `workspace_inherit` → `background` → `hotpants` → `forced_photometry` |

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
| in-memory | Full per-frame `kernel_solution` vector | Yes — **discarded** after each frame |
| `ws/{kernel_fit}/kernel_r2.npz` | Target-level kernel from `kernel_fit` | One per target |

**Per-frame Hotpants kernel vectors are deliberately not written** (see docstring at the top of `stages/hotpants.py`). Consequences:

- "Convolve a new/modified template with the already-derived kernel" is only possible on the **kernel_fit path**: reuse `kernel_r2.npz`, re-run `convolved_templates` (with `skip_existing: false` or cleared outputs) → `kernel_subtract` → `forced_photometry`.
- On the default hotpants-only path, changing a template requires re-running `hotpants` (new per-frame fits).

## 6. `sat_template` — current behavior and known gaps

Intended purpose: model images of the **PS1-removed** (saturated) stars, per WCS group, for later subtraction. Actual behavior has two gaps that matter when planning star-subtraction work:

1. **Star selection**: `execute.py: _load_removed_stars_in_crop()` returns the full Gaia catalog whenever `gaia_df` already carries crop-local `x`/`y` columns — which is always true after `shared_mask`. So in production runs `removed_stars_csv` (i.e. `events/{label}/ps1_removed_stars.csv` produced by downsample) is **ignored**, and the "sat" template contains *all* Gaia stars in the crop, not just removed ones.
2. **Subtraction wiring**: outputs are per-group `group_{gid}.fits.gz`, but the `subtract` stage consumes per-frame `tess{pid}_{label}.fits.gz` planes. No code bridges the two; the example config `config/example/diff_config_b_epsf_sat_bkg.yaml` that pairs them is stale (it also references a removed `background_estimate` kind). `load_group_templates()` is unused outside the module.

There is currently **no mechanism to subtract a single, chosen star from a template** — see `.cursor/plans/exoplanet_star_removed_template.plan.md` for the planned design.

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
| Photometry / ePSF | `difference_imaging/stages/photometry.py`, `epsf.py` |
| Paths / naming | `difference_imaging/support/paths.py`, `ffi_naming.py` |
