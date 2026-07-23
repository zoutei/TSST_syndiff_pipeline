---
name: syndiff-pipeline-map
description: Architecture, data flow, artifact paths, and invariants of the syndiff_pipeline (TESS+PS1 difference imaging and host-star light curves). Use when working on any pipeline stage (tess_ffi_download, wcs_grouping, mapping, ps1_download, ps1_process, downsample, diff, star), when locating artifacts (templates, manifests, Zarr stores, kernels, light curves), or when modifying template creation, difference-imaging, or star-pipeline code.
---

# SynDiff Pipeline Map

## Stage DAG and where the code lives

```
tess_ffi_download → wcs_grouping → mapping → ps1_download → ps1_process → downsample → diff
   (network)          (local)     (Condor)    (network)      (Condor)     (cpu)      (Condor)

completed template + diff artifacts ──verify──→ star
                                                (Condor; separate star_targets.csv)
```

The composed stage registry has eight stages. The first seven form the normal
template+diff DAG; `star` is an independent branch whose prerequisites are
verified from an existing event workspace.

| Stage | Module | Deep-dive doc |
|-------|--------|---------------|
| `tess_ffi_download` | `common/download.py` | `docs/markdown/stages/tess_ffi_download.md` |
| `wcs_grouping` | `common/wcs_grouping.py` + `template_creation/orchestration/handoff.py` | `docs/markdown/stages/wcs_grouping.md` |
| `mapping` (PanCAKES + Gaia download) | `template_creation/processing/pancakes.py` | `docs/markdown/stages/mapping_pancakes.md` |
| `ps1_download` | `template_creation/processing/ps1_download.py` | `docs/markdown/stages/README.md` |
| `ps1_process` | `template_creation/processing/ps1_process.py`, `band_utils.py` | `docs/markdown/stages/ps1_process_technical.md` |
| `downsample` | `template_creation/processing/downsample.py` | `docs/markdown/stages/downsample_technical.md` |
| `diff` (internal sub-pipeline) | `difference_imaging/orchestration/execute.py` + `difference_imaging/stages/` + `difference_imaging/masking/` | `docs/markdown/stages/diff_pipeline.md`, `docs/markdown/masking.md` |
| `star` (host-star light curves) | `star/runner.py`, `star/epsf_runner.py`, `star/orchestration/stages.py` | `docs/markdown/star_lightcurves.md`, `docs/markdown/stages/star_pipeline.md` |

Mask library code lives under **`difference_imaging/masking/`** (consumers: diff stages + star ePSF). Template creation does **not** import it.

Orchestration (scheduler, SQLite state, Condor, verify): `docs/markdown/template_pipeline.md`, `docs/markdown/template_runner_architecture.md`. Read the relevant deep-dive doc before editing a stage.

## Site config layout (`config/`)

| File | Git | Owns |
|------|-----|------|
| `pipeline.yaml` | committed | Template policy: stages, pools, notifications |
| `diff_config.yaml` | committed | Diff sub-pipeline: `defaults` (crop/n_jobs/plots) + `pipeline:` stage knobs. **Omit stage keys that match dataclass defaults.** |
| `mask_settings.yaml` | optional | Mask geometry/policy (empirical/tessreduce, maglims, TNS, asteroids). Sibling of `diff_config`; **not** embedded in `diff_config`. Bare `- kind: shared_mask` uses site file or packaged defaults. |
| `star_config.yaml` | committed | Star-branch policy |
| `deployment.yaml` | **gitignored** | `workspace_root`, `data_root`, Gaia + Discord credentials (copy from `deployment.yaml.example`) |

Targets are always passed on the CLI (`--targets` / `--star-targets`), never embedded in config.

Frozen copies: each run freezes effective template config under `runs/{run_id}/`; each diff workspace freezes a **slim** `ws/diff_config.yaml` (`cfg_to_snapshot_dict` — empties/defaults/bundled straps|BSC paths omitted) plus `ws/mask_settings.yaml` after `shared_mask`. Check the frozen copies when debugging, not only site YAML defaults.

Deep dive: `docs/markdown/stages/diff_pipeline.md` §0 (config ownership), `docs/markdown/masking.md`.

## Artifact map

Two roots: `data_root` (SCC-wide, shared across targets) and `workspace_root` (per-target events). See `docs/markdown/storage_layout.md`.

```
{data_root}/
  tess_ffi/s{SSSS}/cam{C}_ccd{K}/*.fits(.gz)                      ← FFIs
  skycell_pixel_mapping/sector_*/camera_*/ccd_*/
      [oversampling_{N}/]                                         ← present when oversampling_factor > 1
      tess_s*_master_skycells_list.csv                            ← mapping verify gate
      tess_s*_master_pixels2skycells.fits.gz                      ← TESS WCS + skycell-ID map
      tess_s*_skycell.{proj}.{cell}.fits.gz                       ← per-skycell PS1→TESS reg maps
  catalogs/sector_*/camera_*/ccd_*/gaia_catalog_s*.csv            ← Gaia DR3 (downloaded in mapping stage)
  ps1_skycells_zarr/ps1_skycells.zarr                             ← raw PS1 bands (shared, file-locked)
  ps1_skycells.zarr                                               ← star cache default (same schema, different path)
  convolved_results/sector_*_camera_*_ccd_*.zarr                  ← ps1_process output: flat {skycell}_data/_mask arrays
  convolved_results/..._removed_stars.csv                         ← removed-star records
  shifted_downsampled/sector*_camera*_ccd*[_roi]/syndiff_template_*_dx*_dy*.fits.gz

{workspace_root}/events/{target_label}/
  cluster_template_job.json          ← reference FFI, groups (group_dx/dy), crop bounds
  syndiff_ffi_frames.csv             ← per-FFI manifest: drift, btjd, group_id, hotpants status
  ps1_removed_stars.csv              ← removed stars projected to crop-local x,y
  ws/templates → (symlink to shifted_downsampled dir)
  ws/{label}/tess{product_id}_{label}.fits.gz                     ← diff sub-stage outputs (hp_d, ks_d, …)
  ws/{diffs_m}/phot_calib.csv, kernel_reconstruction.npz          ← hotpants meta
  ws/{diffs_label}_kernels/{product_id}_kernel.npz                ← optional per-frame Hotpants kernels
  ws/kernel_fit/kernel_r2.npz                                     ← reusable target-level kernel
  {baseline_ws}/{epsf_label}/{*_gridded_epsf.npz, gridded_epsf_index.json}
                                                                    ← reusable/built-on-demand star gepsf models
  ws/{lc_label}/lightcurve_{method}[_{extra}].csv                 ← light curves
  ws/debug_plots/                                                 ← WCS/diff diagnostics
  {baseline_ws}/host_star/
      batch_manifest.csv
      {gaia_source_id}/{identifier.json, mini_templates/, diff_stamps/,
                        lightcurve_{method}_gaia_{id}.csv, plots/}
```

The star cache defaults to `{data_root}/ps1_skycells.zarr`, while
`ps1_download` writes `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`.
Set top-level `ps1_zarr_path` in `star_config.yaml` to reuse the latter.

## Invariants that bite

1. **Offset quantization**: 1 PS1 px = 0.258″ ≈ 0.0124 TESS px; `offset_threshold` = 0.01 TESS px. Template offsets are realized as **integer PS1-pixel rolls per skycell** (`compute_ps1_shift_for_skycell`); sub-pixel WCS drift never requires re-running `mapping`.
2. **Drift is measured at the target position only** (single point) — templates degrade away from the target. Plans to fix: `.cursor/plans/spatially_varying_wcs_templates.plan.md`.
3. **Coordinates**: crop bounds are `[min, max)` in full-FFI 0-based pixels; diff-stage x/y are **crop-local**. `ensure_gaia_crop_xy` converts. Mixing these silently misplaces stars.
4. **Template filenames are the API**: `syndiff_template_s{S}_{C}_{K}[_x..._y...][_osN]_dx{±D.DDD}_dy{±D.DDD}.fits.gz`; the diff stage matches manifest `group_dx/dy` to filename dx/dy (tolerance `0.01 × offset_threshold`). Swapping templates = pointing `template_dir` at a dir with identical filenames.
5. **Kernels**: Hotpants kernels are not persisted by default, but `write_kernel_solutions: true` writes one `{diffs_label}_kernels/{product_id}_kernel.npz` per frame (required by `star`). The separate `kernel_fit` path persists one target-level reusable kernel (`kernel_r2.npz`) for `convolved_templates` → `kernel_subtract`.
5b. **Hotpants ignores FAINT_CAT (bit 32)** via `hotpants_mask_bool`; ePSF ignores `1|2|32` (very bright / mid-bright / faint catalog tiers). Bit 1 = all BSC + Gaia `tess_mag` &lt; `epsf_mag_lim` (7.5); bit 2 = Gaia 7.5–13. Mask policy lives in `mask_settings.yaml`, not `diff_config` stage keys. Implementation: `syndiff_pipeline.difference_imaging.masking` (not a top-level package).
6. **Verify gates are thin**: mapping = CSV exists; templates = files parse. Partially stale artifacts pass verify — delete outputs when inputs changed.
7. **FFI files may be `.fits` or `.fits.gz`** — always resolve both (helpers exist in `common/download.py`).
8. **`sat_template` is broken for its stated purpose**: it uses the full Gaia catalog (not `removed_stars_csv`) and its per-group outputs aren't consumable by `subtract`. Don't build on it without reading `docs/markdown/stages/diff_pipeline.md` §6.
9. **PS1 star removal zeroes whole SEP segments** (no interpolation), catalog pass at `tess_mag < 13` plus saturation-flag pass; records land in `*_removed_stars.csv`.
10. Band combination weights: r=0.238, i=0.344, z=0.283, y=0.135 (four bands, not three).
11. **Star gepsf uses the baseline workspace**: every `psf_type: epsf` method requires `inputs.epsf: {label}`. An optional `epsf` block builds `{baseline_ws}/{epsf.output}` and must use the same label; without it, that catalog must already exist. `epsf.inputs.diffs` optionally overrides the source diff label. Photometry loads the matching per-frame model for each star stamp.
12. **Streaming PS1 changes the effective DAG**: with `ps1_process.ps1_source: stream`, `ps1_download` is skipped and `ps1_process` depends directly on `mapping`.

## Key schemas

- `cluster_template_job.json`: `schema_version`, `reference_ffi_path`, `reference_ffi_basename`, `sector/camera/ccd`, `offset_threshold`, `groups: [{group_id, group_dx, group_dy, n_frames}]`, `crop_mode`, `crop_box_size`, crop `x_min…y_max`, `shape`.
- `syndiff_ffi_frames.csv`: `filename, path, wcs_ok, DATE-OBS, btjd, x_pix, y_pix, delta_x(_raw), delta_y(_raw), earth_deg, moon_deg, group_id, group_dx, group_dy` (+ hotpants status columns appended by diff).
- SCC `convolved_results/*_removed_stars.csv`: PS1-process records including `source_id`, `ra/dec`, `pixel_x/y`, magnitudes, segment centroid/flux/id, and `removal_reason`.
- Event `ps1_removed_stars.csv`: the target crop subset projected to crop-local `x/y`; star uses it to skip hosts already absent from the production template.

## Improvement plans

- `.cursor/plans/spatially_varying_wcs_templates.plan.md` — drift field + incremental per-skycell template cache.
- `.cursor/plans/exoplanet_star_removed_template.plan.md` — historical design superseded by the implemented `star` branch; consult current star docs instead.

## Verified: rotation/shear is not the translation residual (Jul 2026)

Campaign `experimental/grid_wcs_correction/` (s0020/c3/k2): frozen regmap + single integer PS1 roll leaves **~0.37 PS1 px** centroid residual vs exact per-frame mapping (10 btjd-spaced frames). **Local affine / per-pixel integer shifts do not reduce it** (Jacobian ~10⁻⁵; ~1.2 unique shifts per skycell footprint). Fix = **cached exact per-epoch regmaps** (seam-remap pattern), not rotation modeling. See `experimental/grid_wcs_correction/13_rotation_fix_recommendation.md`.
