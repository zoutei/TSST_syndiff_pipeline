> **Package integration**: diff sub-stages `kernel_fit`, `convolved_templates`, `kernel_subtract` · modules under `difference_imaging/stages/` · configured in `config/diff_config_single_kernel.yaml` and relatives  
> **Related docs**: [diff pipeline internals](diff_pipeline.md) · [background stage](background.md) · [forced photometry](forced_photometry.md) · [oversampled templates](../oversampled_templates.md) · [linear centroids campaign](../linear_centroids_pipeline.md)

# Multi-kernel difference imaging (`kernel_fit` → `convolved_templates` → `kernel_subtract`)

Alternative to per-FFI **Hotpants**: fit **one target-level PSF kernel** on a carefully chosen science frame, convolve each WCS-group template with that fixed kernel, then form per-epoch difference images by **algebraic subtraction** plus a shared robust-TESSreduce background estimate (biharmonic inpainting with KNN boundary sigma-clip; see [background stage](background.md) for the *separate*, optional downstream Savitzky–Golay smoothing stage). Downstream stages (`background`, `subtract`, `hotpants` round 2, `forced_photometry`) consume the `ks_*` / `hp_*` labels this path produces.

---

## When to use vs Hotpants

| Aspect | Kernel-fit path | Default Hotpants (`hotpants`) |
|--------|-----------------|--------------------------------|
| Kernel | One `kernel_r2.npz` per target (min-background FFI) | Per-FFI kernel fit (discarded unless `write_kernel_solutions: true`) |
| Per-frame work | Cheap subtract + robust-TESSreduce bkg | Full Hotpants each FFI |
| Template swap | Re-run `convolved_templates` + `kernel_subtract` with same `kernel_r2.npz` | Must re-run Hotpants |
| PSF variability | Single kernel for all epochs | Adapts per frame |
| Typical configs | `diff_config_single_kernel.yaml`, `diff_config_multi_kernel.yaml`, linear-centroids phase 1b | `diff_config.yaml` (production default) |

**Multi-kernel** (`diff_config_multi_kernel.yaml`) runs this prefix, then `background` on `ks_b`, then a **second Hotpants pass** (`hp_bgo=0`) on temporally smoothed backgrounds — combining a stable kernel with per-epoch Hotpants refinement.

---

## Pipeline order

```text
shared_mask
  → kernel_fit          # HP1(bgo) → bkg1 (robust tessreduce) → HP2(bgo=0) → bkg2 → HP3(bgo=0, final) → kernel_r2.npz
  → convolved_templates # each unique template × kernel_r2
  → kernel_subtract     # ffi − convolved_template → diff_raw; robust-tessreduce bkg → ks_d (bkg-subtracted), ks_b (bkg)
  → background          # optional: further spatial/temporal/strap refinement -- NOTE: since ks_d is now
                         # already background-subtracted, chaining this stage after kernel_subtract will
                         # re-run its own spatial photutils estimate on an already-cleaned diff; verify it's
                         # still wanted before enabling both (see caveat in Stage 3 below).
  → subtract            # optional: ks_d + ks_b − ks_b_s → ks_d_s
  → hotpants            # optional round 2 (multi-kernel configs)
  → forced_photometry
```

Example YAML: [`config/diff_config_single_kernel.yaml`](../../../config/diff_config_single_kernel.yaml).

---

## Stage 1: `kernel_fit` (`kernel_fit.py`)

Fits the shared kernel on the **minimum Earth/Moon angle** FFI (`pick_best_angle_ffi` with configurable `weighting_factor`).

**Three-round hotpants + robust-TESSreduce recipe** (on the chosen frame only; see
[`background/tessreduce_residual.py`](../../../syndiff_pipeline/difference_imaging/stages/background/tessreduce_residual.py)
for the shared estimator both `kernel_fit` and `kernel_subtract` call):

1. **HP1** — `hp_bgo` at the stage-configured order (default 3); kernel params not collected. `conv1` = its convolved template.
2. `diff1 = ffi − conv1`; `bkg1 = estimate_tessreduce_residual_background(diff1, ...)` (biharmonic inpainting + KNN boundary sigma-clip); `cleaned_ffi1 = ffi − bkg1`.
3. **HP2** — `sci=cleaned_ffi1`, `hp_bgo=0`; kernel params not collected. `conv2` = its convolved template.
4. `diff2 = ffi − conv2`; `bkg2 = estimate_tessreduce_residual_background(diff2, ...)`; `cleaned_ffi2 = ffi − bkg2`.
5. **HP3 (final)** — `sci=cleaned_ffi2`, `hp_bgo=0`; extract `kernel_solution` and convolved template -- these are what get persisted.

**Outputs** (under the stage `output` label, e.g. `kernel_fit/`):

| Artifact | Contents |
|----------|----------|
| `kernel_r2.npz` | `kernel_solution`, `kernel_image`, `basis`, Hotpants scalar arrays (from HP3) |
| `kernel_fit_meta.json` | `min_bg_ffi_path`, `product_id`, `group_dx`/`group_dy` or field `group_id`, `reference_kernel_sum`, fit params (including `tessreduce_boundary_k`/`tessreduce_boundary_sigma`/`tessreduce_boundary_rim_width`) |

`skip_existing: true` (default) reloads both files when present.

**Field mode:** template is assembled from the SCC field store (`assemble_field_template_for_ffi`); `mapping_grid` may apply science/template pairing padding. Same Hotpants oversample / stamp wiring as the `hotpants` stage.

---

## Stage 2: `convolved_templates` (`convolved_templates.py` + `kernel.py`)

Loads `kernel_r2.npz`, builds a Hotpants config with `hp_bgo=0`, and convolves each template via `convolve_template_with_kernel_solution()` (pyhotpants `KernelModel` or pure-Python path when `oversample > 1`).

**Linear mode** — one output per unique `(group_dx, group_dy)` from `template_paths`:

```text
convolved_template_dx{±D.DDD}_dy{±D.DDD}.fits.fz
```

**Field mode** — one output per distinct `group_id` in the frame manifest:

```text
convolved_template_gid{N}.fits.fz
```

Manifest: `convolved_templates.csv` with columns `group_id`, `group_dx`, `group_dy`, `template_path`, `convolved_path` (`group_dx`/`group_dy` are NaN in field mode).

At `F>1`, basis sigmas scale as `σ/F²` and output maps are **native resolution** (see [oversampled templates](../oversampled_templates.md)).

### Patch-cache convolution (`use_patch_cache`, field mode + `F>1` only)

`stages.convolved_templates.use_patch_cache: true` (default `false`) replaces
the default "convolve the whole assembled group template fresh, per group"
path with a precomputed per-basis-function convolution cache
(`convolved_templates_patch_cache.py`), motivated by heavy per-skycell reuse
across the ~1000+ distinct `group_id`s in one real SCC (~7h serial bottleneck
observed on s0020/c3/k3 at ~23s/group).

The math: convolution is linear over the DIA kernel's basis-function
decomposition, `conv(image, Σc_i·basis_i) = Σc_i·conv(image, basis_i)`, and
Lane A fits exactly **one** global `kernel_solution` per SCC run (`kernel_fit`
runs `run_hotpants_frame` once, reused unchanged by every group) — so each
skycell patch's basis convolutions can be computed once and reused across
every group sharing that skycell, instead of re-convolving the whole
assembled group template per group.

Mechanics:

- Requires the [interior/seam-delta split store](../field_geometry.md#optional-interior-seam-delta-split-write_split_contribs) (`write_split_contribs: true` on `downsample`) to already be **complete for every skycell touched by every group in the run** — `assemble_group_from_split_contribs`-style assembly raises if even one skycell's interior contrib is missing, so a partial/scoped split-contrib backfill cannot back a real production `use_patch_cache` run; it only supports single-skycell or fully-covered-subset validation.
- Per `(skycell, sx_int, sy_int)`, materializes the interior contrib densely over `footprint ⊕ 2×hw_kernel` (dilated by **twice** the kernel half-width, not once — "valid"-mode convolution itself consumes one `hw_kernel` of padding, and the desired output margin is a *second*, additional `hw_kernel`; getting this factor wrong was caught by `test_patch_scatter_add_matches_whole_image_convolution` during development) and convolves it against each basis function via `hotpants/pure/os_precompute.py::precompute_basis_lr_maps`. Same construction for each skycell's seam delta, restricted to the thin rim band.
- Caches results under the **same** field template store as `basis_conv/interior/{skycell}_sx{±}_sy{±}.npz` and `basis_conv/seam_delta/{skycell}_sx{±}_sy{±}_gid{N}.npz` — see [field_geometry.md storage layout](../field_geometry.md#storage-layout).
- Group assembly scatter-adds each constituent skycell's `(n_basis, ny, nx)` interior + seam-delta maps, then recombines with per-block kernel coefficients (`recombine_basis_maps_full`, a pixel-for-pixel port of `hotpants.pure.convolution.jit_spatial_convolve`'s blocking contract) — this is the one genuinely new numerical code path and is float64 throughout (today's default path runs in float32, so results will not match to float32's own machine epsilon, only somewhat below it).
- Falls back to the default per-group convolution path when `use_patch_cache` is unset/`false`; both paths coexist in `convolved_templates.py` and can be A/B compared on the same SCC.

Status as of 2026-08-23: implemented and verified on synthetic data (halo-exactness, block-recombination-vs-`jit_spatial_convolve`, and a full crop-aware H.1+H.2 integration test) and on real s0020/c3/k3 data restricted to a same-batch split-contrib subset (0 mismatches once compared against a *fresh* plain-contrib snapshot rather than pre-refit orphaned files — see the field_geometry.md verification pitfall note). Not yet validated end-to-end on a full real SCC production run; blocked, independent of this code, on backfilling a subset of that SCC's shared `ps1_combined.zarr` cells to the current recipe fingerprint (see the pipeline-map skill's combined-store fingerprint migration note) before a full-coverage split-contrib store can be built.

---

## Stage 3: `kernel_subtract` (`kernel_subtract.py`)

Per-FFI loop (joblib `loky`, `n_jobs` from config):

```text
diff_raw = ffi_crop − convolved_template
tessreduce_bkg = estimate_tessreduce_residual_background(diff_raw, mask, ...)  # same shared estimator as kernel_fit
diff_final = diff_raw − tessreduce_bkg
```

- **Diff FITS** (`output.diffs`, e.g. `ks_d`) — **background-subtracted** (`diff_final`), science-ready.
- **Background FITS** (`output.phot_bkg`, e.g. `ks_b`) — the robust-TESSreduce background plane, still saved separately for diagnostics/the star branch.

> **Caveat:** because `ks_d` is now already background-subtracted, chaining an optional downstream `background` stage (spatial photutils + temporal Savitzky–Golay + strap, see [background stage](background.md)) after `kernel_subtract` will re-estimate/re-subtract background from an already-cleaned diff. Configs that enable both (`diff_config_single_kernel.yaml`, `diff_config_multi_kernel.yaml`) should be reviewed/re-tuned before relying on this combination; configs that only use `kernel_fit` → `kernel_subtract` are unaffected and now receive a properly background-subtracted diff where previously they did not.

Template lookup:

- Linear: `resolve_template_for_ffi` → `lookup_convolved_path(group_dx, group_dy)`.
- Field: `lookup_convolved_path_by_group_id(group_id_for_ffi(...))`.

**SCC-primary storage:** with `data_root` set, writes go to `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/{label}/` via `resolve_diff_write_path` (not event `ws/`). Provenance indexed in `bookkeeping/provenance.db` when enabled.

The star branch expects `ks_b` (or smoothed `ks_b_s`) for host stamps — see [diff pipeline §6](diff_pipeline.md#6-sat_template--current-behavior-and-known-gaps).

---

## Kernel persistence compared to Hotpants

| Location | Contents | Granularity |
|----------|----------|-------------|
| `kernel_fit/kernel_r2.npz` | Target-level kernel from HP3 (final round) | One per target |
| `convolved_templates/` | Pre-convolved template FITS + CSV | One per offset group or `group_id` |
| `{diffs}_kernels/*.npz` | Per-frame Hotpants kernels | Only when `write_kernel_solutions: true` on a `hotpants` stage |

Reusing a modified template set: keep `kernel_r2.npz`, clear or `skip_existing: false` on `convolved_templates`, re-run `convolved_templates` → `kernel_subtract` → downstream.

---

## Example SCC lane layout (`diff_linear/`)

From the [linear centroids campaign](../linear_centroids_pipeline.md):

| Label | Role |
|-------|------|
| `kernel_fit/` | `kernel_r2.npz`, `kernel_fit_meta.json` |
| `tmpl_conv/` | Convolved templates + `convolved_templates.csv` |
| `ks_d/`, `ks_b/` | Per-FFI background-subtracted diffs and robust-TESSreduce backgrounds |
| `hp_d/`, `hp_b/` | Optional second Hotpants leg |

Paths are under `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/` when using SCC field-mode v2 handoff.

---

## Key YAML parameters

### `kernel_fit`

| Key | Default | Notes |
|-----|---------|-------|
| `weighting_factor` | 0.5 | Earth/Moon angle ranking |
| `tessreduce_smooth_gauss` | 2.0 | Gaussian smoothing after biharmonic gap-fill |
| `tessreduce_anomaly_gauss` | 2.0 | Anomaly-repair smoothing (`fix_bkg_frame_decomposed`) |
| `tessreduce_qe_spline_degree` | 2 | Strap QE B-spline degree |
| `tessreduce_qe_spline_smooth_mult` | 10.0 | Strap QE B-spline smoothing multiplier |
| `tessreduce_boundary_k` | 15 | KNN neighbors for boundary sigma-clip |
| `tessreduce_boundary_sigma` | 3.0 | Boundary sigma-clip threshold |
| `tessreduce_boundary_rim_width` | 1 | Dilation width defining the mask boundary rim |
| `output` | `kernel_fit` | Artifact directory label |
| `hp_*` | HotpantsParams | Round 1 uses stage `hp_bgo`; rounds 2 and 3 force `bgo=0` internally |

### `convolved_templates`

| Key | Role |
|-----|------|
| `inputs.kernel_fit` | Directory with `kernel_r2.npz` |
| `output` | Convolved template workspace label |
| `skip_existing` | Reuse valid FITS + manifest |
| `use_patch_cache` | `false` by default; field mode + `F>1` only — see [patch-cache convolution](#patch-cache-convolution-use_patch_cache-field-mode--f1-only) above. Requires a complete `write_split_contribs` store. |

### `kernel_subtract`

| Key | Role |
|-----|------|
| `inputs.convolved` | Convolved-templates label |
| `tessreduce_smooth_gauss` / `tessreduce_anomaly_gauss` / `tessreduce_qe_spline_degree` / `tessreduce_qe_spline_smooth_mult` | Same robust-TESSreduce knobs as `kernel_fit` (shared estimator) |
| `tessreduce_boundary_k` / `tessreduce_boundary_sigma` / `tessreduce_boundary_rim_width` | Boundary sigma-clip knobs (same defaults as `kernel_fit`) |
| `output.diffs` / `output.phot_bkg` | e.g. `ks_d` (background-subtracted), `ks_b` (background plane) |

---

## Key source files

| Module | Role |
|--------|------|
| [`kernel_fit.py`](../../../syndiff_pipeline/difference_imaging/stages/kernel_fit.py) | Min-background FFI selection, HP1/HP2 fit |
| [`convolved_templates.py`](../../../syndiff_pipeline/difference_imaging/stages/convolved_templates.py) | Template convolution + manifest |
| [`kernel_subtract.py`](../../../syndiff_pipeline/difference_imaging/stages/kernel_subtract.py) | Per-FFI subtract + bkg |
| [`kernel.py`](../../../syndiff_pipeline/difference_imaging/stages/kernel.py) | `convolve_template_with_kernel_solution`, NPZ constants |
| [`execute.py`](../../../syndiff_pipeline/difference_imaging/orchestration/execute.py) | Sub-stage driver |
