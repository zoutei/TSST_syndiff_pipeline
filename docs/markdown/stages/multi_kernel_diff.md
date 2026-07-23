> **Package integration**: diff sub-stages `kernel_fit`, `convolved_templates`, `kernel_subtract` · modules under `difference_imaging/stages/` · configured in `config/diff_config_single_kernel.yaml` and relatives  
> **Related docs**: [diff pipeline internals](diff_pipeline.md) · [background stage](background.md) · [forced photometry](forced_photometry.md) · [oversampled templates](../oversampled_templates.md) · [linear centroids campaign](../linear_centroids_pipeline.md)

# Multi-kernel difference imaging (`kernel_fit` → `convolved_templates` → `kernel_subtract`)

Alternative to per-FFI **Hotpants**: fit **one target-level PSF kernel** on a carefully chosen science frame, convolve each WCS-group template with that fixed kernel, then form per-epoch difference images by **algebraic subtraction** plus a photutils background estimate. Downstream stages (`background`, `subtract`, `hotpants` round 2, `forced_photometry`) consume the `ks_*` / `hp_*` labels this path produces.

---

## When to use vs Hotpants

| Aspect | Kernel-fit path | Default Hotpants (`hotpants`) |
|--------|-----------------|--------------------------------|
| Kernel | One `kernel_r2.npz` per target (min-background FFI) | Per-FFI kernel fit (discarded unless `write_kernel_solutions: true`) |
| Per-frame work | Cheap subtract + photutils bkg | Full Hotpants each FFI |
| Template swap | Re-run `convolved_templates` + `kernel_subtract` with same `kernel_r2.npz` | Must re-run Hotpants |
| PSF variability | Single kernel for all epochs | Adapts per frame |
| Typical configs | `diff_config_single_kernel.yaml`, `diff_config_multi_kernel.yaml`, linear-centroids phase 1b | `diff_config.yaml` (production default) |

**Multi-kernel** (`diff_config_multi_kernel.yaml`) runs this prefix, then `background` on `ks_b`, then a **second Hotpants pass** (`hp_bgo=0`) on temporally smoothed backgrounds — combining a stable kernel with per-epoch Hotpants refinement.

---

## Pipeline order

```text
shared_mask
  → kernel_fit          # HP1 (bgo=3) → photutils bkg on diff → HP2 (bgo=0) → kernel_r2.npz
  → convolved_templates # each unique template × kernel_r2
  → kernel_subtract     # ffi − convolved_template → ks_d; photutils bkg on diff → ks_b
  → background          # optional: ks_b → ks_b_s (temporal Savitzky–Golay)
  → subtract            # optional: ks_d + ks_b − ks_b_s → ks_d_s
  → hotpants            # optional round 2 (multi-kernel configs)
  → forced_photometry
```

Example YAML: [`config/diff_config_single_kernel.yaml`](../../../config/diff_config_single_kernel.yaml).

---

## Stage 1: `kernel_fit` (`kernel_fit.py`)

Fits the shared kernel on the **minimum Earth/Moon angle** FFI (`pick_best_angle_ffi` with configurable `weighting_factor`).

**Hotpants two-pass recipe** (on the chosen frame only):

1. **HP1** — `hp_bgo=3` (background order 3); kernel params not collected.
2. **Photutils** — `photutils_background_masked` on HP1 diff (`phot_box_size`, shared mask).
3. **HP2** — science cleaned: `ffi − hp1_bkg − phot_bkg`; `hp_bgo=0`; extract `kernel_solution`.

**Outputs** (under the stage `output` label, e.g. `kernel_fit/`):

| Artifact | Contents |
|----------|----------|
| `kernel_r2.npz` | `kernel_solution`, `kernel_image`, `basis`, Hotpants scalar arrays |
| `kernel_fit_meta.json` | `min_bg_ffi_path`, `product_id`, `group_dx`/`group_dy` or field `group_id`, `reference_kernel_sum`, fit params |

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

---

## Stage 3: `kernel_subtract` (`kernel_subtract.py`)

Per-FFI loop (joblib `loky`, `n_jobs` from config):

```text
diff_raw = ffi_crop − convolved_template
phot_bkg = photutils_background_masked(diff_raw, mask, phot_box_size)
```

- **Diff FITS** (`output.diffs`, e.g. `ks_d`) — **not** background-subtracted.
- **Background FITS** (`output.phot_bkg`, e.g. `ks_b`) — separate plane when configured.

Template lookup:

- Linear: `resolve_template_for_ffi` → `lookup_convolved_path(group_dx, group_dy)`.
- Field: `lookup_convolved_path_by_group_id(group_id_for_ffi(...))`.

**SCC-primary storage:** with `data_root` set, writes go to `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/{label}/` via `resolve_diff_write_path` (not event `ws/`). Provenance indexed in `bookkeeping/provenance.db` when enabled.

The star branch expects `ks_b` (or smoothed `ks_b_s`) for host stamps — see [diff pipeline §6](diff_pipeline.md#6-sat_template--current-behavior-and-known-gaps).

---

## Kernel persistence compared to Hotpants

| Location | Contents | Granularity |
|----------|----------|-------------|
| `kernel_fit/kernel_r2.npz` | Target-level kernel from HP2 | One per target |
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
| `ks_d/`, `ks_b/` | Per-FFI diffs and photutils backgrounds |
| `hp_d/`, `hp_b/` | Optional second Hotpants leg |

Paths are under `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/` when using SCC field-mode v2 handoff.

---

## Key YAML parameters

### `kernel_fit`

| Key | Default | Notes |
|-----|---------|-------|
| `weighting_factor` | 0.5 | Earth/Moon angle ranking |
| `phot_box_size` | 4 | Photutils mesh on HP1 diff |
| `output` | `kernel_fit` | Artifact directory label |
| `hp_*` | HotpantsParams | Round 1 uses stage `hp_bgo`; round 2 forces `bgo=0` internally |

### `convolved_templates`

| Key | Role |
|-----|------|
| `inputs.kernel_fit` | Directory with `kernel_r2.npz` |
| `output` | Convolved template workspace label |
| `skip_existing` | Reuse valid FITS + manifest |

### `kernel_subtract`

| Key | Role |
|-----|------|
| `inputs.convolved` | Convolved-templates label |
| `phot_box_size` | Photutils background on algebraic diff |
| `output.diffs` / `output.phot_bkg` | e.g. `ks_d`, `ks_b` |

---

## Key source files

| Module | Role |
|--------|------|
| [`kernel_fit.py`](../../../syndiff_pipeline/difference_imaging/stages/kernel_fit.py) | Min-background FFI selection, HP1/HP2 fit |
| [`convolved_templates.py`](../../../syndiff_pipeline/difference_imaging/stages/convolved_templates.py) | Template convolution + manifest |
| [`kernel_subtract.py`](../../../syndiff_pipeline/difference_imaging/stages/kernel_subtract.py) | Per-FFI subtract + bkg |
| [`kernel.py`](../../../syndiff_pipeline/difference_imaging/stages/kernel.py) | `convolve_template_with_kernel_solution`, NPZ constants |
| [`execute.py`](../../../syndiff_pipeline/difference_imaging/orchestration/execute.py) | Sub-stage driver |
