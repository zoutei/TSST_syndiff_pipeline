> **Package integration**: `syndiff` stage `wcs_grouping` · modules `common/wcs_grouping.py` + `template_creation/orchestration/handoff.py` · runner-only (no legacy script)  
> **Orchestration docs**: [template pipeline guide](../template_pipeline.md)

# WCS grouping — Detailed Technical Reference

The `wcs_grouping` stage measures how the science target's **pixel position drifts across the sector** (because each FFI carries its own WCS solution, including SIP distortion), smooths that drift in time, assigns each FFI to a **template offset group**, selects a **reference FFI**, and resolves the **image crop (ROI)**. Its single JSON handoff, `cluster_template_job.json`, drives `mapping`, `downsample`, and `diff`.

With **`geometry_mode: field`**, target-anchored groups from this stage are supplemented
by a separate per-skycell shift schedule built during the `templates` stage — see
[field geometry](../field_geometry.md). The linear-mode flow below still applies for
reference-FFI selection, crop bounds, and `syndiff_ffi_frames.csv`.

This stage is fast (header-only FITS reads) and runs unpooled on the submit host.

---

## Table of contents

1. [Concepts](#1-concepts)
2. [Execution flow](#2-execution-flow)
3. [Drift measurement](#3-drift-measurement)
4. [Savitzky–Golay smoothing](#4-savitzkygolay-smoothing)
5. [Reference FFI selection](#5-reference-ffi-selection)
6. [Template group assignment](#6-template-group-assignment)
7. [Crop bounds (ROI)](#7-crop-bounds-roi)
8. [Output artifacts](#8-output-artifacts)
9. [Downstream consumers](#9-downstream-consumers)
10. [Known limitation: single-point drift](#10-known-limitation-single-point-drift)
11. [Configuration reference](#11-configuration-reference)

---

## 1. Concepts

- **Drift** (`delta_x`, `delta_y`): the target's pixel position in frame *t* minus its position in the **reference FFI**, both computed by projecting the same target RA/Dec through each frame's own WCS (`WCS.world_to_pixel_values`). Drift is caused by pointing jitter, differential velocity aberration, and evolving distortion — it is typically ≲ 0.1 px over a sector but not constant.
- **Template group**: a set of frames whose smoothed drift rounds to the same `(group_dx, group_dy)` on an `offset_threshold` grid (default **0.01 px**). One PS1-based template is built per group by `downsample`, shifted by `(group_dx, group_dy)`.
- **Why 0.01 px**: `downsample` realizes the offset by rolling PS1 images an **integer number of PS1 pixels** per skycell. One PS1 pixel (0.258″) ≈ 0.0124 TESS px (21″), so ~0.01 TESS px is the natural quantization — finer groups would produce identical templates.

## 2. Execution flow

Driver: `run_wcs_grouping()` in `template_creation/orchestration/handoff.py`.

1. List FFIs on disk under `{ffi_dir}/s{SSSS}/cam{C}_ccd{K}/` (`common/download.py: nested_ffi_dir`, `list_local_ffis`), time-sorted by filename.
2. `select_ffis_with_valid_target_wcs(...)` — with `max_ffis`, scan in order and keep only frames whose WCS maps the target to finite pixels; without it, keep all.
3. `build_wcs_table(...)` — per-FFI header read (HDU 1, memmap), target pixel position, provisional drift vs the first valid frame, `btjd` from `DATE-OBS`.
4. `smooth_wcs_drift_savgol(...)` — SG filter over the time-ordered valid frames (see §4).
5. `attach_tessvector_earth_moon_angles(...)` — interpolate TESSVectors FFI CSV onto each frame's `btjd`, adding `earth_deg` / `moon_deg` (camera–body angles).
6. `finalize_wcs_table_with_reference_anchor(...)`:
   - `choose_reference_ffi_path(...)` (unless an explicit `ref_ffi_path` was passed) — see §5.
   - `reanchor_wcs_drift_to_reference(...)` — subtract the reference frame's smoothed drift so the reference sits at (0, 0).
   - `assign_template_groups(...)` — see §6.
7. Write `syndiff_ffi_frames.csv` (the frame manifest).
8. Resolve crop bounds from the **reference FFI header** (`resolve_crop_bounds_from_params`, see §7).
9. Write `cluster_template_job.json` and the debug plot.

## 3. Drift measurement

`build_wcs_table()` produces one row per FFI:

| Column | Meaning |
|--------|---------|
| `filename`, `path` | FFI file (`.fits.fz` / `.fits.gz` / `.fits` all resolved) |
| `wcs_ok` | Header has CRVAL/CRPIX/CD keys and projection succeeded |
| `DATE-OBS`, `btjd` | Frame timestamp (BTJD = BJD − 2457000) |
| `x_pix`, `y_pix` | Target pixel position through **this frame's** WCS (0-based, full-FFI) |
| `delta_x`, `delta_y` | Drift; re-anchored to the reference FFI in step 6 |

Projection uses `world_to_pixel_values` (not iterative `all_world2pix`) to avoid SIP convergence warnings; the WCS is built from the full HDU 1 header, so **SIP distortion is included** in the target-position evaluation.

## 4. Savitzky–Golay smoothing

`smooth_wcs_drift_savgol(window_length=11, polyorder=2)` filters `delta_x`/`delta_y` along the time-ordered valid frames. Raw values are preserved in `delta_x_raw` / `delta_y_raw`. The window is capped/odd-adjusted to the number of valid samples; smoothing is skipped (< 3 samples or window `None`/`< 3`) without adding raw columns. Grouping and reference selection then use the **smoothed** drift, so single-frame WCS noise does not spawn spurious template groups.

## 5. Reference FFI selection

`choose_reference_ffi_path()` picks the frame closest (in smoothed-drift space) to the **median smoothed drift**, subject to quality gates, with graceful fallback:

1. raw−smooth residual ≤ `max_smoothed_residual` (0.05 px) **and** `earth_deg ≥ 45°`, `moon_deg ≥ 25°` (scatter-light screening);
2. angle cuts only;
3. residual gate only;
4. any usable row.

The chosen path is stored as `reference_ffi_path` in the JSON. Everything downstream — mapping geometry, template WCS, Gaia projections, diff crop — is anchored to this frame's WCS.

## 6. Template group assignment

`assign_template_groups(offset_threshold=0.01)`:

```
dx_rounded = round(delta_x / offset_threshold) * offset_threshold   # same for dy
group_id   = index of first occurrence of (dx_rounded, dy_rounded)
```

Adds `group_id`, `group_dx`, `group_dy` per frame (`-1`/NaN for invalid WCS). `summarize_template_groups()` builds the per-group table (`group_id`, `group_dx`, `group_dy`, `n_frames`) embedded in the JSON as `groups`. A typical sector yields a handful of groups.

## 7. Crop bounds (ROI)

`resolve_crop_bounds_from_params()` (shared with diff bootstrap):

| Mode | Behavior |
|------|----------|
| explicit `x_min…y_max` | Any given edge wins; missing edges fall back to the usable rectangle; clamped to FFI |
| `full` | Entire FFI including dead columns |
| `tl` / `tr` / `bl` / `br` | Quadrants of the **usable** rectangle (dead strips removed, chip midlines) |
| `target_box` | Square of `crop_box_size` (default 1024) centered on the target, edge-clamped |

Usable area excludes `x_left_dead`/`x_right_dead` (44 px each) and the top `y_edge_strip` (30 px). Bounds are `[min, max)` in **full-FFI pixels**; the JSON stores them plus `shape` `[ny, nx]`.

## 8. Output artifacts

All under `{workspace_root}/events/{target_label}/`:

| File | Contents |
|------|----------|
| `syndiff_ffi_frames.csv` | The frame manifest: one row per FFI with all §3–§6 columns (plus `earth_deg`, `moon_deg`, and raw drift columns). The diff stage later appends Hotpants status columns to this file. |
| `cluster_template_job.json` | `schema_version`, `reference_ffi_basename`, `reference_ffi_path`, `sector`, `camera`, `ccd`, `offset_threshold`, `groups` (list of `{group_id, group_dx, group_dy, n_frames}`), crop `x_min…y_max` + `shape`, optional `crop_mode` / `crop_box_size` |
| `plots/wcs_drift_template_debug.png` | Four stacked panels: `delta_x`, `delta_y` (raw scatter + smoothed line), `group_id`, Earth/Moon angles vs BTJD, with reference-FFI vline |

## 9. Downstream consumers

| Consumer | What it reads |
|----------|---------------|
| `mapping` (PanCAKES) | `reference_ffi_path` — the *only* WCS the whole mapping is built against; also drives the Gaia catalog footprint |
| `downsample` | `groups` → the unique `(dx, dy)` offset list; crop bounds → ROI; `reference_ffi_basename` → consistency check against the master mapping's `TESS_FFI` header |
| `diff` | Crop bounds (unless `diff_config` overrides with `crop_mode`/explicit bounds); per-frame `group_id`/`group_dx`/`group_dy` from `syndiff_ffi_frames.csv` for template selection (`support/template_resolution.py`) |
| `downsample` post-step | Reference WCS to project PS1 removed stars into crop-local coordinates (`events/{label}/ps1_removed_stars.csv`) |

## 10. Single-point drift (linear mode) and field-mode fix

This section applies to **`geometry_mode: linear`** (the default). For
**`geometry_mode: field`**, drift is measured at every skycell center and hybrid
Exact patches fix seam/rim errors — see
[field geometry](../field_geometry.md).

### Linear-mode limitation

Drift is measured **only at the target's sky position**. The applied template offset is then treated as constant across the chip: `downsample` converts the single `(group_dx, group_dy)` into a per-skycell PS1 shift via a WCS round-trip *at each skycell center*, but the TESS-pixel drift input itself is one number per frame.

In reality the drift field varies over the focal plane (distortion evolves through the sector; changes are larger toward the FFI edges). Consequences:

- Templates are correctly positioned near the science target but degrade with distance from it.
- Light curves of objects far from the target (e.g. `additional_forced_targets` at large offsets, or secondary science targets) see template misregistration residuals.

Two properties of the existing design matter for any fix: (1) the expensive mapping stage does **not** need re-running for sub-pixel WCS changes — only the cheap per-skycell shift computation does; (2) the natural recomputation quantum is one PS1 pixel ≈ 0.0124 TESS px, matching `offset_threshold`.

### Linear-mode workarounds

- **Field mode (recommended for full-chip science):** set `geometry_mode: field` in
  `stages.wcs_grouping` and `stages.templates`. See [field_geometry.md](../field_geometry.md).
- **Re-target workaround:** for an object away from the main target, re-run `wcs_grouping`
  with that object's RA/Dec as the target (new event label), then re-run only `templates`
  (and `diff`). `mapping`, `ps1_download`, and `ps1_process` outputs are SCC-wide and
  are reused as-is.

## 11. Configuration reference

`stages.wcs_grouping` in `pipeline.yaml` (see `WcsGroupingStageParams`):

| Key | Default | Meaning |
|-----|---------|---------|
| `offset_threshold` | `0.01` | Group grid spacing (TESS px) |
| `wcs_drift_savgol_window` | `11` | SG window (odd; capped to sample count) |
| `wcs_drift_savgol_polyorder` | `2` | SG polynomial order |
| `bkg_vector_path` | null | Local TESSVectors data path (else downloaded) |
| `crop_mode` | `"full"` | `full` / `tl` / `tr` / `bl` / `br` / `target_box` |
| `crop_box_size` | `1024` | Side length for `target_box` |
| `x_min` … `y_max` | null | Explicit crop bounds (override presets) |
| `x_left_dead`, `x_right_dead` | `44` | Dead columns excluded from usable area |
| `y_edge_strip` | `30` | Top dead rows excluded from usable area |
| `geometry_mode` | `"linear"` | `"linear"` or `"field"` — field uses per-skycell schedule in `templates`; see [field_geometry.md](../field_geometry.md) |
| `grouping_quantum_ps1_px` | `1.0` | Field-mode signature quantum (PS1 px) |

Reference selection thresholds (`earth_deg_min=45`, `moon_deg_min=25`, `max_smoothed_residual=0.05`) are function defaults in `common/wcs_grouping.py`, not YAML keys.
