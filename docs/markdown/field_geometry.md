# Field (distortion-aware) templates

**Field mode is the default** (`geometry_mode: field` on `stages.wcs_grouping`
and `stages.downsample`). Templates are built once per SCC from per-skycell WCS
drift, hybrid Exact patches, and a sparse contrib store; difference imaging
assembles a full-chip template per `group_id` on demand.

Linear mode (`geometry_mode: linear`) remains available as an explicit opt-out
for target-anchored single-offset templates — see
[WCS grouping](stages/wcs_grouping.md) and
[Multi-offset downsample](stages/downsample_technical.md).

```{contents}
:local:
:depth: 2
```

Also see [oversampled templates](oversampled_templates.md) when combining field
mode with `oversampling_factor F>1`.

---

## Quick start

```bash
mamba activate syndiff

# SCC-only template DAG (remap + downsample are included when geometry_mode=field)
syndiff template submit \
  --site config \
  --scc config/scc_example.csv

# First-time event differencing still needs bind
syndiff diff submit \
  --site config \
  --targets config/targets_example.csv \
  --stages bind,diff
```

Defaults (no YAML required beyond a normal site):

| Knob | Default | Meaning |
|------|---------|---------|
| `stages.wcs_grouping.geometry_mode` | `field` | Event bind records field geometry |
| `stages.downsample.geometry_mode` | `field` | L5 uses `field_downsample` |
| `stages.remap.apply_hybrid_exact` | `true` | L4a Exact on R=1 seam/rim |
| `stages.remap.l4b_policy` | `none` | L4b F2 off until you set `pair_state` |
| `stages.downsample.l4b_policy` | `none` | Must match remap when enabling F2 |

Enable true Type-II (F2) L4b:

```yaml
stages:
  remap:
    l4b_policy: pair_state
    raw_drift_outlier_sigma: 5.0   # pre-SG MAD gate; null disables
  downsample:
    l4b_policy: pair_state
    require_l4b_cache: true       # fail-loud on missing L4b NPZs
```

Opt out to linear:

```yaml
stages:
  wcs_grouping:
    geometry_mode: linear
  downsample:
    geometry_mode: linear
```

---

## Why field mode

### Spatial error (large)

Linear mode applies one `(group_dx, group_dy)` everywhere. True drift is a smooth
field `d(x, y, t)` from velocity aberration. Measuring only at the science target
leaves up to **~6 PS1 px** differential across the FOV — templates are good near
the target and wrong far away.

### Sub-pixel floor (small)

Even with a correct local integer roll, a pure roll of the frozen footprint
disagrees with Exact geometry on **TESS-ownership seams** and the **footprint
rim** (~0.3–0.8% of pixels). Field mode Exact-patches those bands.

### What field mode does

1. Measure drift at **every skycell center** (~10³ points).
2. Savitzky–Golay-smooth each skycell’s drift time series per orbit segment.
3. Integer-quantize each skycell’s PS1 shift with hysteresis.
4. Group frames by full-chip shift **signature** → `group_id`.
5. Roll each skycell’s frozen L0 regmap independently.
6. Exact-patch the R=1 seam/rim (**L4a**) and, when enabled, the inter-skycell
   rim under a shared WCS (**L4b F2**, `l4b_policy: pair_state`).
7. Bin PS1 flux into sparse `contribs/` and assemble per `group_id` at diff time.

---

## Linear vs field

| Aspect | Field (default) | Linear (opt-out) |
|--------|-----------------|------------------|
| Drift measured at | Every skycell center | Science target only |
| Template groups | ~10²–10³ full-chip signatures | ~19 target-anchored |
| Regmap geometry | Roll + L4a (+ optional L4b F2) | Integer roll of frozen L0 |
| Output | SCC `contribs/` + on-demand assemble | Per-event `syndiff_template_*_dx*_dy*.fits.fz` |
| Event dependency at build | None (SCC-wide) | Requires `event_job.json` |
| Template DAG | Includes `remap` before `downsample` | `remap` pre-skipped |
| Deep dive | This document | [wcs_grouping.md](stages/wcs_grouping.md), [downsample_technical.md](stages/downsample_technical.md) |

---

## Architecture (L0–L5)

| Layer | Role | Stage / artifacts | Modules |
|-------|------|-------------------|---------|
| **L0** | Frozen PS1→TESS ownership at mapping-epoch WCS | `mapping` → `mapping/oversampling_{N}/` | `pancakes.py` |
| **L2** | Per-skycell PS1 drift time series | `remap` → `shift_schedule.npz` | `shift_schedule.py`, `field_remap.py` |
| **L3** | Hysteresis integer `(sx,sy)`; signature → `group_id` | Same remap store | `shift_schedule.py` |
| **L4a** | Exact-patch R=1 seam/rim (~9% footprint) | `exact_cache_l4a/` | `hybrid_regmaps.py`, `field_hybrid_exact.py` |
| **L4b (F2)** | Shared-WCS abutting rim (~1.9%) | `exact_cache_l4b/` when `pair_state` | `field_abutting.py`, `field_hybrid_exact.py` |
| **L5** | Compose L4a→L4b; bin PS1; sum contribs per `group_id` | `downsample` → `templates/…/contribs/` | `field_downsample.py`, `field_templates.py` |

```text
L0 frozen regmaps ──► L4a exact_cache_l4a ──┐
                     └► L4b exact_cache_l4b ─┼─► L5 hybrid bin ─► assemble(group_id)
L2 shift_schedule ─► L3 groups ─────────────┘
```

### Template DAG

```text
tess_ffi_download → mapping → ps1_download → ps1_process → downsample
                         └→ remap ──────────────────────────┘
```

`downsample` waits for both `remap` and `ps1_process`. Linear mode pre-skips
`remap` and omits it from `downsample` effective deps.

---

## L2–L3: shift schedule and groups

For each valid FFI frame and each skycell center, remap measures the PS1-pixel
shift of the skycell relative to the mapping-epoch reference WCS, smooths per
orbit with Savitzky–Golay, and hysteresis-rounds to integer `(sx_int, sy_int)`.

A **signature** is the full-chip vector of per-skycell integer shifts. The first
appearance of each distinct signature gets a dense `group_id`. Artifacts:

- `shift_schedule.npz` / `.json`
- `skycell_shift_grid_debug.png` / `skycell_shift_relative_center_debug.png`
- `template_group_shifts.parquet`
- `template_groups.json` (`group_id`, `signature_hash`, frame membership)

### Non-measurable frames (missing WCS / sigma-clipped)

Frames with `wcs_ok=False` (or empty WCS headers) and frames whose raw TESS
drift fails a pre-SG MAD gate (`stages.remap.raw_drift_outlier_sigma`, default
`5.0`) are **not dropped** from remap or template. They are marked
non-measurable for Savitzky–Golay, then **synthesized**:

| Gap position | Policy (v1) |
|--------------|-------------|
| Interior | Hold last measurable quantized `(sx_int, sy_int)`; floats = `float(int)` |
| Leading / trailing | Flat-line extrapolation (constant = first / last measurable values) |

Orbit segments for SG come from MIT `TESS_orbit_times.csv` (auto-downloaded via
`ensure_tess_orbit_times_csv`), not a single sector-wide window.

Provenance is stored in `shift_schedule.npz` as `frame_origin` (`0`=measured,
`1`=synth missing WCS, `2`=synth sigma-clipped) and in the JSON sidecar as
`frames_missing_wcs` / `frames_sigma_clipped` / `frame_origin_counts`. The remap
manifest echoes `shift_schedule_frame_origin_counts`.

---

## L4a: Type I Exact (intra-skycell)

Exact only the **~9% R=1 boundary band** — never the whole skycell.

| Item | Spec |
|------|------|
| Trigger | New nonzero `(skycell, sx, sy)` from L3 |
| Skip | `(0, 0)` |
| Mask | `dilate(ownership_boundary ∪ footprint_edge, R=1)` on the rolled assignment |
| Cache | `exact_cache_l4a/{skycell}_sx{±}_sy{±}_exact.npz` |
| WCS | Realizing frame for that own shift |

Interior pixels stay as cheap integer roll of the frozen L0 map.

**Legacy monolithic `exact_cache/`** (old L4b-lite pollution) is **never** reused
as L4a. Verify rejects it when `exact_cache_l4a/` is missing; rebuild with
`rebuild_remap_cache: true`.

---

## L4b: Type II F2 Exact (inter-skycell rim)

Off by default (`l4b_policy: none`). When `l4b_policy: pair_state`:

| Item | Spec |
|------|------|
| Pairs | Master 4-neighbour abutting undirected pairs |
| Keys | Unique `(sx_A,sy_A,sx_B,sy_B)` per pair over `frame_valid` |
| Scope | Shared abutting border TESS ids only (~1.9%) |
| WCS | First valid frame realizing that 4-tuple (`rep_frame_index`) |
| Cache | `exact_cache_l4b/pair_{id_lo}__{id_hi}_…_rim.npz` |

There is **no** `lite` policy. Manifests with `include_abutting_border_exact` or
`l4b_policy` in `{lite, abutting_under_type1_wcs}` are rejected on verify.

---

## L5: compose, bin, assemble

### Architecture A (group-scoped contribs)

When `l4b_policy=pair_state`, neighbour shifts can differ across `group_id`s that
share the same Type I key `(skycell,sx,sy)` (~48% of keys on s0020/c3/k3). L5
therefore writes **group-qualified** contribs:

```text
contribs/{skycell}_sx{±}_sy{±}_gid{N}.npz     # pair_state
contribs/{skycell}_sx{±}_sy{±}.npz             # l4b_policy=none (legacy key)
```

### Compose order (per group context)

1. Load L0 `TESS_PIXEL_MAP`.
2. Build L4a hybrid (roll + Exact patch from `exact_cache_l4a/`).
3. For each master abutting neighbour B, load the matching L4b rim NPZ using
   **this group’s** `(sx_B, sy_B)`; patch A’s rim (`abutting_rim_ps1_mask`).
   **L4b wins on overlap.**
4. Bin PS1 with `assignment=hybrid_map`, `sx_int=0`, `sy_int=0` (do not data-roll
   PS1 when hybrid is used).
5. Missing required Exact → **fail** (`require_l4b_cache` / require L4a). No
   silent roll fallback when hybrid is required.

### Consumers

| Consumer | Path |
|----------|------|
| Diff Hotpants / shared mask / kernels | `build_field_mode_template_loader` → `assemble_field_group_flux` / `_count` |
| Optional FITS | `materialize_fits: true` → `fits/syndiff_field_*_gid{N}.fits.fz` from the **same** assemble helpers |
| Star | Reads field-assembled templates when the event’s `geometry_mode` is field |

---

## Config reference

```yaml
stages:
  wcs_grouping:                   # consumed by bind (diff DAG); key name unchanged
    geometry_mode: field          # DEFAULT
    grouping_quantum_ps1_px: 1.0
    wcs_drift_savgol_window: 11
    wcs_drift_savgol_polyorder: 2
    crop_mode: target_box         # diff crop only; field store is full-chip
    crop_box_size: 1024

  remap:                          # L2–L4
    apply_hybrid_exact: true
    hybrid_R: 1
    l4b_policy: none              # or pair_state
    raw_drift_outlier_sigma: 5.0  # or null to disable
    rebuild_remap_cache: false
    rebuild_l4b_cache: false
    cache_quantum_ps1_px: 1.0
    keying: absolute
    n_jobs: 16
    executor: condor
    condor_request_memory: 128000

  downsample:                     # L5
    geometry_mode: field          # DEFAULT
    apply_hybrid_exact: true
    hybrid_R: 1
    l4b_policy: none              # must match remap for F2
    require_l4b_cache: null       # auto-true when pair_state
    rebuild_field_store: false
    materialize_fits: false
    n_jobs: 16
```

Set `geometry_mode` under **both** `wcs_grouping` and `downsample`. The
downsample dataclass default is `field`, but an explicit mismatch with
`wcs_grouping` is confusing — keep them aligned.

---

## Storage layout

```text
{data_root}/s{SSSS}/c{C}/k{K}/remap/oversampling_{N}/     # remap (L2–L4)
  remap_manifest.json
  shift_schedule.npz / .json
  skycell_shift_grid_debug.png              # 3×3 SG+quantized vs BTJD
  skycell_shift_relative_center_debug.png   # FoV differential (SG only)
  template_group_shifts.parquet
  template_groups.json
  exact_cache_l4a/{skycell}_sx{±N}_sy{±N}_exact.npz
  exact_cache_l4b/pair_{id}__{id}_…_rim.npz               # pair_state only
  exact_cache_legacy_polluted/                            # migrated lite; do not use
  .lock

{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/ # downsample (L5)
  template_manifest.json
  field_mode_assembly.json          # schema v1|v2
  contribs/…[_gid{N}].npz
  fits/syndiff_field_s{SSSS}_{C}_{K}[_os{N}]_gid{N}.fits.fz  # optional
  materialized_fits.json
  .lock
```

| Stage | Saves | Does not save |
|-------|-------|---------------|
| `mapping` | Frozen L0 regmaps | Schedules, Exact, flux |
| `remap` | Schedule, groups, Exact caches | Full hybrid grids, contribs, FITS |
| `downsample` | Hybrid-binned contribs (+ optional FITS) | PanCAKES Exact (reads caches only) |

Both remap and templates stores are **shared across events** on an SCC. Ordinary
`--force-rerun` does not delete them. Use `rebuild_remap_cache` /
`rebuild_field_store` for intentional rebuilds.

See also [storage layout](storage_layout.md).

### Migration

Older field builds colocated L2–L4 under `templates/`. Use
`migrate_scc_remap_artifacts()` to copy schedule/groups into `remap/`; legacy
`exact_cache/` is archived as `exact_cache_legacy_polluted/` and must be rebuilt
as pure `exact_cache_l4a/` (+ `exact_cache_l4b/` for F2).

---

## Diff and star

1. `bind` writes `event_job.json` / `frames.csv` with `geometry_mode: field` and
   per-frame `group_id` (from SCC group artifacts).
2. Diff resolves the SCC templates store via `data_root` + SCC identity.
3. Hotpants / shared mask / kernel engines call the field template loader, which
   assembles flux (and optionally count) for the frame’s `group_id`, then crops
   to the event ROI.
4. Star consumes the same field-assembled templates when the event is field mode.

Do not parse field products with the linear `dx/dy` filename regex.

---

## Engine support

| Stage | Field-aware |
|-------|-------------|
| `hotpants` | yes (on-demand loader, cached per group); OS-aware — see [oversampled templates](oversampled_templates.md) |
| `shared_mask` | yes (`ps1_min_hit_count>0` uses assembled COUNT) |
| `kernel_fit` / `convolved_templates` / `kernel_subtract` | yes (keyed by `group_id`) |
| `epsf` / `centroids` / `sat_template` / `subtract` / `background` / `forced_photometry` | agnostic (consume diff/ePSF products) |
| `star` | yes when event `geometry_mode` is field |

---

## Verify and rebuild

Verify rejects:

- `include_abutting_border_exact` / lite `l4b_policy` values
- Polluted `exact_cache/` without `exact_cache_l4a/`
- `pair_state` without both cache dirs (and NPZ count fingerprints)
- `pair_state` contrib keys that are not group-qualified 4-tuples

Intentional rebuild:

```yaml
stages:
  remap:
    rebuild_remap_cache: true
    rebuild_l4b_cache: true
  downsample:
    rebuild_field_store: true
```

---

## Performance notes

- Field mode has **~10²–10³ groups**, so `convolved_templates` runs one template
  per `group_id` (can be slow on a full frame set).
- Hybrid Exact workers cap at `min(n_jobs, SYNDIFF_HYBRID_MAX_JOBS=24, CPUs)`.
- L4a+L4b F2 remap is order **~7 h CPU** per SCC-class gate; use Condor memory
  ≥128 GB for remap when enabling `pair_state`.
- Pre-SG MAD outlier gate + missing-WCS synthesis (not a post-hoc median
  PS1-shift drop) keeps L4a keys from exploding while every FFI still gets a
  shift assignment.

---

## Package modules

| Module | Role |
|--------|------|
| `template_creation/processing/shift_schedule.py` | L2–L3 schedule + groups + synthesis / frame_origin |
| `template_creation/processing/shift_schedule_plots.py` | Remap debug PNGs (3×3 grid + relative-to-center) |
| `template_creation/processing/hybrid_regmaps.py` | L4a mask / roll / patch primitives |
| `template_creation/processing/field_hybrid_exact.py` | Exact subsets; L4a/L4b compose |
| `template_creation/processing/field_abutting.py` | Undirected pairs + pair-state enum |
| `template_creation/processing/field_remap.py` | SCC remap store (`run_field_remap_scc`) |
| `template_creation/processing/field_downsample.py` | SCC L5 (`run_field_downsample_scc`) |
| `template_creation/processing/field_templates.py` | Contrib I/O, assemble, materialize FITS |
| `difference_imaging/support/template_resolution.py` | Diff-time field loader |
| `template_creation/orchestration/verify.py` | Dual-cache + lite rejection |

---

## Glossary

| Term | Meaning |
|------|---------|
| Frozen regmap | L0 PS1→TESS assignment at mapping-epoch WCS |
| Linear / roll | Integer PS1 roll of frozen regmap |
| Exact | `process_skycell_pixel_mapping` under a chosen frame WCS |
| Hybrid | Linear everywhere + Exact on the R=1 (and optional L4b) mask |
| Type I / L4a | Intra-skycell seam/rim Exact |
| Type II / L4b F2 | Inter-skycell abutting-rim Exact under shared WCS |
| Signature / `group_id` | Full-chip vector of per-skycell integer shifts |
| Architecture A | Group-qualified contribs when neighbour context collides |
