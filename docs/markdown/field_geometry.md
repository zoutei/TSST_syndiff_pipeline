# Field (distortion-aware) templates

**Field mode is the default** (`geometry_mode: field` on `stages.downsample`).
Templates are built once per SCC from per-skycell WCS drift, hybrid Exact
patches, and a sparse contrib store; difference imaging assembles a full-chip
template per `group_id` on demand via `scc_bootstrap` and subtracts on the
canonical **MappingGrid** science rectangle (~1960×2018 native for 2048 FFI
defaults).

Linear mode (`geometry_mode: linear`) remains available as an explicit opt-out
for target-anchored single-offset templates — see
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

# Field-mode diff (after template rebuild with MAPGRID=2 + sidecar v3):
syndiff diff submit \
  --site config \
  --config config/diff_config_single_kernel.yaml \
  --scc config/scc_example.csv
```

Defaults (no YAML required beyond a normal site):

| Knob | Default | Meaning |
|------|---------|---------|
| `stages.downsample.geometry_mode` | `field` | L5 uses `field_downsample`; diff uses `scc_bootstrap` |
| `stages.remap.intra_skycell_R` | `1` | Intra-skycell Exact dilation radius |
| `stages.downsample.apply_intra_skycell` | `true` | Apply intra-skycell Exact patch at L5 |
| `stages.downsample.apply_inter_skycell` | `true` | Apply inter-skycell rim patches at L5 |
| `stages.mapping.template_conv_pad_spare_px` | `4` | Extra bottom pad rows for Hotpants kernel margin |

Opt out to linear:

```yaml
stages:
  downsample:
    geometry_mode: linear
```

## MappingGrid (canonical SCC grid)

All field-mode coordinates flow through **`MappingGrid`**
(`syndiff_pipeline/common/mapping_grid.py`). Mapping writes `MAPGRID=2` on the
master FITS; remap and downsample sidecars embed the same grid; `scc_bootstrap`
copies `crop_bounds` into `bookkeeping/diff/diff_job.json` for diff.

| Quantity (2048² FFI defaults) | Value |
|-------------------------------|-------|
| Science shape (native) | 2018 × 1960 |
| Template shape (with bottom conv pad) | 2026 × 1960 |
| `ffi_xmin`, `ffi_xmax` | 44, 2004 |
| Science `y` range | 0 … 2018 |
| Template `ffi_ymin` (pad rows) | −8 |

WCS and `tesswcs` projections must use **FFI chip pixels** `(ffi_x, ffi_y)`;
local grid indices `(lx, ly)` are for array indexing only. Pad rows use negative
`ffi_y`. See `coordinate_preflight.py` checks in mapping, bootstrap, and execute.

Diff does **not** use `crop_mode` / `target_box` for geometry — the science
grid is fully determined by the SCC `MappingGrid`.

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
6. Exact-patch the R=1 seam/rim (**intra-skycell / L4a**) and the inter-skycell
   abutting rim under a shared WCS (**inter-skycell / L4b F2**).
7. Bin PS1 flux into sparse `contribs/` and assemble per `group_id` at diff time.

---

## Linear vs field

| Aspect | Field (default) | Linear (opt-out) |
|--------|-----------------|------------------|
| Drift measured at | Every skycell center | Science target only |
| Template groups | ~10²–10³ full-chip signatures | ~19 target-anchored |
| Regmap geometry | Roll + intra-skycell (L4a) + inter-skycell (L4b F2) | Integer roll of frozen L0 |
| Output | SCC `contribs/` + on-demand assemble | Per-event `syndiff_template_*_dx*_dy*.fits.fz` |
| Event dependency at build | None (SCC-wide) | Requires per-event offset list (linear only) |
| Template DAG | Includes `remap` before `downsample` | `remap` pre-skipped |
| Deep dive | This document | [downsample_technical.md](stages/downsample_technical.md) |

---

## Architecture (L0–L5)

| Layer | Role | Stage / artifacts | Modules |
|-------|------|-------------------|---------|
| **L0** | Frozen PS1→TESS ownership at mapping-epoch WCS | `mapping` → `mapping/oversampling_{N}/` | `pancakes.py` |
| **L2** | Per-skycell PS1 drift time series | `remap` → `shift_schedule.npz` | `shift_schedule.py`, `field_remap.py` |
| **L3** | Hysteresis integer `(sx,sy)`; signature → `group_id` | Same remap store | `shift_schedule.py` |
| **L4a** | Intra-skycell Exact | Exact-patch R=1 seam/rim (~9% footprint) | `exact_cache_l4a/` | `hybrid_regmaps.py`, `field_hybrid_exact.py` |
| **L4b (F2)** | Inter-skycell Exact | Shared-WCS abutting rim (~1.9%) | `exact_cache_l4b/` | `field_abutting.py`, `field_hybrid_exact.py` |
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
  (under `{scc}/debug_plots/`; named remap lanes add `_{store_name}` to the basename)
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
| Trigger | Contiguous nonzero `(skycell, sx, sy)` **shift epoch** (RLE; revisits get a new epoch) |
| Skip | `(0, 0)` |
| Mask | `dilate(ownership_boundary ∪ footprint_edge, R=1)` on the rolled assignment |
| Cache | `exact_cache_l4a/{skycell}/e{epoch}_sx{±}_sy{±}_exact.npz` (schema v3) |
| WCS | **Middle measured** frame in the epoch (`rep_frame_index`) |

Interior pixels stay as cheap integer roll of the frozen L0 map.

**Legacy monolithic `exact_cache/`** (old L4b-lite pollution) is **never** reused
as L4a. Verify rejects it when `exact_cache_l4a/` is missing; rebuild with
`rebuild_remap_cache: true`. Flat root `exact_cache_l4a/*.npz` is rejected under
schema v3 — wipe Exact dirs and rebuild.

---

## L4b: Type II F2 Exact (inter-skycell rim)

Remap builds `exact_cache_l4b/` for every abutting master pair and contiguous
**pair epoch** (constant 4-tuple run). Downsample applies those rim patches when
`stages.downsample.apply_inter_skycell` is `true` (default). Set
`apply_inter_skycell: false` to compose L5 without inter-skycell rims (skips the
nonempty `exact_cache_l4b/` verify check) — useful when inter-skycell remap is
incomplete.

| Item | Spec |
|------|------|
| Pairs | Master 4-neighbour abutting undirected pairs |
| Keys | Contiguous pair epochs `(sx_A,sy_A,sx_B,sy_B)` covering sets of `group_id`s |
| Scope | Shared abutting border TESS ids only (~1.9%) |
| WCS | Middle measured frame in the pair epoch (`rep_frame_index`) |
| Cache | `exact_cache_l4b/pair_{id_lo}__{id_hi}/e{epoch}_sx…_rim.npz` |

L5 compose resolves Exact via `gid_epoch_index.npz` (`group_id` → epoch → path).
Missing-WCS / rejected frames still get synthesized shifts and real `group_id`s
(contiguous signature islands; no `group_id=-1`).

Legacy manifests with `include_abutting_border_exact`, `l4b_policy`, or
`apply_hybrid_exact` are rejected on verify — rebuild remap with schema **v3**.

---

## L5: compose, bin, assemble

### Group-scoped contribs (always)

Neighbour shifts can differ across `group_id`s that share the same intra-skycell
key `(skycell,sx,sy)` (~48% of keys on s0020/c3/k3). L5 therefore always writes
**group-qualified** contribs:

```text
contribs/{skycell}_sx{±}_sy{±}_gid{N}.npz
```

### Skycell-major dispatch + composite-key fan-out

L5 parallelizes over **skycells**, not flat `(group_id, skycell, sx, sy)` rows.
Within each skycell worker:

1. Load the regmap FITS and PS1 zarr **once**.
2. Build a **composite geometry key** per group:
   `(l4a_epoch_or_roll0, sorted (neighbour_id, pair_epoch_id)…)` from
   `gid_epoch_index.npz` (schema v3) or legacy neighbour shifts.
3. **Compose + bin once** per distinct key; **fan out** identical sparse arrays to
   every sharing `group_id` (on-disk contract unchanged: one NPZ per group).
4. Own-shift `(0,0)` uses L4a sentinel `"roll0"` (remap skips zero epochs) but
   still composes L4b rims when `apply_inter_skycell` is true and neighbours
   differ. When `apply_inter_skycell` is false, composite keys omit neighbour
   pair-epoch slots (intra epoch / `"roll0"` only).

Progress sidecar (`downsample.progress.json`) reports `ckeys done/total` and
skycell batches. Contrib writes use temp-file + atomic `replace` (no store-wide
lock). Crop/event builds prefilter pixels outside the ROI before `argsort`.

### Compose order (per group context)

1. Load L0 `TESS_PIXEL_MAP`.
2. If `apply_intra_skycell` (default): build intra-skycell hybrid (roll + Exact
   patch from `exact_cache_l4a/`). Otherwise use the rolled linear assignment.
3. If `apply_inter_skycell` (default): for each master abutting neighbour B, load
   the matching inter-skycell rim NPZ using **this group’s** pair-epoch lookup;
   patch A’s rim (`abutting_rim_ps1_mask`). **Inter-skycell wins on overlap.**
4. Bin PS1 with `assignment=hybrid_map`, `sx_int=0`, `sy_int=0` (do not data-roll
   PS1 when hybrid is used).
5. Missing required Exact caches (for layers that are enabled) → **fail**. No
   silent roll fallback when a required layer’s cache is absent.

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
  remap:                          # L2–L4 (field mode only)
    intra_skycell_R: 1            # sole geometry tuning knob
    raw_drift_outlier_sigma: 5.0  # or null to disable
    rebuild_remap_cache: false
    rebuild_inter_skycell_cache: false
    cache_quantum_ps1_px: 1.0
    keying: absolute
    n_jobs: 16
    executor: condor
    condor_request_memory: 128000

  downsample:                     # L5
    geometry_mode: field          # DEFAULT; set linear to opt out
    apply_intra_skycell: true     # use exact_cache_l4a/ at compose
    apply_inter_skycell: true     # use exact_cache_l4b/ rim patches
    rebuild_field_store: false
    materialize_fits: false
    n_jobs: 16
```

At least one of `apply_intra_skycell` / `apply_inter_skycell` must be `true`
(reject both `false` at parse). Remap still builds both Exact caches; the
downsample toggles only control which layers L5 compose applies and which
caches verify requires.

Removed keys (`apply_hybrid_exact`, `l4b_policy`, `require_l4b_cache`, `hybrid_R`
on downsample) are **rejected at parse** — use the shape above.

Set `geometry_mode: field` on `stages.downsample` (default). Linear opt-out
requires `geometry_mode: linear` on the same key.

---

## Storage layout

```text
{data_root}/s{SSSS}/c{C}/k{K}/remap/oversampling_{N}/     # remap (L2–L4), schema v3
  remap_manifest.json                   # schema_version: 3
  shift_schedule.npz / .json
  shift_epochs.parquet                  # L4a shift epochs (skycell, sx, sy, rep_frame)
  pair_epochs.parquet                   # L4b pair epochs (4-tuple runs)
  epoch_group_members.parquet           # epoch_id → group_id membership
  gid_epoch_index.npz                   # group_id → L4a/L4b epoch lookup
  group_id_per_frame.npy
  template_group_shifts.parquet
  template_groups.json
  exact_cache_l4a/{skycell}/e{epoch}_sx{±}_sy{±}_exact.npz
  exact_cache_l4b/pair_{id_lo}__{id_hi}/e{epoch}_sx…_rim.npz
  exact_cache_legacy_polluted/          # migrated lite; do not use
  remap.progress.json                   # L4a/L4b counters + perf metadata
  .lock

{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/ # downsample (L5)
  template_manifest.json
  field_mode_assembly.json              # includes store_root, remap_root, lane names
  downsample.progress.json              # field L5 ckeys / skycell batches
  contribs/…_gid{N}.npz
  fits/syndiff_field_s{SSSS}_{C}_{K}[_os{N}]_gid{N}.fits.fz  # optional
  materialized_fits.json
  .lock

{data_root}/s{SSSS}/c{C}/k{K}/debug_plots/  # template-pipeline diagnostics
  skycell_shift_grid_debug.png
  skycell_shift_relative_center_debug.png
  skycell_shift_grid_debug_{NAME}.png              # when remap.store_name set
  skycell_shift_relative_center_debug_{NAME}.png

# Optional named lanes (do not clobber the default trees):
#   remap_{NAME}/oversampling_{N}/
#   templates_{NAME}/oversampling_{N}/
```

Named store knobs (campaign YAML):

| Knob | Role |
|------|------|
| `stages.remap.store_name` | Remap **write** lane → `remap/` or `remap_{NAME}/` |
| `stages.downsample.remap_store_name` | Downsample **input** (read remap); omit → inherit `remap.store_name` |
| `stages.downsample.output_store_name` | Downsample **output** → `templates/` or `templates_{NAME}/` |
| `paths.template_store_name` (diff) | Which templates lane diff/star reads |

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

1. `scc_bootstrap` (inside `diff` execute) reads `field_mode_assembly.json`
   schema v3 + `mapping_grid`, writes `bookkeeping/diff/{frames.csv,diff_job.json}`,
   and sets per-frame `group_id` from SCC group artifacts.
2. Diff resolves the SCC templates store via `data_root` + SCC identity.
3. Hotpants / shared mask / kernel engines call the field template loader, which
   assembles flux (and optionally count) for the frame’s `group_id` on the full
   science grid (pad/trim at bottom edge for Hotpants).
4. Diff products are written SCC-primary under `diff_{lane}/`; event photometry
   (when using `--targets`) may additionally materialize under `events/.../ws/`.
5. Star consumes field-assembled templates from the SCC diff lane when available.

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

- Legacy manifests with `apply_hybrid_exact`, `l4b_policy`, or `hybrid_R` (schema v1)
- `include_abutting_border_exact` / lite `l4b_policy` values
- Polluted `exact_cache/` without `exact_cache_l4a/`
- Missing `exact_cache_l4a/` when `apply_intra_skycell: true` (default)
- Missing / empty `exact_cache_l4b/` when `apply_inter_skycell: true` (default);
  skipped when `apply_inter_skycell: false`
- Contrib keys that are not group-qualified 4-tuples `(group_id, skycell, sx, sy)`

Intentional rebuild:

```yaml
stages:
  remap:
    rebuild_remap_cache: true
    rebuild_inter_skycell_cache: true
  downsample:
    rebuild_field_store: true
```

---

## Performance notes

- Field mode has **~10²–10³ groups**, so `convolved_templates` runs one template
  per `group_id` (can be slow on a full frame set).
- Hybrid Exact workers cap at `min(n_jobs, SYNDIFF_HYBRID_MAX_JOBS, CPUs)`;
  Condor sets `SYNDIFF_HYBRID_MAX_JOBS` from `condor_request_cpus`.
- **Remap L4a** batches shift epochs by skycell (one regmap/zarr load per skycell,
  not per epoch). **Remap L4b** batches pair epochs by abutting border (~2.4k
  joblib tasks vs one per pair-epoch row). Worker processes hoist TESS coords and
  load the master map from `master_path` (safe L4a→L4b joblib handoff).
- **L5 downsample** batches by skycell and deduplicates compose+bin via composite
  keys (~17–59× reuse on full-chip SCC builds). Optional
  `SYNDIFF_REMAP_BENCHMARK_TAG` attaches perf metadata to `remap.progress.json`.
- Intra-skycell + inter-skycell remap is order **~7 h CPU** per SCC-class gate; use
  Condor memory ≥128 GB for remap on full chips.
- Pre-SG MAD outlier gate + missing-WCS synthesis (not a post-hoc median
  PS1-shift drop) keeps L4a keys from exploding while every FFI still gets a
  shift assignment.

---

## Package modules

| Module | Role |
|--------|------|
| `template_creation/processing/shift_schedule.py` | L2–L3 schedule + groups + synthesis / frame_origin |
| `template_creation/processing/shift_schedule_plots.py` | Remap debug PNGs under SCC `debug_plots/` |
| `template_creation/processing/hybrid_regmaps.py` | L4a mask / roll / patch primitives |
| `template_creation/processing/field_hybrid_exact.py` | Exact subsets; L4a/L4b compose |
| `template_creation/processing/field_abutting.py` | Undirected pairs; epoch cache path helpers |
| `template_creation/processing/field_remap.py` | SCC remap store (`run_field_remap_scc`); epoch artifacts |
| `template_creation/processing/field_downsample.py` | SCC L5 (`run_field_downsample_scc`); composite-key batches |
| `template_creation/processing/field_downsample_progress.py` | L5 progress sidecar (`ckeys`, skycell batches) |
| `template_creation/processing/remap_progress.py` | Remap L4a/L4b progress + perf metadata |
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
| Architecture A | Group-qualified contribs when neighbour context collides (always on) |
