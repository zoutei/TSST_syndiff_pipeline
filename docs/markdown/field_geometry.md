# Field (distortion-aware) templates — `geometry_mode: field`

## What it is

Linear templates measure velocity-aberration drift at **one** point (the science
target) and apply it as a single global PS1-pixel roll, so templates degrade away
from the target (CLAUDE.md invariant #2). **Field mode** instead measures drift at
**every skycell center**, integer-quantizes each skycell's PS1 shift with
hysteresis, groups frames by their full-chip shift **signature** (`group_id`), rolls
each skycell's frozen regmap independently, and Exact-patches the R=1 seam/rim
(hybrid **L4a** + abutting-border **L4b-lite**).

Instead of per-target `dx/dy` template FITS, field mode keeps an **SCC-scoped sparse
contrib store** and **assembles a template per `group_id` on demand**.

`geometry_mode: linear` remains the default and is unaffected.

## Config knobs

```yaml
stages:
  wcs_grouping:
    geometry_mode: field          # opt in
    grouping_quantum_ps1_px: 1.0  # signature quantum
    crop_mode: target_box         # a crop keeps the contrib set to ROI skycells
    crop_box_size: 1024
  downsample:
    geometry_mode: field
    apply_hybrid_exact: true      # L4a R=1 seam/rim Exact (else roll-only)
    hybrid_R: 1
    include_abutting_border_exact: true   # L4b-lite
    rebuild_field_store: false    # true overwrites existing contribs + exact cache
    n_jobs: 32                    # hybrid workers cap at min(n_jobs, 24, CPUs)
```

`mapping_dir` / `convolved_dir` can point downsample at a shared read-only mapping +
convolved tree while writing `field_templates/` to an isolated `data_root`.

## Storage

```
{data_root}/field_templates/sector_{S}_camera_{C}_ccd_{K}/[oversampling_{N}/]
  template_manifest.json          # completeness marker for the SCC store
  shift_schedule.npz              # per-skycell drift schedule
  template_group_shifts.parquet   # (group_id, skycell, sx_int, sy_int, ...)
  field_mode_assembly.json        # roi_bounds, base_tess_shape, zarr, ignore_mask
  contribs/skycell.{proj}.{cell}_sx{±N}_sy{±N}.npz
  exact_cache/…_exact.npz
{event}/field_contrib_keys.json   # per-event crop-filtered key set (verify marker)
{event}/ws/field_templates -> SCC store
```

The store is **shared across events** on an SCC; force-rerun never deletes it, and
each event records exactly the keys it required (crop-aware verify). See
`storage_layout.md`.

## Engine support

Every template-consuming stage is field-aware; templates are assembled per
`group_id` from the store.

| Stage | Field-aware |
|-------|-------------|
| `hotpants` | yes (on-demand loader, cached per group) |
| `shared_mask` | yes (`ps1_min_hit_count>0` uses the assembled COUNT plane) |
| `kernel_fit` / `convolved_templates` / `kernel_subtract` | yes (convolved products keyed by `group_id`) |
| `epsf` / `centroids` / `sat_template` / `subtract` / `background` / `forced_photometry` | agnostic (consume diff/ePSF products) |
| star (host-star LCs) | yes (per-skycell field shifts per `group_id`, deduped to local signatures) |

Assemble a full-FFI ("big") template for any FFI:
`template_resolution.assemble_field_template_for_ffi(ctx, manifest, ffi_name)`.

## Performance caveats

- Field mode has **~10²–10³ groups** (vs ~19 linear), so `convolved_templates`
  convolves one template per distinct `group_id` **serially** — slow on a full
  frame set. Parallelize `run_convolved_templates`, or use a coarser
  `grouping_quantum_ps1_px` for the kernel engine. (The star path deduplicates to
  the few **local** signatures over its ROI, so it stays cheap.)
- Hybrid Exact does one `process_skycell_pixel_mapping` per `(skycell, sx, sy)` key;
  workers cap at `min(n_jobs, SYNDIFF_HYBRID_MAX_JOBS=24, available CPUs)` at
  ~2 GB each.

## Not yet done

- `materialize_fits: true` (optional pre-materialized FITS) is a no-op flag.
- Parallel `convolved_templates` / F2 pair-state strip cache (perf).
