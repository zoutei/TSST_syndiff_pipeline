# SynDiff Bookkeeping — Content-Addressed Provenance Graph (Design & Plan, rev. 2)

**Status:** revised 2026-07-19; supersedes rev. 1 in full.
**Branch context:** `distortion_aware_templates` @ d5569c0 **plus** the in-flight
uncommitted `ffi_list` refactor (see §2.4). The earlier partial implementation in
the `provenance-bookkeeping` worktree is **abandoned as a code source** — the
branch has moved past its base (remap/downsample stage split, `.fits.fz`
canonical format, `ffi_list`). Its *design* and its two **validated findings**
(§12 combined-artifact shape, §13 convolution linearity) are retained; all code
is reimplemented fresh on this branch.
**Scope (extended):** template creation **and** difference imaging. Per TESS
FFI, the background image, difference image, and ePSF are tracked artifacts;
diff imaging (hotpants/kernel matching/background cleaning) is **SCC-scoped**
once parameters are finalized and shared across events; **photometry is
event-scoped** and references the SCC-scoped diff artifacts through the graph.

---

## 0. Executive summary

One abstraction replaces all completeness bookkeeping: a **content-addressed
provenance graph** — a small, purpose-built build system over the pipeline's
artifacts.

- Every artifact is a **node** named by a Merkle fingerprint
  `H(kind, spatial_key, recipe_params, sorted(input_fingerprints), code_version)`.
  Identical work is recognized by string equality; a param change re-fingerprints
  exactly the affected downstream cone.
- **Sky-keyed sharing:** combined and convolved PS1 skycells carry no
  sector/camera/ccd, so overlapping sectors reuse them instead of recomputing.
- **SCC-keyed sharing (new):** per-FFI diff products (background, difference
  image, ePSF) are keyed by `(SCC, tess_product_id, recipe)` — not by event — so
  every event on an SCC shares one set of finalized diff products. Photometry
  nodes are per-event and point at the diff artifacts they consumed.
- **The O(cells) verify scan dies:** completeness becomes an indexed SQLite
  query; the post-run manifest re-scan is replaced by sidecar records emitted at
  publish time. The one remaining hot-path scan (`verify_ps1_process`) is the
  first cutover target.
- **Invariants:** content authority (bytes at the fingerprinted key are the
  truth; the DB is a rebuildable index), atomic publish, single DB writer
  (supervisor) fed by lock-free worker sidecars — the same NFS-safe pattern as
  the run-state DB and the new `ffi_list` writer.

Delivery: one prerequisite PR (land `ffi_list`), five template-side PRs, two
diff-side PRs, one cleanup PR. §19.

---

## 1. Goals

1. **Track inputs** — TESS FFIs (via `ffi_list`) and raw PS1 skycells.
2. **Share across sectors** — sky-keyed combined + convolved PS1 skycells.
3. **Share across events** — SCC-keyed per-FFI diff products (background, diff
   image, ePSF, shared mask) once parameters are finalized.
4. **Track outputs with full recipes** — mapping, remap store, downsample
   products, diff products, photometry — queryable under config drift.
5. **Per-unit incrementality** — per-skycell and per-FFI skip/resume, not
   all-or-nothing per SCC.
6. **Eliminate hot-path filesystem scans** — O(1) indexed queries.

---

## 2. Current state (what the graph replaces / builds on)

### 2.1 Template side — the remaining hot scan
The stage tuple is now six stages (`stages.py:230`): `tess_ffi_download →
mapping → ps1_download → ps1_process → remap → downsample` (the old `templates`
stage is hard-renamed `downsample`; `remap` owns field L2–L4 under
`remap/oversampling_{N}/`; downsample's field mode depends on remap via
`_downsample_effective_deps`, `stages.py:62`).

- `verify_ps1_process` (`verify.py:993`) → `expected_ps1_process_skycells`
  (`:930`) → `_count_convolved_data_arrays` (`:967`): **one stat-walk per
  expected skycell** over `convolved.zarr` — still the O(cells) NFS scan
  (historically ~30 min). `collect_stage_artifacts` (`:1515`) **reruns the same
  count** post-run (`:1561-1567`). This is the primary target.
- `verify_remap` (`:1194`) and `verify_downsample` (`:1124`) are already cheap
  (manifest/marker reads) — they get provenance records for graph completeness,
  not for performance.
- `config_fingerprint` (`verify.py:117-172`) enumerates params-that-matter per
  stage (mapping, ps1_process, remap, downsample, ps1_download) — one hash per
  SCC/stage, values not stored, scattered JSON. These enumerations seed the
  per-kind `recipe_params()` builders.
- Scheduler flow: `_run_verify_pass` (`scheduler.py:1103-1229`) —
  `check_manifests_only` (`verify.py:300`) fast path at `scheduler.py:1141/1151`,
  `stage_absence_probe` at `:1172`, else a `VerifyTask` to the background
  `ArtifactVerifyWorker` whose `_run_verify_task` (`verify_worker.py:88`) runs
  `stage_complete` (manifest-first, then the disk scan). Outcomes land via
  `_apply_verify_outcome` (`:890`) → `state.cache_external_check`;
  `_tick_run` (`:1518`) → `promote_stages` (`:1574`). Supervisor loop:
  `run_supervisor_daemon` (`:1986`), body at `:2060-2090`.
- Post-run: `run_stage.py:272` calls `collect_stage_artifacts` + `write_manifest`
  (`:308-316`).
- `ps1_process.py` is unchanged since the rev.-1 base commit — saver path
  `saver_worker` (`:841`) → `zarr_utils.save_convolved_results`
  (`ps1_process.py:851`); `band_cache` + `regular_cache_hit` fast path
  (`ingest_worker`, `:503-509`); per-SCC store `scc_convolved_zarr`
  (`scc_paths.py:120`).

### 2.2 Diff side — no per-artifact bookkeeping at all
Package `syndiff_pipeline/difference_imaging/`; one run = one event×SCC leaf;
`run_config_pipeline` (`orchestration/execute.py:680`) loops the configured
`pipeline:` stages, and the per-FFI loop is internal (joblib/loky). All outputs
land under the **event** workspace `events/{event}/s{SSSS}_c{C}_k{K}/ws/`:

- Per-FFI files named `tess{product_id}_{label}.fits.fz`
  (`support/ffi_naming.py`): difference images (`stages/hotpants.py` /
  `kernel_subtract.py`), hotpants background (`_save_bkg_fits`,
  `hotpants.py:815`), standalone background stage per-frame FITS + `stack.npz`
  cube (`stages/background/io.py`), gridded ePSF
  `{ffi_stem}_gridded_epsf.npz` + `gridded_epsf_index.json` + optional
  `group_epsf/group_epsf_{gid}.npz` (`stages/gridded_epsf.py`).
- Recipes already exist as strict dataclasses (`orchestration/stage_params.py`):
  `HotpantsParams`, `KernelFitParams`/`KernelSubtractParams`, `EpsfParams`,
  `BackgroundParams` (+ step params), photometry method params (incl. the
  tessreduce ePSF fitter).
- Existing fingerprint machinery: `workspace_lock.diff_config_fingerprint`
  (whole-pipeline sha256) + immutable `ws/diff_config.yaml` snapshot;
  `write_diff_manifest` per stage. Whole-config granularity only.
- `verify_diff` (`orchestration/diff_verify.py`) checks **only the last
  configured stage's** marker (e.g. a lightcurve CSV or "any FITS in the diffs
  dir") — no per-FFI completeness; resume relies on per-stage skip-existing
  checks and progress sidecars.
- The per-FFI spine is `frames.csv` (`support/manifest.py`), joined by
  `tess_product_id`; templates are resolved from the SCC science tree
  (`support/template_resolution.py` → `scc_templates_dir`, per-frame choice by
  dx/dy offset via `find_template_by_offset`).

### 2.3 Cross-sector sharing gap (unchanged motivation)
Raw grizy is globally shared (`ps1_skycells_zarr/ps1_skycells.zarr`, keyed
`projection/skycell`). `convolved.zarr` is per-SCC and rebuilt per sector; the
only per-SCC ingredient is cross-projection padding at projection seams.

### 2.4 New substrate this design builds on
- **`ffi_list` (landed — PR0, commit `7711a86`):** per-SCC
  `ffi_list.parquet`/`.csv` (`scc_ffi_list_parquet`, `scc_paths.py:143`)
  replaces `wcs_cache.parquet`; rows keyed by **logical** `.fits` basename
  (`manifest_basename_from_local`) independent of on-disk variant, carrying the
  full HDU1 header blob + `wcs_ok` + `schema_version`. Populated **during
  download** (`download.py` `_FfiListIngestBuffer`, `download_ffis(...,
  update_ffi_list=True)`); grouping/reference-selection/remap are pure cache
  consumers. Writes are FileLock + tmp + `os.replace`. This is the FFI input
  registry: `ffi` input nodes and per-FFI required sets derive from it, with no
  FITS reads.
- **`.fits.fz` variant system:** readers accept `.fits.fz/.fits.gz/.fits`
  (`fits_variants.resolve_fits_variant:92`); writers always produce `.fits.fz`.
  `fits_logical_path` (`:55`) / `canonical_fits_path_key` (`:123`) provide the
  variant-stable logical identity that reindex and `location` fields key on.
  Publish caveat: the plain-FITS write is atomic (`_atomic_writeto_plain`,
  `fits_io.py:91`) but the fpack step is not a single rename — hardened in §10.
- **Existing bookkeeping tree:** `scc_bookkeeping_dir` (`scc_paths.py:221`) /
  `scc_bookkeeping_stage_dir` (`:231`).
- **Status grid:** 7 canonical columns in `pipeline_spec.STATUS_GRID_STAGES`
  (`pipeline_spec.py:22`), legacy `templates`→`downsample` aliasing (`:33`),
  rendered by `run_report._status_grid_rows` (`:298`) from SQLite
  `StageRunRow`s. The graph feeds this surface via the existing
  `external_check` path — no grid changes needed.

---

## 3. Foundational facts

1. **PS1 products are sky-addressed.** Skycells sit on a fixed tessellation;
   band-combine and star removal are pure functions of the skycell footprint
   (confirmed: `project_gaia_to_skycell` filters Gaia to the footprint; removal
   depends only on footprint + Gaia version + mag threshold). Only
   cross-projection padding is SCC-specific.
2. **Finalized diff products are SCC-addressed.** Hotpants/kernel matching and
   background cleaning depend on (FFI, template, mask, params) — nothing
   event-specific. Per-FFI diff artifacts keyed `(SCC, product_id, recipe)` are
   shared by every event on that SCC. Only photometry (target lists, forced
   positions) is event-specific.
3. **Identity = recipe + inputs, hashed (Merkle).** The artifact's name is the
   hash, so sameness is string equality, invalidation is automatic and exact,
   and old products survive config changes.

---

## 4. Locked decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Bookkeeping model | Content-addressed provenance graph; run-state DB demoted to pure scheduling |
| 2 | Worktree code | Ignored; fresh implementation on this branch. Design + validated findings (§12, §13) carried over |
| 3 | Rollout order | ffi_list PR0 → graph core → ps1_process cutover → shared combined → shared convolved (gated) ∥ diff tracking after core |
| 4 | Padding decouple | Same-projection canonical halo + exact patch-convolve-and-add seam correction (§13; linearity validated) |
| 5 | `code_version` | Hand-bumped `RECIPE_SCHEMA_VERSION` per producer; git SHA stored for forensics only |
| 6 | Raw-input version token | FFIs: `ffi_list` row (logical basename, size, mtime); raw skycells: `(size, mtime, download_batch_id)`; `--checksum` audit in reindex |
| 7 | GC | Reference-counted against active configs; ships report-only |
| 8 | Legacy products at reindex | `legacy_unverified`, lazily rebuilt; `--trust-current-config` escape hatch |
| 9 | Sidecar transport | Per-host `O_APPEND` JSONL spool; supervisor rotates + drains; sole DB writer |
| 10 | Star removal | Shareable (confirmed); Gaia version + mag threshold folded into combined recipe |
| 11 | Diff artifact granularity | **Per-FFI artifacts, SCC-scoped** for background/diff/ePSF/mask; **event-scoped** for photometry |
| 12 | Diff store keying | Per-FFI files grouped under a per-stage **recipe fingerprint directory** (one tree per finalized parameter set), not per-file fingerprint dirs (§14.2) |
| 13 | Exploration runs | Untracked (or tracked opt-in); only publishes into the SCC diff store when run with `publish_scc: true` — "finalized params" is a config state, not a mode switch |
| 14 | Shared-store location | All three PS1 stores under the existing `ps1_skycells_zarr/` folder: `ps1_skycells.zarr`, `ps1_combined.zarr`, `ps1_convolved.zarr` |
| 15 | Per-SCC `convolved.zarr` | **Retired as a write target at PR5** — never produced again; readers resolve shared-store-first with read-only fallback to existing legacy `convolved.zarr` |
| 16 | Existing event workspaces | **Untouched** — no migration, no symlinks, no rewrites; SCC-scoped diff store applies to new runs only |

---

## 5. Core model

```
Artifact
  kind          see §6 registry
  spatial_key   canonical dict: skycell {projection,skycell} | scc {s,c,k[,os]}
                | scc_ffi {s,c,k,product_id} | event {event,s,c,k}
  recipe        full materialized params (stored) + code_version
  inputs        fingerprints of consumed artifacts (graph edges)
  fingerprint   H(kind, spatial_key, recipe_id, sorted(input_fingerprints))
  location      path/zarr key of finalized bytes (logical .fits path for FITS)
  state         building | complete | failed
  meta          bytes, wall_time, produced_by, created_at
```

### The DAG (template + diff + photometry)

```
ffi(s,c,k,pid) ──▶ ffi_set(s,c,k) ─▶ mapping(s,c,k,os) ──────────────┐
   │                    └─▶ remap_store(s,c,k,os) ────────────┐       ▼
   │  raw_skycell(p,cell) ─▶ combined_skycell(p,cell) ─▶ convolved_skycell(p,cell) ─▶ scc_assembly(s,c,k,os) ─▶ downsample(s,c,k,os)
   │      source_catalog(footprint) ┘                                                     (templates/oversampling_N)   │
   │                                                                                                                   ▼
   ├──────────────▶ diff_background(s,c,k,pid) ──┐                                              template FITS (by group dx/dy)
   ├──────────────▶ shared_mask(s,c,k) ──────────┼──▶ diff_image(s,c,k,pid) ─▶ epsf(s,c,k,pid) ─┐
   │                                             │         ▲                                     │
   └─────────────────────────────────────────────┘         └── downsample ───────────────────────┤
                                                                                                 ▼
                                                              photometry(event,s,c,k,method) [per-event]
```

`combined_skycell`/`convolved_skycell` carry no s,c,k (cross-**sector**
sharing); `diff_background`/`diff_image`/`epsf`/`shared_mask` carry no event
(cross-**event** sharing); only `photometry` is event-scoped.

### Invariants
- **Content authority.** An artifact exists iff its finalized bytes sit at its
  fingerprinted location. The DB is a derived, rebuildable index.
- **Atomic publish.** Temp key → single atomic rename. Crashes leave only
  `_tmp_*` orphans, never a partial that looks complete.

---

## 6. Artifact kind registry

One `recipe_params(resolved) -> dict` builder per kind, migrated from
`verify.config_fingerprint` (`verify.py:117-172`) and
`stage_params.py` dataclasses (diff side: `dataclasses.asdict` of the strict
param objects — they are already the exact allow-listed recipe).

| kind | spatial_key | recipe params (source) | inputs |
|---|---|---|---|
| `ffi` (input) | `{s,c,k,product_id}` | — ; version = ffi_list row (logical basename, size, mtime) | — |
| `ffi_set` | `{s,c,k}` | download params | N×`ffi` |
| `raw_skycell` | `{projection,skycell}` | `{}` + version token | — |
| `source_catalog` | `{projection,skycell}` | gaia query params, `gaia_version` | — |
| `mapping` | `{s,c,k,os}` | `oversampling_factor, pad_distance, overwrite` (verify.py:132) | `ffi_set` |
| `remap_store` | `{s,c,k,os}` | `cache_quantum_ps1_px, keying, apply_hybrid_exact, hybrid_R, include_abutting_border_exact` (verify.py:146) | `ffi_set`, `mapping` |
| `combined_skycell` | `{projection,skycell}` | saturation/star-removal params, band-combine consts, `gaia_version` | `raw_skycell`, `source_catalog` |
| `convolved_skycell` | `{projection,skycell}` | `psf_sigma, radius, mode, padding="same_projection_only"` | `combined_skycell` |
| `scc_assembly` | `{s,c,k,os}` | seam-pad params (`PAD_SIZE`, edge exclusion), `projections_limit` | `mapping`, N×`convolved_skycell` |
| `downsample` | `{s,c,k,os}` | `oversampling_factor, single_offset, ignore_mask_bits, output_base` (verify.py:159) | `scc_assembly`, `remap_store` (field mode) |
| `shared_mask` | `{s,c,k}` | masking params | `ffi_set` |
| `diff_background` | `{s,c,k,product_id,label}` | `BackgroundParams` (+step params) or hotpants bg params | `ffi` |
| `diff_image` | `{s,c,k,product_id,label}` | `HotpantsParams` \| `KernelFitParams`+`KernelSubtractParams` | `ffi`, `downsample` (template by group offset), `shared_mask`, [`diff_background`] |
| `epsf` | `{s,c,k,product_id,label}` | `EpsfParams` | `diff_image`, gaia catalog |
| `photometry` | `{event,s,c,k,method,label}` | photometry method params (incl. fitter, aperture/sky controls) | N×`diff_image`, N×`epsf`, centroids, target list |

Notes:
- `scc_assembly`'s input edges enumerate exactly the required convolved cells —
  the scheduler never re-derives the expected set by walking the store.
- Per-FFI diff kinds get their required-set from `ffi_list` (which product ids
  exist for this SCC) filtered by the frame manifest — an indexed count, no
  directory walk.
- Until Phase 2 lands, `scc_assembly` is recorded as a whole-stage checkpoint
  node over today's per-SCC `convolved.zarr` (PR2/PR3) — same trick that killed
  the scan in the earlier spike.

---

## 7. Storage layout

```
data_root/
  ps1_skycells_zarr/
    ps1_skycells.zarr                          # raw grizy (exists, unchanged)
    ps1_combined.zarr                          # NEW  {proj}/{skycell}/{fp}/{arrays.npz,headers.json,removed_stars.json}
    ps1_convolved.zarr                         # NEW  {proj}/{skycell}/{fp}/...
  bookkeeping/
    provenance.db                              # NEW  derived index (rebuildable)
    spool/{host}.{pid}.jsonl                   # NEW  worker sidecars
  s{SSSS}/c{C}/k{K}/
    ffi_list.parquet / ffi_list.csv            # PR0 (in-flight)
    mapping/oversampling_{N}/...
    convolved.zarr                             # LEGACY: read-only back-compat; never written after PR5
    remap/oversampling_{N}/...                 # remap store (exists)
    templates/oversampling_{N}/...             # downsample products (exists)
    diff/{stage_label}/{recipe_fp}/            # NEW (Phase D2): SCC-scoped diff store
      tess{pid}_{label}.fits.fz                #   per-FFI diff/background images
      tess{pid}_gridded_epsf.npz ...           #   per-FFI ePSF products
      shared_mask.fits.fz, indexes, ...
  events/{event}/s{SSSS}_c{C}_k{K}/
    frames.csv, event_job.json
    ws/                                        # exploration runs + photometry (stays event-scoped)
```

FITS `location`s are stored as **logical** paths (`fits_logical_path`); readers
resolve variants. The remap store (npz+JSON), ePSF npz, parquet, and zarr kinds
each declare their own completeness marker for reindex.

---

## 8. Database schema (`bookkeeping/provenance.db`)

Unchanged from rev. 1 (WAL, single writer): `artifacts(fingerprint PK, kind,
spatial_key, recipe_id→recipes, location, state, bytes, wall_time_s,
produced_by, created_at)`, `recipes(recipe_id PK, kind, params_json,
code_version, git_sha, created_at)`, `artifact_inputs(fingerprint,
input_fingerprint)`, `input_files(kind, key, spatial_key, bytes, mtime,
checksum, source, batch_id)`; indexes on `(kind,spatial_key)`, `(recipe_id)`,
`(kind,state)`.

Scale note: per-FFI diff kinds add ~3–4 rows × #FFIs × #SCCs per sector
(~10⁴–10⁵ rows/sector) — trivial for SQLite; no sharding needed yet.
`input_files` for FFIs is fed from `ffi_list` rows at ingest (no re-stat).

---

## 9. Fingerprinting spec (`common/provenance/fingerprint.py`)

Unchanged from rev. 1:

```python
RECIPE_SCHEMA_VERSION = 1  # bump on ANY producer algorithm change

def canonical(obj) -> bytes: ...   # sorted keys; floats rounded 1e-9, no -0.0;
                                   # tuples→lists; NaN/inf rejected; golden-tested
def recipe_id(kind, params, code_version) -> str: ...      # sha256[:16]
def fingerprint(kind, spatial_key, recipe_id, input_fps) -> str: ...  # sha256[:24]
```

- Diff recipes come from `dataclasses.asdict(stage_params)` — already strict
  allow-lists, so drift-proof by construction.
- FFI input "fingerprints" = `H("ffi", logical_basename, size, mtime)` from the
  `ffi_list` row.
- Golden tests pin `canonical` bytes.

---

## 10. Publish / ingest / query protocol

**Publish** (`common/provenance/publish.py`): write under
`_tmp_{fp}_{pid}` → atomic rename to the fingerprinted key → append one JSON
line to `bookkeeping/spool/{host}.{pid}.jsonl` (`O_APPEND`, lock-free). A
`_provenance.json` inside each published directory makes stores
self-describing for reindex.

**FITS hardening:** for `.fits.fz` products, fpack to a temp name in the
destination directory and `os.replace` onto the final `.fits.fz` (today
`fpack_plain_fits`, `fits_io.py:44`, writes the final name directly and
pre-unlinks — a mid-write window). Small change in `fits_io`, benefits every
FITS product.

**Ingest** (`ingest.py`): supervisor rotates each spool file (rename → fresh
fd), drains into `provenance.db` in one transaction (idempotent
`INSERT OR REPLACE`), deletes the rotated file. Sole writer. Hook: the
supervisor loop body around `scheduler.py:2080-2083` (global, once per pass,
throttled), alongside `write_verify_in_flight`.

**Query** (`store.py`):

```python
def scc_stage_complete(required_fps) -> bool   # one indexed count
def missing_fingerprints(required_fps) -> list[str]
```

Authoritative fallback on index lag: `stat` only the missing fingerprinted
keys — never the whole set. A fault-injection test store raises on any
directory walk to enforce the no-scan hot path.

---

## 11. Killing the scans — exact call-site changes

### Template side (the O(cells) scan)
- `run_stage.py:272`: after a successful `ps1_process`, emit the
  `scc_assembly` checkpoint sidecar (deterministic fingerprint recomputed from
  the resolved config; recipe = psf_sigma/saturation/removal params +
  `projections_limit`; location = existing `scc_convolved_zarr` path — no bytes
  move). Try/except-guarded, non-fatal. Manifests still written (dual-write
  window).
- `scheduler.py:1103` `_run_verify_pass`: before enqueuing a `VerifyTask`, try
  `store.scc_stage_complete([expected_fp])` with the fingerprint **recomputed
  fresh from the current config** (config drift ⇒ miss ⇒ fall open to the
  legacy path). On hit, route through the existing `_apply_verify_outcome`
  (`:890`) → `cache_external_check`, so `promote_stages` and the status grid
  behave identically. On miss: legacy `check_manifests_only` → verify-worker
  scan, unchanged.
- `verify.py` scan helpers (`expected_ps1_process_skycells`,
  `_count_convolved_data_arrays`) survive only under reindex/fallback; the
  post-run recount in `collect_stage_artifacts:1561` is bypassed once the
  checkpoint write is trusted (PR6).
- `mapping`/`remap`/`downsample`/downloads: same checkpoint pattern, emitted at
  `run_stage` success, for graph completeness — their verifies are already
  cheap, so this is provenance value, not perf (can trail PR3).

### Diff side (no completeness → exact per-FFI completeness)
- Producers (`stages/hotpants.py` save site, `stages/gridded_epsf.py` npz save,
  `stages/background/io.py`, masking, kernel_subtract) emit one sidecar per
  published per-FFI file (kind, product_id, recipe fp, input fps, location).
- Required set for a stage = product ids from `frames.csv`/`ffi_list` (minus
  frames excluded by grouping/quality flags recorded in `frames.csv`).
- `diff_verify` gains `diff_stage_complete(scc, stage_label, recipe_fp)` — an
  indexed count against the required set — replacing the last-stage-only marker
  check as the primary path (marker check remains fallback).
- Per-FFI **resume** becomes an index query (`missing_fingerprints`) instead of
  per-file `resolve_pipeline_fits_path` existence probes (which remain as
  belt-and-braces).

---

## 12. Phase 1 — shared combined store (cross-sector)

- Store: `ps1_skycells_zarr/ps1_combined.zarr/{proj}/{skycell}/{fp}/…` (decision #14).
- **Validated finding (carried over):** star removal runs *inside*
  `process_single_cell` downstream of band combine, and the uncertainty array
  does not survive it. The shareable artifact is exactly what
  `ps1_process.band_cache` already holds for padding-source cells:
  `{combined_image (star-removed), combined_mask, headers_data, removed_stars}`.
  The shared store persists that shape cross-run.
- Wiring (fail-open): seed `band_cache` before workers start for regular
  (non-padding-role) skycells under the current recipe fingerprint — reusing the
  existing `regular_cache_hit` fast path (`ingest_worker`, `ps1_process.py:503`)
  unchanged; publish freshly computed regular results at `process_coordinator`'s
  completion sites via optional trailing kwargs defaulting to old behavior.
  Padding-source-role sharing deferred (separate, riskier seam).
- Recipe: saturation/star-removal params + band-combine constants +
  `gaia_version`. Inputs: `raw_skycell`, `source_catalog`.
- Payoff: overlapping sectors skip the raw read + band combine + star removal
  per cell. Final template numerics unchanged (convolution still per-SCC).

---

## 13. Phase 2 — shared convolved store + padding decouple (blocking gate)

- Canonical convolved cell = convolve on a master array padded by
  **same-projection** neighbors only (sector-independent); store sky-keyed in
  `ps1_skycells_zarr/ps1_convolved.zarr`. Cross-projection seams are corrected
  at `scc_assembly`.
- **Hard cut on writes (decision #15):** once PR5 lands, `ps1_process` never
  writes per-SCC `convolved.zarr` again — canonical cells go only to the shared
  store. Consumers (`scc_assembly`/`downsample`, the zarr export tools, verify)
  resolve through one helper: shared store first, then read-only fallback to an
  existing legacy `convolved.zarr` for SCCs built before the cut. Legacy stores
  are never migrated or deleted (GC handles them later, report-only).
- **Validated finding (carried over, test-locked in the earlier spike):**
  Gaussian convolution linearity holds to ~1e-15, so the **exact** correction is
  `convolve(canonical cell with gap zeroed) + convolve(reprojected neighbor
  patch alone at its true position)` — convolve the patch separately, add the
  result. The naive zero-gap shortcut is **not** safe: up to ~50% flux deficit
  at the seam edge, tapering over ~1 truncation radius (~470 px). The
  correction is real production code in the per-row loop
  (`ps1_process.py`/`cross_projection_padding.py`), not a value splice.
  Re-land the linearity/bias-guard tests with the fresh implementation.
- **Blocking numeric-equivalence gate:** a real SCC assembled from shared +
  seam-corrected cells must match today's `convolved.zarr` within tolerance in
  the downsample/template comparison harness. Requires real data access —
  a supervised run, not CI-only.

---

## 14. Diff-side tracking (new scope)

### 14.1 Phase D1 — track in place (event-scoped locations unchanged)
Wrap the existing per-FFI save sites with publish sidecars; recipes from
`stage_params` dataclasses; input edges: `ffi` (product id → `ffi_list` row),
template FITS chosen by `find_template_by_offset` (→ the `downsample` node),
`shared_mask`, upstream label (e.g. epsf ← diff_image). `diff_verify` gains the
indexed per-FFI completeness query. No storage moves; `frames.csv` and progress
sidecars unchanged. This alone answers "which template/ePSF/params produced
this diff frame?" as a SELECT.

### 14.2 Phase D2 — SCC-scoped diff store (cross-event sharing)
- New tree `s/c/k/diff/{stage_label}/{recipe_fp}/` (helper in `scc_paths.py`
  beside `scc_templates_dir`): per-FFI files keyed inside a per-recipe
  directory — one tree per finalized parameter set (decision #12), so
  exploration configs never collide with finalized ones and a param change
  mints a sibling tree instead of clobbering.
- `run_config_pipeline` gains a `publish_scc` config switch (decision #13):
  when set, per-FFI stage outputs write to (and skip-check against) the SCC
  store; the event workspace keeps only references (index/symlink) for the
  stages that ran SCC-scoped. First event on an SCC pays the cost; later
  events' diff stages become index hits.
- **Existing event workspaces are untouched (decision #16):** no migration, no
  symlink injection, no rewrites of any `events/…/ws/` tree that already
  exists. The SCC store applies only to runs launched with `publish_scc`;
  legacy workspaces keep working exactly as today.
- Photometry stages keep writing under `events/…/ws/`, published as per-event
  `photometry` nodes whose input edges point at the SCC-scoped diff/epsf
  fingerprints — the graph records exactly which shared products each
  lightcurve consumed.
- Orchestration: dispatch stays one job per event×SCC; the win is that shared
  stages inside the run are skip-hits. (A later optimization can hoist
  finalized diff imaging into its own SCC-level stage between `downsample` and
  `diff`, but that is orchestration surgery and is deliberately out of scope.)

---

## 15. Orchestrator / scheduler / status integration

- Completeness for template stages and diff moves to `store` queries surfaced
  through the **existing** `external_check` mechanism — `promote_stages`
  (`scheduler.py:1574`), `stage_absence_probe`, and the 7-column status grid
  (`pipeline_spec.STATUS_GRID_STAGES`) need no changes.
- Supervisor gains the spool-ingest drain in its loop (§10); it remains the
  sole writer of both DBs.
- Run-state DB keeps scheduling only (Condor ids, retries, leases,
  notifications); it stops being a completeness oracle.
- Work units become per-fingerprint where it pays: `ps1_process` per-cell
  (Phase 1/2), diff per-FFI (Phase D1/D2).

---

## 16. Migration & bootstrap

1. **PR0:** landed (`7711a86`): `ffi_list`/download refactor; consumers converted,
   atomic writes, `tests/test_ffi_list.py` on branch.
2. Land `common/provenance/` + empty DB + `reindex` + `syndiff bookkeeping`
   CLI (no behavior change). `reindex` walks: raw/combined/convolved stores,
   per-SCC trees (convolved.zarr, remap manifest, templates dir), diff
   workspaces — using `fits_variants` logical keys and per-kind markers; legacy
   products → `legacy_unverified` (decision #8).
3. **Dual-write window:** manifests + sidecars both written; scheduler prefers
   the DB, falls open to manifests/scans. Remove manifests after one green
   campaign.
4. Migration discipline as proven by `migrate_scc_event_layout.py` /
   `migrate_field_remap_store.py`: copy-never-delete, idempotent, verified,
   supervisor drained first.

---

## 17. Concurrency, failure matrix, GC

| Event | Guarantee |
|---|---|
| Worker crash mid-write | only `_tmp_*` orphan; no sidecar; index unaffected |
| Two workers build same fp | both rename to same key; identical bytes; idempotent sidecars |
| Sidecar written, ingest lagging | `stat` fallback on missing keys only |
| DB lost | `reindex` rebuilds from content |
| Config change mid-campaign | new fingerprints; old artifacts untouched |
| FFI re-downloaded | `ffi_list` row changes → per-FFI diff cone re-fingerprints |
| fpack mid-write crash | temp `.fz` orphan only (after §10 hardening) |

GC (`syndiff bookkeeping gc`): mark reachable from active configs, report-only
first; per-recipe diff trees make "sweep abandoned exploration params" a
directory-level operation. Never touches raw inputs.

---

## 18. Testing strategy

- Golden `canonical`/fingerprint bytes; per-kind `recipe_params` builders;
  Merkle invalidation (flip one param → exactly the downstream cone).
- Concurrency publish + idempotent ingest; `reindex` == live index.
- Fault-injection store proving **no directory walk** on the scheduling path.
- ps1_process checkpoint: hit routes through `_apply_verify_outcome`
  (promotion works); config drift forces a miss; fail-open on cold SCC.
- Phase-2 blocking gate (§13) + re-landed linearity/bias-guard tests.
- Diff: fixture SCC with two events — second event's diff stages are index
  hits; photometry nodes carry correct input edges; per-FFI resume rebuilds
  only missing product ids.
- Full suite via `/home/kshukawa/miniforge3/envs/syndiff/bin/python -m pytest`
  with `PYTHONHASHSEED=0` (base env has no pytest).

---

## 19. Phased PRs

| PR | Content | Accept when |
|---|---|---|
| **PR0** | Land in-flight `ffi_list` + download refactor | existing suite green; grouping/remap consume `ffi_list` with zero FITS reads |
| **PR1** | Provenance core: `common/provenance/{fingerprint,model,store,publish,ingest,reindex,cli}.py`, schema, kind registry (template **and** diff kinds), fpack atomic hardening in `fits_io` | golden tests; `reindex` populates from an existing tree; no compute-path change |
| **PR2** | Publish/ingest plumbing: `scc_assembly` checkpoint at `run_stage` success + supervisor spool drain (dual-write with manifests) | checkpoint sidecars ingested idempotently after a real ps1_process run |
| **PR3** | Scheduler cutover: checkpoint-first in `_run_verify_pass`, fail-open to legacy scan; other template stages' checkpoints | fault-injection proves no walk on hit; promotion identical; skip decision ms not minutes |
| **PR4** | Shared combined store + per-cell skip (§12) | two-sector fixture reuses combined cells; templates bit-identical |
| **PR-D1** | Diff per-FFI tracking in place (§14.1) + `diff_verify` indexed completeness | per-FFI rows correct after a real diff run; resume via index |
| **PR-D2** | SCC-scoped diff store + `publish_scc` (§14.2) + per-event photometry nodes | two-event fixture: second event hits the shared store; lightcurve edges correct |
| **PR5** | Shared convolved + padding decouple + hard-cut of per-SCC `convolved.zarr` writes with legacy read fallback (§13) | **blocking** real-SCC numeric-equivalence gate; legacy SCCs still readable |
| **PR6** | GC (report-only), manifest retirement, dead verify-scan removal, docs | GC report correct; scheduling unaffected without manifests |

PR-D1 depends only on PR1 and can proceed in parallel with PR2–PR4. PR-D2
needs PR-D1. PR5 is last among numeric-risk changes and needs real data.

---

## 20. File inventory

**New:** `syndiff_pipeline/common/provenance/` (7 modules);
`ps1_skycells_zarr/ps1_combined.zarr`, `ps1_skycells_zarr/ps1_convolved.zarr`,
`data_root/bookkeeping/`;
`scc_paths.py` helpers (`scc_diff_store_dir`, provenance/spool paths);
`template_creation/processing/combined_store.py`, `convolved_store.py`;
`template_creation/orchestration/provenance_checkpoint.py`;
diff publish shims in `difference_imaging/orchestration/`.

**Modified:** `fits_io.py` (atomic fpack); `run_stage.py` (checkpoint emit);
`scheduler.py` (checkpoint-first verify + ingest drain); `ps1_process.py`
(Phase 1 seeding/publish; Phase 2 canonical halo + seam correction);
`cross_projection_padding.py` (Phase 2); `stages/hotpants.py`,
`gridded_epsf.py`, `background/io.py`, `kernel_subtract.py`, `masking.py`
(publish sidecars; D2 store paths); `orchestration/execute.py` (`publish_scc`);
`diff_verify.py`; `support/template_resolution.py` (D2 references).

**Deleted (PR6):** hot-path scan in `verify_ps1_process` path; post-run
recount in `collect_stage_artifacts`; JSON manifest machinery after the
dual-write window.

---

## 21. Open questions (non-blocking)

- **D2 event-side referencing:** symlinks from `ws/{label}/` into the SCC store
  vs an index file consumers resolve through. Symlinks are simplest but NFS
  tooling sometimes fights them; decide at PR-D2 with the photometry reader
  code in front of us.
- **Hotpants-internal background vs standalone background stage:** both are
  tracked (different labels/recipes); whether the standalone stage's
  `stack.npz` cube is one artifact or per-frame + cube nodes — decide from how
  photometry consumes it.
- **Gaia version granularity** (scalar vs per-footprint artifacts): start
  scalar.
- **`scc_assembly` materialization** (keep per-SCC `convolved.zarr` on disk vs
  assemble on demand): decide with Phase-2 numbers.
- **DB sharding:** defer until row counts warrant.
