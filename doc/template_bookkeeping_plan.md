# Template-Creation Bookkeeping — Comprehensive Design & Implementation Plan

**Status:** approved-in-principle; this is the authoritative spec we implement from.
**Branch context:** `distortion_aware_templates` (post SCC-workspace split).
**Premise:** the current bookkeeping (coarse per-SCC JSON manifests + a run-state
DB that doubles as a completeness oracle + O(cells) NFS verify scans) is being
**replaced**, not extended. We are free to build the correct thing.

---

## 0. Executive summary

We replace all template-creation bookkeeping with a single abstraction: a
**content-addressed provenance graph** — a small, purpose-built build system for
the pipeline's artifacts.

- Every artifact (FFI set, raw skycell, combined skycell, convolved skycell,
  mapping, SCC assembly, template) is a **node** named by a **Merkle
  fingerprint** = `H(kind, spatial_key, recipe_params, sorted(input_fingerprints),
  code_version)`.
- **Combined and convolved skycells are keyed by sky position only** (no
  sector/camera/ccd), so they are automatically **shared across every sector**
  pointed at the same sky. PS1 skycells sit on a fixed sky tessellation, and — now
  confirmed — the band-combine and star-removal steps are pure functions of the
  skycell footprint. This is the big storage/compute win the user asked for.
- **The slow "verify" scan disappears.** Completeness becomes an indexed database
  query over a set of required fingerprints; the post-run manifest scan is
  replaced by sidecar records the worker emits at publish time. No more
  `os.scandir`-per-cell on NFS.
- **Config drift is first-class.** Every fingerprint maps to a stored, fully
  materialized recipe (`params_json`), so "what parameters produced this cell?" is
  a `SELECT`, and a config change mints new fingerprints without clobbering old
  products.
- **Correctness invariants:** content authority (the bytes at the fingerprinted
  key *are* the truth; the DB is a rebuildable index), atomic publish, and a
  single DB writer (the supervisor) fed by lock-free worker sidecars — the same
  NFS-safe pattern already proven by the run-state DB.

Delivered in six reversible PRs; the verify scan dies in PR3, cross-sector sharing
lands in PR4 (combined) and PR5 (convolved, behind a blocking numeric gate).

---

## 1. Goals

1. **Track inputs** — TESS FFIs and raw PS1 skycells (when not streamed).
2. **Share intermediates across sectors** — persist and reuse *linearly-combined
   (pre-convolution)* skycells and *convolved* skycells, sky-addressed.
3. **Track outputs** — mapping, WCS groups, catalogs, SCC assemblies, templates —
   each with the exact recipe that produced it.
4. **Recompute only what changed** — per-skycell incrementality, not all-or-nothing
   per SCC.
5. **Provenance under config drift** — full materialized parameters per artifact.
6. **Eliminate the slow verify/scan** — replace O(cells) NFS walks with O(1) index
   queries.

---

## 2. Current state and why it must change

### 2.1 The verify scan (the primary pain)
Verification runs the expensive scan in **two** hot places:

- **Scheduling-time skip decision.** `scheduler.py` gates stage promotion on
  upstream-artifact existence. The fast path `check_manifests_only`
  (`scheduler.py:1141`) reads a JSON manifest and `Path.exists()`-checks every
  listed artifact; on any miss it falls back to `verify_worker.py` running
  `stage_complete` **in a subprocess** (comment: on-disk checks "can walk large
  NFS trees and parse"). For `ps1_process`, `verify_ps1_process`
  (`verify.py:944`) re-derives the expected skycell set from the mapping CSV
  (`expected_ps1_process_skycells`) and `_count_convolved_data_arrays` does **one
  `os.scandir` per expected skycell** (thousands per SCC). `verify_ps1_download`
  is per skycell × 12 arrays. The header comment records **~30 min on NFS** before
  the metadata-only optimization — and it is *still* O(cells) stats each verify.
- **Post-execution manifest write.** `collect_stage_artifacts` (`verify.py:1383`)
  re-runs the same scan to compute `produced_count`/`artifacts`, then
  `write_manifest` persists it.

### 2.2 Coarse, lossy, file-scattered provenance
`verify.config_fingerprint` (`verify.py:111`) hashes the params-that-matter per
stage, but: one hash per whole SCC/stage (never per-cell), only the *hash* is
stored (not the values), and it lives in scattered JSON files — not queryable.

### 2.3 No cross-sector sharing of the expensive product
Raw grizy is already globally shared (`ps1_skycells.zarr`, keyed
`projection/skycell`). But `convolved.zarr` is **per-SCC**, ~100k files each, and
is rebuilt from scratch for every sector even when footprints overlap heavily. The
*only* reason it is per-SCC is cross-projection padding at projection seams.

### 2.4 Run-state DB conflates scheduling with completeness
`common/orchestration/state.py` (single-writer supervisor, WAL/NFS-hardened) is
good at scheduling but is wrongly also the completeness oracle (`artifacts` table,
`external_verify_*`). We keep it for scheduling and remove the completeness role.

---

## 3. The two foundational facts

1. **PS1 products are sky-addressed, not sector-addressed.** Skycells live on a
   fixed sky tessellation (`projection.skycell`). The band-combined image
   (`process_skycell_bands`) and the convolved-cell interior are pure functions of
   the raw cell + parameters, independent of any TESS SCC. **Star removal is
   shareable** (confirmed): `project_gaia_to_skycell` filters Gaia to the skycell
   footprint and `remove_background` removes segments by in-footprint stars below a
   magnitude threshold — a pure function of `(skycell footprint, Gaia catalog
   version, mag threshold)`. The **only** SCC-specific part of `ps1_process` is
   cross-projection padding at projection seams.

2. **Identity = recipe + inputs, hashed (Merkle).** Two computations are the same
   iff the same algorithm ran with the same params on the same inputs. Making the
   artifact's *name* that hash makes sameness decidable by string equality and
   makes invalidation automatic and correct — change any upstream param or byte and
   the entire downstream cone re-fingerprints, while old products survive.

---

## 4. Locked decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Bookkeeping model | Content-addressed provenance graph; run-state DB demoted to scheduling | Correct shape; everything derives from it |
| 2 | Rollout order | Graph core + shared **combined** store first; convolved second | Foundation + zero-numeric-risk win before the risky refactor |
| 3 | Padding decouple | Same-projection canonical halo + post-hoc cross-projection seams | Max reuse; physically correct decomposition |
| 4 | `code_version` | Hand-bumped `RECIPE_SCHEMA_VERSION` per producer + git SHA stored for forensics | Deliberate invalidation; auto-SHA would re-fingerprint on unrelated commits |
| 5 | Raw-skycell version token | `(size, mtime, download_batch_id)`; `--checksum` audit mode in reindex | Per-cell checksums too costly on hot path |
| 6 | GC | Reference-counted against active configs; **ships report-only**, deletion later | Bound storage without irreversible early deletes |
| 7 | reindex bootstrap of legacy products | Mark `legacy_unverified`, rebuild lazily; `--trust-current-config` escape hatch | Never silently bless products made with unknown params |
| 8 | Sidecar spool transport | Per-host `O_APPEND` JSONL under `bookkeeping/spool/`; supervisor rotates+drains | Simplest correct; single-writer preserved |
| 9 | Non-template externals | Keep run-state `artifacts` table during transition; migrate later | Bounds PR scope/risk |
| 10 | §9.1 star-removal coupling | **Resolved: shareable**; fold Gaia catalog version + mag threshold into combined recipe | Confirmed pure function of footprint |

---

## 5. Core model: the provenance graph

Every produced/consumed unit is an **artifact node**:

```
Artifact
  kind          raw_skycell | combined_skycell | convolved_skycell
                | ffi_set | mapping | wcs_group | source_catalog
                | scc_assembly | template
  spatial_key   canonical dict identifying the region:
                  skycell:  {"projection":..,"skycell":..}
                  scc:      {"sector":..,"camera":..,"ccd":..,"oversampling":..}
  recipe        full materialized params (stored) + code_version
  inputs        fingerprints of upstream artifacts consumed (graph edges)
  fingerprint   = H(kind, spatial_key, recipe_id, sorted(input_fingerprints))
  location      zarr key / path where finalized bytes live
  state         building | complete | failed
  meta          bytes, wall_time, produced_by (run_id/host), created_at
```

### The Merkle property
A node's fingerprint depends on its inputs' fingerprints. Therefore changing a
`psf_sigma`, a `bright_star_mag_threshold`, or re-downloading a raw cell
re-fingerprints exactly the affected downstream cone and nothing else. "What would
rerun?" is a graph diff, not a heuristic.

### The dependency DAG for one template
```
ffi_set(s,c,k) ─▶ mapping(s,c,k,os) ─────────────────────┐
                                                          ▼
raw_skycell(p,cell) ─▶ combined_skycell(p,cell) ─▶ convolved_skycell(p,cell) ─▶ scc_assembly(s,c,k,os) ─▶ template(s,c,k,os)
       ▲ version         ▲ band-combine + star-removal      ▲ psf_sigma,radius,mode   ▲ seam-pad params
       │                 │ (mag threshold, gaia version)    │                         │ (+ N convolved cells,
source_catalog(footprint)┘                                                              mapping)
```
`combined_skycell` and `convolved_skycell` carry **no s,c,k** → shared across all
overlapping sectors. Only `scc_assembly`/`template` are sector-scoped.

### Two invariants that make it correct
- **Content authority.** An artifact exists iff its finalized bytes sit at its
  fingerprinted key. The database is a **derived, rebuildable index** — no
  correctness depends on a DB write. (Inherited from the raw-zarr "path = truth".)
- **Atomic publish.** Bytes go to a temp key, then a single atomic rename to the
  fingerprinted key. A crash leaves only an orphan temp; never a partial artifact
  that looks present. `state=building` rows are advisory hints only.

---

## 6. Artifact kind registry

Recipe params come from one place per kind (`recipe_params(resolved) -> dict`),
migrated out of today's `verify.config_fingerprint` enumeration.

| kind | spatial_key | recipe params | inputs |
|---|---|---|---|
| `ffi_set` | `{s,c,k}` | download source/params | — |
| `raw_skycell` | `{projection,skycell}` | `{}` + `version_token` (size,mtime,batch) | — |
| `source_catalog` | `{projection,skycell}` (footprint) | gaia query params, `gaia_version` | — |
| `mapping` | `{s,c,k,os}` | `oversampling_factor`, `pad_distance`, `tess_buffer` | `ffi_set` |
| `wcs_group` | `{s,c,k}` | `offset_threshold`, savgol window/order, crop_mode | `ffi_set` |
| `combined_skycell` | `{projection,skycell}` | `enable_saturation_correction`, `remove_saturated_stars`, `bright_star_mag_threshold`, band-combine consts, `gaia_version` | `raw_skycell`, `source_catalog` |
| `convolved_skycell` | `{projection,skycell}` | `psf_sigma`, `radius`, `mode` | `combined_skycell` |
| `scc_assembly` | `{s,c,k,os}` | seam-pad params (`PAD_SIZE`, edge exclusion) | `mapping`, N×`convolved_skycell` |
| `template` | `{s,c,k,os}` | `oversampling_factor`, `single_offset`, `ignore_mask_bits`, geometry_mode | `scc_assembly` |

`scc_assembly` is the node whose `artifact_inputs` edges enumerate exactly the
required convolved cells for an SCC — computed once from mapping output and stored,
so the scheduler never re-derives the expected set by walking the store.

---

## 7. Storage layout

```
data_root/
  ps1_skycells_zarr/ps1_skycells.zarr        # raw (exists, unchanged)
  ps1_combined_zarr/ps1_combined.zarr         # NEW  proj/skycell/<fp> -> {data,mask,uncert}
  ps1_convolved_zarr/ps1_convolved.zarr       # NEW  proj/skycell/<fp> -> {data,mask,weight}
  bookkeeping/
    provenance.db                             # NEW  derived index (rebuildable)
    spool/<host>.<pid>.jsonl                   # NEW  worker sidecar append logs
  s{SSSS}/c{C}/k{K}/                           # SCC leaves (existing)
    mapping/oversampling_{N}/ ...
    convolved.zarr                             # Phase 1: still per-SCC ASSEMBLY (references shared cells later)
    templates/oversampling_{N}/ ...
```

Fingerprint in the key means recipes never collide and old products survive config
changes. Storage growth is bounded by GC (§16).

---

## 8. Database schema (`bookkeeping/provenance.db`)

```sql
PRAGMA journal_mode=WAL;   -- single writer (supervisor), many readers

CREATE TABLE artifacts (
    fingerprint  TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    spatial_key  TEXT NOT NULL,     -- canonical json
    recipe_id    TEXT NOT NULL REFERENCES recipes(recipe_id),
    location     TEXT,
    state        TEXT NOT NULL,     -- building | complete | failed
    bytes        INTEGER,
    wall_time_s  REAL,
    produced_by  TEXT,              -- run_id / host
    created_at   TEXT
);
CREATE TABLE recipes (
    recipe_id    TEXT PRIMARY KEY,  -- H(kind, params, code_version)
    kind         TEXT NOT NULL,
    params_json  TEXT NOT NULL,     -- FULL resolved params (human-readable)
    code_version TEXT,              -- RECIPE_SCHEMA_VERSION
    git_sha      TEXT,              -- forensics
    created_at   TEXT
);
CREATE TABLE artifact_inputs (
    fingerprint       TEXT NOT NULL,   -- child
    input_fingerprint TEXT NOT NULL,   -- parent
    PRIMARY KEY (fingerprint, input_fingerprint)
);
CREATE TABLE input_files (             -- FFIs & raw skycells (input tracking)
    kind         TEXT NOT NULL,        -- ffi | raw_skycell
    key          TEXT NOT NULL,        -- basename or projection.skycell
    spatial_key  TEXT NOT NULL,
    bytes        INTEGER, mtime TEXT, checksum TEXT,
    source       TEXT, batch_id TEXT, recorded_at TEXT,
    PRIMARY KEY (kind, key)
);

CREATE INDEX ix_art_kind_spatial ON artifacts(kind, spatial_key);
CREATE INDEX ix_art_recipe       ON artifacts(recipe_id);
CREATE INDEX ix_art_state        ON artifacts(kind, state);
```

This single schema subsumes: per-stage JSON manifests, the scattered
`config_fingerprint`, and the run-state `artifacts` completeness role.
`recipes.params_json` is the queryable answer to config drift.

---

## 9. Fingerprinting spec (`common/provenance/fingerprint.py`)

```python
RECIPE_SCHEMA_VERSION = 1          # bump on ANY producer algorithm change

def canonical(obj) -> bytes:
    # deterministic: sorted keys; floats normalized (round to 1e-9, no -0.0);
    # tuples->lists; reject NaN/inf; utf-8. Golden-tested byte output.

def recipe_id(kind, params: dict, code_version: str) -> str:
    return sha256(canonical([kind, params, code_version])).hexdigest()[:16]

def fingerprint(kind, spatial_key: dict, recipe_id: str,
                input_fingerprints: list[str]) -> str:
    return sha256(canonical([kind, spatial_key, recipe_id,
                             sorted(input_fingerprints)])).hexdigest()[:24]
```

- **One definition, reused everywhere** (producers, scheduler, reindex). Move the
  per-stage param enumeration out of `verify.config_fingerprint` into per-kind
  `recipe_params()` builders in `common/provenance/model.py`.
- **Golden tests** pin `canonical` bytes so fingerprints never silently drift
  across python/library versions.
- **code_version**: `str(RECIPE_SCHEMA_VERSION)`; `git_sha` recorded separately.

---

## 10. Publish / ingest / query protocol

**Producer publish** (`common/provenance/publish.py`), replacing the write side of
`zarr_utils.save_convolved_results` and the new combined writer:
```python
def publish_skycell(store, kind, projection, skycell, fp, arrays, *, recipe, inputs):
    tmp = f"{projection}/{skycell}/_tmp_{fp}_{pid}"
    write arrays under tmp
    atomic_rename(tmp -> f"{projection}/{skycell}/{fp}")
    emit_sidecar(kind, fp, spatial_key, recipe, inputs, location, bytes, wall_time)
```
`emit_sidecar` appends one JSON line to `bookkeeping/spool/<host>.<pid>.jsonl`
(`O_APPEND`, lock-free).

**Supervisor ingest** (`ingest.py`, in the daemon loop): rotate each spool file
(rename → new fd), drain into `provenance.db` in one transaction
(`INSERT OR REPLACE` on fingerprint; recipes `INSERT OR IGNORE`; edges), delete the
rotated file. **Sole writer.** Idempotent.

**Scheduler query** (`store.py`):
```python
def scc_stage_complete(store, kind, scc_key, required_fps) -> bool:
    # SELECT count(*) FROM artifacts
    # WHERE state='complete' AND fingerprint IN (required_fps)
    # -> == len(required_fps).  One indexed query. No filesystem walk.
def missing_fingerprints(store, required_fps) -> list[str]: ...
```
Authoritative fallback (rare, index-lag only): `stat` **only** the missing
fingerprinted keys — never the whole set.

---

## 11. Killing the verify scan — before/after and exact call sites

| Question | Today | New |
|---|---|---|
| ps1_process complete for SCC+config? | re-derive expected set + `os.scandir`/cell in subprocess (NFS walk) | `scc_stage_complete` — one indexed query |
| What did the run produce? | re-scan store, `write_manifest` | sidecars already emitted at publish |
| Does this cell exist (skip it)? | not answerable per-cell | index lookup / `stat(fp key)` |

**Call-site changes:**
- `ps1_process.saver_worker` (`ps1_process.py:841`) / `zarr_utils.save_convolved_results`
  → wrap with `publish_skycell` (fingerprinted key + sidecar).
- `ps1_process` main loop → per-cell skip: consult index/`stat` before enqueuing;
  today's all-or-nothing `overwrite` becomes per-fingerprint.
- `scheduler.py:1141` `check_manifests_only` → `store.scc_stage_complete`.
  `verify_worker.py` subprocess NFS walk → **deleted for template stages.**
- `run_stage.py:272` / `cli.py:1167` post-run `collect_stage_artifacts` +
  `write_manifest` → **deleted** (completeness recorded at publish).
- `state.py` `external_verify_*` (run-state `artifacts`) → retained only for
  non-template externals (decision #9).
- `verify.py` per-stage `verify_*`, `_count_convolved_data_arrays`,
  `expected_ps1_process_skycells` scan → retained **only** behind `reindex` (bulk,
  offline), never on the hot path.

---

## 12. Phase 1 — shared combined store + per-cell incrementality

- New `ps1_combined.zarr`, keyed `projection/skycell/<combined_fp>` →
  `{data, mask, uncert}` (outputs of `process_skycell_bands` + star removal).
- `combined_fp` recipe = saturation/star-removal params + band-combine constants +
  `gaia_version` (decision #10). Inputs: `raw_skycell`, `source_catalog`.
- `ps1_process` flow becomes: for each needed skycell, compute `combined_fp`;
  if `complete` → load shared combined cell; else build + `publish_skycell` to the
  shared store. Convolution still runs per-SCC over the combined inputs (so **no
  change to final template numerics** — this is the safe win).
- **Payoff:** every sector overlapping an earlier one skips the ~1.6 GB/cell raw
  read + the band-combine + star removal. Given many sectors share pointing, this
  is large.
- **Gaia versioning:** stamp a `gaia_version` (catalog release id / ingest date)
  into `source_catalog` recipe; a Gaia refresh re-fingerprints combined cells.

---

## 13. Phase 2 — shared convolved store + padding decouple (blocking gate)

**Problem:** today padding is applied to a master row array **before** convolution,
so each stored convolved cell bakes neighbor flux (SCC-specific) into its border
(`cross_projection_padding.py`, `PAD_SIZE=480`, Gaussian `radius≈470`). A convolved
cell needs a neighbor halo to be correct at its own edges.

**Decouple (decision #3):** convolve each cell on a master array padded by its
**same-projection** neighbors only (always present, sector-independent), and record
the **valid region**. Store that canonical convolved cell sky-keyed in
`ps1_convolved.zarr`. Remaining **cross-projection** seams (different PS1
projections meeting inside one SCC footprint) are re-padded at **`scc_assembly`
time**, drawing from other *shared* convolved cells — so even seam cells rarely
recompute.

**`scc_assembly`** becomes a cheap reference-and-stitch step producing the per-SCC
mosaic that `templates` consumes; its inputs are the shared convolved cells +
mapping.

**Blocking numeric-equivalence gate:** assemble a real SCC from shared + re-padded
cells and require agreement with today's baked-in `convolved.zarr` within tolerance,
wired into the existing downsample/template comparison harness. Given this branch's
roll-sign/seam history, this is non-negotiable and gates the PR.

---

## 14. Orchestrator / scheduler integration

- **Completeness** for `mapping`, `ps1_process`(→`scc_assembly`), `templates`
  moves to `store.scc_stage_complete`. The scheduler promotes a stage when its
  required fingerprints are `complete`.
- **Work units become per-fingerprint** for skycell kinds: a run's `ps1_process`
  work = `missing_fingerprints(required)`, enabling changed-only reruns without
  rebuilding the SCC.
- **Supervisor** gains the ingest loop (drain spool → DB) alongside its existing
  duties; it remains the sole writer of both DBs.
- **Run-state DB** keeps scheduling/execution (Condor ids, retries, leases,
  pause/resume, notifications). It stops answering "is it done?".

---

## 15. Migration & bootstrap

1. Land `common/provenance/` + empty `provenance.db` + `reindex` + `bookkeeping`
   CLI (no behavior change).
2. `syndiff bookkeeping reindex` walks existing raw/combined/convolved zarr + SCC
   dirs and registers artifacts. Legacy on-disk products get
   `kind=..._legacy_unverified` (decision #7) and rebuild lazily on first use;
   `--trust-current-config` blesses known-safe products. This is the one-time
   expensive scan — offline, not on the hot path.
3. **Dual-write window:** keep writing JSON manifests **and** sidecars; scheduler
   reads the DB, manifests remain a fallback. Remove manifests after one green
   campaign (PR6).
4. Follow the proven migration discipline (`scripts/migrate_scc_event_layout.py`):
   copy/never-delete, idempotent, guarded, supervisor drained first.

---

## 16. Concurrency, failure matrix, and GC

**Failure matrix (must hold):**

| Event | Guarantee |
|---|---|
| Worker crash mid-write | only `_tmp_*` orphan; no publish, no sidecar; index unaffected |
| Two workers build same fp | both atomic-rename same key; identical bytes; sidecars idempotent |
| Sidecar written, ingest lagging | `stat` fallback finds key → correct; index catches up |
| DB lost/corrupted | `reindex` rebuilds from content; zero data loss |
| Config changes mid-campaign | new fingerprints; old artifacts untouched; only changed cone rebuilds |
| Raw skycell re-downloaded | `version_token` changes → combined/convolved re-fingerprint downstream |
| Orphan temp keys accumulate | periodic sweep removes `_tmp_*` older than a grace window |

**GC (decision #6):** `syndiff bookkeeping gc` marks every fingerprint reachable
from the **active configs** in `config/`, then reports (initially) unreferenced
`complete` artifacts older than a grace window. Flip to actual deletion after a
campaign of clean reports. Never touches raw inputs.

---

## 17. Testing strategy

- **Unit:** `canonical`/`fingerprint` golden bytes; per-kind `recipe_params`
  builders; **Merkle invalidation** (flip one param → exactly the downstream cone
  re-fingerprints).
- **Concurrency:** N processes publishing overlapping fingerprints to a temp store
  → no partial artifacts; idempotent ingest; `reindex` output == live index.
- **NFS-append:** targeted test of `O_APPEND` JSONL semantics + supervisor
  rotate/drain on the actual filesystem.
- **Perf regression:** scheduling-time completeness check must issue no `scandir`
  on the hot path — enforced via a fault-injection store that raises on directory
  walk.
- **Phase-2 equivalence gate (blocking):** SCC assembled from shared+seam cells vs
  today's `convolved.zarr`, within tolerance, in the downsample/template harness.
- **End-to-end:** a two-sector overlap fixture proving the second sector reuses
  combined (Phase 1) and convolved (Phase 2) cells and rebuilds only genuinely new
  ones.

---

## 18. Phased PR breakdown (implementation order + acceptance)

**PR1 — provenance core (read-only, no behavior change).**
`fingerprint`, `model` (kinds + `recipe_params`), `store`, schema, `reindex`,
`bookkeeping` CLI. *Accept:* golden fingerprint tests; `reindex` populates DB from
existing disk; queries return correct counts; nothing in compute path changed.

**PR2 — publish/ingest plumbing (dual-write).**
Sidecar spool + supervisor ingest; wrap `save_convolved_results` with
`publish_skycell` (still writing manifests too). *Accept:* concurrency test green;
sidecars ingested idempotently; DB matches disk after a real ps1_process run.

**PR3 — scheduler cutover for template stages (the scan dies).**
`scc_stage_complete` replaces `check_manifests_only`/`verify_worker` for template
stages; delete the hot-path `scandir`. *Accept:* perf test proves no walk on
scheduling; a full campaign schedules identically to before but without the
subprocess NFS scan; wall-clock of the skip decision drops from minutes to ms.

**PR4 — shared combined store + per-cell skip (Phase 1).**
`ps1_combined.zarr`; per-fingerprint build/skip in `ps1_process`; `source_catalog`
+ `gaia_version`. *Accept:* two-sector overlap fixture reuses combined cells; final
templates bit-identical to pre-PR (convolution unchanged).

**PR5 — shared convolved store + padding decouple (Phase 2).**
`ps1_convolved.zarr`; `scc_assembly` reference-and-stitch; same-projection halo +
post-hoc seams. *Accept:* **blocking** numeric-equivalence gate vs current
`convolved.zarr`; overlap fixture reuses convolved cells.

**PR6 — GC + manifest removal + docs.**
Reference-counted GC (report-only); retire JSON manifests + dead verify code after
a green campaign; document the model and CLI. *Accept:* GC report correct on a real
tree; manifest removal leaves scheduling unaffected.

Each PR is independently shippable and reversible (additive stores, derived DB).

---

## 19. Module / file change inventory

**New:** `syndiff_pipeline/common/provenance/{__init__,fingerprint,model,store,
publish,ingest,reindex,gc,cli}.py`; `ps1_combined_zarr/`, `ps1_convolved_zarr/`,
`bookkeeping/` stores; `scc_paths.py` helpers for the new stores + `provenance.db`
+ `spool/`.

**Modified:** `ps1_process.py` (publish + per-cell skip + Phase-2 canonical-halo
convolution + `scc_assembly`), `zarr_utils.py` (publish integration),
`cross_projection_padding.py` (seam padding at assembly), `scheduler.py`
(completeness via store), supervisor loop (ingest), `runner_config.py` /
`stages.py` (recipe_params wiring), `common/orchestration/cli.py`,
`common/orchestration/run_stage.py`.

**Deleted (PR3/PR6):** template-stage paths in `verify_worker.py`; post-run
`collect_stage_artifacts`+`write_manifest`; hot-path `_count_convolved_data_arrays`
/ `expected_ps1_process_skycells` scans (kept only under `reindex`); JSON manifest
machinery once dual-write window closes.

---

## 19b. Corrections from PR0 code investigation (authoritative)

Detailed implementer maps: `doc/bookkeeping_pr3_seam_map.md` and
`doc/bookkeeping_pr245_dataflow_map.md`. Key corrections to assumptions above:

- **PR3 promotion-stall (highest risk).** The scheduler's `promote_stages`
  (`state.py:698`) promotes a stage only when `external_verify_attempted` is true,
  and the status UI (`run_report.py:425`) reads the `external_check` cache. The
  cutover must therefore **still record** the `scc_stage_complete` result into the
  run-state `external_check` rows (rewire `scheduler.py:909-918`), not just delete
  the scan — otherwise template stages never promote. `verify_worker.py` is an
  in-process ThreadPool, not a subprocess.
- **PR2 write seam.** Convolved cells are written **flat** (`{skycell}_data` at
  store root, non-atomic delete-then-create) by `save_convolved_results`
  (`zarr_utils.py:78-145`), and the live write path is
  `data_root/convolved_results/sector_..._ccd_.zarr` (`ps1_process.py:1587`) —
  reconcile with `scc_convolved_zarr` before wiring publish. The wrapper changes
  both key structure (→ `{projection}/{skycell}/{fp}`) and atomicity.
- **PR4 combined seam.** Star removal happens **downstream** of the band combiner,
  in the subprocess `process_single_cell` (`ps1_process.py:436-443`). The shareable
  "combined + star-removed" product first exists at the coordinator result
  (`ps1_process.py:711-722`) — publish there, not at the combiner. `combined_fp`
  recipe = band weights + flux constants + `sigma=2.5`/`sigma_mask=50` +
  `remove_saturated_stars` + `bright_star_mag_threshold` + `gaia_version`.
- **PR4 per-cell skip** goes in the task-list build loop (`ps1_process.py:1745-1773`).
- **PR5 decouple** confirmed: drop `apply_cross_projection_padding`
  (`ps1_process.py:1369-1404`) from the pre-convolution path, keep same-projection
  `apply_cross_row_padding` (`:1364`), move the cross-projection reproject/stitch
  (`cross_projection_padding.py:511-536`) into a post-convolution `scc_assembly`.

## 20. Remaining open questions (non-blocking)

- **Gaia version granularity:** a single global `gaia_version` scalar vs per-footprint
  catalog artifacts. Start with the scalar; upgrade if partial Gaia refreshes appear.
- **`scc_assembly` materialization:** keep a per-SCC `convolved.zarr` mosaic on disk
  (simplest for `templates`) vs assemble on-demand into memory at template time.
  Decide with Phase-2 storage/latency numbers.
- **provenance.db sharding:** one global DB vs per-(sector) shards if the artifacts
  table grows very large. Defer until row counts warrant.
```
