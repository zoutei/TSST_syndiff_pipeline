> **Package integration**: `syndiff` stage `ps1_process` · module `template_creation/processing/ps1_process.py` · legacy script `process_ps1.py`  
> **Orchestration docs**: [template pipeline guide](../template_pipeline.md) · [HTCondor](../template_pipeline.md#htcondor-integration)

> **Current-store note:** the current supervised field pipeline uses the shared
> convolved-cell store rooted at `{data_root}/ps1_convolved.zarr` (with legacy
> SCC `convolved.zarr` fallback where configured). Historical descriptions of
> a top-level `convolved_results/` flat Zarr layout below are retained only to
> explain legacy inputs; they are not the live field-template contract. L5
> validates every required master skycell before assembly and fails rather than
> silently skipping a missing convolved cell. See
> [storage layout](../storage_layout.md), [field geometry](../field_geometry.md),
> and [coordinate frames and cropping](../coordinate_frames_and_cropping.md).

# PS1 Template Processing Pipeline — Detailed Technical Reference

`process_ps1.py` builds a deep-sky reference template by reading Pan-STARRS 1 (PS1) skycell images from a Zarr store, combining multi-band exposures, removing backgrounds and saturated stars, stitching cells into large "master arrays" using a sliding window, applying padding at projection and cross-projection boundaries, convolving with a Gaussian PSF, and saving convolved results back to a Zarr store. This document explains every stage in detail.

---

## Table of Contents

1. [Concepts and Data Model](#1-concepts-and-data-model)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Startup: Configuration and Task Scheduling](#3-startup-configuration-and-task-scheduling)
4. [Stage 1 — Reader Workers](#4-stage-1--reader-workers)
5. [Stage 1.5 — Band Combiner Workers](#5-stage-15--band-combiner-workers)
6. [Stage 2 — Process Coordinator and ProcessPoolExecutor](#6-stage-2--process-coordinator-and-processpoolexecutor)
7. [Cell Processing: process_single_cell](#7-cell-processing-process_single_cell)
8. [Stage 3 — Sequential Assembler (Main Thread)](#8-stage-3--sequential-assembler-main-thread)
9. [Cross-Row Padding](#9-cross-row-padding)
10. [Cross-Projection Padding](#10-cross-projection-padding)
11. [Band Cache and Padding Source Pre-loading](#11-band-cache-and-padding-source-pre-loading)
12. [Manual Fallback Loader](#12-manual-fallback-loader)
13. [Stage 4 — Saver Worker](#13-stage-4--saver-worker)
14. [Saturation Correction](#14-saturation-correction)
15. [Memory Management and OOM Prevention](#15-memory-management-and-oom-prevention)
16. [Concurrency Control and Throttling](#16-concurrency-control-and-throttling)
17. [Queue and Buffer Reference](#17-queue-and-buffer-reference)
18. [Key Constants](#18-key-constants)
19. [Running the Pipeline](#19-running-the-pipeline)
20. [Log Prefixes Quick Reference](#20-log-prefixes-quick-reference)

---

## 1. Concepts and Data Model

### Skycells

The sky is divided into a grid of **skycells** by the PS1 survey. Each skycell has a name like `skycell.2556.082` where `2556` is the **projection** (a large sky tile) and `082` is the **cell index within that projection**. Each skycell is ~4800×4800 pixels and has four band images (r, i, z, y), corresponding mask arrays (bit-packed uint16), and per-band variance ("weight") maps stored in the Zarr archive.

### Projections

A **projection** is a group of skycells that share the same tangent-point WCS. Projections are processed one at a time, sequentially. Within a projection, cells are arranged in rows (y-coordinate) and columns (x-coordinate).

### The Sliding Window

Processing a projection requires assembling cells row by row. The pipeline uses a **two-row sliding window**: at each step it holds:

- `current_array`: the master array for the row being convolved and saved
- `next_array`: the master array for the immediately following row

After convolution of `current_array`, the window advances — `next_array` becomes `current_array`, and a new empty `next_array` is prepared for the next row.

### Master Array

Each master array is a large 2-D float32 image tall enough for one cell height plus padding zones:

```
height = cell_height + 2 × PAD_SIZE    (PAD_SIZE = 480 px)
width  = PAD_SIZE + (N_cells × (cell_width - CELL_OVERLAP)) + CELL_OVERLAP + PAD_SIZE
```

Cells are placed at fixed x-offsets accounting for the `CELL_OVERLAP = 480` px between adjacent cells.

---

## 2. Pipeline Overview

```mermaid
flowchart TD
    subgraph startup [Startup]
        CSV["CSV metadata\n(pancakes_v2 output)"]
        ZarrIn["Zarr store\n(PS1 skycells)"]
        TaskBuild["Build interleaved\ntask list"]
        CSV --> TaskBuild
    end

    subgraph stage1 [Stage 1 - Ingest Threads xN (default 16)]
        TQ["task_queue\nFIFO"]
        RW1["ingest_worker 1"]
        RW2["ingest_worker N"]
        TQ --> RW1 & RW2
        ZarrIn --> RW1 & RW2
    end

    subgraph stage15 [Stage 1.5 - Band Combiner Threads x4]
        RCQ["raw_cell_queue\nmaxsize=max(6, 2xN)"]
        BC1["band_combiner_worker 1"]
        BC2["band_combiner_worker 2"]
        BC3["band_combiner_worker 3"]
        BC4["band_combiner_worker 4"]
        RCQ --> BC1 & BC2 & BC3 & BC4
    end

    subgraph stage2 [Stage 2 - Coordinator Thread]
        CRQ["combined_raw_queue\nmaxsize=12"]
        Coord["process_coordinator"]
        PPE["ProcessPoolExecutor\nN workers"]
        PSC["process_single_cell\nremove_background only"]
        BCCache["band_cache\npadding source results"]
        PendR["pending_results deque"]
        CRQ --> Coord
        Coord -->|"submit"| PPE
        PPE --> PSC
        PSC -->|"padding_source"| BCCache
        PSC -->|"regular"| PendR
        Coord -->|"cache_hit fast-path"| PendR
    end

    subgraph stage3 [Stage 3 - Main Thread]
        CCQ["combined_cell_queue\nmaxsize=30"]
        Gather["_gather_cells_for_row"]
        Assemble["assemble_row_from_bundles"]
        CRPad["apply_cross_row_padding"]
        CPPad["apply_cross_projection_padding\nreproject_interp"]
        Conv["Gaussian convolution"]
        Extract["extract_cell_results"]
        CCQ --> Gather --> Assemble --> CRPad --> CPPad --> Conv --> Extract
        BCCache --> CPPad
    end

    subgraph stage4 [Stage 4 - Saver Thread]
        RQ["results_queue\nmaxsize=3"]
        Saver["saver_worker"]
        ZarrOut["Zarr output store"]
        RQ --> Saver --> ZarrOut
    end

    TaskBuild -->|"all tasks upfront"| TQ
    RW1 & RW2 -->|"raw bundles ~1.6GB each"| RCQ
    BC1 & BC2 & BC3 & BC4 -->|"combined bundles ~0.4GB each"| CRQ
    PendR -->|"drain each iteration"| CCQ
    Extract -->|"bundle"| RQ
```

The five stages run concurrently:
- **Stage 1** (`num_ingest_workers` ingest threads, default 16): I/O-bound reads (zarr, or direct PS1 download in stream mode), produces raw bundles ~1.6 GB each
- **Stage 1.5** (4 band combiner threads): CPU-bound NumPy band combination, compresses 1.6 GB → ~0.4 GB per cell
- **Stage 2** (1 coordinator thread + N PPE processes): CPU-bound SEP source extraction only
- **Stage 3** (main thread): sequential sliding-window assembly, padding, convolution
- **Stage 4** (1 saver thread): writes results to output Zarr

### Why Five Stages?

Previously, Stages 1.5 and 2 were merged: raw 4-band bundles (~1.6 GB each) were passed directly from reader threads through `raw_cell_queue` into subprocess workers, where both band combination and source extraction ran together. With `maxsize=20` and 8 workers, up to ~48 GB of raw band data could be resident in memory at once, causing the Linux OOM killer to terminate the process.

The split separates two very different workloads:
- **Band combination** (`process_skycell_bands`) is fast (~4 s), NumPy-only, thread-safe, and produces a 4× smaller output. Running it in threads (no process fork, no CoW overhead) eliminates the per-process memory tax.
- **Source extraction** (`remove_background` via SEP) is slow (~60–120 s), benefits from subprocess isolation, and now receives a much lighter ~0.4 GB bundle instead of a 1.6 GB one.

---

## 3. Startup: Configuration and Task Scheduling

### CSV Metadata

`pancakes_v2.py` produces a CSV file that maps each TESS sector/camera/CCD to a set of PS1 skycells. Each row in the CSV contains:

| Column | Meaning |
|---|---|
| `projection` | PS1 projection ID |
| `NAME` | Full skycell name (e.g. `skycell.2556.082`) |
| `x` | Column position within the projection |
| `y` | Row position (row ID for the sliding window) |
| `NAXIS1/2` | Cell pixel dimensions |

The pipeline reads this CSV, groups cells by `(projection, y)`, and determines the processing order.

### Cross-Projection Padding Identification

Before any tasks are dispatched, `identify_all_padding_sources()` scans every projection and every row in the CSV to find which skycells from *other* projections are needed to fill the padding borders of each step's master arrays. This is a pure DataFrame operation — no I/O. It returns:

- `padding_sources`: `{skycell_name → source_projection}` — the full set of unique padding source cells
- `band_cache_uses`: `{skycell_name → use_count}` — how many times each cell will be needed (for cache eviction)
- `row_padding_map`: `{(projection, row_id) → set of skycell_names}` — which cells each step needs

### Per-Projection Canonical Classification (tiered ingest, count-based cut)

When `use_shared_convolved_store` is active (`combined_store_recipe` and
`convolved_store_recipe` both resolved), every projection is classified
**before** the interleaved task list is built — this decides whether the
projection needs the sliding-row machinery at all. See
`doc/ps1_process_tiered_ingest_architecture_plan.md` for the full design
rationale; summary:

1. For each projection, `convolved_store.classify_projection_missing_cells`
   does a cheap, read-only, per-cell fingerprint check (no pixel loads,
   reusing `skycell_already_canonical`) and returns the subset of cells
   that are **not** already published for the caller's exact recipe.
2. **Zero missing cells** → the projection is skipped entirely; its cells
   are added to `already_canonical_cells` and never enter the task list.
3. **`1 <= missing_count <= MISSING_CELL_THRESHOLD` (default 5)** →
   `process_sparse_projection` handles it directly, in-process, with no
   worker pipeline involvement at all (see below). If any required cell
   (the missing cell itself or one of its up-to-8 neighbors) turns out to
   be a genuine cold miss needing raw fetch + band-combine + SEP star
   removal, `process_sparse_projection` returns `None` and the whole
   projection falls back to the dense path instead — sparse publish is
   all-or-nothing per projection, decided before any partial state is
   written.
4. **`missing_count > MISSING_CELL_THRESHOLD`** (or classification is
   disabled/unavailable) → the projection is added to `dense_projections`
   and goes through the **unchanged** sliding-row task-list/worker
   pipeline described in the rest of this document. As of the initial
   rollout, dense projections process their *entire* row range exactly as
   before this change — the row-range-bounding refinement described in the
   architecture plan (start one row before the first missing row, stop
   after the last) is a documented future optimization, not yet
   implemented.

Only projections in `dense_projections` (not the full `projections` list)
proceed to the interleaved-task-list and `sequential_processor` steps
below. `produced_skycells` for the run is the union of
`sequential_processor`'s own result, `already_canonical_cells`, and
whatever `process_sparse_projection` published.

#### The per-skycell path (`process_sparse_projection` / `convolve_single_skycell`)

For each missing cell, `convolve_single_skycell` builds one small
`(cell_height + 2*radius) x (cell_width + 2*radius)` array — **not** a
full neighbor cell or a stitched multi-cell mosaic — with the cell's own
full pixel data in the center and only a `radius`-wide border strip (or
`radius × radius` corner block) from each of up to 8 same-row/adjacent-row
neighbors filling the padded margin. Any side with no neighbor (true grid
edge) stays `NaN`, matching `assemble_row_from_bundles`'s fill convention.
Because the Gaussian kernel is truncated at `radius` pixels
(`truncate = radius / sigma`), this reproduces byte-for-byte what the
existing whole-row/local-window convolution would have produced for that
cell. All required cells (center + neighbors, deduplicated across the
projection's missing cells) are fetched once via the same tier-1/2/3 lazy
lookup Part A uses, kept in a small dict bounded by roughly
`missing_count × 9`, then convolved and published
(`_publish_single_convolved_cell`, mirroring
`_publish_canonical_convolved_snapshot`'s fingerprint-chain logic) — no
`ingest_worker`/`band_combiner_worker`/`ProcessPoolExecutor`/`saver_worker`
involvement.

### Interleaved Task List

All tasks are placed into `task_queue` in a carefully computed order **before** the sequential processor starts. The order is designed so that padding source cells enter the pipeline immediately after the regular cells they accompany, giving the PPE workers maximum lead time before the main thread needs them.

The order for one projection with rows `[R0, R1, R2, R3]`:

```
[R0 regular cells]
[R1 regular cells]
[Padding cells for step 0 (uses R0 + R1)]   ← inserted here
[R2 regular cells]
[Padding cells for step 1 (uses R1 + R2)]   ← inserted here
[R3 regular cells]
[Padding cells for step 2 (uses R2 + R3)]   ← inserted here
[Padding cells for last step (R3, no next)] ← appended at end
```

Cells needed by multiple steps are only dispatched once (tracked by `already_dispatched_padding`). Subsequent steps find the cell already in `band_cache`.

After putting all tasks in `task_queue`, `None` signals are sent to terminate the reader threads. Readers are fully done — and may have exited — by the time the sequential processor starts.

---

## 4. Stage 1 — Ingest Workers

**`num_ingest_workers` `ingest_worker` threads** (default **16**; `reader_worker` remains as a backward-compatible alias) consume from `task_queue`. Each task is either a 4-tuple (regular cell) or a 5-tuple with task type (padding source):

```python
(skycell_id, projection, row_id, x_coord)               # regular
(skycell_id, projection, -1, 0, "padding_source")        # padding
```

For each task the ingest worker:

1. **Checks `band_cache`** (regular tasks only, tier 1 — already loaded this run): if already processed and cached, emits a lightweight `"regular_cache_hit"` bundle — no I/O.
2. **Tier 2 — shared combined-store lookup (regular tasks only)**: if `combined_store_data_root`/`combined_store_recipe` are set (shared-store campaigns), calls `seed_band_cache_from_combined_store()` for **just this one cell** (a single-item list). On a hit, the cell's `combined_image`/`combined_mask` are populated into `band_cache` and the same lightweight `regular_cache_hit` bundle is forwarded — no raw fetch, no band-combine, no SEP star removal for this cell.
3. **Loads the four bands** (only on a tier-1 and tier-2 miss) — in the default `ps1_source: "zarr"` mode via `load_skycell_bands_masks_and_headers()` (r, i, z, y band arrays, mask arrays, variance arrays, FITS header strings); in `ps1_source: "stream"` mode by downloading the skycell FITS files directly (no shared Zarr store required).
4. **Puts a `raw_bundle`** onto `raw_cell_queue` with all the loaded arrays plus task type metadata.

`raw_cell_queue` has `maxsize = max(6, 2 × num_ingest_workers)` (32 at the default worker count). The many ingest threads exist to hide network/disk latency; the queue bound provides the memory backstop.

**Tier 2 lookup is per-cell, not bulk.** An earlier version of this pipeline called `seed_band_cache_from_combined_store()` once, eagerly, for *every* regular cell in the SCC before any row processing began — for an SCC that shares most of its projections with already-processed neighboring sectors (high store-hit rate), this loaded hundreds of cells' full `combined_image`/`combined_mask` arrays (~248 MB/cell) into one unbounded dict at once, causing an OOM (measured: 865 cells × ~248 MB ≈ 214 GB, tripping a 300 GB Condor cgroup limit). The per-cell lookup above bounds tier-2 memory to at most `num_ingest_workers` cells in flight, matching the memory-safety envelope raw ingest already had. See §15 and `doc/ps1_process_tiered_ingest_architecture_plan.md` (Part A).

---

## 5. Stage 1.5 — Band Combiner Workers

**Four `band_combiner_worker` threads** consume raw bundles from `raw_cell_queue` and output reduced bundles to `combined_raw_queue`. This is the **compression stage** of the pipeline.

### What it does

For each raw bundle (carrying r, i, z, y band arrays, masks, variance maps, and header strings), the band combiner:

1. Calls `process_skycell_bands` — applies PS1 flux conversion per band and combines the four bands into a single `combined_image` (float32), `combined_mask` (uint16), and `combined_uncert` (float32).
2. Drops the raw band arrays (`del raw_bundle`) immediately after combination to free ~1.2 GB.
3. Puts a **reduced bundle** onto `combined_raw_queue` — only `combined_image`, `combined_mask`, `combined_uncert`, `headers_data`, and metadata.

### Memory impact

| Bundle type | Approximate size |
|---|---|
| `raw_bundle` (in `raw_cell_queue`) | ~1.6 GB (4 bands × data + mask + weight) |
| `reduced_bundle` (in `combined_raw_queue`) | ~0.4 GB (combined_image + mask + uncert) |

With `raw_cell_queue` maxsize `max(6, 2 × num_ingest_workers)` and `combined_raw_queue` maxsize 12, the theoretical in-queue peak at the default 16 ingest workers is `32 × 1.6 GB + 12 × 0.4 GB`; in practice ingest rarely keeps the raw queue full because band combination drains it in ~4 s per cell. The coordinator's memory-fraction guard (§16) is the hard backstop.

### Passthrough behaviour

Bundles without `bands_data` (cache-hit fast-path bundles) are forwarded to `combined_raw_queue` unchanged, since they carry no raw arrays.

### Thread count: why 4?

Band combination is CPU-bound NumPy work (~4 s per cell). Four threads keep `combined_raw_queue` well-fed without starving the readers or over-subscribing the CPU relative to the more expensive source extraction stage downstream.

### Shutdown

Each band combiner thread exits when it reads a `None` sentinel from `raw_cell_queue` and forwards one `None` to `combined_raw_queue`. The pipeline sends exactly one sentinel per thread (`num_band_combiners = 4`), so the coordinator receives its shutdown signal after the last combiner exits.

---

## 6. Stage 2 — Process Coordinator and ProcessPoolExecutor

The **coordinator thread** bridges `combined_raw_queue` (output of band combiners) with a `ProcessPoolExecutor` (N source-extractor workers). It runs a tight loop:

```mermaid
flowchart TD
    A["Drain pending_results\ninto combined_cell_queue\n(non-blocking put_nowait)"]
    B["Collect completed futures"]
    C{"task type?"}
    D["Store in band_cache\n(padding_source)"]
    E["Append to pending_results\n(regular)"]
    F["Capacity check\ntotal_pending < 30?"]
    Mem["Memory pressure check\navailable > 15% total?"]
    G["Pause check\npipeline_paused_event?"]
    H["Get bundle from\ncombined_raw_queue (0.1s timeout)"]
    I{"task_type?"}
    J["Cache-hit fast-path:\nbuild result from band_cache\nappend to pending_results"]
    K["Submit to PPE:\nprocess_single_cell"]

    A --> B --> C
    C -->|"padding_source"| D
    C -->|"regular / cache_hit"| E
    D & E --> F
    F -->|"at capacity: sleep 0.1s"| A
    F -->|"ok"| Mem
    Mem -->|"low memory: sleep 0.5s"| A
    Mem -->|"ok"| G
    G -->|"paused: sleep 0.1s"| A
    G -->|"ok"| H --> I
    I -->|"regular_cache_hit"| J --> A
    I -->|"regular or padding_source"| K --> A
```

### Non-blocking output (pending_results deque)

Rather than blocking on `combined_cell_queue.put()`, the coordinator appends to a local `deque`. At the top of every loop iteration it attempts `put_nowait()` into `combined_cell_queue` and stops if the queue is full. This ensures the coordinator thread can never freeze even when the main thread is occupied with multi-minute padding operations.

### Routing by task type

| Task type | What happens |
|---|---|
| `regular` | Submitted to PPE → `process_single_cell` → result appended to `pending_results` → flows to `combined_cell_queue` |
| `regular_cache_hit` | Result built directly from `band_cache` → appended to `pending_results` → flows to `combined_cell_queue`. Zero subprocess cost. |
| `padding_source` | Submitted to PPE → `process_single_cell` → result stored in `band_cache`. Never goes to `combined_cell_queue`. |

### Capacity throttling

`total_pending = len(active_tasks) + combined_cell_queue.qsize()`. When this exceeds `MAX_TOTAL_PENDING_WORK = 30`, the coordinator sleeps 0.1 s without fetching new work.

### Runtime memory pressure guard

Every iteration the coordinator checks `psutil.virtual_memory().available`. If available RAM drops below `MIN_AVAILABLE_MEMORY_FRACTION = 0.15` (15%) of total, it sleeps 0.5 s and does not submit new work. A rate-limited warning is logged at most once per 30 s. This acts as a last-resort brake against OOM kills when all other throttles are insufficient.

### Pause during cross-projection padding

When the main thread begins `apply_cross_projection_padding`, it sets a `pipeline_paused_event` (a `threading.Event`). The coordinator sees this flag, stops submitting new tasks to PPE, and sleeps 0.1 s per iteration. This frees PPE workers to finish their current tasks, giving the `reproject_interp` calls inside the padding threads more CPU bandwidth.

### PPE worker count — dynamic scaling

At startup, `num_source_extractors` is computed from available RAM and CPU count:

```python
mem_limit = max(2, int(available_gb // 4))   # each worker may dirty ~4 GB
num_source_extractors = max(2, min(ncpus // 2, mem_limit))
```

Previously workers received raw 4-band bundles and dirtied ~10 GB each (CoW fork); now they receive pre-combined ~0.4 GB bundles, so the per-worker memory budget is ~4 GB, roughly doubling the number of workers that can run safely on the same machine.

---

## 7. Cell Processing: process_single_cell

`process_single_cell` runs inside each PPE subprocess. It now receives a **pre-combined bundle** from `band_combiner_worker` — containing `combined_image`, `combined_mask`, and `combined_uncert` — and performs only the **source extraction** step.

### Background removal and source detection — `remove_background`

Uses [SEP](https://sep.readthedocs.io/) (Source Extractor as a Python library):

1. Identifies bright-star pixels: `data > nanmedian(uncert) × sigma_mask (50)`.
2. Calls `sep.extract()` with `sigma=2.5` to detect sources, producing a segmentation map.
3. Sets all non-source, non-bright-star pixels to zero (background suppression).
4. **If `remove_saturated_stars`** (`band_utils.remove_background`), two segment-removal passes run. The segmentation map is first extended into masked/gap pixels with a distance transform so catalog stars sitting under the bright-star mask still land on a segment.
   - **Primary (catalog) pass**: Gaia stars projected into the skycell (`project_gaia_to_skycell()`, using the mapping-stage Gaia catalog with `tess_mag` computed from G/BP/RP) are matched to SEP segments; every segment containing a star with `tess_mag < bright_star_mag_threshold` (default 13.0) is **zeroed in its entirety** (no interpolation). Both the bright star (`removal_reason="catalog_bright_star"`) and any fainter Gaia stars sharing the segment (`"catalog_neighbor"`) are recorded.
   - **Secondary (quality-flag) pass**: PS1 saturation mask bits (`0x0020` and `0x1000`) identify additional saturated segments not already removed; these are also zeroed, recorded as `"quality_flag_star"` (Gaia star found in the segment) or `"quality_flag_no_star"` (SEP-only, `source_id=-1`).

### Removed-star records

Each record already carries `ra`/`dec` from the Gaia catalog (no WCS back-projection needed); `skycell_id` is attached after the removal call. Full column set: `source_id`, `ra`, `dec`, `pixel_x`, `pixel_y`, `tess_mag`, `phot_g/bp/rp_mean_mag`, `seg_centroid_x/y`, `seg_flux`, `segment_id`, `removal_reason`, `skycell_id`. All records are accumulated and written to **`{convolved_zarr_path}_removed_stars.csv`** at the end of the run; `downsample` later dedups and projects them to crop-local coordinates as `events/{label}/ps1_removed_stars.csv`.

### Shared memory output

Rather than returning large NumPy arrays through the `ProcessPoolExecutor`'s internal pickle pipe (which can deadlock on pipe buffer overflow), `process_single_cell` writes `combined_image` and `combined_mask` to POSIX shared memory blocks and returns only lightweight descriptors:

```python
{"shm_name": str, "shape": tuple, "dtype": str}
```

The coordinator calls `_materialize_shm_result` to reconstruct the arrays in the coordinator's address space and immediately unlinks the shared memory block.

### Output bundle (after materialization in coordinator)

```python
{
    "skycell_id": str,
    "projection": str,
    "row_id": int,
    "x_coord": int,
    "combined_image": np.ndarray,   # float32, 4800×4800
    "combined_mask":  np.ndarray,   # uint16, 4800×4800
    "headers_data":   dict,         # band → FITS header string
    "removed_stars":  list[dict],   # one dict per removed star
}
```

---

## 8. Stage 3 — Sequential Assembler (Main Thread)

The main thread runs `sequential_processor`, which loops over projections and, within each projection, over row steps.

### Per-step logic in `process_row_step_from_queue`

```mermaid
flowchart TD
    A["Determine needed_padding_cells\nfrom row_padding_map"]
    B["Gather current row\n_gather_cells_for_row"]
    C["Assemble current_array\nassemble_row_from_bundles"]
    D["Optional: saturation correction\napply_saturation_to_row"]
    E["Gather next row\n_gather_cells_for_row"]
    F["Assemble next_array"]
    G["Optional: saturation correction"]
    H["Cross-row padding\napply_cross_row_padding"]
    I["Wait for padding cells\n_wait_for_padding_cells"]
    J["Cross-projection padding\napply_cross_projection_padding"]
    K["Gaussian convolution\nconvolution_utils"]
    L["Extract results\nextract_cell_results"]
    M["Queue results\nresults_queue.put"]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K --> L --> M
```

### _gather_cells_for_row

Waits for all cells of a given `(projection, row_id)` to arrive in `combined_cell_queue`. The cell buffer `cell_buffer[(projection, row_id)]` may already hold cells that arrived early (out-of-order arrivals from the coordinator).

**Timeout logic**: If no relevant cell arrives for `GATHER_TIMEOUT_SECONDS = 180` s, the function:
1. Does a non-blocking sweep of `combined_cell_queue` to recover any late arrivals.
2. Calls `_manually_process_cell` for each still-missing cell (see Section 12).

A "still gathering..." log message is emitted at most once per minute.

### assemble_row_from_bundles

Places each cell's `combined_image` into the master array at the correct x-offset. Cells overlap by `CELL_OVERLAP = 480` px; only the non-overlapping portion of each cell (past the `EDGE_EXCLUSION = 10` px guard zone) is placed. The PAD_SIZE padding areas at left, right, top, and bottom remain NaN until filled by padding steps.

### Advancing the window

After step `i` is complete, `advance_sliding_window` moves `next_array` → `current_array`, resets `next_array` to NaN, and updates all position tracking. Step `i+1` does not re-gather row `i+1` (it becomes the new current row that was the previous next row).

### Step 3b — Shared convolved-store canonical snapshot (per-cell skip)

When the caller opts into the shared, content-addressed convolved store (`use_shared_convolved_store: true`, i.e. `convolved_store_recipe`/`convolved_store_data_root`/`combined_store_recipe` are all set), `process_row_step_from_queue` calls `_publish_canonical_convolved_snapshot` immediately after `apply_cross_row_padding` and *before* `apply_cross_projection_padding` — the one point in the row's lifecycle where `state.current_array` still holds the same-projection-only master array that the canonical cell definition requires. It operates on an independent copy and never mutates `state.current_array`, so the unchanged step 5 full convolution/save path is unaffected either way.

This function is per-cell-skip-aware and unconditional (no config flag disables it):

1. **Pass 1 (cheap, read-only)**: for every cell currently placed in the row, resolve its `combined_fingerprint` the same deterministic way `combined_store.py` does, and check `convolved_store.skycell_already_canonical` — this recomputes the resolved `convolved_fingerprint` on top of that `combined_fingerprint` and confirms an actually-published `convolved_skycell` payload exists for the caller's exact recipe chain (band weights, `apply_flux_conv`, saturated-star removal params, `gaia_version`, `psf_sigma`/`radius`/`mode` all baked in). No convolution happens in this pass.
2. **If every cell in the row is already canonical**, the function returns immediately — zero convolution calls for that row.
3. **Otherwise**, only the non-canonical cells are (re)published:
   - If the missing fraction is below `ROW_FALLBACK_THRESHOLD = 0.2`, `_convolve_local_windows_for_missing_cells` convolves merged, `±radius`-padded local windows around just those cells (adjacent missing cells whose windows would overlap are merged into one convolution call — "contiguous-run batching"). Because the Gaussian kernel is truncated at `radius` pixels, this reproduces byte-for-byte what convolving the whole row and cropping would have produced.
   - Otherwise (≥20% of the row missing), `_convolve_whole_row_snapshot` falls back to the pre-optimization behavior of convolving the entire row once — at that missing fraction, per-cell call/I-O overhead outweighs the FLOPs saved by skipping already-canonical cells.

Cells with no actually-published `combined_skycell` record on disk (e.g. a dual-role cell only ever cached as a cross-projection padding source) are skipped for this publish rather than fabricating a fingerprint edge to nothing.

---

## 8. Cross-Row Padding

Before cross-projection padding and convolution, `apply_cross_row_padding` fills the top/bottom padding zones of each master array using data from the other:

```
current_array[bottom_pad] ← next_array[overlap_source]
next_array[top_pad]       ← current_array[overlap_source]
```

The `CELL_OVERLAP = 480` px between adjacent rows means there is real shared sky data to copy. The `EDGE_EXCLUSION = 10` px prevents copying the very edge pixels that may have artifacts. This step is pure in-memory numpy slicing — no I/O.

---

## 9. Cross-Projection Padding

Adjacent PS1 projections have different WCS tangent points, so their cells cannot be directly overlaid. When a TESS pixel falls near the boundary of projection A, it may need data from a skycell belonging to projection B. This is **cross-projection padding**.

The CSV produced by `pancakes_v2.py` encodes where each boundary cell needs padding from and which source skycell in an adjacent projection provides it.

### apply_cross_projection_padding

Called once per row step (for both current and next row simultaneously). The full flow:

```mermaid
flowchart TD
    A["Parse padding requirements\nparse_row_padding_requirements x2"]
    B["Deduplicate into PaddingJobs\nanalyze_padding_jobs"]
    C["Create master WCS\ncreate_master_array_wcs"]
    D["ThreadPoolExecutor 8 threads\none thread per PaddingJob"]
    E["_process_padding_job\nper thread"]
    F["Check band_cache\ncombined_image available?"]
    G["Cache HIT:\nuse cached data directly"]
    H["Cache MISS:\nzarr load + process_skycell_bands\n+ remove_background\n(WARNING: should not happen)"]
    I["exclude_edge_pixels\nmask 10px border"]
    J["create_cell_wcs\nsource WCS from header"]
    K["create_padding_wcs\nlocalized target WCS"]
    L["reproject_interp\nbilinear interpolation"]
    M["Stitch into target array\nthread-safe write_lock"]

    A --> B --> C --> D --> E
    E --> F
    F -->|"HIT"| G
    F -->|"MISS"| H
    G & H --> I --> J --> K --> L --> M
```

### PaddingJob deduplication

A single source skycell may be needed by both `current_array` and `next_array`. `analyze_padding_jobs` creates one `PaddingJob` per unique source cell with a `targets` list so each cell is loaded only once per padding call, even if it contributes to multiple arrays.

### reproject_interp

This is the most expensive operation. It uses `astropy`'s `reproject_interp` to resample the source skycell (in its native WCS) into the localized WCS of each target padding location. The GIL is released during the C-level interpolation, so multiple padding threads make real parallel progress.

To give `reproject_interp` full CPU access, the coordinator pauses new PPE submissions (`pipeline_paused_event.set()`) for the duration of this call.

---

## 10. Band Cache and Padding Source Pre-loading

The `band_cache` is a shared dict used to avoid redundant processing of cells that serve dual roles.

```
band_cache[skycell_name] = {
    "combined_image": np.ndarray,   # fully processed float32 image
    "combined_mask":  np.ndarray,
    "headers_data":   dict,
    "removed_stars":  list,
}
```

### How padding sources enter the cache

At startup, all padding source cells are identified and inserted into the interleaved task list as `"padding_source"` tasks. The reader loads them from zarr; the band combiner combines the raw bands; the coordinator submits them through `process_single_cell` (background removal + optional star removal); and the coordinator routes the result to `band_cache` instead of `combined_cell_queue`.

### How the cache is used in _process_padding_job

When `_process_padding_job` runs, it first checks `band_cache[job.skycell_name]`. On a cache hit it copies the `combined_image` directly and proceeds to `exclude_edge_pixels` + `reproject_interp` — skipping zarr I/O and all band processing entirely.

### Dual-role cells

A cell like `skycell.2556.082` might be:
- A **padding source** for projection `2557`
- Also a **regular cell** in projection `2556`'s own rows

When the coordinator first processes it as `"padding_source"`, the result goes to `band_cache`. When the reader later sees the regular task for the same cell, it detects the cache hit and emits a `"regular_cache_hit"` bundle. The coordinator builds the result bundle directly from cache — zero subprocess overhead, no zarr load, no band processing.

`band_cache_uses` tracks how many times each cell will be needed. After each row step, `_evict_band_cache_for_step` decrements the use counter for all cells consumed in that step and deletes entries whose count reaches zero, keeping memory bounded.

### _wait_for_padding_cells

Before calling `apply_cross_projection_padding`, the main thread calls `_wait_for_padding_cells`:

1. Computes `remaining = needed_cells - already_in_cache`.
2. If empty: proceeds immediately.
3. Otherwise: polls `band_cache` every 0.5 s.
4. Logs "still waiting..." at most once per minute with elapsed time and timeout remaining.
5. If `timeout = 180` s passes since the last cell arrived: triggers `_manually_process_cell` for each still-missing cell (places result directly in `band_cache`).

---

## 11. Manual Fallback Loader

`_manually_process_cell` is the shared fallback used when the pipeline has not delivered a result within the timeout. It runs directly in the main thread:

```python
zarr.open() → load_skycell_bands_masks_and_headers()
            → process_skycell_bands()
            → remove_background()          # identical to pipeline path
            → RA/Dec enrichment of star records
```

Both fallback paths use this single function:

- `_gather_cells_for_row` calls it for missing **regular cells**, wraps the result in a full bundle dict.
- `_wait_for_padding_cells` calls it for missing **padding cells**, places the result directly in `band_cache`.

**Why not push to the reader queue instead?** By the time a timeout fires in `sequential_processor`, the reader threads have already exited (they exit after consuming their `None` shutdown signals, which are sent before `sequential_processor` starts). Executing directly in the main thread is the only reliable path. It completes in a known amount of time (typically 30–90 s per cell) without depending on pipeline capacity.

---

## 13. Stage 4 — Saver Worker

`saver_worker` runs as a **daemon thread**. It receives `processed_bundle` dicts from `results_queue` and calls `zarr_utils.save_convolved_results()` to write into the output Zarr store as **flat root-level arrays named `{skycell_name}_data`** (float32, NaN fill) **and `{skycell_name}_mask`** (uint16, the *pre-convolution* `combined_mask`), zstd-compressed with ≤1024² chunks. Weights are not written in the standard flow.

`results_queue` has `maxsize=3`, which provides backpressure — if the saver is slow, the main thread will block on `results_queue.put()`, naturally throttling the pipeline.

---

## 14. Saturation Correction

When `--enable-saturation-correction` is passed (and `--remove-saturated-stars` is not), a Gaia star catalog is loaded. After assembling each row into the master array, `apply_saturation_to_row` uses the Gaia RA/Dec coordinates of bright stars to correct saturated pixels in the master array. This step runs after each row is assembled, before padding and convolution.

---

## 15. Memory Management and OOM Prevention

Previous versions of the pipeline were vulnerable to the Linux OOM killer terminating the process when too many large raw bundles accumulated in queues. The current architecture employs a layered defence:

### Layer 1: Queue size limits on raw data

`raw_cell_queue` is capped at `maxsize = max(6, 2 × num_ingest_workers)` (32 at the default 16 ingest workers). With ~1.6 GB per raw bundle, this bounds raw-bundle queue memory.

### Layer 2: Band compression before subprocesses

`band_combiner_worker` threads consume raw bundles and output ~0.4 GB combined bundles to `combined_raw_queue` (maxsize=12), so subprocesses never see raw 4-band bundles. Previously raw bundles went straight to subprocess workers (`20 × 1.6 GB = ~32 GB` in-queue plus subprocess memory).

### Layer 3: Runtime memory pressure guard

The coordinator polls `psutil.virtual_memory().available` every iteration. If available RAM falls below 15% of total (`MIN_AVAILABLE_MEMORY_FRACTION = 0.15`), no new tasks are submitted to the PPE until memory recovers.

### Layer 4: PPE capacity cap

`MAX_TOTAL_PENDING_WORK = 30` limits the combined count of in-flight PPE tasks and buffered results in `combined_cell_queue`, preventing the subprocess pool from holding unbounded processed results.

### Layer 5: Process cleanup on exit

`atexit.register(_cleanup_child_processes)` and a `try/finally` block in the pipeline orchestrator ensure that PPE worker processes are forcefully terminated if the main process exits for any reason (including an OOM kill of a subprocess), preventing orphaned worker processes from accumulating across restarts. `_cleanup_child_processes` reads `getattr(_active_executor, "_processes", None) or {}` — the executor's `_processes` attribute can be present but `None` post-shutdown, not merely absent, so a bare `getattr(..., {})` default is insufficient.

### Layer 6: Per-cell (not bulk) shared combined-store lookup

Fixed 2026-08-20: `seed_band_cache_from_combined_store()` used to be called once, eagerly, for every regular skycell in the SCC (shared-store campaigns only) before any row processing started — for a re-run sharing most projections with already-processed sectors, this loaded hundreds of cells' full arrays (~248 MB/cell) into one unbounded dict at once (865 cells × ~248 MB ≈ 214 GB in the incident that surfaced this, tripping a 300 GB Condor cgroup limit). It is now called per-cell, lazily, inside `ingest_worker` on a tier-1 (`band_cache`) miss (§4) — bounded by `num_ingest_workers` cells in flight, same envelope as raw ingest. See `doc/ps1_process_tiered_ingest_architecture_plan.md`.

---

## 16. Concurrency Control and Throttling

| Mechanism | Effect |
|---|---|
| `raw_cell_queue maxsize=max(6, 2×ingest)` | Backpressure on ingest workers — bounds raw in-flight memory |
| `combined_raw_queue maxsize=12` | Backpressure on band combiners — caps combined in-flight memory at ~4.8 GB |
| `combined_cell_queue maxsize=30` | Backpressure on coordinator — prevents SEP results from filling RAM |
| `results_queue maxsize=3` | Backpressure on main thread — prevents convolved results piling up if saver is slow |
| `MAX_TOTAL_PENDING_WORK=30` | Coordinator stops submitting new PPE tasks when `active + queue_size >= 30` |
| `MIN_AVAILABLE_MEMORY_FRACTION=0.15` | Coordinator pauses submissions when available RAM < 15% of total |
| `pipeline_paused_event` | Main thread signals coordinator to pause during `reproject_interp` |
| `pending_results deque` | Non-blocking buffer between PPE results and `combined_cell_queue`; prevents coordinator freeze |
| `band_cache_uses` eviction | Evicts fully-used entries after each row step; bounds peak cache memory |

---

## 17. Queue and Buffer Reference

| Name | Type | maxsize | Direction | Purpose |
|---|---|---|---|---|
| `task_queue` | `queue.Queue` | unbounded | Startup → Ingest workers | Ordered task tuples (cell name, projection, row, x, type) |
| `raw_cell_queue` | `queue.Queue` | max(6, 2×ingest) | Ingest workers → Band Combiners | Raw band bundles (~1.6 GB each) |
| `combined_raw_queue` | `queue.Queue` | 12 | Band Combiners → Coordinator | Pre-combined bundles (~0.4 GB each) |
| `combined_cell_queue` | `queue.Queue` | 30 | Coordinator → Main thread | Fully processed cell bundles (SEP complete) |
| `results_queue` | `queue.Queue` | 3 | Main thread → Saver | Convolved row results |
| `cell_buffer` | `dict[(proj, row_id) → list]` | — | In-memory | Out-of-order cell bundles waiting for their row |
| `band_cache` | `dict[skycell_name → dict]` | — | In-memory | Fully processed images for padding and dual-role reuse |
| `pending_results` | `deque` | — | Coordinator-internal | Non-blocking staging area before `combined_cell_queue` |

All queues use `queue.Queue` (threading) rather than `multiprocessing.Queue`. This avoids POSIX semaphore corruption that occurs when `ProcessPoolExecutor` workers inherit pipe file descriptors from a `multiprocessing.Queue` created before the fork.

---

## 18. Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `CELL_OVERLAP` | 480 px | Pixel overlap between adjacent cells in a row |
| `EDGE_EXCLUSION` | 10 px | Edge guard zone masked during padding |
| `EFFECTIVE_OVERLAP` | 470 px | `CELL_OVERLAP - EDGE_EXCLUSION` |
| `PAD_SIZE` | 480 px | Height of top/bottom/left/right padding zones in master array |
| `GATHER_TIMEOUT_SECONDS` | 180 s | Max wait since last cell arrival before manual fallback |
| `MAX_TOTAL_PENDING_WORK` | 30 | PPE active tasks + combined queue size cap |
| `MIN_AVAILABLE_MEMORY_FRACTION` | 0.15 | RAM fraction below which coordinator pauses submissions |
| `num_ingest_workers` | 16 (configurable) | Ingest thread count (zarr or stream mode) |
| `num_band_combiners` | 4 | Band combiner thread count (fixed) |
| `num_source_extractors` | dynamic | SEP subprocess count: `min(ncpus // 2, available_gb // 4)` |
| `band weights` | r=0.238, i=0.344, z=0.283, y=0.135 | PS1 band combination weights |
| `ROW_FALLBACK_THRESHOLD` | 0.2 | Fraction of non-canonical cells in a row at/above which step 3b falls back to whole-row convolution instead of local windows |

---

## 19. Running the Pipeline

```bash
python process_ps1.py SECTOR CAMERA CCD [OPTIONS]
```

| Argument | Description |
|---|---|
| `sector` | TESS sector number |
| `camera` | TESS camera number (1–4) |
| `ccd` | TESS CCD number (1–4) |
| `--data-root DIR` | Root directory for data (default: `data`) |
| `--limit N` | Process only the first N projections (useful for testing) |
| `--psf-sigma FLOAT` | Gaussian PSF sigma in pixels (default: **60.0**; the convolution kernel is truncated at radius 470 px in `convolution_utils`) |
| `--enable-saturation-correction` | Apply Gaia-catalog-based saturation correction (**orchestrated runs default this to `true`**) |
| `--remove-saturated-stars` | Zero out bright/saturated star segments and record them to `{convolved_zarr}_removed_stars.csv` (see §7); uses the Gaia catalog for the primary pass |
| `--catalog-path PATH` | Override default Gaia catalog path (catalog is used by both saturation correction and star removal) |
| `--ps1-source {zarr,stream}` | `zarr` (default) reads the shared PS1 Zarr; `stream` downloads skycell FITS on the fly (no shared store needed) |

**Expected directory structure:**

```
data/
  ps1_skycells_zarr/ps1_skycells.zarr   ← input
  convolved_results/                    ← output (created automatically)
  catalogs/sector_NNNN/camera_N/ccd_N/  ← Gaia catalog (if saturation correction)
  <CSV file discovered via find_csv_file>
```

**Typical run (sector 20, camera 3, CCD 3, remove saturated stars):**

```bash
python process_ps1.py 20 3 3 --remove-saturated-stars 2>&1 | tee log/comb_20_3_3_sat_rm.log
```

---

## 20. Log Prefixes Quick Reference

| Prefix | Stage | Meaning |
|---|---|---|
| `[Pipeline]` | Startup | Initialization, task dispatch, shutdown |
| `[Reader]` | Stage 1 | Zarr loads, cache hits |
| `[BandCombiner]` | Stage 1.5 | Band combination per cell (thread) |
| `[ProcessCoordinator]` | Stage 2 | Capacity, memory pressure, routing, PPE submissions |
| `[PreProcessor]` | Stage 2/PPE | Source extraction (SEP) per cell |
| `[Gather]` | Stage 3 | Waiting for cells, buffer hits, queue sweeps |
| `[ManualLoader]` | Stage 3 fallback | Synchronous zarr load when pipeline times out |
| `[Assembler]` | Stage 3 | Master array cell placement |
| `[SequentialProcessor]` | Stage 3 | Per-row step timing and progress |
| `[PaddingGather]` | Stage 3 | Waiting for padding cells in band_cache |
| `[CrossPadding]` | Stage 3 | reproject_interp job progress per cell |
| `[Saver]` | Stage 4 | Output zarr writes |
| `[Band]` | band_utils | Flux conversion, mask combination |

---

## Orchestrator integration

In the supervised pipeline, convolution runs as the `ps1_process` stage (default executor: HTCondor). The worker reads the shared PS1 Zarr at `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr` and writes `{data_root}/convolved_results/sector_{SSSS}_camera_{C}_ccd_{K}.zarr`. See the [template pipeline guide](../template_pipeline.md) for Condor wrappers, memory pools, and skipping when convolved Zarr already verifies on disk.
