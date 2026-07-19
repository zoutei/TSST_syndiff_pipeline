# PR2/PR4/PR5 data-flow map — PS1 processing seams

Reference for the publish wrapper (PR2), shared combined store (PR4), and shared
convolved store + padding decouple (PR5). Anchors are `file:line` in
`syndiff_pipeline/template_creation/processing/`. Verify before editing.

## Critical corrections discovered (update the plan's assumptions)
1. **Convolved output path is NOT the SCC `convolved.zarr` in code.** The pipeline
   writes `data_root/convolved_results/sector_{SSSS}_camera_{C}_ccd_{K}.zarr`
   (`ps1_process.py:1587`). (The SCC migration moved files on disk; confirm how the
   live write path maps to `scc_convolved_zarr` before PR2.)
2. **Writes are FLAT and non-atomic.** `save_convolved_results` (`zarr_utils.py:78-145`)
   writes arrays keyed `{skycell_name}_data/_mask/_weight` at the **store root** (no
   projection/row nesting), via delete-then-`create_array` (`:103-107,121-124,140-143`).
   `projection`/`row_id` are used only for the log line. The PR2 `publish_skycell`
   wrapper must change BOTH the key structure (→ `{projection}/{skycell}/{fp}`) and add
   atomicity (tmp key → rename).
3. **Star removal is DOWNSTREAM of the band combiner.** `band_combiner_worker`
   (`ps1_process.py:588-638`) produces the pre-star-removal combined image; star removal
   runs later in the subprocess `process_single_cell` (`:436-443`). The shareable
   "combined + star-removed" product first exists at the coordinator result
   (`ps1_process.py:711-722`) — that is the PR4 publish seam, not the combiner output.

## Pipeline stages / queues / workers (`ps1_process.py`)
- Constants: `CELL_OVERLAP=480, EDGE_EXCLUSION=10, EFFECTIVE_OVERLAP=470, PAD_SIZE=480` (`:83-86`; dup in `cross_projection_padding.py:30-32`).
- Queues built in `run_modern_sliding_window_pipeline` (`:1645-1656`); all in-process `queue.Queue` (the only true process boundary is the `ProcessPoolExecutor` in `process_coordinator`). SHM descriptors cross it: `_array_to_shm` (`:312-321`) / `_shm_to_array` (`:324-330`).
- **Stage 1 `ingest_worker`** (`:475-547`): loads raw bands (zarr `:523` / stream `:517`); in-run `band_cache` hit → `regular_cache_hit` passthrough (`:503-513`). Output → `raw_cell_queue`.
- **Stage 1.5 `band_combiner_worker`** (`:588-638`): `process_skycell_bands(...)` (`:613-619`) → `(combined_image, combined_mask, combined_uncert)`; the ~1.6 GB→~0.4 GB reduction. Output → `combined_raw_queue`.
- **Stage 2 `process_coordinator`** (`:651-838`): one thread owns `ProcessPoolExecutor` (`:690`); submits `process_single_cell` (`:804`); injects Gaia (`:800-802`). `process_single_cell` (`:387-472`, subprocess): `project_gaia_to_skycell` (`:424`) → `remove_background` (`:436-443`) → SHM (`:450-451`). Regular results → `combined_cell_queue`; padding-source → `band_cache` (`:713-720`). **PR4 publish seam: `:711-722`.**
- **Stage 3 assembler** (main thread): `sequential_processor` (`:1421-1547`) → `process_row_step_from_queue` (`:1262-1418`): `_gather_cells_for_row` (`:987-1117`), `assemble_row_from_bundles` (`:862-904`, PAD_SIZE offsets), `apply_cross_row_padding` (`:193-211`), cross-projection wait+pad (`:1122-1221`, `:1390-1394`), **convolution `:1406-1412`**, `extract_cell_results` (`:1415`) → `results_queue`.
- **Stage 4 `saver_worker`** (`:841-856`): `zarr.open(mode="a")` (`:844`), `save_convolved_results(...)` (`:851`). **PR2 write seam.**
- Orchestrator `run_modern_sliding_window_pipeline` (`:1553-1858`): NO `overwrite` param; output path `:1587`; task-list build+dispatch `:1738-1787`; returns `produced_skycells`/`expected_skycells`/`artifacts` (`:1844-1852`).

## Combined product purity (PR4)
`process_skycell_bands` (`band_utils.py:251-275`) → float32 `combined_image`/`combined_uncert`
(`combine_rizy_bands` `:150-214`, weights `[0.238,0.344,0.283,0.135]` `:163`, flux via header
`BOFFSET/BSOFTEN/EXPTIME` `:188-193`), uint16 `combined_mask` (`combine_masks` `:217-248`).
Footprint filter (the shareability proof): `project_gaia_to_skycell` `ps1_process.py:365-369`
(`in_footprint` mask → `gaia_catalog[in_footprint]`). Per-SCC Gaia load `load_gaia_catalog`
(`:98-128`, path `data/catalogs/sector_.../gaia_catalog_*.csv`).
**combined_fp recipe params:** band weights + flux constants + `sigma=2.5`/`sigma_mask=50`
(`band_utils.py:382-383`) + `remove_saturated_stars` + `bright_star_mag_threshold` + `gaia_version`.
**Inputs:** `raw_skycell` fp + `source_catalog` fp.
Load path: mirror the `band_cache` hit in `ingest_worker` (`:503-513`) or check before task
enqueue (`:1751-1773`); on hit inject as a `pending_results`-style bundle bypassing stages 1/1.5/2.

## Convolution + padding (PR5)
`apply_gaussian_convolution` (`convolution_utils.py:14-46`): `sigma` (pipeline passes `psf_sigma`),
`radius=470`, `truncate=radius/sigma`, `mode="constant"`, `cval=np.nan`. Runs on the fully padded
master row array (`:1410`), then cells extracted.
Padding, BOTH before convolution:
- same-projection cross-row `apply_cross_row_padding` (`ps1_process.py:193-211`) + in-row overlap
  (`assemble_row_from_bundles` `:862-904`) — **canonical / sector-independent.**
- cross-projection `apply_cross_projection_padding` (`cross_projection_padding.py:547-610`;
  `_process_padding_job` `:437-544`: load → `exclude_edge_pixels` `:480` → `reproject_interp` `:516`
  → stitch under lock `:531-536`) — **SCC-specific.**
**§13 decouple:** drop the cross-projection call (`ps1_process.py:1369-1404`) from the
pre-convolution path, keep same-projection (`:1364`); publish the canonical convolved cell
sky-keyed (replace `zarr_utils.py:93-143`); move `_process_padding_job` reproject/stitch
(`cross_projection_padding.py:511-536`) into a post-convolution `scc_assembly` drawing from shared
convolved cells. **convolved_fp recipe:** `sigma`/`radius` + same-projection pad geometry consts;
inputs: `combined_fp`.

## Skip / overwrite today (PR4 per-cell skip)
`band_cache` is per-invocation only (`:1661`), never persisted — NO cross-run dedup. No `overwrite`
param; `force_rerun` → `clear_ps1_process_artifacts` rmtrees the whole store (`verify.py:845-865`,
dispatch `:270-296`). Completeness answered only by post-hoc scan (`_count_convolved_data_arrays`
`verify.py:918-932`). **Per-fingerprint skip goes in the task-list build loop
`ps1_process.py:1745-1773`** (before `master_task_list.append` `:1753`), using
`expected_convolved_skycells` (`:284-306`) as the planned set.

## Numeric-equivalence gate (PR5, blocking)
Extend `tests/test_convolution_utils.py` (flux-conservation asserts, `sigma=60,radius=470`) and
`tests/test_downsample_binning_golden.py` (golden loops, `assert_allclose` atol 1e-6/0). Assemble a
real SCC from shared+re-padded convolved cells and `assert_allclose` vs current baked-in convolved
store. No dev notebooks exist. Convolved path helper `scc_convolved_zarr` (`common/scc_paths.py`),
live write `ps1_process.py:1587`.
