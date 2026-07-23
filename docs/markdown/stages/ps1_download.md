> **Package integration**: `syndiff` stage `ps1_download` · module `template_creation/processing/ps1_download.py` · legacy script `download_and_store_zarr.py`  
> **Related docs**: [template pipeline guide](../template_pipeline.md) · [PS1 process (technical)](ps1_process_technical.md) · [storage layout](../storage_layout.md) · [PanCAKES mapping](mapping_pancakes.md)

# PS1 skycell download (`ps1_download`)

Downloads Pan-STARRS 1 (PS1) **rings v3** skycell FITS from MAST, decompresses in memory, and writes directly into one **shared Zarr store** under `data_root`. The stage is network-bound and SCC-scoped: each run reads the skycell list for one sector/camera/CCD (SCC) from the mapping stage, but all SCCs on the same `data_root` share the same Zarr file.

---

## Role in the template DAG

```text
tess_ffi_download → wcs_grouping → mapping → ps1_download → ps1_process → downsample → diff
```

| Property | Value |
|----------|-------|
| Scheduler stage | `ps1_download` |
| Depends on | `mapping` (skycell list CSV) |
| Downstream | `ps1_process` reads the shared Zarr (`ps1_source: zarr`, default) |
| Executor | Network pool (Condor by default) |
| Scope | Per SCC run; **shared** Zarr across all SCCs on one `data_root` |

### Skipped when `ps1_source: stream`

When `pipeline.yaml` sets `ps1_process.ps1_source: stream`, `ps1_process` fetches skycells on demand via `fetch_skycell_bands_masks_and_headers()` and **does not depend on** `ps1_download`. The scheduler drops `ps1_download` from the effective dependency chain (`stages.py`: `ps1_process` deps become `("mapping",)` only). Verify for `ps1_download` is not required on stream-ingest pipelines.

Use **stream** for pilots that avoid populating the shared Zarr; use **zarr** (default) when many SCCs reuse the same skycells and you want download-once, read-many behavior.

---

## Canonical paths (`scc_paths`)

Path helpers live in [`syndiff_pipeline/common/scc_paths.py`](../../../syndiff_pipeline/common/scc_paths.py):

| Helper | Resolves to |
|--------|-------------|
| `ps1_skycells_zarr_dir(data_root)` | `{data_root}/ps1_skycells_zarr/` |
| `ps1_skycells_zarr_path(data_root)` | `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr` |
| `ps1_skycells_zarr_lock_path(data_root)` | `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr.lock` |

The pipeline passes `zarr_output_dir=resolved.zarr_dir`, which defaults to `ps1_skycells_zarr_dir(data_root)` — not a per-SCC subdirectory.

### Zarr hierarchy

```text
{data_root}/ps1_skycells_zarr/
  ps1_skycells.zarr/
    {projection_id}/
      skycell.{projection}.{cell}/
        r, r_mask, r_wt
        i, i_mask, i_wt
        z, z_mask, z_wt
        y, y_mask, y_wt
  ps1_skycells.zarr.lock
```

Each skycell has **12 arrays** (four bands `r,i,z,y` × image, mask, weight). FITS headers are stored in each array's Zarr attrs (`header` key). Compression: zstd level 3; chunk size up to 1024×1024.

---

## Inputs: mapping CSV

`download_and_store_ps1_data()` resolves the skycell list via `scc_mapping_master_skycells_csv()` (SCC layout), with fallback to legacy:

```text
{data_root}/skycell_pixel_mapping/sector_{SSSS}/camera_{C}/ccd_{K}/
  tess_s{sector}_{camera}_{ccd}_master_skycells_list.csv
```

From the `NAME` column it collects unique skycells, then merges **padding skycells** from `csv_utils.get_all_padding_cells()` (same helper mapping uses for cross-projection edges). If padding discovery fails, the stage logs a warning and continues with main skycells only.

---

## Algorithm summary

1. **Initialize** the Zarr store (`initialize_zarr_store`) under a file lock.
2. **Dask bag** over skycell names (`process_skycells_with_dask`, threaded scheduler).
3. Per skycell (`download_and_store_skycell`):
   - Skip if all 12 arrays are complete (`skycell_array_status` / `is_array_complete`).
   - Download missing band/mask/weight triples in parallel (HTTP from `ps1images.stsci.edu`, or local FITS when `use_local_files=True`).
   - Batch writes through a background `ZarrWriter` queue (serialized under the same lock).
4. Return a completion dict with `produced_skycells`, `zarr_path`, and per-skycell Zarr group paths.

`fetch_skycell_bands_masks_and_headers()` exposes the same download/decompress path **without** writing Zarr — used by `ps1_process` stream ingest and the star branch cache miss path.

---

## File lock and concurrent writers

All Zarr initialization and writes acquire `FileLock` on `ps1_skycells.zarr.lock`. Parallel Dask workers may download concurrently, but **writes are serialized** through `ZarrWriter` + lock. Multiple `ps1_download` jobs on the same `data_root` therefore queue at write time; a stuck lock stalls every waiter. Tune `resources.network.max_concurrent` in `pipeline.yaml` accordingly (see [template pipeline guide](../template_pipeline.md)).

---

## Resume, idempotency, and verify

| Behavior | Mechanism |
|----------|-----------|
| **Resume** | Per-array completeness check before download; partial skycells fill in missing arrays only |
| **Corruption** | `_array_complete_unlocked` probes `[0:1,0:1]`; corrupted arrays are re-downloaded |
| **Overwrite** | Stage param `overwrite: true` (CLI `--overwrite`) forces re-fetch |
| **Verify** | `verify_ps1_download`: every expected skycell from mapping must have all 12 complete arrays |

Re-running `ps1_download` on an interrupted SCC is safe: completed skycells are skipped, incomplete ones resume.

---

## Stage configuration (`pipeline.yaml`)

Typical keys under `stages.ps1_download`:

| Key | Role |
|-----|------|
| `num_workers` | Dask thread workers (default in module CLI: 32; dispatch uses stage param) |
| `use_local_files` | Prefer pre-downloaded FITS tree (`local_data_path`) |
| `local_data_path` | Root for `rings.v3.skycell.{proj}.{cell}.stk.{band}.unconv.fits` layout |
| `log_level` | `ERROR` default in Python API; CLI default `WARNING` |
| `overwrite` | Re-download existing arrays |

---

## Standalone CLI

```bash
mamba activate syndiff

python -m syndiff_pipeline.template_creation.processing.ps1_download \
  20 3 3 \
  --zarr-output-dir /path/to/data/ps1_skycells_zarr \
  --num-workers 32
```

Positional args: `sector`, `camera`, `ccd`. Optional: `--use-local-files`, `--local-data-path`, `--overwrite`, `--log-level`.

---

## Relation to the star branch

Host-star light curves (`star`) may read the same raw-band store when `ps1_source: zarr_download` or `zarr_local_only` is set. [`star/ps1_cache.py`](../../../syndiff_pipeline/star/ps1_cache.py) resolves the default path with `ps1_skycells_zarr_path(data_root)` — the **same** layout as `ps1_download`.

Older deployments sometimes kept a top-level `{data_root}/ps1_skycells.zarr` (without the `ps1_skycells_zarr/` wrapper). If your cache lives there, set `ps1_zarr_path` in [`star_config.yaml`](star_config.md) so star and template stages agree. See [star pipeline (technical)](star_pipeline.md) and [storage layout](../storage_layout.md#ps1-raw-skycells-zarr).

---

## Key source files

| File | Responsibility |
|------|----------------|
| [`ps1_download.py`](../../../syndiff_pipeline/template_creation/processing/ps1_download.py) | Download, Zarr write, `download_and_store_ps1_data` |
| [`scc_paths.py`](../../../syndiff_pipeline/common/scc_paths.py) | Canonical Zarr and lock paths |
| [`dispatch.py`](../../../syndiff_pipeline/template_creation/orchestration/dispatch.py) | Stage runner |
| [`verify.py`](../../../syndiff_pipeline/template_creation/orchestration/verify.py) | `verify_ps1_download` |
| [`stages.py`](../../../syndiff_pipeline/template_creation/orchestration/stages.py) | `ps1_process` deps when `ps1_source: stream` |
