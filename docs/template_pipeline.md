# SynDiff PS1 Template Pipeline (`syndiff-template`)

This document describes the **template-building pipeline** orchestrated by the `syndiff-template` CLI. It turns TESS Full Frame Images (FFIs) and Pan-STARRS1 (PS1) skycell data into per-offset PS1 template FITS files that feed the main SynDiff difference-imaging pipeline (`run_pipeline.py`).

For difference imaging, Hotpants, ePSF, and forced photometry, see the [main README](../README.md).

**Documentation index**: [`docs/README.md`](README.md)

---

## Table of Contents

- [Overview](#overview)
- [Documentation layers and code lineage](#documentation-layers-and-code-lineage)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Concepts](#concepts)
  - [Targets](#targets)
  - [Runs and stages](#runs-and-stages)
  - [Resource pools](#resource-pools)
  - [Local vs HTCondor execution](#local-vs-htcondor-execution)
- [Pipeline Stages](#pipeline-stages)
- [Configuration Reference](#configuration-reference)
- [Targets CSV Formats](#targets-csv-formats)
- [CLI Reference](#cli-reference)
- [Run Lifecycle](#run-lifecycle)
- [Logging and Artifacts](#logging-and-artifacts)
- [Verification](#verification)
- [HTCondor Integration](#htcondor-integration)
- [Force Rerun Behavior](#force-rerun-behavior)
- [Per-SCC Overrides](#per-scc-overrides)
- [Troubleshooting](#troubleshooting)
- [Relationship to SynDiff Diff Imaging](#relationship-to-syndiff-diff-imaging)
- [Stage algorithm deep-dives](#stage-algorithm-deep-dives)
- [Module Map](#module-map)

---

## Overview

The template pipeline produces **PS1-based templates on the TESS pixel grid** for one or more science targets (sector / camera / CCD, or “SCC”). A typical end-to-end flow:

1. Download TESS FFIs (optional if already on disk).
2. **WCS grouping** — measure target pixel drift across epochs; assign template offset groups; write handoff JSON.
3. **Mapping** (“pancakes”) — map TESS pixels to PS1 skycells; download Gaia catalog for the reference FFI.
4. **PS1 download** — fetch PS1 skycell cutouts into a shared Zarr store.
5. **PS1 process** — convolve PS1 data onto the TESS grid (CPU-heavy; optionally on HTCondor).
6. **Downsample** — combine convolved skycells at multiple sub-pixel offsets → `syndiff_template_*.fits`.

The runner is designed for **batch operation across many SCCs**:

- A detached **scheduler** dequeues work subject to resource-pool limits.
- Progress is tracked in **SQLite** and on disk (logs, summaries).
- Stages can be run **subset-by-subset** (e.g. only `ps1_process,downsample`) when upstream artifacts already exist.
- **`ps1_process`** can run on a shared **HTCondor** pool; all other stages run as local subprocesses on the submit host.

---

## Documentation layers and code lineage

This guide covers **orchestration** — how to configure and run `syndiff-template` across many targets. The **algorithms** behind each stage are documented separately because they were developed and originally documented in the standalone [`syndiff`](../../syndiff/) research repository before being integrated into `syndiff_pipeline`.

| Layer | Location | What it covers |
|-------|----------|----------------|
| Orchestration | This file (`docs/template_pipeline.md`) | YAML config, scheduler, SQLite, Condor, CLI, logs |
| Stage algorithms | [`docs/stages/`](stages/README.md) | PanCAKES mapping, PS1 convolution, downsampling internals |
| Legacy standalone workflow | [`docs/stages/standalone_pipeline_overview.md`](stages/standalone_pipeline_overview.md) | Original `pipeline.py` + per-script CLI |
| Diff imaging | [`README.md`](../README.md) | Hotpants → photometry after templates exist |

### Script → module → stage mapping

| Legacy script (`syndiff/`) | Package module | `syndiff-template` stage |
|----------------------------|----------------|--------------------------|
| — | `download.py` | `tess_ffi_download` |
| — | `wcs_grouping.py` + `template_runner/handoff.py` | `wcs_grouping` |
| `pancakes_v2.py` | `template/pancakes.py` | `mapping` |
| `download_and_store_zarr.py` | `template/ps1_download.py` | `ps1_download` |
| `process_ps1.py` | `template/ps1_process.py` | `ps1_process` |
| `multi_offset_downsampling.py` | `template/downsample.py` | `downsample` |

The runner adds capabilities not present in the standalone scripts: **multi-target batching**, **WCS drift grouping** for transients, **artifact verification**, **force-rerun cleanup**, **pause/kill/retry**, and **HTCondor** for `ps1_process`.

If you previously used `syndiff/run.sh` one-liners, the equivalent production path is `syndiff-template submit` with `example/template_runner/config_real.yaml` (paths aligned to the same data layout).

---

## Architecture

```mermaid
flowchart TB
    subgraph CLI["syndiff-template CLI"]
        submit[submit]
        monitor[status / progress / logs]
        control[pause / resume / kill / retry]
    end

    subgraph Scheduler["Scheduler (detached process)"]
        pools[Resource pools]
        sqlite[(SQLite state DB)]
        skip[Artifact skip / verify]
    end

    subgraph Launch["Stage launcher"]
        local[Local subprocess]
        condor[HTCondor submit]
    end

    subgraph Stages["Stage workers (run_stage.py)"]
        s1[tess_ffi_download]
        s2[wcs_grouping]
        s3[mapping]
        s4[ps1_download]
        s5[ps1_process]
        s6[downsample]
    end

    submit --> Scheduler
    monitor --> sqlite
    control --> Scheduler
    Scheduler --> pools
    Scheduler --> skip
    pools --> Launch
    Launch --> local
    Launch --> condor
    local --> Stages
    condor --> s5
    Stages --> sqlite
```

**Hybrid execution model**

| Stage | Default executor | Resource pool | Notes |
|-------|------------------|---------------|-------|
| `tess_ffi_download` | local | `network` | MAST / tesscurl downloads |
| `wcs_grouping` | local | `cpu_light` | Writes per-target handoff under `handoff_root` |
| `mapping` | local | `cpu_light` | Gaia + skycell mapping (pancakes) |
| `ps1_download` | local | `network` | Shared Zarr at `{data_root}/ps1_skycells_zarr/` |
| `ps1_process` | **condor** | `cpu_heavy` | Whole-node jobs; configurable |
| `downsample` | local | `cpu_light` | Reads convolved Zarr + mapping |

**Stage dependency graph**

```text
tess_ffi_download
       │
       ▼
  wcs_grouping ─────────────────────────────┐
       │                                     │
       ▼                                     │
   mapping                                   │
       │                                     │
       ▼                                     │
 ps1_download                                │
       │                                     │
       ▼                                     │
  ps1_process ───────────────────────────────┤
                                             ▼
                                       downsample
```

`downsample` requires both `wcs_grouping` (crop bounds / ROI from `cluster_template_job.json`) and `ps1_process` (convolved Zarr).

When you run a **stage subset**, dependencies outside the subset are satisfied if **on-disk artifacts pass verification** (see [Verification](#verification)).

---

## Installation

The template runner ships with the `syndiff-pipeline` package.

```bash
# From a clone of this repository
pip install -e .

# Or install in your conda/mamba environment
mamba activate syndiff   # recommended env name in this project
pip install -e /path/to/syndiff_pipeline
```

This registers the console script:

```bash
syndiff-template --help
```

**Python**: ≥ 3.10 (see `pyproject.toml`).

**Core dependencies** (shared with the rest of SynDiff): `numpy`, `pandas`, `astropy`, `zarr`, `pyyaml`, `sep`, `scipy`, `shapely`, `numba`, `tqdm`, `filelock`, and others used by the `template/` modules.

**Mapping-specific**: the PanCAKES stage requires a **modified MOCPy** build with `MOC.filter_points_in_polygons` (Rust backend). See [`docs/stages/mapping_pancakes.md`](stages/mapping_pancakes.md) and the standalone repo’s `install_mocpy.sh`. Standard `pip install mocpy` is not sufficient.

**Cluster / Condor** (optional): HTCondor client tools (`condor_submit`, `condor_q`, `condor_history`, `condor_rm`) on the submit node. No `python-htcondor` package is required.

**Hardware** (from production experience): `ps1_process` expects a **whole node** (~64 cores, 512 GB RAM on the STScI science cluster). Mapping and downsample are lighter but benefit from multi-core hosts and fast NFS.

---

## Quick Start

### 1. Prepare config and targets

Copy and edit the examples under `example/template_runner/`:

```bash
cp example/template_runner/config_example.yaml my_config.yaml
cp example/template_runner/targets_example.csv my_targets.csv
```

Set at minimum:

- `data_root` — science data tree (FFIs, mapping, Zarr, templates).
- `ffi_dir` — TESS FFI directory (often `{data_root}/tess_ffi`).
- `handoff_root` — per-target WCS handoffs and pipeline metadata.
- `skycell_wcs_csv` — PS1 skycell WCS table (SkyCells list).
- `gaia_credentials` — Gaia archive credentials file (for mapping).

See [Configuration Reference](#configuration-reference).

### 2. Verify prerequisites (optional but recommended)

```bash
syndiff-template verify \
  --config my_config.yaml \
  --targets my_targets.csv \
  --stages tess_ffi_download,wcs_grouping,mapping,ps1_download
```

### 3. Submit a detached run

Always activate your conda environment first so the scheduler records the correct Python path in stage commands:

```bash
mamba activate syndiff

syndiff-template submit \
  --config my_config.yaml \
  --targets my_targets.csv \
  --stages ps1_process,downsample
```

Example output:

```text
Submitted run_id=20260607_210919 scheduler_pid=2692578
  logs: /path/to/runs/20260607_210919/scheduler.log
Monitor: syndiff-template progress --config ... --run-id 20260607_210919
```

### 4. Monitor

```bash
syndiff-template progress --config my_config.yaml --run-id 20260607_210919
syndiff-template status --watch --config my_config.yaml --run-id 20260607_210919
syndiff-template tail --config my_config.yaml --run-id 20260607_210919 \
  --target s0023_c1_k3_2020ftl --stage ps1_process
```

### 5. Use templates in SynDiff

Downsampled FITS appear under `{data_root}/shifted_downsampled/` (or `stages.downsample.output_base`). Point the main SynDiff config’s `template_dir` at that tree before running `run_pipeline.py`.

---

## Concepts

### Targets

A **target** is one SCC (sector, camera, CCD) plus transient coordinates and a name. Targets are loaded from CSV (see [Targets CSV Formats](#targets-csv-formats)).

Each target gets a stable **label** used in logs and SQLite:

```text
s{sector:04d}_c{camera}_k{ccd}_{target_name}
```

Example: `s0023_c1_k3_2020ftl` for sector 23, camera 1, CCD 3, SN 2020ftl.

### Runs and stages

A **run** is one scheduler session identified by `run_id` (default: UTC timestamp `YYYYMMDD_HHMMSS`). For each `(target, stage)` pair the scheduler creates a **stage run** row with status:

| Status | Meaning |
|--------|---------|
| `pending` | Not yet eligible (waiting on dependencies) |
| `ready` | Dependencies satisfied; waiting for pool capacity |
| `running` | Stage command launched |
| `success` | Exit code 0 |
| `failed` | Non-zero exit; downstream stages blocked |
| `skipped` | Artifact already on disk (normal submit without `--force-rerun`) |
| `blocked` | Never started (upstream failure or run killed) |
| `killed` | Terminated by user `kill` |

Run-level status (`runs.status`): `running`, `success`, `failed`, `killed`.

### Resource pools

Concurrency is limited per **pool** (not globally):

| Pool | Stages | Typical limit | Purpose |
|------|--------|---------------|---------|
| `network` | `tess_ffi_download`, `ps1_download` | 3 | Throttle MAST / PS1 API |
| `cpu_light` | `wcs_grouping`, `mapping`, `downsample` | 2 | Moderate CPU / I/O |
| `cpu_heavy` | `ps1_process` | 1–4 | Heavy convolution (local or Condor slot count) |

Configure under `resources:` in YAML. For Condor, `cpu_heavy.max_concurrent` caps **simultaneous Condor submissions**, not CPUs per job.

### Local vs HTCondor execution

- **Local**: `subprocess.Popen` with `start_new_session=True` (own process group for clean kill).
- **Condor**: only `ps1_process` by default (`stages.ps1_process.executor: condor`).

The Condor path:

1. Writes a `.condor.submit` file next to the stage log.
2. Submits via `condor_submit`.
3. Stores the **cluster ID** in SQLite as `pid`.
4. Polls with `condor_q` / `condor_history`.

Execute nodes run `template_runner/condor_wrapper.sh`, which activates the `syndiff` conda env and `exec`s the same `run_stage.py` command the local launcher would use.

---

## Pipeline Stages

### `tess_ffi_download`

**Module**: `syndiff_pipeline.download`

Downloads calibrated TESS FFIs for the target SCC into `ffi_dir` using the shared download helpers.

**Verification**: at least one FFI file present under the nested sector/camera/ccd directory.

---

### `wcs_grouping`

**Module**: `template_runner/handoff.py` → `syndiff_pipeline.wcs_grouping`

**Inputs**: FFIs on disk; target RA/Dec from targets CSV.

**Outputs** (under `{handoff_root}/{target_label}/`):

| File | Description |
|------|-------------|
| `syndiff_ffi_frames.csv` | Per-FFI WCS drift, template group IDs |
| `cluster_template_job.json` | Reference FFI, crop bounds, offsets for downsample |

**Verification**: valid `cluster_template_job.json` with existing `reference_ffi_path`.

---

### `mapping`

**Module**: `template/pancakes.py` (ported from `pancakes_v2.py`)

Builds TESS↔PS1 skycell pixel mappings for the reference FFI from `cluster_template_job.json`. Optionally downloads a Gaia catalog (`skip_download_catalog: false` by default).

**Algorithm summary** (see [PanCAKES deep-dive](stages/mapping_pancakes.md)):

1. Build a TESS MOC footprint and filter the PS1 skycell catalog to overlapping cells.
2. Assign every TESS pixel to a skycell index (mocpy + Numba point-in-polygon).
3. In parallel, project each skycell’s TESS pixel footprints onto the PS1 grid → per-skycell registration FITS.
4. Compute padding skycells at projection edges for downstream convolution.

**Outputs** (under `{data_root}/skycell_pixel_mapping/`):

```text
sector_{SSSS}/camera_{C}/ccd_{K}/
  tess_s{SSSS}_{C}_{K}_master_skycells_list.csv
  tess_s{SSSS}_{C}_{K}_master_pixels2skycells.fits.gz
  tess_s{SSSS}_{C}_{K}_{skycell}.fits   (per skycell)
```

With `oversampling_factor > 1`, paths include an `oversampling_{N}/` prefix and `_os{N}` suffixes.

**Verification**: master skycells CSV exists.

**Deep dive**: [mapping_pancakes.md](stages/mapping_pancakes.md)

---

### `ps1_download`

**Module**: `template/ps1_download.py` (ported from `download_and_store_zarr.py`)

Downloads PS1 skycell data listed in the mapping CSV into a **shared Zarr store**:

```text
{data_root}/ps1_skycells_zarr/ps1_skycells.zarr
```

Uses a lock file (`ps1_skycells.zarr.lock`) so concurrent downloads for different SCCs on the same `data_root` serialize safely. Tune `resources.network.max_concurrent` accordingly.

**Verification**: Zarr store exists and is non-empty.

**Standalone CLI reference**: [standalone pipeline overview — Download PS1](stages/standalone_pipeline_overview.md#2-download-ps1-data)

---

### `ps1_process`

**Module**: `template/ps1_process.py` (ported from `process_ps1.py`)

Reads PS1 Zarr + mapping CSV; runs the **modern sliding-window convolution pipeline**. Sizes worker counts from **whole-machine** `os.cpu_count()` and available RAM — on Condor this expects a **whole-node** claim (`request_cpus=64`, large memory).

**Algorithm summary** (see [PS1 process technical reference](stages/ps1_process_technical.md)):

- Five concurrent stages: zarr readers → band combiners → SEP source extraction (process pool) → sequential sliding-window assembler (padding + Gaussian convolution) → Zarr saver.
- Master arrays use a two-row sliding window with 480 px cell overlap and cross-projection padding via `reproject_interp`.
- Optional `--remove-saturated-stars` writes a removed-star CSV used later by downsample / SynDiff sat templates.

**Outputs**:

| Path | Description |
|------|-------------|
| `{data_root}/convolved_results/sector_{SSSS}_camera_{C}_ccd_{K}.zarr` | Convolved skycell arrays (`*_data`, masks) |
| `{data_root}/convolved_results/sector_{SSSS}_camera_{C}_ccd_{K}_removed_stars.csv` | Optional removed-star records (when enabled) |

**Verification**: convolved Zarr contains the expected number of non-empty `*_data` arrays (derived from mapping CSV and `projections_limit`).

**Key parameters**: `psf_sigma`, `remove_saturated_stars`, `projections_limit` (smoke testing), Condor resource requests.

**Deep dive**: [ps1_process_technical.md](stages/ps1_process_technical.md) (architecture diagrams, queue reference, log prefixes)

---

### `downsample`

**Module**: `template/downsample.py` (ported from `multi_offset_downsampling.py`)

Combines convolved Zarr data at multiple sub-pixel offsets from `cluster_template_job.json`. Produces template FITS for SynDiff Hotpants.

**Algorithm summary** (see [downsample technical reference](stages/downsample_technical.md)):

1. Load TESS WCS + master registration map; filter skycells to the WCS-grouping ROI.
2. Precompute per-skycell PS1 pixel shifts for each `(dx, dy)` offset via WCS round-trip.
3. Parallel joblib workers bin shifted PS1 flux into TESS pixels using registration FITS.
4. Deduplicate overlapping skycell contributions; write one multi-extension FITS per offset (`FLUX_SUM`, `COUNT`, `MASK`).

Default production offsets are the calibrated dither list from the standalone script (10 pairs); WCS grouping supplies the subset needed for each transient’s template groups.

**Outputs** (under `output_base`, default `{data_root}/shifted_downsampled/`):

```text
sector{SSSS}_camera{C}_ccd{K}[_x..._y...][_os{N}]/
  syndiff_template_s{SSSS}_{C}_{C}_dx{X.XXX}_dy{Y.YYY}.fits
  ...
```

**Verification**: at least one `syndiff_template_*.fits` under the target directory glob.

**Deep dive**: [downsample_technical.md](stages/downsample_technical.md)

---

## Configuration Reference

Configuration is YAML loaded by `template_runner/runner_config.py`. Paths may be absolute or relative to the **config file’s directory**.

### Top-level keys

| Key | Required | Description |
|-----|----------|-------------|
| `data_root` | yes | Root for mapping, Zarr, convolved results, default template output |
| `ffi_dir` | no | TESS FFI root (defaults to `data_root`) |
| `handoff_root` | yes | Per-target WCS handoffs (`{handoff_root}/{target_label}/`) |
| `runs_root` | no | Run logs and summaries (default: `{handoff_root}/runs`) |
| `state_db_path` | no | SQLite DB (default: `{handoff_root}/pipeline_state.sqlite`) |
| `skycell_wcs_csv` | yes | PS1 SkyCells WCS CSV |
| `gaia_credentials` | no | Gaia archive credentials (mapping) |
| `stages` | no | Per-stage parameters (see below) |
| `resources` | no | Pool concurrency limits |
| `scheduler` | no | Scheduler tuning |
| `overrides` | no | Per-SCC parameter overrides |

### Scheduler

```yaml
scheduler:
  heartbeat_interval_s: 30.0
```

### Resource pools

```yaml
resources:
  network:
    max_concurrent: 3
  cpu_light:
    max_concurrent: 2
  cpu_heavy:
    max_concurrent: 3   # max simultaneous ps1_process jobs (Condor or local)
```

Defaults if omitted: `network=3`, `cpu_light=2`, `cpu_heavy=1`.

### Stage parameters

Unknown keys under `stages.*` raise `ValueError` at load time (strict allow-list).

#### `stages.wcs_grouping`

| Key | Default | Description |
|-----|---------|-------------|
| `offset_threshold` | `0.01` | Max pixel drift before new template group |
| `wcs_drift_savgol_window` | `11` | Savitzky–Golay window for drift smoothing |
| `wcs_drift_savgol_polyorder` | `2` | SG polynomial order |
| `bkg_vector_path` | null | Optional TESSVectors path for Earth/Moon angles |
| `crop_quadrant` | `"full"` | Default crop mode if bounds not set (`full` = entire FFI; `full_science` = usable area only) |
| `x_min`, `x_max`, `y_min`, `y_max` | null | Explicit crop bounds (pixels) |
| `x_left_dead`, `x_right_dead` | `44` | Horizontal dead columns |
| `y_edge_strip` | `30` | Vertical edge strip |

#### `stages.mapping`

| Key | Default | Description |
|-----|---------|-------------|
| `buffer`, `tess_buffer`, `pad_distance` | various | Pancakes geometry buffers |
| `edge_exclusion`, `edge_buffer_large`, `edge_buffer_small` | various | Edge handling |
| `n_threads` | `8` | Thread count |
| `max_workers` | null | Optional process pool cap |
| `oversampling_factor` | `1` | Sub-pixel oversampling |
| `overwrite` | `true` | Overwrite mapping FITS |
| `skip_download_catalog` | `false` | Skip Gaia download if catalog exists |

#### `stages.ps1_download`

| Key | Default | Description |
|-----|---------|-------------|
| `num_workers` | `8` | Download parallelism |
| `use_local_files` | `false` | Read from local PS1 tree instead of API |
| `local_data_path` | `data/ps1_skycells` | Local PS1 path when `use_local_files` |
| `overwrite` | `false` | Re-download into Zarr |
| `log_level` | `INFO` | Logging level |

#### `stages.ps1_process`

| Key | Default | Description |
|-----|---------|-------------|
| `projections_limit` | null | Limit skycell rows (smoke tests); null = all |
| `psf_sigma` | `60.0` | Gaussian convolution sigma |
| `enable_saturation_correction` | `true` | Saturation handling |
| `remove_saturated_stars` | `false` | Track removed stars → CSV |
| `catalog_path` | null | Override Gaia catalog path |
| `bright_star_mag_threshold` | `13.0` | Bright-star cutoff |
| `executor` | `"condor"` | `"condor"` or `"local"` |
| `condor_request_cpus` | `64` | HTCondor `request_cpus` |
| `condor_request_memory` | `500000` | HTCondor `request_memory` (MB) |
| `condor_requirements` | `Memory >= 500000 && LoadAvg < 10` | Machine requirements expression |
| `condor_rank` | `-LoadAvg` | Prefer lower load average |

#### `stages.downsample`

| Key | Default | Description |
|-----|---------|-------------|
| `ignore_mask_bits` | `[12]` | PS1 mask bits to ignore |
| `oversampling_factor` | `1` | Must match mapping |
| `mapping_dir` | null | Override mapping root |
| `convolved_dir` | null | Override convolved Zarr directory |
| `output_base` | null | Template FITS output root |
| `single_offset` | `false` | Single `[0,0]` offset only (smoke) |

### Resolved per-target paths

For each target, `resolve_config()` derives:

| Field | Path |
|-------|------|
| `handoff_dir` | `{handoff_root}/{target_label}/` |
| `mapping_root` | `{data_root}/skycell_pixel_mapping/` |
| `zarr_dir` | `{data_root}/ps1_skycells_zarr/` |
| `template_output_base` | `{data_root}/shifted_downsampled/` |

---

## Targets CSV Formats

### Normalized format (recommended)

Header (all columns required):

```csv
sector,camera,ccd,target_ra,target_dec,target_name,enabled
23,1,3,185.015708,5.343289,2020ftl,true
```

Rows with `enabled=false` are skipped.

### SN event catalog format

Header:

```csv
ID,redshift,type,ra,dec,...,tess_coverage,...
```

`ID` may be prefixed with `SN `. `tess_coverage` uses tokens like `S23C1D3` or `S44C2D1; S45C1D4` for multi-SCC events. One target row is expanded per SCC token.

See `example/template_runner/events_example.csv`.

---

## CLI Reference

All commands require `--config PATH` unless noted.

| Command | Description |
|---------|-------------|
| `submit` | Detached scheduler (production) |
| `run` | Foreground scheduler (debugging) |
| `status` | Per-target stage status grid (`--watch`, `--interval`) |
| `progress` | Aggregate status counts |
| `runs` | List recent runs from SQLite |
| `active` | Runs whose scheduler PID is still alive |
| `show` | Print `run_meta.json` |
| `logs` | Print scheduler or stage log (`--target`, `--stage`, `--follow`) |
| `tail` | Alias for `logs --follow` |
| `verify` | Check on-disk artifacts (`--targets`, `--scc`, `--stages`) |
| `retry` | Re-queue one failed stage for one SCC (`--scc`, `--stage`, `--targets`) |
| `pause` | Stop dequeuing new stages (running stages continue) |
| `resume` | Resume dequeuing |
| `kill` | Terminate scheduler, Condor clusters, local jobs; mark run killed |

### Common flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--targets PATH` | submit, run, verify, retry | Targets CSV |
| `--stages LIST` | submit, run, verify | Comma-separated subset (default: all stages) |
| `--run-id ID` | most | Explicit run (default: latest symlink) |
| `--force-rerun` | submit, run | Re-run stages; see [Force Rerun](#force-rerun-behavior) |

### Examples

```bash
# Full pipeline, detached
syndiff-template submit --config cfg.yaml --targets targets.csv

# Re-process PS1 convolution + downsample from scratch
syndiff-template submit --config cfg.yaml --targets targets.csv \
  --stages ps1_process,downsample --force-rerun

# Verify one SCC
syndiff-template verify --config cfg.yaml --targets targets.csv \
  --scc 23,1,3 --stages ps1_process

# Retry mapping after a fix
syndiff-template retry --config cfg.yaml --targets targets.csv \
  --run-id 20260607_210919 --scc 23,1,3 --stage mapping

# Kill a run (Condor + local)
syndiff-template kill --config cfg.yaml --run-id 20260607_210919
```

---

## Run Lifecycle

### Submit (`submit`)

1. Creates `{runs_root}/{run_id}/` layout and `run_meta.json`.
2. Inserts run + `(target, stage)` rows in SQLite (`pending`).
3. Spawns detached scheduler; writes `scheduler.pid`.
4. Symlinks `{runs_root}/latest` → `run_id`.

### Scheduler loop

1. **Skip existing artifacts** (unless `--force-rerun`): verified stages → `skipped`.
2. **Promote** `pending` → `ready` when dependencies satisfied.
3. **Launch** up to pool capacity; update status → `running`, record `pid`.
4. **Poll** local PIDs / Condor clusters; on exit → `success` or `failed`.
5. On failure, **block** downstream stages for that target.
6. Write `summary.json` / `summary.csv` each iteration.
7. Exit when no running jobs and no pending/ready work remain.

### Pause / resume

Pause sets a flag in SQLite; the scheduler sleeps without launching new work. Running stages are not stopped.

### Kill

`kill`:

1. SIGTERM scheduler process tree.
2. For each `running` stage with a `pid`:
   - Condor stages → `condor_rm <cluster_id>`
   - Local stages → terminate process group
3. Mark running stages `killed` (exit 143); pending/ready → `blocked`.
4. Set run status `killed`; remove `scheduler.pid`.

The scheduler’s shutdown handler also calls `terminate()` on active handles (including Condor) when it exits on SIGTERM.

### Retry

`retry` resets one `(target, stage)` to `ready` and blocks downstream stages in SQLite. The scheduler must still be running (or you must submit a new run) to pick up the work.

---

## Logging and Artifacts

### Run directory layout

```text
{runs_root}/{run_id}/
  run_meta.json          # submit metadata, config paths, force_rerun flag
  scheduler.log          # orchestration log
  scheduler.pid          # while scheduler alive
  summary.json           # live status counts
  summary.csv            # flat stage table
  per_target/
    {target_label}/
      {stage}.log                    # primary stage log (stdout/stderr tee)
      ps1_process.condor.submit      # Condor only
      ps1_process.condor.stdout
      ps1_process.condor.stderr
      ps1_process.condor.log         # Condor job event log
```

**Primary debugging path**: `{stage}.log` — written by `run_stage.py` on NFS, including on Condor execute nodes.

Condor `.condor.*` files capture wrapper/submit diagnostics when the job fails before Python starts.

### SQLite state

Default: `{handoff_root}/pipeline_state.sqlite`

Tables: `runs`, `targets`, `stage_runs`. Safe to query while scheduler runs (WAL timeout 60s). Used by all status/progress commands.

---

## Verification

`syndiff-template verify` checks **on-disk artifacts**, not SQLite run state.

| Stage | Check |
|-------|-------|
| `tess_ffi_download` | FFI files exist |
| `wcs_grouping` | Valid `cluster_template_job.json` |
| `mapping` | Master skycells CSV |
| `ps1_download` | Non-empty `ps1_skycells.zarr` |
| `ps1_process` | Convolved Zarr complete (counts `*_data` arrays vs mapping) |
| `downsample` | `syndiff_template_*.fits` present |

Partial convolved Zarr (interrupted run) reports e.g. `Partial convolved zarr: 3/120 skycells saved`.

Use verify before subset runs to confirm upstream stages are satisfied off-run.

---

## HTCondor Integration

### Requirements

- Submit host: Condor client tools, `syndiff` conda env, NFS access to `data_root`, `handoff_root`, and `runs_root`.
- Execute nodes: same NFS mounts for `/home` (conda) and science data; no inbound file transfer (`should_transfer_files = NO`).
- Jobs run as the submitting Unix user (`getenv = false` — wrapper sets up environment).

### Wrapper script

`template_runner/condor_wrapper.sh`:

1. Resolves `HOME` via `getent` (Condor does not export it).
2. `source ~/miniforge3/etc/profile.d/conda.sh && conda activate syndiff`
3. `exec` the stage command (absolute Python path from submit host).

**Important**: Run `syndiff-template submit` with `syndiff` activated so `sys.executable` in the command points at the correct env.

Adjust the miniforge path in the wrapper if your install location differs.

### Submit file (generated)

Per job, written to `per_target/{target}/ps1_process.condor.submit`:

- `executable = .../condor_wrapper.sh`
- `arguments = /path/to/python -m syndiff_pipeline.template_runner.run_stage ...`
- `request_cpus`, `request_memory`, `requirements`, `rank`
- `output`, `error`, `log` → sibling `.condor.*` files

### Resource sizing (example: STScI science cluster)

Typical `ps1_process` settings for 64-core / 512 GB nodes:

```yaml
stages:
  ps1_process:
    condor_request_cpus: 64
    condor_request_memory: 500000
    condor_requirements: "Memory >= 500000 && LoadAvg < 10"
    condor_rank: "-LoadAvg"
resources:
  cpu_heavy:
    max_concurrent: 3
```

`ps1_process` auto-scales workers to the allocated machine; partial-node claims are not supported.

### Monitoring Condor jobs

```bash
condor_q -submitter $(whoami)
condor_history <cluster_id>
```

Cluster IDs match SQLite `stage_runs.pid` for running `ps1_process` stages.

### Local fallback

For laptops or debugging:

```yaml
stages:
  ps1_process:
    executor: local
```

---

## Force Rerun Behavior

`--force-rerun` on `submit` or `run`:

1. **Scheduler bookkeeping**: resets selected stages to `pending` in SQLite (even if previously `success` or `skipped`).
2. **Skips artifact-exists checks** for those stages.
3. **Does not** automatically delete upstream artifacts for other stages.

### `ps1_process` artifact cleanup

When `ps1_process` runs with `--force-rerun`, it **deletes existing outputs first**:

- `{data_root}/convolved_results/sector_{SSSS}_camera_{C}_ccd_{K}.zarr`
- `{data_root}/convolved_results/sector_{SSSS}_camera_{C}_ccd_{K}_removed_stars.csv`

Deletion is logged in `ps1_process.log`. This ensures a clean Zarr rewrite (`ps1_process` opens Zarr in append mode otherwise).

Other stages are **not** auto-deleted on force rerun. Remove mapping CSV, shared PS1 Zarr, or template FITS manually if you need a full rebuild.

---

## Per-SCC Overrides

The `overrides` map keys SCC as `"sector/camera/ccd"` or `"sector/camera/ccd"` matching `Target.scc_key()`:

```yaml
overrides:
  "23/2/1":
    stages:
      ps1_process:
        projections_limit: 1   # smoke test on one SCC
```

Optional per-override `data_root` redirects that SCC’s data paths.

---

## Troubleshooting

### Condor job exits immediately (exit code 1)

Check `ps1_process.condor.stderr` first.

| Symptom | Likely cause |
|---------|----------------|
| `HOME: unbound variable` | Old wrapper; upgrade to current `condor_wrapper.sh` |
| `cannot find miniforge3` | Execute node lacks NFS home or different install path |
| Empty `ps1_process.log` | Wrapper failed before Python started |

### Partial or stale convolved Zarr

Verify reports partial counts. Use `--force-rerun` with `ps1_process` (auto-deletes Zarr + removed-stars CSV) or delete the Zarr directory manually.

### Stage stuck in `pending`

Upstream dependency not `success`/`skipped` in-run, or off-run artifact missing. Run `verify` for dependency stages.

### `ps1_download` contention

Multiple SCCs share one Zarr. Lock file serializes writers; excessive `network.max_concurrent` may queue internally — normal.

### Scheduler died but Condor jobs still running

Run `syndiff-template kill` (or `condor_rm` manually using cluster IDs from `condor_q`). Check `active` and `runs` commands.

### Import errors on Condor execute nodes

Ensure the same conda env exists on NFS and the submit host used `syndiff` when submitting. Mapping imports `pancakes` at module load — even `ps1_process`-only runs pull in heavy deps through `stages.py`.

---

## Relationship to SynDiff Diff Imaging

```text
┌─────────────────────────────────────┐
│  syndiff-template (this document)   │
│  PS1 templates on TESS grid         │
│  → shifted_downsampled/*.fits       │
└─────────────────┬───────────────────┘
                  │ template_dir
                  ▼
┌─────────────────────────────────────┐
│  run_pipeline.py / SynDiff          │
│  WCS grouping → Hotpants → ePSF →   │
│  background → forced photometry     │
└─────────────────────────────────────┘
```

Template pipeline **handoff** (`cluster_template_job.json`, frame manifest) is written under `handoff_root`, separate from SynDiff’s main `output_dir`. You can reuse WCS grouping logic in both contexts; paths are configured independently.

---

## Stage algorithm deep-dives

For maintainers and algorithm reviewers, full step-by-step technical references (originally in `../syndiff/`) are vendored under [`docs/stages/`](stages/README.md):

| Stage | Document | Highlights |
|-------|----------|------------|
| `mapping` | [mapping_pancakes.md](stages/mapping_pancakes.md) | MOC filtering, master pixel map, padding skycells, output FITS layout |
| `ps1_process` | [ps1_process_technical.md](stages/ps1_process_technical.md) | 5-stage pipeline, queues, memory guards, cross-projection padding |
| `downsample` | [downsample_technical.md](stages/downsample_technical.md) | Shift precompute, sparse binning, ROI/oversampling, FITS HDUs |
| All (legacy CLI) | [standalone_pipeline_overview.md](stages/standalone_pipeline_overview.md) | `pipeline.py`, per-script invocations, `run.sh` equivalents |

---

## Module Map

| Module | Role |
|--------|------|
| `template_runner/cli.py` | `syndiff-template` entry point |
| `template_runner/scheduler.py` | Resource-pool orchestration |
| `template_runner/state.py` | SQLite schema and queries |
| `template_runner/runner_config.py` | YAML loading, path resolution, overrides |
| `template_runner/stage_params.py` | Typed stage parameters + validation |
| `template_runner/stages.py` | Stage registry and in-process execution |
| `template_runner/run_stage.py` | Subprocess/Condor worker entry point |
| `template_runner/launcher.py` | Local vs Condor launch |
| `template_runner/condor.py` | Condor submit file + CLI polling |
| `template_runner/condor_wrapper.sh` | Conda activation on execute nodes |
| `template_runner/daemon.py` | Detached scheduler spawn + process trees |
| `template_runner/logs.py` | Log paths and tee helper |
| `template_runner/targets.py` | CSV target loading |
| `template_runner/verify.py` | Artifact verification + force-rerun cleanup |
| `template_runner/handoff.py` | WCS grouping wrapper |
| `template/pancakes.py` | Mapping stage |
| `template/ps1_download.py` | PS1 Zarr download |
| `template/ps1_process.py` | Convolution pipeline |
| `template/downsample.py` | Multi-offset template FITS |

---

## Example files

| File | Purpose |
|------|---------|
| `example/template_runner/config_example.yaml` | Annotated starter config |
| `example/template_runner/config_real.yaml` | Production-style STScI cluster config |
| `example/template_runner/targets_example.csv` | Normalized multi-target CSV |
| `example/template_runner/events_example.csv` | SN catalog format |
| `example/template_runner/README.md` | Quick-start pointer |
| `docs/stages/` | Algorithm deep-dives (from `../syndiff/` step READMEs) |
| `../syndiff/run.sh` | Historical per-SCC command log (reference only) |

---

*For questions, bug reports, or contributions, use the project’s GitHub issue tracker once published.*
