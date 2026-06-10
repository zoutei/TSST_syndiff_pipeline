# `syndiff-template` scripts reference

This document explains **what each script and command in the `syndiff-template` toolchain does**. It is a script-oriented companion to the orchestration guide in [`template_pipeline.md`](template_pipeline.md) and the algorithm deep-dives in [`stages/`](stages/README.md).

`syndiff-template` is installed as a single console entry point (`pyproject.toml` → `template_runner/cli.py`). Everything below is reachable through that CLI or through the internal modules it launches.

---

## Table of contents

- [CLI commands](#cli-commands)
- [Pipeline stages](#pipeline-stages)
- [Internal worker entry points](#internal-worker-entry-points)
- [Template package modules](#template-package-modules)
- [Runner support modules](#runner-support-modules)
- [Related documentation](#related-documentation)

---

## CLI commands

Run `syndiff-template --help` or `syndiff-template <command> --help` for flags. Commands are grouped by what they operate on.

### Submit and run

| Command | What it does |
|---------|--------------|
| **`submit`** | Production entry point. Creates a run directory under `{handoff_root}/runs/<run_id>/`, freezes `config.yaml` and `targets.csv`, registers the run in SQLite, starts the supervisor daemon (if needed), and returns immediately. The daemon schedules stage workers in the background. |
| **`run`** | Same as `submit` but runs the scheduler loop in the **foreground** until the run finishes. Intended for debugging on a laptop, not long production batches. |

### Monitor

| Command | What it does |
|---------|--------------|
| **`progress`** | Prints aggregate stage counts (`pending=`, `running=`, `success=`, …) for active runs. With detail enabled (default), parses running workers' logs or sidecars for fractional progress (e.g. `ps1_dl: 342/1009`, `down: 45/84`). |
| **`status`** | Prints a per-target stage grid using short names (`tess_dl`, `wcs`, `map`, `ps1_dl`, `ps1_pr`, `down`). Supports `--watch` for live refresh. |
| **`show`** | Dumps `run_meta.json` (submit time, source config path, selected stages, `force_rerun` flag). |
| **`logs`** | Prints log text. Without `--target`/`--stage`, shows the workspace daemon log. With both, shows the stage worker log under `per_target/<label>/<stage>.log`. |
| **`tail`** | Alias for `logs --follow` (like `tail -f`). |

With no `--run-dir` or `--run-id`, monitor commands auto-discover the workspace and show **all active runs** (or the latest run if none are active).

### Workspace

| Command | What it does |
|---------|--------------|
| **`runs`** | Lists recent runs from SQLite with status and whether the supervisor daemon is alive. |
| **`active`** | Shows runs in `running` or `stalled` status plus supervisor PID and heartbeat age. |
| **`daemon start`** | Starts the host-level supervisor for a `handoff_root` (flock-guarded, one owner per workspace). Starts the Discord bot when the supervisor was already running; otherwise the new supervisor starts the bot on startup. Normally invoked automatically by `submit`. |
| **`daemon stop`** | Stops the supervisor and all Discord bots for that workspace. |
| **`daemon status`** | JSON summary: supervisor alive/wedged, PID, heartbeat age, Discord bot state. |

### Run control

These insert **command intents** into SQLite; the supervisor daemon applies them on its next tick.

| Command | What it does |
|---------|--------------|
| **`retry`** | Re-queues failed or canceled stages. Bulk retry (no flags) or single `--scc` + `--stage`. Resets downstream dependencies for targeted retries. |
| **`pause`** | Stops dequeuing new stages for the run; in-flight workers continue until they exit. |
| **`resume`** | Clears pause and resumes scheduling. |
| **`kill`** | Cancels the run: marks stages `canceled`, terminates local subprocesses, sweeps HTCondor clusters. |

### Verification and manifests

| Command | What it does |
|---------|--------------|
| **`verify`** | Read-only check of **on-disk artifacts** (not SQLite state). Use before submit to confirm prerequisites, or after a partial run to debug one SCC. Writes completion manifests when checks pass. |
| **`reconcile-manifests`** | One-shot backfill of stable completion manifests under `{runs_root}/.manifests/` for data that already exists. Lets future runs skip expensive NFS scans. |

### Discord (optional)

| Command | What it does |
|---------|--------------|
| **`notify test`** | Sends a read-only Discord preview (same progress + status text as CLI). `--dry-run` prints locally without posting. |
| **`discord bot`** | Runs the on-demand status-reply bot in the **foreground** (debug). Production bots are auto-started by the supervisor or by `submit`/`retry` when the supervisor is already up. |

---

## Pipeline stages

When the scheduler launches work, it runs one of six stages per target. Stages are executed by `template_runner/run_stage.py`, which calls into the modules listed here.

```text
tess_ffi_download → wcs_grouping → mapping → ps1_download → ps1_process → downsample
                                      └──────────────────────────────────────┘
                                      (downsample also depends on wcs_grouping)
```

| Stage | Module | What it does |
|-------|--------|--------------|
| **`tess_ffi_download`** | `download.py` | Downloads calibrated TESS FFIs for the target sector/camera/CCD from MAST (tesscurl manifest or astroquery). Writes FITS under `{data_root}/tess_ffi/`. |
| **`wcs_grouping`** | `wcs_grouping.py` via `template_runner/handoff.py` | Measures pixel drift of the transient across FFIs, smooths drift, assigns template offset groups, and writes handoff files: `cluster_template_job.json`, `syndiff_ffi_frames.csv`, `wcs_drift_template_debug.png` under `{handoff_root}/{target_label}/`. |
| **`mapping`** | `template/pancakes.py` | PanCAKES v2: builds TESS↔PS1 skycell pixel mappings for the reference FFI. Optionally downloads a Gaia catalog. Outputs master skycell CSV, pixel-to-skycell map, and per-skycell registration FITS under `{data_root}/skycell_pixel_mapping/`. |
| **`ps1_download`** | `template/ps1_download.py` | Downloads PS1 skycell cutouts listed in the mapping CSV into a shared Zarr store at `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`. Uses a file lock so concurrent SCCs serialize safely. Skipped when `ps1_source: stream` (download happens inside `ps1_process`). |
| **`ps1_process`** | `template/ps1_process.py` | Convolve PS1 data onto the TESS grid using a sliding-window pipeline (zarr readers → band combiners → SEP extraction → Gaussian convolution with cross-projection padding → Zarr saver). Writes `{data_root}/convolved_results/sector_*_camera_*_ccd_*.zarr`. CPU-heavy; defaults to HTCondor. |
| **`downsample`** | `template/downsample.py` | Combines convolved Zarr at multiple sub-pixel offsets from `cluster_template_job.json` into template FITS (`syndiff_template_*.fits`) under `{data_root}/shifted_downsampled/`. These feed SynDiff Hotpants difference imaging. |

**Executors**: `mapping` and `ps1_process` default to HTCondor; all other stages run as local subprocesses on the submit host.

---

## Internal worker entry points

These are not usually typed at a shell prompt, but they are the scripts `syndiff-template` actually runs.

| Script | Invocation | What it does |
|--------|------------|--------------|
| **`template_runner/run_stage.py`** | `python -m syndiff_pipeline.template_runner.run_stage --run-id … --stage … --run-dir … --target-label …` | Worker for a single target + stage. Configures logging, writes `per_target/<label>/<stage>.log` and durable `*.status.json`, calls `stages.execute_stage()`, and writes completion manifests on success. Used for both local subprocesses and Condor jobs. |
| **`template_runner/scheduler.py`** | `python -m syndiff_pipeline.template_runner.scheduler --daemon --deployment …` or foreground `--run-id` + `--run-dir` | Supervisor loop: ingests command intents, reconciles running jobs, verifies artifacts, promotes stages, claims pool slots, launches workers (local or Condor), detects stalls. The detached daemon is the long-lived process behind `submit`. |
| **`template_runner/condor_wrapper.sh`** | HTCondor `executable` on execute nodes | Activates the `syndiff` conda environment on NFS-mounted home and `exec`s the `run_stage.py` command. Required because Condor jobs do not inherit the submit host shell. |
| **`template_runner/discord_bot.py`** | `syndiff-template discord bot --deployment …` or auto-spawned by daemon | Listens in a configured Discord channel and replies to messages with live `progress` + `status` output. |

---

## Template package modules

Modules under `template/` implement the science algorithms. Several can still be run **standalone** for debugging outside the scheduler.

### Runnable standalone (have `__main__` / CLI)

| Module | Standalone command | What it does |
|--------|-------------------|--------------|
| **`template/pancakes.py`** | `python -m syndiff_pipeline.template.pancakes` | PanCAKES mapping only: TESS FITS → skycell pixel maps and registration FITS. Same core logic as the `mapping` stage. |
| **`template/ps1_download.py`** | `python -m syndiff_pipeline.template.ps1_download` | Download PS1 skycells from the mapping CSV into the shared Zarr store. |
| **`template/ps1_process.py`** | `python -m syndiff_pipeline.template.ps1_process` | Run the sliding-window convolution pipeline for one SCC. |
| **`template/downsample.py`** | `python -m syndiff_pipeline.template.downsample` | Multi-offset downsampling for one SCC (offsets and ROI can be passed on the CLI). |
| **`template/compute_ps1_skycell_shifts.py`** | `python -m syndiff_pipeline.template.compute_ps1_skycell_shifts` | Utility: compute per-skycell PS1 pixel shifts from a small TESS pixel offset using WCS round-trips. Used in shift precompute for downsample; runnable alone for debugging WCS shifts. |
| **`download.py`** | `python -m syndiff_pipeline.download` | Download TESS FFIs for one sector/camera/CCD. Same as the `tess_ffi_download` stage. |

### Library-only (imported by stages)

| Module | What it does |
|--------|--------------|
| **`template/csv_utils.py`** | Load and parse skycell mapping and padding CSVs used by `ps1_process` and cross-projection padding. |
| **`template/band_utils.py`** | Combine PS1 r/i/z/y bands, background removal, SEP source extraction helpers, TESS-equivalent magnitude from Gaia photometry. |
| **`template/zarr_utils.py`** | Load and save PS1 skycell bands, masks, weights, and headers from the shared Zarr store. |
| **`template/convolution_utils.py`** | Gaussian convolution (Dask-backed) to apply the TESS PSF during `ps1_process`. |
| **`template/correct_saturation.py`** | Identify and correct saturated stars using Gaia catalog + ePSF fitting; supports optional removed-star CSV output. |
| **`template/cross_projection_padding.py`** | Apply padding across PS1 projection boundaries during sliding-window assembly (uses mapping padding metadata). |
| **`template/downsample_progress.py`** | Atomic JSON sidecar (`downsample.progress.json`) updated by parallel downsample workers; read by `syndiff-template progress`. |

---

## Runner support modules

These modules have no user-facing CLI of their own but implement orchestration pieces used by `syndiff-template`.

| Module | Role |
|--------|------|
| `template_runner/cli.py` | `syndiff-template` entry point and argument parsing. |
| `template_runner/runner_config.py` | Load site `config.yaml` + `deployment.yaml`, resolve per-target paths, apply SCC overrides. |
| `template_runner/stage_params.py` | Typed, validated parameters for each stage section in config. |
| `template_runner/stages.py` | Stage registry, dependency list, and `execute_stage()` dispatch to science modules. |
| `template_runner/state.py` | SQLite schema, stage status machine, command intents, resource pools. |
| `template_runner/launcher.py` | Launch local subprocesses or submit HTCondor jobs. |
| `template_runner/condor.py` | Generate `.condor.submit` files, poll `condor_q` / `condor_history`, sweep clusters on kill. |
| `template_runner/daemon.py` | Spawn and manage the detached supervisor process tree. |
| `template_runner/scheduler_control.py` | Start/stop/status the supervisor; flock and PID files under `handoff_root`. |
| `template_runner/handoff.py` | Thin wrapper around `wcs_grouping` for the `wcs_grouping` stage. |
| `template_runner/verify.py` | On-disk artifact checks, force-rerun cleanup for `ps1_process`, completion manifest I/O. |
| `template_runner/verify_worker.py` | Background artifact verification pool used by the scheduler (avoids blocking the main loop on NFS). |
| `template_runner/verify_status.py` | Exposes in-flight verify count for `status` / `progress` display. |
| `template_runner/run_context.py` | Resolve frozen config and targets from a run directory. |
| `template_runner/targets.py` | Load and normalize targets CSV (including SN catalog format). |
| `template_runner/workspace.py` | Discover live supervisors, record deployment paths, resolve `runs_root` and state DB path. |
| `template_runner/deployment.py` | Load gitignored `deployment.yaml` (paths, Gaia and Discord credentials). |
| `template_runner/logs.py` | Run directory layout, frozen input materialization, log paths, atomic JSON writes. |
| `template_runner/run_report.py` | Format `progress` and `status` output for CLI and Discord. |
| `template_runner/stage_progress.py` | Parse in-flight progress from stage log files (PS1 download, PS1 process). |
| `template_runner/notifications.py` | Discord webhook alerts on run/stage events. |
| `template_runner/discord_bot_control.py` | Start/stop/status the detached Discord bot (flock-guarded, one per workspace). |
| `template_runner/resources.py` | Resource pool bookkeeping for concurrent stage limits. |
| `wcs_grouping.py` | WCS drift measurement and template offset grouping (repo root; used by handoff wrapper). |

---

## Related documentation

| Document | Contents |
|----------|----------|
| [`template_pipeline.md`](template_pipeline.md) | Full orchestration guide: config, Condor, run lifecycle, troubleshooting, CLI flags |
| [`template_runner_architecture.md`](template_runner_architecture.md) | Deep dive: supervisor, scheduler tick, SQLite, verify, launch, recovery |
| [`stages/README.md`](stages/README.md) | Algorithm deep-dives for mapping, PS1 process, and downsample |
| [`../example/template_runner/README.md`](../example/template_runner/README.md) | Example configs and quick-start commands |
| [`README.md`](README.md) | Documentation index |

---

*Install: `pip install -e .` from the repo root registers `syndiff-template`. Activate the `syndiff` conda environment before submit so stage commands record the correct Python interpreter.*
