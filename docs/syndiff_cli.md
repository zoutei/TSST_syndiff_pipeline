# `syndiff` CLI reference

This document explains **what each `syndiff` command and internal module does**. It is a script-oriented companion to the orchestration guide in [`template_pipeline.md`](template_pipeline.md) and the algorithm deep-dives in [`stages/`](stages/README.md).

`syndiff` is the single console entry point (`pyproject.toml` → `syndiff_pipeline/cli.py`). Commands use a **noun/verb** structure:

```text
syndiff <noun> <verb>     # execution presets
syndiff <verb>            # monitoring, control, verify, daemon
```

---

## Table of contents

- [Execution presets (nouns)](#execution-presets-nouns)
- [Monitoring and control verbs](#monitoring-and-control-verbs)
- [Pipeline stages (7-stage DAG)](#pipeline-stages-7-stage-dag)
- [Internal worker entry points](#internal-worker-entry-points)
- [Science modules](#science-modules)
- [Orchestration modules](#orchestration-modules)
- [Related documentation](#related-documentation)

---

## Execution presets (nouns)

| Command | Stages selected | What it does |
|---------|-----------------|--------------|
| **`syndiff all submit`** | All seven stages | End-to-end: template build then diff per target. |
| **`syndiff all run`** | All seven stages | Foreground debug loop (blocks until terminal). |
| **`syndiff template submit`** | `tess_ffi_download` … `downsample` | Template building only. |
| **`syndiff template run`** | Template stages | Foreground template-only debug. |
| **`syndiff diff submit`** | `diff` | Diff only; upstream template stages start `external` and are skipped when artifacts verify on disk. |
| **`syndiff diff run`** | `diff` (one target) | Foreground diff for `--target-name` (no daemon/DB). Supports `--validate-only`. |

Common flags: `--site DIR` (loads `pipeline.yaml` + `diff_config.yaml` + `deployment.yaml`; `pipeline.yaml` may set `diff_config: diff_config.yaml`), `--config`, `--deployment`, `--targets`, `--run-id`, `--stages` (override preset), `--force-rerun`, `--local` (on `syndiff diff submit` or `syndiff all submit`: patches the frozen run config so `stages.diff.executor` is `local` instead of Condor).

**`--local` (submit only):** After materializing the run directory, the CLI rewrites frozen `config.yaml` to set `stages.diff.executor: local`. Use this for cluster smoke tests or when Condor is unavailable; template stages still follow their normal executors.

**`syndiff diff submit` verify closure:** Only `tess_ffi_download`, `wcs_grouping`, and `downsample` are artifact-verified on disk (`DIFF_VERIFY_UPSTREAM`). `mapping`, `ps1_download`, and `ps1_process` are marked **n/a** without scanning. See [`pipeline_state_machine_reference.md`](pipeline_state_machine_reference.md#diff-only-artifact-verify-closure).

---

## Monitoring and control verbs

Run `syndiff <verb> --help` for flags. These operate on the **workspace** (one `workspace_root` → one SQLite DB and supervisor).

### Monitor

| Command | What it does |
|---------|--------------|
| **`syndiff progress`** | Aggregate stage counts; optional per-task detail from stage logs / `downsample.progress.json`. |
| **`syndiff status`** | Per-target stage grid (`tess_dl`, `wcs`, `map`, `ps1_dl`, `ps1_pr`, `down`, `diff`). `--watch` for live refresh. |
| **`syndiff show`** | Dump `run_meta.json`. |
| **`syndiff logs`** / **`syndiff tail`** | Daemon log or `per_target/<label>/<stage>.log`. |

### Workspace

| Command | What it does |
|---------|--------------|
| **`syndiff runs`** | List recent runs from SQLite. |
| **`syndiff active`** | Running/stalled runs + supervisor health. |
| **`syndiff daemon start\|stop\|status`** | Supervisor lifecycle (normally auto-started by `submit`). |

### Run control

Insert **command intents** into SQLite; the supervisor applies them on the next tick.

| Command | What it does |
|---------|--------------|
| **`syndiff retry`** | Re-queue failed/canceled stages (bulk or `--scc` + `--stage`). |
| **`syndiff pause`** / **`syndiff resume`** | Stop/resume dequeuing new stages. |
| **`syndiff kill`** | Cancel run; terminate local workers and Condor clusters. |

### Verification

| Command | What it does |
|---------|--------------|
| **`syndiff verify`** | Read-only on-disk artifact check (site or `--run-dir`). |
| **`syndiff reconcile-manifests`** | Backfill stable manifests under `runs/.manifests/`. |

### Discord (optional)

| Command | What it does |
|---------|--------------|
| **`syndiff notify test`** | Discord preview (`--dry-run` prints locally). |
| **`syndiff discord bot`** | Foreground status-reply bot (normally auto-started). |

---

## Pipeline stages (7-stage DAG)

```text
tess_ffi_download → wcs_grouping → mapping → ps1_download → ps1_process → downsample → diff
                                      └──────────────── downsample also needs wcs_grouping
```

| Stage | Module | What it does |
|-------|--------|--------------|
| **`tess_ffi_download`** | `common/download.py` | Download TESS FFIs for the target SCC. |
| **`wcs_grouping`** | `common/wcs_grouping.py` via `handoff.py` | WCS drift, template groups; writes handoff under `{workspace_root}/events/{label}/`. |
| **`mapping`** | `template_creation/.../pancakes.py` | PanCAKES TESS↔PS1 skycell mapping. |
| **`ps1_download`** | `template_creation/.../ps1_download.py` | PS1 skycells into shared Zarr (skipped when `ps1_source: stream`). |
| **`ps1_process`** | `template_creation/.../ps1_process.py` | Convolution onto TESS grid (defaults to Condor). |
| **`downsample`** | `template_creation/.../downsample.py` | Multi-offset template FITS + `ps1_removed_stars.csv` in `event_dir`. |
| **`diff`** | `difference_imaging/.../execute.py` | Config-driven Hotpants → photometry; outputs in `events/{label}/ws/`. |

**Executors**: `mapping`, `ps1_process`, and `diff` can run on HTCondor; other stages are local subprocesses on the submit host.

---

## Internal worker entry points

| Script | Invocation | What it does |
|--------|------------|--------------|
| **`common/orchestration/run_stage.py`** | `python -m syndiff_pipeline.common.orchestration.run_stage --run-id … --stage …` | Single target + stage worker. Writes log + `*.status.json`, runs spec-driven `execute_stage()`, writes manifests. |
| **`common/orchestration/scheduler.py`** | `--daemon --deployment …` | Supervisor loop: verify, promote, launch, reconcile. |
| **`common/orchestration/condor_wrapper.sh`** | HTCondor `executable` | Parameterized conda activation + `exec` of `run_stage.py`. |
| **`template_creation/.../discord_bot.py`** | `syndiff discord bot` | On-demand status replies. |

---

## Science modules

Template and diff science code lives under `template_creation/processing/` and `difference_imaging/stages/`. Several modules retain standalone `__main__` entry points for debugging outside the scheduler — see [`stages/README.md`](stages/README.md).

---

## Orchestration modules

| Module | Role |
|--------|------|
| `syndiff_pipeline/cli.py` | Noun/verb CLI entry; delegates to `common/orchestration/cli.py`. |
| `common/orchestration/cli.py` | Monitoring, control, verify, daemon verbs. |
| `common/orchestration/spec.py` | `StageSpec` / `PipelineSpec`. |
| `pipeline_spec.py` | Composed 7-stage SynDiff DAG. |
| `common/orchestration/state.py` | SQLite schema, status machine, promotion, attempts/backoff. |
| `common/orchestration/scheduler.py` | Supervisor tick, verify scheduling, launch, stall detection. |
| `common/orchestration/condor.py` | Submit, batched poll, held-job handling. |
| `common/orchestration/launcher.py` | Local `Popen` vs Condor submit. |
| `template_creation/orchestration/stages.py` | Template stage registry. |
| `difference_imaging/orchestration/stages.py` | `diff` stage registry. |
| `difference_imaging/orchestration/site_config.py` | Resolve/freeze per-target diff config from site folder. |
| `template_creation/orchestration/verify.py` | On-disk verifiers + completion manifests. |
| `template_creation/orchestration/runner_config.py` | YAML load, `event_dir` = `events/{label}/`, path resolution. |

---

## Related documentation

| Document | Contents |
|----------|----------|
| [`template_pipeline.md`](template_pipeline.md) | User guide: config, Condor, run lifecycle, troubleshooting |
| [`template_runner_architecture.md`](template_runner_architecture.md) | Maintainer deep dive: scheduler, verify, recovery |
| [`pipeline_state_machine_reference.md`](pipeline_state_machine_reference.md) | SQLite status transition matrix |
| [`../config/`](../config/) | Site config examples |
| [`README.md`](README.md) | Documentation index |

---

*Install: `pip install -e .` registers `syndiff`. Activate the `syndiff` conda environment before submit so stage commands record the correct Python interpreter.*
