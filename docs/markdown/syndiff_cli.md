# `syndiff` CLI reference

This document explains **what each `syndiff` command and internal module does**. Companion guides: [`template_pipeline.md`](template_pipeline.md) (orchestration), [`stages/README.md`](stages/README.md) (algorithms).

`syndiff` is the single console entry point (`pyproject.toml` → `syndiff_pipeline/cli.py`). Commands use a **noun/verb** structure:

```text
syndiff <noun> <verb>     # execution presets (template|diff|photometry|star)
syndiff <verb>            # monitoring, control, verify, daemon, bookkeeping
```

There is **no `all` noun**. Invoking `syndiff all ...` prints a guiding error telling you to use `template` and `diff` separately (photometry and star are separate nouns as well).

---

## Table of contents

- [Execution presets (nouns)](#execution-presets-nouns)
- [Monitoring and control verbs](#monitoring-and-control-verbs)
- [Pipeline stages (main DAG + branches)](#pipeline-stages-main-dag--branches)
- [Internal worker entry points](#internal-worker-entry-points)
- [Science modules](#science-modules)
- [Orchestration modules](#orchestration-modules)
- [Related documentation](#related-documentation)

---

## Execution presets (nouns)

| Noun | Input | Stages selected by default |
|------|-------|------------------------------|
| **`template`** | `--scc` or `--sector/--camera/--ccd` | All template stages: `tess_ffi_download`, `mapping`, `ps1_download`, `ps1_process`, `remap`, `downsample` |
| **`diff`** | `--scc` (SCC subtract) **or** `--targets` (event-oriented submit; mutually exclusive) | `["diff"]` only |
| **`photometry`** | `--targets` + photometry config | `["photometry"]` |
| **`star`** | `--star-targets` | `["star"]` |

Override selected stages with `--stages` where the noun supports it (template/diff). There is no combined preset that runs template+diff+photometry in one submit.

| Command | What it does |
|---------|--------------|
| **`syndiff template submit`** | Template building only. Input: `--scc` or `--sector/--camera/--ccd`. |
| **`syndiff template run`** | Foreground template-only debug. |
| **`syndiff diff submit`** | Diff imaging. **`--scc`** (field-mode v2, SCC-primary products) **or** `--targets`. Upstream verify: `DIFF_VERIFY_UPSTREAM` = `{tess_ffi_download, downsample}`. |
| **`syndiff diff run`** | Foreground: `--target-name` + `--targets`, **or** `--scc` / inline SCC. |
| **`syndiff photometry submit`** | Event photometry batch. Requires `--site`, `--targets`, and `--photometry-config` or `site/photometry_config.yaml`. |
| **`syndiff photometry run`** | Foreground one event (`--target-name`). See [photometry.md](photometry.md). |
| **`syndiff star submit`** | Supervised batch over `star_targets.csv`. |
| **`syndiff star run`** | Foreground single-SCC star run. See [star_lightcurves.md](star_lightcurves.md). |

### Field-mode v2 diff (SCC-only)

`scc_bootstrap` loads `field_mode_assembly.json` v3 + `mapping_grid`, writes `bookkeeping/diff/{frames.csv,diff_job.json}`, and products land under `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/`.

```bash
# After template rebuild (mapping, remap, downsample):
syndiff diff submit \
  --config config/diff_config_single_kernel.yaml \
  --scc config/scc_s20_c1_k1.csv \
  --run-id s20_v2_diff
```

See [field_geometry.md](field_geometry.md).

### Event photometry

Prefer **`syndiff photometry`** after SCC diffs exist. Diff `--targets` submits still register the `diff` stage only unless your `diff_config` includes a `kind: photometry` delegator.

```bash
syndiff photometry submit \
  --site config/ \
  --photometry-config config/photometry_config_2020ut_gepsf_lc.yaml \
  --targets config/targets_example.csv \
  --run-id my_phot
```

Common flags (`template`/`diff`/`photometry`): `--site DIR`, `--config` / `--photometry-config`, `--deployment`, `--run-id`, `--force-rerun`, `--local`.

**`syndiff star` flags** (`--site` required for both verbs):

| Flag | `submit` | `run` | Description |
|------|----------|-------|-------------|
| `--star-config` | yes | yes | Star policy YAML (default: `<site>/star_config.yaml`) |
| `--star-targets` | yes | yes | Star targets CSV |
| `--workspace-run-id` | yes | yes | Deprecated/ignored; outputs under baseline `host_star/` |
| `--config` | yes | — | Orchestrator `pipeline.yaml` |
| `--run-id` | yes | — | Unique run name |
| `--force-rerun` | yes | — | Ignore existing star artifacts for this run |
| `--local` | yes | — | `stages.star.executor: local` |
| `--target-name` | — | yes | SCC key or label from `star_targets.csv` |

**`--local` (submit only):** Rewrites frozen run config so the noun’s stage executor is `local` (`diff`, `photometry`, or `star`).

**`syndiff diff submit` verify closure:** `DIFF_VERIFY_UPSTREAM` = `{tess_ffi_download, downsample}` (`common/orchestration/spec.py`). Mapping/remap are not scanned on diff-only submits. See [`pipeline_state_machine_reference.md`](pipeline_state_machine_reference.md#diff-only-artifact-verify-closure).

---

## Monitoring and control verbs

Run `syndiff <verb> --help` for flags. These operate on the **workspace** (one `workspace_root` → one SQLite DB and supervisor).

### Run scope (`--site` vs `--run-dir` / `--run-id`)

**`--site` is only for submit/run and some monitor/verify commands** (`progress`, `status`, `verify`). It is **not** accepted by run-control or run-scoped log commands (`retry`, `pause`, `resume`, `kill`, `show`, `logs`, `tail`, `launch`).

To target a specific batch run, use either:

| Scope | Example |
|-------|---------|
| Full run directory | `--run-dir /path/to/workspace/runs/batch_no5` |
| Run ID + deployment | `--run-id batch_no5 --deployment config/deployment.yaml` |

```bash
syndiff retry \
  --deployment config/deployment.yaml \
  --run-id test_multi_hp_temp_calib_20260623 \
  --scc s0023_c2_k1_2020ghq \
  --stage diff
```

### Monitor

| Command | What it does |
|---------|--------------|
| **`syndiff progress`** | Aggregate stage counts; optional per-task detail from stage logs and progress sidecars (`downsample.progress.json`, `diff.hotpants.progress.json`, `diff.epsf.progress.json`, `diff.centroids.progress.json`, `diff.photometry.progress.json`). Condor detail lines show `condor_q` state. Use `--no-detail` for summary-only. |
| **`syndiff status`** | Per-target stage grid: `tess_dl \| map \| ps1_dl \| ps1_pr \| remap \| down \| diff` (`photometry` and `star` omitted). `--watch` for live refresh. |
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
| **`syndiff launch`** | Force-launch a ready stage once (bypasses pool `max_concurrent`). |
| **`syndiff pause`** / **`syndiff resume`** | Stop/resume dequeuing new stages. |
| **`syndiff kill`** | Cancel run; terminate local workers and Condor clusters. |

### Verification

| Command | What it does |
|---------|--------------|
| **`syndiff verify`** | Read-only on-disk artifact check (site or `--run-dir`). |
| **`syndiff reconcile-manifests`** | Backfill stable manifests under `runs/.manifests/`. |

### Bookkeeping

| Command | What it does |
|---------|--------------|
| **`syndiff bookkeeping …`** | Provenance graph ops (`reindex`, `stats`, `query`, `verify`, `gc`, `pilot`, `convolved-gate`). See [bookkeeping.md](bookkeeping.md). |

### Discord (optional)

| Command | What it does |
|---------|--------------|
| **`syndiff notify test`** | Discord preview (`--dry-run` prints locally). |

There is **no** `syndiff discord bot` CLI. When `notifications.bot.enabled` is true, the status-reply bot runs **in-process inside the supervisor daemon**.

---

## Pipeline stages (main DAG + branches)

```text
Template DAG (SCC-scoped):
tess_ffi_download → mapping → ps1_download → ps1_process → remap → downsample
                         ↘ (ps1_source: stream skips ps1_download)
                         ↘ (geometry_mode: linear skips remap)

Field-mode diff (SCC-scoped):
downsample → diff
  └─ scc_bootstrap reads templates sidecar v3 + mapping_grid
  └─ products under data_root/.../diff_{lane}/

Event photometry (independent stage):
completed diff lane ──verify──→ photometry
  └─ outputs under events/{event}/s…/phot_{run_id}/

Host-star branch (independent stage):
completed template + diff artifacts ──verify──→ star
```

Composed registry (**9** stages): six template + `diff` + `photometry` + `star`.

| Stage | Module | What it does |
|-------|--------|--------------|
| **`tess_ffi_download`** | `common/download.py` | Download TESS FFIs into `{data_root}/s{SSSS}/c{C}/k{K}/ffi/`. |
| **`mapping`** | `template_creation/.../pancakes.py` | PanCAKES TESS↔PS1 skycell mapping + Gaia. |
| **`ps1_download`** | `template_creation/.../ps1_download.py` | PS1 skycells into shared Zarr (skipped when `ps1_source: stream`). |
| **`ps1_process`** | `template_creation/.../ps1_process.py` | Convolution onto TESS grid (defaults to Condor). |
| **`remap`** | `template_creation/.../field_remap.py` | Field-mode L2–L4 drift / Exact cache (skipped in linear mode). |
| **`downsample`** | `template_creation/.../field_downsample.py` (field) or `linear_downsample.py` (linear) | L5 template store under `{data_root}/…/templates/oversampling_{N}/` (or `templates_{NAME}/`). Sidecar `field_mode_assembly.json` schema v3 + `mapping_grid`. Stage names `templates` / `tmpl` are **rejected** in strict config parse; legacy SQLite rows may still display as aliases. |
| **`diff`** | `difference_imaging/.../execute.py` | Config-driven Hotpants / kernel / ePSF stack. Field mode: `scc_bootstrap`, SCC-primary `diff_{lane}/`. |
| **`photometry`** | `photometry/runner.py` | Astrometry + forced photometry on SCC diffs → `phot_{run_id}/`. |
| **`star`** | `star/runner.py` | Host-star light curves from diff side products. |

**Product path vs stage name:** on-disk directory is still `templates/…`; the scheduler stage is **`downsample`**.

**Executors:** `mapping`, `ps1_process`, `remap`, `downsample`, `diff`, `photometry`, and `star` can run on HTCondor (per `pipeline.yaml`); network stages are local on the submit host.

---

## Internal worker entry points

| Script | Invocation | What it does |
|--------|------------|--------------|
| **`common/orchestration/run_stage.py`** | `python -m syndiff_pipeline.common.orchestration.run_stage --run-id … --stage …` | Single target + stage worker. |
| **`common/orchestration/scheduler.py`** | `--daemon --deployment …` | Supervisor loop: verify, promote, launch, reconcile. |
| **`common/orchestration/condor_wrapper.sh`** | HTCondor `executable` | Conda activation + `run_stage.py`. |
| **`template_creation/.../discord_bot.py`** | In-process inside supervisor | On-demand status replies when bot enabled. |

---

## Science modules

Template and diff science code lives under `template_creation/processing/` and `difference_imaging/stages/`. Photometry and star packages are `syndiff_pipeline/photometry/` and `syndiff_pipeline/star/`. See [`stages/README.md`](stages/README.md).

---

## Orchestration modules

| Module | Role |
|--------|------|
| `syndiff_pipeline/cli.py` | Noun/verb CLI; delegates to orch / photometry / star CLIs; `all` prints removal error. |
| `photometry/cli.py` | `syndiff photometry submit\|run`. |
| `star/cli.py` | `syndiff star submit\|run`. |
| `common/orchestration/cli.py` | Monitoring, control, verify, daemon; `preset_stages()` (`template` → six template stages, `diff` → `["diff"]`). |
| `common/orchestration/spec.py` | `StageSpec` / `PipelineSpec`; `DIFF_VERIFY_UPSTREAM = {tess_ffi_download, downsample}`. |
| `pipeline_spec.py` | Composed registry: template + diff + photometry + star; `STATUS_GRID_STAGES` (7 columns). |
| `difference_imaging/orchestration/scc_bootstrap.py` | Field-mode diff handoff. |
| `difference_imaging/orchestration/stages.py` | Diff registry (`diff`; deps=`downsample`). |
| `photometry/orchestration/stages.py` | Photometry registry. |
| `star/orchestration/stages.py` | Star registry + verifier. |
| `common/orchestration/state.py` | SQLite schema, status machine. |
| `common/orchestration/scheduler.py` | Supervisor tick. |
| `common/orchestration/condor.py` | Submit / poll / held jobs. |
| `common/scc_paths.py` | SCC + event path helpers. |
| `template_creation/orchestration/stages.py` | Template registry (`…`, `remap`, `downsample`). |
| `template_creation/orchestration/verify.py` | On-disk verifiers + manifests. |
| `template_creation/orchestration/runner_config.py` | YAML load, path resolution. |

---

## Related documentation

| Document | Contents |
|----------|----------|
| [`photometry.md`](photometry.md) | Event photometry quick start |
| [`field_geometry.md`](field_geometry.md) | MappingGrid, field templates, rebuild runbook |
| [`storage_layout.md`](storage_layout.md) | `diff_{lane}/`, `bookkeeping/diff/`, `phot_{run_id}/` |
| [`bookkeeping.md`](bookkeeping.md) | Provenance CLI |
| [`template_runner_architecture.md`](template_runner_architecture.md) | Scheduler, verify, recovery |
| [`pipeline_state_machine_reference.md`](pipeline_state_machine_reference.md) | SQLite status transitions |
| [`../config/`](../../config/) | Site config examples |
| [`README.md`](README.md) | Documentation index |

---

*Install: `pip install -e .` registers `syndiff`. Activate the `syndiff` conda environment before submit so stage commands record the correct Python interpreter.*
