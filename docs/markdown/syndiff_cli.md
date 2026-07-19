# `syndiff` CLI reference

This document explains **what each `syndiff` command and internal module does**. It is a script-oriented companion to the orchestration guide in [`template_pipeline.md`](template_pipeline.md) and the algorithm deep-dives in [`stages/`](stages/README.md).

`syndiff` is the single console entry point (`pyproject.toml` → `syndiff_pipeline/cli.py`). Commands use a **noun/verb** structure:

```text
syndiff <noun> <verb>     # execution presets (template|diff|star)
syndiff <verb>            # monitoring, control, verify, daemon
```

There is **no `all` noun**. Invoking `syndiff all ...` prints a guiding error:
`"The 'all' preset was removed. Use 'syndiff template submit|run ...' and
'syndiff diff submit|run ...' separately."`

---

## Table of contents

- [Execution presets (nouns)](#execution-presets-nouns)
- [Monitoring and control verbs](#monitoring-and-control-verbs)
- [Pipeline stages (main DAG + star branch)](#pipeline-stages-main-dag--star-branch)
- [Internal worker entry points](#internal-worker-entry-points)
- [Science modules](#science-modules)
- [Orchestration modules](#orchestration-modules)
- [Related documentation](#related-documentation)

---

## Execution presets (nouns)

`template` and `diff` are **separate DAGs with separate input formats** — `template` takes an SCC-only CSV (`--scc`), `diff` takes an event targets CSV (`--targets`). There is no combined preset that runs both from one command.

| Command | Stages selected | What it does |
|---------|-----------------|--------------|
| **`syndiff template submit`** | `tess_ffi_download`, `mapping`, `ps1_download`, `ps1_process`, `templates` (default; override with `--stages`) | Template building only, one SCC (or SCC list) at a time. Input: `--scc sccs.csv` or `--sector/--camera/--ccd`. |
| **`syndiff template run`** | Template stages | Foreground template-only debug. |
| **`syndiff diff submit`** | `diff` **only** by default — `bind` is *not* auto-included; pass `--stages bind,diff` explicitly | Diff (and, if requested, `bind`); upstream template stages start `external` and are skipped when artifacts verify on disk. Input: `--targets targets.csv`. |
| **`syndiff diff run`** | `diff` (one target, in-process; never runs `bind`) | Foreground diff for `--target-name` (no daemon/DB). Supports `--validate-only`. Requires the event's `bind` handoff to already exist on disk. |
| **`syndiff star submit`** | `star` | Supervised batch over enabled rows in `star_targets.csv` (Condor or `--local`). |
| **`syndiff star run`** | — | Foreground single-SCC star run (`--target-name` required; no daemon/DB). See [star_lightcurves.md](star_lightcurves.md). |

**Important**: `syndiff diff submit --targets targets.csv` with no `--stages`
selects only `["diff"]` (`preset_stages("diff")` in
`common/orchestration/cli.py`). `bind` is then marked `skipped` (reason
`not_selected`) rather than run, because `DIFF_VERIFY_UPSTREAM`
(`common/orchestration/spec.py`) covers only `tess_ffi_download` and
`templates` — not `bind`. The first time you diff a new event, always pass
**`--stages bind,diff`** so `bind` actually writes `event_job.json` +
`frames.csv`; otherwise `diff` fails at runtime if that handoff doesn't
already exist.

Common flags (`template`/`diff` presets): `--site DIR`, `--config`,
`--deployment`, `--scc`/`--targets`, `--run-id`, `--stages` (override preset),
`--force-rerun`, and `--local`. Supervised submit reads `pipeline.yaml` and
the `diff_config:` it names. Foreground `syndiff diff run --site DIR` always
reads `<site>/diff_config.yaml`; to choose another diff policy, omit `--site`
and pass `--config <diff.yaml> --deployment <deployment.yaml>`. On
`diff submit`, `--local` patches the frozen config so
`stages.diff.executor` is local.

**`syndiff star` flags** (`--site` required for both verbs):

| Flag | `submit` | `run` | Description |
|------|----------|-------|-------------|
| `--star-config` | yes | yes | Star policy YAML (default: `<site>/star_config.yaml`) |
| `--star-targets` | yes | yes | Star targets CSV (default: `<site>/star_targets_example.csv`) |
| `--workspace-run-id` | yes | yes | Deprecated and ignored; star outputs always land in `{baseline_ws}/host_star/` |
| `--config` | yes | — | Orchestrator `pipeline.yaml` (default: `<site>/pipeline.yaml`) |
| `--run-id` | yes | — | Unique run name (must not already exist) |
| `--force-rerun` | yes | — | Ignore existing star artifacts for this run |
| `--local` | yes | — | Patch frozen config so `stages.star.executor` is `local` |
| `--target-name` | — | yes | SCC key (`20/3/2`) or full label from `star_targets.csv` |
| `--targets`, `--stars-file`, `--baseline-*`, `--cutout-size`, … | — | yes | Foreground overrides; see `syndiff star run --help` |

`star submit` materializes the run directory (frozen `star_config.yaml` + targets), registers the `star` stage in SQLite, and ensures the supervisor daemon — same pattern as `template submit`.

**`--local` (submit only):** For `diff`, the CLI rewrites frozen
`config.yaml` to set `stages.diff.executor: local`; for `star`, it sets
`stages.star.executor: local`. Other selected stages keep their configured
executors.

**`syndiff diff submit` verify closure:** Only `tess_ffi_download` and `templates` are artifact-verified on disk (`DIFF_VERIFY_UPSTREAM`). `bind`, `mapping`, `ps1_download`, and `ps1_process` are marked **n/a** without scanning (see the callout above — this means `bind` needs `--stages bind,diff` to actually run). See [`pipeline_state_machine_reference.md`](pipeline_state_machine_reference.md#diff-only-artifact-verify-closure).

---

## Monitoring and control verbs

Run `syndiff <verb> --help` for flags. These operate on the **workspace** (one `workspace_root` → one SQLite DB and supervisor).

### Run scope (`--site` vs `--run-dir` / `--run-id`)

**`--site` is only for submit/run and some monitor/verify commands** (`progress`, `status`, `verify`). It is **not** accepted by run-control or run-scoped log commands (`retry`, `pause`, `resume`, `kill`, `show`, `logs`, `tail`).

To target a specific batch run, use either:

| Scope | Example |
|-------|---------|
| Full run directory | `--run-dir /path/to/workspace/runs/batch_no5` |
| Run ID + deployment | `--run-id batch_no5 --deployment config/deployment.yaml` |

`--deployment` is optional when exactly one supervisor is already running (auto-discovered). `--run-id` alone resolves `runs/{run_id}/` under the workspace in `deployment.yaml`.

```bash
# Retry one target's diff stage (run control — no --site)
syndiff retry \
  --deployment config/deployment.yaml \
  --run-id test_multi_hp_temp_calib_20260623 \
  --scc s0023_c2_k1_2020ghq \
  --stage diff
```

### Monitor

| Command | What it does |
|---------|--------------|
| **`syndiff progress`** | Aggregate stage counts; optional per-task detail from stage logs and progress sidecars (`downsample.progress.json` — filename unchanged by the `templates` rename — `diff.hotpants.progress.json`, `diff.epsf.progress.json`, `diff.centroids.progress.json`, `diff.photometry.progress.json` beside `diff.log`). For Condor stages, detail lines also show queue state from `condor_q` (`condor idle cN.0`, `condor running cN.0`, `condor held cN.0`, or `condor unsubmitted` before a cluster id is recorded). Use `--no-detail` for summary-only output. |
| **`syndiff status`** | Per-target stage grid: `tess_dl | map | ps1_dl | ps1_pr | remap | down | diff` (`bind` and `star` omitted). `--watch` for live refresh. |
| **`syndiff show`** | Dump `run_meta.json`. |
| **`syndiff logs`** / **`syndiff tail`** | Daemon log or `per_target/<label>/<stage>.log`. |

### Workspace

| Command | What it does |
|---------|--------------|
| **`syndiff runs`** | List recent runs from SQLite. |
| **`syndiff active`** | Running/stalled runs + supervisor health. |
| **`syndiff daemon start\|stop\|status`** | Supervisor lifecycle (normally auto-started by `submit`). Ownership is recorded in `control/daemon.lease` (cross-host source of truth). `daemon stop` works from any host via `control/daemon.stop`. |

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

There is **no** `syndiff discord bot` CLI. When `notifications.bot.enabled` is true and token/channel are configured, the status-reply bot runs **in-process inside the supervisor daemon** (started on `submit` / `daemon start`). Check `syndiff daemon status` (`discord_bot.expected_in_process`). Legacy detached bot processes are cleaned up on supervisor start.

---

## Pipeline stages (template DAG + diff DAG + star branch)

```text
Template DAG (SCC-scoped, syndiff template submit --scc sccs.csv):
tess_ffi_download → mapping → ps1_download → ps1_process → templates

Diff DAG (event-scoped, syndiff diff submit --targets targets.csv):
bind → diff

completed template + diff artifacts ──verify──→ star
```

The composed registry contains eight stages: five template + `bind` + `diff`
+ `star`. There is **no** `all` noun selecting all of them at once — `template`
and `diff` are separate submits with separate SQLite run rows (though
`state.py::create_run` always materializes all eight stage rows per target
regardless of which noun submitted the run; see
[`pipeline_state_machine_reference.md`](pipeline_state_machine_reference.md)).
`star` is submitted separately with `star_targets.csv`; its stage spec has no
SQLite dependency edges because it validates the existing event artifacts
when it starts.

| Stage | Module | What it does |
|-------|--------|--------------|
| **`tess_ffi_download`** | `common/download.py` | Download TESS FFIs for the target SCC into `{data_root}/s{SSSS}/c{C}/k{K}/ffi/`. |
| **`mapping`** | `template_creation/.../pancakes.py` + `.../scc_reference_ffi.py` | SCC-scoped reference-FFI chooser, then PanCAKES TESS↔PS1 skycell mapping. |
| **`ps1_download`** | `template_creation/.../ps1_download.py` | PS1 skycells into shared Zarr (skipped when `ps1_source: stream`). |
| **`ps1_process`** | `template_creation/.../ps1_process.py` | Convolution onto TESS grid (defaults to Condor). |
| **`templates`** (config key/legacy alias: `downsample`) | `template_creation/.../downsample.py` (+ `field_downsample.py`) | Multi-offset template store under `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/`; linear mode also writes `ps1_removed_stars.csv` in `event_dir`. |
| **`bind`** | `difference_imaging/.../bind.py` via `common/wcs_grouping.py` | Event WCS drift, template groups; writes handoff (`event_job.json` + `frames.csv`) under `{workspace_root}/events/{event_name}/s{SSSS}_c{C}_k{K}/`. |
| **`diff`** | `difference_imaging/.../execute.py` | Config-driven Hotpants → photometry; outputs in `events/{event_name}/{scc_label}/ws/`. |
| **`star`** | `star/runner.py` | Host-star mini-templates, kernel-reused stamps, and light curves under `{baseline_ws}/host_star/`. |

Legacy stage-name aliases accepted by `resolve_stage_name`: `downsample` → `templates`, `down` → `templates`, `wcs_grouping` → `bind`, `wcs` → `bind`.

**Executors**: `mapping`, `ps1_process`, `diff`, and `star` can run on
HTCondor; other stages are local subprocesses on the submit host.

---

## Internal worker entry points

| Script | Invocation | What it does |
|--------|------------|--------------|
| **`common/orchestration/run_stage.py`** | `python -m syndiff_pipeline.common.orchestration.run_stage --run-id … --stage …` | Single target + stage worker. Writes log + `*.status.json`, runs spec-driven `execute_stage()`, writes manifests. |
| **`common/orchestration/scheduler.py`** | `--daemon --deployment …` | Supervisor loop: verify, promote, launch, reconcile. |
| **`common/orchestration/condor_wrapper.sh`** | HTCondor `executable` | Parameterized conda activation + `exec` of `run_stage.py`. |
| **`template_creation/.../discord_bot.py`** | In-process inside supervisor daemon | On-demand status replies when `notifications.bot.enabled`. |

---

## Science modules

Template and diff science code lives under `template_creation/processing/` and `difference_imaging/stages/`. Several modules retain standalone `__main__` entry points for debugging outside the scheduler — see [`stages/README.md`](stages/README.md).

---

## Orchestration modules

| Module | Role |
|--------|------|
| `syndiff_pipeline/cli.py` | Noun/verb CLI entry; delegates to `common/orchestration/cli.py` and `star/cli.py`; `all` prints a removal error. |
| `star/cli.py` | `syndiff star submit|run` — host-star light curves. |
| `common/orchestration/cli.py` | Monitoring, control, verify, daemon verbs; `preset_stages()` (`template` → 5 stages, `diff` → `["diff"]` only). |
| `common/orchestration/spec.py` | `StageSpec` / `PipelineSpec`; `DIFF_VERIFY_UPSTREAM = {tess_ffi_download, templates}`; legacy stage-name aliases. |
| `pipeline_spec.py` | Composed registry: five template stages + `bind` + `diff` + independent `star`. |
| `common/orchestration/state.py` | SQLite schema, status machine, promotion, attempts/backoff. |
| `common/orchestration/scheduler.py` | Supervisor tick, verify scheduling, launch, stall detection. |
| `common/orchestration/condor.py` | Submit, batched poll, held-job handling. |
| `common/orchestration/launcher.py` | Local `Popen` vs Condor submit. |
| `common/scc_paths.py` | SCC-scoped + event-scoped path helpers. |
| `template_creation/orchestration/stages.py` | Template stage registry (`tess_ffi_download`, `mapping`, `ps1_download`, `ps1_process`, `templates`). |
| `template_creation/processing/scc_reference_ffi.py` | SCC-scoped mapping reference-FFI chooser + bookkeeping. |
| `difference_imaging/orchestration/stages.py` | Diff DAG registry (`bind`, `diff`). |
| `difference_imaging/orchestration/bind.py` | `bind` stage implementation. |
| `star/orchestration/stages.py` | Independent `star` stage registry and artifact verifier. |
| `difference_imaging/orchestration/site_config.py` | Resolve/freeze per-target diff config from site folder. |
| `template_creation/orchestration/verify.py` | On-disk verifiers + completion manifests. |
| `template_creation/orchestration/runner_config.py` | YAML load, `event_dir` = `events/{event_name}/{scc_label}/`, path resolution. |

---

## Related documentation

| Document | Contents |
|----------|----------|
| [`template_pipeline.md`](template_pipeline.md) | User guide: config, Condor, run lifecycle, troubleshooting |
| [`template_runner_architecture.md`](template_runner_architecture.md) | Maintainer deep dive: scheduler, verify, recovery |
| [`pipeline_state_machine_reference.md`](pipeline_state_machine_reference.md) | SQLite status transition matrix |
| [`../config/`](../../config/) | Site config examples |
| [`README.md`](README.md) | Documentation index |

---

*Install: `pip install -e .` registers `syndiff`. Activate the `syndiff` conda environment before submit so stage commands record the correct Python interpreter.*
