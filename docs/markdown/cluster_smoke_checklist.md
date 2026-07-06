# Cluster smoke test checklist

Manual smoke test for the supervised `syndiff` pipeline on a shared cluster (HTCondor + NFS). Run on the **daemon host** for the workspace — the machine where `syndiff * submit` starts the supervisor.

**Prerequisites**

- [ ] `mamba activate syndiff` and `pip install -e .` from the repo root
- [ ] `config/deployment.yaml` exists (copy from `deployment.yaml.example`) with valid `workspace_root` and `data_root` on shared storage
- [ ] Submit host has Condor client tools and NFS mounts for `workspace_root`, `data_root`, and your conda env
- [ ] Pick one smoke target: label like `s0023_c1_k3_2020ftl` (sector 23, camera 1, CCD 3, name `2020ftl` from `config/targets_example.csv`), or any target with templates already on disk for a diff-only path

Record results in a dated note (optional): `docs/cluster_smoke_YYYY-MM-DD.md`.

---

## 1. Submit end-to-end run (one target)

```bash
mamba activate syndiff

syndiff all submit \
  --site config \
  --config config/pipeline.yaml \
  --targets config/targets_smoke.csv \
  --run-id smoke_01
```

For a **single-target** smoke (`<test>` = transient name such as `2020ftl`, label `s0023_c1_k3_2020ftl`): use a one-row `targets_smoke.csv` (copy the matching row from `targets_example.csv`). Supervised `submit` schedules every enabled row in `--targets`; `--target-name` is for foreground `syndiff diff run`, not batch submit. The `--config` flag is optional when `--site` is set.

Optional shortcuts for faster smoke:

- `--stages tess_ffi_download,wcs_grouping,mapping,downsample,diff` — skip heavy `ps1_process` if convolved Zarr already exists
- `--local` — run `diff` on the submit host instead of Condor (template stages still use their normal executors)
- `--force-rerun` — ignore existing artifacts for selected stages (new run only)

**Pass criteria**

- [ ] Command exits 0 and prints monitor hints including `syndiff progress --run-id smoke_01`
- [ ] `{workspace_root}/runs/smoke_01/` created with frozen `config.yaml`, `targets.csv`, `run_meta.json`
- [ ] `syndiff daemon status --site config` shows a live supervisor PID
- [ ] `runs/latest` symlink points at `smoke_01`

---

## 1b. Diff-only smoke (`syndiff diff submit`)

Use when **template artifacts already exist** on disk (`tess_ffi`, WCS handoff JSON, downsampled templates). The scheduler verifies upstream outputs and runs only `diff` (mapping / PS1 stages are marked `external` and skipped when artifacts pass verify).

```bash
mamba activate syndiff

syndiff diff submit \
  --site config \
  --targets config/targets_smoke.csv \
  --run-id smoke_diff_01
```

Use a one-row `targets_smoke.csv` for a target with templates ready (e.g. `s0023_c1_k3_2020ftl`). Optional: `--local` to run `diff` on the submit host instead of Condor.

**Pass criteria**

- [ ] Command exits 0; `syndiff progress --run-id smoke_diff_01` shows only `diff` (and brief verify scans) for the smoke target
- [ ] `{workspace_root}/runs/smoke_diff_01/config.yaml` frozen; upstream template stages not re-queued when artifacts verify
- [ ] After completion, sections 2–6 below apply with `--run-id smoke_diff_01` (monitor, verify diff, retry diff, reconcile manifests, inspect `events/{label}/ws/`)

---

## 2. Monitor run progress

(`syndiff monitor` is not a subcommand — use **`progress`** or **`status --watch`**.)

```bash
syndiff progress --site config --run-id smoke_01

# optional: refresh every few seconds
syndiff status --site config --run-id smoke_01 --watch
```

**Pass criteria**

- [ ] Summary line shows stage counts (`pending`, `running`, `success`, etc.) and run status `running` (or terminal state when finished)
- [ ] While stages run, detail lines appear for the smoke target (e.g. `ps1_dl: …`, `down: …`, or diff log progress)
- [ ] `scan_queued` / `scan_running` appear briefly during artifact verify, then return to zero
- [ ] No persistent `stalled` status unless genuinely blocked (check `stall_reason` in `progress` output)

**Logs**

```bash
syndiff tail --site config --run-id smoke_01 \
  --target s0023_c1_k3_2020ftl --stage diff
```

---

## 3. Verify diff artifacts on disk

`verify` checks **filesystem outputs**, not SQLite schedule state.

```bash
syndiff verify --site config --run-id smoke_01 --stages diff
```

Scope to one target:

```bash
syndiff verify --site config --run-id smoke_01 \
  --scc s0023_c1_k3_2020ftl --stages diff
```

**Pass criteria**

- [ ] `[OK] <label>/diff: Frame manifest and N workspace label(s) present` after diff completes
- [ ] Before diff finishes: `[FAIL]` with message like `Missing frame manifest CSV` or `Missing ws/ under event_dir` is expected

---

## 4. Retry diff stage (force re-execution)

Re-queue a completed or failed `diff` stage for one target. Downstream reset is on by default.

```bash
syndiff retry --deployment config/deployment.yaml --run-id smoke_01 \
  --scc s0023_c1_k3_2020ftl --stage diff
```

To **force a diff re-run** (re-execute even when artifacts exist):

- Submit a new run with `--force-rerun`, or
- Use `retry` as above (scheduler bypasses artifact-skip for the selected stage)

The diff worker overwrites outputs in place; stale files under `ws/` are not removed automatically. Delete `ws/` manually if you need a fully clean workspace.

`--scc` accepts the target label (`s0023_c1_k3_2020ftl`), SCC triple (`23,1,3`), or transient name (`2020ftl`).

**Pass criteria**

- [ ] CLI prints `Queued retry for diff on <label> in run smoke_01`
- [ ] `diff` row returns to `pending` then `running` in `syndiff progress`
- [ ] `{label}/diff.log` appends new output; `diff.status.json` updates with a new `launch_token`
- [ ] After success, `verify --stages diff` is `[OK]` again

**Optional:** `--no-reset-downstream` to reopen only `diff` without touching downstream (none for `diff` on a full run).

---

## 5. Reconcile completion manifests

Backfill stable manifests under `{runs_root}/.manifests/` so future runs can skip expensive NFS scans.

```bash
syndiff reconcile-manifests --site config --run-id smoke_01
```

Quiet mode (only lines where a manifest was written):

```bash
syndiff reconcile-manifests --site config --run-id smoke_01 --quiet
```

**Pass criteria**

- [ ] Summary: `reconcile-manifests: wrote N manifest(s), M stage(s) not complete`
- [ ] For each complete stage, `{workspace_root}/runs/.manifests/{label}/{stage}.manifest.json` exists
- [ ] Re-running reconcile is idempotent (mostly `skipped` / no new writes)

---

## 6. Expected artifacts under `events/{label}/` and workspace layout

Let `LABEL` = target label (e.g. `s0023_c1_k3_2020ftl`) and `WS` = `{workspace_root}` from `deployment.yaml`.

### Per-target event directory

```text
{WS}/events/{LABEL}/
  cluster_template_job.json       # after wcs_grouping
  syndiff_ffi_frames.csv          # after wcs_grouping
  wcs_drift_template_debug.png    # optional (plots enabled)
  ps1_removed_stars.csv           # after downsample
  ws/
    templates/                    # after downsample: symlink → {data_root}/shifted_downsampled/...
    master/                       # after diff: flat FITS mirror + tess_ffi link
    <pipeline_label>/             # e.g. hp_d, ep, lc_prf_on_diffs — per diff_config.yaml
      syndiff_ffi_frames.csv      # or at event_dir root depending on stage
      shared_mask.fits
      hotpants_substamp_stars.csv
      lightcurve.csv              # when forced_photometry ran
      ...
```

Quick checks:

```bash
WS=/path/from/deployment.yaml
LABEL=s0023_c1_k3_2020ftl

test -f "$WS/events/$LABEL/cluster_template_job.json"
test -f "$WS/events/$LABEL/ps1_removed_stars.csv"
test -L "$WS/events/$LABEL/ws/templates" && test -d "$(readlink -f "$WS/events/$LABEL/ws/templates")"
test -d "$WS/events/$LABEL/ws"
test -f "$WS/events/$LABEL/ws/"*/hotpants_substamp_stars.csv 2>/dev/null || \
  test -f "$WS/events/$LABEL/shared_mask.fits"
```

(Adjust glob paths to match your `diff_config.yaml` pipeline labels.)

### Run directory (orchestration)

```text
{WS}/runs/{run_id}/
  config.yaml
  targets.csv
  run_meta.json
  summary.json
  per_target/{LABEL}/
    {stage}.log
    {stage}.status.json
    {stage}.manifest.json
    {stage}.condor.*          # when executor=condor
```

### Science caches (under `data_root`, not `events/`)

- `{data_root}/skycell_pixel_mapping/…` — mapping
- `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr` — PS1 download
- `{data_root}/shifted_downsampled/…/syndiff_template_*.fits.gz` — downsample templates

**Pass criteria**

- [ ] Handoff JSON and frame manifest exist under `events/{LABEL}/` before diff
- [ ] `ps1_removed_stars.csv` present after downsample
- [ ] After downsample (before diff): `events/{LABEL}/ws/templates` exists, is a symlink, and resolves to the downsample output directory under `{data_root}/shifted_downsampled/…` (contains `syndiff_template_*.fits*`)
- [ ] Targets that completed downsample before the templates symlink feature: run `scripts/backfill_template_symlinks.py --site config` (add `--force` to refresh stale links)
- [ ] `events/{LABEL}/ws/` contains at least one non-`master`, non-`templates` workspace label after diff
- [ ] Run logs and status sidecars exist under `runs/{run_id}/per_target/{LABEL}/`

---

## 7. Condor hold timeout (`condor_hold_timeout_s`)

Held Condor jobs (`condor_q` status `H`) are removed by the supervisor after **`condor_hold_timeout_s`** seconds (default **600**). The held timer is persisted in `{stage}.condor.hold` beside the submit file so restarts do not reset the clock.

Config (`config/pipeline.yaml`):

```yaml
scheduler:
  condor_hold_timeout_s: 600.0   # remove held jobs after N seconds
```

**Smoke observation (optional)**

- [ ] With a deliberately broken hold (e.g. invalid requirements), job enters `H` in `condor_q`
- [ ] After timeout, supervisor marks stage failed and removes the cluster; `diff.condor.log` / stage log explain the hold
- [ ] Lower value (e.g. `120.0`) speeds up this check during development

See [template_pipeline.md → HTCondor](template_pipeline.md#htcondor-integration) and [template_runner_architecture.md](template_runner_architecture.md#sqlite-and-nfs).

---

## 8. NFS and single-host SQLite expectations

`pipeline_state.sqlite` at `{workspace_root}/control/pipeline_state.sqlite` uses **WAL mode**. Only one host should run the **supervisor daemon** and heavy **CLI control** against that database.

| Expectation | Detail |
|-------------|--------|
| Daemon host | One submit host per `workspace_root`; daemon holds `control/daemon.lock` via flock |
| CLI monitoring | Run `progress`, `status`, `retry`, `verify`, etc. on the **same host** as the daemon when possible |
| Host mismatch warning | If CLI hostname ≠ daemon hostname, commands print: `SQLite WAL mode is unsafe across NFS clients; run CLI commands on the daemon host.` |
| NFS for science data | `data_root`, `events/`, and run logs **may** live on NFS; Condor workers read/write artifacts via mounts |
| Heartbeats | Supervisor liveness uses a **host-local** heartbeat file; NFS SQLite heartbeats are best-effort only |
| Safe to ignore (sometimes) | Read-only `verify` from a login node may work; **retry/kill/submit** from the wrong host risks WAL corruption or stale views |

**Pass criteria**

- [ ] Submit and monitor from the daemon host — no host-mismatch warning
- [ ] If testing from a login node, warning appears and you still route control commands to the daemon host
- [ ] `control/pipeline_state.sqlite` is not placed on a filesystem shared for multi-writer access across hosts

---

## Quick reference

| Step | Command |
|------|---------|
| Submit | `syndiff all submit --site config --targets … --run-id <id>` |
| Monitor | `syndiff progress --site config --run-id <id>` |
| Verify diff | `syndiff verify --site config --run-id <id> --stages diff` |
| Retry diff | `syndiff retry --deployment config/deployment.yaml --run-id <id> --scc <label> --stage diff` |
| Manifests | `syndiff reconcile-manifests --site config --run-id <id>` |

**Further reading:** [template_pipeline.md](template_pipeline.md), [template_runner_architecture.md](template_runner_architecture.md), [syndiff_cli.md](syndiff_cli.md), [storage_layout.md](storage_layout.md).
