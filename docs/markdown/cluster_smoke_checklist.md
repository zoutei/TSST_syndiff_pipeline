# Cluster smoke test checklist

Manual smoke test for the supervised `syndiff` pipeline on a shared cluster (HTCondor + NFS). Run on the **daemon host** for the workspace — the machine where `syndiff * submit` starts the supervisor.

**Prerequisites**

- [ ] `mamba activate syndiff` and `pip install -e .` from the repo root
- [ ] `config/deployment.yaml` exists (copy from `deployment.yaml.example`) with valid `workspace_root` and `data_root` on shared storage
- [ ] Submit host has Condor client tools and NFS mounts for `workspace_root`, `data_root`, and your conda env
- [ ] Pick one smoke SCC (e.g. sector 23, camera 1, CCD 3) and one smoke event (transient name such as `2020ftl` from `config/targets_example.csv`, label `s0023_c1_k3_2020ftl`)

Record results in a dated note (optional): `docs/cluster_smoke_YYYY-MM-DD.md`.

There is **no combined end-to-end preset** — `syndiff template` (SCC-scoped) and `syndiff diff` (event-scoped) are separate submits. Run template first, then diff.

---

## 1. Submit template run (one SCC)

```bash
mamba activate syndiff

syndiff template submit \
  --site config \
  --config config/pipeline.yaml \
  --scc config/scc_smoke.csv \
  --run-id smoke_template_01
```

Use a one-row `scc_smoke.csv` (`sector,camera,ccd,enabled` — copy the SCC from the matching row of `targets_example.csv`; template submit does **not** take `--targets`). Supervised `submit` schedules every enabled row in `--scc`.

Optional shortcuts for faster smoke:

- `--stages ps1_process,downsample` — skip earlier stages if mapping/Zarr already exist
- `--force-rerun` — ignore existing artifacts for selected stages (new run only)

**Pass criteria**

- [ ] Command exits 0 and prints monitor hints including `syndiff progress --run-id smoke_template_01`
- [ ] `{workspace_root}/runs/smoke_template_01/` created with frozen `config.yaml`, `targets.csv` (normalized SCC CSV), `run_meta.json`
- [ ] `syndiff daemon status --site config` shows a live supervisor PID
- [ ] `runs/latest` symlink points at `smoke_template_01`
- [ ] `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_1/` populated after `downsample` succeeds

---

## 1b. Diff smoke (`syndiff diff submit --stages bind,diff`)

Use once **template artifacts already exist** on disk for the smoke SCC (FFIs, mapping, template store). This submits the two-stage diff DAG for one event: `bind` (event WCS grouping + handoff) then `diff`.

```bash
mamba activate syndiff

syndiff diff submit \
  --site config \
  --targets config/targets_smoke.csv \
  --stages bind,diff \
  --run-id smoke_diff_01
```

**Important — always pass `--stages bind,diff` explicitly.** The bare `syndiff diff submit --targets ...` (no `--stages`) selects only `["diff"]`; `bind` is then marked `skipped` (not_selected) and never runs, so `diff` fails at runtime if `event_job.json` doesn't already exist for this event. See [template_pipeline.md → Overview](template_pipeline.md#overview) and [pipeline_state_machine_reference.md §3.2](pipeline_state_machine_reference.md).

Use a one-row `targets_smoke.csv` for the smoke event (e.g. `s0023_c1_k3_2020ftl`). Optional: `--local` to run `diff` on the submit host instead of Condor.

**Pass criteria**

- [ ] Command exits 0; `syndiff progress --run-id smoke_diff_01` shows `bind` then `diff` (and brief verify scans) for the smoke target
- [ ] `{workspace_root}/runs/smoke_diff_01/config.yaml` frozen; `mapping`/`ps1_download`/`ps1_process` marked n/a (not re-queued)
- [ ] `bind` succeeds and writes `event_job.json` + `frames.csv` under `events/{event_name}/s{SSSS}_c{C}_k{K}/`
- [ ] After completion, sections 2–6 below apply with `--run-id smoke_diff_01` (monitor, verify diff, retry diff, reconcile manifests, inspect `events/{event_name}/{scc_label}/ws/`)

---

## 2. Monitor run progress

(`syndiff monitor` is not a subcommand — use **`progress`** or **`status --watch`**.)

```bash
syndiff progress --site config --run-id smoke_diff_01

# optional: refresh every few seconds
syndiff status --site config --run-id smoke_diff_01 --watch
```

**Pass criteria**

- [ ] Summary line shows stage counts (`pending`, `running`, `success`, etc.) and run status `running` (or terminal state when finished)
- [ ] While stages run, detail lines appear for the smoke target (e.g. `ps1_dl: …`, `down: …`, or diff log progress)
- [ ] `scan_queued` / `scan_running` appear briefly during artifact verify, then return to zero
- [ ] No persistent `stalled` status unless genuinely blocked (check `stall_reason` in `progress` output)

**Logs**

```bash
syndiff tail --site config --run-id smoke_diff_01 \
  --target s0023_c1_k3_2020ftl --stage diff
```

---

## 3. Verify diff artifacts on disk

`verify` checks **filesystem outputs**, not SQLite schedule state.

```bash
syndiff verify --site config --run-id smoke_diff_01 --stages diff
```

Scope to one target:

```bash
syndiff verify --site config --run-id smoke_diff_01 \
  --scc s0023_c1_k3_2020ftl --stages diff
```

**Pass criteria**

- [ ] `[OK] <label>/diff: Frame manifest and N workspace label(s) present` after diff completes
- [ ] Before diff finishes: `[FAIL]` with message like `Missing frame manifest CSV` or `Missing ws/ under event_dir` is expected

---

## 4. Retry diff stage (force re-execution)

Re-queue a completed or failed `diff` stage for one target. Downstream reset is on by default.

```bash
syndiff retry --deployment config/deployment.yaml --run-id smoke_diff_01 \
  --scc s0023_c1_k3_2020ftl --stage diff
```

To **force a diff re-run** (re-execute even when artifacts exist):

- Submit a new run with `--force-rerun`, or
- Use `retry` as above (scheduler bypasses artifact-skip for the selected stage)

The diff worker overwrites outputs in place; stale files under `ws/` are not removed automatically. Delete `ws/` manually if you need a fully clean workspace.

`--scc` accepts the target label (`s0023_c1_k3_2020ftl`), SCC triple (`23,1,3`), or transient name (`2020ftl`).

**Pass criteria**

- [ ] CLI prints `Queued retry for diff on <label> in run smoke_diff_01`
- [ ] `diff` row returns to `pending` then `running` in `syndiff progress`
- [ ] `{label}/diff.log` appends new output; `diff.status.json` updates with a new `launch_token`
- [ ] After success, `verify --stages diff` is `[OK]` again

**Optional:** `--no-reset-downstream` to reopen only `diff` without touching downstream (none for `diff` on a full run).

---

## 5. Reconcile completion manifests

Backfill stable manifests under `{runs_root}/.manifests/` so future runs can skip expensive NFS scans.

```bash
syndiff reconcile-manifests --site config --run-id smoke_diff_01
```

Quiet mode (only lines where a manifest was written):

```bash
syndiff reconcile-manifests --site config --run-id smoke_diff_01 --quiet
```

**Pass criteria**

- [ ] Summary: `reconcile-manifests: wrote N manifest(s), M stage(s) not complete`
- [ ] For each complete stage, `{workspace_root}/runs/.manifests/{label}/{stage}.manifest.json` exists
- [ ] Re-running reconcile is idempotent (mostly `skipped` / no new writes)

---

## 6. Expected artifacts under `events/{event_name}/{scc_label}/` and workspace layout

Let `EVENT` = event name (e.g. `2020ftl`), `SCC` = SCC label (e.g. `s0023_c1_k3`), `LABEL` = full target label (`s0023_c1_k3_2020ftl`, used for `per_target/` and SQLite), and `WS` = `{workspace_root}` from `deployment.yaml`.

### Per-event, per-SCC directory

```text
{WS}/events/{EVENT}/{SCC}/
  event_job.json                  # after bind (legacy name: cluster_template_job.json)
  frames.csv                      # after bind (legacy name: syndiff_ffi_frames.csv)
  wcs_drift_template_debug.png    # optional (plots enabled)
  ps1_removed_stars.csv           # written by `downsample` (linear geometry_mode)
  ws/
    master/                       # after diff: flat FITS mirror + tess_ffi link
    <pipeline_label>/             # e.g. hp_d, ep, lc_prf_on_diffs — per diff_config.yaml
      shared_mask.fits
      hotpants_substamp_stars.csv
      lightcurve.csv              # when forced_photometry ran
      ...
```

There is **no** `ws/templates` symlink — `diff` resolves templates directly from `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/`.

Quick checks:

```bash
WS=/path/from/deployment.yaml
DATA=/path/from/deployment.yaml   # data_root
EVENT=2020ftl
SCC=s0023_c1_k3                   # event leaf label (unchanged)
SCC_DATA=s0023/c1/k3              # nested data_root leaf

test -f "$WS/events/$EVENT/$SCC/event_job.json"
test -f "$WS/events/$EVENT/$SCC/frames.csv"
test -d "$DATA/$SCC_DATA/templates/oversampling_1" && \
  ls "$DATA/$SCC_DATA/templates/oversampling_1"/*.fits.fz >/dev/null 2>&1
test -d "$WS/events/$EVENT/$SCC/ws"
test -f "$WS/events/$EVENT/$SCC/ws/"*/hotpants_substamp_stars.csv 2>/dev/null || \
  test -f "$WS/events/$EVENT/$SCC/shared_mask.fits"
```

(Adjust glob paths to match your `diff_config.yaml` pipeline labels.)

### Run directory (orchestration)

```text
{WS}/runs/{run_id}/
  config.yaml
  targets.csv
  run_meta.json
  summary.json
  per_target/{LABEL}/            # flat, keyed by Target.label() — e.g. s0023_c1_k3_2020ftl
    {stage}.log
    {stage}.status.json
    {stage}.manifest.json
    {stage}.condor.*          # when executor=condor
```

### Science caches (under `data_root/s{SSSS}/c{C}/k{K}/`, not `events/`)

- `{data_root}/s{SSSS}/c{C}/k{K}/mapping/oversampling_1/…` — mapping
- `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr` — PS1 download (unchanged path)
- `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_1/…/syndiff_template_*.fits.fz` — `downsample` stage output

**Pass criteria**

- [ ] `event_job.json` and `frames.csv` exist under `events/{EVENT}/{SCC}/` before diff (written by `bind`)
- [ ] `ps1_removed_stars.csv` present after `downsample` (linear geometry_mode)
- [ ] Before diff: `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_1/` contains `syndiff_template_*.fits*` (or a complete field-mode manifest)
- [ ] `events/{EVENT}/{SCC}/ws/` contains at least one non-`master` workspace label after diff
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
| Submit template | `syndiff template submit --site config --scc … --run-id <id>` |
| Submit diff | `syndiff diff submit --site config --targets … --stages bind,diff --run-id <id>` |
| Monitor | `syndiff progress --site config --run-id <id>` |
| Verify diff | `syndiff verify --site config --run-id <id> --stages diff` |
| Retry diff | `syndiff retry --deployment config/deployment.yaml --run-id <id> --scc <label> --stage diff` |
| Manifests | `syndiff reconcile-manifests --site config --run-id <id>` |

**Further reading:** [template_pipeline.md](template_pipeline.md), [template_runner_architecture.md](template_runner_architecture.md), [syndiff_cli.md](syndiff_cli.md), [storage_layout.md](storage_layout.md).
