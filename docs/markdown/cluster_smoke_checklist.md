# Cluster smoke test checklist

Manual smoke test for the supervised `syndiff` pipeline on a shared cluster (HTCondor + NFS). Run on the **daemon host** for the workspace — the machine where `syndiff * submit` starts the supervisor.

**Prerequisites**

- [ ] `mamba activate syndiff` and `pip install -e .` from the repo root
- [ ] `config/deployment.yaml` exists (copy from `deployment.yaml.example`) with valid `workspace_root` and `data_root` on shared storage
- [ ] Submit host has Condor client tools and NFS mounts for `workspace_root`, `data_root`, and your conda env
- [ ] Pick one smoke SCC (e.g. sector 20, camera 1, CCD 1 from `config/scc_s20_c1_k1.csv`)

Record results in a dated note (optional): `docs/cluster_smoke_YYYY-MM-DD.md`.

There is **no combined end-to-end preset** — `syndiff template` (SCC-scoped) and `syndiff diff` (SCC-scoped field subtract or event-scoped photometry) are separate submits. Run template first, then diff.

---

## 1. Submit template run (one SCC)

```bash
mamba activate syndiff

syndiff template submit \
  --site config \
  --config config/pipeline_s20_field_l4_split_smoke.yaml \
  --scc config/scc_s20_c1_k1.csv \
  --run-id smoke_template_01
```

Use a one-row SCC CSV (`sector,camera,ccd,enabled`). Supervised `submit` schedules every enabled row in `--scc`.

Optional shortcuts for faster smoke:

- `--stages mapping,remap,downsample` — skip earlier stages if FFIs/Zarr already exist
- `--force-rerun` — ignore existing artifacts for selected stages (new run only)

**Pass criteria**

- [ ] Command exits 0 and prints monitor hints including `syndiff progress --run-id smoke_template_01`
- [ ] `{workspace_root}/runs/smoke_template_01/` created with frozen `config.yaml`, `targets.csv` (normalized SCC CSV), `run_meta.json`
- [ ] `syndiff daemon status --site config` shows a live supervisor PID
- [ ] `runs/latest` symlink points at `smoke_template_01`
- [ ] `{data_root}/s0020/c1/k1/templates/oversampling_1/` populated after `downsample` succeeds
- [ ] Master mapping FITS has `MAPGRID=3`; `field_mode_assembly.json` has `schema_version: 3`

---

## 1b. Diff smoke (`syndiff diff submit --scc`)

Use once **template artifacts already exist** on disk for the smoke SCC (FFIs, mapping with MAPGRID=3, remap, template store v3). This submits the single `diff` stage; `scc_bootstrap` runs inside execute. Missing or non-3 MAPGRID metadata fails closed and requires a mapping rebuild.

```bash
mamba activate syndiff

syndiff diff submit \
  --site config \
  --scc config/scc_s20_c1_k1.csv \
  --run-id smoke_diff_01
```

`--site config` alone resolves `config/pipeline.yaml`, whose embedded `diff:`
block already carries the single-kernel recipe (`shared_mask` → `kernel_fit` →
`convolved_templates` → `background_estimate` → `background_temporal_smoothing`
→ `subtract`). `--config` and `--site` both always resolve a `pipeline.yaml`-
style file now — pointing `--config` straight at a standalone
`config/diff_config_single_kernel.yaml` (a bare diff-only YAML) no longer
works, since that file has neither `stages:` nor a `diff:`/`diff_config:` key
of its own for the loader to find. See
[config_schema_v2.md](config_schema_v2.md).

Optional: `--local` to run `diff` on the submit host instead of Condor.

**Pass criteria**

- [ ] Command exits 0; `syndiff progress --run-id smoke_diff_01` shows brief verify scans for `tess_ffi_download`/`downsample`, then `diff`
- [ ] `{workspace_root}/runs/smoke_diff_01/config.yaml` frozen; template stages marked n/a or verified external
- [ ] `{data_root}/s0020/c1/k1/bookkeeping/diff/frames.csv` and `diff_job.json` written by `scc_bootstrap`
- [ ] Diff FITS under `{data_root}/s0020/c1/k1/diff_{lane}/` with science shape 1960×2018
- [ ] After completion, sections 2–6 below apply with `--run-id smoke_diff_01`

### 1c. Event photometry smoke (optional)

When testing per-target forced photometry under `events/{name}/ws/`:

```bash
syndiff diff submit \
  --site config \
  --targets config/targets_smoke.csv \
  --run-id smoke_diff_event_01
```

`--targets` and `--scc` are mutually exclusive. Templates must still exist on disk (`DIFF_VERIFY_UPSTREAM`).

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
  --target s0020_c1_k1 --stage diff
```

---

## 3. Verify artifacts on disk

```bash
syndiff verify --site config --run-id smoke_diff_01
```

**Pass criteria**

- [ ] `diff` stage reports complete for the smoke SCC target
- [ ] No `[FAIL]` lines for `tess_ffi_download` or `downsample` when those were verified external

---

## 4. Retry a failed stage

If a stage failed during smoke:

```bash
syndiff retry --run-id smoke_diff_01 --target s0020_c1_k1 --stage diff
```

**Pass criteria**

- [ ] Retry resets `diff` to `pending` and supervisor relaunches it
- [ ] No duplicate Condor clusters for the same `(run, target, stage)` after reconcile

---

## 5. Reconcile manifests (stable skip cache)

After a successful run:

```bash
ls {workspace_root}/runs/.manifests/s0020_c1_k1/diff.manifest.json
```

**Pass criteria**

- [ ] Stable manifest exists with matching `config_fingerprint`
- [ ] A second `syndiff diff submit --scc ...` with the same config skips `diff` when artifacts are complete (unless `--force-rerun`)

---

## 6. Inspect outputs

SCC-primary diff store:

```text
{data_root}/s0020/c1/k1/
  bookkeeping/diff/
    frames.csv
    diff_job.json              # mapping_grid, crop_bounds, store names
  diff_{lane}/                 # e.g. diff_l4_split_smoke/
    {workspace_label}/{recipe_fp}/
      tess<digits>_{label}.fits.fz
```

Event workspace (when using `--targets`):

```text
{workspace_root}/events/{event_name}/s0020_c1_k1/
  ws/                          # photometry tree
    hp_d/
    ...
```

**Pass criteria**

- [ ] `bookkeeping/diff/diff_job.json` contains `mapping_grid` and `crop_bounds`
- [ ] Difference FITS PRIMARY data shape matches science grid (1960×2018 native at F=1)
- [ ] `syndiff progress` shows terminal `success` for all selected stages

---

## Quick reference

| Step | Command |
|------|---------|
| Submit template | `syndiff template submit --site config --scc … --run-id <id>` |
| Submit field diff | `syndiff diff submit --site config --scc … --run-id <id>` |
| Submit event diff | `syndiff diff submit --site config --targets … --run-id <id>` |
| Monitor | `syndiff progress --site config --run-id <id>` |
| Verify | `syndiff verify --site config --run-id <id>` |
| Daemon status | `syndiff daemon status --site config` |

See [template_pipeline.md](template_pipeline.md), [field_geometry.md](field_geometry.md), and [storage_layout.md](storage_layout.md) for full layout and rebuild guidance.

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
