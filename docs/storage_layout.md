# Storage layout and filesystem reference

Canonical on-disk layout for the supervised `syndiff` pipeline. Two deployment roots are configured in `deployment.yaml`:

| Key | Role |
|-----|------|
| `workspace_root` | Orchestration state, run bookkeeping, per-target persistent outputs |
| `data_root` | Shared SCC-scoped science caches (mapping, Zarr, template FITS) |

Example paths (adjust for your site):

```text
/astro/.../syndiff/workspace     # workspace_root
/astro/.../syndiff/data          # data_root
```

---

## Naming glossary

| Term | Meaning |
|------|---------|
| **workspace** | The `workspace_root` directory — one SQLite DB, one supervisor, one `runs/` tree |
| **event dir** | `{workspace_root}/events/{target_label}/` — handoff JSON + diff outputs for one target |
| **run dir** | `{workspace_root}/runs/{run_id}/` — frozen config and per-run stage sidecars |
| **control dir** | `{workspace_root}/control/` — orchestrator-only files (SQLite, daemon, Discord) |

Deprecated in prose: *handoff root*, *template_handoffs* (old path name).

---

## `{workspace_root}/` tree

Only three top-level subtrees belong here long-term:

```text
{workspace_root}/
  control/                         # orchestrator state (see below)
  runs/                            # batch run bookkeeping
  events/                          # per-target persistent outputs
    {target_label}/                # e.g. s0023_c1_k3_2020ftl
      cluster_template_job.json    # wcs_grouping handoff
      syndiff_ffi_frames.csv       # frame manifest
      wcs_drift_template_debug.png
      ps1_removed_stars.csv        # crop-local Gaia (downsample)
      ws/                          # diff stage workspaces
        templates/                 # symlink → {data_root}/shifted_downsampled/... (after downsample)
        master/                    # flat FITS mirror + tess_ffi link (diff stage)
        {workspace_label}/         # one Hotpants/photometry workspace per offset group
          hotpants_substamp_stars.csv
          ...
```

### `control/` — orchestrator only

```text
{workspace_root}/control/
  pipeline_state.sqlite            # WAL-mode SQLite; all runs share one DB
  pipeline_state.sqlite-wal        # SQLite WAL sidecar (when active)
  pipeline_state.sqlite-shm
  daemon.lock                      # flock: only one supervisor process
  daemon.pid
  daemon.log
  discord_bot.lock                 # flock: one Discord bot per workspace
  discord_bot.pid
  discord_bot.log
  discord_bot_config.path          # site pipeline.yaml path for bot (re)start
  workspace_deployment.path        # recorded path to deployment.yaml
```

Code resolves these via `control_root()` and `state_db_path()` in `syndiff_pipeline.common.orchestration.workspace`.

**Supervisor heartbeat** is host-local (not under `control/`): `$TMPDIR/syndiff-daemon/{hash}.heartbeat`. See [SQLite and NFS](template_runner_architecture.md#sqlite-and-nfs).

### `runs/` — batch bookkeeping

```text
{workspace_root}/runs/
  latest -> {run_id}               # symlink to most recent run
  {run_id}/
    config.yaml                    # frozen site config with absolute paths
    targets.csv
    run_meta.json
    summary.json / summary.csv
    per_target/
      {target_label}/
        {stage}.log
        {stage}.status.json        # worker liveness / exit code
        {stage}.manifest.json      # per-run completion manifest
        {stage}.condor.*           # when executor=condor
  .manifests/                      # cross-run stable skip cache
    {target_label}/
      {stage}.manifest.json
```

---

## `{data_root}/` tree (science caches)

Shared across targets on the same SCC where noted. Paths are derived in `runner_config.resolve_config()`.

```text
{data_root}/
  tess_ffi/                        # tess_ffi_download (optional override via ffi_dir)
    s{sector:04d}/cam{camera}_ccd{ccd}/
      tess*_ffic.fits.gz           # calibrated FFIs (gzip after download)
  skycell_pixel_mapping/           # mapping (PanCAKES)
    sector_{SSSS}/camera_{C}/ccd_{K}/
      tess_s{SSSS}_{C}_{K}_master_skycells_list.csv
      ...
  ps1_skycells_zarr/
    ps1_skycells.zarr                # ps1_download (shared store; lock file alongside)
    ps1_skycells.zarr.lock
  convolved_results/
    sector_{SSSS}_camera_{C}_ccd_{K}.zarr
    sector_{SSSS}_camera_{C}_ccd_{K}_removed_stars.csv
  shifted_downsampled/             # downsample template FITS (default output_base)
    sector_{SSSS}_camera_{C}_ccd_{K}_x{X0}-{X1}_y{Y0}-{Y1}/
      syndiff_template_*.fits.gz
```

| Subtree | Stage |
|---------|-------|
| `tess_ffi/` | `tess_ffi_download` |
| `skycell_pixel_mapping/` | `mapping` |
| `ps1_skycells_zarr/` | `ps1_download` |
| `convolved_results/` | `ps1_process` |
| `shifted_downsampled/` | `downsample` |

After `downsample`, the orchestrator creates `events/{label}/ws/templates` as a symlink to the physical template directory under `{data_root}/shifted_downsampled/…` (the sector/camera/CCD/crop subtree containing `syndiff_template_*.fits.gz`). Diff imaging resolves templates via this symlink (or an explicit `paths.template_dir` override in `diff_config.yaml`) and writes science products under `events/{label}/ws/`.

`events/{label}/ws/master/` is a **flat FITS mirror** for Condor/shared-FS access: every `ws/{workspace_label}/*.fits` appears as a basename symlink, plus `ws/master/tess_ffi` → `{data_root}/tess_ffi/` when configured. It does **not** hold template FITS.

---

## NFS and SQLite

`pipeline_state.sqlite` in `control/` uses **WAL mode**. Treat it like any SQLite WAL database:

- Run the **supervisor daemon** on one submit host per `workspace_root`.
- Prefer running **CLI control/monitor** on that same host.
- `data_root`, `events/`, and `runs/` **may** live on NFS; Condor workers read/write artifacts via mounts.

Full daemon/liveness details: [template_runner_architecture.md — SQLite and NFS](template_runner_architecture.md#sqlite-and-nfs).

---

## Legacy layout (pre-migration)

Older workspaces may have:

1. **Flat target dirs** at `{workspace_root}/{target_label}/` instead of `events/{target_label}/`.
2. **Orchestrator files at workspace root** (`pipeline_state.sqlite`, `daemon.*`, `discord_bot.*`) instead of under `control/`.
3. **Path name `template_handoffs`** instead of `workspace`.

Use the one-time migration script (below) before running a current `syndiff` supervisor against such a tree.

---

## Migrating an existing workspace

**Prerequisites:** stop the supervisor if it is running (`syndiff daemon stop --site …`).

```bash
mamba activate syndiff

# Preview changes
python scripts/migrate_workspace_layout.py \
  --workspace-root /path/to/template_handoffs \
  --rename-to /path/to/workspace \
  --dry-run

# Apply
python scripts/migrate_workspace_layout.py \
  --workspace-root /path/to/template_handoffs \
  --rename-to /path/to/workspace
```

Then update `deployment.yaml`:

```yaml
workspace_root: /path/to/workspace
```

Restart the daemon (`syndiff daemon start --site …`) or let the next `submit` auto-start it.

The script:

1. Optionally renames the workspace directory (`template_handoffs` → `workspace`).
2. Creates `control/` and moves SQLite, daemon, Discord, and deployment pointer files.
3. Normalizes `events/{label}/` (moves flat `{label}/` dirs; resolves symlink indirection).

Related one-time utility (not part of layout migration): `scripts/backfill_ps1_removed_stars.py` — writes missing `events/{label}/ps1_removed_stars.csv` for targets that completed downsample before that file was added.

---

## Related docs

| Document | Contents |
|----------|----------|
| [template_pipeline.md](template_pipeline.md) | CLI, config, deployment setup |
| [template_runner_architecture.md](template_runner_architecture.md) | Scheduler internals, daemon lifecycle |
| [cluster_smoke_checklist.md](cluster_smoke_checklist.md) | Manual validation on a cluster |
| [stages/README.md](stages/README.md) | Per-stage output paths |
