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
| **workspace tree** | `events/{label}/ws/` (canonical) or `events/{label}/ws_{workspace_run_id}/` — diff sub-pipeline artifacts |
| **workspace_run_id** | Namespaces diff (`ws_{id}/`) and star (`star_{id}/`) output trees when set in `diff_config` / `star_config` |

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
      ws/                          # canonical diff workspace tree (no workspace_run_id)
      ws_{workspace_run_id}/       # namespaced tree when diff_config sets workspace_run_id
        diff_config.yaml           # frozen copy for this tree
        templates/                 # symlink → {data_root}/shifted_downsampled/... (after downsample)
        master/                    # flat FITS mirror + tess_ffi link (diff stage)
        debug_plots/               # PNG diagnostics when pipeline_plots: true
        shared_mask.fits.gz        # ws-root artifacts (see diff workspace table below)
        hotpants_substamp_stars.csv
        gaia_catalog_pipeline.csv
        targets.reg
        tile_centers.json
        {diffs_label}/             # e.g. hp_d — per-FFI difference FITS
          tess{pid}_{diffs_label}.fits.gz
        {diffs_label}_m/           # meta workspace paired with diffs (hp_d → hp_m)
          hotpants.progress.json
          kernel_reconstruction.npz
          phot_calib.csv
        {diffs_label}_kernels/     # sibling of diffs dir; write_kernel_solutions: true only
          {product_id}_kernel.npz
        {epsf_label}/              # e.g. epsf_r1 — gridded ePSF
          {ffi_stem}_gridded_epsf.npz
          gridded_epsf_index.json
          epsf.progress.json
          group_epsf/              # optional group medians
            group_epsf_{gid}.npz
        {centroids_label}/         # e.g. centroids_r1
          {ffi_stem}_photresults.ecsv
          centroids_index.json
          centroids.progress.json
          centroids.progress.json
        {lc_label}/                # forced_photometry output; e.g. lc_gepsf_on_hp_diffs
          lightcurve_{method}.csv  # e.g. lightcurve_gepsf.csv
          lightcurve_{method}_{extra}.csv
      star/                        # default host-star outputs (syndiff star)
      star_{workspace_run_id}/       # namespaced star outputs (star_config workspace_run_id)
        batch_manifest.csv
        {gaia_source_id}/
          identifier.json
          host_gaia_row.csv
          mini_templates/
          diff_stamps/
          lightcurve_{method}_gaia_{id}.csv
          plots/                   # when debug_plots: true
```

### Diff workspace trees (`ws/` / `ws_{workspace_run_id}/`)

Filesystem name comes from `workspace_tree_name()` in `difference_imaging/support/paths.py`: canonical `ws/` when `workspace_run_id` is unset, otherwise `ws_{workspace_run_id}/`. Each tree holds one ordered diff sub-pipeline (labels from stage `output:` keys). Per-FFI FITS use `{tess_product_id}_{label}.fits.gz` (`support/ffi_naming.py`).

| Path (under active `ws*` tree) | Stage / role |
|--------------------------------|--------------|
| `templates/` | Symlink to downsampled `syndiff_template_*.fits.gz` (or `paths.template_dir` override) |
| `master/` | Flat basename symlinks to all workspace FITS + optional `tess_ffi` link (`master_fits_mirror`) |
| `debug_plots/` | Diagnostic PNGs when `pipeline_plots: true` (ePSF montages, light-curve figures, background GIFs) |
| `shared_mask.fits.gz`, `hotpants_substamp_stars.csv`, `gaia_catalog_pipeline.csv`, `targets.reg`, `tile_centers.json` | `shared_mask`, ePSF, `sat_template`, legacy photometry |
| `{diffs_label}/` | Hotpants or `kernel_subtract` difference images (e.g. `hp_d/`) |
| `{diffs_label}_m/` | Meta workspace for a diffs label (`hp_d` → `hp_m`): `hotpants.progress.json`, `kernel_reconstruction.npz`, `phot_calib.csv` |
| `{diffs_label}_kernels/{product_id}_kernel.npz` | Per-frame Hotpants `kernel_solution` (sibling of `{diffs_label}/`; only when `write_kernel_solutions: true`) |
| `{epsf_label}/*_gridded_epsf.npz` | Per-frame gridded ePSF archives (`data`, `grid_xypos`, `oversampling`) |
| `{epsf_label}/gridded_epsf_index.json` | `ffi_stem` → npz path index |
| `{epsf_label}/epsf.progress.json` | Frame progress sidecar (CLI mirror: `runs/.../diff.epsf.progress.json` beside `diff.log`) |
| `{epsf_label}/group_epsf/group_epsf_{gid}.npz` | Optional median gridded cube per WCS group |
| `{centroids_label}/*_photresults.ecsv` | Per-frame Gaia PSF photometry on diffs |
| `{centroids_label}/centroids_index.json` | `ffi_stem` → photresults path index |
| `{centroids_label}/centroids.progress.json` | Frame progress sidecar (CLI mirror: `runs/.../diff.centroids.progress.json` beside `diff.log`) |
| `{centroids_label}/centroids.progress.json` | Frame progress sidecar (CLI mirror: `runs/.../diff.centroids.progress.json` beside `diff.log`) |
| `{lc_label}/lightcurve_{method}.csv` | Forced photometry (e.g. `lc_gepsf_on_hp_diffs/lightcurve_gepsf.csv`) |
| `{lc_label}/lightcurve_{method}_{extra}.csv` | Additional forced targets (`additional_forced_targets`) |

**`workspace_inherit`** (`difference_imaging/support/workspace_inherit.py`): preamble entry that symlinks selected labels and root artifacts from a parent `ws_{from_run_id}/` into a child `ws_{run_id}/` without modifying the parent tree. Relative links look like `../ws_{from_run_id}/{label}`. Typical inherited labels: `hp_d`, `hp_m`; typical root artifacts: `shared_mask.fits`, `gaia_catalog_pipeline.csv`, `hotpants_substamp_stars.csv`, `targets.reg`.

### `control/` — orchestrator only

```text
{workspace_root}/control/
  pipeline_state.sqlite            # WAL-mode SQLite; all runs share one DB
  pipeline_state.sqlite-wal        # SQLite WAL sidecar (when active)
  pipeline_state.sqlite-shm
  daemon.lease                     # cross-host ownership lease (JSON; authoritative)
  daemon.stop                      # cross-host stop request (JSON)
  daemon.lock                      # best-effort same-host flock (lease wins)
  daemon.pid
  daemon.log
  discord_bot_config.path          # site pipeline.yaml for in-process bot.enabled
  workspace_deployment.path        # recorded path to deployment.yaml
```

The Discord status bot runs **inside the supervisor process** (no separate `discord_bot.pid` / lock). Legacy detached bot pid/lock files may still exist on older workspaces and are cleaned up on supervisor start.

Code resolves these via `control_root()` and `state_db_path()` in `syndiff_pipeline.common.orchestration.workspace`.

**Supervisor ownership** uses `daemon.lease` on the shared filesystem (renewed ~every 15s; stale after 120s). Host-local liveness remains at `$TMPDIR/syndiff-daemon/{hash}.heartbeat`. See [SQLite and NFS](template_runner_architecture.md#sqlite-and-nfs).

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
  ps1_skycells.zarr                  # star PS1 cache (same layout; used by syndiff star)
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

After `downsample`, the orchestrator creates `events/{label}/ws/templates` (or `ws_{workspace_run_id}/templates`) as a symlink to the physical template directory under `{data_root}/shifted_downsampled/…` (the sector/camera/CCD/crop subtree containing `syndiff_template_*.fits.gz`). Diff imaging resolves templates via this symlink (or an explicit `paths.template_dir` override in `diff_config.yaml`) and writes science products under the active workspace tree (see **Diff workspace trees** above).

`events/{label}/ws*/master/` is a **flat FITS mirror** for Condor/shared-FS access: every workspace-label `*.fits` appears as a basename symlink, plus `master/tess_ffi` → `{data_root}/tess_ffi/` when configured. It does **not** hold template FITS.

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
| [stages/diff_pipeline.md](stages/diff_pipeline.md) | Diff sub-stages, kernels, ePSF/gepsf photometry |
