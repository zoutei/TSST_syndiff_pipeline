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
| **event dir** | `{workspace_root}/events/{event_name}/s{SSSS}_c{C}_k{K}/` — handoff JSON + diff outputs for one event×SCC leaf |
| **run dir** | `{workspace_root}/runs/{run_id}/` — frozen config and per-run stage sidecars |
| **control dir** | `{workspace_root}/control/` — orchestrator-only files (SQLite, daemon, Discord) |
| **workspace tree** | `events/{label}/ws/` (canonical) or `events/{label}/ws_{workspace_run_id}/` — diff sub-pipeline artifacts (and star outputs under `host_star/`) |
| **workspace_run_id** | Namespaces the diff workspace tree (`ws_{id}/`) when set in `diff_config` / `star_config.baseline` |

Deprecated in prose: *handoff root*, *template_handoffs* (old path name).

---

## `{workspace_root}/` tree

Only three top-level subtrees belong here long-term:

```text
{workspace_root}/
  control/                         # orchestrator state (see below)
  runs/                            # batch run bookkeeping
  events/                          # per-event nested SCC leaves
    {event_name}/                  # e.g. 2020ftl
      s{SSSS}_c{C}_k{K}/           # e.g. s0023_c1_k3
      event_job.json               # bind handoff (was cluster_template_job.json)
      frames.csv                   # frame manifest (was syndiff_ffi_frames.csv)
      ps1_removed_stars.csv        # crop-local Gaia (templates stage; linear geometry_mode)
      ws/                          # canonical diff workspace tree (no workspace_run_id)
      ws_{workspace_run_id}/       # namespaced tree when diff_config sets workspace_run_id
        diff_config.yaml           # frozen copy for this tree
        templates/                 # (removed) templates resolve from data_root/s{SSSS}/c{C}/k{K}/templates/
        master/                    # flat FITS mirror + tess_ffi link (diff stage)
        debug_plots/               # PNG diagnostics when pipeline_plots: true
          wcs_drift_template_debug.png
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
        {lc_label}/                # forced_photometry output; e.g. lc_gepsf_on_hp_diffs
          lightcurve_{method}.csv  # e.g. lightcurve_gepsf.csv
          lightcurve_{method}_{extra}.csv
        host_star/                 # syndiff star outputs (nested in baseline ws)
          batch_manifest.csv
          {gaia_source_id}/
            identifier.json
            host_gaia_row.csv
            mini_templates/
            diff_stamps/
            lightcurve_{method}_gaia_{id}.csv
            plots/                 # when debug_plots: true
```

### Diff workspace trees (`ws/` / `ws_{workspace_run_id}/`)

Filesystem name comes from `workspace_tree_name()` in `difference_imaging/support/paths.py`: canonical `ws/` when `workspace_run_id` is unset, otherwise `ws_{workspace_run_id}/`. Each tree holds one ordered diff sub-pipeline (labels from stage `output:` keys). Per-FFI FITS use `{tess_product_id}_{label}.fits.gz` (`support/ffi_naming.py`). Star-branch outputs live under `host_star/` inside the baseline workspace (not a sibling `star_*` tree).

| Path (under active `ws*` tree) | Stage / role |
|--------------------------------|--------------|
| `templates/` | **Removed.** Diff resolves `cfg.template_dir` from `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/` |
| `master/` | Flat basename symlinks to all workspace FITS + optional `tess_ffi` link (`master_fits_mirror`); skips `host_star/` |
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
| `{lc_label}/lightcurve_{method}.csv` | Forced photometry (e.g. `lc_gepsf_on_hp_diffs/lightcurve_gepsf.csv`) |
| `{lc_label}/lightcurve_{method}_{extra}.csv` | Additional forced targets (`additional_forced_targets`) |
| `host_star/` | Host-star light curves (`syndiff star`): per-Gaia stamps, mini-templates, LCs |

**Legacy:** older star runs wrote sibling trees `events/{label}/star/` or `star_{id}/`. Verify still accepts those when `host_star/batch_manifest.csv` is absent.

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
  s{SSSS}/c{C}/k{K}/               # nested SCC science leaf (scc_paths.scc_root)
    ffi/                           # tess_ffi_download (all FFIs for this SCC)
    catalogs/                      # Gaia DR3 (mapping stage)
    convolved.zarr                 # ps1_process output
    convolved_removed_stars.csv
    wcs_cache.parquet              # shared WCS keyword cache (+ wcs_cache.csv twin)
    mapping/
      oversampling_{N}/            # PanCAKES skycell maps
    templates/
      oversampling_{N}/            # full-chip sparse template store (field mode)
    legacy/                        # archived pre-cutover artifacts
    bookkeeping/                   # per-stage run_meta (mapping reference FFI, etc.)
  ps1_skycells_zarr/               # shared PS1 raw-band cache (ps1_download + syndiff star)
    ps1_skycells.zarr
```

Path helpers live in `syndiff_pipeline/common/scc_paths.py`: `scc_label()` builds the orchestration/event label `s{SSSS}_c{C}_k{K}`; `scc_root()` builds the nested filesystem leaf `s{SSSS}/c{C}/k{K}/`. `scc_ffi_dir()`, `scc_catalogs_dir()`, `scc_convolved_zarr()`, `scc_convolved_removed_stars_csv()`, `scc_wcs_cache_parquet()` / `scc_wcs_cache_csv()`, `scc_mapping_dir()`, `scc_templates_dir()`, `scc_legacy_dir()`, `scc_bookkeeping_dir()` / `scc_bookkeeping_stage_dir()` all build off `scc_root()`. `ps1_skycells_zarr_dir()` / `ps1_skycells_zarr_path()` / `ps1_skycells_zarr_lock_path()` resolve the shared PS1 store under `data_root`. `oversampling_dirname(N)` always nests as `oversampling_{N}/`, including `N=1`. Event-scoped helpers: `event_root()` and `event_scc_leaf()` (`events/{event_name}/s{SSSS}_c{C}_k{K}/` — still a flat label leaf under the event).

Older top-level science trees (`tess_ffi/`, `skycell_pixel_mapping/`, `field_templates/`, `shifted_downsampled/`, `convolved_results/`, flat `catalogs/`, and `scc/s{SSSS}_c{C}_k{K}/`) are obsolete and are not read by current code.

`ps1_download` and `syndiff star` share one PS1 Zarr store at
`{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`. Optional top-level
`ps1_zarr_path` in `star_config.yaml` overrides that location for unusual
deployments.

Diff imaging resolves templates from `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/` (or an explicit `paths.template_dir` override). The `ws/templates` symlink is no longer created.

With `geometry_mode: field`, templates live under `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/` (sparse contribs + assemble by `group_id` at diff time). When `N>1`, field `base_tess_shape` / `roi_bounds` are oversampled pixels; see [oversampled templates §9](oversampled_templates.md#9-field-mode--oversampling).

Linear template FITS at `N>1` carry native `XMIN`/`XMAX`/`YMIN`/`YMAX` plus `OVERSAMP=N`; array planes are shape `(native_h·N, native_w·N)`. Diff crops stay native and are scaled at load time (`common/template_coverage.py`).

`events/{event}/s_c_k/ws*/master/` is a **flat FITS mirror** for Condor/shared-FS access: every workspace-label `*.fits` appears as a basename symlink, plus `master/tess_ffi` → SCC `ffi/` when configured. It does **not** hold template FITS.

---

## NFS and SQLite

`pipeline_state.sqlite` in `control/` uses **WAL mode**. Treat it like any SQLite WAL database:

- Run the **supervisor daemon** on one submit host per `workspace_root`.
- Prefer running **CLI control/monitor** on that same host.
- `data_root`, `events/`, and `runs/` **may** live on NFS; Condor workers read/write artifacts via mounts.

Full daemon/liveness details: [template_runner_architecture.md — SQLite and NFS](template_runner_architecture.md#sqlite-and-nfs).

---

## Related docs

| Document | Contents |
|----------|----------|
| [template_pipeline.md](template_pipeline.md) | CLI, config, deployment setup |
| [template_runner_architecture.md](template_runner_architecture.md) | Scheduler internals, daemon lifecycle |
| [cluster_smoke_checklist.md](cluster_smoke_checklist.md) | Manual validation on a cluster |
| [stages/diff_pipeline.md](stages/diff_pipeline.md) | Diff sub-stages, kernels, ePSF/gepsf photometry |
