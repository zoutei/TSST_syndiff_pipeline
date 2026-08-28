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
| **event dir** | `{workspace_root}/events/{event_name}/s{SSSS}_c{C}_k{K}/` — handoff JSON; photometry under `phot_{run_id}/` |
| **workspace tree** | Legacy `ws/` trees (optional); SCC diff products live under `data_root/.../diff_{lane}/` |
| **photometry_run_id** | Namespaces event photometry tree (`phot_{id}/`) from `photometry_config.yaml` |

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
      phot_{photometry_run_id}/   # astrometry + forced LCs (photometry stage) -- the ONLY thing under events/ now
        targets.reg
        astrometry_result.json
        {lc_label}/lightcurve_*.csv
      frames.csv                   # optional event copy of SCC frame manifest
      ps1_removed_stars.csv        # crop-local Gaia (downsample stage; linear geometry_mode)
```

SCC subtract/ePSF/centroids products are **not** stored under the event tree. They live on the shared lane at `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/` (flat label dirs, `tess{digits}-s{SSSS}-{C}-{K}_{label}.fits.fz` stems). Fingerprints are recorded in `provenance.db` only.

Legacy `ws/` / `ws_{workspace_run_id}/` trees may still exist from older runs but are no longer written by the diff stage.

```text
# removed from current diff writes (legacy layout)
      ws/
      ws_{workspace_run_id}/
        master/
        tile_centers.json
        {diffs_label}/           # now on data_root diff lane only
```

**Deprecated event `ws/` trees:** older runs stored diff FITS, masks, ePSF, and forced photometry under `events/{name}/s…/ws/` (or `ws_{workspace_run_id}/`). Current diff writes are **SCC-primary** on `data_root` (below). Event trees may still hold progress sidecars, frozen `diff_config.yaml`, and `templates`/`ffis` symlinks for exploration; they are not the source of truth for subtract/ePSF completeness. Per-event photometry lives under `phot_{photometry_run_id}/`, not `ws/{lc_label}/`.

**Legacy star layout:** older runs used `events/{label}/star/`, `star_{id}/`, or `phot_{run_id}/host_star/`. Star is now per-SCC, not per-event: SCC-only verify requires `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/host_star/batch_manifest.csv`.

**`workspace_inherit`** (removed): SCC-only diff no longer supports inheriting labels from a parent event `ws/` tree. Re-run upstream diff stages on the SCC lane instead.

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
  discord_bot.lease.json           # bot singleton lease (authoritative; mirrors daemon.lease)
  discord_bot.lock                  # best-effort flock (lease wins)
  discord_bot.pid
  discord_bot.log
  discord_bot_config.path          # site pipeline.yaml for bot.enabled
  workspace_deployment.path        # recorded path to deployment.yaml
```

The Discord status bot is a **supervisor-managed subprocess** with an NFS lease (`discord_bot.lease.json`), pid file, and best-effort lock under `control/`. Legacy detached bot pid/lock files may still exist on older workspaces and are cleaned up on supervisor start/stop.

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
    ffi_list.parquet               # shared FFI header inventory (+ ffi_list.csv twin)
    mapping/
      oversampling_{N}/            # PanCAKES skycell maps
    remap/
      oversampling_{N}/            # field L2–L4 (schedule, groups, exact_cache)
    remap_{NAME}/                  # optional named remap lane (stages.remap.store_name)
      oversampling_{N}/
    templates/
      oversampling_{N}/            # L5 sparse contribs + template_manifest (field mode); see field_geometry.md for
                                    # the optional interior/seam-delta split store and convolved_templates'
                                    # use_patch_cache basis_conv/ cache, both under this same directory
    templates_{NAME}/              # optional named templates lane (downsample.output_store_name)
      oversampling_{N}/
    diff_{NAME}/                   # named diff lane (paths.output_store_name)
      shared_mask.fits.fz
      hotpants_substamp_stars.csv
      gaia_catalog_pipeline.csv
      diff_config.yaml             # immutable per-lane config lock (chmod 444)
      diff_config.fingerprint      # its fingerprint sibling, `v2:`-prefixed;
                                   #   records the RECIPE only -- execution
                                   #   resources (n_jobs, condor requests) are
                                   #   hashed nowhere and live per-run in
                                   #   runs/{run_id}/config.yaml instead, since
                                   #   one lane is built by many runs at
                                   #   different worker counts. A bare
                                   #   (pre-v2) fingerprint self-migrates on
                                   #   the lane's next run.
      mask_settings.yaml           # frozen effective mask policy
      host_star/                   # star branch outputs (per-SCC, not per-event)
      hp_d/tess{digits}-s{SSSS}-{C}-{K}_{label}.fits.fz
      hp_b/
      hp_c/
      hp_d_kernels/
      epsf_r1/
      centroids_r1/
      debug_plots/                 # diff-stage diagnostics (default lane: diff/debug_plots/)
        epsf_r1/                   # ePSF montage PNGs ({label}_{ffi_stem}.png)
        masks/                     # shared_mask QA when pipeline_plots: true
        background/                # background GIFs when pipeline_plots: true
    debug_plots/                   # template-pipeline diagnostics only
      wcs_drift_linear_template.png  # written by linear downsample / remap drift_source:point (ref-FFI-center point-drift groups; not by mapping)
      mapping_projection_overlay.png
      skycell_shift_*_debug.png
    bookkeeping/diff_{NAME}/oversampling_{N}/
      frames.csv
      diff_job.json
    legacy/                        # archived pre-cutover artifacts
    bookkeeping/                   # per-stage run_meta (mapping reference FFI, diff handoff, …)
      mapping/                     # run_meta.json = reference FFI path only (no drift PNG)
      diff/
        frames.csv
        diff_job.json
  ps1_skycells_zarr/               # shared PS1 raw-band cache (ps1_download + syndiff star)
    ps1_skycells.zarr
```

Path helpers live in `syndiff_pipeline/common/scc_paths.py`: `scc_label()` builds the orchestration/event label `s{SSSS}_c{C}_k{K}`; `scc_root()` builds the nested filesystem leaf `s{SSSS}/c{C}/k{K}/`. `scc_ffi_dir()`, `scc_catalogs_dir()`, `scc_convolved_zarr()`, `scc_convolved_removed_stars_csv()`, `scc_ffi_list_parquet()` / `scc_ffi_list_csv()`, `scc_mapping_dir()`, `scc_remap_dir()`, `scc_templates_dir()`, `scc_legacy_dir()`, `scc_bookkeeping_dir()` / `scc_bookkeeping_stage_dir()` all build off `scc_root()`. `ps1_skycells_zarr_dir()` / `ps1_skycells_zarr_path()` / `ps1_skycells_zarr_lock_path()` resolve the shared PS1 store under `data_root`. `oversampling_dirname(N)` always nests as `oversampling_{N}/`, including `N=1`. Event-scoped helpers: `event_root()` and `event_scc_leaf()` (`events/{event_name}/s{SSSS}_c{C}_k{K}/` — still a flat label leaf under the event).

Older top-level science trees (`tess_ffi/`, `skycell_pixel_mapping/`, `field_templates/`, `shifted_downsampled/`, `convolved_results/`, flat `catalogs/`, and `scc/s{SSSS}_c{C}_k{K}/`) are obsolete and are not read by current code.

`ps1_download` and `syndiff star` share one PS1 Zarr store at
`{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`. Optional top-level
`ps1_zarr_path` in `star_config.yaml` overrides that location for unusual
deployments.

Diff imaging resolves templates from `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/`
(or `templates_{NAME}/…` when `paths.template_store_name` is set; or an explicit
`paths.template_dir` override). The `ws/templates` symlink is no longer created.

With `geometry_mode: field`, L2–L4 artifacts live under
`{data_root}/s{SSSS}/c{C}/k{K}/remap/oversampling_{N}/`:

```text
remap/oversampling_{N}/
  remap_manifest.json
  shift_schedule.npz / .json
  template_group_shifts.parquet
  template_groups.json
  exact_cache/{skycell}_sx{±N}_sy{±N}_exact.npz
  .lock
```

For a temporal lane, mapping/remap/template metadata additionally carries the
temporal frame-contract fingerprint. The MappingGrid serialization records
the coordinate frame, explicit science/template bounds, pad geometry, and
geometry fingerprint. Consumers reject a mismatch rather than silently reuse
the store. See [coordinate frames and cropping](coordinate_frames_and_cropping.md).

L5 sparse contribs and `template_manifest.json` live under
`templates/oversampling_{N}/` (product path name; stage name is `downsample`).
Code dual-reads legacy L2–L4 files colocated under `templates/` when
`remap_manifest.json` is absent. Migrate with
`syndiff_pipeline.template_creation.processing.migrate_field_remap_store.migrate_scc_remap_artifacts`
(copy+verify; sources left in place). When `N>1`, field
`base_tess_shape` / `roi_bounds` are oversampled pixels; see
[oversampled templates §9](oversampled_templates.md#9-field-mode--oversampling).

Linear template FITS at `N>1` carry native `XMIN`/`XMAX`/`YMIN`/`YMAX` plus `OVERSAMP=N`; array planes are shape `(native_h·N, native_w·N)`. Diff crops stay native and are scaled at load time (`common/template_coverage.py`).

`events/{event}/s_c_k/phot_{run_id}/` holds per-event astrometry and forced photometry only. Star is per-SCC, not per-event: outputs default to `host_star/` under the SCC diff lane (`{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/host_star/`), derived directly from `data_root`/sector/camera/ccd — never from an event workspace or photometry run id.

---

## NFS and SQLite

`pipeline_state.sqlite` in `control/` uses **WAL mode**. Treat it like any SQLite WAL database:

- Run the **supervisor daemon** on one submit host per `workspace_root`.
- Prefer running **CLI control/monitor** on that same host.
- `data_root`, `events/`, and `runs/` **may** live on NFS; Condor workers read/write artifacts via mounts.

Full daemon/liveness details: [template_runner_architecture.md — SQLite and NFS](template_runner_architecture.md#sqlite-and-nfs).

---

## Provenance bookkeeping (`data_root/bookkeeping/`)

Content-addressed provenance lives under `data_root`. For a full guide (concepts, DAG, scheduler integration, operator runbook), see **[bookkeeping.md](bookkeeping.md)**.

```text
{data_root}/bookkeeping/
  provenance.db          # derived index (rebuildable via ``syndiff bookkeeping reindex``)
  spool/                 # per-process JSONL sidecars drained by the supervisor
```

Shared PS1 stores (decision #14):

```text
{data_root}/ps1_skycells_zarr/
  ps1_skycells.zarr/     # raw bands
  ps1_combined.zarr/     # star-removed combined cells (PR4)
  ps1_convolved.zarr/    # canonical convolved cells (PR5, gated)
```

`ps1_combined.zarr`/`ps1_convolved.zarr` are content-addressed by
`{projection}/{cell}/{fingerprint}/` and shared **across every sector/run**
that touches the same sky cell — publishing is append-only and immutable, so
more than one recipe (e.g. differing `remove_saturated_stars`,
`bright_star_mag_threshold`, `psf_sigma`) can legitimately accumulate
side-by-side for the same cell. Readers must never select "whichever
fingerprint has the newest mtime" — a newer, unrelated publish under a
different recipe can silently reintroduce saturated stars. Always resolve
via `combined_store.resolve_combined_fingerprint_for_recipe` /
`convolved_store.resolve_convolved_fingerprint_for_recipe` (deterministically
recomputed from the caller's own recipe), falling back to the `current.json`
pointer (`resolve_current_combined_ref`/`resolve_current_convolved_ref`) and
only then to mtime, with a loud warning, when no recipe context is available
at all. `scripts/audit_shared_store_recipes.py` reports cells with genuinely
conflicting recipes (read-only, does not mutate the store).

SCC-scoped diff store (field mode v2 — SCC-primary write-through):

```text
{data_root}/s{SSSS}/c{C}/k{K}/
  diff/                              # default lane (store_name null)
  diff_{lane}/                       # named lane (e.g. diff_linear/)
    shared_mask.fits.fz
    diff_config.yaml                 # immutable per-lane config lock (chmod 444)
    diff_config.fingerprint
    mask_settings.yaml
    hp_d/tess{digits}-s{SSSS}-{C}-{K}_hp_d.fits.fz
    hp_d_kernels/
    epsf_r1/gridded_epsf_index.json
    host_star/                       # star branch outputs (per-SCC)
  bookkeeping/
    diff/oversampling_{N}/           # lane bookkeeping (or legacy flat bookkeeping/diff/)
      frames.csv                     # SCC frame manifest (bootstrap)
      diff_job.json                  # v2: mapping_grid, store names, crop_bounds
```

Recipe fingerprints are recorded in `provenance.db` only; they are **not** encoded in the on-disk directory layout. Event workspaces do not receive mirrored diff FITS under `ws/{label}/`. Every diff artifact's `provenance.db` recipe row also carries `git_sha` (the full 40-char SHA of the checkout that produced it, stored but not part of the fingerprint hash) and, in the FITS/NPZ `meta`, `run_id` (the orchestrator run that produced it, empty for foreground/ad-hoc runs) — see [config_schema_v2.md](config_schema_v2.md#provenance).

Operator commands:

| Command | Purpose |
|---------|---------|
| `syndiff bookkeeping stats` | Row counts by kind/state |
| `syndiff bookkeeping reindex` | Offline DB rebuild from disk |

**Reindex warning:** A full reindex (default, without `--incremental`) clears `provenance.db`. Per-FFI diff rows are spool-ingested only and are **not** rebuilt from FITS on disk — drain `bookkeeping/spool/` first (supervisor ingest) and re-emit diff runs if needed.
| `syndiff bookkeeping gc` | Report-only orphan/missing scan |
| `syndiff bookkeeping pilot` | Phase-5 go/no-go checklist |
| `syndiff bookkeeping convolved-gate` | PR5 gate before write cutover |

**PR5 convolved-store cutover gate (s20/c1/k1 smoke SCC)**

Run after a dual-write or legacy `ps1_process` completion on a representative SCC, before setting `stages.ps1_process.use_shared_convolved_store: true` in any site config:

```bash
mamba activate syndiff
# data_root from config/deployment.yaml
syndiff bookkeeping convolved-gate \
  --data-root /path/to/data \
  --sector 20 --camera 1 --ccd 1 \
  --sample-cells 10
```

Exit code 0 and `"pass": true` in the JSON report means shared-store cells match legacy per-SCC `convolved.zarr` on padding-free skycells. Only then enable `use_shared_convolved_store` (with `write_per_scc_convolved_zarr: false`) for that campaign.

**Index-trust cutover** (`bookkeeping.trust_index` in `config/pipeline.yaml`, default `false`):

| Flag | Template stages with checkpoints | Scheduler verify | `run_stage` after success |
|------|----------------------------------|------------------|---------------------------|
| `false` (default) | Manifest + provenance dual-write | Checkpoint-first, fail-open to manifest/scan | Unchanged |
| `true` | Provenance emit only (no manifest JSON) | Index-only for checkpoint stages; no NFS scandir | Skip `collect_stage_artifacts` / manifest write |

Enable only after a green campaign with warm `provenance.db` (or run `syndiff bookkeeping reindex` first). Details: [bookkeeping.md §11](bookkeeping.md#11-bookkeepingtrust_index).

---

## Related docs

| Document | Contents |
|----------|----------|
| [field_geometry.md](field_geometry.md) | MappingGrid, field templates, rebuild runbook |
| [template_pipeline.md](template_pipeline.md) | CLI, config, deployment setup |
| [template_runner_architecture.md](template_runner_architecture.md) | Scheduler internals, daemon lifecycle |
| [cluster_smoke_checklist.md](cluster_smoke_checklist.md) | Manual validation on a cluster |
| [stages/diff_pipeline.md](stages/diff_pipeline.md) | Diff sub-stages, kernels, ePSF/gepsf photometry |
