---
name: syndiff-run-ops
description: Run, monitor, debug, and retry syndiff_pipeline batch runs (the syndiff CLI, supervisor daemon, HTCondor stages, host-star light curves, logs, verify). Use when executing pipeline stages, investigating a failed or stalled run, reading stage logs, retrying targets, or testing pipeline changes on real data.
---

# SynDiff Run Operations

## Environment — always first

```bash
mamba activate syndiff
```

Required before running any Python in this repo (pipeline, scripts, tests).

## Launching

```bash
syndiff all submit --site config/ --targets config/targets_example.csv      # full 7-stage DAG
syndiff template submit --site config/ --targets config/targets_example.csv # through downsample
syndiff diff submit --site config/ --targets config/targets_example.csv     # diff only
syndiff diff run --site config/ --targets config/targets_example.csv --target-name {label}
syndiff star submit --site config/ --star-targets {star_targets.csv}
syndiff star run --site config/ --star-targets {star_targets.csv} --target-name {scc_or_label}
```

- `--site DIR` loads `pipeline.yaml` + `diff_config.yaml` + `deployment.yaml`.
- Optional sibling `mask_settings.yaml` owns mask policy (not required in `diff_config`; bare `- kind: shared_mask` uses site file or packaged defaults). See `docs/markdown/masking.md`.
- `star` additionally loads `star_config.yaml` and a separate `star_targets.csv`; it verifies an existing template+diff workspace rather than depending on the main DAG in SQLite.
- `--local` on submit rewrites the frozen run config so `stages.diff.executor: local` (Condor bypass for smoke tests).
- `--local` on `star submit` instead rewrites `stages.star.executor: local`.
- `--stages`, `--targets`, `--force-rerun`, `--run-id` override presets.
- Heavy stages run on HTCondor by default: `mapping`, `ps1_process`, `diff`, `star`.

## Monitoring and control

```bash
syndiff status --site config/ --watch        # per-target stage grid
syndiff progress --site config/              # aggregate + per-task progress
syndiff tail --run-id {rid} --deployment config/deployment.yaml --scc {label} --stage {stage}
syndiff retry --run-id {rid} --deployment config/deployment.yaml --scc {label} --stage {stage}
syndiff pause|resume|kill --run-id {rid} --deployment config/deployment.yaml
syndiff verify --site config/ --targets config/targets_example.csv  # pre-run artifact check
```

Run-control verbs (`retry`, `pause`, `kill`, `logs`, `tail`, `show`) take `--run-dir` or `--run-id` + `--deployment`, **not** `--site`. Control commands write intents to SQLite; the supervisor applies them on its next tick.

## Where to look when something fails

1. **Per-stage log**: `{workspace_root}/runs/{run_id}/per_target/{target_label}/{stage}.log` (e.g. `mapping.log`, `diff.log`). Condor stderr lands alongside.
2. **Daemon log**: `syndiff logs` (supervisor scheduling decisions, verify results, Condor submit errors).
3. **Run / workspace metadata**:
   - `runs/{run_id}/run_meta.json` + frozen `config.yaml` / per-target `diff_config.yaml`
   - Event workspace slim freeze: `events/{label}/ws[_id]/diff_config.yaml` (`cfg_to_snapshot_dict` — empties/defaults omitted) and `…/mask_settings.yaml` after `shared_mask`
   - Check frozen copies before assuming site YAML defaults applied. Config ownership: `docs/markdown/stages/diff_pipeline.md` §0.
4. **State DB**: `{workspace_root}/control/pipeline_state.sqlite`; `syndiff runs` / `syndiff active` list runs and supervisor health. Status semantics: `docs/markdown/pipeline_state_machine_reference.md`.
5. **Event artifacts**: `{workspace_root}/events/{label}/` — `cluster_template_job.json`, `syndiff_ffi_frames.csv`, `ws/…`, and `star[_workspace_run_id]/batch_manifest.csv` (see the syndiff-pipeline-map skill for the artifact map).

## Pitfalls

- **Verify gates are shallow** (mapping = one CSV; templates = parseable files). After changing an upstream stage's behavior, delete its outputs or use `--force-rerun`; stale artifacts will otherwise be "verified" and skipped.
- **`ps1_download` writes a shared Zarr with a file lock** — concurrent runs on the same `data_root` serialize there; a stuck lock file can stall the stage.
- **`ps1_process.ps1_source: stream` skips `ps1_download`** and fetches directly after `mapping`; seeing `ps1_download` marked skipped/n/a is expected in that mode.
- **Star's default Zarr path differs from `ps1_download`**: star uses `{data_root}/ps1_skycells.zarr`; the template stage uses `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`. Set `ps1_zarr_path` in `star_config.yaml` when `zarr_local_only` should reuse the template-stage store.
- **Star needs baseline side products**, not merely finished diffs: convolved templates, photutils backgrounds, shared mask, and `{diffs}_kernels/*_kernel.npz`. Backfill older workspaces with `config/diff_config_star_full_backfill.yaml`.
- **Star gepsf lives in the baseline workspace**: each `psf_type: epsf` method requires `inputs.epsf: {label}`. An optional `epsf` block builds `{baseline_ws}/{epsf.output}` and requires `epsf.output == inputs.epsf`; without that block, the labeled catalog must already exist. `epsf.inputs.diffs` selects the source baseline diffs (default: `baseline.diffs`). `overwrite: true` rebuilds a configured catalog.
- **Workspace suffixes are independent**: diff baselines use `ws/` or `ws_{baseline.workspace_run_id}/`; star outputs use `star/` or `star_{defaults.workspace_run_id}/`.
- Expected stage durations (native res, one SCC): mapping ~13 min (Gaia ~2.5 min + ~10 min skycell loop), ps1_process tens of minutes to hours, downsample minutes, diff depends on frame count. A mapping stage running for hours is stuck, not slow.
- `syndiff diff submit` only artifact-verifies `tess_ffi_download`, `wcs_grouping`, `downsample`; `mapping`/`ps1_download`/`ps1_process` are assumed present (marked n/a).
- Reference docs: `docs/markdown/template_pipeline.md` (orchestration), `docs/markdown/syndiff_cli.md` (all verbs/flags), `docs/markdown/cluster_smoke_checklist.md` (Condor/NFS validation).

## Testing changes cheaply

- Single target, foreground: `syndiff diff run --site config/ --targets {csv} --target-name {label}` (add `--validate-only` for config checks without executing).
- Single SCC star branch: `syndiff star run --site config/ --star-targets {csv} --target-name {scc_or_label}`; use `max_ffis` in `star_config.yaml` for a short debug run. For gepsf testing, pass `--star-config config/star_config_epsf_gepsf.yaml`.
- Stage modules have CLI entry points for standalone runs (e.g. `python -m syndiff_pipeline.template_creation.processing.pancakes <cluster_template_job.json>`); see `docs/markdown/syndiff_cli.md` "Internal worker entry points".
- Use a small `crop_mode: target_box` event to keep downsample/diff fast.
