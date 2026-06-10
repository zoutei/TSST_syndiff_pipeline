# Template pipeline examples

These files configure the **`syndiff-template`** CLI, which builds PS1 templates on the TESS pixel grid before running the main SynDiff difference-imaging pipeline.

## Documentation

| Guide | Contents |
|-------|----------|
| [Template pipeline](../../docs/template_pipeline.md) | Full orchestration guide — **start with [CLI Reference](../../docs/template_pipeline.md#cli-reference)** |
| [Stage algorithms](../../docs/stages/README.md) | PanCAKES, PS1 process, downsample deep-dives |
| [Docs index](../../docs/README.md) | Documentation map |

## Files in this directory

| File | Description |
|------|-------------|
| [`config.yaml`](config.yaml) | Site policy: stages, resource pools, notifications (`ps1_source: zarr` default; stream option documented in-file) |
| [`deployment.yaml.example`](deployment.yaml.example) | Copy to `deployment.yaml` — deployment paths (`handoff_root`, `data_root`) + credentials |
| [`config_s0020_c3_k3_roi.yaml`](config_s0020_c3_k3_roi.yaml) | ROI test run with downsample overrides |
| [`targets_example.csv`](targets_example.csv) | Multi-target example (sector, camera, ccd, coordinates) |
| [`events_example.csv`](events_example.csv) | Supernova catalog format with `tess_coverage` tokens |

## First-time setup

```bash
mamba activate syndiff
pip install -e ../..   # from repo root, once

cp deployment.yaml.example deployment.yaml
```

Edit `deployment.yaml` for your machine:

```yaml
handoff_root: /path/to/template_handoffs   # daemon, SQLite, runs/, handoffs
data_root: /path/to/syndiff/data           # FFIs, mapping, Zarr, templates
gaia_username: ...                         # optional; mapping stage
gaia_password: ...
discord_webhook_url: ...                    # optional; notifications
```

`config.yaml` holds stage parameters only — paths are never set there.

## Typical workflow

```bash
# 1. Check prerequisites on disk
syndiff-template verify --config config.yaml --targets targets_example.csv

# 2. Submit detached run (starts daemon if needed)
syndiff-template submit \
  --config config.yaml \
  --targets targets_example.csv \
  --stages ps1_process,downsample \
  --run-id my_batch_01

# 3. Monitor (no flags — shows all active runs, or latest if none active)
syndiff-template progress
syndiff-template status --watch

# Or one specific run
syndiff-template progress --run-id my_batch_01
syndiff-template tail --run-dir /path/to/template_handoffs/runs/my_batch_01 \
  --target s0023_c1_k3_2020ftl --stage ps1_process

# 4. Workspace overview (zero flags OK when one supervisor is running)
syndiff-template active
syndiff-template runs
```

## Command reference

### Starting runs

| Command | Usage |
|---------|-------|
| **submit** | `syndiff-template submit --config CONFIG --targets TARGETS [--stages LIST] [--run-id ID] [--force-rerun]` |
| **run** | Same flags as submit; blocks in foreground (debug) |

`--stages` is a comma-separated list (`mapping,ps1_process,downsample`). Default: all six stages.

### Monitoring runs

Default (no flags): all **active** runs in the auto-discovered workspace (latest run if none active).

| Command | Usage |
|---------|-------|
| **progress** | `syndiff-template progress` or `--run-id ID` or `--run-dir RUN` |
| **status** | `syndiff-template status [--watch] [--interval 10]` or `--run-id` / `--run-dir` |
| **show** | `syndiff-template show --run-dir RUN` — prints `run_meta.json` |
| **logs** | `syndiff-template logs --run-dir RUN` — daemon log |
| **logs** | `syndiff-template logs --run-dir RUN --target LABEL --stage STAGE` |
| **tail** | Same as `logs --follow` for a stage log |

Target labels look like `s0023_c1_k3_2020ftl` (sector, camera, ccd, target name).

### Controlling a run

| Command | Usage |
|---------|-------|
| **retry** | `syndiff-template retry --run-dir RUN` — all failed/canceled |
| **retry** | `syndiff-template retry --run-dir RUN --scc 23,1,3 --stage mapping` — one stage |
| **pause** | `syndiff-template pause --run-dir RUN` |
| **resume** | `syndiff-template resume --run-dir RUN` |
| **kill** | `syndiff-template kill --run-dir RUN` — cancel + stop workers |

### Workspace-wide

Use optional `--deployment path/to/deployment.yaml`, or omit it when exactly one supervisor is running:

| Command | Usage |
|---------|-------|
| **runs** | `syndiff-template runs [--deployment DEPLOYMENT] [--limit 20]` |
| **active** | `syndiff-template active [--deployment DEPLOYMENT]` |
| **daemon start** | `syndiff-template daemon start [--deployment DEPLOYMENT]` |
| **daemon stop** | `syndiff-template daemon stop [--deployment DEPLOYMENT]` |
| **daemon status** | `syndiff-template daemon status [--deployment DEPLOYMENT]` |

### Verification

| Command | Usage |
|---------|-------|
| **verify** | `syndiff-template verify --config config.yaml --targets targets_example.csv` |
| **verify** | `syndiff-template verify --run-dir RUN --scc 23,1,3 --stages ps1_process` |
| **reconcile-manifests** | `syndiff-template reconcile-manifests --config config.yaml --targets targets_example.csv` |

### Discord (optional)

Requires `notifications.enabled: true` in `config.yaml` and keys in `deployment.yaml`.

| Command | Usage |
|---------|-------|
| **notify test** | `syndiff-template notify test --run-dir RUN [--dry-run] [-v]` |
| **discord bot** | `syndiff-template discord bot --deployment deployment.yaml` (foreground; normally auto-started) |

## Variant configs

**Stream mode** — download PS1 skycells inside `ps1_process` (no shared Zarr). In `config.yaml`, set `stages.ps1_process.ps1_source: stream` and `num_ingest_workers: 16` (see commented lines in that file):

```bash
syndiff-template submit \
  --config config.yaml \
  --targets targets_example.csv \
  --stages mapping,ps1_process,downsample
```

**ROI test** — small WCS crop + downsample overrides:

```bash
syndiff-template submit \
  --config config_s0020_c3_k3_roi.yaml \
  --targets targets_s0020_c3_k3_roi.csv \
  --run-id s0020_roi_test
```

## Smoke testing

Limit work on one SCC via `overrides` in `config.yaml` (`projections_limit: 1`) or submit a short `--stages` list after verifying upstream artifacts exist.

Use `--force-rerun` to re-run `ps1_process` from a clean convolved Zarr. See [force rerun](../../docs/template_pipeline.md#force-rerun-behavior).
