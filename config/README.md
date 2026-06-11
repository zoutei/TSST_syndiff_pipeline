# SynDiff configuration

This directory is the **config root** passed to `syndiff --site config`.

## Files

| File | Role |
|------|------|
| `pipeline.yaml` | Orchestrator policy: 7-stage DAG params, resource pools, scheduler, notifications |
| `diff_config.yaml` | Site diff policy: `pipeline:` stage list, defaults (`n_jobs`), SCC overrides, Condor |
| `deployment.yaml` | Gitignored: `workspace_root`, `data_root`, credentials (copy from `deployment.yaml.example`) |
| `targets_example.csv` | Example targets list for `--targets` |

## Foreground diff (two entry points)

| Path | When to use | Command |
|------|-------------|---------|
| **Site policy** | Normal debugging with live `diff_config.yaml` | `syndiff diff run --site config --targets targets_example.csv --target-name 2020ut` |
| **Materialized YAML** | Frozen per-target config with absolute paths | `python -m syndiff_pipeline.difference_imaging.orchestration.cli --config example/diff_config_a_prf.yaml` |

Materialized examples live under `example/diff_config_*.yaml`; legacy names are in `example/legacy/recipe_*.yaml` (read-only reference).

## Supervised pipeline

```bash
cp config/deployment.yaml.example config/deployment.yaml   # first time
syndiff all submit --site config --targets config/targets_example.csv --run-id my_run
```

## Runtime frozen configs

On submit, the orchestrator copies policy into the workspace:

- `{workspace_root}/runs/{run_id}/config.yaml` — frozen orchestrator
- `{workspace_root}/runs/{run_id}/per_target/{label}/diff_config.yaml` — frozen per-target diff

See [docs/storage_layout.md](../docs/storage_layout.md).
