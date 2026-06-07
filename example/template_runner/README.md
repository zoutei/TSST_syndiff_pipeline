# Template pipeline examples

These files configure the **`syndiff-template`** CLI, which builds PS1 templates on the TESS pixel grid before running the main SynDiff difference-imaging pipeline.

## Documentation

| Guide | Contents |
|-------|----------|
| [Template pipeline](../docs/template_pipeline.md) | Config, scheduler, Condor, CLI, troubleshooting |
| [Stage algorithms](../docs/stages/README.md) | PanCAKES, PS1 process, downsample deep-dives (from `../syndiff/`) |
| [Docs index](../docs/README.md) | Full documentation map |

## Files

| File | Description |
|------|-------------|
| [`config_example.yaml`](config_example.yaml) | Starter config with placeholder paths and comments |
| [`config_real.yaml`](config_real.yaml) | Production-style config (STScI cluster paths, Condor settings) |
| [`targets_example.csv`](targets_example.csv) | Normalized targets (sector, camera, ccd, coordinates, enabled flag) |
| [`events_example.csv`](events_example.csv) | Supernova catalog format with `tess_coverage` tokens |

## Minimal workflow

```bash
mamba activate syndiff
pip install -e ../..   # from repo root, once

# Edit paths in config_example.yaml, then:
syndiff-template verify \
  --config config_example.yaml \
  --targets targets_example.csv

syndiff-template submit \
  --config config_example.yaml \
  --targets targets_example.csv \
  --stages ps1_process,downsample

syndiff-template progress --config config_example.yaml
syndiff-template status --watch --config config_example.yaml
```

## Smoke testing

Limit work on one SCC via `overrides` in `config_real.yaml` (`projections_limit: 1`) or run a single stage subset after verifying upstream artifacts exist.

Use `--force-rerun` to re-run `ps1_process` from a clean convolved Zarr (auto-deletes Zarr + removed-stars CSV). See the [force rerun section](../../docs/template_pipeline.md#force-rerun-behavior) in the full docs.
