# Event photometry (`syndiff photometry`)

Quick start for the **photometry** orchestrator stage: forced photometry (and optional astrometry) on completed SCC difference-imaging products.

> **Related:** [CLI reference](syndiff_cli.md) · [photometry pipeline (technical)](stages/photometry_pipeline.md) · [photometry config](stages/photometry_config.md) · [forced photometry methods](stages/forced_photometry.md) · [storage layout](storage_layout.md)

## When to use it

| Goal | Command |
|------|---------|
| Build SCC templates | `syndiff template submit …` |
| Subtract (Hotpants / kernel stack) | `syndiff diff submit --scc …` |
| Astrometry + forced LCs on events | **`syndiff photometry submit\|run`** |
| Host-star LCs | `syndiff star submit\|run` |

The default schema v1 site config [`config/diff_config.yaml`](../../config/diff_config.yaml) is **`shared_mask` + `hotpants` only**. Event light curves are **not** part of that default; they run via this noun (or, on schema v1 sites only, an optional in-diff `kind: photometry` delegator — a schema v2 `diff.pipeline` rejects that kind outright, since diff is SCC-scoped, not event-scoped; see [diff pipeline](stages/diff_pipeline.md) and [config_schema_v2.md](config_schema_v2.md)).

## Prerequisites

1. Template + remap + downsample complete for the SCC (`geometry_mode: field` default).
2. Diff lane complete under `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/` (e.g. `hp_d/`, optional `epsf_r1/` for gepsf).
3. Handoff present: `{data_root}/…/bookkeeping/diff/{frames.csv,diff_job.json}` (schema v2 + MappingGrid).
4. Site files: `deployment.yaml`, `pipeline.yaml`, **`photometry_config.yaml`** (or pass `--photometry-config`), and an event **`--targets`** CSV.

Verify gates check the named SCC diff lane before launch — see [photometry pipeline](stages/photometry_pipeline.md#verify).

**Submit only once those prerequisites are actually on disk.** Unlike `diff`
(which waits on `downsample`), `photometry` has no declared upstream
dependency and will attempt a real Condor launch immediately on submit —
launching before the diff lane exists means repeated real Condor submissions
that fail fast and retry with a short backoff before giving up, not a benign
hold. Use [`scripts/submit_photometry_when_ready.py`](../../scripts/submit_photometry_when_ready.py)
instead of `syndiff photometry submit` directly when any target's diff lane
might not be finished yet — it polls the same on-disk readiness check the
daemon uses and only submits a target (one Condor run per SCC, via a
single-row targets CSV) once it will actually succeed:

```bash
python scripts/submit_photometry_when_ready.py \
  --photometry-config config/photometry_config.yaml \
  --targets config/targets_example.csv \
  --run-id-prefix my_phot_run
```

See [photometry pipeline §No automatic upstream gate](stages/photometry_pipeline.md#no-upstream-gate)
for why this is necessary.

## Quick start

```bash
mamba activate syndiff

# Supervised batch (daemon + Condor by default)
syndiff photometry submit \
  --site config/ \
  --photometry-config config/photometry_config_2020ut_gepsf_lc.yaml \
  --targets config/targets_example.csv \
  --run-id phot_2020ut_gepsf

# Foreground, one event
syndiff photometry run \
  --site config/ \
  --photometry-config config/photometry_config_2020ut_gepsf_lc.yaml \
  --targets config/targets_example.csv \
  --target-name s0020_c3_k3_2020ut
```

Common flags: `--force-rerun`, `--local` (sets `stages.photometry.executor: local` on the frozen run config).

Monitor like other runs: `syndiff status --site config/`, `syndiff progress`, `syndiff logs --run-id …`. Photometry is **not** a column in the default status grid (`tess_dl … diff`); use `progress` / per-target `photometry.log`.

## Outputs

Under `{workspace_root}/events/{event}/s{SSSS}_c{C}_k{K}/phot_{photometry_run_id}/`:

| Path | Contents |
|------|----------|
| `astrometry_result.json` | When `kind: astrometry` is in the photometry pipeline |
| `targets.reg` | DS9 regions for forced targets |
| `{lc_label}/lightcurve_*.csv` | Forced-photometry light curves |
| `debug_plots/` | Optional LC / QA plots when `pipeline_plots: true` |

Diff FITS stay SCC-primary under `data_root`; photometry does not re-run Hotpants.

## Config sketch

```yaml
deployment_file: deployment.yaml
defaults:
  photometry_run_id: my_phot_run
  n_jobs: 16
paths:
  inputs:
    diffs: hp_d
    epsf: epsf_r1          # required for psf_type: epsf / gepsf
pipeline:
  - kind: forced_photometry
    inputs: {diffs: hp_d, epsf: epsf_r1}
    output: lc_gepsf
    methods:
      - name: gepsf
        type: psf
        psf_type: epsf
        fit_shape: 11
```

Full schema: [photometry config](stages/photometry_config.md). Examples: [`config/photometry_config_*.yaml`](../../config/).

## Related stages

- Diff subtract: [diff pipeline](stages/diff_pipeline.md), [multi-kernel path](stages/multi_kernel_diff.md)
- ePSF models: [gridded ePSF](stages/gridded_epsf.md)
- Method knobs: [forced photometry](stages/forced_photometry.md)
- Host stars (separate branch): [star light curves](star_lightcurves.md)
