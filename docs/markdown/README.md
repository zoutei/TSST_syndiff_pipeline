# SynDiff documentation

Documentation for the unreleased **syndiff-pipeline** project.

## Start here

| Document | Audience | Contents |
|----------|----------|----------|
| [Main README](../../README.md) | All users | Project overview, pyhotpants + custom MOCPy, install, quick start |
| [Unified pipeline guide](template_pipeline.md) | All users | `syndiff` CLI, five-stage template DAG + two-stage diff DAG (`bind`→`diff`), independent star branch, Condor, config, workspace layout |
| [Host-star light curves](star_lightcurves.md) | All users | `syndiff star submit|run`, config, prerequisites, outputs |
| [Storage layout](storage_layout.md) | All users | `workspace_root` + `data_root` (SCC + nested-event layout) filesystem reference |
| [Field (distortion-aware) geometry](field_geometry.md) | Users / maintainers | `geometry_mode: field` — per-skycell drift, hybrid Exact, SCC contrib store |
| [Oversampled templates + Hotpants stamp modes](oversampled_templates.md) | Users / maintainers | `oversampling_factor` (`F`), native vs HR coords, Hotpants `stamp_mode` / `region_*`, star OS |
| [`syndiff` CLI reference](syndiff_cli.md) | All users | Noun/verb commands, stages, internal modules |
| [Cluster smoke checklist](cluster_smoke_checklist.md) | Ops | Manual validation on HTCondor + NFS after setup |
| [Orchestration architecture](template_runner_architecture.md) | Maintainers | Spec-driven scheduler, SQLite state machine, verify/launch internals |
| [State machine reference](pipeline_state_machine_reference.md) | Maintainers | Status transitions, partial runs, retry/cancel matrices |
| [Config guide](../../config/README.md) | All users | `pipeline.yaml`, `diff_config.yaml`, deployment, example YAMLs |
| [Site configs](../../config/) | Quick start | `pipeline.yaml`, `diff_config.yaml`, deployment template |

## Template pipeline — two documentation layers

The pipeline has **orchestration docs** (how to run SCCs with `syndiff template` and events with `syndiff diff`) and **algorithm docs** (what each stage does internally).

```
docs/markdown/
├── template_pipeline.md          ← orchestration, scheduler, Condor, config, run lifecycle
├── syndiff_cli.md                ← CLI noun/verb reference
├── storage_layout.md             ← SCC + nested-event on-disk layout
├── field_geometry.md             ← distortion-aware templates (geometry_mode: field)
├── oversampled_templates.md      ← F>1 templates + Hotpants stamp_mode / region_*
└── stages/
    ├── README.md                 ← index + script/module mapping
    ├── standalone_pipeline_overview.md   ← legacy single-FFI pipeline.py workflow
    ├── tess_ffi_download.md      ← FFI download stage
    ├── wcs_grouping.md           ← drift measurement, template groups, reference FFI, crop (now the `bind` stage)
    ├── mapping_pancakes.md       ← PanCAKES (TESS↔PS1 pixel mapping) + Gaia download
    ├── ps1_process_technical.md  ← sliding-window convolution architecture + star removal
    ├── downsample_technical.md   ← multi-offset downsampling onto TESS grid (now the `templates` stage)
    ├── diff_pipeline.md          ← diff internal sub-stages, kernels, photometry
    ├── star_pipeline.md          ← host-star branch (technical)
    ├── star_config.md            ← star_config.yaml / star_targets schema
    └── background.md             ← Savitzky–Golay temporal smooth of ks_b (kernel_subtract bkg)
├── star_lightcurves.md           ← host-star quick start (syndiff star)
```

**Improvement plans** live under [`.cursor/plans/`](../../.cursor/plans/).
Field (distortion-aware) templates are implemented — see [field_geometry.md](field_geometry.md)
(including hybrid Exact cache-key / reuse notes). Historical design memos under
`doc/` were retired into that guide.
The historical target-star plan is superseded by the implemented `syndiff star` branch.

The stage deep-dives originated in an earlier standalone research workflow and
are vendored here so this repository is self-contained.

## Code lineage

| Legacy script (`syndiff/`) | Package module (`syndiff_pipeline/`) | `syndiff` stage |
|----------------------------|--------------------------------------|--------------------------|
| `pancakes_v2.py` | `template_creation/processing/pancakes.py` + `processing/scc_reference_ffi.py` | `mapping` |
| `download_and_store_zarr.py` | `template_creation/processing/ps1_download.py` | `ps1_download` |
| `process_ps1.py` | `template_creation/processing/ps1_process.py` | `ps1_process` |
| `multi_offset_downsampling.py` | `template_creation/processing/downsample.py` (+ `field_downsample.py`) | `templates` (legacy config key/alias: `downsample`) |
| — | `difference_imaging/orchestration/bind.py` + `common/wcs_grouping.py` | `bind` (diff DAG; legacy alias: `wcs_grouping`) |
| — | `common/download.py` | `tess_ffi_download` |
| — | `difference_imaging/orchestration/execute.py` | `diff` |

The **`syndiff` orchestrator** adds event-scoped WCS grouping (`bind`) for
transients, an SCC-scoped five-stage template DAG plus a two-stage diff DAG
(`bind`→`diff`) and an independent `star` branch, SQLite bookkeeping, resource
pools, detached scheduling, artifact verification, and HTCondor for
`mapping`, `ps1_process`, `diff`, and `star`. The core template science
algorithms match the standalone scripts.

## Example diff configs

Materialized diff configs under [`config/example/`](../../config/example/) are for foreground `python -m syndiff_pipeline.difference_imaging.orchestration.cli --config …` (not `syndiff diff run`, which reads live site `diff_config.yaml`). Legacy recipe YAMLs are in [`config/example/legacy/`](../../config/example/legacy/). See [`config/README.md`](../../config/README.md).
