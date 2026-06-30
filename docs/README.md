# SynDiff documentation

Documentation for the **syndiff-pipeline** open-source release.

## Start here

| Document | Audience | Contents |
|----------|----------|----------|
| [Main README](../README.md) | All users | Project overview, pyhotpants + custom MOCPy, install, quick start |
| [Unified pipeline guide](template_pipeline.md) | All users | `syndiff` CLI, 7-stage DAG, Condor, config, workspace layout |
| [Storage layout](storage_layout.md) | All users | `workspace_root` + `data_root` filesystem reference |
| [`syndiff` CLI reference](syndiff_cli.md) | All users | Noun/verb commands, stages, internal modules |
| [Cluster smoke checklist](cluster_smoke_checklist.md) | Ops | Manual validation on HTCondor + NFS after setup |
| [Orchestration architecture](template_runner_architecture.md) | Maintainers | Spec-driven scheduler, SQLite state machine, verify/launch internals |
| [State machine reference](pipeline_state_machine_reference.md) | Maintainers | Status transitions, partial runs, retry/cancel matrices |
| [Config guide](../config/README.md) | All users | `pipeline.yaml`, `diff_config.yaml`, deployment, example YAMLs |
| [Site configs](../config/) | Quick start | `pipeline.yaml`, `diff_config.yaml`, deployment template |

## Template pipeline — two documentation layers

The pipeline has **orchestration docs** (how to run many SCCs with `syndiff all|template|diff`) and **algorithm docs** (what each stage does internally).

```
docs/
├── template_pipeline.md          ← orchestration, scheduler, Condor, config, run lifecycle
├── syndiff_cli.md                ← CLI noun/verb reference
└── stages/
    ├── README.md                 ← index + script/module mapping
    ├── standalone_pipeline_overview.md   ← legacy single-FFI pipeline.py workflow
    ├── mapping_pancakes.md       ← PanCAKES v2 (TESS↔PS1 pixel mapping)
    ├── ps1_process_technical.md  ← sliding-window convolution architecture
    ├── downsample_technical.md   ← multi-offset downsampling onto TESS grid
    └── phot_bkg_temporal_smooth.md  ← Savitzky–Golay temporal smooth of ks_b (kernel_subtract bkg)
```

The stage deep-dives were originally maintained in the standalone [`syndiff`](../../syndiff/) research repository (`README_pancakes.md`, `README_process_ps1.md`, `README_downsample_offset.md`). They are copied here so this release is self-contained.

## Code lineage

| Legacy script (`syndiff/`) | Package module (`syndiff_pipeline/`) | `syndiff` stage |
|----------------------------|--------------------------------------|--------------------------|
| `pancakes_v2.py` | `template_creation/processing/pancakes.py` | `mapping` |
| `download_and_store_zarr.py` | `template_creation/processing/ps1_download.py` | `ps1_download` |
| `process_ps1.py` | `template_creation/processing/ps1_process.py` | `ps1_process` |
| `multi_offset_downsampling.py` | `template_creation/processing/downsample.py` | `downsample` |
| — | `template_creation/orchestration/handoff.py` + `common/wcs_grouping.py` | `wcs_grouping` |
| — | `common/download.py` | `tess_ffi_download` |
| — | `difference_imaging/orchestration/execute.py` | `diff` |

The **`syndiff` orchestrator** adds WCS grouping for transient targets, a unified 7-stage DAG (template + diff), SQLite bookkeeping, resource pools, detached scheduling, artifact verification, and HTCondor for `mapping`, `ps1_process`, and `diff`. The core science algorithms match the standalone scripts.

## Example diff configs

Materialized diff configs under [`config/example/`](../config/example/) are for foreground `python -m syndiff_pipeline.difference_imaging.orchestration.cli --config …` (not `syndiff diff run`, which reads live site `diff_config.yaml`). Legacy recipe YAMLs are in [`config/example/legacy/`](../config/example/legacy/). See [`config/README.md`](../config/README.md).
