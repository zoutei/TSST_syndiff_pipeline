# SynDiff documentation

Documentation for the unreleased **syndiff-pipeline** project.

## Start here

| Document | Audience | Contents |
|----------|----------|----------|
| [Main README](../../README.md) | All users | Project overview, install, quick start |
| [Unified pipeline guide](template_pipeline.md) | All users | `syndiff` CLI, template DAG + diff + photometry + star, Condor, config |
| [`syndiff` CLI reference](syndiff_cli.md) | All users | Noun/verb commands, stages, operators |
| [Storage layout](storage_layout.md) | All users | `workspace_root` + `data_root` filesystem reference |
| [Bookkeeping (provenance graph)](bookkeeping.md) | All users / ops | Fingerprints, spool/ingest, skip/resume, CLI |
| [Field (distortion-aware) templates](field_geometry.md) | Users / maintainers | Default `geometry_mode: field` — L0–L5, storage, verify, ops |
| [Event photometry](photometry.md) | All users | `syndiff photometry submit\|run`, prerequisites, outputs |
| [Host-star light curves](star_lightcurves.md) | All users | `syndiff star submit\|run`, config, prerequisites |
| [Oversampled templates + Hotpants stamp modes](oversampled_templates.md) | Users / maintainers | `oversampling_factor` (`F`), Hotpants `stamp_mode` |
| [Linear → centroids campaign](linear_centroids_pipeline.md) | Users / maintainers | Phase 1 linear + centroids; phase 2 TV WCS planned |
| [Static masking](masking.md) | Diff/star users | Empirical/TNS/asteroid masks; `mask_settings.yaml` |
| [Cluster smoke checklist](cluster_smoke_checklist.md) | Ops | HTCondor + NFS validation |
| [Orchestration architecture](template_runner_architecture.md) | Maintainers | Scheduler, SQLite, verify/launch |
| [State machine reference](pipeline_state_machine_reference.md) | Maintainers | Status transitions, retry/cancel |
| [Config guide](../../config/README.md) | All users | Site YAML layout and examples |

## Two documentation layers

**Orchestration** (how to run) vs **algorithms** (what each stage does):

```
docs/markdown/
├── template_pipeline.md          ← orchestration, scheduler, Condor, run lifecycle
├── syndiff_cli.md                ← CLI noun/verb reference
├── storage_layout.md             ← SCC + nested-event on-disk layout
├── bookkeeping.md                ← provenance graph
├── field_geometry.md             ← default field templates (L0–L5)
├── oversampled_templates.md      ← F>1 templates + Hotpants stamp modes
├── masking.md                    ← empirical/TNS/asteroid masks
├── photometry.md                 ← event photometry quick start
├── star_lightcurves.md           ← host-star quick start
├── linear_centroids_pipeline.md  ← linear bootstrap → centroids campaign
└── stages/
    ├── README.md                 ← stage index + module mapping
    ├── tess_ffi_download.md
    ├── wcs_grouping.md           ← linear-mode drift reference (field: field_geometry)
    ├── mapping_pancakes.md
    ├── ps1_download.md
    ├── ps1_process_technical.md
    ├── downsample_technical.md   ← downsample stage (product path: templates/)
    ├── diff_pipeline.md
    ├── multi_kernel_diff.md
    ├── gridded_epsf.md
    ├── centroids.md
    ├── forced_photometry.md
    ├── background.md
    ├── photometry_pipeline.md
    ├── photometry_config.md
    ├── star_pipeline.md
    ├── star_config.md
    └── standalone_pipeline_overview.md   ← LEGACY
```

Field (distortion-aware) templates are implemented — see [field_geometry.md](field_geometry.md).
Historical design notes: [`doc/padded_scc_v2_implementation.md`](../../doc/padded_scc_v2_implementation.md),
[`doc/template_bookkeeping_plan.md`](../../doc/template_bookkeeping_plan.md).
Improvement plans live under [`.cursor/plans/`](../../.cursor/plans/).

## Code lineage

| Legacy script (`syndiff/`) | Package module (`syndiff_pipeline/`) | `syndiff` stage |
|----------------------------|--------------------------------------|-----------------|
| `pancakes_v2.py` | `template_creation/processing/pancakes.py` | `mapping` |
| `download_and_store_zarr.py` | `template_creation/processing/ps1_download.py` | `ps1_download` |
| `process_ps1.py` | `template_creation/processing/ps1_process.py` | `ps1_process` |
| — | `template_creation/processing/field_remap.py` | `remap` |
| `multi_offset_downsampling.py` | `field_downsample.py` / `linear_downsample.py` | **`downsample`** (product dir still `templates/`) |
| — | `difference_imaging/orchestration/execute.py` | `diff` |
| — | `photometry/runner.py` | `photometry` |
| — | `star/runner.py` | `star` |

The **`syndiff` orchestrator** runs a six-stage template DAG, SCC-primary `diff`, independent **`photometry`** and **`star`** branches, SQLite bookkeeping, and HTCondor for heavy stages.

## Example configs

Site configs under [`config/`](../../config/). Materialized diff examples under [`config/example/`](../../config/example/) are for foreground debugging. See [`config/README.md`](../../config/README.md).
