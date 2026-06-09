# SynDiff documentation

Documentation for the **syndiff-pipeline** open-source release.

## Start here

| Document | Audience | Contents |
|----------|----------|----------|
| [Main README](../README.md) | Diff-imaging users | Hotpants, ePSF, background, forced photometry (`run_pipeline.py`) |
| [Template pipeline guide](template_pipeline.md) | Template builders | Multi-target orchestration (`syndiff-template`), Condor, config, CLI |
| [batch_no2 PS1 download investigation](batch_no2_ps1_download_investigation.md) | Ops / debugging | NFS Zarr lock contention, config tuning, restart after kill |
| [Example configs](../example/template_runner/README.md) | Quick start | Setup, workflow, per-command cheat sheet |

## Template pipeline — two documentation layers

The template workflow has **orchestration docs** (how to run many SCCs with `syndiff-template`) and **algorithm docs** (what each stage does internally).

```
docs/
├── template_pipeline.md          ← orchestration, scheduler, Condor, config, full CLI reference
└── stages/
    ├── README.md                 ← index + script/module mapping
    ├── standalone_pipeline_overview.md   ← legacy single-FFI pipeline.py workflow
    ├── mapping_pancakes.md       ← PanCAKES v2 (TESS↔PS1 pixel mapping)
    ├── ps1_process_technical.md  ← sliding-window convolution architecture
    └── downsample_technical.md   ← multi-offset downsampling onto TESS grid
```

The stage deep-dives were originally maintained in the standalone [`syndiff`](../../syndiff/) research repository (`README_pancakes.md`, `README_process_ps1.md`, `README_downsample_offset.md`). They are copied here so this release is self-contained.

## Code lineage

| Legacy script (`syndiff/`) | Package module (`syndiff_pipeline/`) | `syndiff-template` stage |
|----------------------------|--------------------------------------|--------------------------|
| `pancakes_v2.py` | `template/pancakes.py` | `mapping` |
| `download_and_store_zarr.py` | `template/ps1_download.py` | `ps1_download` |
| `process_ps1.py` | `template/ps1_process.py` | `ps1_process` |
| `multi_offset_downsampling.py` | `template/downsample.py` | `downsample` |
| — | `template_runner/handoff.py` + `wcs_grouping.py` | `wcs_grouping` |
| — | `download.py` | `tess_ffi_download` |

The **`syndiff-template` runner** adds WCS grouping for transient targets, SQLite bookkeeping, resource pools, detached scheduling, artifact verification, and HTCondor submission for `ps1_process`. The core science algorithms match the standalone scripts.

## Example recipes (diff imaging)

YAML pipeline recipes for `run_pipeline.py` live under [`example/`](../example/README.md).
