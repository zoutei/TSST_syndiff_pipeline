# SynDiff Pipeline

[![Docs](https://github.com/zoutei/TSST_syndiff_pipeline/actions/workflows/docs.yml/badge.svg)](https://zoutei.github.io/TSST_syndiff_pipeline/)

TESS Full Frame Image (FFI) template building and difference-imaging pipeline for transient detection and forced photometry.

> **This project has not been released.** All modules are under active development.

---

## Overview

SynDiff is an end-to-end pipeline for TESS Full Frame Image (FFI) transient work: it builds PS1-based templates on the TESS pixel grid, runs SCC-primary difference imaging, then optional **event photometry** and **host-star** light curves. Template building is **SCC-scoped** (sector/camera/CCD). CLI nouns are separate: `syndiff template`, `syndiff diff`, `syndiff photometry`, `syndiff star` — there is no combined "run everything" preset.

### Template creation (TESS FFIs + PS1 → SCC-shared template store)

`syndiff template submit --scc sccs.csv` (SCC-only input; no event coordinates):

1. **FFI download** (`tess_ffi_download`) — calibrated TESS FFIs from MAST into `{data_root}/s{SSSS}/c{C}/k{K}/ffi/`.
2. **Mapping (PanCAKES)** (`mapping`) — reference FFI + TESS↔PS1 skycell maps + Gaia. Uses a customized **[MOCPy](#forked-dependencies)** fork.
3. **PS1 download** (`ps1_download`) — PS1 skycells into a shared Zarr store (skipped when `ps1_source: stream`).
4. **PS1 process** (`ps1_process`) — convolve PS1 onto the TESS grid (often HTCondor).
5. **Remap** (`remap`) — field-mode L2–L4 drift / Exact cache (skipped when `geometry_mode: linear`).
6. **Downsample** (`downsample`) — L5 templates under `{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/` (stage name is `downsample`; directory name remains `templates/`).

### Field-mode difference imaging (`syndiff diff submit --scc sccs.csv`)

After templates exist (with `MAPGRID=2` and `field_mode_assembly.json` schema v3), **`diff`** runs `scc_bootstrap` inside execute and writes SCC-primary products under `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/`. Default site `diff_config.yaml` is `shared_mask` → `hotpants`. See [`docs/markdown/field_geometry.md`](docs/markdown/field_geometry.md).

### Event photometry (`syndiff photometry`)

Forced photometry (and optional astrometry) on completed SCC diff lanes → `events/{event}/s…/phot_{run_id}/`. See [`docs/markdown/photometry.md`](docs/markdown/photometry.md).

### Host-star light curves (`syndiff star`)

Independent branch that reuses Hotpants kernels / convolved templates / backgrounds without re-running Hotpants. See [`docs/markdown/star_lightcurves.md`](docs/markdown/star_lightcurves.md).

### Difference imaging sub-stages

The **`diff`** stage runs a YAML-ordered pipeline ([`config/diff_config.yaml`](config/diff_config.yaml)). A fuller recipe might include:

1. **Shared masking** — bitmask from Gaia / straps / optional TNS / asteroids.
2. **Image differencing** — **[pyhotpants](#forked-dependencies)** or the multi-kernel path.
3. **ePSF / centroids** — optional gridded ePSF and Gaia centroid tables on the SCC lane.
4. **Background / subtract** — unified background and algebraic plane combinations.

Optional steps you can add to the `pipeline:` list:

- **Empirical PSF fitting** — tiled gridded ePSF via photutils (`epsf` stage; see [`docs/markdown/stages/gridded_epsf.md`](docs/markdown/stages/gridded_epsf.md)).
- **Gaia centroids** — PSF photometry on bright Gaia stars (`centroids` stage).
- **Saturated star templates** — known limitations; see [`docs/markdown/stages/diff_pipeline.md`](docs/markdown/stages/diff_pipeline.md) §6.
- **Background removal** — unified `background` stage (spatial / temporal / strap).
- **Second round of differencing** — Hotpants again on background-subtracted science images.

Run templates with `syndiff template submit --scc sccs.csv`, SCC subtract with `syndiff diff submit --scc sccs.csv`, event LCs with `syndiff photometry submit --targets …`, host stars with `syndiff star submit`.

---

## Forked dependencies

SynDiff relies on two customized libraries that are **not** satisfied by the stock PyPI packages alone.

### pyhotpants (difference imaging)

The diff stage uses **[pyhotpants](https://github.com/zoutei/pyhotpants)** — a Python/C wrapper around HOTPANTS for Alard–Lupton kernel-matching subtraction. The PyPI package is **`hotpants`** (not `pyhotpants`):

1. **Default:** `pip install -e .` installs `hotpants>=0.1.1` from PyPI automatically.
2. **From source:** `pip install git+https://github.com/zoutei/pyhotpants`
3. **Dev checkout:** clone that repo and either `pip install -e` it, or place a `pyhotpants/` directory where the import fallback in `difference_imaging/stages/hotpants.py` can find it.

Used for: per-frame FFI vs PS1 template differencing, optional second-pass subtraction, polynomial background products.

### Custom MOCPy (template mapping)

The **mapping** (PanCAKES) stage requires a **modified MOCPy** build with `MOC.filter_points_in_polygons` (Rust backend). Standard `pip install mocpy` does not include this API.

- Source: [github.com/zoutei/mocpy_syndiff](https://github.com/zoutei/mocpy_syndiff/)
- Install: follow that repository’s build instructions (Rust + `maturin develop --release`), or see [`docs/markdown/stages/mapping_pancakes.md`](docs/markdown/stages/mapping_pancakes.md) and [`docs/markdown/stages/standalone_pipeline_overview.md`](docs/markdown/stages/standalone_pipeline_overview.md#custom-mocpy-installation).

### Other external packages

| Package | Role |
|---------|------|
| **TGLC** | ePSF fitting (`tglc.effective_psf`); must be importable (clone + `PYTHONPATH`) |
| **PRF** | Optional; official TESS PRF when `psf_type: prf` |
| **numpy, pandas, astropy, scipy, matplotlib, joblib, pyyaml, scikit-image** | Core diff-imaging stack (conda-forge / pip) |

---

## Installation

```bash
mamba activate syndiff
pip install -e .    # registers `syndiff`; installs hotpants>=0.1.1
```

For full template + diff runs, also install **custom MOCPy** (above) and ensure **TGLC** is on `PYTHONPATH` if using ePSF stages.

---

## Unified SynDiff pipeline (`syndiff`)

The **`syndiff`** CLI orchestrates the full workflow behind one supervisor daemon and one SQLite state database — template building (TESS FFIs + PS1 → SCC-shared template store) and difference imaging (config-driven Hotpants → photometry). `template` and `diff` are separate DAGs with separate submits:

```text
syndiff template submit --scc sccs.csv                   # template stages only
syndiff diff submit --scc sccs.csv                       # field-mode subtract (default --stages diff)
syndiff diff submit --targets t.csv                      # event photometry under events/{name}/ws/
syndiff progress                                          # monitoring works the same for any run
```

Foreground debugging: `syndiff diff run --site config --scc sccs.csv` for SCC-only, or `--targets t.csv --target-name 2020ut` for one event.

| | Foreground (`syndiff diff run`) | Supervised (`syndiff * submit`) |
|---|--------------------------------|----------------------------------|
| **Purpose** | One target, current process | Multi-target batch + daemon |
| **Config** | `--site config` (site policy) | `--site` → `pipeline.yaml` + `diff_config.yaml` + `deployment.yaml` |
| **State** | No SQLite | `{workspace_root}/control/pipeline_state.sqlite` |
| **Outputs** | `events/{event_name}/s{SSSS}_c{C}_k{K}/ws/` | Same layout under `workspace_root` |

### Setup (first time)

Use the site folder at `config/`:

| File | Git | Contains |
|------|-----|----------|
| `pipeline.yaml` | committed | Template policy: stages, pools, notifications |
| `diff_config.yaml` | committed | Diff pipeline policy + Condor resources |
| `deployment.yaml` | **gitignored** | `workspace_root`, `data_root`, Gaia + Discord credentials |

```bash
cp config/deployment.yaml.example config/deployment.yaml
# Edit workspace_root, data_root, optional gaia_username/password, Discord keys
```

Targets are always passed on the CLI (`--scc` for template, `--targets` for diff/star), never embedded in config files.

### Quick start

```bash
mamba activate syndiff

syndiff verify --site config --targets config/targets_example.csv

syndiff template submit \
  --site config \
  --scc config/scc_example.csv \
  --run-id batch_no5

syndiff progress
syndiff status --watch

# once templates exist, run field-mode diff on the SCC
syndiff diff submit \
  --site config \
  --config config/diff_config_single_kernel.yaml \
  --scc config/scc_example.csv \
  --run-id diff_batch_no5

syndiff retry --deployment config/deployment.yaml --run-id diff_batch_no5 \
  --scc s0023_c1_k3_2020ftl --stage diff
```

### Command overview

| Pattern | Examples |
|---------|----------|
| **Execute** | `syndiff template submit`, `syndiff diff submit --scc` or `--targets`, `syndiff diff run`, `syndiff star submit` |
| **Monitor** | `syndiff progress`, `syndiff status --watch`, `syndiff logs`, `syndiff tail` |
| **Control** | `syndiff retry`, `syndiff pause`, `syndiff resume`, `syndiff kill` |
| **Workspace** | `syndiff runs`, `syndiff active`, `syndiff daemon status`, `syndiff verify` |

---

## Difference imaging (config-driven)

After PS1 templates exist, the orchestrator **`diff`** stage runs a YAML-ordered
diff policy (`shared_mask`, `hotpants`, `epsf`, `background`,
`forced_photometry`, …). Foreground `--site config` uses
[`config/diff_config.yaml`](config/diff_config.yaml); supervised submit uses
the `diff_config:` path selected by `pipeline.yaml`. Field-mode diff uses
`scc_bootstrap` for geometry (`bookkeeping/diff/diff_job.json`); templates
are resolved directly from the SCC's shared store
(`{data_root}/s{SSSS}/c{C}/k{K}/templates/oversampling_{N}/`) — there is
no `ws/templates` symlink.

**Two foreground paths** (no daemon):

| Path | Command | Config source |
|------|---------|---------------|
| Site policy | `syndiff diff run --site config --targets t.csv --target-name 2020ut` | `diff_config.yaml` + `deployment.yaml` |
| Alternate diff policy | `syndiff diff run --config config/other_diff.yaml --deployment config/deployment.yaml --targets t.csv --target-name 2020ut` | Explicit diff policy + deployment |
| Materialized YAML | `python -m syndiff_pipeline.difference_imaging.orchestration.cli --config config/example/diff_config_a_prf.yaml` | Pre-built per-target YAML under [`config/example/`](config/example/) |

See [`config/README.md`](config/README.md) for site layout. Outputs live under `{workspace_root}/events/{event_name}/s{SSSS}_c{C}_k{K}/ws/`; full directory reference: [`docs/markdown/storage_layout.md`](docs/markdown/storage_layout.md).

---

## Documentation

**Online docs:** https://zoutei.github.io/TSST_syndiff_pipeline/ (publish via Actions → docs → Run workflow)

| Document | Contents |
|----------|----------|
| [`docs/README.md`](docs/README.md) | Documentation index and HTML build instructions |
| [`docs/markdown/template_pipeline.md`](docs/markdown/template_pipeline.md) | `syndiff` orchestration, Condor, config, run lifecycle |
| [`docs/markdown/syndiff_cli.md`](docs/markdown/syndiff_cli.md) | CLI noun/verb commands and stage modules |
| [`docs/markdown/storage_layout.md`](docs/markdown/storage_layout.md) | `workspace_root`, `data_root` (SCC + nested-event layout), on-disk layout |
| [`docs/markdown/star_lightcurves.md`](docs/markdown/star_lightcurves.md) | Host-star quick start, prerequisites, and outputs |
| [`docs/markdown/stages/`](docs/markdown/stages/README.md) | PanCAKES, PS1 process, template-build algorithms |
| [`docs/markdown/cluster_smoke_checklist.md`](docs/markdown/cluster_smoke_checklist.md) | Cluster smoke test after setup |
| [`config/`](config/) | Site configs and example diff YAMLs |
