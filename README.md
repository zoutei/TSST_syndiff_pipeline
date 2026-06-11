# SynDiff Pipeline

TESS Full Frame Image (FFI) difference imaging pipeline for transient detection and forced photometry.

> **This project has not been released.** All modules are under active development.

---

## Overview

SynDiff performs difference imaging on TESS FFIs against PS1 (Pan-STARRS1) templates to produce clean difference images and extract forced PSF photometry light curves at a target coordinate. The workflow covers:

1. **FFI download** — bulk download of calibrated TESS FFIs from MAST.
2. **WCS drift analysis** — per-frame WCS, pixel drift of the science target, template offset groups.
3. **Shared masking** — bitmask from Gaia (bright stars, saturation crosses, TESS straps).
4. **Image differencing** — kernel-matching subtraction via **[pyhotpants](#forked-dependencies)** (FFI crops vs PS1 templates).
5. **Empirical PSF fitting** — tiled ePSF via TGLC.
6. **Saturated star templates** — model and subtract bright saturated sources when needed.
7. **Background estimation** — rough per-frame background, then adaptive temporal smoothing (TESSVectors).
8. **Forced photometry** — PSF-fitted flux at the target (ePSF or official TESS PRF).

Template building (TESS↔PS1 mapping, PS1 convolution, multi-offset downsampling) uses a customized **[MOCPy](#forked-dependencies)** fork for the PanCAKES mapping stage.

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
- Install: follow that repository’s build instructions (Rust + `maturin develop --release`), or see [`docs/stages/mapping_pancakes.md`](docs/stages/mapping_pancakes.md) and [`docs/stages/standalone_pipeline_overview.md`](docs/stages/standalone_pipeline_overview.md#custom-mocpy-installation).

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

The **`syndiff`** CLI orchestrates the full workflow behind one supervisor daemon and one SQLite state database. A single **seven-stage DAG** covers template building (TESS FFIs + PS1 → `syndiff_template_*.fits`) and difference imaging (Hotpants → photometry):

```text
syndiff all submit      # template stages → diff
syndiff template submit # template stages only
syndiff diff submit     # diff only (verifies tess_dl + wcs handoff + downsample on disk)
syndiff progress        # monitoring works the same for any run
```

Foreground debugging: `syndiff diff run --site config --targets t.csv --target-name 2020ut` (optional `--validate-only`).

| | Foreground (`syndiff diff run`) | Supervised (`syndiff * submit`) |
|---|--------------------------------|----------------------------------|
| **Purpose** | One target, current process | Multi-target batch + daemon |
| **Config** | `--site config` (site policy) | `--site` → `pipeline.yaml` + `diff_config.yaml` + `deployment.yaml` |
| **State** | No SQLite | `{workspace_root}/control/pipeline_state.sqlite` |
| **Outputs** | `events/{label}/ws/` | Same layout under `workspace_root` |

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

Targets are always passed on the CLI (`--targets`), never embedded in config files.

### Quick start

```bash
mamba activate syndiff

syndiff verify --site config --targets config/targets_example.csv

syndiff all submit \
  --site config \
  --targets config/targets_example.csv \
  --run-id batch_no5

syndiff progress
syndiff status --watch
syndiff retry --run-id batch_no5 --scc s0023_c1_k3_2020ftl --stage diff
```

### Command overview

| Pattern | Examples |
|---------|----------|
| **Execute** | `syndiff all submit`, `syndiff template submit`, `syndiff diff submit`, `syndiff diff run --target-name …` |
| **Monitor** | `syndiff progress`, `syndiff status --watch`, `syndiff logs`, `syndiff tail` |
| **Control** | `syndiff retry`, `syndiff pause`, `syndiff resume`, `syndiff kill` |
| **Workspace** | `syndiff runs`, `syndiff active`, `syndiff daemon status`, `syndiff verify` |

---

## Difference imaging (config-driven)

After PS1 templates exist, the orchestrator **`diff`** stage runs the YAML-ordered pipeline in [`config/diff_config.yaml`](config/diff_config.yaml) (`shared_mask`, `hotpants`, `epsf`, `background_*`, `forced_photometry`, …). WCS handoff and templates come from template creation (`cluster_template_job.json`, `syndiff_ffi_frames.csv`, `events/{label}/ws/templates`).

**Two foreground paths** (no daemon):

| Path | Command | Config source |
|------|---------|---------------|
| Site policy | `syndiff diff run --site config --targets t.csv --target-name 2020ut` | `pipeline.yaml` + `diff_config.yaml` + `deployment.yaml` |
| Materialized YAML | `python -m syndiff_pipeline.difference_imaging.orchestration.cli --config config/example/diff_config_a_prf.yaml` | Pre-built per-target YAML under [`config/example/`](config/example/) |

See [`config/README.md`](config/README.md) for site layout. Outputs live under `{workspace_root}/events/{label}/ws/`; full directory reference: [`docs/storage_layout.md`](docs/storage_layout.md).

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/README.md`](docs/README.md) | Documentation index |
| [`docs/template_pipeline.md`](docs/template_pipeline.md) | `syndiff` orchestration, Condor, config, run lifecycle |
| [`docs/syndiff_cli.md`](docs/syndiff_cli.md) | CLI noun/verb commands and stage modules |
| [`docs/storage_layout.md`](docs/storage_layout.md) | `workspace_root`, `data_root`, on-disk layout |
| [`docs/stages/`](docs/stages/README.md) | PanCAKES, PS1 process, downsample algorithms |
| [`docs/cluster_smoke_checklist.md`](docs/cluster_smoke_checklist.md) | Cluster smoke test after setup |
| [`config/`](config/) | Site configs and example diff YAMLs |

---

## Known limitations

- Second-pass Hotpants accepts `inputs.convolved` in YAML but does not consume it yet (warning logged).
- Hotpants `science` must be `"ffi"`; workspace science input is not implemented.
- No per-stage checkpointing — trim the `pipeline:` list and declare pre-existing workspaces with a preamble entry (`- external_workspaces: [hp_d]`) before the stages that consume them; legacy `pipeline_external_workspace_labels` is still supported.
- Mission WCS must be adequate for drift grouping; external astrometry may be required beforehand.
- PS1 templates must exist before diff imaging (`syndiff template submit` or `syndiff all submit`).
