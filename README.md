# SynDiff Pipeline

TESS Full Frame Image (FFI) template building and difference-imaging pipeline for transient detection and forced photometry.

> **This project has not been released.** All modules are under active development.

---

## Overview

SynDiff is an end-to-end pipeline for TESS Full Frame Image (FFI) transient work: it builds PS1-based templates on the TESS pixel grid, then runs difference imaging and forced photometry at a science target. The unified **`syndiff`** CLI runs template creation first, then difference imaging — either end-to-end or one phase at a time.

### Template creation (TESS FFIs + PS1 → `syndiff_template_*.fits`)

1. **FFI download** — bulk download of calibrated TESS FFIs from MAST.
2. **WCS grouping** — per-frame WCS, pixel drift of the science target, template offset groups; writes handoff JSON for diff.
3. **Mapping (PanCAKES)** — map TESS pixels to PS1 skycells; Gaia catalog for the reference FFI. Uses a customized **[MOCPy](#forked-dependencies)** fork.
4. **PS1 download** — fetch PS1 skycell cutouts into a shared Zarr store.
5. **PS1 process** — convolve PS1 data onto the TESS grid (CPU-heavy; optionally on HTCondor).
6. **Downsample** — combine convolved skycells at multiple sub-pixel offsets → `syndiff_template_*.fits.gz`.

### Difference imaging (templates + FFIs → light curves)

After templates exist, the **`diff`** stage runs a YAML-ordered pipeline ([`config/diff_config.yaml`](config/diff_config.yaml)). You choose which steps to include; the default site config is a short path (mask → Hotpants → forced photometry with the official TESS PRF). A fuller recipe might look like:

1. **Shared masking** — bitmask from Gaia (bright stars, saturation crosses, TESS straps).
2. **Image differencing** — kernel-matching subtraction via **[pyhotpants](#forked-dependencies)** (FFI crops vs PS1 templates).
3. **Forced photometry** — PSF-fitted flux at the target (official TESS PRF by default, or ePSF when that stage is enabled).

Optional steps you can add to the `pipeline:` list:

- **Empirical PSF fitting** — tiled ePSF via TGLC (`epsf` stage; use with `psf_type: epsf`).
- **Saturated star templates** — model and subtract bright saturated sources (`sat_template` + `subtract`).
- **Background removal** — rough per-frame background, then adaptive temporal smoothing (`background_estimate` / `background_rough`, then `subtract` from the diffs).
- **Second round of differencing** — run Hotpants again on background-subtracted science images for cleaner residuals (see commented blocks in [`config/diff_config.yaml`](config/diff_config.yaml) and [`config/example/diff_config_c_second_hotpants.yaml`](config/example/diff_config_c_second_hotpants.yaml)).

Run the full workflow with `syndiff all submit`, template building only with `syndiff template submit`, or diff only (when upstream artifacts already exist) with `syndiff diff submit`.

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

The **`syndiff`** CLI orchestrates the full workflow behind one supervisor daemon and one SQLite state database — template building (TESS FFIs + PS1 → `syndiff_template_*.fits`) and difference imaging (config-driven Hotpants → photometry):

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
