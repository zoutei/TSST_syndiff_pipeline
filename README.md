# SynDiff Pipeline

[![Docs](https://github.com/zoutei/TSST_syndiff_pipeline/actions/workflows/docs.yml/badge.svg)](https://zoutei.github.io/TSST_syndiff_pipeline/)

TESS Full Frame Image (FFI) template building and difference-imaging pipeline for transient detection and forced photometry.

> **This project has not been released.** All modules are under active development.

---

## Overview

SynDiff is an end-to-end pipeline for TESS Full Frame Image (FFI) transient work: it builds PS1-based templates on the TESS pixel grid, then runs difference imaging and forced photometry at a science target. Template building is **SCC-scoped** (sector/camera/CCD; shared across every event that lands on it), while difference imaging is **event-scoped**. The `syndiff template` and `syndiff diff` CLI nouns submit these as two separate DAGs — there is no combined "run everything" preset.

An independent **`star`** branch can then use a completed template+diff
workspace to produce TIC/Gaia host-star light curves without re-running
Hotpants.

### Template creation (TESS FFIs + PS1 → SCC-shared template store)

`syndiff template submit --scc sccs.csv` (SCC-only input; no event coordinates):

1. **FFI download** (`tess_ffi_download`) — bulk download of calibrated TESS FFIs from MAST into `{data_root}/scc/s{SSSS}_c{C}_k{K}/ffi/`.
2. **Mapping (PanCAKES)** (`mapping`) — choose the SCC's mapping-epoch reference FFI (median-CRVAL anchor + Earth/Moon-angle cuts), map TESS pixels to PS1 skycells, and download the Gaia catalog for that reference FFI. Uses a customized **[MOCPy](#forked-dependencies)** fork.
3. **PS1 download** (`ps1_download`) — fetch PS1 skycell cutouts into a shared Zarr store.
4. **PS1 process** (`ps1_process`) — convolve PS1 data onto the TESS grid (CPU-heavy; optionally on HTCondor).
5. **Templates** (`templates`; legacy config key/alias: `downsample`) — combine convolved skycells at multiple sub-pixel offsets into the SCC's shared template store under `{data_root}/scc/s{SSSS}_c{C}_k{K}/templates/oversampling_{N}/`.

### Event binding (`syndiff diff submit --targets targets.csv --stages bind,diff`)

The first diff-DAG stage, **`bind`**, measures the transient's per-frame WCS and pixel drift and writes the event handoff (`event_job.json` + `frames.csv`) under `{workspace_root}/events/{event_name}/s{SSSS}_c{C}_k{K}/`. **Important**: the bare `syndiff diff submit --targets ...` (no `--stages`) selects only `diff`, not `bind` — pass `--stages bind,diff` explicitly the first time you diff a new event. See [`docs/markdown/template_pipeline.md`](docs/markdown/template_pipeline.md#overview).

### Difference imaging (templates + FFIs → light curves)

After templates exist, the **`diff`** stage runs a YAML-ordered pipeline ([`config/diff_config.yaml`](config/diff_config.yaml)). You choose which steps to include; the default site config is a short path (mask → Hotpants → forced photometry with the official TESS PRF). A fuller recipe might look like:

1. **Shared masking** — bitmask from Gaia (bright stars, saturation crosses, TESS straps).
2. **Image differencing** — kernel-matching subtraction via **[pyhotpants](#forked-dependencies)** (FFI crops vs PS1 templates).
3. **Forced photometry** — PSF-fitted flux at the target (official TESS PRF by default, or ePSF when that stage is enabled).

Optional steps you can add to the `pipeline:` list:

- **Empirical PSF fitting** — tiled gridded ePSF via photutils (`epsf` stage; `GriddedPSFModel` with configurable `tile_nx`×`tile_ny`; use with `psf_type: epsf` in forced photometry for spatially varying **gepsf**).
- **Gaia centroids** — PSF photometry on bright Gaia stars (`centroids` stage; outputs per-frame `*_photresults.ecsv`).
- **Saturated star templates** — model and subtract bright saturated sources (`sat_template` + `subtract`).
- **Background removal** — the unified `background` stage composes spatial, temporal, and strap corrections before optional `subtract`.
- **Second round of differencing** — run Hotpants again on background-subtracted science images for cleaner residuals (see commented blocks in [`config/diff_config.yaml`](config/diff_config.yaml) and [`config/example/diff_config_c_second_hotpants.yaml`](config/example/diff_config_c_second_hotpants.yaml)).

Run template building with `syndiff template submit --scc sccs.csv`, then diff imaging (once templates exist) with `syndiff diff submit --targets targets.csv --stages bind,diff`.

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
syndiff diff submit --targets t.csv --stages bind,diff   # bind (event WCS) + diff
syndiff progress                                          # monitoring works the same for any run
```

Foreground debugging: `syndiff diff run --site config --targets t.csv --target-name 2020ut` (optional `--validate-only`) — this path never runs `bind`; the event's handoff must already exist.

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

# once templates exist, bind + diff the event
syndiff diff submit \
  --site config \
  --targets config/targets_example.csv \
  --stages bind,diff \
  --run-id diff_batch_no5

syndiff retry --deployment config/deployment.yaml --run-id diff_batch_no5 \
  --scc s0023_c1_k3_2020ftl --stage diff
```

### Command overview

| Pattern | Examples |
|---------|----------|
| **Execute** | `syndiff template submit`, `syndiff diff submit --stages bind,diff`, `syndiff diff run --target-name …`, `syndiff star submit` |
| **Monitor** | `syndiff progress`, `syndiff status --watch`, `syndiff logs`, `syndiff tail` |
| **Control** | `syndiff retry`, `syndiff pause`, `syndiff resume`, `syndiff kill` |
| **Workspace** | `syndiff runs`, `syndiff active`, `syndiff daemon status`, `syndiff verify` |

---

## Difference imaging (config-driven)

After PS1 templates exist, the orchestrator **`diff`** stage runs a YAML-ordered
diff policy (`shared_mask`, `hotpants`, `epsf`, `background`,
`forced_photometry`, …). Foreground `--site config` uses
[`config/diff_config.yaml`](config/diff_config.yaml); supervised submit uses
the `diff_config:` path selected by `pipeline.yaml`. The event handoff
(`event_job.json`, `frames.csv`) comes from the **`bind`** stage; templates
are resolved directly from the SCC's shared store
(`{data_root}/scc/s{SSSS}_c{C}_k{K}/templates/oversampling_{N}/`) — there is
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
| [`docs/markdown/scc_migration.md`](docs/markdown/scc_migration.md) | One-time migration script to the SCC + nested-event layout |
| [`docs/markdown/star_lightcurves.md`](docs/markdown/star_lightcurves.md) | Host-star quick start, prerequisites, and outputs |
| [`docs/markdown/stages/`](docs/markdown/stages/README.md) | PanCAKES, PS1 process, template-build algorithms |
| [`docs/markdown/cluster_smoke_checklist.md`](docs/markdown/cluster_smoke_checklist.md) | Cluster smoke test after setup |
| [`config/`](config/) | Site configs and example diff YAMLs |
