# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SynDiff: a TESS Full Frame Image (FFI) difference-imaging pipeline. It builds PS1-based templates on the TESS pixel grid, runs Hotpants subtraction + forced photometry at a science target, and can additionally produce TIC/Gaia host-star light curves from a completed workspace. Everything is driven by the `syndiff` CLI (entry point: `syndiff_pipeline/cli.py`).

## Environment and commands

```bash
mamba activate syndiff        # ALWAYS first — required for any Python in this repo
pip install -e .              # registers the `syndiff` CLI; pyhotpants must be installed from github. it is not on pypi
```

```bash
pytest tests/                          # full suite (~109 files, no conftest magic)
pytest tests/test_diff_stage.py        # one file
pytest tests/test_diff_stage.py -k name  # one test
```

Docs build: `pip install -e ".[docs]"` then `cd docs/sphinx && make html` (third-party imports are mocked, so the runtime stack isn't needed). Publishing to GitHub Pages is manual via Actions → docs.

### Running the pipeline

```bash
syndiff template submit --site config/ --scc config/scc_example.csv   # template DAG through downsample
syndiff diff submit --site config/ --scc …                            # SCC subtract (verifies upstream on disk)
syndiff photometry submit --site config/ --targets …                  # event forced photometry
syndiff diff run --site config/ --targets t.csv --target-name {label} # foreground diff debug
syndiff star submit --site config/ --star-targets {star_targets.csv}  # host-star branch
syndiff verify --site config/ --targets ...                           # pre-run artifact check
```

Monitoring/control: `syndiff progress`, `syndiff status --watch`, `syndiff cluster` (execute-host sampler table; `--check` for placement preview), `syndiff tail`, `syndiff retry`, `syndiff launch`, `syndiff pause|resume|kill`, `syndiff runs`, `syndiff active`, `syndiff logs`, `syndiff bookkeeping`. Control verbs take `--run-id` + `--deployment` (or `--run-dir`), **not** `--site`; they write intents to SQLite which the supervisor applies on its next tick.

Cheap testing of changes: foreground `syndiff diff run` (add `--validate-only` for config checks), `--local` on submit to bypass Condor, `max_ffis` in `star_config.yaml` for short star runs, `crop_mode: target_box` for small/fast events. Stage modules also have standalone entry points (e.g. `python -m syndiff_pipeline.template_creation.processing.pancakes <cluster_template_job.json>`).

### Site config (`config/`)

| File | Git | Contains |
|------|-----|----------|
| `pipeline.yaml` | committed | Template policy: stages, pools, notifications |
| `diff_config.yaml` | committed | Diff sub-pipeline: `defaults` + `pipeline:` stage knobs (omit dataclass defaults). Mask policy is **not** here. Default = `shared_mask` + `hotpants`. |
| `mask_settings.yaml` | optional | Mask geometry/policy (sibling of `diff_config`; bare `- kind: shared_mask` uses this or packaged defaults) |
| `photometry_config.yaml` | optional site | Event photometry policy (`syndiff photometry`); examples `photometry_config_*.yaml` |
| `star_config.yaml` | committed | Star-branch policy |
| `deployment.yaml` | **gitignored** | `workspace_root`, `data_root`, Gaia + Discord credentials (copy from `deployment.yaml.example`) |

Targets are always passed on the CLI (`--targets` / `--star-targets`), never embedded in config. Each run freezes its effective config under `runs/{run_id}/`; each diff workspace freezes a slim `ws/diff_config.yaml` plus `ws/mask_settings.yaml` after `shared_mask` — check the frozen copies, not the YAML defaults, when debugging.

## Architecture

### Stage DAG

```
tess_ffi_download → mapping → ps1_download → ps1_process → remap → downsample → diff
   (network)         (Condor)   (network)      (Condor)    (Condor)  (cpu/Condor) (Condor)

completed diff lane ──verify──→ photometry   (event targets)
completed template + diff ──verify──→ star   (star_targets.csv)
```

`pipeline_spec.py` composes `TEMPLATE_STAGES + DIFF_STAGES + PHOTOMETRY_STAGES + STAR_STAGES` (**nine** stages). `wcs_grouping` is config-only (linear drift), not a scheduler stage. With `ps1_process.ps1_source: stream`, `ps1_download` is skipped. With `geometry_mode: linear`, `remap` is skipped.

### Package layout

- `syndiff_pipeline/cli.py` — noun/verb CLI (`syndiff template|diff|photometry|star`, monitor/control verbs; `all` removed).
- `syndiff_pipeline/common/orchestration/` — supervisor, scheduler, SQLite state, Condor, verify.
- `syndiff_pipeline/template_creation/processing/` — pancakes, ps1_download/process, field_remap, field_downsample / linear_downsample.
- `syndiff_pipeline/difference_imaging/` — `execute.py` + `stages/` + `masking/`. Default site diff = shared_mask + hotpants.
- `syndiff_pipeline/photometry/` — event astrometry + forced photometry on SCC diff lanes.
- `syndiff_pipeline/star/` — host-star light curves.

Heavy stages on HTCondor by default: `mapping`, `ps1_process`, `remap`, `downsample`, `diff`, `photometry`, `star`.

### Storage: two roots

`data_root` holds SCC trees under `s{SSSS}/c{C}/k{K}/` (ffi, mapping, remap, templates, convolved.zarr, `diff_{lane}/`, bookkeeping). `workspace_root` holds `events/{event}/s{SSSS}_c{C}_k{K}/` (photometry under `phot_{run_id}/`, star under `host_star/`). Full reference: `docs/markdown/storage_layout.md`.

In this checkout, `data` and `workspace` are symlinks to `/astro` storage; `pyhotpants`, `TESSreduce`, `syndiff_viewer` are symlinks to sibling checkouts. `scripts/`, `dev/`, `experimental/`, `config/example/` are locally gitignored scratch/backfill areas — don't assume they're in git history.
## Invariants that bite

1. **Offset quantization**: 1 PS1 px = 0.258″ ≈ 0.0124 TESS px; `offset_threshold` = 0.01 TESS px. Template offsets are realized as integer PS1-pixel rolls per skycell; sub-pixel WCS drift never requires re-running `mapping`.
2. **Drift is measured at the target position only** — templates degrade away from the target. (Known limitation; a cached exact per-epoch regmap fix was validated in `experimental/grid_wcs_correction/`, not rotation/affine modeling.)
3. **Coordinates**: crop bounds are `[min, max)` in full-FFI 0-based pixels; diff-stage x/y are **crop-local**. `ensure_gaia_crop_xy` converts. Mixing these silently misplaces stars.
4. **Template filenames are the API**: `syndiff_template_s{S}_{C}_{K}[_x..._y...][_osN]_dx{±D.DDD}_dy{±D.DDD}.fits.gz` — the diff stage matches manifest `group_dx/dy` to filename dx/dy. Swapping templates = pointing `template_dir` at a dir with identically-named files.
5. **Kernels**: Hotpants kernels are not persisted unless `write_kernel_solutions: true` (required by `star`). The separate `kernel_fit` path persists one reusable target-level `kernel_r2.npz`.
6. **Verify gates are thin** (mapping = one CSV exists; templates = files parse). After changing an upstream stage, delete its outputs or `--force-rerun` — stale artifacts pass verify and get skipped.
7. **FFI files may be `.fits` or `.fits.gz`** — always resolve both (helpers in `common/download.py`).
8. **`sat_template` is broken for its stated purpose** — don't build on it without reading `docs/markdown/stages/diff_pipeline.md` §6.
9. Band combination weights: r=0.238, i=0.344, z=0.283, y=0.135 (four bands, not three).
10. **Star vs template Zarr paths differ**: star defaults to `{data_root}/ps1_skycells.zarr`; `ps1_download` writes `{data_root}/ps1_skycells_zarr/ps1_skycells.zarr`. Set `ps1_zarr_path` in `star_config.yaml` to share.
11. **Star gepsf**: every `psf_type: epsf` method requires `inputs.epsf: {label}`; an optional `epsf` block builds `{baseline_ws}/{epsf.output}` and must use the same label. Star also needs baseline side products (convolved templates, backgrounds, shared mask, `{diffs}_kernels/*.npz`) — backfill older workspaces with `config/diff_config_star_full_backfill.yaml`.
12. `ps1_download` writes to a shared, file-locked Zarr — concurrent runs on one `data_root` serialize there; a stuck lock stalls the stage.

## Debugging a failed/stalled run

1. Per-stage log: `{workspace_root}/runs/{run_id}/per_target/{target_label}/{stage}.log` (Condor stderr alongside).
2. Daemon log: `syndiff logs` (scheduling decisions, verify results, Condor submit errors).
3. Frozen config: `runs/{run_id}/run_meta.json` + per-target `diff_config.yaml`.
4. State DB: `{workspace_root}/control/pipeline_state.sqlite`; status semantics in `docs/markdown/pipeline_state_machine_reference.md`.
5. Expected durations (native res, one SCC): mapping ~13 min, ps1_process tens of minutes–hours, downsample minutes. A mapping stage running for hours is stuck, not slow.

## Forked/external dependencies

- **pyhotpants** — PyPI package is `hotpants` (installed by `pip install -e .`); dev checkout fallback: a `pyhotpants/` dir found by the import fallback in `difference_imaging/stages/hotpants.py`.
- **Custom MOCPy** — the `mapping` stage needs `MOC.filter_points_in_polygons` from [zoutei/mocpy_syndiff](https://github.com/zoutei/mocpy_syndiff/) (Rust + maturin build); stock `mocpy` will not work.
- **TGLC** — must be importable (clone + `PYTHONPATH`) for ePSF stages; **PRF** optional for `psf_type: prf`.
- **Asteroid mask generate (optional)** — consuming SCC `pixel_intervals.parquet` needs nothing extra. Generating requires `sbident` (`pip install git+https://github.com/bengebre/sbident`) + `tess-ephem` (`pip install --no-deps tess-ephem` if astropy is already present). MIT `TESS_orbit_times.csv` auto-downloads to `{data_root}/catalogs/`. See `docs/markdown/masking.md`.

## Documentation map

Deep-dive docs live in `docs/markdown/` — read the relevant stage doc before editing a stage: `template_pipeline.md` (orchestration/Condor/run lifecycle), `syndiff_cli.md` (all verbs/flags + worker entry points), `storage_layout.md`, `field_geometry.md`, `photometry.md`, `star_lightcurves.md`, `stages/` (per-stage algorithms), `pipeline_state_machine_reference.md`. Docs may lag the code; the frozen run configs and `pipeline_spec.py` are ground truth. Historical design notes and improvement plans are in `.cursor/plans/`; active design documents live in `doc/` (e.g. `padded_scc_v2_implementation.md`, `template_bookkeeping_plan.md`). Claude skills for this repo are in `.claude/skills/` (`syndiff-pipeline-map`, `syndiff-run-ops`).
