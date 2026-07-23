> **Package integration**: `syndiff` stage `photometry` · package `syndiff_pipeline/photometry/` · configured by `photometry_config*.yaml`  
> **Related docs**: [photometry quick start](../photometry.md) · [config schema](photometry_config.md) · [forced photometry](forced_photometry.md) · [diff pipeline](diff_pipeline.md) · [storage layout](../storage_layout.md)

# Photometry pipeline (technical)

The orchestrator stage **`photometry`** (`photometry/orchestration/stages.py`) is independent of the template→diff DAG. It verifies that an SCC diff lane is complete, then runs an ordered YAML pipeline of **photometry kinds** via `photometry/runner.py: run_photometry_pipeline()`.

Photometry kinds (not the same set as diff `STAGE_KINDS`):

| Kind | Module | Role |
|------|--------|------|
| `astrometry` | `difference_imaging/stages/astrometry.py` | Refine transient RA/Dec; write `astrometry_result.json` under the photometry tree |
| `forced_photometry` | `difference_imaging/stages/photometry.py` | Aperture / PRF / ePSF / gepsf light curves on SCC diffs |

## Relationship to `syndiff diff`

| Path | How photometry runs |
|------|---------------------|
| **`syndiff photometry submit\|run`** | Preferred. Freezes `photometry_config.yaml`, registers stage `photometry` in SQLite. |
| Diff `pipeline: [{kind: photometry, config: …}]` | Delegator: `execute.py` → `run_photometry_delegator()` loads that YAML and runs the same runner in-process during a diff stage. |

Default site [`diff_config.yaml`](../../../config/diff_config.yaml) does **not** include the delegator. Astrometry and forced photometry are **not** diff `STAGE_KINDS` (except the `photometry` delegator kind).

## Execution flow

1. Load `PhotometrySitePolicy` + merge `PhotometryRunConfig` for the event (`site_config.py`).
2. Build a `SynDiffConfig` pointed at SCC templates / `data_root` / named inputs (`diffs`, optional `epsf`).
3. Require `data_root`; load SCC handoff via `scc_bootstrap.load_scc_diff_handoff_for_config()` (`bookkeeping/diff/`).
4. For each pipeline entry:
   - **`astrometry`** — may overwrite `cfg.target_ra` / `cfg.target_dec` from survey mix.
   - **`forced_photometry`** — resolve crop-local positions, load gridded ePSF catalogs when `psf_type: epsf`, write LCs under `phot_{run_id}/{output}/`.
5. Write DS9 `targets.reg` as needed; optional debug plots under `phot_{run_id}/debug_plots/`.

## Verify {#verify}

Upstream (before launch): `scc_diff_lane_complete()` requires:

- `{data_root}/…/bookkeeping/diff/diff_job.json` and `frames.csv`
- At least one pipeline FITS under `diff_{lane}/{diffs_label}/`
- If `paths.inputs.epsf` is set: `{epsf_label}/gridded_epsf_index.json`

Completion: `photometry_complete()` checks expected light-curve CSVs (and astrometry JSON when configured) under `events/{event}/s…/phot_{photometry_run_id}/`.

## Outputs

```text
{workspace_root}/events/{event}/s{SSSS}_c{C}_k{K}/phot_{photometry_run_id}/
  astrometry_result.json          # if kind: astrometry ran
  targets.reg
  {lc_label}/lightcurve_*.csv
  debug_plots/                    # optional
```

SCC products remain under `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/` — photometry is read-only on those planes.

## Condor / CLI

- Stage default executor: Condor (`stages.photometry` in `pipeline.yaml`).
- CLI: `syndiff photometry submit|run` (`photometry/cli.py`); `--local` patches frozen run config to `executor: local`.
- Status grid (`syndiff status`) omits `photometry` and `star`; use `syndiff progress` and `per_target/{label}/photometry.log`.

## Code map

| Path | Role |
|------|------|
| `photometry/cli.py` | `submit` / `run` |
| `photometry/site_config.py` | Policy + run merge + `SynDiffConfig` builder |
| `photometry/runner.py` | Pipeline loop + forced-photometry driver + diff delegator |
| `photometry/orchestration/stages.py` | `PHOTOMETRY_STAGES` registry |
| `photometry/orchestration/verify.py` | Lane + completion checks |
