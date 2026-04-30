# Example recipes

YAML files in this directory are **reference `pipeline:` recipes**. Paths are relative to **each YAML file’s directory** (see `syndiff_pipeline/config.py` `load_config`).

| File | Summary |
|------|---------|
| [`recipe_simple.yaml`](recipe_simple.yaml) | Hotpants + PRF photometry; optional background / subtract chain (commented tail). |
| [`recipe_simple_photometry_only.yaml`](recipe_simple_photometry_only.yaml) | Re-run `forced_photometry` only (`pipeline_external_workspace_labels` + existing `ws/`). |
| [`recipe_a_prf.yaml`](recipe_a_prf.yaml) | PRF on Hotpants diffs → `background_rough` + `background_adaptive` → subtract → PRF again (lower peak RAM than a single combined stage). |
| [`recipe_a_prf_from_background.yaml`](recipe_a_prf_from_background.yaml) | Re-run `background_adaptive` when `rough_bkg_r1.npz` or `rough_bkg_r1.npy` already exists (see file header for `inputs.rough` workspace label). |
| [`recipe_b_epsf_sat_bkg.yaml`](recipe_b_epsf_sat_bkg.yaml) | Full ePSF + sat template + background + photometry chain; see header for `SynDiffConfig` field reference (`config.py`). |
| [`recipe_c_second_hotpants.yaml`](recipe_c_second_hotpants.yaml) | Two Hotpants passes and repeated middle chain; notes on `inputs.convolved`. |

## Smoke / local test

[`test_pipeline/`](test_pipeline/) — one-FFI layout, `test_config.yaml`, and optional **smoke** YAMLs (`smoke_prf_hotpants_lc.yaml`, `smoke_early_background.yaml`) for developer runs. Not user-facing recipes.

## Other

Post-pipeline **`lightcurve.csv` analysis**: [`development/lightcurve_24h_rolling.ipynb`](../../development/lightcurve_24h_rolling.ipynb) (repo root `development/`).

Pipeline run artifacts under `syndiff_pipeline/example/output/` are **gitignored**; create that tree locally when recipes point `output_dir` there.
