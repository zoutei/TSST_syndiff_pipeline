# Host-star light curves with syndiff star

Star measures TIC or Gaia targets on a completed SCC difference lane. It restores the retained PS1 light of each target using that lane's template recipe and saved Hotpants kernels.

## Run a lane

Set `template_recipe` to the operator descriptor for the producing template, or place it at `star_template_recipe.yaml` in the producing template directory. See the [operator descriptor](stages/star_pipeline.md#template-operator-descriptor).

```bash
syndiff star run \
  --site config \
  --star-config config/star_config.yaml \
  --star-targets config/star_targets_example.csv \
  --target-name 20/3/2
```

Use `syndiff star submit` with the same site, config and targets options to run all enabled SCC rows through the supervisor. The host list CSV has `gaia_source_id`, `tic_id`, and `label` columns. Supply one identifier per row. Preserve Gaia IDs as decimal integers.

The default stamp uses the lane's `hp_d` and its saved `hp_d_kernels/*_kernel.npz`. The template, registration maps, Gaia catalogue, shared mask, and group schedule must be available. Save `NOISE` when producing the Hotpants differences. Use `stamp_source: reconstruct` to use the raw science, `baseline.convolved` (typically `hp_c`), and `baseline.phot_bkg` (typically `ks_b`) for the base image. Use the convolved and background planes from the same subtraction.

## Options

Set these in `defaults:` or in an SCC entry under `overrides:` in `star_config.yaml`.

| Option | Meaning |
|---|---|
| `template_recipe` | Path to the producing operator descriptor. Relative paths resolve from the site directory. |
| `template_store_name` | Name of the template and bootstrap store, for example `linear`. |
| `output_store_name` | Name of the existing difference lane, for example `ownwcs_t1`. |
| `output_directory` | Optional SCC-specific output directory. Otherwise Star writes to `host_star/` beside the difference products. |
| `stamp_source` | `difference` by default. `reconstruct` uses the legacy science minus convolved template minus science background formula for the base. |
| `stamp_size` | Native output side length, 35 by default. Allow room for the entire aperture and sky annulus. |
| `cutout_size` | Native pre-kernel component side length, 96 by default. Star flags clipped component support. |
| `max_ffis` | Optional cap on cadences processed. |
| `cadence_positions` | Optional CSV or Parquet with integer `source_id`, timestamp-only `product_id`, and crop-local `x,y` for targets and potential neighbours. A row with `product_id: reference` gives the template reference position. Without the file, Star computes cadence positions from the FFI WCS and Gaia proper motions. |
| `physical_mask_dir` | Optional directory of native science-crop `<SCC-bearing-stem>.npy` detector masks. Zero means usable. Without it, Star accepts Hotpants output flags 0 and 64. |
| `perband` | Default `true`. Set `false` to run the historical weighted mini-template and CSV workflow for older datasets. The legacy workflow also reads `ps1_source` and `ps1_zarr_path`. |

A per-band run rebuilds its target archives and tables from the configured inputs. Reuse a table for alternate weights. `kernel_margin_px` is deprecated. The PS1 Gaussian radius is specified in the template operator descriptor, and the native convolution guard comes from the fitted kernel.

## Photometry configuration

The default methods are circular pixel-centre apertures at radii 4, 6 and 8 native pixels. Every aperture uses the same detector support for the difference and the four bands. The difference measurement uses a 3σ-clipped median sky in the outer annulus. The band measurements are aperture sums of isolated retained components.

```yaml
photometry:
  methods:
    - {name: r6, type: aperture, radius: 6, annulus: [12, 16]}
    - name: joint
      type: psf
      psf_type: epsf
      inputs: {epsf: epsf_r1}
      fit_radius: 6
```

A joint ePSF fit uses fixed catalogue centroids, lane noise, and a shared projection that removes constant sky and neighbouring profiles. Supply the existing ePSF catalogue referenced by `inputs.epsf`. An enabled `epsf` build block can populate it in the lane when using the standard lane output directory. The legacy `perband: false` path retains square `tar_ap` apertures and PRF methods.

## Outputs and rescoring

Each target directory contains `pre_kernel/group_<id>.npz`, `post_kernel/<product_id>.npz` and `perband.parquet`. The top-level `batch_manifest.csv` reports the completed targets and failed cadences. The [technical schema](stages/star_pipeline.md#output-schema) defines the arrays and table columns.

```python
import pandas as pd
from syndiff_pipeline.star.perband_photometry import score

table = pd.read_parquet("perband.parquet")
result = score(table, {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438})
```

`restored_flux` always uses the producing lane weights. `template_flux` uses the supplied trial weights. `flux_ratio` divides restored flux by trial template flux on valid rows with positive model flux. Formal errors propagate the saved pixel noise. Total-flux calibration, kernel uncertainty and residual crowding require separate assessment.

The [manifest-driven band commands](stages/star_band_extraction.md) remain available for externally produced component images.
