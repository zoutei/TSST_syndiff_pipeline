# Star pipeline

The [host-star workflow](../star_lightcurves.md) runs on a completed SCC template and subtraction. It replays the target component through the producing template geometry and the fitted kernel of each cadence.

## Template operator descriptor

Per-band mode currently reads native OS1 linear lanes. Supply a YAML descriptor with the producing PS1 cells, registration geometry and calibration. The default location is `star_template_recipe.yaml` in the lane's frozen `template_dir`. Paths in the descriptor may be absolute or relative to the descriptor. References inside the frozen template config retain their original config-relative interpretation.

```yaml
frozen_config: /data/workspace/runs/t1/config.yaml
kernel_radius: 470
geometry: canonical_v3
removal_recipe_id: 073d63be286b4da0
cell_index: retained_cells.json
ps1_crpix: 0.5
registration_grid:
  width: 2064
  origin_full_ffi: [36, -8]
# registration_dir: /data/s0026/c2/k2/mapping/oversampling_1
# reference_ffi: /data/s0026/c2/k2/ffi/tess-reference.fits.fz
```

The frozen config supplies `stages.ps1_process.psf_sigma` and `stages.downsample.band_weights`, `ignore_mask_bits` and `zpcell_factors_path`. The descriptor can also declare these as top-level `psf_sigma`, `band_weights`, `ignore_mask_bits` and `zpcell_factors_path` for a historical recipe. Use `null` for zpcell disabled. The Gaussian radius and width are in PS1 pixels. Assembly weights, where present, must agree.

`cell_index` is JSON mapping each contributing skycell name to the directory containing `arrays.npz`, `headers.json` and `_provenance.json` for its producing generation. The archive must hold four unweighted, flux-converted, post-removal bands with a common retained support. Archive provenance must match the declared removal recipe. Each source cell's zpcell factors divide its bands before canonical assembly and cross-projection warping.

A June-style descriptor uses `geometry: june_row`. Supply `raw_store` with the original PS1 band Zarr and let `cell_index` point to the archived combined-image support. Supply `padding_patches` as JSON keyed by recipient, with an ordered list of entries containing `source`, `bounds` (half-open recipient PS1 `x0,y0,x1,y1`), and `wcs` (the historical FITS keyword mapping of that patch). Use the June registration width and full-FFI origin and, for the later reflected FITS, set `reflect_below` to the original full-FFI lower y bound. The historical support and coefficients need independent lane closure.

The released S4 June recipe also specifies `left_writer_cells`, a JSON list of right-hand cells whose left neighbour owns the 10-pixel overlap strip. `raw_fallback` accepts a JSON list path at `cells` and its `psf_sigma`. S4 used 60 PS1 pixels for ten cells. These receivers use their own cell boundaries and attribute the conditional target add-back only to the retained footprint. Measurable fallback light sets `identity: raw_fallback_provisional` and excludes that target from validated scores. `removal_recipe_ids` accepts the generation IDs of the archives selected in the index. Use `mapping_table` for the historical mapping CSV or Parquet and `registration_pattern` for a filename such as `tess_s{sector}_{camera}_{ccd}_{cell}.fits.gz` under `registration_dir`.

## Coordinates and replay

1. The retained PS1 island is selected near the host at the stack epoch, aided by the removal ledger when available. Multiple catalogue centres in the same retained component are flagged. Star transports all recipient cells covered by the target window with canonical column and row ownership, ordered cross-projection patches, the frozen bit veto, and integer PS1 rolls.
2. Master and registration array origins are expressed in 0-based full-FFI coordinates. Registration assignments index the producing grid. Science crops, output origins and fractional catalogue positions use 0-based crop-local native coordinates. All bounds are half-open.
3. Star bins the four bands once per target and offset group, reusing each result for all frames in its group. It reads the lane `frames.csv`, with bookkeeping fallback, and accepts both SCC-bearing FITS names and timestamp-only legacy names. Saved kernel identifiers remain timestamp-only.
4. The kernel window adds one fitted native kernel radius per side. Hotpants spatial polynomials use the shape and origin of the full padded fit grid, with block centres phased from its edge. An isolated stamp uses the same block centres. The local Gaussian footprint and Hotpants native kernel guard have different units.

The restored image is

$$
I_\mathrm{star}=D+\sum_{b\in\{r,i,z,y\}}w_{b,\mathrm{lane}}S_b.
$$

Here `D` is the saved Hotpants difference in electrons/s per native pixel, `S_b` is the unweighted target-only band after that frame's kernel in the same units, and `w` is the dimensionless frozen template coefficient. For `stamp_source: reconstruct`, Star uses $D=I-C-B$, where $I$ is the raw science crop, $C$ is the saved convolved template and $B$ is the photometric background, all in the same units. A parity check uses identical cadences, masks, background and convolution direction.

## Output schema

| Artifact | Fields |
|---|---|
| `pre_kernel/group_<id>.npz` | `flux[band,y,x]` in producing template flux-sum units, `bands` in r/i/z/y order, `origin_crop_xy`, `group_id`, `shift` in native pixels, registration origin, kernel fit geometry, blur and removal identifiers. |
| `post_kernel/<product_id>.npz` | `images[plane,y,x]` in electrons/s per native pixel, with `bands` ordered D/r/i/z/y. The raw difference plane still contains sky. Also stores `noise` (1σ), `good` (the common measurement support), `solver_mask`, `origin_crop_xy`, `center_crop_xy`, `group_id` and SCC-bearing `stem`. |
| `perband.parquet` | One row per source, cadence and estimator. Identifiers are `source_id`, `stem`, `group_id`, `btjd` and `method`. Coordinates are `x,y,x0,y0`. `D` and `S_r,S_i,S_z,S_y` are amplitudes in electrons/s. `lane_w_r` through `lane_w_y` store the producing weights. `error` is formal difference-flux uncertainty. `sky` and `n_sky` record aperture sky in electrons/s/pixel and its retained pixel count. `identity` and `valid` record attribution and usable support. |
| `batch_manifest.csv` | Paths to each target's table, usable frame count, failed frame count and errors. |

Use `score(table, weights)` for plotting and trial mixtures. Call `restore_stamp(images, lane_weights)` to evaluate the stamp equation on saved planes. The score's restored flux remains fixed when the trial weights change.

Aperture support is fractional-centred but uses pixel-centre inclusion. The estimator subtracts the clipped sky only from `D`. The ePSF path selects its spatial shape at the science-crop source coordinates, normalizes every cropped oversampled node so its integral is one, and applies one noise-weighted target projection to all five planes. The nuisance subspace contains constant sky and fixed nearby catalogue profiles. No fit is reported when the target is inseparable from this subspace.

The weighted legacy workflow, selected with `perband: false`, retains `mini_templates/`, `diff_stamps/`, and `lightcurve_*.csv`. Its square-aperture and PRF conventions remain available. It also benefits from SCC FITS resolution and padded-grid kernel replay.
