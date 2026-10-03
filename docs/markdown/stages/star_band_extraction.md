# Per-band target add-back and photometry

`syndiff star extract-band --manifest /absolute/path/manifest.json` measures targets
on saved per-band subtraction products. It consumes frozen target-component and
PSF/WCS products; it does not retrain the forward model or rebuild a subtraction
lane. The older `star run`/`submit` path continues to consume Hotpants products.

This entry point needs only the ordinary pipeline runtime, not the development
forward-model package. A producer exports profiles and transported components
before invoking it. The reference implementation and 1,000-target investigation
are recorded in `/astro/armin/koji/syndiff/dev_runs/faint_target_photometry_20261002/`.
Its demonstration inputs are explicitly provisional, not a final paper generation.

## Measurement contract

The restored stamp is

$$D_t(x,y)=D(x,y)+a(x,y)M_t(x,y).$$

Here $D$ is the saved difference image in electrons per second per native TESS
pixel; $D_t$ is the restored target stamp in the same units. $M_t$ is the retained
PS1 target component after the producing blur, seam handling, registration and
per-band convolution, in the template's linear flux-rate units per native pixel.
The saved photometric scale $a$ converts those units to TESS electrons per second
and is evaluated at each output pixel. $x,y$ are science-local native pixel
coordinates. The subtraction's fitted constant offset is already in $D$ and is
not added back.

For crowded stamps, the solver fits

$$D_t(p)=f_tP_t(p)+\sum_j f_jP_j(p)+\beta.$$

The index $p$ labels an accepted native pixel. $P_t$ and $P_j$ are the dimensionless
pixel fractions of the target and neighbouring-source profiles, exported from
the fitted ePSF/WCS model. $f_t$ is the target flux in electrons per second;
$f_j$ are freely fitted neighbour **residual** amplitudes in the same units,
not necessarily their total stellar flux. The fitted constant $\beta$ is in
electrons per second per pixel. No catalogue flux prior is imposed on $f_t$.

The target is projected against the nuisance-profile and sky column space.
Degenerate nuisance columns do not by themselves invalidate an identifiable
target. Targets with insufficient pixels or effectively no independent profile
support receive an explicit failure status. Negative fluxes are retained;
magnitudes are only reported for positive fluxes.

`unit_sum_full_support` normalizes the complete exported profile before masks or
window cropping. Never normalize just the surviving pixels. The flux zero point
must use the same normalization convention and independent calibration stars.

## Manifest version 1

All paths should be absolute. Required fields:

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `artifact_state` | `provisional` or `validated`, describing the supplied producer generation; not certification by this command |
| `output_dir` | Task-owned output directory |
| `targets_csv` | Unique `target_index`, `objID`, `gaia_source_id`; optional PS1-predicted magnitude/error, colour and bin columns |
| `profiles_npz` | Arrays `profile`, `x`, `y`, `objID`, indexed by `target_index`; profiles centred on rounded science-local positions |
| `transport_dirs` | Ordered component directories; each target is `NNNN.npz` with native `addback_unscaled` and half-open science-local `bounds=[x0,x1,y0,y1]` |
| `flux_zero_point` | Independently established magnitude zero point for the exported profile convention |
| `frames` | Nonempty list of frame records below |

Optional fields include `profile_normalization: unit_sum_full_support`,
`neighbour_profiles_npz` (arrays `profile,x,y,source_id,tmag`), `stamp_half_size`
(default 10), `neighbour_radius_px` (default 14), and `write_stamps` (default true).

A frame specifies `id`, preferably `time_btjd`, `difference_fits`, `noise_fits`,
`physical_mask_fits`, and either `scale_image` (NumPy/FITS) or scalar `scale`.
Defaults are difference/mask HDU 1 and noise HDU 2. `noise_origin_xy` maps the
science crop into the raw noise image. Optional per-frame `profiles_npz` and
`neighbour_profiles_npz` override the defaults. Set `reference_profile_reused`
when a frame deliberately reuses the reference profile; this flag is retained
in the measurements. Optional `raw_science_fits` and `background_fits` add a
joint-fit control without PS1 add-back, using the same pixels and source basis.

Physical masks and Hotpants output flags have **different semantics**. Supply a
physical mask, including available per-frame temporal bits. The default allows
physical bit index 5 (value 32, faint-source selection). Allowing bit index 1
(value 2, bright-source circles) requires neighbour profiles. The demonstrator
also excludes a configurable radius around very bright stars (`hard_bright_tmag`
9 and `hard_bright_radius_px` 6). This does not replace a proper saturation mask.

The target component must reproduce the actual template generation. In particular:

- Read blur width, band weights, keep masks and publisher/seam ownership from the
  producing records. Do not redetect a different segment and assume equivalence.
- Derive the adopted colour offsets from **that fit's** kernel metadata. An older
  weight file's example offsets can belong to a different colour reference.
- Treat a blended retained component explicitly: restoring it can restore multiple
  sources, which must be modelled or flagged.
- A magnitude-limited Gaia preparation catalogue may be too shallow for faint
  extraction. Check its metadata; the historical F1 catalogue stops at T=18.

## Outputs and evidence limits

`measurements.csv` contains every requested target/epoch, including failures,
signed fluxes, conditional uncertainties, profile support, neighbour counts,
fit statistics, additive flux closure and optional raw-image controls.
`lightcurves/<objID>.csv` groups the same records per target;
`batch_manifest.csv` counts each target's statuses. `stamps.npz` retains data,
add-back, model, residual and optional raw-control planes, plus accepted pixels.
`run_record.json`, the copied manifest and `input_hashes.json` preserve source
and input provenance.

Formal errors condition on fixed profiles, positions, templates and background.
They include the fitted neighbour covariance, but not uncertainty in those fixed
inputs. Two epochs demonstrate repeated extraction, not full-sector precision.
Validate variability through any learned calibration and compare independent
profiles before claiming temporal or absolute photometric accuracy.

## Preparing transported components

The runtime producer is also available in the main package:

```bash
syndiff star prepare-band --manifest /absolute/path/prepare_manifest.json
# A bounded worker can select one or more targets:
syndiff star prepare-band --manifest /absolute/path/prepare_manifest.json \
  --target-index 123 --target-index 456
```

The manifest supplies `targets_csv`, `prepared_components_dir`, and a
`template_snapshot` object. Targets additionally require PS1 `raMean,decMean`
and approximate full-FFI `x_ffi,y_ffi` for window placement. Photometry still
uses the fitted positions in the profile product, not these approximate centres.

Snapshot fields are `operator_version`, `mapping_dir`, `master_filename`,
`skycell_list`, `band_cells_dir`, `publisher_lists`, `contribution_dir`,
`regmap_pattern` (containing `{skycell}`), `kernel_npz`, `adopted_weights_json`,
`store_weights`, `psf_sigma`, `blur_radius`, and `mask_cache_dir`.
The implemented, tested version is `publisher_row_v1`, the row/padding operator
used by the saved demonstration. Other versions are rejected until their own
operator closure is established; a newer store must not silently inherit an
older operator.

Frozen mask caches are keyed to band-image content and recipe. The explicit
`mask_resolution: legacy_archive_image_match` option can recover historical masks
from `historical_store_root`: it requires one matching recipe whose combined image
agrees with the saved band sum. It never selects by modification time or a current
pointer. `data_root` identifies the producing storage tree for padding metadata.

The producer replays the recorded publisher row, cross-projection padding,
registration, ignored PS1 pixel bit 12, source-position kernel interpolation,
fit-specific colour offsets and band-weight rescaling. A finite-support Gaussian
optimization reproduces the full convolution, including its boundary convention.
Outputs retain the band templates, global oversampled origin, kernel offsets and
source-component records. Cached preparation does not certify the input model.

The present source-isolation policy selects retained connected components. A
component can contain multiple sources and must then be jointly modelled or
flagged. A zero component is recorded separately; it is not proof by itself that
a complete stellar PSF was erased. Runs are process-isolated because the inherited
padding implementation uses a temporary module-level loader.
