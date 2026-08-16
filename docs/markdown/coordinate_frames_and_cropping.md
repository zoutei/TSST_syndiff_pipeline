# Coordinate frames, MappingGrid, and temporal-WCS cropping

This reference defines the coordinate contract used by distortion-aware field
templates. It is the authoritative operational guide for `mapping`, `remap`,
field `downsample` (L5), `scc_bootstrap`, and diff template loading.

The pipeline uses four-sided padded science geometry: templates provide PS1
support over the convolution halo, while science FFIs are neutral-padded and
invalid-masked to the same array shape before Hotpants. Final science products
are then selected with the declared MappingGrid science slice.

```{contents}
:local:
:depth: 2
```

See [field geometry](field_geometry.md) for the L0–L5 algorithm and
[oversampled templates](oversampled_templates.md) for Hotpants behavior at
`F > 1`.

## 1. WCS-frame safety

The temporal Chebyshev/B-spline WCS is fitted in science-local pixels;
mapping and Exact remap use physical full-FFI pixels. `TemporalWcsAdapter`
is the required boundary between those frames. It is the only component that
translates full-FFI coordinates to/from the raw temporal model; array padding
is owned only by `MappingGrid`.

## 2. The four coordinate concepts

| Name | Symbol / example | Used for | Never use it as |
| --- | --- | --- | --- |
| Full FFI | `(x_ffi, y_ffi)` | Physical detector coordinates; external SPOC/TESS WCS, PS1 cross-projection, mapping and Exact remap | A local NumPy index |
| Science local | `(x_sci, y_sci) = (x_ffi-XSCI0, y_ffi-YSCI0)` | Raw temporal-model fit domain | An external-WCS input |
| Template local | `(x_tmp, y_tmp)` | Indexing the approved `MappingGrid` template array | A temporal-model input |
| Oversampled template | `(x_os, y_os)` | Array index at factor `F` | A physical detector coordinate |

All bounds are half-open. For a science rectangle `S` and native convolution
margin `P`, the geometry is:

```text
science full-FFI:  S=[XSCI0,XSCI1) × [YSCI0,YSCI1)
template support:  T=[XSCI0-P,XSCI1+P) × [YSCI0-P,YSCI1+P)
science local:     [0,science_width) × [0,science_height)
template local:    [0,template_width) × [0,template_height)
```

The template-support halo is mapped PS1/WCS support. The matching science halo
is not observed science data: it is neutral-filled and invalid-masked.
`MappingGrid` owns full-FFI ↔ array conversions and the science slice. Do not
reconstruct them from CRPIX, array dimensions, or literal offsets.

## 3. Oversampling

Oversampling changes array sampling only. It does not multiply a physical FFI
coordinate. With template origin `(XTMP0,YTMP0)` and factor `F`, the center of
oversampled pixel `(ix,iy)` is:

```text
x_ffi = XTMP0 + (ix + 0.5)/F - 0.5
y_ffi = YTMP0 + (iy + 0.5)/F - 0.5
```

`create_coords_for_grid()` is the canonical producer of these full-FFI
coordinates. Mapping and remap must continue to use it. `MappingGrid` then
converts native full-FFI crop bounds to the corresponding template-local or
oversampled slice.

## 4. Temporal-WCS boundary

The serialized temporal manifest carries a frame contract:

```text
model_pixel_frame: science_local
model_origin_ffi: [XSCI0, YSCI0]
science_shape: [height, width]
pixel_origin: 0
frame_contract_fingerprint: <stable fingerprint>
```

`TemporalChebWcsStore.for_stem(stem)` returns a full-FFI
`TemporalWcsAdapter` plus cadence. The adapter accepts and returns full-FFI
pixels, subtracting/adding `model_origin_ffi` only at the raw model boundary.
It provides the Astropy-compatible methods required by mapping and remap:

```python
adapter, btjd = store.for_stem(stem)
ra_dec = adapter.all_pix2world(full_ffi_pixels, 0)
x_ffi, y_ffi = adapter.world_to_pixel_values(ra, dec)
```

Production mapping, L2 scheduling, and L4 Exact workers must use this adapter.
The crop-local raw model is reserved for fitting/serialization; callers must
not call it with full-FFI pixels. A missing or internally inconsistent frame
contract is a hard load failure, not an implicit `(0,0)` fallback.

## 5. Artifact and cache contract

The following artifacts must agree on the MappingGrid recipe and, for
a temporal lane, the temporal frame-contract fingerprint:

```text
mapping master + skycell CSV + regmaps
→ remap shift schedule + L4a/L4b Exact caches
→ L5 contribs + field_mode_assembly.json
→ scc_bootstrap handoff + on-demand template/FITS materialization
```

Mapping FITS metadata includes `MAPGRID`, full-FFI template bounds,
science bounds, `CONVPAD`, `OVERSAMP`, coordinate-frame declaration, and a
geometry fingerprint. A mismatch is a rebuild condition. In particular, do
not reuse an L4 cache or L5 store built with a raw crop-local temporal model.

## 6. Skycell completeness is a hard gate

Before L5 writes or reuses contribs, it compares the exact master skycell set
to convolved-store availability. The diagnostic distinguishes cells absent
from the master, source, processing result, or convolved store. Any required
missing cell raises `L5CompletenessError`; L5 must not drop the batch and
leave a zero-flux area.

An OS4 mapping list must be compared with the OS4 convolved inputs. A native
cell list or a non-empty Zarr group is not evidence of completeness.

## 7. Required validation

1. At left/centre/right and top/middle/bottom samples, compare original
   full-FFI WCS and the temporal adapter at the same physical pixel.
2. Test full-FFI → world → full-FFI round trips at science corners and the
   declared template pad boundary for `F=1` and `F=4`.
3. Verify MappingGrid full-FFI/local/oversampled conversions and science slice
   placement for the lane's factor.
4. Validate mapping, remap, and L5 provenance/fingerprints before reuse.
5. Require an exact master-versus-convolved skycell match before L5.
6. Materialize representative flux and COUNT outputs; inspect science corners,
   right edge, and declared padded rows separately.

These checks are required for every padded template geometry.

## 8. Four-sided padded operation

Four-sided padding preserves every declared science pixel after convolution.
The template-support rectangle is mapped PS1/WCS support. The matching science
halo is neutral-filled and invalid-masked; it is never treated as an observed
FFI sample or accepted for fitting.

### Paired Hotpants inputs

Hotpants receives science and template images of the **same padded shape**.
`MappingGrid` pads the native science FFI and provides the matching template
geometry; Hotpants/kernel fitting/convolution run on that pair; only afterward
are diff, convolved, noise, mask, and background planes trimmed with the exact
four-sided science slice. A fabricated science pad is neutral-filled and
invalid-masked, so it cannot be selected for substamps or kernel fitting.

Do not crop the template to science size before Hotpants and do not trim only
one member of the pair. This belongs in `grid_pairing` and must not be recreated
with literal pad widths in individual diff stages.

The effective kernel support is recorded as `P`. `MappingGrid` records the
science rectangle, template-support rectangle, four pad widths, and exact
native/oversampled science slices. Mapping/remap/L5 operate over template
support; paired diff processing applies the corresponding neutral-invalid
science padding. No stage may replace these declarations with a local constant.

The FITS and sidecar contract records both science and template-support bounds,
the four pad widths, the science slice, oversampling, kernel-support margin,
temporal extrapolation limit, and the geometry fingerprint.
