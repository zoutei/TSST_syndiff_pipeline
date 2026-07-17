# Static masking (empirical / TESSreduce / TNS / asteroids)

Default **static** mask style is **empirical** (behavior change on upgrade). Set
`shared.style: tessreduce` in site `mask_settings.yaml` to keep the old
TESSreduce-only bit layout (bits 1/2/4/8/16 only; no 32/64/128).

The on-disk artifact is still named `shared_mask.fits.gz` (pipeline stage kind
`shared_mask`) for backward compatibility; it stores only the **static** layer.
Per-FFI temporal bits (asteroids) are applied in memory via `MaskCatalog.mask_at`.

## Package

Code lives in [`syndiff_pipeline/masking/`](../../syndiff_pipeline/masking/):

| Module | Role |
|--------|------|
| `bits.py` | Bit constants + `epsf_reject_mask` / `hotpants_mask_bool` / `full_mask_bool` / strap helpers |
| `geometry.py` | Packaged `mask_geometry.yaml` radii / crosses (numba painters) |
| `settings.py` | `MaskSettings` load/resolve/freeze |
| `shared.py` | Hybrid `build_static_mask` + tessreduce `make_shared_mask` / `Cat_mask` |
| `faint_star_squares.py` | Mag-binned square stamps (bit 32 `FAINT_CAT`; numba) |
| `tns.py` | Public CSV ensure + `transient_fixed` + bit 64 |
| `asteroids.py` | SCC intervals load/generate + bit 128 |
| `asteroid_discover.py` | `sbident` discover + MIT orbit-times auto-download |
| `catalog.py` | `MaskCatalog.mask_at` |
| `api.py` | `generate_shared_mask_catalog` entry point |

[`difference_imaging/stages/masking.py`](../../syndiff_pipeline/difference_imaging/stages/masking.py)
re-exports public names for backward compatibility.

## Bit layout

| Value | Name | Geometry |
|------:|------|----------|
| 1 | `BRIGHT_CAT` | Empirical circles (T≥9) + cross body; T &lt; `bright_maglim` (13) |
| 2 | `SAT_CROSS` | Cross arms/body only; T &lt; 9; Gaia ∪ BSC |
| 4 | `STRAP` | Strap columns |
| 8 | `EDGE` | Detector dead zones |
| 16 | `PS1` | Template COUNT &lt; threshold |
| 32 | `FAINT_CAT` | `faint_star_squares` (numba mag-binned squares); `bright_maglim` ≤ T &lt; `faint_maglim` (18) |
| 64 | `TNS` | Circles from `transient_fixed` (usually static) |
| 128 | `ASTEROID` | Per-cadence only — never in static FITS |

Catalog source is **Gaia** (+ BSC for crosses). Never `ps1_removed_stars`.

## Settings

Mask policy lives in **`mask_settings.yaml`**, not in `diff_config.yaml`.

You do **not** need to set anything under `- kind: shared_mask` for mask
geometry/policy. A bare `- kind: shared_mask` is enough.

Resolve order ([`resolve_mask_settings`](../../syndiff_pipeline/masking/settings.py)):

1. Optional stage path: `mask_settings: /path/to.yaml` under `- kind: shared_mask`
2. Already-frozen `{ws}/mask_settings.yaml` (from a prior `shared_mask` run)
3. Site `{site_dir}/mask_settings.yaml` (sibling of `diff_config.yaml`)
4. Packaged code defaults (`MaskSettings()` — empirical; TNS/asteroids **enabled**)

Because step 2 wins over the site file, editing site `mask_settings.yaml`
alone does **not** change an existing workspace until you remove/replace
`{ws}/mask_settings.yaml` (or force-rerun `shared_mask`).

Copy [`config/mask_settings.example.yaml`](../../config/mask_settings.example.yaml)
to site `mask_settings.yaml` when you want non-default maglims / style / TNS /
asteroids.

`tns.download_url` is omitted from the example — code default
`DEFAULT_TNS_PUBLIC_ZIP_URL`.

**Do not** put maglims / strapsize / PS1 thresholds in the `shared_mask` stage.
Legacy stage keys `gaia_mag_bright` / `strapsize` / `ps1_min_hit_count` are still
accepted for BC and override the corresponding settings fields **only when
explicitly present** (they no longer clobber `mask_settings.yaml` via dataclass
defaults). Prefer editing `mask_settings.yaml`.

`shared_mask` stage knobs that remain stage-local are Hotpants **ref-star
selection** only: `ref_mag_min` / `ref_mag_max` / `ref_isolation_*` /
`ref_separation_px`.

Two freeze locations (do not confuse them):

- `{ws}/mask_settings.yaml` — **effective** settings written on `shared_mask`
  execute (includes any stage-path YAML and legacy stage-key overrides).
- `runs/{run_id}/mask_settings.yaml` — on submit, a copy of the **site** sibling
  only (audit trail; not post-override effective settings).

## Artifacts

| Artifact | Location |
|----------|----------|
| `shared_mask.fits.gz` | event `ws/` — **static** int16 bitmask (may include bit 64; never bit 128). Pipeline stage kind remains `shared_mask` for BC. |
| `mask_settings.yaml` | event `ws/` (frozen effective) |
| `transient_fixed.parquet` | event `ws/` |
| TNS public CSV | `{data_root}/catalogs/tns/tns_public_objects.csv` |
| `pixel_intervals.parquet` + `asteroid_ffi_times.parquet` | `{data_root}/catalogs/sector_*/camera_*/ccd_*/asteroids/` |
| `TESS_orbit_times.csv` | `{data_root}/catalogs/` (auto-downloaded from MIT) |
| QA PNGs | `{pipeline_plots}/masks/` when `pipeline_plots: true` — bit planes, ePSF/Hotpants predicates, plus TSST-style `tns_locations.png` and `asteroid_tracks_by_epoch.png` |

## `MaskCatalog.mask_at`

```python
cat.mask_at(time, which="full"|"static"|"temporal", out=None, as_bool=False)
```

- `time`: cadence `int` or BTJD `float` (nearest entry in `asteroid_ffi_times` within ~½ cadence).
- `which="full"`: static copy then OR bit 128 for active cadence.
- Asteroid row/col are **1-based full-FFI**; converted to crop-local once on load.

### Per-FFI mask FITS helper

```python
from syndiff_pipeline.masking import (
    load_catalog_for_event,
    write_mask_fits_for_ffi,
    write_sector_sample_mask_fits,
)

cat = load_catalog_for_event(".../ws", crop_bounds=crop, data_root=data_root, sector=20, camera=3, ccd=3)
write_mask_fits_for_ffi(cat, "tess2020007215923", "mask.fits", wcs_table=manifest)
# or begin / mid / end of sector:
write_sector_sample_mask_fits(cat, manifest, "out_dir/")
```

`ffi_id` may be a product id (`tess…`), bare digits, or an FFI path/basename.
## Consumers

| Stage | Predicate |
|-------|-----------|
| Hotpants | `hotpants_mask_bool(mask_at(..., which="full"))` — ignore bit 32 (`FAINT_CAT`) |
| kernel_fit / kernel_subtract / spatial bkg | `full_mask_bool(mask_at(..., which="full"))` |
| Strap QE / phot_mask | source bits `1\|32`; strap bit `4` |
| ePSF / star ePSF | ignore all star stamps `1\|2\|32` (bright / crosses / faint squares); reject any other set bit via per-FFI `mask_at` |

Hotpants and ePSF loky workers install the catalog once and resolve BTJD→mask per frame (no static-mask FITS I/O per frame).
On-disk artifact remains `shared_mask.fits.gz` (static layer only; bit 128 never written there).

## Graceful degrade

TNS and asteroids default **on**. Missing public CSV → auto-download; failure →
warn and omit bit 64. Missing asteroid parquet → try generate if deps present;
else warn and omit bit 128. Set `enabled: false` to skip entirely.

## Asteroid generate deps (optional)

Consuming an existing SCC `pixel_intervals.parquet` needs **no** extra packages.
Generating candidates/tracks requires:

```bash
mamba activate syndiff
pip install tess-ephem
pip install git+https://github.com/bengebre/sbident
# or: pip install "syndiff-pipeline[asteroids]"
```

- **`sbident`** — JPL Small-Body Identification (TESS `xobs-hel` observer). Correct for discover.
- **`tess-ephem`** — on-CCD confirmation + per-FFI tracks. Still required for generate; not installed by `sbident` alone.
- **Orbit times** — MIT [`TESS_orbit_times.csv`](https://tess.mit.edu/public/files/TESS_orbit_times.csv) is auto-downloaded to `{data_root}/catalogs/TESS_orbit_times.csv` when missing or when the requested sector is absent; if still missing after re-fetch, discover falls back to `tesswcs.pointings` Start–End.

## Manual checklist (not CI)

1. One empirical `syndiff diff run` with defaults (TNS+asteroids on).
2. Compare Hotpants residuals vs a tessreduce rollback (`style: tessreduce`).
3. Confirm ePSF keeps catalog stars (bits 1|2|32 ignored) and rejects straps/TNS/asteroids.
4. Spot-check `debug_plots/masks/` including bits 64/128 on a known SCC.
