# Shared masking (empirical / TESSreduce / TNS / asteroids)

Default shared-mask style is **empirical** (behavior change on upgrade). Set
`shared.style: tessreduce` in site `mask_settings.yaml` to keep the old
TESSreduce-only bit layout (bits 1/2/4/8/16 only; no 32/64/128).

## Package

Code lives in [`syndiff_pipeline/masking/`](../../syndiff_pipeline/masking/):

| Module | Role |
|--------|------|
| `bits.py` | Bit constants + `epsf_reject_mask` / `full_mask_bool` / strap helpers |
| `geometry.py` | Packaged `mask_geometry.yaml` radii / crosses (numba painters) |
| `settings.py` | `MaskSettings` load/resolve/freeze |
| `shared.py` | Hybrid `build_static_mask` + tessreduce `make_shared_mask` / `Cat_mask` |
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
| 32 | `FAINT_CAT` | TESSreduce squares; `bright_maglim` ≤ T &lt; `faint_maglim` (18) |
| 64 | `TNS` | Circles from `transient_fixed` (usually static) |
| 128 | `ASTEROID` | Per-cadence only — never in static FITS |

Catalog source is **Gaia** (+ BSC for crosses). Never `ps1_removed_stars`.

## Settings

Copy [`config/mask_settings.example.yaml`](../../config/mask_settings.example.yaml)
to site `mask_settings.yaml`. Resolve order: stage `mask_settings` path →
`{ws}/mask_settings.yaml` → `{site}/mask_settings.yaml` → packaged defaults
(empirical; TNS/asteroids **enabled**).

`tns.download_url` is omitted from the example — code default
`DEFAULT_TNS_PUBLIC_ZIP_URL`. Stage keys `gaia_mag_bright` / `strapsize` /
`ps1_min_hit_count` still override the corresponding settings fields (BC).

On `shared_mask` execute, effective YAML is frozen to `{ws}/mask_settings.yaml`.
On submit, if the site file exists it is also copied to `runs/{run_id}/mask_settings.yaml`.

## Artifacts

| Artifact | Location |
|----------|----------|
| `shared_mask.fits.gz` | event `ws/` (int16 static; may include bit 64) |
| `mask_settings.yaml` | event `ws/` (frozen effective) |
| `transient_fixed.parquet` | event `ws/` |
| TNS public CSV | `{data_root}/catalogs/tns/tns_public_objects.csv` |
| `pixel_intervals.parquet` + `asteroid_ffi_times.parquet` | `{data_root}/catalogs/sector_*/camera_*/ccd_*/asteroids/` |
| `TESS_orbit_times.csv` | `{data_root}/catalogs/` (auto-downloaded from MIT) |
| QA PNGs | `{pipeline_plots}/masks/` when `pipeline_plots: true` |

## `MaskCatalog.mask_at`

```python
cat.mask_at(time, which="full"|"static"|"temporal", out=None, as_bool=False)
```

- `time`: cadence `int` or BTJD `float` (nearest entry in `asteroid_ffi_times` within ~½ cadence).
- `which="full"`: static copy then OR bit 128 for active cadence.
- Asteroid row/col are **1-based full-FFI**; converted to crop-local once on load.

## Consumers

| Stage | Predicate |
|-------|-----------|
| Hotpants / kernel_fit / kernel_subtract / spatial bkg | `full_mask_bool(mask_at(..., which="full"))` |
| Strap QE / phot_mask | source bits `1\|32`; strap bit `4` |
| ePSF / star ePSF | ignore `1\|2`; reject any other set bit |

Hotpants loky workers install the catalog once and resolve BTJD→cadence per frame.
ePSF loads `ws_root/shared_mask.fits.gz` (not the event `out/` root).

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
3. Confirm ePSF still gets T&lt;13 stars (bits 1|2 ignored).
4. Spot-check `debug_plots/masks/` including bits 64/128 on a known SCC.
