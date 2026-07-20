# Pipeline resources

Static files required to run the template pipeline, versioned with the code.

| File | Purpose |
|------|---------|
| `skycell_wcs.csv` | PS1 SkyCells WCS table (mapping stage) |
| `bsc5/catalog` | Bright Star Catalogue 5th Ed. fixed-width table (Hotpants saturation crosses) |
| `bsc5/ybsc5.readme` | BSC5 byte layout spec |
| `tess_straps.csv` | TESS detector strap column list (`shared_mask` stage) |
| `tess_orbit_times.csv` | MIT TESS orbit start/end times (shift-schedule SG segments); auto-downloaded via `ensure_tess_orbit_times_csv()` from `https://tess.mit.edu/public/files/TESS_orbit_times.csv` |

Machine-specific paths and credentials belong in the gitignored deployment file (`deployment.yaml` beside your site `pipeline.yaml`), not here.
