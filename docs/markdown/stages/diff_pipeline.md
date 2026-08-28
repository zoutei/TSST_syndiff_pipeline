> **Package integration**: `syndiff` stage `diff` · package `syndiff_pipeline/difference_imaging/` · configured by `pipeline.yaml`'s embedded `diff:` block (schema v2; see [config_schema_v2.md](../config_schema_v2.md)) or a legacy standalone `config/diff_config*.yaml`  
> **Related docs**: [template pipeline guide](../template_pipeline.md) · [photometry](../photometry.md) · [multi-kernel](multi_kernel_diff.md) · [gridded ePSF](gridded_epsf.md) · [background](background.md) · [field geometry](../field_geometry.md)

> **Field mode:** with `geometry_mode: field` there are no event-local `dx/dy`
> template FITS — templates are **assembled per `group_id`** on demand from the
> SCC store `{data_root}/…/templates/oversampling_{N}/`. `shared_mask`,
> `hotpants`, and the multi-kernel engine are field-aware (convolved products
> keyed by `group_id`). See [field geometry](../field_geometry.md).
>
> **Oversampling / stamp modes:** templates may be built at
> `oversampling_factor F>1`. Hotpants accepts `oversample`, `stamp_mode`,
> `region_*`. See [oversampled templates](../oversampled_templates.md).

# Difference-Imaging (`diff`) Stage — Internal Pipeline Reference

The orchestrator runs this as **three** Condor stages — `diff_prep` (deps=`("downsample",)`, pool `diff_prep`), `background_estimate` (deps=`("diff_prep",)`, pool `background_estimate` — the memory-hungry one, needs its own big-RAM `condor:` profile), and `diff` (deps=`("background_estimate",)`, pool `diff`) — but `syndiff status`/`progress` still show all three as one `diff` row/column, and `syndiff diff submit`'s default preset activates all three together (`orchestration/stages.py`). Internally each stage runs the same **ordered YAML pipeline of sub-stages** (`orchestration/execute.py: run_config_pipeline()`, restricted to its own kind subset via the `kinds` parameter — `cfg.pipeline` itself is never filtered, so the workspace config lock's fingerprint agrees across all three), validated against `STAGE_KINDS` in `orchestration/validate.py`:

`shared_mask`, `hotpants`, `kernel_fit`, `convolved_templates`, `background_estimate` (formerly `kernel_subtract`), `epsf`, `centroids`, `sat_template`, `subtract`, `background_temporal_smoothing` (formerly `background`), `photometry` (delegator → [`syndiff photometry`](../photometry.md))

`diff_prep` owns `shared_mask`/`kernel_fit`/`convolved_templates`; `background_estimate` owns `background_estimate` alone; `diff` owns everything else (`background_temporal_smoothing`, `hotpants`, `epsf`, `centroids`, `temporal_wcs`, `per_ffi_wcs`, `sat_template`, `subtract`).

**Default site config** ([`config/diff_config.yaml`](../../../config/diff_config.yaml), schema v1): `shared_mask` → `hotpants` only. The committed schema v2 reference, [`config/pipeline.yaml`](../../../config/pipeline.yaml)'s embedded `diff:` block, instead carries the single-kernel recipe (`shared_mask` → `kernel_fit` → `convolved_templates` → `background_estimate` → `background_temporal_smoothing` → `subtract`) — see §3. Astrometry and forced photometry are **not** default diff kinds — use [`syndiff photometry`](../photometry.md) (kinds `astrometry` / `forced_photometry` in `photometry_config.yaml`), or (schema v1 standalone `diff_config.yaml` sites only — a v2 `diff.pipeline` rejects it) add an optional `kind: photometry` delegator that points at a photometry YAML.

Preamble entries (no `kind`, must precede the first stage): `external_workspaces` only. `workspace_inherit` is **not supported** under SCC-only diff storage.

### Field-mode v2 handoff (`scc_bootstrap`)

When `data_root` is set and templates exist with `field_mode_assembly.json` **schema v3** + `mapping_grid`, `execute.py` loads handoff via `scc_bootstrap.load_scc_diff_handoff_for_config()`:

- `bookkeeping/diff/frames.csv` — per-FFI manifest with `group_id`
- `bookkeeping/diff/diff_job.json` v2 — `mapping_grid`, `crop_bounds`, store names
- Diff products written SCC-primary to `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/`

No separate scheduler stage. The diff-side `crop_mode` / `crop_box_size` knobs are removed entirely (they were a silent no-op since commit `041e996`): the SCC diff crop is always `mapping_grid.science_ffi_bounds()`, a property of the SCC, never overridable per diff config.

The bootstrap also validates frozen MappingGrid geometry and, for temporal-WCS
lanes, the temporal frame-contract fingerprint before it permits on-demand
assembly. This prevents a template/remap product built with a crop-local WCS
being mixed with a full-FFI mapping lane. The science crop remains native;
template-local/oversampled conversion is owned by `MappingGrid`. See
[coordinate frames and cropping](../coordinate_frames_and_cropping.md).

Per-FFI diff completeness and resume use `data_root/bookkeeping/provenance.db` when indexed (BK-5). Frame manifests for verify come from SCC handoff `bookkeeping/diff/frames.csv` only. See [storage layout](../storage_layout.md#provenance-bookkeeping-data_rootbookkeeping) and [`doc/template_bookkeeping_plan.md`](../../doc/template_bookkeeping_plan.md).

---

## 0. Config ownership (`diff:` vs `mask_settings`)

Site authoring is intentionally split. As of schema v2 the diff policy is
embedded in `pipeline.yaml` rather than a standalone file — see
[config_schema_v2.md](../config_schema_v2.md) for the full old→new key map
and the storage/edit-safety tiers below only summarize it:

| File | Owns |
|------|------|
| `pipeline.yaml` `diff.defaults:` | Multi-stage knobs: `n_jobs`, `max_ffis`, `workspace_run_id`, `pipeline_plots*`. (Crop is **not** here — see above.) |
| `pipeline.yaml` `diff.pipeline:` `kind:` blocks | Stage-only knobs. **Omit keys that match dataclass defaults** in `stage_params.py` |
| `mask_settings.yaml` (optional sibling of `pipeline.yaml`) | Mask geometry/policy (style, maglims, strap/edge/PS1 *policy*, TNS, asteroids). **Not** embedded in `diff:` — `diff.mask_settings` exists in the schema but is not yet consumed by any resolver |
| `deployment.yaml` | `workspace_root`, `data_root`, credentials |

`- kind: shared_mask` does **not** require a `mask_settings:` path. Resolve order is stage path → `{lane}/mask_settings.yaml` → site `mask_settings.yaml` → packaged defaults, where `{lane}` is the SCC diff lane root (`cfg.output_dir`). An existing `{lane}/mask_settings.yaml` wins over later site edits until removed/replaced. See [masking.md](../masking.md).

The frozen per-lane record `{lane}/diff_config.yaml` (`chmod 444`, written once by `write_immutable_workspace_config_snapshot`) is a **slim snapshot** (`cfg_to_snapshot_dict`): empties, SynDiffConfig defaults, and bundled `straps_csv`/`bsc_catalog` paths are omitted; pipeline stages drop keys equal to their param-dataclass defaults. It is an immutable audit record, not an input — nothing reads it back into a running pipeline. Legacy full dumps still load via `load_config`. The file that **is** read every tick is `runs/{run_id}/config.yaml`, frozen verbatim at submit — see [config_schema_v2.md](../config_schema_v2.md#the-three-edit-safety-tiers-of-the-frozen-run-config) for what's safe to hand-edit there.

Bundled straps / BSC: leave `straps_csv` / `bsc_catalog` unset (empty = packaged resource at use time). Do not expect absolute bundled paths in the frozen snapshot.

---

## 1. Workspace layout and naming

**SCC-primary (field mode v2):** subtract, ePSF, centroids, and shared-mask FITS are written under `{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/` (flat label subdirs such as `hp_d/`, `epsf_r1/`). Per-FFI stems use `tess{digits}-s{SSSS}-{C}-{K}_{label}.fits.fz` (`support/ffi_naming.py`). Completeness is indexed in `provenance.db`; event `ws/` trees are not populated with diff FITS.

- Event root: `{workspace_root}/events/{event_name}/s{SSSS}_c{C}_k{K}/` — holds only per-event photometry now (`phot_{run_id}/`); there is no `ws/` tree.
- Photometry tree: `phot_{photometry_run_id}/` (astrometry JSON, `targets.reg`, `{lc_label}/lightcurve_*.csv`).
- Lane root artifacts: `shared_mask.fits.fz`, `hotpants_substamp_stars.csv`, `gaia_catalog_pipeline.csv`, the frozen `diff_config.yaml` + `diff_config.fingerprint` lock, and `mask_settings.yaml`, all directly under `diff_{lane}/`.
- Meta paired with a diffs label (`hp_d` → `hp_m`): `kernel_reconstruction.npz`, `phot_calib.csv`, `hotpants.progress.json` (lane-local when written).
- Astrometry for photometry runs: `phot_{run_id}/astrometry_result.json` (not the legacy `ws/` root).

## 2. Sub-stages

### Astrometry and forced photometry (not default diff kinds)

`astrometry` and `forced_photometry` run in the **`photometry`** orchestrator stage
([photometry pipeline](photometry_pipeline.md)), not as entries in diff
`STAGE_KINDS`. Prefer:

```bash
syndiff photometry run --site config/ \
  --photometry-config config/photometry_config_2020ut_gepsf_lc.yaml \
  --targets config/targets_example.csv \
  --target-name s0020_c3_k3_2020ut
```

Optional in-diff hook (schema v1 standalone `diff_config.yaml` only — a
schema v2 `diff.pipeline` rejects the `photometry`/`astrometry`/
`forced_photometry` kinds outright): `- kind: photometry` with
`config: path/to/photometry_config.yaml` delegates to the same runner during
a diff job. Method details:
[forced_photometry.md](forced_photometry.md). Astrometry implementation:
`difference_imaging/stages/astrometry.py`.

### `shared_mask` (`stages/masking.py` → `syndiff_pipeline.difference_imaging.masking`)
Builds the shared bitmask and selects isolated Hotpants reference stars.
**Default style is empirical** (see [masking.md](../masking.md)): Gaia
`tess_mag` &lt; `epsf_mag_lim` (7.5) + all BSC → bit 1 (crosses);
`epsf_mag_lim` ≤ `tess_mag` &lt; `bright_maglim` (13) → bit 2 (circles);
13≤T&lt;18 `faint_star_squares` (bit 32), straps/edges/PS1 (4/8/16), optional TNS (64) and per-cadence asteroids
(128 via `MaskCatalog`). Rollback: site `mask_settings.yaml` with
`shared.style: tessreduce` (bits 1/2/4/8/16 only).

**Config:** mask policy comes from `mask_settings.yaml` (sibling of
`pipeline.yaml`, or packaged defaults). Stage YAML only needs
`- kind: shared_mask` plus optional Hotpants ref-star selection keys
(`ref_mag_*`, `ref_isolation_*`, `ref_separation_px`) or an explicit
`mask_settings:` path override. Do not put maglims/strapsize/PS1 thresholds on
the stage (legacy keys still work if explicit; prefer the mask file).

Writes `shared_mask.fits.fz`, `hotpants_substamp_stars.csv`,
`gaia_catalog_pipeline.csv`, frozen `mask_settings.yaml`. See [masking.md](../masking.md).

When templates are oversampled (`OVERSAMP=F` or field OS store), the PS1
`COUNT` plane is HR; syndiff **block-sums** each `F×F` block to native before
comparing to `ps1_min_hit_count` so the mask stays science-shaped. See
[oversampled templates §8](../oversampled_templates.md#8-shared-mask-and-ps1-count).

### `hotpants` (`stages/hotpants.py`)

Per-FFI Hotpants: cropped science frame vs the PS1 template of that frame's WCS group. Runs in-memory through pyhotpants `Hotpants.run_pipeline()`; FITS written afterward.

**Science crops are always native.** Template crops scale by `OVERSAMP` (or
field OS) so an HR template of shape `science * F` is passed with
`oversample=F` (inferred from shapes unless YAML sets `oversample` explicitly).
`F>1` and `stamp_mode: connected_regions` force the pure-Python backend;
`hp_force_convolve` must be `"t"`. Diffs / noise / mask / bkg outputs remain
**native (LR)**.

Optional YAML (beyond classical `hp_*` keys):

| Key | Default | Notes |
|-----|---------|-------|
| `oversample` | inferred | Usually omit |
| `use_c_extension` | auto | Forced off for `F>1` / connected regions |
| `stamp_mode` | `grid` | `grid` \| `connected_regions` |
| `region_*` | see guide | Only for `connected_regions` |

Full tables: [oversampled templates §6](../oversampled_templates.md#6-hotpants-parameter-reference).

Outputs (per YAML `output:` block) under `{data_root}/…/diff_{lane}/{diffs}/`:
`tess{pid}_{diffs}.fits.fz` (PRIMARY + NOISE + MASK), optional convolved model,
Hotpants background, and stamps. Production default (`config/diff_config.yaml`):
`write_convolved: false`, `write_bkg: true`, `write_stamps: false`,
`write_kernel_solutions: true`.

When `write_kernel_solutions: true`, per-frame kernel vectors are persisted as
`{diffs_label}_kernels/{product_id}_kernel.npz` on the SCC lane. See §5 and
[multi-kernel diff](multi_kernel_diff.md).

### Multi-kernel path (`kernel_fit` → `convolved_templates` → `background_estimate`)

Fits one reusable target-level kernel, convolves group templates, then
algebraically subtracts. Full guide: [multi_kernel_diff.md](multi_kernel_diff.md).
Artifacts land under `diff_{lane}/` (e.g. `kernel_fit/kernel_r2.npz`, convolved
templates, `ks_d` / `ks_b`).

### `background_temporal_smoothing` (`stages/background/pipeline.py`)

Unified background cube (spatial photutils, temporal Savitzky–Golay, strap correction). See [background.md](background.md). Writes `stack.npz`/`stack.npy` and optional per-frame FITS.

### `subtract` (`support/subtract.py` + `execute.py`)

Per-frame linear combination of SCC lane planes (or the virtual cropped `ffi`
label), e.g. `expression: "ks_d + ks_b - ks_b_s"` → `diff_{lane}/ks_d_s/`.

### `epsf` (`stages/epsf.py`, `stages/gridded_epsf.py`)

Per-frame gridded empirical PSF on difference images. Deep dive:
[gridded_epsf.md](gridded_epsf.md). Outputs under `diff_{lane}/{output}/`:
`*_gridded_epsf.npz`, `gridded_epsf_index.json`, progress sidecars. Legacy
tile-stack bundles remain for `sat_template` only.

### `centroids` (`stages/centroids.py`)

Gaia-star PSF photometry on diffs using gridded ePSF. Deep dive:
[centroids.md](centroids.md). Outputs: `*_photresults.ecsv`,
`centroids_index.json` under `diff_{lane}/{output}/`.

### `sat_template` (`stages/sat_template.py`) — see §6

Builds per-group model images of bright stars as flux-scaled ePSF stamps under
`diff_{lane}/{output}/` (known gaps — §6).

### `photometry` (delegator)

YAML `- kind: photometry` with `config: <photometry_config.yaml>` runs
`run_photometry_delegator()` inside the diff job. Schema v1 standalone
`diff_config.yaml` only — `parse_unified_diff_policy` rejects this kind in a
schema v2 `diff.pipeline` (diff is SCC-scoped, not event-scoped; this kind
belongs in `photometry_config.yaml`). Prefer the standalone
`syndiff photometry` noun for event LCs either way. See [photometry_pipeline.md](photometry_pipeline.md).

## 3. Production pipeline orders

| Config | Order |
|--------|-------|
| `config/diff_config.yaml` (schema v1, **default**) | `shared_mask` → `hotpants` |
| `config/pipeline.yaml` `diff:` (schema v2 reference) / `config/diff_config_single_kernel.yaml` (schema v1) | `shared_mask` → `kernel_fit` → `convolved_templates` → `background_estimate` → `background_temporal_smoothing` → `subtract` |
| `config/archive/diff_config_multi_kernel.yaml` | multi-kernel prefix → optional round-2 `hotpants` |
| `config/archive/diff_config_*_epsf*.yaml` | often `epsf` (± `centroids`) on an existing `hp_d` lane |
| Event LCs | **`syndiff photometry`** + `photometry_config*.yaml` (`astrometry` / `forced_photometry`) |

## 4. Template resolution

Template filename pattern (parsed by `parse_syndiff_template_filename()` in `stages/hotpants.py`):

```
syndiff_template_s{sector}_{camera}_{ccd}[_x{x0}-{x1}_y{y0}-{y1}][_osN]_dx{dx}_dy{dy}.fits[.gz]
```

Per-frame selection: the manifest row gives `(group_id, group_dx, group_dy)`; `find_template_by_offset()` matches the filename `dx`/`dy` against the manifest offsets with tolerance `max(1e-5, 0.01 × offset_threshold)`; `cfg.template_paths[group_id]` then holds the absolute path. Discovery prefers flat `syndiff_template_*` files under `template_dir` (the SCC template store; no `ws/templates` symlink is created); fallback is `group_{id}/ps1_template.fits`.

This is the hook for swapping in modified templates: point `template_dir` at a directory of alternative templates with the **same filenames**, and every sub-stage picks them up.

## 5. Kernel persistence — what is and is not saved

| Location | What | Per-frame? |
|----------|------|------------|
| `diff_{lane}/{diffs_m}/kernel_reconstruction.npz` | Hotpants **basis** stack + config scalars | No (one per hotpants pass) |
| `diff_{lane}/{diffs_m}/phot_calib.csv` | `kernel_sum`, `tess_zp` per FFI | Yes (scalars only) |
| in-memory (default) | Full per-frame `kernel_solution` vector | Yes — **discarded** after each frame |
| `diff_{lane}/{diffs_label}_kernels/{product_id}_kernel.npz` | Per-frame `kernel_solution` + Hotpants config scalars | Yes — only when `write_kernel_solutions: true` |
| `diff_{lane}/{kernel_fit}/kernel_r2.npz` | Target-level kernel from `kernel_fit` | One per target |

**Per-frame Hotpants kernel vectors are not written by default** (see docstring at the top of `stages/hotpants.py`). Set `write_kernel_solutions: true` on a `hotpants` stage to persist them under `{diffs_label}_kernels/`. Consequences:

- "Convolve a new/modified template with the already-derived kernel" is only possible on the **kernel_fit path**: reuse `kernel_r2.npz`, re-run `convolved_templates` (with `skip_existing: false` or cleared outputs) → `background_estimate` → `forced_photometry`.
- On the default hotpants-only path, changing a template requires re-running `hotpants` (new per-frame fits).

## 6. `sat_template` — current behavior and known gaps

Intended purpose: model images of the **PS1-removed** (saturated) stars, per WCS group, for later subtraction. Actual behavior has two gaps that matter when planning star-subtraction work:

1. **Star selection**: `execute.py: _load_removed_stars_in_crop()` returns the full Gaia catalog whenever `gaia_df` already carries crop-local `x`/`y` columns — which is always true after `shared_mask`. So in production runs `removed_stars_csv` (i.e. `events/{label}/ps1_removed_stars.csv` produced by downsample) is **ignored**, and the "sat" template contains *all* Gaia stars in the crop, not just removed ones.
2. **Subtraction wiring**: outputs are per-group `group_{gid}.fits.fz`, but the `subtract` stage consumes per-frame `tess{pid}_{label}.fits.fz` planes. No code bridges the two; the example config `config/example/diff_config_b_epsf_sat_bkg.yaml` that pairs them is stale (it also references a removed `background_estimate` kind). `load_group_templates()` is unused outside the module.

There is currently **no mechanism to subtract a single, chosen star from a template** via the main diff pipeline — see [host-star light curves](../star_lightcurves.md) and [star pipeline](star_pipeline.md) for the `syndiff star` workflow that builds per-host mini templates and star-only diff stamps from persisted per-frame Hotpants artifacts.

### Downstream: host-star light curves

The **`star`** stage reads baseline diff workspaces (not `hp_d` images directly):

- `convolved` (e.g. `hp_c`) — per-frame convolved template windows
- `{diffs}_kernels` — per-frame Hotpants kernels
- `phot_bkg` (e.g. `ks_b_s` or `ks_b`) — photutils background subtracted in star stamps (**not** `hp_b`)

Produce `ks_b` via `background_estimate`; smooth to `ks_b_s` with the `background_temporal_smoothing` stage. Star config sets `baseline.phot_bkg` explicitly.

## 7. Config schema highlights

Keys under schema v2 `diff:` (`pipeline.yaml`) — or top-level in a schema v1 standalone `diff_config.yaml`: `deployment_file` (v1 only; v2 always uses the site's top-level `deployment_file`), `defaults` (merged into `SynDiffConfig`: `n_jobs`, `pipeline_plots`, `workspace_run_id`, … — `crop_mode`/`crop_box_size` are removed, not just omittable), `pipeline` (ordered stage list; unknown keys per stage fail validation via `*_ALLOWED` frozensets in `orchestration/stage_params.py`; v2 additionally rejects event-scoped kinds), `overrides` (keyed `"sector/camera/ccd"`), `condor` (`request_cpus`, `request_memory`, `host_stats_min_mem_mb`, `host_stats_max_load15` — legacy `requirements` / `rank` rejected). `additional_forced_targets` / `per_event_force_targets` are dead on the diff side in both schemas (silently ignored in v1, rejected outright in v2) — set them in `photometry_config.yaml`. See [config_schema_v2.md](../config_schema_v2.md) for the full key map.

A frozen per-target copy of the effective config is written to `runs/{run_id}/per_target/{label}/diff_config.yaml` at stage execution (`stages._frozen_diff_config_path` + `site_config.write_frozen_diff_config()`) — write-only debug output, not read back. It is resolved frozen-first from `runs/{run_id}/config.yaml`'s embedded `diff:` policy when present (`diff_verify.frozen_diff_config_for_verify()`), falling back to a live `site_config.freeze_target_diff_config()` load only for schema v1 sites with no embedded policy.

## 8. Recipes: reusing prior work

| Goal | What to reuse | What to re-run |
|------|---------------|----------------|
| New photometry on existing diffs | `{data_root}/…/diff_{lane}/hp_d/*.fits.fz` | `forced_photometry` only (or `syndiff` photometry stage) |
| Modified templates, kernel-fit path | `kernel_r2.npz` + `kernel_fit_meta.json` | `convolved_templates` → `background_estimate` → (`background_temporal_smoothing`/`subtract`) → `forced_photometry` |
| Modified templates, hotpants path | shared mask, substamp stars | full `hotpants` (per-frame kernels re-fit) |
| Continue a multi-kernel run | re-run upstream diff stages on the SCC lane | remaining stages in a new run |

## 9. Key files

| Concern | Path |
|---------|------|
| Pipeline driver | `difference_imaging/orchestration/execute.py` |
| Stage kinds / validation | `difference_imaging/orchestration/validate.py`, `stage_params.py` |
| YAML → config | `difference_imaging/orchestration/config.py`, `site_config.py` |
| Template matching | `difference_imaging/support/template_resolution.py` |
| Kernel fit / convolve / subtract | `difference_imaging/stages/kernel_fit.py`, `convolved_templates.py`, `kernel_subtract.py`, `kernel.py` |
| Sat templates | `difference_imaging/stages/sat_template.py` |
| Photometry / ePSF / centroids | `difference_imaging/stages/photometry.py`, `epsf.py`, `gridded_epsf.py`, `centroids.py`, `epsf_progress.py` |
| Paths / naming | `difference_imaging/support/paths.py`, `ffi_naming.py` |
