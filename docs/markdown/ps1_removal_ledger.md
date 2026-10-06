# PS1 removal ledger (isolated implementation)

This package records pixel operations and catalogue associations without changing
the `footprint_v1` image algorithm.

## Inline capture in `ps1_process` (`stages.ps1_process.removal_ledger: true`)

With the flag on, every freshly processed cell records its ledger while SEP runs
(`removal_ledger/inline.py`): the exact pixel operations through
`remove_background(recorder=...)`, the SEP result (`segmentation_union`,
`sep_bright_mask`, `sep_objects` in `geometry.npz`), and Gaia associations from the
**uncut** per-projection Gaia catalogue (proper motion propagated to the PS1 header
`MJD-OBS`, image-calibrated, 5 px candidate radius, bright-star halo radius for
T<13). No bulk PS1 stack catalogue is downloaded. Ledgers are published to
`{data_root}/ps1_removal_ledger/v1/{projection}/{cell}/{combined_fp}/` with a
`ledger.json` pointer; a failed capture leaves `error.json` there and the cell is
produced by the unchanged removal path. When a projection's rows finish, PS1 mean
magnitudes for the Gaia stars linked to a removal are fetched from the Gaia archive
cross-match (`gaiadr3.panstarrs1_best_neighbour`) into
`{data_root}/catalogs/gaia_ps1_best_neighbour/v1/proj_PPPP.parquet`, and
`ps1_removal_ledger/v1/projection_status/PPPP.json` lists cells without a ledger
(cache hits, manual path, failures). The flag never changes a template pixel and is
not part of the combined recipe. Checked bit-exact against paper cell
skycell.2528.005, with pixel operations identical to the standalone C4 ledger
(`dev_runs/inline_ledger_check_20261005`).

The standalone campaign, pilot, catalogue-query and seam-transport validation
modules described below now live in `tools/removal_ledger_campaign/` (run from the
repository root, e.g. `python -m tools.removal_ledger_campaign.campaign`); they are
for historical backfills and tests, not part of the pipeline. Use the selected
source/worktree explicitly and do not change the live editable installation to run one.

## Scope and guarantees

- Capture actual background, bright-trigger and saturation zeroing separately.
  The 480-pixel cap is part of the recorded support. Repeated operations can
  select already-zero pixels; finite signal changes are charged once.
- Record changed and nonfinite pixels, support geometry, causal trigger IDs,
  component centroids, source identities and catalogue candidates separately.
  A component is not automatically a star.
- Store the measured explicit deleted pixel values, so later seam transport does
  not need SEP or another raw download. Background support/statistics and the
  raw-input hash are retained; background noise values are not duplicated in full.
- A saved SEP union may be used directly. It can also be recovered from saved
  pre/post-background images only if no original pixel is exactly zero and the
  cache is verified to be pure zeroing. Otherwise recompute rather than guess.
- Existing pixel ledgers can be loaded with `CellLedger.from_published` for new
  catalogue associations, without SEP. The original capture implementation stays
  in provenance when a refreshed ledger is published.
- Gaia queries apply no magnitude/RP/BP cut and must agree with independent query
  counts. PS1 DR2 stack queries preserve non-primary/split measurements, page by a
  stable detection key, check counts, and keep original metadata and page hashes.
- IDs stay lossless strings (Gaia/PS1 identities) or namespaced component IDs.
  Read CSV ID columns as strings; spreadsheet numeric inference can corrupt them.
- Gaia/PS1 identity candidates use proper motion and PS1 mean epochs. Missing
  motion/epoch and blends remain unconfirmed. Catalogue-to-image calibration uses
  independent isolated pre-removal stars; original coordinates are preserved.
- Source membership at a centre and overlap with a declared candidate support are
  different fields. The candidate aperture/halo radius is not a measured total
  stellar PSF. Individual blended source flux is not inferred from a region sum.

## Artifacts and interpretation

`CellLedger.publish` writes an atomic, content-keyed directory with:

- `manifest.json`: input/output hashes, image identity, code/capture hashes,
  catalogue snapshots, WCS and calibration provenance, member checksums.
- `regions.parquet`: pixel operations, actual support bounds, counts, measured
  signed/absolute flux changes and causal triggers.
- `geometry.npz`: packed support masks, first-change ownership, component labels,
  saved segmentation union and explicit deleted values; native SEP products where
  available. It does not generate a dense image for every source.
- `sources.parquet`, `associations.parquet`: Gaia, PS1 measurement and unmatched
  component identities; candidate links, partial-support flags and coordinates.
- Immutable auxiliary tables: original records, Gaia/PS1 candidates and image
  calibrators. Original `source_id=-1` values are not used as unique keys.

`complete_cell_pixels` means the cell's pixel operations were validated. It does
**not** mean all light has an astrophysical identity or that final-template
transport has been completed. Catalogue records remain catalogue-scoped.

`ra`/`dec` describe the stated catalogue epoch; `original_ra`/`original_dec` retain
Gaia's reference positions. `pixel_x`/`pixel_y` may include measured PS1-image
offsets. Preserve these distinctions when building a TESS-epoch scene catalogue.

## Seam transport

Historical v2 ownership is snapshotted verbatim from paper code `648c0bc` as
`historical_v2.py`. Its rendering/geometry functions are used explicitly; store
schema resolution is explicit and does not fall back to a current pointer.
Current v3 helpers remain separately available. Never use one ownership version
to backfill the other product.

`transport_cell` uses the actual canonical ownership and ordered signed
cross-projection correction over the full halo. Counterfactuals share validity
masks and fixed kernels. Integer source labels are not blurred or reprojected as
flux. `FrozenFieldOperator` reuses the actual intra/inter-skycell hybrid assignment
and sparse L5 binning, including the zero-shift rim patches and float32 narrowing.

`field_validation.validate_recipient` checks a recipient against its frozen
contribution, with all contributing input ledgers. `reduce_field` requires every
recipient and checks the reconstructed full template. A numerical pass is marked
`numerically_validated_pending_visual_review`. Existing template artifacts and
physical cross-projection inconsistencies are not changed by accounting.

## Commands

Activate the `syndiff` environment before any Python. From the isolated worktree:

```bash
mamba activate syndiff
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m syndiff_pipeline.template_creation.processing.removal_ledger.pilot \
  --out /astro/armin/koji/syndiff/dev_runs/ps1_ledger_fix_20261002 --inventory
python -m syndiff_pipeline.template_creation.processing.removal_ledger.pilot \
  --out /astro/armin/koji/syndiff/dev_runs/ps1_ledger_fix_20261002 \
  --field C4 --cell skycell.2528.005
```

`--download-raw` is needed only when no reusable original-input cache is present.
Results have version-specific pointers at `cell_versions/<cell>/<combined_fp>/`;
`pilot/<cell>/result.json` is only the latest human-facing view.

The campaign generator creates owned Condor DAG tasks from the immutable inventory.
It does not submit jobs. Jobs check their pinned checkout, preserve atomic outputs,
have a repeated-failure circuit breaker and a 150-GiB storage floor. Source tasks
request two CPUs/8 GiB (streamed pilot peak 3.85 GiB); recipient validators request
two CPUs/16 GiB. Default concurrency is eight source jobs and four validators.
Submitted campaign scripts cannot be repointed to a different code version.

No job control, pipeline merge, template replacement or new removal physics is
performed by this package. Those remain separate operational/review decisions.
