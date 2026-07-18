# SCC + nested-event layout migration

One-time migration from the legacy flat `data_root` / `workspace_root` layout to
the storage-first **SCC + nested-event** layout described in
[storage_layout.md](storage_layout.md). Implemented in
[`scripts/migrate_scc_event_layout.py`](../../scripts/migrate_scc_event_layout.py).
Not invoked by the pipeline supervisor — run it by hand, once, per site.

This is a **separate** migration from the older
[`scripts/migrate_workspace_layout.py`](../../scripts/migrate_workspace_layout.py)
(`template_handoffs` → `workspace` rename, `control/` creation). Run that one
first if your workspace predates `control/`; this script assumes a
`workspace_root` that already has `events/` and a `data_root` science cache.

---

## DATA SAFETY — read before running `--apply`

- The script **moves** files and directories (`shutil.move`); it does **not copy**.
- It **never deletes anything**, at any point, in any mode.
- Because it moves rather than copies, the legacy top-level directories
  (`tess_ffi/`, `skycell_pixel_mapping/`, `field_templates/`, `shifted_downsampled/`,
  `convolved_results/`, `catalogs/`) will end up **empty** (or contain only
  entries the script skipped) once a migration completes successfully — they
  are not deleted, just drained.
- **Operators must verify the new `data_root/scc/` tree is complete** (compare
  the printed inventory / JSON manifest counts against what existed before)
  **before manually removing** any old top-level directory. The script never
  does this cleanup for you.
- `--apply` is refused unless `--confirm I_ACCEPT_SCC_MIGRATION` is passed
  verbatim, and it prints a loud warning banner before executing any move.
- Paths containing the substring `field_pilot` are rejected outright
  (`_reject_field_pilot`) — this script must never touch that environment.
- It **never writes live `event_job.json` or `frames.csv`** — those are
  written by the `bind` stage, which must run **after** migration completes.
- It **never promotes** `field_templates/` or `shifted_downsampled/` content
  into a live `templates/` directory — those always land under
  `legacy/` (`_execute_move` raises `RuntimeError` if a phase-A `move_dir`
  would land inside a non-legacy `templates/` path).
- **Idempotent / resumable:** if a planned destination already exists, the
  move is skipped with a `[SKIP]` warning instead of overwriting or erroring
  out — safe to re-run `--apply` after a partial run or interruption.

---

## Safety contract (summary)

| Rule | Enforcement |
|------|-------------|
| Dry-run by default | `--dry-run` is the default; `--apply` is opt-in |
| Explicit confirmation | `--apply` requires `--confirm I_ACCEPT_SCC_MIGRATION` (exact token) |
| Never deletes | Every move is `shutil.move`; no `unlink`/`rmtree` on source data |
| Idempotent | `_skip_if_dst_exists` skips (does not overwrite) when destination exists |
| `field_pilot` guard | `_reject_field_pilot` refuses `--data-root` / `--workspace-root` containing that substring |
| No live handoff writes | `_execute_move` refuses to write `event_job.json` / `frames.csv` |
| No promotion into live templates | `_execute_move` refuses phase-A `move_dir` into non-`legacy` `templates/` |

---

## What it relocates

### Phase A — `data_root` (SCC-scoped science caches)

| Legacy path | New path | Notes |
|-------------|----------|-------|
| `tess_ffi/s{S}/cam{C}_ccd{K}/` | `scc/s{SSSS}_c{C}_k{K}/ffi/` | One move per SCC directory |
| `skycell_pixel_mapping/sector_{S}/camera_{C}/ccd_{K}/` | `scc/s{SSSS}_c{C}_k{K}/mapping/oversampling_1/` | Native (un-oversampled) mapping promoted to `oversampling_1` |
| `skycell_pixel_mapping/oversampling_{N}/sector_{S}/camera_{C}/ccd_{K}/` | `scc/s{SSSS}_c{C}_k{K}/mapping/oversampling_{N}/` | Oversampled mapping |
| `field_templates/sector_{S}_camera_{C}_ccd_{K}/[oversampling_{N}/]` | `scc/s{SSSS}_c{C}_k{K}/legacy/templates_legacy_pre_cutover/` | **Never** promoted to live `templates/`; archived once per SCC |
| `shifted_downsampled/sector{S}_camera{C}_ccd{K}_x.._y../` | `scc/s{SSSS}_c{C}_k{K}/legacy/templates_legacy_{event}/…` (attributed) or `.../legacy/templates_legacy_unattributed/…` (no `ws/templates` symlink to trace back to an event) | Attribution comes from resolving each event's `ws*/templates` symlink target |
| `convolved_results/sector_{S}_camera_{C}_ccd_{K}.zarr` | `scc/s{SSSS}_c{C}_k{K}/convolved.zarr` | `scc_convolved_zarr()` |
| `convolved_results/sector_{S}_camera_{C}_ccd_{K}_removed_stars.csv` | `scc/s{SSSS}_c{C}_k{K}/convolved_removed_stars.csv` | `scc_convolved_removed_stars_csv()` |
| `catalogs/sector_{S}/camera_{C}/ccd_{K}/` | `scc/s{SSSS}_c{C}_k{K}/catalogs/` | Whole leaf, including any `asteroids/` subdir |

### Phase B — `workspace_root` (nested event layout)

| Legacy path | New path | Notes |
|-------------|----------|-------|
| `events/s{SSSS}_c{C}_k{K}_{event_name}/` | `events/{event_name}/s{SSSS}_c{C}_k{K}/` | Nests the flat `Target.label()` directory under `{event_name}/{scc_label}/` |
| `events/{label}/cluster_template_job.json` | `data_root/scc/s{SSSS}_c{C}_k{K}/legacy/handoff_legacy_{event_name}/cluster_template_job.json` | Archived, not migrated into the new `event_job.json` — `bind` regenerates the live handoff |
| `events/{label}/syndiff_ffi_frames.csv` | `data_root/scc/s{SSSS}_c{C}_k{K}/legacy/handoff_legacy_{event_name}/syndiff_ffi_frames.csv` | Same archival treatment |
| `ws*/templates`, `ws*/field_templates` symlinks | unlinked (`unlink_symlink` action) | Stale legacy-data-path symlinks; rebuilt automatically post-cutover once templates resolve from `scc/.../templates/` |

Both phases are recorded in one `MigrationManifest` (`generated_utc`, `dry_run`,
`inventory`, `warnings`, `moves`), optionally written to disk via
`--manifest-out path.json`.

---

## Usage

```bash
mamba activate syndiff

# 1. Dry run — inventory + planned moves only, no filesystem changes
python scripts/migrate_scc_event_layout.py \
  --data-root /astro/.../syndiff/data \
  --workspace-root /astro/.../syndiff/workspace \
  --manifest-out /tmp/scc_migration_dryrun.json

# 2. Inspect /tmp/scc_migration_dryrun.json — confirm move counts and warnings

# 3. Apply (stop the supervisor first: syndiff daemon stop --site config)
python scripts/migrate_scc_event_layout.py \
  --data-root /astro/.../syndiff/data \
  --workspace-root /astro/.../syndiff/workspace \
  --apply --confirm I_ACCEPT_SCC_MIGRATION \
  --manifest-out /tmp/scc_migration_apply.json
```

`--dry-run` is the default even without the flag; pass `--apply` to execute.
Warnings (first 10 printed; full list in the manifest) commonly flag
`[SKIP]` (destination already exists — safe, idempotent) or unattributed
`shifted_downsampled/` directories (no traceable `ws/templates` symlink).

---

## Operator cutover order

1. **Stop the supervisor** (`syndiff daemon stop --site config`) and let any
   in-flight Condor jobs drain (`condor_q -submitter $(whoami)`).
2. **Dry-run** the migration; review the manifest and warning counts.
3. **`--apply --confirm I_ACCEPT_SCC_MIGRATION`**.
4. **Verify** the new `data_root/scc/` tree is complete (FFIs, mapping,
   convolved Zarr, catalogs per SCC) before touching any old top-level
   directory by hand.
5. **Deploy the current code** (this refactor: `bind`/`templates` stage
   names, `scc_paths.py` helpers).
6. **Rebuild / bind**: run `syndiff template submit` (or `run`) to confirm
   template stages verify complete against the new `scc/` paths, then
   `syndiff diff submit` so `bind` writes fresh `event_job.json` +
   `frames.csv` under the nested `events/{event}/s{SSSS}_c{C}_k{K}/` leaf.
7. **Smoke test** one SCC end-to-end (see
   [cluster_smoke_checklist.md](cluster_smoke_checklist.md)) before resuming
   full production submits.

---

## Related docs

| Document | Contents |
|----------|----------|
| [storage_layout.md](storage_layout.md) | Target SCC + nested-event layout this script migrates *to* |
| [template_pipeline.md](template_pipeline.md) | Template DAG, CLI, config |
| [cluster_smoke_checklist.md](cluster_smoke_checklist.md) | Post-cutover smoke test |
