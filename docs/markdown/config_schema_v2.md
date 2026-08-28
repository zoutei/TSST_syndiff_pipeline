# Diff config schema v2 (single authored file)

An SCC used to need two authored files: `pipeline.yaml` (template policy) plus
a standalone `diff_config.yaml` (diff policy), wired together by a
`diff_config:` (or `diff_site_config:`) pointer key in `pipeline.yaml`. As of
this change, `pipeline.yaml` carries the diff recipe **inline** under a `diff:`
block, marked with a top-level `config_schema_version: 2`. `config/pipeline.yaml`
is the reference example.

`deployment.yaml` (machine paths + credentials) stays a separate file either
way. `photometry_config.yaml` and `star_config.yaml` remain their own files
too — this migration is diff-only.

The standalone-file form (schema v1: `diff_config:` pointing at a bare
`diff_config.yaml`) is **no longer supported**. A `diff_config:`,
`diff_site_config:` or `diff_config_path:` key in an authored `pipeline.yaml`
now raises `ValueError` at load, naming the file and pointing here. Fold the
pointed-at file's content under `diff:` instead.

`site_config.load_diff_site_policy()` still exists as a parser for a standalone
diff-policy *file*, and `config/archive/` is deliberately kept in v1 form as a
provenance record — but nothing wires such a file into a run any more.

`config_schema_version: 2` is an **informational marker only** — no loader
reads or validates it. Which form is in effect is decided purely by whether
`pipeline.yaml` has a `diff:` key; the version number does not gate anything.

## Old → new key map

Everything that used to be a top-level key in the standalone `diff_config.yaml`
now sits one level deeper, under `diff:` in `pipeline.yaml`:

| v1 (`diff_config.yaml` top-level) | v2 (`pipeline.yaml`, under `diff:`) | Notes |
|---|---|---|
| `deployment_file` | *(not settable here)* | The site's single top-level `deployment_file` always governs; a `diff.deployment_file` key is parsed but silently ignored by `parse_unified_diff_policy`. |
| `defaults` | `diff.defaults` | Unchanged shape (`n_jobs`, `pipeline_plots`, `workspace_run_id`, …). `crop_mode` / `crop_box_size` are gone in both forms — see below. |
| `paths` | `diff.paths` | Unchanged shape (`output_store_name`, `template_store_name`, `remap_store_name`, `template_dir`, `gaia_catalog`, …). |
| `pipeline` | `diff.pipeline` | Unchanged stage-list shape, with one new restriction: a v2 `diff.pipeline` may not contain an event-scoped stage kind (`astrometry`, `forced_photometry`, or the `photometry` delegator) — `parse_unified_diff_policy` raises `ValueError`. Diff is SCC-scoped; those kinds belong in `photometry_config.yaml`. v1 standalone files are not checked and may still declare them. |
| `overrides` | `diff.overrides` | Unchanged (`"sector/camera/ccd"` keys). |
| `condor` | `diff.condor` | Unchanged flat-vs-nested-per-stage rule (see CLAUDE.md invariant 13). |
| `additional_forced_targets`, `per_event_force_targets` | *(rejected)* | Already dead on the diff side in v1 (`resolve_diff_config()` zeroes/never reads them — photometry has its own copies in `photometry_config.yaml`), but v1 silently accepted and ignored them. v2 raises `ValueError` if either key is present, so a config author doesn't believe they still do something. |
| *(none)* | `diff.mask_settings` | New in v2: carries former `mask_settings.yaml` content verbatim when authored inline. **Not yet consumed** — `resolve_mask_settings()` still only reads sibling `mask_settings.yaml` files (stage path → `{lane}/mask_settings.yaml` → site `mask_settings.yaml` → packaged defaults). Author mask policy in `mask_settings.yaml` for now. |
| *(implicit: same dir as `diff_config.yaml`)* | `diff.source_dir` | Absolute authoring directory that relative `diff.paths` entries resolve against. Recorded automatically; a reloaded frozen `runs/{run_id}/config.yaml` carries it forward as the *original* authoring directory, not wherever the frozen file now lives. |

## Breaking CLI notes

- `syndiff diff run --target-name` (foreground debug) and `diff submit`'s
  `--config` flag previously accepted either a `--site` directory (which
  resolved `{site}/diff_config.yaml` directly) or a `--config` path pointed
  straight at a `diff_config.yaml`. **Both now always resolve a
  `pipeline.yaml`-style file** — `--site DIR` means `{DIR}/pipeline.yaml`, and
  `--config PATH` must be that same unified file, never a bare
  `diff_config.yaml`. That file must carry an embedded `diff:` block; a legacy
  pointer key is rejected.
- `crop_mode` / `crop_box_size` are removed from `SynDiffConfig` entirely (both
  v1 and v2). They had been a silent no-op since commit `041e996`; the SCC
  diff crop is always `mapping_grid.science_ffi_bounds()` (or, on the legacy
  event-dir handoff branch, `wcs_grouping.load_crop_bounds()`), never
  overridable per diff config. This is unrelated to `stages.wcs_grouping.crop_mode`
  in the *template* side of `pipeline.yaml`, which still exists and still
  drives `cluster_template_job.json`'s crop.

## The three storage tiers

The name `diff_config.yaml` used to mean up to four different things
depending on context. After this change there are three concrete tiers, and
only two files still carry that name:

| Tier | Location | What it is |
|---|---|---|
| Authored | `pipeline.yaml`'s `diff:` block (or a legacy standalone `diff_config.yaml`) | Site policy as written by an operator. Not read at stage-execution time. |
| Frozen run config | `runs/{run_id}/config.yaml` | The **sole runtime authority**. The whole `RunnerConfig` — template stages, resources, scheduler, notifications, and the entire `diff:` block frozen **verbatim** — written once at submit. Every stage, retry, verify tick, and Condor submit reads this file; the supervisor re-reads it every tick (no caching) and every Condor worker re-reads it at process start. Hand-editing it retunes a live run — see edit safety below. |
| Frozen per-lane record | `{data_root}/s{S}/c{C}/k{K}/diff_{lane}/diff_config.yaml` (`chmod 444`) | The immutable record of what policy actually built that lane's FITS, written once by `write_immutable_workspace_config_snapshot()` on first `shared_mask`/etc. run and never rewritten (a fingerprint mismatch on a later run raises `WorkspaceConfigMismatchError` rather than overwriting). This is **not an input** — nothing reads it back into a running pipeline. Its fingerprint sibling is `{lane}/diff_config.fingerprint`. |

A fourth location exists but is write-only debug output, not a config tier:
`runs/{run_id}/per_target/{label}/diff_config.yaml` — a per-target dump of the
frozen config, written by each stage for operator inspection, never read back.

The old `events/{event}/…/ws[_{workspace_run_id}]/` tree that used to hold a
per-workspace `diff_config.yaml` snapshot is gone — diff is SCC-scoped, not
event-scoped (see below), so the lock lives at the lane root next to
`mask_settings.yaml`, not under an event workspace.

### Diff is SCC-scoped; only photometry is per-event

`cfg.output_dir` is the SCC diff lane —
`{data_root}/s{S}/c{C}/k{K}/diff_{store}/` — not an event workspace. The
workspace config lock (`diff_config.yaml` + `.fingerprint` above) lives there,
beside the already-existing `mask_settings.yaml`. Star output moved with it,
to `{lane}/host_star/`. The event workspace tier
(`{workspace_root}/events/{event}/s{S}_c{C}_k{K}/`) now holds only
per-event photometry, under `phot_{run_id}/`.

## The three edit-safety tiers of the frozen run config

Because `runs/{run_id}/config.yaml` is re-read live, hand-editing it is the
supported way to retune a live run — but not every key is safe to touch, and
the pipeline enforces that in three different ways:

| Tier | Keys | What happens if you edit it mid-run |
|---|---|---|
| **Safe anytime** | `diff.condor.*`, `scheduler.*`, `resources.*`, `notifications.*` | Takes effect on the next tick / next Condor submit. None of these participate in the workspace config lock fingerprint. |
| **Stops you rather than diverging** | `diff.pipeline`, `pipeline_plots`, `workspace_run_id` | `workspace_lock.diff_config_fingerprint()` hashes these (plus `sector`/`camera`/`ccd`); `assert_workspace_config_lock()` compares the incoming fingerprint against the one frozen in `{lane}/diff_config.fingerprint` and raises `WorkspaceConfigMismatchError` on mismatch rather than silently building against a different recipe. `epsf`, `centroids`, `per_ffi_wcs`, and `temporal_wcs` stage kinds are exempt from the pipeline hash specifically so you can append them without bumping `workspace_run_id`. |
| **Silent, no guard** | `diff.defaults.*` (besides `pipeline_plots`/`workspace_run_id`), `diff.paths.*`, `diff.overrides.*`, `diff.mask_settings`, and the four lock-exempt stage kinds above | Not fingerprinted at all. An edit here changes behavior on the next run of that stage with no mismatch error — it is on you to know whether the change is safe for artifacts already on disk. |

Freezing buys isolation between the authored file and a running run — editing
the site `pipeline.yaml` mid-run cannot affect it. It does **not** buy
immutability *within* a run, and cannot, because the frozen file is
deliberately editable: **to retune a live run (most commonly, to bump a
Condor `request_memory` after an OOM), edit `runs/{run_id}/config.yaml`
directly.** That is the intended workflow, not a workaround.

## Provenance

Recipes now record `git_sha` (`common/provenance/publish.py:git_sha()` — the
full 40-char SHA of the running checkout at recipe-upsert time, cached
per-process, never raises; stored but not hashed — it does not participate in
`recipe_id`). Diff artifact emit sites additionally stamp `meta["run_id"]`
(from `SynDiffConfig.run_id`, empty for foreground/ad-hoc runs) into every
FITS/NPZ they publish. Together an artifact traces to both the run and the
exact code revision that produced it. `shared_mask` recipes now hash the
*resolved* `MaskSettings` policy rather than always falling back to packaged
defaults (previously `SharedMaskParams.mask_settings` — a path, `None` in
every live config — meant the recipe never reflected the lane's real mask
policy even though it embedded a full `mask_settings` block).

See [bookkeeping.md](bookkeeping.md) for the full recipe/fingerprint model.
