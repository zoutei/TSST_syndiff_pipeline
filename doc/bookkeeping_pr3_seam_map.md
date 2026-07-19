# PR3 seam map — replacing the verify scan with `scc_stage_complete`

Reference for the scheduler cutover (plan §11/§14). All anchors are `file:line`
in `syndiff_pipeline/`. Produced by a read-only investigation; verify lines before
editing (code drifts).

## Key correction
`verify_worker.py` is **not** a subprocess — it is an in-process
`ThreadPoolExecutor` (`ArtifactVerifyWorker`, `common/orchestration/verify_worker.py:145-343`).
The NFS walk runs on daemon threads inside the supervisor. There is no process
boundary to delete; you delete the thread-pool scheduling in `scheduler.py` and the
`stage_complete` call the pool runs.

## Hot-path decision flow (`common/orchestration/scheduler.py`)
`_resolve_external_and_pending_skips` (`:1232`) / `_schedule_external_and_pending_skips`
(`:1254`) → `_run_verify_pass` (`:1103-1229`):
- **Branch A — manifest fast path** (main thread): `check_manifests_only(...)`
  `scheduler.py:1141-1148` (+ stable recheck `:1151-1157`, `BackfillTask` `:1158-1163`).
- **Branch B — absence probe**: `stage_absence_probe(...)` `:1172-1177`; `ABSENT` → not complete, no scan.
- **Branch C — the NFS scan**: `VerifyTask` enqueue `:1193-1202` → pool runs
  `_run_verify_task` (`verify_worker.py:88-142`) → **`stage_complete`** (`verify_worker.py:99-106`),
  which is manifest-first (`verify.py:1369-1376`) then falls back to `verify_stage`
  (`verify.py:1377`) → per-stage `VERIFY_FUNCS` scanner.
- Outcomes applied by `_apply_verify_outcome` (`scheduler.py:890-919`): on complete →
  `mark_skipped` + `cache_skip_reason(SKIP_REASON_ARTIFACTS)` + `cache_external_check(complete=True)`
  (`:904-915`); incomplete → `cache_external_check(complete=False)` (`:916-918`).

**Cutover:** replace branches A **and** C for template stages with one
`store.scc_stage_complete(required_fps)` query.

## Required-fingerprint-set source per template stage
- `tess_ffi_download` → `expected_ffi_basenames` + `list_local_ffis` (`verify.py:1448-1455`).
- `mapping` → `_mapping_csv_path` (single artifact, `verify.py:1418-1421`).
- `ps1_download` → `_expected_ps1_download_skycells` (`verify.py:506`).
- `ps1_process`(→`scc_assembly`) → **`expected_ps1_process_skycells`** (`verify.py:881-915`);
  today counted by `_count_convolved_data_arrays` (`verify.py:918-932`, the `scandir`/cell).
- `templates`/`downsample` → `expected_downsample_fits_paths` (`verify.py:1061`).

`stage_absence_probe` (`verify.py:1220-1310`) enumerates each stage's output root
(mapping `:1240`, tess `:1248`, ps1_download `:1258`, ps1_process `:1266`, templates `:1274`).

## Call-site inventory (role)
- `check_manifests_only` (def `verify.py:281`): ONLY `scheduler.py:1141,1151` — hot path; both die for template stages.
- `stage_complete` (def `verify.py:1351`): `verify_worker.py:99` (hot); `cli.py:1202`, `stages.py:139` (`StageDefinition.verify_complete`).
- `verify_stage` (def `verify.py:1324`): `verify.py:1377,1456,1516`, `cli.py:1133`.
- `collect_stage_artifacts` (def `verify.py:1383`): `verify.py:347,1474`, `run_stage.py:272` (post-exec), `cli.py:1214`, `stages.py:151`. Expected/produced branches: ps1_process `:1411-1417`, mapping `:1418-1421`, ps1_download `:1440-1447`, tess `:1448-1455` — move behind `reindex`.
- `write_manifest` (def `verify.py:182`): `verify.py:350,1479`, `run_stage.py:309` (post-exec), `cli.py:1217`.
- `write_stable_manifest` (def `verify.py:332`): `verify_worker.py:109` (scan-time) — delete with scan.
- `persist_completion_manifests` (def `verify.py:1461`): `cli.py:1144`.

## state.py cache methods
- `external_verify_complete` (`state.py:1882`): `scheduler.py:856,996,1361`, `state.py:589`, `run_report.py:425` (UI).
- `external_verify_attempted` (`state.py:1902`): `scheduler.py:994`, **`state.py:698` inside `promote_stages`**.
- `cache_external_check` (`state.py:1926`): `scheduler.py:909,916`.
- `is_artifact_verified`/`cache_artifact_verified` (`state.py:577,591`): dead-ish shims, no external callers.

## Post-execution writers to delete (sidecar-at-publish replaces)
- `run_stage.py:272` (`collect_stage_artifacts`) + `run_stage.py:308-317` (`write_manifest`).
- `verify_worker.py:109` (`write_stable_manifest`) — with the scan.
- Keep `cli.py cmd_reconcile_manifests` only behind `reindex`.

## RISKS (must handle in PR3)
- **HIGHEST: promotion stall.** `promote_stages` (`state.py:689-707`) promotes PENDING→READY
  only when `external_verify_attempted` is true (`:698`). If the cutover stops writing
  `external_check` rows for template stages, they never promote. **PR3 must still record the
  `scc_stage_complete` result into the run-state `external_check` cache** (rewire
  `scheduler.py:909-918`) — or special-case promotion for template stages.
- **UI/status** reads `external_check` rows: `run_report.py:422-435`, `state.py:1458-1487`
  (`RunDisplayContext`), `state.py:1982-1993` (`list_unchecked_external_stages`). Keep them fed.
- **Retry/reset** deletes `artifacts` rows: `state.py:824,1152-1153,1292-1293,1820` — audit so a
  retry still forces re-derivation.
- **Diff/star are NOT template SCC stages** — keep them on the manifest/`external_check` path;
  do not route through `scc_stage_complete`. Boundary markers: `VERIFY_FUNCS` (`verify.py:1313-1321`),
  `stage=='diff'` special-cases (`verify.py:1346,1391,1286-1308`), `run_stage.py:263,266,269,289,299`.
- **Manifest-reading tests**: `tests/test_verify_worker.py:114-118`, `tests/test_verify_diff_cli.py:177`.
- **Dead config after pool removal**: `RunnerConfig.verify_max_workers` / `verify_budget_per_tick`
  and the `verify_worker` singleton lifecycle (`verify_worker.py:349-401`).
