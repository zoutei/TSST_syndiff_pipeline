# C1 recovery and F1 preparation

User authorization: recover C1 and prepare the isolated F1 pilot; background and source-removal decisions remain with the user. No new ePSF fit or final campaign should start from these tools.

This is an independent local clone on branch `codex/paper1-recovery-20261002`, sharing read-only objects with the parent repository. It avoids changing live source or shared worktree metadata. Base: `3609d46`; v3 row-overlap fix: cherry-pick of `1edfce9`; per-band implementation and its tests: selected files from `d128192`. The unvalidated starmodel removal code is deliberately absent; `footprint_v1` is unchanged.

Validation: 84 tests passed, 4 slow tests deselected, 1 expected cross-projection failure. Tested canonical-cell geometry, convolved store, per-band canonical sums, bootstrap, scene and per-band chain units. Run `python -m pytest` with `PYTHONPATH` explicitly set to this checkout; the environment's editable install points to the live main checkout. Initial test collection using the console entry point found the live package and failed; no result from that attempt is used.

Read-only C1 inventory on October 2: exact recipe `e17a198a4942aa2d`, convolution recipe `f4d8a7b322fb8cb5`. OS4: 944/944 payloads present with correct neighbour fingerprints. OS1: 936/942; missing variants are 2057.030, 2057.031, 2266.022, 2266.023, 2266.024, 2266.034. File presence and fingerprint checks are not a full pixel-level certification. The saved log's failed-final-task cell, 2130.004, resolves to a complete v3 payload.

## Preparation

Activate `syndiff`, set `PYTHONPATH` to this checkout, and commit owned changes first. `python tools/paper1_recovery/prepare.py` is read-only. Add `--apply` to create a fresh `/astro/armin/koji/syndiff/dev_runs/paper1_recovery_20261002` directory, stage configs and two submit files. It refuses an existing output directory and never submits jobs. Register a central INDEX entry when preparation is applied.

## C1 execution boundary

First inspect `condor_q --global -submitter kshukawa` and record the owning scheduler and GlobalJobId. Do not assume scheduler identity from our current host or a cluster number. The existing template_v3_C1 job is an ad-hoc Condor job, not a daemon-supervised submission. Quiesce only its verified job before resuming; preserve all outputs and logs. The new entry point independently refuses to run if the old batch is still visible globally or the global query is incomplete/fails.

The prepared C1 job rechecks OS4 completeness, runs the missing OS1 pass with native recipe-aware reuse, then native downsample and F4 template construction. It uses the existing v3 data root for cached cells/native template output, but writes new stage records and the F4 template under the recovery run. Check the removed-star CSV before/after the OS1 pass and preserve the full OS4 record if the smaller mapping replaces it with an incomplete subset. This aggregate bookkeeping check remains a manual acceptance step, not an implemented repair. Inspect expected FITS, manifest and image checks before accepting recovery.

## F1 probe boundary

The F1 job runs three deterministic cells (first, middle, last in the sorted authoritative cell list), sequentially in one process, with 2 CPUs and a 32 GB memory ceiling. This is an initial measured-memory probe, not a guarantee those resources suffice or a final concurrency setting. It records elapsed time and process peak RSS, rejects per-cell errors/NaN-pattern mismatch, and requires band-sum closure within the existing 1e-5 peak-relative threshold.

It reuses the OLD F1 fitted mapping and band cells read-only, but uses v3 convolution and fresh contribution outputs. This deliberately separates resource/operator diagnosis from the final scientific recipe. Nothing from F1_probe should become a publication product. The full F1 pipeline configuration is completed only after the user's preparation choices are settled.

## Execution status

The global queue query and elevated command execution are blocked by repeated automatic approval-service timeouts despite explicit user approval. Source integration, syntax checks, read-only inventory and the tests above completed inside the sandbox. Neither recovery nor memory-probe job has been submitted. No old job or launcher was killed or released. No merge into the live branch was performed.

Conclusion: source and staged preparation tools are ready for execution review; C1 needs six OS1 variants rather than a full OS4 rebuild. Recovery is incomplete until scheduler access, execution and final product checks succeed.
