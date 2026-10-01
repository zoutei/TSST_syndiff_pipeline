# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Cross-fit (held-out fold) harness for single-FFI scene fits (Paper 1 gate G1, 2026-09-29).

build_folds -> make_init -> scene_fit on each fold scene -> score_oof -> merge_oof; xp_tess gives the flux truth.
Plan: docs/BRIGHT_WIDTH_CROSSFIT_PLAN_20260928.md §2 (A1-A4) + §6 V0. Run record:
/astro/armin/koji/syndiff/dev_runs/heldout_harness_20260929/README.md
"""
