"""
Content-addressed provenance graph for SynDiff template + diff artifacts.

See ``doc/template_bookkeeping_plan.md`` for the design. This package is the
PR1 "provenance core": fingerprinting, the artifact-kind registry, the
SQLite-backed store, publish/ingest sidecar plumbing, offline reindex, and the
``syndiff bookkeeping`` CLI. Nothing in this package is wired into the
scheduler/verify hot path yet (that is PR2/PR3); importing it has zero effect
on existing compute paths.
"""

from __future__ import annotations

from syndiff_pipeline.common.provenance.fingerprint import (
    RECIPE_SCHEMA_VERSION,
    canonical,
    fingerprint,
    recipe_id,
)

__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "canonical",
    "fingerprint",
    "recipe_id",
]
