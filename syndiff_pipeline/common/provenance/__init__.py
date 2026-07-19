"""Content-addressed provenance graph for template-creation bookkeeping.

See ``doc/template_bookkeeping_plan.md`` for the full design. A few
dependency-light convenience names are re-exported here; heavier submodules
(``store``, ``ingest``, ``reindex``) are imported directly by callers to keep
this package import cheap for scheduler/daemon code paths.

Note: the Merkle-hash *function* is deliberately NOT re-exported at package
level, because that would shadow the ``fingerprint`` *submodule* attribute
(``syndiff_pipeline.common.provenance.fingerprint``). Import it explicitly::

    from syndiff_pipeline.common.provenance.fingerprint import fingerprint
"""

from __future__ import annotations

from .fingerprint import (
    RECIPE_SCHEMA_VERSION,
    canonical,
    code_version,
    git_sha,
    recipe_id,
)

__all__: list[str] = [
    "RECIPE_SCHEMA_VERSION",
    "canonical",
    "code_version",
    "git_sha",
    "recipe_id",
]
