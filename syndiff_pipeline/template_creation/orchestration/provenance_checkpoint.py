"""``scc_assembly`` provenance checkpoint for ``ps1_process`` (PR2).

See ``doc/template_bookkeeping_plan.md`` §11 "Killing the scans -- exact
call-site changes" (template side). Until the real per-skycell provenance
graph lands (Phase 1/2, §12-13), ``ps1_process``'s known-slow completeness
scan (``verify.verify_ps1_process`` -> ``expected_ps1_process_skycells`` ->
``_count_convolved_data_arrays``, one ``os.scandir`` per expected skycell) is
stood in for by one coarse checkpoint artifact per SCC -- kind
``scc_assembly`` -- whose *recipe* is exactly the param set
``verify.config_fingerprint`` already enumerates for this stage
(``model.scc_assembly_recipe_params`` / ``model.ps1_process_recipe_params``:
``projections_limit``, ``psf_sigma``, ``enable_saturation_correction``,
``remove_saturated_stars``, ``bright_star_mag_threshold``) and whose
*location* is the existing, unchanged per-SCC convolved Zarr path
(``scc_paths.scc_convolved_zarr``) -- no bytes move, only identity is
recorded.

:func:`expected_scc_assembly_fingerprint` is a pure function of *resolved*
(no filesystem access), so this emitter and the future PR3 scheduler
pre-check always compute the identical fingerprint for the same config --
the whole point of the Merkle design (plan §5/§9): a config change is
detected exactly like today (a miss), without any filesystem access. Until
Phase 2 lands there is no tracked input graph for this checkpoint (§6 note:
"Until Phase 2 lands, scc_assembly is recorded as a whole-stage checkpoint
node over today's per-SCC convolved.zarr"), so it fingerprints with an empty
input-fingerprint list; it will re-fingerprint automatically once real
``convolved_skycell``/``mapping`` input edges are wired in a later PR.

:func:`emit_scc_assembly_checkpoint` is the PR2 write side: called from
``run_stage.py`` after a successful ``ps1_process`` run, dual-write alongside
the existing manifest (manifests remain authoritative during this window).
It is best-effort and **never raises** -- any failure (including the
provenance package being mid-authoring / absent / broken) is logged and
swallowed, since the legacy manifest/scan path is still the pipeline's real
completion signal at this phase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        ResolvedTargetConfig,
    )

log = logging.getLogger(__name__)

SCC_ASSEMBLY_KIND = "scc_assembly"

__all__ = [
    "SCC_ASSEMBLY_KIND",
    "expected_scc_assembly_fingerprint",
    "emit_scc_assembly_checkpoint",
]


def _scc_assembly_spatial_key_dict(resolved: "ResolvedTargetConfig") -> Dict[str, Any]:
    """Build the ``scc_assembly`` spatial key dict via ``provenance.model.SccKey``.

    ``os`` here is the *mapping* stage's oversampling factor -- the same value
    used to key ``mapping``/``downsample`` and to build ``mapping_root`` in
    ``runner_config.resolve_config`` -- since ``scc_assembly`` sits between
    those two stages in the DAG (plan §5).
    """
    from syndiff_pipeline.common.provenance.model import SccKey

    t = resolved.target
    oversampling = int(resolved.stages.mapping.oversampling_factor)
    key = SccKey(s=int(t.sector), c=int(t.camera), k=int(t.ccd), os=oversampling)
    return key.to_dict()


def expected_scc_assembly_fingerprint(resolved: "ResolvedTargetConfig") -> str:
    """Deterministic ``scc_assembly`` fingerprint for one SCC's ``ps1_process`` output.

    Pure function of *resolved*: identical config in always yields an
    identical fingerprint out, with no filesystem access. Unlike
    :func:`emit_scc_assembly_checkpoint`, this may raise if the provenance
    package itself is absent/broken -- callers that need a never-raises
    guarantee should go through the emitter instead.
    """
    from syndiff_pipeline.common.provenance.fingerprint import (
        RECIPE_SCHEMA_VERSION,
        fingerprint as merkle_fingerprint,
        recipe_id as compute_recipe_id,
    )
    from syndiff_pipeline.common.provenance.model import scc_assembly_recipe_params

    params = scc_assembly_recipe_params(resolved)
    rid = compute_recipe_id(SCC_ASSEMBLY_KIND, params, RECIPE_SCHEMA_VERSION)
    spatial_key = _scc_assembly_spatial_key_dict(resolved)
    # Phase-1 checkpoint (plan §6 note): no tracked convolved_skycell/mapping
    # input edges yet, so identity is recipe + spatial_key only.
    return merkle_fingerprint(SCC_ASSEMBLY_KIND, spatial_key, rid, [])


def _scc_assembly_location(resolved: "ResolvedTargetConfig") -> str:
    """Existing per-SCC convolved Zarr path -- the checkpoint's location.

    No bytes move for this checkpoint (plan §11): the location simply records
    where ``ps1_process`` already wrote its output today.
    """
    from syndiff_pipeline.common.scc_paths import scc_convolved_zarr

    t = resolved.target
    return str(scc_convolved_zarr(resolved.data_root, t.sector, t.camera, t.ccd))


def emit_scc_assembly_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    """Best-effort ``scc_assembly`` checkpoint sidecar emit (PR2 dual-write).

    Call this after a successful ``ps1_process`` stage run (see
    ``run_stage.py``). Never raises: any failure -- provenance package
    absent/mid-authoring, spool write error, unexpected config shape, ... --
    is logged and swallowed. The existing manifest write remains the
    pipeline's real completion signal until a later PR trusts the checkpoint.
    """
    try:
        _emit_scc_assembly_checkpoint(resolved)
    except Exception:
        log.exception(
            "scc_assembly checkpoint emit failed (non-fatal; manifest still authoritative)"
        )


def _emit_scc_assembly_checkpoint(resolved: "ResolvedTargetConfig") -> None:
    """Build and publish the checkpoint record. May raise; callers must guard."""
    from syndiff_pipeline.common.provenance.fingerprint import (
        RECIPE_SCHEMA_VERSION,
        recipe_id as compute_recipe_id,
    )
    from syndiff_pipeline.common.provenance.model import scc_assembly_recipe_params
    from syndiff_pipeline.common.provenance.publish import (
        append_spool_record,
        build_record,
    )
    from syndiff_pipeline.common.scc_paths import provenance_spool_dir

    params = scc_assembly_recipe_params(resolved)
    rid = compute_recipe_id(SCC_ASSEMBLY_KIND, params, RECIPE_SCHEMA_VERSION)
    spatial_key = _scc_assembly_spatial_key_dict(resolved)
    fp = expected_scc_assembly_fingerprint(resolved)
    location = _scc_assembly_location(resolved)

    # Bytes already live at ``location`` (the existing, unchanged convolved
    # Zarr path) -- this is the "record-only" publish path (plan §10):
    # build_record() + append_spool_record() records the location and
    # appends one spool line without going through publish_dir()/
    # publish_record()'s atomic-rename step, since there is nothing to
    # rename -- no bytes move.
    record = build_record(
        fp,
        SCC_ASSEMBLY_KIND,
        spatial_key,
        rid,
        RECIPE_SCHEMA_VERSION,
        [],
        location,
        recipe_params=params,
        state="complete",
    )
    append_spool_record(provenance_spool_dir(resolved.data_root), record)
