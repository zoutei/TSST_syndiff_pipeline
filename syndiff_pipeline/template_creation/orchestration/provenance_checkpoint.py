"""Provenance checkpoint emission for template pipeline stages (PR2/PR3 glue).

Scope (deliberately narrow): this wires the provenance graph
(``syndiff_pipeline.common.provenance``) into the *one* template stage whose
completeness scan is the pipeline's known slow path — ``ps1_process``, whose
``verify.verify_ps1_process`` does an ``os.scandir`` per expected skycell
(historically ~30 min on NFS; see ``doc/bookkeeping_pr3_seam_map.md``). Other
template stages (mapping/templates/downloads) follow the same pattern but are
intentionally deferred to keep this change small and reviewable; their scans
are already cheap by comparison.

Because the real per-skycell provenance graph (shared combined/convolved
stores, Phase 1/2) has not landed yet, ``ps1_process`` is represented here as
one coarse checkpoint artifact per SCC — kind ``scc_assembly`` — whose recipe
is the union of the params that today's ``verify.config_fingerprint`` already
enumerates for this stage (psf_sigma, saturation/removal knobs,
projections_limit). This checkpoint's *location* is the existing, unchanged
per-SCC convolved Zarr path (``scc_convolved_zarr``) — we are not moving any
bytes, only recording "this recipe produced what's at this path" so
completeness becomes an indexed lookup instead of a directory scan.

Both the emitter (called after a successful stage run, see ``run_stage.py``)
and the checker (called by the scheduler's verify pass, see ``scheduler.py``)
call :func:`ps1_process_checkpoint_record` so the fingerprint they compare is
guaranteed identical — the whole point of the Merkle design (plan §5, §9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from syndiff_pipeline.common.provenance import model
from syndiff_pipeline.common.provenance.publish import emit_sidecar
from syndiff_pipeline.common.scc_paths import scc_convolved_zarr

if TYPE_CHECKING:
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        ResolvedTargetConfig,
    )

PS1_PROCESS_STAGE = "ps1_process"


def _ps1_process_recipe_params(resolved: "ResolvedTargetConfig") -> Dict[str, Any]:
    """Union of the params that determine ``ps1_process`` output (pre-Phase-2).

    Mirrors ``verify.config_fingerprint``'s ``ps1_process`` branch exactly, plus
    the convolution params from :func:`model.convolved_skycell_params`, since the
    coarse per-SCC checkpoint stands in for both the (not-yet-shared) combined
    and convolved steps.
    """
    pp = resolved.stages.ps1_process
    params = model.combined_skycell_params(resolved)
    params.update(model.convolved_skycell_params(resolved))
    params["projections_limit"] = pp.projections_limit
    return params


def ps1_process_checkpoint_record(resolved: "ResolvedTargetConfig") -> Dict[str, Any]:
    """Build the (deterministic) provenance record for one SCC's ps1_process output.

    Pure function of ``resolved`` — no filesystem access, no randomness — so the
    scheduler can call this to compute the *expected* fingerprint for the
    *current* config and compare it against the store, independent of whether
    or when the stage last actually ran.
    """
    t = resolved.target
    recipe = model.Recipe(
        kind=model.SCC_ASSEMBLY,
        params=_ps1_process_recipe_params(resolved),
        code_version=_code_version(),
        git_sha=_git_sha(),
    )
    artifact = model.Artifact(
        kind=model.SCC_ASSEMBLY,
        spatial_key=model.scc_no_os_spatial_key(t.sector, t.camera, t.ccd),
        recipe=recipe,
        inputs=[],
        location=str(scc_convolved_zarr(resolved.data_root, t.sector, t.camera, t.ccd)),
        state="complete",
    )
    return artifact.to_record()


def _code_version() -> str:
    from syndiff_pipeline.common.provenance.fingerprint import code_version

    return code_version()


def _git_sha():
    from syndiff_pipeline.common.provenance.fingerprint import git_sha

    return git_sha()


def emit_ps1_process_checkpoint(
    resolved: "ResolvedTargetConfig", *, produced_by: str | None = None
) -> Dict[str, Any]:
    """Emit (sidecar-only, non-blocking) the checkpoint for a completed ps1_process run.

    Called after the stage's existing manifest write succeeds (dual-write, see
    ``run_stage.py``). Never raises: a failure here must not affect the
    pipeline's real completion, since the legacy manifest/scan path remains the
    fallback until the checkpoint is ingested and used.
    """
    record = ps1_process_checkpoint_record(resolved)
    if produced_by is not None:
        record["produced_by"] = produced_by
    emit_sidecar(record, resolved.data_root)
    return record


__all__ = [
    "PS1_PROCESS_STAGE",
    "emit_ps1_process_checkpoint",
    "ps1_process_checkpoint_record",
]
