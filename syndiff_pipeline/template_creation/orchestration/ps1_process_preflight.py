"""Pre-launch delta/small-job policy for ``ps1_process``.

Decides, *before* any Condor submission, whether a target SCC's
``ps1_process`` stage can be skipped outright (every skycell its mapping
requires is already canonical in the shared combined/convolved store),
launched as a small/cheap job (only a handful of skycells actually need
(re)convolving), or needs the full configured resource profile.

This does not duplicate the per-cell canonical/sparse machinery already in
``ps1_process.py`` (see ``doc/ps1_process_tiered_ingest_architecture_plan.md``)
-- it reuses the exact same recipe-aware check
(``convolved_store.classify_projection_missing_cells``) so the scheduler's
decision and the worker's own runtime classification can never disagree.
Everything here is read-only and side-effect-free; it never writes to the
store or submits anything itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from syndiff_pipeline.common.orchestration import condor

DECISION_SKIP = "skip"
DECISION_SMALL = "small"
DECISION_FULL = "full"


@dataclass(frozen=True)
class Ps1ProcessLaunchPlan:
    """Result of :func:`plan_ps1_process_launch`."""

    decision: str  # DECISION_SKIP | DECISION_SMALL | DECISION_FULL
    oversampling_factor: int
    target_cells: frozenset[str]
    os1_cells: frozenset[str]
    target_only_cells: frozenset[str]
    os1_only_cells: frozenset[str]
    missing_cells: frozenset[str]
    reason: str
    resources: Optional[condor.CondorResourceRequest] = None


def _small_job_resources(params: Any, missing_count: int) -> condor.CondorResourceRequest:
    memory_mb = max(
        int(getattr(params, "small_job_min_memory_mb")),
        int(getattr(params, "small_job_memory_per_skycell_mb")) * max(1, missing_count),
    )
    cpus = max(1, min(int(getattr(params, "small_job_request_cpus")), missing_count))
    return condor.CondorResourceRequest(
        request_cpus=cpus,
        request_memory_mb=memory_mb,
        host_stats_min_mem_mb=int(getattr(params, "small_job_host_stats_min_mem_mb")),
        host_stats_max_load15=float(getattr(params, "host_stats_max_load15")),
    )


def plan_ps1_process_launch(
    *,
    data_root: str,
    sector: int,
    camera: int,
    ccd: int,
    oversampling_factor: int,
    params: Any,
    mapping_store_name: Optional[str] = None,
) -> Ps1ProcessLaunchPlan:
    """Side-effect-free preflight decision for one SCC's ``ps1_process`` launch.

    ``params`` is the resolved ``stages.ps1_process`` config for this target
    (a ``Ps1ProcessStageParams``-shaped object). ``mapping_store_name`` is the
    target's ``stages.mapping.store_name`` (e.g. ``"tvwcs"``) -- the OS-target
    mapping master-skycells CSV lives under that named lane
    (``mapping_{store_name}/``) whenever one is configured; the OS1
    comparison (diagnostics only, not load-bearing for the decision) always
    resolves against the default/legacy ``mapping/`` lane, since an OS1
    inventory predating named lanes is the common case this compares against.
    Fails closed to :data:`DECISION_FULL` whenever the shared-store reuse
    path can't be established (mapping CSVs unreadable, shared store
    disabled, recipe resolution fails, or the per-cell canonical check
    errors) -- it never silently under-requests resources or claims a skip
    it can't support.
    """
    from syndiff_pipeline.common.scc_paths import scc_mapping_master_skycells_csv
    from syndiff_pipeline.template_creation.processing.ps1_process import (
        expected_convolved_skycells,
    )

    oversampling_factor = int(oversampling_factor)

    try:
        target_mapping_csv = scc_mapping_master_skycells_csv(
            data_root, sector, camera, ccd,
            oversampling_factor=oversampling_factor,
            store_name=mapping_store_name,
        )
        target_cells = frozenset(
            expected_convolved_skycells(
                data_root, sector, camera, ccd,
                oversampling_factor=oversampling_factor,
                mapping_csv_path=str(target_mapping_csv),
            )
        )
    except Exception as exc:
        return Ps1ProcessLaunchPlan(
            decision=DECISION_FULL,
            oversampling_factor=oversampling_factor,
            target_cells=frozenset(),
            os1_cells=frozenset(),
            target_only_cells=frozenset(),
            os1_only_cells=frozenset(),
            missing_cells=frozenset(),
            reason=f"could not resolve OS{oversampling_factor} mapping inventory: {exc}",
        )

    if oversampling_factor == 1:
        os1_cells = target_cells
    else:
        try:
            os1_cells = frozenset(
                expected_convolved_skycells(data_root, sector, camera, ccd, oversampling_factor=1)
            )
        except Exception:
            # OS1 inventory is only used for diagnostics here (the actual
            # skip/small decision is driven by the canonical-store check
            # below, which does not depend on OS1 at all); an unreadable OS1
            # CSV must not block an otherwise-valid OS-target run.
            os1_cells = frozenset()

    target_only = target_cells - os1_cells
    os1_only = os1_cells - target_cells

    if not bool(getattr(params, "use_shared_convolved_store", False)):
        return Ps1ProcessLaunchPlan(
            decision=DECISION_FULL,
            oversampling_factor=oversampling_factor,
            target_cells=target_cells,
            os1_cells=os1_cells,
            target_only_cells=target_only,
            os1_only_cells=os1_only,
            missing_cells=frozenset(),
            reason="use_shared_convolved_store is disabled; full profile required",
        )

    try:
        from syndiff_pipeline.template_creation.processing.combined_store import (
            production_combined_recipe,
        )
        from syndiff_pipeline.template_creation.processing.convolved_store import (
            _projection_and_cell,
            classify_projection_missing_cells,
            convolved_recipe as build_convolved_recipe,
        )

        combined_recipe = production_combined_recipe(
            params, data_root=data_root, sector=sector, camera=camera, ccd=ccd
        )
        convolved_recipe_dict = build_convolved_recipe(params)

        by_projection: dict[str, list[str]] = {}
        unparsed: list[str] = []
        for cell_name in target_cells:
            parsed = _projection_and_cell(cell_name)
            if parsed is None:
                unparsed.append(cell_name)
                continue
            projection, _cell = parsed
            by_projection.setdefault(projection, []).append(cell_name)

        missing_cells: set[str] = set(unparsed)
        for projection, cell_names in by_projection.items():
            missing_cells |= classify_projection_missing_cells(
                data_root, projection, cell_names, combined_recipe, convolved_recipe_dict
            )
    except Exception as exc:
        return Ps1ProcessLaunchPlan(
            decision=DECISION_FULL,
            oversampling_factor=oversampling_factor,
            target_cells=target_cells,
            os1_cells=os1_cells,
            target_only_cells=target_only,
            os1_only_cells=os1_only,
            missing_cells=frozenset(),
            reason=f"canonical-store classification failed, falling back to full profile: {exc}",
        )

    missing_frozen = frozenset(missing_cells)
    if not missing_frozen:
        return Ps1ProcessLaunchPlan(
            decision=DECISION_SKIP,
            oversampling_factor=oversampling_factor,
            target_cells=target_cells,
            os1_cells=os1_cells,
            target_only_cells=target_only,
            os1_only_cells=os1_only,
            missing_cells=missing_frozen,
            reason=f"all {len(target_cells)} OS{oversampling_factor} skycells already canonical",
        )

    small_max = int(getattr(params, "small_job_max_skycells", 0))
    if small_max > 0 and len(missing_frozen) <= small_max:
        return Ps1ProcessLaunchPlan(
            decision=DECISION_SMALL,
            oversampling_factor=oversampling_factor,
            target_cells=target_cells,
            os1_cells=os1_cells,
            target_only_cells=target_only,
            os1_only_cells=os1_only,
            missing_cells=missing_frozen,
            reason=f"{len(missing_frozen)} skycell(s) missing (<= {small_max}); small profile",
            resources=_small_job_resources(params, len(missing_frozen)),
        )

    return Ps1ProcessLaunchPlan(
        decision=DECISION_FULL,
        oversampling_factor=oversampling_factor,
        target_cells=target_cells,
        os1_cells=os1_cells,
        target_only_cells=target_only,
        os1_only_cells=os1_only,
        missing_cells=missing_frozen,
        reason=f"{len(missing_frozen)} skycell(s) missing (> {small_max}); full profile",
    )
