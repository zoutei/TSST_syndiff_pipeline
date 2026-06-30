"""Pipeline-agnostic stage and DAG specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ManifestTuple = Optional[Tuple[int, int, List[str], Optional[dict]]]

# Upstream stages whose on-disk artifacts diff-only runs verify before launch.
DIFF_VERIFY_UPSTREAM = frozenset(
    {
        "tess_ffi_download",
        "wcs_grouping",
        "downsample",
    }
)


@dataclass
class StageRunContext:
    """Runtime inputs for a single stage execution or verification."""

    run_id: str
    runs_root: str
    target_label: str
    target: Any
    runner_cfg: Any
    template_resolved: Any | None = None
    meta: Dict[str, Any] = field(default_factory=dict)
    force_rerun: bool = False
    progress_path: str | None = None


@dataclass(frozen=True)
class StageSpec:
    name: str
    short_name: str
    deps: Tuple[str, ...]
    pool: Optional[str] = None
    default_executor: str = "local"
    effective_deps: Optional[Callable[[Any], Tuple[str, ...]]] = None
    execute: Callable[..., ManifestTuple] = None  # type: ignore[assignment]
    verify_complete: Callable[..., bool] = None  # type: ignore[assignment]
    collect_artifacts: Callable[..., Tuple[int, int, List[str]]] = None  # type: ignore[assignment]
    config_fingerprint: Callable[..., str] = None  # type: ignore[assignment]
    condor_resources: Optional[Callable[[Any], Any]] = None
    stage_snapshot: Optional[Callable[..., dict]] = None

    def deps_for(self, resolved: Any) -> Tuple[str, ...]:
        if self.effective_deps is not None and resolved is not None:
            return self.effective_deps(resolved)
        return self.deps

    def resolve_executor(self, cfg: Any) -> str:
        stages = getattr(cfg, "stages", None)
        if stages is not None:
            if self.name == "mapping":
                return stages.mapping.executor
            if self.name == "ps1_process":
                return stages.ps1_process.executor
            if self.name == "diff":
                return stages.diff.executor
        return self.default_executor


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    stages: Tuple[StageSpec, ...]

    @property
    def stage_names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.stages)

    def stage_short_names(self) -> Dict[str, str]:
        return {s.name: s.short_name for s in self.stages}

    def stage_deps(self) -> Dict[str, List[str]]:
        return {s.name: list(s.deps) for s in self.stages}

    def stage_pools(self) -> Dict[str, str]:
        return {s.name: s.pool for s in self.stages if s.pool}

    def get(self, name: str) -> StageSpec | None:
        for spec in self.stages:
            if spec.name == name:
                return spec
        return None

    def require(self, name: str) -> StageSpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"Unknown stage: {name!r}")
        return spec

    def resolve_stage_name(self, name: str) -> str:
        """Map a full or short stage name to the canonical internal name."""
        raw = str(name).strip()
        if not raw:
            raise ValueError("Stage name must not be empty")
        if raw in self.stage_names:
            return raw
        short_to_full = {spec.short_name: spec.name for spec in self.stages}
        if raw in short_to_full:
            return short_to_full[raw]
        full_names = ", ".join(self.stage_names)
        short_names = ", ".join(sorted(short_to_full))
        raise ValueError(
            f"Unknown stage {raw!r}; use a full name ({full_names}) "
            f"or short name ({short_names})"
        )

    def parse_stage_list(self, stages_arg: str | None) -> List[str]:
        if not stages_arg or not str(stages_arg).strip():
            return list(self.stage_names)
        names = [s.strip() for s in str(stages_arg).split(",") if s.strip()]
        return [self.resolve_stage_name(n) for n in names]

    def stages_in_pool(self, pool: str) -> List[str]:
        return [s.name for s in self.stages if s.pool == pool]

    def unpooled_stages(self) -> List[str]:
        return [s.name for s in self.stages if not s.pool]

    def upstream_stages_for(self, active_stages: Sequence[str]) -> frozenset[str]:
        required: set[str] = set()
        stack = list(active_stages)
        deps_map = self.stage_deps()
        while stack:
            stage = stack.pop()
            for dep in deps_map.get(stage, []):
                if dep not in required:
                    required.add(dep)
                    stack.append(dep)
        return frozenset(required)

    def effective_stage_deps(self, stage: str, stages=None) -> List[str]:
        spec = self.get(stage)
        if spec is None:
            return []
        if spec.effective_deps is not None and stages is not None:
            return list(spec.effective_deps(stages))
        return list(spec.deps)

    def downstream_stages(self, stage: str) -> List[str]:
        out: set[str] = set()
        deps_map = self.stage_deps()
        changed = True
        while changed:
            changed = False
            for s, deps in deps_map.items():
                if s in out:
                    continue
                if stage in deps or any(d in out for d in deps):
                    out.add(s)
                    changed = True
        return [s for s in self.stage_names if s in out]

    def run_stage_closure(self, active_stages: Sequence[str]) -> frozenset[str]:
        return frozenset(set(active_stages) | self.upstream_stages_for(active_stages))

    def artifact_verify_closure(self, active_stages: Sequence[str]) -> frozenset[str]:
        """Stages eligible for artifact verify in a partial run.

        Diff-only runs verify tess_dl, wcs handoff, and downsample — not the
        full template chain (mapping, ps1_download, ps1_process).
        """
        active = set(active_stages)
        if active == {"diff"}:
            return frozenset(active | DIFF_VERIFY_UPSTREAM)
        return self.run_stage_closure(active_stages)

    def direct_dependents(self, stage: str) -> List[str]:
        return [s.name for s in self.stages if stage in s.deps]
