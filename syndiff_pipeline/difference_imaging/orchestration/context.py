"""
Invocation context for a config-driven pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass

from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.support.paths import (
    normalize_workspace_run_id,
    resolve_manifest_path,
    workspace_artifact_path,
    workspace_dir,
    workspace_root,
)


@dataclass
class PipelineInvocationContext:
    """Holds resolved paths for one ``run_config_pipeline`` execution."""

    cfg: SynDiffConfig
    manifest_path: str

    @classmethod
    def from_config(cls, cfg: SynDiffConfig) -> PipelineInvocationContext:
        mp = resolve_manifest_path(cfg.output_dir, cfg.manifest or None)
        return cls(cfg=cfg, manifest_path=mp)

    @property
    def workspace_run_id(self) -> str | None:
        return normalize_workspace_run_id(getattr(self.cfg, "workspace_run_id", None))

    def workspace_root_path(self) -> str:
        return workspace_root(self.cfg.output_dir, run_id=self.workspace_run_id)

    def workspace(self, label: str) -> str:
        return workspace_dir(
            self.cfg.output_dir, label, run_id=self.workspace_run_id
        )

    def workspace_artifact(self, basename: str) -> str:
        return workspace_artifact_path(
            self.cfg.output_dir, basename, run_id=self.workspace_run_id
        )
