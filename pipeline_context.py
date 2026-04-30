"""
Invocation context for a config-driven pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import SynDiffConfig
from .paths import resolve_manifest_path, workspace_dir


@dataclass
class PipelineInvocationContext:
    """Holds resolved paths for one ``run_config_pipeline`` execution."""

    cfg: SynDiffConfig
    manifest_path: str

    @classmethod
    def from_config(cls, cfg: SynDiffConfig) -> PipelineInvocationContext:
        mp = resolve_manifest_path(cfg.output_dir, cfg.manifest or None)
        return cls(cfg=cfg, manifest_path=mp)

    def workspace(self, label: str) -> str:
        return workspace_dir(self.cfg.output_dir, label)
