"""
Invocation context for a config-driven pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass

from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.support.paths import resolve_manifest_path


@dataclass
class PipelineInvocationContext:
    """Holds resolved paths for one ``run_config_pipeline`` execution.

    ``cfg.output_dir`` is the SCC diff lane root directly -- there is no
    per-run ``ws/`` workspace tree under it (removed in wave A-3; see
    ``workspace_lock.py`` for where the config lock now lives).
    """

    cfg: SynDiffConfig
    manifest_path: str

    @classmethod
    def from_config(cls, cfg: SynDiffConfig) -> PipelineInvocationContext:
        """From config.

        Parameters
        ----------
        cfg : SynDiffConfig

        Returns
        -------
        PipelineInvocationContext"""
        mp = resolve_manifest_path(cfg.output_dir, cfg.manifest or None)
        return cls(cfg=cfg, manifest_path=mp)
