"""Backward-compatible re-export of the shared orchestration CLI."""

from syndiff_pipeline.common.orchestration.cli import *  # noqa: F403
from syndiff_pipeline.common.orchestration.cli import main

__all__ = [name for name in globals() if not name.startswith("_")]
