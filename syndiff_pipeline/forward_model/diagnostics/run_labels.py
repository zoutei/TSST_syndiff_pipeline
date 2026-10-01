# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Run display labels for diagnostic plot titles."""

from __future__ import annotations

from pathlib import Path


def resolve_run_id(run_dir: Path, run_id: str | None = None) -> str:
    """Plot title label: parent folder when ``run_dir`` is a Colab ``artifacts/`` dir."""
    if run_id:
        return str(run_id)
    run_dir = Path(run_dir).resolve()
    if run_dir.name == "artifacts":
        return run_dir.parent.name
    return run_dir.name
