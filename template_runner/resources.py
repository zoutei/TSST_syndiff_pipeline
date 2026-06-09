"""Bundled static assets for the template pipeline."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def skycell_wcs_csv() -> Path:
    """PS1 SkyCells WCS table shipped with the repository."""
    path = _PACKAGE_ROOT / "resources" / "skycell_wcs.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing bundled resource: {path}. "
            "Ensure resources/skycell_wcs.csv is present in the repo checkout."
        )
    return path
