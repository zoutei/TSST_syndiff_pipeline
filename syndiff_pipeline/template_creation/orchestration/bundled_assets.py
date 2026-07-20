"""Bundled static assets for the template pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

# syndiff_pipeline/template_creation/orchestration/bundled_assets.py -> package root
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger(__name__)

TESS_ORBIT_TIMES_URL = "https://tess.mit.edu/public/files/TESS_orbit_times.csv"


def skycell_wcs_csv() -> Path:
    """PS1 SkyCells WCS table shipped with the repository."""
    path = _PACKAGE_ROOT / "resources" / "skycell_wcs.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing bundled resource: {path}. "
            "Ensure syndiff_pipeline/resources/skycell_wcs.csv is present."
        )
    return path


def bright_star_catalog_path() -> Path:
    """Decompressed BSC5 fixed-width catalog shipped with the repository."""
    path = _PACKAGE_ROOT / "resources" / "bsc5" / "catalog"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing bundled resource: {path}. "
            "Ensure syndiff_pipeline/resources/bsc5/catalog is present."
        )
    return path


def tess_straps_csv() -> Path:
    """TESS detector strap column list shipped with the repository."""
    path = _PACKAGE_ROOT / "resources" / "tess_straps.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing bundled resource: {path}. "
            "Ensure syndiff_pipeline/resources/tess_straps.csv is present."
        )
    return path


def gaia_alerts_csv() -> Path:
    """Gaia Photometric Science Alerts index shipped with the repository."""
    path = _PACKAGE_ROOT / "resources" / "gaia_alerts.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing bundled resource: {path}. "
            "Ensure syndiff_pipeline/resources/gaia_alerts.csv is present."
        )
    return path


def tess_orbit_times_csv() -> Path:
    """Local path for the MIT TESS orbit-times CSV (may not exist until ensured)."""
    return _PACKAGE_ROOT / "resources" / "tess_orbit_times.csv"


def ensure_tess_orbit_times_csv(*, force: bool = False) -> Path:
    """
    Return the local MIT TESS orbit-times CSV, downloading it if missing.

    Source: ``https://tess.mit.edu/public/files/TESS_orbit_times.csv``
    (same URL used by TESSreduce ``update_tess_sectors.ipynb``; ``skipfooter=1``
    applies when reading, not when downloading).
    """
    path = tess_orbit_times_csv()
    if path.is_file() and not force:
        return path

    import requests

    log.info("Downloading TESS orbit times from %s -> %s", TESS_ORBIT_TIMES_URL, path)
    response = requests.get(TESS_ORBIT_TIMES_URL, timeout=60)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    log.info("Wrote %s (%d bytes)", path, len(response.content))
    return path
