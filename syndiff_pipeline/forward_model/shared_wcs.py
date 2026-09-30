# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""One-off shared linear (no-SIP) WCS, fit once and frozen.

Per plan: a single ``RA---TAN`` WCS (CRVAL/CRPIX/CD only, ``sip_degree=0``)
supplies ``(alpha_0, delta_0)`` and the CD matrix for the Chebyshev model in
``cheb_wcs.py``. All per-frame boresight drift and all distortion are carried
by the Chebyshev/B-spline coefficients, never by refitting this WCS.
"""

from __future__ import annotations

import pandas as pd
from astropy.wcs import WCS

from . import _bootstrap  # noqa: F401
from .data import RegionSpec

from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import FrameRecord, fit_shared_linear_wcs, load_merged_stars, select_qc_stars  # noqa: E402
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.fit_wcs_from_centroids import StarSelectionConfig  # noqa: E402


def fit_region_shared_wcs(
    frame: FrameRecord,
    gaia_full: pd.DataFrame,
    region: RegionSpec,
    *,
    margin_px: float = 8.0,
    star_cfg: StarSelectionConfig | None = None,
) -> tuple[WCS, pd.DataFrame]:
    """Fit the frozen linear WCS from one reference frame's QC stars in ``region``.

    Returns the WCS and the QC+region-filtered star table it was fit from
    (useful for warm-start / residual diagnostics).
    """
    merged = load_merged_stars(frame, gaia_full)
    qc = select_qc_stars(merged, star_cfg)
    x_lo, x_hi = region.x_min - margin_px, region.x_max + margin_px
    y_lo, y_hi = region.y_min - margin_px, region.y_max + margin_px
    in_region = (
        (qc["x_fit"] >= x_lo) & (qc["x_fit"] < x_hi) & (qc["y_fit"] >= y_lo) & (qc["y_fit"] < y_hi)
    )
    region_qc = qc.loc[in_region].reset_index(drop=True)
    wcs = fit_shared_linear_wcs(region_qc, frame.crop_shape)
    return wcs, region_qc
