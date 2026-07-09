"""
Simple convolution utilities for TESS PSF application.
"""

import logging

import dask.array as da
import numpy as np
from dask_image.ndfilters import gaussian_filter as dask_gaussian_filter

logger = logging.getLogger(__name__)


def apply_gaussian_convolution(
    image: np.ndarray,
    sigma: float = 60.0,
    radius: int = 470,
    *,
    cval: float = np.nan,
) -> np.ndarray:
    """Apply Gaussian convolution to simulate TESS PSF.

    Args:
        image: Input image array
        sigma: Gaussian sigma parameter
        radius: Kernel truncation radius in pixels (truncate = radius / sigma)
        cval: Constant fill value for ``mode="constant"`` boundary padding.
            Production PS1 mosaics use the default ``np.nan`` for masked gaps;
            isolated star cutouts should pass ``0.0`` so tight cutouts do not
            lose flux to NaN contamination at the edges.

    Returns:
        Convolved image array
    """
    truncate = radius / sigma
    dimage = da.from_array(image, chunks=(1024, 1024))
    # with ProgressBar():
    convolved = dask_gaussian_filter(
        dimage,
        sigma=sigma,
        mode="constant",
        cval=cval,
        truncate=truncate,
    ).compute()
    logger.debug(f"Applied Gaussian convolution (sigma={sigma}): {image.shape}")
    return convolved
