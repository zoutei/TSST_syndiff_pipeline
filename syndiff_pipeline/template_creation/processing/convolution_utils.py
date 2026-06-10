"""
Simple convolution utilities for TESS PSF application.
"""

import logging

import dask.array as da
import numpy as np
from dask_image.ndfilters import gaussian_filter as dask_gaussian_filter

logger = logging.getLogger(__name__)


def apply_gaussian_convolution(image: np.ndarray, sigma: float = 60.0, radius: int = 470) -> np.ndarray:
    """Apply Gaussian convolution to simulate TESS PSF.

    Args:
        image: Input image array
        sigma: Gaussian sigma parameter
        truncate: Truncate radius for Gaussian filter

    Returns:
        Convolved image array
    """
    truncate = radius / sigma
    dimage = da.from_array(image, chunks=(1024, 1024))
    # with ProgressBar():
    convolved = dask_gaussian_filter(dimage, sigma=sigma, mode="constant", cval=np.nan, truncate=truncate).compute()
    logger.debug(f"Applied Gaussian convolution (sigma={sigma}): {image.shape}")
    return convolved
