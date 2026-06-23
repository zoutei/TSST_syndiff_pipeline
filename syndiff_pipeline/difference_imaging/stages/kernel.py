"""Kernel reconstruction and template convolution via pyhotpants."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    _calculate_kernel_basis,
    _kernel_sigma_deg_for_basis,
)

log = logging.getLogger(__name__)

KERNEL_R2_NPZ_BASENAME = "kernel_r2.npz"
KERNEL_FIT_META_BASENAME = "kernel_fit_meta.json"
CONVOLVED_TEMPLATES_CSV_BASENAME = "convolved_templates.csv"


def build_kernel_basis(hp: HotpantsParams) -> np.ndarray:
    """Stacked basis images ``(n_basis, H, W)`` matching Hotpants geometry."""
    rkernel, sigma_gauss, deg_fixe, _ = _kernel_sigma_deg_for_basis(hp)
    size = 2 * rkernel + 1
    shape = (size, size)
    basis_list = _calculate_kernel_basis(shape, sigma_gauss, deg_fixe)
    return np.stack([np.asarray(b, dtype=np.float64) for b in basis_list], axis=0)


def convolve_template_with_kernel_solution(
    template: np.ndarray,
    kernel_solution: np.ndarray,
    hp_config,
) -> np.ndarray:
    """Convolve template with a full Hotpants ``kernel_solution`` vector."""
    from hotpants.convolve import KernelModel, convolve_template

    tmpl = np.asarray(template, dtype=np.float64)
    model = KernelModel(
        kernel_solution=np.asarray(kernel_solution, dtype=np.float64).ravel(),
        config=hp_config,
        fit_shape=tmpl.shape,
    )
    return np.asarray(convolve_template(tmpl, model), dtype=np.float64)


def kernel_from_hotpants_result(
    kernel_params_arrays: Optional[dict[str, np.ndarray]],
    hp_config,
    image_shape: Tuple[int, int],
    *,
    at_coords: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """Build kernel image from :func:`run_hotpants_frame` kernel_params_arrays."""
    if not kernel_params_arrays:
        return None
    ks = kernel_params_arrays.get("kernel_solution")
    if ks is None:
        return None
    return kernel_image_at_coords(
        ks, hp_config, image_shape, at_coords=at_coords
    )


def kernel_image_at_coords(
    kernel_solution: np.ndarray,
    hp_config,
    image_shape: Tuple[int, int],
    *,
    at_coords: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Approximate the spatial kernel at ``at_coords`` by convolving a unit impulse."""
    ny, nx = int(image_shape[0]), int(image_shape[1])
    if at_coords is None:
        cx, cy = nx // 2, ny // 2
    else:
        cx, cy = int(at_coords[0]), int(at_coords[1])
    delta = np.zeros((ny, nx), dtype=np.float64)
    delta[cy, cx] = 1.0
    convolved = convolve_template_with_kernel_solution(delta, kernel_solution, hp_config)
    r = int(getattr(hp_config, "rkernel", 2)) * 2 + 1
    half = r // 2
    y0 = max(0, cy - half)
    y1 = min(ny, cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(nx, cx + half + 1)
    patch = convolved[y0:y1, x0:x1]
    out = np.zeros((r, r), dtype=np.float64)
    oy = half - (cy - y0)
    ox = half - (cx - x0)
    out[oy : oy + patch.shape[0], ox : ox + patch.shape[1]] = patch
    return out


def kernel_arrays_to_npz_dict(
    kernel_image: np.ndarray,
    kernel_params_arrays: Optional[dict[str, np.ndarray]],
    basis: np.ndarray,
    hp: HotpantsParams,
) -> dict[str, Any]:
    """Bundle kernel artifacts for NPZ export."""
    out: dict[str, Any] = {
        "kernel_image": np.asarray(kernel_image, dtype=np.float64),
        "basis": np.asarray(basis, dtype=np.float64),
        "sci_fwhm": np.float64(hp.sci_fwhm),
    }
    if kernel_params_arrays:
        for key, val in kernel_params_arrays.items():
            out[key] = np.asarray(val)
    return out
