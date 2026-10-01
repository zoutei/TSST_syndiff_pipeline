# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Convert photutils pooled ePSF stamps to forward_epsf_wcs node grids."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.forward_model import epsf_model as EM


def photutils_density_to_flux_fraction(stamp: np.ndarray, *, oversample: int) -> np.ndarray:
    """photutils ePSF density (sum = oversample^2) -> flux fraction (sum ~ 1)."""
    stamp = np.asarray(stamp, dtype=np.float64)
    total = float(oversample) ** 2
    out = stamp / total
    s = float(np.sum(out))
    if s <= 0 or not np.isfinite(s):
        raise ValueError(f"invalid photutils stamp sum={s}")
    return out / s


def resample_oversampled_stamp_to_node(
    stamp_frac: np.ndarray,
    *,
    oversample_src: int = 4,
    stamp_physical: int = EM.STAMP_PHYSICAL,
) -> np.ndarray:
    """Area-overlap resample a centered oversampled stamp to the forward node grid."""
    stamp_frac = np.asarray(stamp_frac, dtype=np.float64)
    if stamp_frac.ndim != 2 or stamp_frac.shape[0] != stamp_frac.shape[1]:
        raise ValueError(f"expected square 2D stamp, got {stamp_frac.shape}")
    n_src = stamp_frac.shape[0]
    center_src = (n_src - 1) / 2.0
    _, node, center_dst = EM.node_geometry(stamp_physical)
    spacing_src = 1.0 / float(oversample_src)
    spacing_dst = 1.0 / float(EM.OVERSAMPLE)
    R = EM.area_overlap_matrix(
        n_src,
        spacing_src,
        center_src,
        node,
        spacing_dst,
        center_dst,
    )
    out = R @ stamp_frac @ R.T
    s = float(np.sum(out))
    if s <= 0 or not np.isfinite(s):
        raise ValueError(f"resampled node grid has invalid sum={s}")
    return (out / s).astype(np.float32)


def gridded_stack_to_epsf_base(
    stack: np.ndarray,
    *,
    tile_ny: int,
    tile_nx: int,
    oversample: int = 4,
    stamp_physical: int = EM.STAMP_PHYSICAL,
) -> np.ndarray:
    """Flatten photutils tile stack (n_tiles, ny, nx) -> (tile_ny, tile_nx, G, G)."""
    stack = np.asarray(stack, dtype=np.float64)
    n_tiles = int(tile_ny) * int(tile_nx)
    if stack.shape[0] != n_tiles:
        raise ValueError(f"stack has {stack.shape[0]} tiles, expected {n_tiles}")

    base = np.zeros((tile_ny, tile_nx, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE), dtype=np.float32)
    for idx in range(n_tiles):
        i, j = divmod(idx, tile_nx)
        frac = photutils_density_to_flux_fraction(stack[idx], oversample=oversample)
        node = resample_oversampled_stamp_to_node(
            frac,
            oversample_src=oversample,
            stamp_physical=stamp_physical,
        )
        node = np.asarray(
            EM.recenter_grid_core(
                np.asarray(node, dtype=np.float32),
                clip_nonneg=True,
            ),
            dtype=np.float32,
        )
        base[i, j] = node
    return base


def save_epsf_base_init(path, base: np.ndarray) -> None:
    """Write npz for ``--init-epsf-base`` (key ``epsf_base``)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, epsf_base=np.asarray(base, dtype=np.float32))
