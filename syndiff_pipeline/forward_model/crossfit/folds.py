# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Spatially coherent fold assignment (A1).

The science region (2048 x 2048 px, science-local 0-based pixel centres) is cut into square tiles of ``tile`` px.
Tiles are grouped into super-tiles of ``group`` x ``group`` tiles; inside every super-tile the tiles are split as
evenly as possible between the folds by a seeded permutation. The fold of ANY source is the fold of the tile its
position falls in, so the assignment extends to catalogues beyond the scene (the G4 scorer's T 13-16 stars).

Why tiles and not random stars: a scene-mode fit uses 15 x 15 stamps, so a randomly held-out star usually sits
inside a trained star's stamp. With 128 px tiles only stars within ~7 px of a tile edge can share pixels with the
other fold; ``pix_disjoint`` flags the strict subset that shares none.
"""
from __future__ import annotations

import numpy as np

NPIX = 2048


def tile_map(seed: int, tile: int = 128, group: int = 2, n_folds: int = 2) -> np.ndarray:
    """(ny, nx) int8 array: fold of each tile."""
    n = -(-NPIX // tile)
    rng = np.random.default_rng(seed)
    m = np.full((n, n), -1, np.int8)
    for gy in range(0, n, group):
        for gx in range(0, n, group):
            cells = [(y, x) for y in range(gy, min(gy + group, n)) for x in range(gx, min(gx + group, n))]
            labels = np.arange(len(cells)) % n_folds
            rng.shuffle(labels)
            for (y, x), f in zip(cells, labels):
                m[y, x] = f
    assert (m >= 0).all()
    return m


def tile_map_diagonal(seed: int, tile: int = 128, n_folds: int = 5) -> np.ndarray:
    """(ny, nx) int8 tile folds: fold = (ix + 2 iy + offset) mod n_folds, offset from ``seed``.

    Every n_folds x n_folds block of tiles holds each fold n_folds times, so folds are spatially uniform and
    balanced to +-1 tile; with n_folds = 5 no two edge- or corner-adjacent tiles share a fold. Used for K = 5
    (D19, 2026-09-30): the group-shuffle layout (``tile_map``) splits only evenly when group**2 is a multiple of K.
    """
    n = -(-NPIX // tile)
    iy, ix = np.mgrid[0:n, 0:n]
    off = int(np.random.default_rng(seed).integers(n_folds))
    return ((ix + 2 * iy + off) % n_folds).astype(np.int8)


def fold_of(tmap: np.ndarray, tile: int, x, y) -> np.ndarray:
    """Fold of science-local positions (x, y); positions off the CCD are clipped to the nearest tile."""
    n = tmap.shape[0]
    ix = np.clip(np.floor(np.asarray(x, float) / tile).astype(int), 0, n - 1)
    iy = np.clip(np.floor(np.asarray(y, float) / tile).astype(int), 0, n - 1)
    return tmap[iy, ix].astype(np.int8)


def stamp_coverage(cx, cy, stamp: int) -> np.ndarray:
    """Boolean (2048, 2048) image: True where any of the given stamps (centre cx, cy; size stamp) has a pixel."""
    cov = np.zeros((NPIX, NPIX), bool)
    h = stamp // 2
    for x, y in zip(np.asarray(cx, int), np.asarray(cy, int)):
        cov[max(y - h, 0):max(min(y + h + 1, NPIX), 0), max(x - h, 0):max(min(x + h + 1, NPIX), 0)] = True
    return cov


def core_disjoint(cov: np.ndarray, x, y, r: float = 3.0) -> np.ndarray:
    """True if no pixel whose centre is within r px of (x, y) is covered by ``cov``."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    R = int(np.ceil(r))
    d = np.arange(-R, R + 1)
    DX, DY = np.meshgrid(d, d)
    out = np.ones(len(x), bool)
    for i in range(len(x)):
        px = np.round(x[i]).astype(int) + DX
        py = np.round(y[i]).astype(int) + DY
        ok = (np.hypot(px - x[i], py - y[i]) <= r) & (px >= 0) & (px < NPIX) & (py >= 0) & (py < NPIX)
        out[i] = not cov[py[ok], px[ok]].any()
    return out
