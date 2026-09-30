# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""CPU-only, spatially resolved stacks of frozen temporal ePSF residuals.

The diagnostic restricts itself to K=1 (single-source) packed groups: for a
one-source packed group the already solved flux makes the pixel model
``flux[t] * template(x[t], y[t], t)``, with no new flux fitting or training
performed.  Blended (K>=2) groups are always excluded -- their residual is
shared across multiple simultaneous flux solves, so attributing it to one
member's stamp would manufacture apparent morphology.

Within K=1, both the bright, trained ``is_epsf_contributor`` stars and the
fainter, catalog-isolated WCS-only anchors (stop-gradiented from ePSF
training) are stacked, tagged by ``population``.  The anchors' isolation
check at export time only screens against ``Tmag<=13`` sources, so a stamp
can still contain an uncatalogued-for-training, sub-threshold neighbour; each
selected group's stamp pixels are checked against the *full* reference
catalogue (``x_lin``/``y_lin``, every source in the export region, not just
packed members) and any pixel within ``neighbor_exclusion_radius_px`` of a
foreign source is dropped from that group's stack at every cadence.  This is
a static, catalogue-geometry mask, not a residual-value sigma clip: clipping
on the observed residual would risk censoring exactly the time-varying
defocus/astigmatism signal this diagnostic is trying to detect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .. import cheb_wcs as CW
from .. import fit as FIT
from . import stamp_counterfactuals as SC


DEFAULT_OVERSAMPLE = 4
DEFAULT_HALF_WIDTH = 4.0
DEFAULT_MIN_GROUPS = 4
VARIANCE_FLOOR = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def time_blocks(btjd: np.ndarray, strict: np.ndarray) -> dict[str, np.ndarray]:
    """Disjoint equal-duration early/middle/late partitions of clean cadences."""
    btjd = np.asarray(btjd, dtype=float)
    strict = np.asarray(strict, dtype=bool)
    clean = strict & np.isfinite(btjd)
    if clean.sum() < 3:
        raise ValueError("need at least three clean cadences for temporal stacks")
    lo, hi = float(np.min(btjd[clean])), float(np.max(btjd[clean]))
    edges = np.linspace(lo, hi, 4)
    names = ("early", "middle", "late")
    result = {}
    for index, name in enumerate(names):
        keep = clean & (btjd >= edges[index])
        keep &= btjd <= edges[index + 1] if index == 2 else btjd < edges[index + 1]
        result[name] = np.flatnonzero(keep)
    if not np.array_equal(
        np.sort(np.concatenate(list(result.values()))), np.flatnonzero(clean)
    ):
        raise RuntimeError("temporal blocks do not partition clean cadences")
    return result


def _bundle_group_metadata(bundle_path: Path, params: dict) -> pd.DataFrame:
    """Read small global metadata and fitted median positions without pixels."""
    with np.load(bundle_path, allow_pickle=False) as raw:
        members = np.asarray(raw["members"], dtype=int)
        valid = np.asarray(raw["valid"], dtype=bool)
        contributor = np.asarray(raw["is_epsf_contributor"], dtype=bool)
        x_lin = np.asarray(raw["x_lin"], dtype=float)
        y_lin = np.asarray(raw["y_lin"], dtype=float)
        basis = np.asarray(raw["cheb_basis"], dtype=float)
        frame_basis = np.asarray(raw["wcs_frame_basis"], dtype=float)
        n_terms = int(np.asarray(raw["cheb_exponents"]).shape[0])
        n_tiers = int(np.asarray(raw["n_packed_tiers"]))
        tier = np.full(len(members), -1, dtype=int)
        local_row = np.full(len(members), -1, dtype=int)
        support = np.full(len(members), np.nan, dtype=float)
        for ti in range(n_tiers):
            ids = np.asarray(raw[f"pt{ti}_group_idx"], dtype=int)
            tier[ids] = ti
            local_row[ids] = np.arange(len(ids))
            support[ids] = np.asarray(raw[f"pt{ti}_pix_valid"], dtype=bool).sum(axis=1)
    x_all, y_all = CW.eval_all_positions(
        x_lin, y_lin, basis, np.asarray(params["wcs_coeff"]), frame_basis, n_terms,
    )
    first = members[:, 0]
    safe_first = np.clip(first, 0, len(x_all) - 1)
    return pd.DataFrame({
        "group_index": np.arange(len(members), dtype=int),
        "slot_index": np.zeros(len(members), dtype=int),
        "star_row": safe_first,
        "group_size": valid.sum(axis=1),
        "is_epsf_contributor": contributor,
        "tier": tier,
        "local_row": local_row,
        "support_size": support,
        "x": np.nanmedian(x_all[safe_first], axis=1),
        "y": np.nanmedian(y_all[safe_first], axis=1),
    })


def _selected_positions(bundle_path: Path, params: dict, star_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fitted positions and packed static arrays for selected K=1 rows."""
    with np.load(bundle_path, allow_pickle=False) as raw:
        x_lin = np.asarray(raw["x_lin"], dtype=float)[star_rows]
        y_lin = np.asarray(raw["y_lin"], dtype=float)[star_rows]
        basis = np.asarray(raw["cheb_basis"], dtype=float)[star_rows]
        frame_basis = np.asarray(raw["wcs_frame_basis"], dtype=float)
        w_frame_basis = np.asarray(raw["w_frame_basis"], dtype=float)
        n_terms = int(np.asarray(raw["cheb_exponents"]).shape[0])
        node_x = np.asarray(raw["epsf_node_x"], dtype=float)
        node_y = np.asarray(raw["epsf_node_y"], dtype=float)
    x, y = CW.eval_all_positions(x_lin, y_lin, basis, np.asarray(params["wcs_coeff"]), frame_basis, n_terms)
    return x, y, w_frame_basis, node_x, node_y, basis


def _bilinear_sample_numpy(grid: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Sample grids ``(B,G,G)`` at ``(B,... )`` coordinates, OOB -> zero."""
    grid = np.asarray(grid, dtype=float)
    b, size, _ = grid.shape
    inside = (gx >= 0) & (gx <= size - 1) & (gy >= 0) & (gy <= size - 1)
    x0 = np.clip(np.floor(gx).astype(int), 0, size - 2)
    y0 = np.clip(np.floor(gy).astype(int), 0, size - 2)
    fx, fy = gx - x0, gy - y0
    bi = np.arange(b).reshape((b,) + (1,) * (gx.ndim - 1))
    value = (
        grid[bi, y0, x0] * (1 - fx) * (1 - fy)
        + grid[bi, y0, x0 + 1] * fx * (1 - fy)
        + grid[bi, y0 + 1, x0] * (1 - fx) * fy
        + grid[bi, y0 + 1, x0 + 1] * fx * fy
    )
    return np.where(inside, value, 0.0)


def _chroma_dilation_field_numpy(
    node_field: np.ndarray, *, radius_px: float = 6.0, oversample: int = 4
) -> np.ndarray:
    """Pure-NumPy twin of ``epsf_model.chroma_dilation_field`` for ``(..., G, G)``.

    Must stay pure NumPy: this module is used precisely because dispatching XLA
    on this bundle segfaults the S52 CPU path, and calling the JAX original from
    inside the render loop reintroduces exactly that crash.

    Index space, matching the original, so the two OVERSAMPLE factors cancel:
    ``D[P] = 2P + (j-c) dP/dj + (i-c) dP/di``. Then both gauges -- zero grid mean,
    and projection off the field in the fixed r <= 6 px disk metric.

    Sign: positive coefficient means a NARROWER PSF (see ``dilation_generator``).
    """
    P = np.asarray(node_field, dtype=float)
    size = P.shape[-1]
    centre = (size - 1) / 2.0
    ax = np.arange(size, dtype=float) - centre
    raw = (2.0 * P
           + ax * np.gradient(P, axis=-1)
           + ax[:, None] * np.gradient(P, axis=-2))
    gauged = raw - raw.mean(axis=(-2, -1), keepdims=True)
    ref = P - P.mean(axis=(-2, -1), keepdims=True)
    coord = (np.arange(size, dtype=float) - centre) / oversample
    disk = (np.sqrt(coord[None, :] ** 2 + coord[:, None] ** 2) <= radius_px).astype(float)
    den = (disk * ref ** 2).sum(axis=(-2, -1), keepdims=True) + 1e-12
    num = (gauged * disk * ref).sum(axis=(-2, -1), keepdims=True)
    return gauged - (num / den) * ref


def _chroma_aniso_field_numpy(
    node_field: np.ndarray, *, radius_px: float = 6.0, oversample: int = 4
) -> np.ndarray:
    """Pure-NumPy twin of ``epsf_model.chroma_aniso_field`` (C1). See
    ``_chroma_dilation_field_numpy`` for why this must stay pure NumPy.

    Generator ``x*Px - y*Py`` in index space (matches ``epsf_model.aniso_generator``),
    then the same two gauges as the dilation twin above.
    """
    P = np.asarray(node_field, dtype=float)
    size = P.shape[-1]
    centre = (size - 1) / 2.0
    ax = np.arange(size, dtype=float) - centre
    raw = ax * np.gradient(P, axis=-1) - ax[:, None] * np.gradient(P, axis=-2)
    gauged = raw - raw.mean(axis=(-2, -1), keepdims=True)
    ref = P - P.mean(axis=(-2, -1), keepdims=True)
    coord = (np.arange(size, dtype=float) - centre) / oversample
    disk = (np.sqrt(coord[None, :] ** 2 + coord[:, None] ** 2) <= radius_px).astype(float)
    den = (disk * ref ** 2).sum(axis=(-2, -1), keepdims=True) + 1e-12
    num = (gauged * disk * ref).sum(axis=(-2, -1), keepdims=True)
    return gauged - (num / den) * ref


def _chroma_shear_field_numpy(
    node_field: np.ndarray, *, radius_px: float = 6.0, oversample: int = 4
) -> np.ndarray:
    """Pure-NumPy twin of ``epsf_model.chroma_shear_field`` (C1).

    Generator ``x*Py + y*Px`` in index space (matches ``epsf_model.shear_generator``),
    then the same two gauges as the dilation twin above.
    """
    P = np.asarray(node_field, dtype=float)
    size = P.shape[-1]
    centre = (size - 1) / 2.0
    ax = np.arange(size, dtype=float) - centre
    raw = ax[:, None] * np.gradient(P, axis=-1) + ax * np.gradient(P, axis=-2)
    gauged = raw - raw.mean(axis=(-2, -1), keepdims=True)
    ref = P - P.mean(axis=(-2, -1), keepdims=True)
    coord = (np.arange(size, dtype=float) - centre) / oversample
    disk = (np.sqrt(coord[None, :] ** 2 + coord[:, None] ** 2) <= radius_px).astype(float)
    den = (disk * ref ** 2).sum(axis=(-2, -1), keepdims=True) + 1e-12
    num = (gauged * disk * ref).sum(axis=(-2, -1), keepdims=True)
    return gauged - (num / den) * ref


def _chroma_kurt_field_numpy(
    node_field: np.ndarray, *, radius_px: float = 6.0, oversample: int = 4
) -> np.ndarray:
    """Pure-NumPy twin of ``epsf_model.chroma_kurt_field`` (C1, optional leaf).

    Generator ``r^2*P - <r^2 P>/<P> * P`` with ``r`` in physical px (matches
    ``epsf_model.kurt_generator``), then the same two gauges as the other twins.
    """
    P = np.asarray(node_field, dtype=float)
    size = P.shape[-1]
    centre = (size - 1) / 2.0
    coord = (np.arange(size, dtype=float) - centre) / oversample
    r2 = coord[None, :] ** 2 + coord[:, None] ** 2
    mean_field = P.mean(axis=(-2, -1), keepdims=True)
    mean_r2field = (r2 * P).mean(axis=(-2, -1), keepdims=True)
    ratio = np.where(mean_field != 0, mean_r2field / (mean_field + 1e-30), 0.0)
    raw = r2 * P - ratio * P
    gauged = raw - raw.mean(axis=(-2, -1), keepdims=True)
    ref = P - P.mean(axis=(-2, -1), keepdims=True)
    disk = (np.sqrt(r2) <= radius_px).astype(float)
    den = (disk * ref ** 2).sum(axis=(-2, -1), keepdims=True) + 1e-12
    num = (gauged * disk * ref).sum(axis=(-2, -1), keepdims=True)
    return gauged - (num / den) * ref


# Name -> pure-NumPy gauged generator, keyed exactly like loss.CHROMA_FIELD_GENERATORS
# so the render loop below can iterate whichever chroma_* leaves are present in
# ``params`` without a name-by-name if/elif ladder.
_CHROMA_FIELD_GENERATORS_NUMPY = {
    "dilation": _chroma_dilation_field_numpy,
    "aniso": _chroma_aniso_field_numpy,
    "shear": _chroma_shear_field_numpy,
    "kurt": _chroma_kurt_field_numpy,
}


def _recenter_numpy(grid: np.ndarray) -> np.ndarray:
    """One production-compatible core-centroid shift plus flux renormalisation."""
    grid = np.asarray(grid, dtype=float)
    b, size, _ = grid.shape
    coordinate = (np.arange(size, dtype=float) - (size - 1) / 2) / 4.0
    yy, xx = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float), indexing="ij")
    core = np.exp(-0.5 * ((xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2) / 3.0**2)
    weighted = grid * core
    total = weighted.sum(axis=(1, 2)) + 1e-12
    cx = (weighted * coordinate[None, None, :]).sum(axis=(1, 2)) / total
    cy = (weighted * coordinate[None, :, None]).sum(axis=(1, 2)) / total
    # production recenter shifts grid content by (-cx, -cy): output(q) samples
    # input(q + cx), in oversampled-grid coordinates.
    iy, ix = np.mgrid[:size, :size]
    gx = ix[None] + cx[:, None, None] * 4.0
    gy = iy[None] + cy[:, None, None] * 4.0
    shifted = _bilinear_sample_numpy(grid, gx, gy)
    return shifted / (shifted.sum(axis=(1, 2), keepdims=True) + 1e-12)


def _decode_epsf_numpy(
    params: dict, *, weight_grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy equivalent of the frozen ePSF decode gauges (no XLA dispatch).

    ``weight_grid`` (task T2, additive, default None -> unchanged r<=6px
    canonical disk): the mode/base-orthogonality gauge (2) metric, matching
    ``epsf_model.decode_epsf_modes``'s own ``weight_grid`` kwarg -- pass
    ``epsf_model.footprint_weight_grid(...)`` here to decode exactly as a
    checkpoint whose ``epsf_modes`` leaf was built with that gauge (e.g.
    ``E1_submission_prep/params_warm_ALL.npz``, footprint-gauged K=2 modes).
    """
    raw = np.asarray(params["epsf_base_raw"], dtype=float)
    # Stable softplus; exact enough for the float32 checkpoint representation.
    pos = np.maximum(raw, 0) + np.log1p(np.exp(-np.abs(raw))) + 1e-8
    base = pos / (pos.sum(axis=(-2, -1), keepdims=True) + 1e-12)
    for _ in range(2):
        base = _recenter_numpy(base.reshape((-1, *base.shape[-2:]))).reshape(base.shape)
        base = np.clip(base, 0, None)
        base /= base.sum(axis=(-2, -1), keepdims=True) + 1e-12
    modes = np.asarray(params["epsf_modes"], dtype=float).copy()
    size = base.shape[-1]
    if weight_grid is None:
        coord = (np.arange(size, dtype=float) - (size - 1) / 2) / 4.0
        yy, xx = np.meshgrid(coord, coord, indexing="ij")
        weight = (np.sqrt(xx**2 + yy**2) <= 6.0).astype(float)
    else:
        # Either a single (G,G) grid shared by every node, or a per-node
        # (n_rows,n_cols,G,G) grid (e.g. E1's per-node footprint_weight_grid
        # shot_noise profile, which depends on that node's own base) -- the
        # gauge math below operates on the trailing two axes only (no
        # cross-node coupling), so both shapes broadcast correctly and give
        # results identical to gauging each node separately.
        weight = np.asarray(weight_grid, dtype=float)
        if weight.shape not in {(size, size), base.shape}:
            raise ValueError(
                f"weight_grid must be ({size},{size}) or {base.shape} (per-node), "
                f"got {weight.shape}"
            )
    base_meansub = base - base.mean(axis=(-2, -1), keepdims=True)
    denominator = (weight * base_meansub**2).sum(axis=(-2, -1), keepdims=True) + 1e-12
    iy, ix = np.mgrid[:size, :size]
    core = np.exp(-0.5 * ((ix - (size - 1) / 2) ** 2 + (iy - (size - 1) / 2) ** 2) / 3.0**2)
    basis_x = (ix - (size - 1) / 2) * core / core.sum()
    basis_y = (iy - (size - 1) / 2) * core / core.sum()
    var_x, var_y = (basis_x**2).sum() + 1e-12, (basis_y**2).sum() + 1e-12
    for _ in range(2):
        modes -= modes.mean(axis=(-2, -1), keepdims=True)
        numerator = (modes * weight * base_meansub[None]).sum(axis=(-2, -1), keepdims=True)
        modes -= numerator / denominator[None] * base_meansub[None]
        dipole_x = (modes * basis_x).sum(axis=(-2, -1), keepdims=True)
        dipole_y = (modes * basis_y).sum(axis=(-2, -1), keepdims=True)
        modes -= dipole_x / var_x * basis_x + dipole_y / var_y * basis_y
    return base, modes


def render_isolated_packed_numpy(
    params: dict,
    x: np.ndarray,
    y: np.ndarray,
    frame_basis: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    pix_valid: np.ndarray,
    *,
    frame_block: int = 16,
    chroma_delta: np.ndarray | None = None,
    chroma_shift_px: np.ndarray | None = None,
    weight_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-NumPy reference packed renderer for isolated K=1 group rows.

    ``weight_grid`` (task T2, additive, default None -> unchanged): forwarded
    to ``_decode_epsf_numpy`` as its mode/base gauge metric -- see that
    function's docstring.

    This is intentionally not the JAX packed hot path: the latter has a known
    S52 CPU XLA crash.  The calculation follows its reference block-sum
    physics and is validated against stored GPU-exported chi2 before use.

    Chromatic term (optional, and it must be supplied for any run fitted with
    ``--chroma`` or the measured residual is that of a *different* model than
    the one that was fitted). Mirrors ``loss._render_occ_templates_packed``:

    - ``chroma_delta`` is ``(n_group,)``, the star's ``bp_rp - c_ref``.
    - ``chroma_shift_px`` is ``(n_group, 2)`` in physical px, **already
      multiplied by delta** (as ``loss.chroma_slot_terms`` returns it), blended
      at the star's static ``x_lin``/``y_lin`` by the caller.
    - the dilation is read from ``params["chroma_dilation"]`` and folded in the
      same row-concat form the JAX path uses, so it is
      ``blend(P) + delta * D[sum_rc Wn_rc eps_rc P_rc]`` -- NOT
      ``eps_blend * D[blend(P)]``, which is a different quantity. The C1
      colour-affine leaves ``chroma_aniso``/``chroma_shear`` (and the optional
      ``chroma_kurt``) are folded in exactly the same way, one term per key
      found in ``params`` -- see ``_CHROMA_FIELD_GENERATORS_NUMPY``.

    Order matters: the perturbation is applied BEFORE ``_recenter_numpy``, so
    the core-centroid gauge sees the grid that is actually rendered. Applying it
    after would let the dilation (or aniso/shear/kurt) masquerade as a
    chromatic shift.
    """
    base, modes = _decode_epsf_numpy(params, weight_grid=weight_grid)
    _chroma = chroma_delta is not None
    if _chroma:
        if "chroma_dilation" not in params:
            raise ValueError("chroma_delta given but params carry no chroma_dilation")
        # Ordered like loss.chroma_slot_terms's field_terms dict: dilation always
        # first, then aniso/shear/kurt when present in the checkpoint.
        _field_coeffs = {"dilation": np.asarray(params["chroma_dilation"], dtype=float)}
        if "chroma_aniso" in params and "chroma_shear" in params:
            _field_coeffs["aniso"] = np.asarray(params["chroma_aniso"], dtype=float)
            _field_coeffs["shear"] = np.asarray(params["chroma_shear"], dtype=float)
            if "chroma_kurt" in params:
                _field_coeffs["kurt"] = np.asarray(params["chroma_kurt"], dtype=float)
        _delta = np.asarray(chroma_delta, dtype=float)
        _chsh = (np.zeros((x.shape[0], 2), dtype=float) if chroma_shift_px is None
                 else np.asarray(chroma_shift_px, dtype=float))
    n_group, n_frame = x.shape
    p = pix_x.shape[1]
    out = np.zeros((n_group, n_frame, p), dtype=np.float32)
    w = np.asarray(params["w_coeff"], dtype=float) @ np.asarray(frame_basis, dtype=float).T
    w = (w.T - np.mean(w.T, axis=0, keepdims=True))
    size = base.shape[-1]
    center = (size - 1) / 2.0
    sub = (np.arange(4, dtype=float) - 1.5) / 4.0
    # Loop order: frame-block OUTER, group INNER. ``field`` (the node-grid
    # composite base+w(t)*modes) and, when chroma is on, each chroma
    # generator field, depend ONLY on the frame block (via ``wb``), not on
    # which group is being rendered -- the original group-outer loop order
    # recomputed both from scratch once per GROUP per block (an
    # ``n_group``-fold redundant cost that scales with the node grid's
    # n_rows*n_cols and, for chroma, with the generator's own cost too).
    # Hoisting them here to compute once per block is a pure performance
    # change: every per-group quantity below (i0/j0/wy/wx from that group's
    # own xb/yb, the delta-weighted chroma blend, recentering, sampling) is
    # untouched and still computed once per (group, block) exactly as
    # before -- see ``test_render_isolated_packed_numpy_block_hoist_matches_naive_loop``
    # for a byte-for-byte regression check against the original group-outer
    # formula on a small case.
    for start in range(0, n_frame, max(1, int(frame_block))):
        stop = min(start + max(1, int(frame_block)), n_frame)
        wb = w[start:stop]
        nb = np.arange(stop - start)
        field = base[None] + np.einsum("bk,kijxy->bijxy", wb, modes, optimize=True)
        gfields = ({name: _CHROMA_FIELD_GENERATORS_NUMPY[name](field) for name in _field_coeffs}
                   if _chroma else {})

        for group in range(n_group):
            xb, yb = x[group, start:stop], y[group, start:stop]
            # This run has a 2x2 ePSF grid, but use the general bilinear form.
            j0 = np.clip(np.searchsorted(node_x, xb, side="right") - 1, 0, len(node_x) - 2)
            i0 = np.clip(np.searchsorted(node_y, yb, side="right") - 1, 0, len(node_y) - 2)
            wx = np.clip((xb - node_x[j0]) / (node_x[j0 + 1] - node_x[j0]), 0, 1)
            wy = np.clip((yb - node_y[i0]) / (node_y[i0 + 1] - node_y[i0]), 0, 1)

            def _blend(f, scale=None):
                """Bilinear node blend; ``scale`` is an optional per-node factor."""
                s00 = 1.0 if scale is None else scale[i0, j0][:, None, None]
                s01 = 1.0 if scale is None else scale[i0, j0 + 1][:, None, None]
                s10 = 1.0 if scale is None else scale[i0 + 1, j0][:, None, None]
                s11 = 1.0 if scale is None else scale[i0 + 1, j0 + 1][:, None, None]
                return (
                    (1 - wy)[:, None, None] * ((1 - wx)[:, None, None] * s00 * f[nb, i0, j0]
                                               + wx[:, None, None] * s01 * f[nb, i0, j0 + 1])
                    + wy[:, None, None] * ((1 - wx)[:, None, None] * s10 * f[nb, i0 + 1, j0]
                                           + wx[:, None, None] * s11 * f[nb, i0 + 1, j0 + 1])
                )

            local = _blend(field)
            gx_shift = gy_shift = 0.0
            if _chroma:
                # Row-concat equivalent: each generator is linear, so blending
                # coeff*Generator[P] then summing equals concatenating
                # [P ; Generator_1[P] ; ...] with weights [Wn ; Wn*coeff*delta ; ...],
                # which is what the fitted model does (see loss._render_occ_templates's
                # banded branch).
                for _name, _coeff in _field_coeffs.items():
                    local = local + _delta[group] * _blend(gfields[_name], scale=_coeff)
                gx_shift, gy_shift = float(_chsh[group, 0]), float(_chsh[group, 1])
            local = _recenter_numpy(local)
            ox = pix_x[group][None, :, None, None] - (xb + gx_shift)[:, None, None, None] + sub[None, None, None, :]
            oy = pix_y[group][None, :, None, None] - (yb + gy_shift)[:, None, None, None] + sub[None, None, :, None]
            gx, gy = center + 4 * ox, center + 4 * oy
            out[group, start:stop] = (_bilinear_sample_numpy(local, gx, gy).sum(axis=(2, 3)) * pix_valid[group][None]).astype(np.float32)
    return out


def select_stack_groups(
    metadata: pd.DataFrame,
    flux: np.ndarray,
    stamp_active: np.ndarray,
    strict: np.ndarray,
    *,
    min_coverage: float = 0.50,
) -> pd.DataFrame:
    """Choose isolated K=1 groups and assign stable 3x3 detector cells.

    Keeps both bright ``is_epsf_contributor`` stars and fainter WCS-only
    anchors (tagged via the ``population`` column); always excludes K>=2
    blended groups.
    """
    strict = np.asarray(strict, dtype=bool)
    active = np.asarray(stamp_active, dtype=bool) & strict[None, :]
    coverage = active.mean(axis=1)
    usable_flux = np.isfinite(flux) & (flux > 0) & active[:, :, None]
    usable = usable_flux[:, :, 0].mean(axis=1)
    work = metadata.copy()
    work["clean_coverage"] = coverage[work.group_index.to_numpy(int)]
    work["positive_flux_coverage"] = usable[work.group_index.to_numpy(int)]
    work = work[
        (work.group_size == 1)
        & (work.tier >= 0)
        & np.isfinite(work.x)
        & np.isfinite(work.y)
        & (work.clean_coverage >= min_coverage)
        & (work.positive_flux_coverage >= min_coverage)
    ].copy()
    if work.empty:
        raise ValueError("no isolated K=1 groups meet coverage requirements")
    work["population"] = np.where(work.is_epsf_contributor, "epsf_bright", "wcs_anchor")
    # Packed tiers have different numbers of stored pixels, but their pixel
    # coordinates share the same detector system.  They can therefore be
    # rendered separately and co-stacked after rasterisation.  Keeping all
    # tiers is important here: otherwise the compact 64-pixel tier alone
    # discards a large fraction of clean isolated contributors.
    # Use this selected population's detector extent, keeping the upper edge
    # in the final cell even when a source lies exactly at the maximum.
    x_edges = np.linspace(float(work.x.min()), float(work.x.max()), 4)
    y_edges = np.linspace(float(work.y.min()), float(work.y.max()), 4)
    x_cell = np.clip(np.searchsorted(x_edges, work.x, side="right") - 1, 0, 2)
    y_cell = np.clip(np.searchsorted(y_edges, work.y, side="right") - 1, 0, 2)
    work["cell_col"] = x_cell.astype(int)
    work["cell_row"] = y_cell.astype(int)
    work["cell"] = [f"r{r}c{c}" for r, c in zip(work.cell_row, work.cell_col)]
    return work.sort_values("group_index").reset_index(drop=True)


def rasterize_packed_values(
    values: np.ndarray,
    x_star: np.ndarray,
    y_star: np.ndarray,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    coverage: np.ndarray,
    *,
    oversample: int = DEFAULT_OVERSAMPLE,
    half_width: float = DEFAULT_HALF_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly splat one group's time×pixel values onto a centred grid."""
    values = np.asarray(values, dtype=float)
    x_star = np.asarray(x_star, dtype=float)
    y_star = np.asarray(y_star, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    n = int(round(2 * half_width * oversample)) + 1
    numerator = np.zeros((n, n), dtype=float)
    denominator = np.zeros((n, n), dtype=float)
    rel_x = np.asarray(pix_x, dtype=float)[None, :] - x_star[:, None]
    rel_y = np.asarray(pix_y, dtype=float)[None, :] - y_star[:, None]
    gx = (rel_x + half_width) * oversample
    gy = (rel_y + half_width) * oversample
    valid = np.isfinite(values) & np.isfinite(gx) & np.isfinite(gy) & np.isfinite(coverage) & (coverage > 0)
    valid &= (gx >= 0) & (gx <= n - 1) & (gy >= 0) & (gy <= n - 1)
    if not np.any(valid):
        return np.full((n, n), np.nan), denominator
    x = gx[valid]
    y = gy[valid]
    value = values[valid]
    weight = coverage[valid]
    ix0 = np.minimum(np.floor(x).astype(int), n - 2)
    iy0 = np.minimum(np.floor(y).astype(int), n - 2)
    fx = x - ix0
    fy = y - iy0
    for dx, dy, frac in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                         (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(numerator, (iy0 + dy, ix0 + dx), weight * frac * value)
        np.add.at(denominator, (iy0 + dy, ix0 + dx), weight * frac)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0), denominator


def robust_group_stack(group_maps: np.ndarray, *, min_groups: int = DEFAULT_MIN_GROUPS) -> tuple[np.ndarray, np.ndarray]:
    """MAD-clipped componentwise median over per-group maps."""
    maps = np.asarray(group_maps, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(maps, axis=0)
        mad = np.nanmedian(np.abs(maps - median[None]), axis=0)
    scale = 1.4826 * mad
    keep = np.isfinite(maps) & ((scale[None] == 0) | (np.abs(maps - median[None]) <= 5 * scale[None]))
    counts = keep.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stacked = np.nanmedian(np.where(keep, maps, np.nan), axis=0)
    stacked[counts < int(min_groups)] = np.nan
    return stacked, counts


DEFAULT_NEIGHBOR_EXCLUSION_PX = 3.0


def catalog_neighbor_pixel_mask(
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    star_row: int,
    catalog_x: np.ndarray,
    catalog_y: np.ndarray,
    *,
    exclusion_radius_px: float,
    search_margin_px: float,
) -> tuple[np.ndarray, int]:
    """Static per-group mask of stamp pixels sitting near a foreign catalog source.

    Uses the frozen linear reference positions, not per-cadence fitted
    positions: two catalog sources this close together share almost the same
    field distortion, so their relative offset is stable to well under a
    pixel across one orbit, and this avoids a per-cadence WCS evaluation of
    the whole catalog for every selected group.  The mask is identical at
    every cadence by construction, so it cannot selectively remove a
    time-varying residual -- unlike a sigma clip on the observed residual
    value, which would.
    """
    catalog_x = np.asarray(catalog_x, dtype=float)
    catalog_y = np.asarray(catalog_y, dtype=float)
    x_star, y_star = float(catalog_x[star_row]), float(catalog_y[star_row])
    reach = float(exclusion_radius_px) + float(search_margin_px)
    dist_to_star = np.hypot(catalog_x - x_star, catalog_y - y_star)
    is_other = np.arange(len(catalog_x)) != star_row
    neighbor_rows = np.flatnonzero(is_other & (dist_to_star <= reach))
    if not len(neighbor_rows):
        return np.zeros(pix_x.shape[0], dtype=bool), 0
    dx = np.asarray(pix_x, dtype=float)[:, None] - catalog_x[neighbor_rows][None, :]
    dy = np.asarray(pix_y, dtype=float)[:, None] - catalog_y[neighbor_rows][None, :]
    contaminated = (np.hypot(dx, dy) <= exclusion_radius_px).any(axis=1)
    return contaminated, int(len(neighbor_rows))


def morphology_projections(image: np.ndarray, counts: np.ndarray, *, oversample: int) -> dict[str, float]:
    """Project a stack on constant, dipole, radial-even and quadrupole maps."""
    n = image.shape[0]
    coordinate = (np.arange(n) - (n - 1) / 2) / oversample
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    rr2 = xx**2 + yy**2
    bases = np.column_stack([
        np.ones(n * n), xx.ravel(), yy.ravel(),
        (rr2 - np.nanmean(rr2)).ravel(), (xx**2 - yy**2).ravel(),
    ])
    names = ("constant", "x_dipole", "y_dipole", "radial_even", "quadrupole")
    value = image.ravel()
    weight = counts.ravel().astype(float)
    ok = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if ok.sum() < len(names) + 2:
        return {name: np.nan for name in names}
    coef, *_ = np.linalg.lstsq(bases[ok] * np.sqrt(weight[ok, None]), value[ok] * np.sqrt(weight[ok]), rcond=None)
    return {name: float(val) for name, val in zip(names, coef)}


def _plot_grid(images: np.ndarray, counts: np.ndarray, path: Path, *, title: str, unit: str) -> None:
    finite = np.abs(images[np.isfinite(images)])
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    vmax = max(vmax, 1e-8)
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    image_artist = None
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            image_artist = ax.imshow(images[row, col], origin="lower", cmap="coolwarm", norm=norm, interpolation="nearest")
            ax.set_title(f"row {row + 1}, col {col + 1}; peak N={int(np.nanmax(counts[row, col]))}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(image_artist, ax=axes.ravel().tolist(), shrink=0.8, label=unit)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _params_path(artifact_dir: Path) -> Path:
    return SC._params_path(Path(artifact_dir))


def run_stacks(
    artifact_dir: Path,
    bundle_path: Path,
    output_dir: Path,
    *,
    frame_block: int = 16,
    min_coverage: float = 0.50,
    min_groups: int = DEFAULT_MIN_GROUPS,
    oversample: int = DEFAULT_OVERSAMPLE,
    half_width: float = DEFAULT_HALF_WIDTH,
    neighbor_exclusion_radius_px: float = DEFAULT_NEIGHBOR_EXCLUSION_PX,
    verify_chi2: bool = True,
    require_chi2_agreement: bool = False,
) -> dict:
    """Render and write the complete read-only temporal residual diagnostic."""
    artifact_dir, bundle_path, output_dir = map(Path, (artifact_dir, bundle_path, output_dir))
    qa_path = artifact_dir.parent / "investigation" / "outputs" / "cadence_quality.csv"
    with np.load(artifact_dir / "flux_solved.npz", allow_pickle=False) as solved:
        flux_all = np.asarray(solved["flux"], dtype=float)
        chi2_all = np.asarray(solved["chi2_red"], dtype=float)
        active_all = np.asarray(solved["stamp_active"], dtype=bool)
        btjd = np.asarray(solved["btjd"], dtype=float)
    qa = pd.read_csv(qa_path)
    strict = qa.strict_mask.to_numpy(dtype=bool)
    if strict.shape != btjd.shape or not np.allclose(qa.btjd.to_numpy(float), btjd):
        raise ValueError("cadence QA table does not align with flux_solved")
    params = FIT.load_params_npz(_params_path(artifact_dir))
    metadata = _bundle_group_metadata(bundle_path, params)
    selected = select_stack_groups(metadata, flux_all, active_all, strict, min_coverage=min_coverage)
    blocks = time_blocks(btjd, strict)

    global_groups = selected.group_index.to_numpy(dtype=int)
    x_slot, y_slot, w_basis, node_x, node_y, _ = _selected_positions(
        bundle_path, params, selected.star_row.to_numpy(dtype=int),
    )
    if x_slot.shape[1] != len(btjd):
        raise ValueError("compact render cadence grid differs from flux_solved")
    # A payload is deliberately per-group: packed support widths differ by
    # tier, while rasterize_packed_values maps every payload to one common
    # output coordinate system.
    payloads: dict[int, dict[str, np.ndarray]] = {}
    active_by_group: dict[int, np.ndarray] = {}
    chi2_rel_parts: list[np.ndarray] = []
    contamination_stats: dict[int, dict[str, int]] = {}
    with np.load(bundle_path, allow_pickle=False) as raw:
        catalog_x = np.asarray(raw["x_lin"], dtype=float)
        catalog_y = np.asarray(raw["y_lin"], dtype=float)
        for tier_index, tier_selected in selected.groupby("tier", sort=True):
            tier_index = int(tier_index)
            selected_rows = tier_selected.index.to_numpy(dtype=int)
            local_rows = tier_selected.local_row.to_numpy(dtype=int)
            tier_groups = tier_selected.group_index.to_numpy(dtype=int)
            tier_star_rows = tier_selected.star_row.to_numpy(dtype=int)
            if int(np.asarray(raw[f"pt{tier_index}_k_tier"])) != 1:
                raise RuntimeError(f"selected tier {tier_index} has K != 1")
            pix_x = np.asarray(raw[f"pt{tier_index}_pix_x"], dtype=float)[local_rows]
            pix_y = np.asarray(raw[f"pt{tier_index}_pix_y"], dtype=float)[local_rows]
            pix_valid = np.asarray(raw[f"pt{tier_index}_pix_valid"], dtype=float)[local_rows]
            data = SC.read_selected_npy_rows(bundle_path, f"pt{tier_index}_data.npy", local_rows).astype(float)
            noise = SC.read_selected_npy_rows(bundle_path, f"pt{tier_index}_noise.npy", local_rows).astype(float)
            weight = SC.read_selected_npy_rows(bundle_path, f"pt{tier_index}_weight_u8.npy", local_rows).astype(float)
            templates = render_isolated_packed_numpy(
                params, x_slot[selected_rows], y_slot[selected_rows], w_basis,
                node_x, node_y, pix_x, pix_y, pix_valid, frame_block=frame_block,
            )
            model = templates * flux_all[tier_groups, :, 0, None]
            active = active_all[tier_groups] & strict[None, :]
            neighbor_mask = np.zeros_like(pix_valid, dtype=bool)
            for ti, star_row in enumerate(tier_star_rows):
                mask, n_neighbors = catalog_neighbor_pixel_mask(
                    pix_x[ti], pix_y[ti], int(star_row), catalog_x, catalog_y,
                    exclusion_radius_px=neighbor_exclusion_radius_px, search_margin_px=half_width,
                )
                neighbor_mask[ti] = mask
                contamination_stats[int(selected_rows[ti])] = {
                    "n_nearby_catalog_sources": n_neighbors,
                    "n_contaminated_pixels": int(mask.sum()),
                    "n_valid_pixels": int(pix_valid[ti].sum()),
                }
            full_coverage = weight * pix_valid[:, None, :] * active[:, :, None]
            coverage = full_coverage * (~neighbor_mask)[:, None, :]
            for ti, selected_row in enumerate(selected_rows):
                payloads[int(selected_row)] = {
                    "data": data[ti], "noise": noise[ti], "model": model[ti],
                    "coverage": coverage[ti], "pix_x": pix_x[ti], "pix_y": pix_y[ti],
                }
                active_by_group[int(selected_row)] = active[ti]
            if verify_chi2:
                # Deliberately uses full_coverage (not the neighbor-masked
                # coverage): this checks the renderer against the stored
                # chi2, which was computed over the whole stamp, not our
                # subsequent contamination exclusion.
                trial = np.sum(full_coverage * (data - model) ** 2 / (noise**2 + VARIANCE_FLOOR), axis=-1)
                trial /= np.maximum(np.sum(full_coverage, axis=-1), 1.0)
                usable = active & np.isfinite(trial) & np.isfinite(chi2_all[tier_groups])
                chi2_rel_parts.append(np.abs(trial[usable] - chi2_all[tier_groups][usable]) /
                                      np.maximum(np.abs(chi2_all[tier_groups][usable]), 1e-6))
    chi2_smoke: dict[str, float | bool | str] = {"performed": False}
    if verify_chi2:
        rel = np.concatenate(chi2_rel_parts) if chi2_rel_parts else np.empty(0, dtype=float)
        median_rel = float(np.nanmedian(rel)) if len(rel) else float("nan")
        chi2_smoke = {
            "performed": True, "n_group_frames": int(len(rel)),
            "median_relative_difference": median_rel,
            "passes_exact_gate": bool(np.isfinite(median_rel) and median_rel <= 5e-3),
            "scope": "pure_numpy_reference_renderer; exact GPU packed renderer is unavailable on CPU",
        }
        if require_chi2_agreement and not chi2_smoke["passes_exact_gate"]:
            raise RuntimeError("compact CPU render does not reproduce stored chi2_red (median relative difference > 0.5%)")

    n_grid = int(round(2 * half_width * oversample)) + 1
    fractional = np.full((3, 3, 3, n_grid, n_grid), np.nan)
    standardized = np.full_like(fractional, np.nan)
    frac_counts = np.zeros_like(fractional, dtype=np.int16)
    std_counts = np.zeros_like(fractional, dtype=np.int16)
    selection_by_group = selected.set_index("group_index")
    metric_rows: list[dict] = []
    for bi, (block_name, frames) in enumerate(blocks.items()):
        for row in range(3):
            for col in range(3):
                group_rows = [i for i, group in enumerate(global_groups)
                              if int(selection_by_group.loc[int(group), "cell_row"]) == row
                              and int(selection_by_group.loc[int(group), "cell_col"]) == col]
                fmaps, smaps = [], []
                for gi in group_rows:
                    payload = payloads[gi]
                    active = active_by_group[gi]
                    use = frames[active[frames] & np.isfinite(flux_all[global_groups[gi], frames, 0]) & (flux_all[global_groups[gi], frames, 0] > 0)]
                    if not len(use):
                        continue
                    resid = payload["data"][use] - payload["model"][use]
                    fval = resid / flux_all[global_groups[gi], use, 0, None]
                    sval = resid / payload["noise"][use]
                    x = x_slot[gi, use]
                    y = y_slot[gi, use]
                    fmap, _ = rasterize_packed_values(fval, x, y, payload["pix_x"], payload["pix_y"], payload["coverage"][use], oversample=oversample, half_width=half_width)
                    smap, _ = rasterize_packed_values(sval, x, y, payload["pix_x"], payload["pix_y"], payload["coverage"][use], oversample=oversample, half_width=half_width)
                    fmaps.append(fmap)
                    smaps.append(smap)
                if fmaps:
                    fractional[bi, row, col], frac_counts[bi, row, col] = robust_group_stack(np.asarray(fmaps), min_groups=min_groups)
                    standardized[bi, row, col], std_counts[bi, row, col] = robust_group_stack(np.asarray(smaps), min_groups=min_groups)
                for unit, image, count in (("fractional", fractional[bi, row, col], frac_counts[bi, row, col]),
                                           ("standardized", standardized[bi, row, col], std_counts[bi, row, col])):
                    projection = morphology_projections(image, count, oversample=oversample)
                    metric_rows.append({
                        "time_block": block_name, "cell_row": row, "cell_col": col, "unit": unit,
                        "n_groups_selected": len(group_rows), "n_frames": len(frames),
                        "median_pixel_group_count": float(np.nanmedian(count)),
                        "peak_pixel_group_count": int(np.nanmax(count)),
                        "fraction_pixels_meeting_min_groups": float(np.mean(count >= min_groups)), **projection,
                    })

    output_dir.mkdir(parents=True, exist_ok=True)
    contamination = pd.DataFrame.from_dict(contamination_stats, orient="index")
    contamination.index.name = "selected_row"
    selected = selected.join(contamination, how="left")
    selected.to_csv(output_dir / "selected_groups.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "stack_metrics.csv", index=False)
    for bi, block_name in enumerate(blocks):
        _plot_grid(fractional[bi], frac_counts[bi], output_dir / f"fractional_residual_{block_name}.png",
                   title=f"Fractional residual stack: {block_name}", unit="(data − model) / solved flux")
        _plot_grid(standardized[bi], std_counts[bi], output_dir / f"standardized_residual_{block_name}.png",
                   title=f"Standardized residual stack: {block_name}", unit="(data − model) / noise")
    diff = np.stack([fractional[0] - fractional[1], fractional[2] - fractional[1]])
    diff_counts = np.minimum(frac_counts[0], frac_counts[1])
    _plot_grid(diff[0], diff_counts, output_dir / "fractional_residual_early_minus_middle.png",
               title="Fractional residual: early − middle", unit="fractional residual difference")
    _plot_grid(diff[1], np.minimum(frac_counts[2], frac_counts[1]), output_dir / "fractional_residual_late_minus_middle.png",
               title="Fractional residual: late − middle", unit="fractional residual difference")
    np.savez_compressed(output_dir / "stack_arrays.npz", fractional=fractional, standardized=standardized,
                        fractional_group_counts=frac_counts, standardized_group_counts=std_counts,
                        btjd=btjd, strict_mask=strict, **{f"frames_{name}": value for name, value in blocks.items()})
    excluded = qa.loc[~strict, ["frame_index", "btjd", "strict_flag_reason"]].to_dict(orient="records")
    manifest = {
        "schema_version": "temporal-residual-stacks.v2", "artifact_dir": str(artifact_dir),
        "bundle": str(bundle_path), "params": str(_params_path(artifact_dir)),
        "flux_solved_sha256": _sha256(artifact_dir / "flux_solved.npz"), "qa_sha256": _sha256(qa_path),
        "bundle_stat": {"size": bundle_path.stat().st_size, "mtime_ns": bundle_path.stat().st_mtime_ns},
        "selection": (
            "K=1 only (blended K>=2 groups excluded); both is_epsf_contributor=True bright stars "
            "and is_epsf_contributor=False WCS-only anchors, all packed tiers, >=50% clean active "
            "and positive-flux coverage; stamp pixels within neighbor_exclusion_radius_px of any "
            "foreign reference-catalog source are dropped from that group's stack at every cadence"
        ),
        "n_selected_groups": int(len(selected)),
        "n_selected_by_population": selected["population"].value_counts().to_dict(),
        "n_groups_with_any_contaminated_pixel": int((selected["n_contaminated_pixels"] > 0).sum()),
        "neighbor_exclusion_radius_px": neighbor_exclusion_radius_px,
        "time_blocks": {name: value.tolist() for name, value in blocks.items()},
        "excluded_cadences": excluded, "oversample": oversample, "half_width_px": half_width,
        "min_groups_per_output_pixel": min_groups, "chi2_smoke": chi2_smoke,
        "interpretation_scope": "descriptive CPU reconstruction; use --require-chi2-agreement to fail closed on exact chi2 mismatch",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-block", type=int, default=16)
    parser.add_argument("--neighbor-exclusion-radius-px", type=float, default=DEFAULT_NEIGHBOR_EXCLUSION_PX)
    parser.add_argument("--no-verify-chi2", action="store_true")
    parser.add_argument("--require-chi2-agreement", action="store_true")
    args = parser.parse_args(argv)
    result = run_stacks(args.artifact_dir, args.bundle, args.output_dir, frame_block=args.frame_block,
                        neighbor_exclusion_radius_px=args.neighbor_exclusion_radius_px,
                        verify_chi2=not args.no_verify_chi2,
                        require_chi2_agreement=args.require_chi2_agreement)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
