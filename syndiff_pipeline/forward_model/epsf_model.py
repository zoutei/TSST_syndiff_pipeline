# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Node-grid ePSF: PRF init, exact area-overlap resampler, bilinear blend, renderer.

Storage convention (chosen for internal consistency, distinct from the PRF
fork's own "density, sums to oversample**2" convention — see plan Finding 3):
every stored ePSF grid (``P_base``, each ``P_k``, and any composite built from
them) is a **flux fraction** map that sums to ~1.

Representation (see CONTRACT_pixel_integrated_epsf.md): the DEFAULT stored/
rendered grid is ``E``, the pixel-integrated (Anderson & King) cell -- the
exact box-sum of the old 4x-oversampled sub-pixel grid ``P`` that the
rendered data can ever constrain (``render(P)`` at any sub-pixel phase
equals plain bilinear sampling of ``E``, to 7e-16 -- the 4-sample box sum
annihilates any period-4/2 pattern in ``P``, which is exactly what let the
optimizer fill those directions with 1-native-pixel stripes). Rendering a
physical pixel's flux is then a plain 2-tap bilinear POINT-SAMPLE of ``E`` at
native-pixel stride, scaled by ``OVERSAMPLE**2`` (no block-sum) -- see
``render_stamps``'s docstring for the exact index derivation. A basis vector
with ``sum(P_k) == 0`` is still exactly flux-neutral in the rendered stamp,
since both the box-sum conversion and the bilinear render are linear.
``SUBPIXEL_*`` constants and ``to_pixel_integrated`` remain for legacy-grid
conversion and tests only.

Grid geometry: 13 physical pixels at 4x oversampling = 52 "core" samples;
the legacy sub-pixel grid ``P`` pads 3 samples on each side (58 total) so
bilinear subpixel sampling never runs out of samples for any offset in
[-0.5, 0.5) physical pixels from the nominal (rounded) stamp center. The
pixel-integrated grid ``E`` derived from it is 3 samples narrower (55) --
see the module constants below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import _bootstrap  # noqa: F401

_LOGGER = logging.getLogger(__name__)

# TESS_PRF is imported lazily inside init_epsf_from_prf so --from-bundle
# train (jax/optax/numpy only) never needs the PRF fork or astropy.

# TESS_PRF's own default localdatadir doesn't exist in this environment
# (syndiff_pipeline/difference_imaging/stages/photometry.py:resolve_tess_prf_localdatadir
# points at a stale path); the real files live under data_old/.
PRF_ROOT_DEFAULT = "/astro/armin/koji/syndiff/data_old/tess_prf/prf_fitsfiles"

STAMP_PHYSICAL = 13
OVERSAMPLE = 4
PAD_SAMPLES = 3
CORE_SAMPLES = STAMP_PHYSICAL * OVERSAMPLE  # 52

# --- Legacy "sub-pixel" node grid P (kept for conversion + tests only) -----
# P is the old storage/render representation: 4x-oversampled sub-pixel cells,
# rendered by bilinearly sampling at 4x4 sub-positions inside each native
# pixel and SUMMING them. That 4-sample box sum annihilates any pattern of
# period 4 (or 2) samples per axis, so P has invisible directions the
# optimizer fills with noise (1-native-pixel stripes). See module docstring.
SUBPIXEL_GRID_SIZE = CORE_SAMPLES + 2 * PAD_SAMPLES  # 58
# Array index of physical offset 0: core samples are centered *between*
# indices 25 and 26 (52 is even), so this is a half-integer even before
# padding; PAD shifts it by an integer, so it stays a half-integer.
SUBPIXEL_CENTER_INDEX = (SUBPIXEL_GRID_SIZE - 1) / 2.0  # 28.5

# --- Pixel-integrated node grid E (default representation) -----------------
# E[j] is the Anderson & King-style pixel-integrated cell: the exact O-sample
# box-sum (i.e. the *only* linear combination of P the rendered data ever
# constrain -- verified: render(P) at any sub-pixel phase equals plain
# bilinear sampling of E, to 7e-16). Rendering E is a plain 2-tap bilinear
# point-sample at native-pixel stride -- no invisible directions.
# G_E = S*O + 2*PAD - (O-1); see CONTRACT_pixel_integrated_epsf.md for the
# derivation (E[j] represents the native pixel whose centre sits at P-sample
# position j+1.5, so index j of E and block-start j of P coincide).
NODE_GRID_SIZE = SUBPIXEL_GRID_SIZE - (OVERSAMPLE - 1)  # 55
NODE_CENTER_INDEX = (NODE_GRID_SIZE - 1) / 2.0  # 27.0 (an exact centre sample: G_E is odd)

EPSF_REPR = "pixel_integrated_v1"
LEGACY_EPSF_REPR = "subpixel_v0"

NATIVE_PRF_SAMPLES = 117
NATIVE_OVERSAMPLE = 9
# TESS_PRF FITS files carry CRPIX1P=CRPIX2P=59 (1-based) -> 0-based 58.
NATIVE_CENTER_INDEX = 58.0


def node_geometry(
    stamp_physical: int = STAMP_PHYSICAL, *, legacy: bool = False
) -> tuple[int, int, float]:
    """Return ``(core_samples, node_grid_size, node_center_index)`` for stamp size ``S``.

    Pixel-integrated (``legacy=False``, the default representation everywhere
    now) by default; ``legacy=True`` returns the old sub-pixel ``P`` grid's
    geometry (58 for ``S=13``) -- needed only where the code must build a
    legacy grid on purpose (``resample_prf_native_to_node``, conversion/tests).
    """
    if int(stamp_physical) < 3 or int(stamp_physical) % 2 == 0:
        raise ValueError(f"stamp_physical must be an odd integer >= 3, got {stamp_physical}")
    core = int(stamp_physical) * OVERSAMPLE
    if legacy:
        node = core + 2 * PAD_SAMPLES
    else:
        node = core + 2 * PAD_SAMPLES - (OVERSAMPLE - 1)
    center = (node - 1) / 2.0
    return core, node, center


def is_subpixel_grid(g_size: int) -> bool:
    """True if ``g_size`` is a legacy sub-pixel ``P`` grid size (58 by default).

    General test: a pixel-integrated ``E`` grid built from any odd
    ``stamp_physical`` satisfies ``(G_E - 2*PAD) % OVERSAMPLE == OVERSAMPLE -
    1`` (never 0 for ``OVERSAMPLE > 1``), so this is unambiguous.
    """
    return (int(g_size) - 2 * PAD_SAMPLES) % OVERSAMPLE == 0


def stamp_physical_from_node_size(n_grid: int) -> int:
    """Invert ``node_geometry``: physical stamp size from node-grid side length.

    Detects legacy (sub-pixel) vs. pixel-integrated grid sizes via
    ``is_subpixel_grid`` and inverts the matching formula.
    """
    n_grid = int(n_grid)
    if is_subpixel_grid(n_grid):
        return (n_grid - 2 * PAD_SAMPLES) // OVERSAMPLE
    return (n_grid - 2 * PAD_SAMPLES + OVERSAMPLE - 1) // OVERSAMPLE


def node_center_for_grid(g_size: int) -> float:
    """Node-grid center index for an arbitrary grid side length ``g_size``.

    General identity: ``center = (g_size - 1) / 2`` for *any* square node
    grid built by ``node_geometry`` (legacy or pixel-integrated, any
    ``stamp_physical``) -- both ``node_geometry``'s ``core + 2*PAD [-
    (OVERSAMPLE-1)]`` construction and this formula agree exactly (see
    CONTRACT_pixel_integrated_epsf.md), so no legacy/E branch or
    ``stamp_physical`` round-trip is needed here. ``g_size`` is a static
    shape (``composite.shape[-1]``), so this is plain Python arithmetic, not
    a traced op.
    """
    return (int(g_size) - 1) / 2.0


def to_pixel_integrated(P: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Exact, linear conversion of a legacy sub-pixel grid ``P`` (..., G_P, G_P)
    to the pixel-integrated grid ``E`` (..., G_E, G_E) it fully determines.

    ``E_raw[..., j, i] = sum_{k,l=0..O-1} P[..., j+k, i+l]``, ``E = E_raw /
    O**2`` (so ``E`` still sums to ~1, like ``P``) -- the exact O x O
    box-sum/"effective PSF" the rendered data can ever constrain (module
    docstring). Linear, so it applies identically to a decoded base, a mode,
    or a decoded chroma image. Works on numpy or jax arrays; always returns a
    jax array (callers ``np.asarray`` it if a plain ndarray is needed).
    """
    P = jnp.asarray(P)
    O = OVERSAMPLE
    g_p = P.shape[-1]
    n_out = g_p - O + 1
    if n_out <= 0:
        raise ValueError(f"grid too small to convert: last dim {g_p} < OVERSAMPLE={O}")
    acc_x = P[..., 0:n_out]
    for k in range(1, O):
        acc_x = acc_x + P[..., k : k + n_out]
    acc_y = acc_x[..., 0:n_out, :]
    for k in range(1, O):
        acc_y = acc_y + acc_x[..., k : k + n_out, :]
    return acc_y / (O ** 2)


def convert_legacy_epsf_array(array: jnp.ndarray | np.ndarray, *, name: str = "epsf array") -> jnp.ndarray:
    """Shared load-site helper: if ``array``'s last (grid) axis is a legacy
    sub-pixel size, convert it (a DECODED array -- positive flux fraction,
    NOT a raw/pre-softplus leaf) to the pixel-integrated default via
    ``to_pixel_integrated`` and log one warning; otherwise return unchanged
    (as a jax array). Every disk-load site for an ePSF array (fit bundles,
    checkpoints, warm starts, resume) should route through this so a legacy
    (58-grid) artifact silently upgrades instead of silently mis-rendering.
    """
    array = jnp.asarray(array)
    g = int(array.shape[-1])
    if is_subpixel_grid(g):
        _LOGGER.warning(
            "%s is a legacy sub-pixel ePSF grid (last dim=%d); converting to "
            "the pixel-integrated default representation (%s).",
            name, g, EPSF_REPR,
        )
        return to_pixel_integrated(array)
    return array


# ---------------------------------------------------------------------------
# Exact area-overlap resampling (flux-fraction preserving, arbitrary grids)
# ---------------------------------------------------------------------------


def _cell_edges(n: int, spacing: float, center_index: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=float)
    centers = (idx - center_index) * spacing
    return centers - spacing / 2.0, centers + spacing / 2.0


def area_overlap_matrix(
    n_src: int,
    spacing_src: float,
    center_src: float,
    n_dst: int,
    spacing_dst: float,
    center_dst: float,
) -> np.ndarray:
    """(n_dst, n_src) matrix ``R`` with ``output = R @ input`` exactly flux-conserving.

    ``input`` is treated as the *total* value in each source cell (not a
    density): ``R[k, j] = overlap_length(j, k) / spacing_src`` so that, for a
    target grid fully covering a source cell, ``sum(R[:, j]) == 1``.
    """
    left_s, right_s = _cell_edges(n_src, spacing_src, center_src)
    left_d, right_d = _cell_edges(n_dst, spacing_dst, center_dst)
    ov_left = np.maximum(left_d[:, None], left_s[None, :])
    ov_right = np.minimum(right_d[:, None], right_s[None, :])
    overlap = np.clip(ov_right - ov_left, 0.0, None)
    return overlap / spacing_src


def resample_prf_native_to_node(
    native: np.ndarray,
    *,
    stamp_physical: int = STAMP_PHYSICAL,
    native_center_index: float = NATIVE_CENTER_INDEX,
) -> np.ndarray:
    """(117,117) native 9x PRF (density, sum=81) -> LEGACY sub-pixel node grid
    (sum~1). Deliberately still the ``P`` (sub-pixel) representation, not the
    pixel-integrated default -- PRF init converts explicitly afterwards
    (``init_epsf_from_prf``) so the outer-ring zeroing happens on ``P``."""
    n_src = native.shape[0]
    frac = native / (float(NATIVE_OVERSAMPLE) ** 2)
    _, node, center = node_geometry(stamp_physical, legacy=True)
    R = area_overlap_matrix(
        n_src, 1.0 / NATIVE_OVERSAMPLE, native_center_index,
        node, 1.0 / OVERSAMPLE, center,
    )
    return R @ frac @ R.T


# ---------------------------------------------------------------------------
# Node grid geometry + parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpsfGridStatic:
    node_x: np.ndarray  # (n_cols,), strictly increasing, region-local px
    node_y: np.ndarray  # (n_rows,), strictly increasing, region-local px
    node_col_ccd: np.ndarray  # (n_cols,), full-CCD column (for PRF lookup)
    node_row_ccd: np.ndarray  # (n_rows,), full-CCD row (for PRF lookup)

    @property
    def n_rows(self) -> int:
        return len(self.node_y)

    @property
    def n_cols(self) -> int:
        return len(self.node_x)

    @classmethod
    def from_region(
        cls, region, *, n_rows: int, n_cols: int, crop_origin: tuple[int, int],
        placement: str = "center",
    ):
        """Node positions for an ``n_rows x n_cols`` grid over ``region``.

        ``placement="center"`` (default, unchanged from the original
        implementation): nodes sit at the centers of an ``n_rows x n_cols``
        even split of the region -- the outermost nodes are half a cell
        width/height inside the region bounds, so any query outside the
        node box is flat-extrapolated (clamped) by ``bilinear_cell``.

        ``placement="edge"`` (new, default-off): the outermost nodes sit
        exactly AT the region bounds (``x0``/``x1``, ``y0``/``y1``), with
        the remaining nodes evenly spaced between them -- i.e. a plain
        ``linspace(x0, x1, n_cols)`` / ``linspace(y0, y1, n_rows)``. This
        removes the need for out-of-box clamped extrapolation anywhere
        inside the region (S3/S4 investigation: edge-anchored node grids
        measurably beat center-anchored grids at matched node count in a
        half-split cross-validation of the static field-position defect --
        see ``dev/forward_epsf_wcs/diagnostics/s3_node_fit.py``).
        """
        if n_rows < 2 or n_cols < 2:
            raise ValueError("bilinear node blend needs n_rows >= 2 and n_cols >= 2")
        x0, y0, x1, y1 = region.x_min, region.y_min, region.x_max, region.y_max
        if placement == "center":
            col_w = (x1 - x0) / n_cols
            row_h = (y1 - y0) / n_rows
            node_x = x0 + (np.arange(n_cols) + 0.5) * col_w
            node_y = y0 + (np.arange(n_rows) + 0.5) * row_h
        elif placement == "edge":
            node_x = np.linspace(x0, x1, n_cols)
            node_y = np.linspace(y0, y1, n_rows)
        else:
            raise ValueError(f"placement must be 'center' or 'edge', got {placement!r}")
        ox, oy = crop_origin
        return cls(
            node_x=node_x, node_y=node_y,
            node_col_ccd=node_x + ox, node_row_ccd=node_y + oy,
        )


@dataclass
class EpsfGridParams:
    base: jnp.ndarray  # (n_rows, n_cols, G, G)
    modes: jnp.ndarray  # (K, n_rows, n_cols, G, G), K = number of deformation modes


def node_coord_1d(
    dtype=jnp.float32,
    *,
    stamp_physical: int | None = None,
    n_grid: int | None = None,
) -> jnp.ndarray:
    """Physical-px coordinates of oversampled node-grid samples (length G).

    When ``n_grid`` is given directly, it is used as-is (grid length AND its
    center via ``node_center_for_grid``, which is representation-agnostic --
    ``(G-1)/2`` -- so this works for a legacy sub-pixel grid too, e.g. during
    load-site conversion, without a lossy stamp_physical round-trip through
    ``node_geometry``'s now pixel-integrated-by-default sizing).
    """
    if n_grid is None:
        if stamp_physical is None:
            stamp_physical = STAMP_PHYSICAL
        _, n_grid, _ = node_geometry(stamp_physical)
    center = node_center_for_grid(int(n_grid))
    idx = jnp.arange(int(n_grid), dtype=dtype)
    return (idx - center) / OVERSAMPLE


def flux_centroid_xy(grid: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Flux-weighted COM in physical px for ``grid`` shaped ``(..., G, G)``.

    Global (unwindowed) first moment — a diagnostic, not the origin gauge.
    The hard gauge is ``core_centroid_xy`` / ``recenter_grid_core``.
    Last axis is detector-x, second-to-last is detector-y.
    """
    coord = node_coord_1d(dtype=grid.dtype, n_grid=int(grid.shape[-1]))
    total = jnp.sum(grid, axis=(-2, -1)) + 1e-12
    cx = jnp.sum(grid * coord, axis=(-2, -1)) / total
    cy = jnp.sum(grid * coord[:, None], axis=(-2, -1)) / total
    return cx, cy


# Gaussian core window: 3 sub-pixels = 0.75 physical px at 4× oversampling.
# Isolates the high-S/N symmetric core so coma/wings cannot pull the origin.
CORE_CENTROID_SIGMA_SUBPIXELS = 3.0


def core_gaussian_weight(g_size: int, *, dtype=jnp.float32) -> jnp.ndarray:
    """``(G, G)`` Gaussian window centered on the node-grid origin.

    Fully differentiable (no boolean mask / hard truncation). ``g_size`` is a
    static Python int (a traced grid's trailing shape).
    """
    sigma = jnp.asarray(CORE_CENTROID_SIGMA_SUBPIXELS, dtype=dtype)
    cy = (g_size - 1) / 2.0
    cx = (g_size - 1) / 2.0
    y = jnp.arange(g_size, dtype=dtype)[:, None]
    x = jnp.arange(g_size, dtype=dtype)[None, :]
    r_sq = (y - cy) ** 2 + (x - cx) ** 2
    return jnp.exp(-0.5 * r_sq / (sigma * sigma))


def core_centroid_xy(grid: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gaussian-core-weighted centroid in physical px for ``grid`` ``(..., G, G)``.

    Same axis convention as ``flux_centroid_xy``. The window is the fixed
    geometric origin, so this is the quantity ``recenter_grid_core`` drives to 0.
    """
    g_size = int(grid.shape[-1])
    w = core_gaussian_weight(g_size, dtype=grid.dtype)
    weighted = grid * w
    total = jnp.sum(weighted, axis=(-2, -1)) + 1e-12
    coord = node_coord_1d(dtype=grid.dtype, n_grid=g_size)
    cx = jnp.sum(weighted * coord, axis=(-2, -1)) / total
    cy = jnp.sum(weighted * coord[:, None], axis=(-2, -1)) / total
    return cx, cy


def bilinear_shift_physical(
    grid: jnp.ndarray,
    dx_phys: jnp.ndarray,
    dy_phys: jnp.ndarray,
) -> jnp.ndarray:
    """Shift grid content by ``(dx_phys, dy_phys)`` physical pixels (JIT-safe).

    Positive ``dx_phys`` moves flux toward higher detector-x (higher last-axis
    index), matching ``diagnostics.shift_epsf_base_physical``. Leading batch
    dims of ``dx_phys``/``dy_phys`` broadcast to ``grid.shape[:-2]``. Outside
    the ``G×G`` support is treated as 0 (constant boundary).
    """
    G = grid.shape[-1]
    leading = grid.shape[:-2]
    flat = grid.reshape((-1, G, G))
    n_batch = flat.shape[0]
    dx = jnp.broadcast_to(jnp.asarray(dx_phys, dtype=grid.dtype).reshape(-1), (n_batch,))
    dy = jnp.broadcast_to(jnp.asarray(dy_phys, dtype=grid.dtype).reshape(-1), (n_batch,))

    # output[q] = input[q - shift]; content moves +dx → shift = +dx*os in index
    iy = jnp.arange(G, dtype=grid.dtype)
    ix = jnp.arange(G, dtype=grid.dtype)
    # (n_batch, G, G) sample coordinates into the source
    gy = iy[None, :, None] - dy[:, None, None] * OVERSAMPLE
    gx = ix[None, None, :] - dx[:, None, None] * OVERSAMPLE

    x0 = jnp.floor(gx).astype(jnp.int32)
    y0 = jnp.floor(gy).astype(jnp.int32)
    fx = gx - x0
    fy = gy - y0

    def _gather(y_off: int, x_off: int) -> jnp.ndarray:
        yy = y0 + y_off
        xx = x0 + x_off
        inside = (yy >= 0) & (yy < G) & (xx >= 0) & (xx < G)
        yy_c = jnp.clip(yy, 0, G - 1)
        xx_c = jnp.clip(xx, 0, G - 1)
        b = jnp.arange(n_batch)[:, None, None]
        return jnp.where(inside, flat[b, yy_c, xx_c], jnp.asarray(0.0, dtype=grid.dtype))

    sampled = (
        _gather(0, 0) * (1 - fx) * (1 - fy)
        + _gather(0, 1) * fx * (1 - fy)
        + _gather(1, 0) * (1 - fx) * fy
        + _gather(1, 1) * fx * fy
    )
    return sampled.reshape(leading + (G, G))


# Decode keeps 2 iters for a clean gauge; the per-slot hot path uses 1
# (TESS cores converge to ≲1e-6 px in one pass — see docstring below).
DECODE_RECENTER_N_ITER = 2
HOTPATH_RECENTER_N_ITER = 1


def recenter_grid_core(
    grid: jnp.ndarray,
    *,
    clip_nonneg: bool = False,
    n_iter: int = DECODE_RECENTER_N_ITER,
) -> jnp.ndarray:
    """Hard gauge: bilinear-shift so the Gaussian-core centroid → origin, then renorm Σ=1.

    Uses a soft Gaussian window (``CORE_CENTROID_SIGMA_SUBPIXELS=3``, 0.75
    physical px at 4×) so asymmetric optical wings (coma) cannot pull the
    ePSF origin. Fully differentiable — no boolean masks or hard truncation.

    ``clip_nonneg=True`` for decoded ``P_base`` (strictly positive flux fraction).
    Leave ``False`` for mode-augmented composites (may be slightly negative).

    A single shift is exact for compact cores well inside the pad; ``n_iter``
    (default 2 for decode) repeats shift+renorm. Real TESS PRF/ePSF cores
    converge to ≲1e-6 px in one pass — use ``HOTPATH_RECENTER_N_ITER`` (=1)
    on the per-slot render path.
    """
    out = grid
    for _ in range(n_iter):
        cx, cy = core_centroid_xy(out)
        out = bilinear_shift_physical(out, -cx, -cy)
        if clip_nonneg:
            out = jnp.clip(out, 0.0)
        total = jnp.sum(out, axis=(-2, -1), keepdims=True) + 1e-12
        out = out / total
    return out


def recenter_grid_com(
    grid: jnp.ndarray,
    *,
    clip_nonneg: bool = False,
    n_iter: int = DECODE_RECENTER_N_ITER,
) -> jnp.ndarray:
    """Deprecated alias for ``recenter_grid_core`` (core-focused Gaussian centroid)."""
    return recenter_grid_core(grid, clip_nonneg=clip_nonneg, n_iter=n_iter)


def phase_class_sums(grid: jnp.ndarray) -> jnp.ndarray:
    """``(..., O, O) sums * O**2`` per node, for phase class ``(index_y mod O,
    index_x mod O)`` (``O = OVERSAMPLE``); each entry is 1 exactly when the
    flux rule (every sub-pixel phase carries an equal share of a star's flux)
    holds. ``grid`` need not have ``G % O == 0`` (``G_E = 55`` doesn't); each
    class is picked out by a strided slice, not a reshape.
    """
    O = OVERSAMPLE
    sums = []
    for a in range(O):
        row_a = grid[..., a::O, :]
        for b in range(O):
            sums.append(jnp.sum(row_a[..., :, b::O], axis=(-2, -1)))
    stacked = jnp.stack(sums, axis=-1)
    return stacked.reshape(stacked.shape[:-1] + (O, O)) * (O ** 2)


def phase_flux_rms(grid: jnp.ndarray) -> jnp.ndarray:
    """Scalar RMS over classes and nodes of ``(phase_class_sums(grid) - 1)``.

    Metric only (logged in training history), not a loss term -- see
    ``enforce_phase_flux_rule``, which is what actually holds this at ~0.
    """
    return jnp.sqrt(jnp.mean((phase_class_sums(grid) - 1.0) ** 2))


def enforce_phase_flux_rule(grid: jnp.ndarray) -> jnp.ndarray:
    """Multiplicative per-phase-class rescale so a star's total flux cannot
    depend on its sub-pixel phase (hard rule, CONTRACT ``Flux rule``):
    ``E[a::O, b::O] *= (1/O**2) / sum(E[a::O, b::O])`` per node. Positive,
    parameter-free, differentiable; applied in ``decode_epsf_base`` after
    softplus+normalize and before the core recenter (which renormalizes the
    grid sum to 1 again and, being a bilinear shift, preserves equal class
    sums away from the edges).
    """
    O = OVERSAMPLE
    G = grid.shape[-1]
    class_sum = phase_class_sums(grid) / (O ** 2)  # (..., O, O), actual per-class sum
    mult = (1.0 / (O ** 2)) / (class_sum + 1e-12)  # (..., O, O)
    idx = jnp.arange(G)
    cls = idx % O
    mult_rows = jnp.take(mult, cls, axis=-2)  # (..., G, O)
    mult_full = jnp.take(mult_rows, cls, axis=-1)  # (..., G, G)
    return grid * mult_full


def decode_epsf_base(raw: jnp.ndarray) -> jnp.ndarray:
    """Positive flux-fraction grid with hard core-centroid=0 (WCS carries all translation).

    Also enforces the flux rule (``enforce_phase_flux_rule``): every 16-way
    (for O=4) sub-pixel phase class must sum to exactly ``1/O**2``, so a
    star's flux does not depend on its sub-pixel phase. Applied after
    softplus+normalize, before the core recenter.
    """
    pos = jax.nn.softplus(raw) + 1e-8
    base = pos / jnp.sum(pos, axis=(-2, -1), keepdims=True)
    base = enforce_phase_flux_rule(base)
    return recenter_grid_core(base, clip_nonneg=True)


def blur_epsf_base(base: jnp.ndarray, dsigma: jnp.ndarray, *, oversample: int | None = None) -> jnp.ndarray:
    """Convolve every node image with a zero-mean Gaussian of covariance ``dsigma`` (TESS px^2).

    Temporal ePSF model (dev_runs/temporal_epsf_20260929): a frame's ePSF is the static ePSF blurred by that
    frame's pointing jitter. Blur commutes with pixel integration, so blurring the pixel-integrated node grid is
    exact. ``dsigma`` is (n_rows, n_cols, 3) = (Sxx, Syy, Sxy) per node, RELATIVE to the static ePSF's own
    jitter, so it may be slightly negative (a mild sharpening; the Fourier factor stays bounded for |S| << 1).
    Flux and centroid are preserved (the kernel is normalised and zero-mean).
    """
    os_ = int(OVERSAMPLE if oversample is None else oversample)
    G = int(base.shape[-1])
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(G, d=1.0 / os_)            # rad per TESS px
    ky, kx = jnp.meshgrid(k, k, indexing="ij")
    sxx, syy, sxy = dsigma[..., 0, None, None], dsigma[..., 1, None, None], dsigma[..., 2, None, None]
    H = jnp.exp(-0.5 * (sxx * kx ** 2 + syy * ky ** 2 + 2.0 * sxy * kx * ky))
    out = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(base) * H))
    return out.astype(base.dtype)


def encode_epsf_base(base: jnp.ndarray) -> jnp.ndarray:
    """Inverse softplus for a strictly positive flux-fraction grid."""
    clipped = jnp.clip(base, 1e-8, None)
    return clipped + jnp.log(-jnp.expm1(-clipped))


def _decode_epsf_base_legacy(raw: jnp.ndarray) -> jnp.ndarray:
    """Pre-representation-change decode (softplus + normalize + recenter,
    NO phase-flux-rule): used ONLY to correctly interpret an old raw leaf
    before converting it -- see ``convert_legacy_epsf_base_raw``."""
    pos = jax.nn.softplus(raw) + 1e-8
    base = pos / jnp.sum(pos, axis=(-2, -1), keepdims=True)
    return recenter_grid_core(base, clip_nonneg=True)


def convert_legacy_epsf_base_raw(raw: jnp.ndarray | np.ndarray, *, name: str = "epsf_base_raw") -> jnp.ndarray:
    """Load-site helper for the ``epsf_base_raw`` RAW (pre-softplus) leaf.

    If ``raw``'s last dim is a legacy sub-pixel size: decode it with the OLD
    (pre-phase-rule) semantics -- the only semantics that raw leaf could have
    been encoded under -- convert the DECODED array to the pixel-integrated
    default (``to_pixel_integrated``), and re-encode (``encode_epsf_base``)
    so the result is a valid raw leaf for the CURRENT ``decode_epsf_base``
    (which also applies the phase-flux rule). Logs one warning. No-op
    (returned unchanged, as a jax array) if already the current size.
    """
    raw = jnp.asarray(raw)
    if not is_subpixel_grid(int(raw.shape[-1])):
        return raw
    _LOGGER.warning(
        "%s is a legacy sub-pixel ePSF raw leaf (last dim=%d); converting to "
        "the pixel-integrated default representation (%s).",
        name, raw.shape[-1], EPSF_REPR,
    )
    legacy_decoded = _decode_epsf_base_legacy(raw)
    converted = to_pixel_integrated(legacy_decoded)
    return encode_epsf_base(converted)


def convert_legacy_epsf_modes_raw(
    raw: jnp.ndarray | np.ndarray,
    base_decoded: jnp.ndarray | np.ndarray,
    *,
    name: str = "epsf_modes",
) -> jnp.ndarray:
    """Load-site helper for the ``epsf_modes`` RAW leaf.

    ``base_decoded`` must be the MATCHING (same representation/size as
    ``raw``) decoded base -- i.e. the legacy (58) decoded base if ``raw`` is
    legacy-sized, needed to reproduce ``decode_epsf_modes``'s gauge
    projections exactly as they were computed when ``raw`` was written
    (those gauges/``decode_epsf_modes`` are already generic in the grid size
    -- CONTRACT_pixel_integrated_epsf.md -- so no separate "legacy decode"
    variant is needed here, unlike ``epsf_base_raw``'s softplus/phase-rule
    change). Converts the DECODED modes (linear in ``raw`` for fixed base,
    so decode-then-convert-then-reencode round-trips exactly) and re-encodes
    against the matching converted (pixel-integrated) base. Logs one warning.
    No-op if already the current size.
    """
    raw = jnp.asarray(raw)
    if not is_subpixel_grid(int(raw.shape[-1])):
        return raw
    _LOGGER.warning(
        "%s is a legacy sub-pixel ePSF raw leaf (last dim=%d); converting to "
        "the pixel-integrated default representation (%s).",
        name, raw.shape[-1], EPSF_REPR,
    )
    base_decoded = jnp.asarray(base_decoded)
    legacy_decoded_modes = decode_epsf_modes(raw, base_decoded)
    converted_modes = to_pixel_integrated(legacy_decoded_modes)
    converted_base = to_pixel_integrated(base_decoded)
    return encode_epsf_modes(converted_modes, converted_base)


def convert_legacy_chroma_image_raw(
    raw: jnp.ndarray | np.ndarray,
    base_decoded: jnp.ndarray | np.ndarray,
    *,
    name: str = "chroma_image",
) -> jnp.ndarray:
    """Load-site helper for the ``chroma_image`` RAW leaf -- same reasoning
    as ``convert_legacy_epsf_modes_raw`` (``decode_chroma_image`` is
    ``decode_epsf_modes`` applied to a single ``(G, G)`` image against the
    node-mean base, already generic in ``G``). ``base_decoded`` is the
    matching (legacy-sized) decoded base field ``(n_rows, n_cols, G, G)``.
    """
    raw = jnp.asarray(raw)
    if not is_subpixel_grid(int(raw.shape[-1])):
        return raw
    _LOGGER.warning(
        "%s is a legacy sub-pixel ePSF raw leaf (last dim=%d); converting to "
        "the pixel-integrated default representation (%s).",
        name, raw.shape[-1], EPSF_REPR,
    )
    base_decoded = jnp.asarray(base_decoded)
    legacy_decoded_image = decode_chroma_image(raw, base_decoded)
    converted_image = to_pixel_integrated(legacy_decoded_image)
    converted_base = to_pixel_integrated(base_decoded)
    return encode_chroma_image(converted_image, converted_base)


def convert_legacy_params_raw(params: dict, *, name: str = "params") -> dict:
    """Convert every legacy-sized ePSF RAW leaf in a params dict (as produced
    by ``loss.init_params`` / an optimizer checkpoint / a fit bundle's
    ``params0``) to the pixel-integrated default. Returns a NEW dict (does
    not mutate the input); a no-op copy if ``params`` has no ``epsf_base_raw``
    or it is already the current representation. Keys handled:
    ``epsf_base_raw``, ``epsf_modes``, ``chroma_image``; any other key passes
    through unchanged.
    """
    out = dict(params)
    if "epsf_base_raw" not in params:
        return out
    raw_base = jnp.asarray(params["epsf_base_raw"])
    if not is_subpixel_grid(int(raw_base.shape[-1])):
        return out
    legacy_decoded_base = _decode_epsf_base_legacy(raw_base)
    out["epsf_base_raw"] = convert_legacy_epsf_base_raw(raw_base, name=f"{name}['epsf_base_raw']")
    if "epsf_modes" in params:
        out["epsf_modes"] = convert_legacy_epsf_modes_raw(
            params["epsf_modes"], legacy_decoded_base, name=f"{name}['epsf_modes']"
        )
    if "chroma_image" in params:
        out["chroma_image"] = convert_legacy_chroma_image_raw(
            params["chroma_image"], legacy_decoded_base, name=f"{name}['chroma_image']"
        )
    return out


MODE_GAUGE_RADIUS_PX = 6.0  # widest fit_radius tier (mag<9; see loss.fit_radius_from_mag).
# A narrower radius (e.g. the tightest mag>=11 tier, 2.5px) was tried and found to strip
# real, loss-visible wing shape from wide-r_fit populations -- see
# TEMPORAL_EPSF_GAUGE_PLAN_20260808.md Sec.3.1/4.1 for the empirical falsification check.


def canonical_mode_weight_grid(g_size: int, *, radius_px: float = MODE_GAUGE_RADIUS_PX) -> jnp.ndarray:
    """Fixed uniform disk mask ``(G, G)`` in node-grid physical-px space.

    Star-independent by design: the mode/base gauge below is a property of the
    stored ePSF parameterization, not of any one star's geometry.
    """
    coord = node_coord_1d(n_grid=g_size)
    r = jnp.sqrt(coord[None, :] ** 2 + coord[:, None] ** 2)
    return (r <= radius_px).astype(jnp.float32)


def footprint_weight_grid(
    g_size: int,
    *,
    radius_px: float = 4.243,
    profile: str = "uniform",
    base: jnp.ndarray | None = None,
    flux: float = 1500.0,
    sky: float = 100.0,
) -> jnp.ndarray:
    """Weight grid matching the flux-solve metric of a packed footprint tier
    (task T1-3; default ``radius_px=4.243`` is the measured majority K=1
    packed-tier (P=64) radius -- see ``diagnostics/wfix_mode_footprint_flux.py``
    and ``S2_mode_footprint/findings.md``). Not used anywhere by default --
    pass the result as ``decode_epsf_modes(..., weight_grid=...)`` to gauge
    the mode/base orthogonality (gauge 2) in this metric instead of the wider
    ``canonical_mode_weight_grid`` (r<=6px, the widest ``fit_radius`` tier).

    ``profile``:
      - ``"uniform"``: a flat disk of the given radius (matches an
        equal-weight aperture sum, i.e. mechanism (A) in S2's findings).
      - ``"shot_noise"``: weights each oversampled sample by an approximate
        inverse-variance ``1 / (base * flux + sky / oversample**2)`` for a
        representative star of the given ``flux`` (native e-/s) and sky
        rate ``sky`` (native e-/s/px), then restricts to the same disk.
        ``base`` (a decoded, node-local flux-fraction grid, ``(G, G)``,
        summing to ~1) is required for this profile. The ``sky`` term is
        spread over the ``oversample**2`` sub-cells per physical pixel so
        that block-summing an oversampled-cell rate recovers the physical
        pixel's total sky rate; the flux term needs no such factor because
        ``base`` is already a per-cell flux *fraction*, not a density (see
        module docstring), so ``base * flux`` block-sums to the correct
        per-physical-pixel source rate directly.

    This is a diagnostic/reporting knob (S2/T1-3), not part of the trained
    gauge unless explicitly opted into a run.
    """
    coord = node_coord_1d(n_grid=g_size)
    r = jnp.sqrt(coord[None, :] ** 2 + coord[:, None] ** 2)
    disk = (r <= radius_px).astype(jnp.float32)
    if profile == "uniform":
        return disk
    if profile == "shot_noise":
        if base is None:
            raise ValueError("profile='shot_noise' requires base (flux-fraction grid, (G,G))")
        base = jnp.asarray(base)
        if base.shape != (g_size, g_size):
            raise ValueError(f"base must be ({g_size},{g_size}), got {base.shape}")
        rate = jnp.clip(base, 0.0, None) * float(flux) + float(sky) / (OVERSAMPLE ** 2)
        ivar = 1.0 / jnp.clip(rate, 1e-6, None)
        return disk * ivar
    raise ValueError(f"unknown profile {profile!r}; expected 'uniform' or 'shot_noise'")


def project_core_dipoles_out(modes: jnp.ndarray) -> jnp.ndarray:
    """Force each mode's core-weighted dipole moment to exactly zero.

    A quadrupole (or noisy) mode with a parasitic local dipole at the core
    would shift the star when ``w_k(t)`` updates, leaking into the jointly-fit
    WCS. Subtracting the rank-2 correction in the span of
    ``{dx * w_core, dy * w_core}`` (same Gaussian as ``recenter_grid_core``)
    removes that translation without touching even (defocus/astig) structure.

    ``modes`` is ``(K, n_rows, n_cols, G, G)``; leading axes are preserved via
    ellipsis so a packed ``(K, G, G)`` also works.
    """
    h, w = int(modes.shape[-2]), int(modes.shape[-1])
    dtype = modes.dtype
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    y = jnp.arange(h, dtype=dtype)
    x = jnp.arange(w, dtype=dtype)
    dy = jnp.broadcast_to((y - cy)[:, None], (h, w))
    dx = jnp.broadcast_to((x - cx)[None, :], (h, w))
    w_core = core_gaussian_weight(h, dtype=dtype)
    w_core = w_core / jnp.sum(w_core)
    basis_x = dx * w_core
    basis_y = dy * w_core

    # Constraint is sum(mode * w_core * dx) == 0. The min-norm correction in
    # the span of {basis_x, basis_y} needs var = sum(basis^2), not sum(w dx^2);
    # mixing those two leaves a residual dipole.
    dipole_x = jnp.einsum("...yx,yx->...", modes, basis_x)
    dipole_y = jnp.einsum("...yx,yx->...", modes, basis_y)
    var_x = jnp.sum(basis_x * basis_x) + 1e-12
    var_y = jnp.sum(basis_y * basis_y) + 1e-12
    correction_x = jnp.einsum("...,yx->...yx", dipole_x / var_x, basis_x)
    correction_y = jnp.einsum("...,yx->...yx", dipole_y / var_y, basis_y)
    return modes - (correction_x + correction_y)


def decode_epsf_modes(
    raw: jnp.ndarray, base: jnp.ndarray, *, weight_grid: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Flux-neutral, base-orthogonal, core-dipole-free modes: three hard gauges, per node.

    (1) Subtract per-mode/node mean over the GxG grid (exact flux-neutrality; float32
    leaves ~1e-4 residual on the sum for a 58x58 grid, acceptable vs mode RMS ~1).
    (2) Project off ``base`` in ``canonical_mode_weight_grid``'s fixed, r_fit-radius
    -weighted metric, so a mode can't carry a component that acts as a pure template
    rescale inside the region the loss actually scores. That component has zero
    data-term gradient (``FS.solve_group_fluxes`` profiles it out of the residual) and
    otherwise leaks into the analytic flux solve as an unconstrained bias -- see plan
    Sec.1. ``base`` is stop-gradient'd so this stays a pure reparameterization of
    ``epsf_modes``; it must not couple a spurious gradient back into ``epsf_base_raw``.
    (3) Project out the core-weighted dipole (``project_core_dipoles_out``) so a
    mode cannot translate the star and fight the jointly-fit WCS.

    Projects against the *mean-subtracted* base (not raw base, which sums to ~1, not
    0): subtracting any multiple of raw base would reintroduce a nonzero grid sum --
    silently breaking gauge (1) for exactly the populations where the base-overlap
    coefficient is large. Mean-subtracting base first changes the projection axis by
    a negligible constant (``mean(base) = 1/N ~ 3e-4``, versus base's peak values) while
    keeping the result exactly zero-sum by construction, so both gauges hold jointly.
    The dipole correction is odd about the origin, so it preserves the zero-sum
    gauge; residual base-overlap after (3) is ~0 because a core-centered base is
    approximately even.
    """
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(int(raw.shape[-1]))
    base_sg = jax.lax.stop_gradient(base)
    base_meansub = base_sg - jnp.mean(base_sg, axis=(-2, -1), keepdims=True)
    den = jnp.sum(weight_grid * base_meansub ** 2, axis=(-2, -1), keepdims=True) + 1e-12
    # Two alternating-projection passes so flux-neutral, base-orthogonal, and
    # core-dipole-free gauges hold jointly (sequential projectors do not
    # commute; one pass left a ~1e-3 residual that broke decode idempotency).
    gauged = raw
    for _ in range(2):
        gauged = gauged - jnp.mean(gauged, axis=(-2, -1), keepdims=True)
        num = jnp.sum(gauged * weight_grid * base_meansub[None], axis=(-2, -1), keepdims=True)
        gauged = gauged - (num / den) * base_meansub[None]
        gauged = project_core_dipoles_out(gauged)
    return gauged


def decode_chroma_image(raw: jnp.ndarray, base: jnp.ndarray) -> jnp.ndarray:
    """Gauge the global free dP/dcolour image with the three MODE gauges.

    ``raw`` is ``(G, G)``; ``base`` is the ``(n_rows, n_cols, G, G)`` node field.
    The image is gauged against the node-MEAN base, since it is one global image
    shared by every field position.

    A free colour image is structurally an ePSF mode whose amplitude is the
    per-star colour offset instead of ``w_k(t)``, so it needs exactly the same
    three gauges, for exactly the same reasons (``decode_epsf_modes``):

    1. flux-neutral -- a colour-dependent flux scale is profiled out by
       ``FS.solve_group_fluxes`` per stamp and carries no data-term gradient;
    2. base-orthogonal in the r_fit-weighted metric -- same pure-rescale
       direction, same unconstrained-bias leak into the analytic flux solve;
    3. core-dipole-free -- and this one is not optional bookkeeping: every
       forward pass re-centres each rendered slot on its own Gaussian core
       centroid, so a dipole in this image is removed per star before it can
       reach the data. The chromatic DISPLACEMENT is therefore carried by
       ``chroma_shift`` (a render offset, applied outside the recenter) and this
       image carries only the chromatic SHAPE change. Leaving the dipole
       ungauged would not measure the shift -- it would leave a direction with
       no gradient for Adam to wander along.
    """
    if raw.ndim != 2:
        raise ValueError(f"chroma_image must be a single (G, G) image, got {raw.shape}")
    base_mean = jnp.mean(base, axis=tuple(range(base.ndim - 2)))
    return decode_epsf_modes(raw[None], base_mean)[0]


def encode_chroma_image(image: jnp.ndarray, base: jnp.ndarray) -> jnp.ndarray:
    """Init helper: store an already-gauged colour image as the raw leaf."""
    return decode_chroma_image(image, base)


def encode_epsf_modes(modes: jnp.ndarray, base: jnp.ndarray) -> jnp.ndarray:
    """Init helper: store already-gauged modes as the trainable raw leaf."""
    return decode_epsf_modes(modes, base)


FD_MODE_NAMES = ("x_smear", "y_smear", "iso_defocus", "astig0", "astig45", "kurt")
# Dipole (translation) modes: jointly fitting WCS + ePSF makes these a null
# space — the ePSF can translate the star and perfectly cancel the WCS
# B-splines. Even modes (iso_defocus, astig0, astig45, kurt) remain allowed.
FORBIDDEN_MODES = ("x_smear", "y_smear")


def validate_mode_names(mode_names: tuple[str, ...]) -> tuple[str, ...]:
    """Canonicalize ``mode_names``; reject unknown names and dipole (K=5) modes."""
    unknown = set(mode_names) - set(FD_MODE_NAMES)
    if unknown:
        raise ValueError(
            f"unknown mode name(s) {sorted(unknown)}; expected subset of {FD_MODE_NAMES}"
        )
    for mode in mode_names:
        if mode in FORBIDDEN_MODES:
            raise ValueError(
                f"Mode '{mode}' is a dipole (translation). Joint WCS+ePSF fitting "
                f"requires strictly even modes (e.g., iso_defocus, astig0) to avoid "
                f"position-shape degeneracies. Do not use K=5."
            )
    return tuple(name for name in FD_MODE_NAMES if name in mode_names)


def _finite_diff_modes(base: np.ndarray, *, mode_names: tuple[str, ...] = ("iso_defocus",)) -> np.ndarray:
    """Deformation-mode init vectors from finite differences of ``base``.

    Full vocabulary (plan section 2.2): X-smear, Y-smear, isotropic defocus,
    astigmatism 0deg, astigmatism 45deg, and the core/wing kurtosis generator
    (``kurt``, added for the K=2 iso_defocus+kurt temporal model -- reuses
    ``kurt_generator`` exactly, the same ``r^2*P - <r^2 P>/<P> * P``
    construction the chromatic ``chroma_kurt`` leaf's generator uses, applied
    here to a whole per-node ``base`` rather than a per-star stacked
    template). ``mode_names`` selects a subset, in ``FD_MODE_NAMES``
    canonical order, regardless of the order requested. Dipole names
    ``x_smear`` / ``y_smear`` are rejected: they translate the star and form
    a null space with the jointly-fit WCS.
    Default is a single isotropic-defocus mode -- the dominant, physically
    identified driver of TESS's focus-breathing systematic (see
    TEMPORAL_EPSF_GAUGE_PLAN_20260808.md Sec.5); all modes are free to train
    thereafter, this only sets the init and the count K.
    Each returned mode is mean-subtracted (exact flux-neutrality, sum==0) and
    unit-RMS normalized.
    """
    selected = validate_mode_names(tuple(mode_names))
    gy, gx = np.gradient(base)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    iso = gxx + gyy
    astig0 = gxx - gyy
    astig45 = 0.5 * (gxy + gyx)
    by_name = {"x_smear": gx, "y_smear": gy, "iso_defocus": iso, "astig0": astig0, "astig45": astig45}
    if "kurt" in selected:
        by_name["kurt"] = np.asarray(kurt_generator(jnp.asarray(base, dtype=jnp.float32)))
    if not selected:
        # K = 0, i.e. --mode-init "": no temporal ePSF mode at all. The loss and the
        # optimizer already support zero-sized epsf_modes/w_coeff (loss.py's n_modes
        # == 0 branch, test_optimizer_and_provenance's "K=0 must not produce a NaN
        # loss"), but np.stack of an empty list raises, so the empty case has to be
        # built explicitly. This is what the single-FFI static fit uses: with one
        # frame the zero-temporal-mean gauge makes every w_k identically zero, so the
        # modes are unidentifiable by construction and must not be carried.
        return np.zeros((0,) + np.asarray(base).shape, dtype=np.float64)
    modes = np.stack([by_name[name] for name in selected], axis=0)
    modes = modes - modes.mean(axis=(1, 2), keepdims=True)
    scale = np.sqrt(np.mean(modes**2, axis=(1, 2), keepdims=True)) + 1e-12
    return modes / scale


def init_epsf_from_prf(
    *,
    camera: int,
    ccd: int,
    sector: int,
    grid: EpsfGridStatic,
    localdatadir: str = PRF_ROOT_DEFAULT,
    stamp_physical: int = STAMP_PHYSICAL,
    mode_names: tuple[str, ...] = ("iso_defocus",),
) -> EpsfGridParams:
    """Init P_base per node from the TESS SPOC PRF, P_k from its finite differences.

    Builds the legacy sub-pixel grid ``P`` exactly as before (``resample_prf_native_to_node``),
    zeroes its outer ``PAD_SAMPLES``-sample ring (the equivalence precondition --
    CONTRACT_pixel_integrated_epsf.md), converts it to the pixel-integrated
    default representation ``E`` (``to_pixel_integrated``), and only THEN
    core-recenters + derives FD modes -- so init sits on the same hard gauge
    enforced by ``decode_epsf_base`` / composite recenter, on ``E``, not ``P``.
    ``mode_names`` sets K (the mode count) and which FD vectors seed it -- see
    ``_finite_diff_modes``.
    """
    from PRF import TESS_PRF  # noqa: E402  (prep-only; heavy)

    n_rows, n_cols = grid.n_rows, grid.n_cols
    _, node, _ = node_geometry(stamp_physical)  # pixel-integrated (E) size
    n_modes = len(mode_names)
    base = np.zeros((n_rows, n_cols, node, node), dtype=np.float32)
    modes = np.zeros((n_modes, n_rows, n_cols, node, node), dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            prf = TESS_PRF(
                camera, ccd, sector,
                float(grid.node_col_ccd[j]), float(grid.node_row_ccd[i]),
                localdatadir=localdatadir,
            )
            node_grid_p = resample_prf_native_to_node(
                np.asarray(prf.prf, dtype=np.float64),
                stamp_physical=stamp_physical,
            )
            # Zero the outer PAD_SAMPLES-sample ring on P -- the equivalence
            # precondition for to_pixel_integrated's exact box-sum conversion.
            node_grid_p[:PAD_SAMPLES, :] = 0.0
            node_grid_p[-PAD_SAMPLES:, :] = 0.0
            node_grid_p[:, :PAD_SAMPLES] = 0.0
            node_grid_p[:, -PAD_SAMPLES:] = 0.0
            node_grid_e = np.asarray(to_pixel_integrated(node_grid_p), dtype=np.float32)
            node_grid_e = np.asarray(
                recenter_grid_core(jnp.asarray(node_grid_e, dtype=jnp.float32), clip_nonneg=True),
                dtype=np.float32,
            )
            base[i, j] = node_grid_e
            modes[:, i, j] = _finite_diff_modes(node_grid_e, mode_names=mode_names).astype(np.float32)
    return EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))


def init_epsf_from_base(
    base: np.ndarray, *, mode_names: tuple[str, ...] = ("iso_defocus",)
) -> EpsfGridParams:
    """Like ``init_epsf_from_prf``, but seeds ``P_base`` from an already-fit array
    (e.g. a previous run's converged, decoded ``epsf_base``) instead of resampling the
    TESS SPOC PRF. ``base`` must already satisfy ``decode_epsf_base``'s gauges (positive,
    sums to 1 per node, core-centroid=0) -- it is used as-is, not re-decoded.

    ``P_k`` are still (re-)derived via finite differences of *this* base, per node, so
    the FD init reflects genuine local deformations of the actual seed shape rather than
    the PRF's. All modes remain free to train thereafter; this only sets the init and K.

    ``base`` may be a legacy sub-pixel (58) grid (e.g. a previous run's
    checkpoint predating this representation) -- detected and converted to
    the pixel-integrated default via ``convert_legacy_epsf_array`` (one
    warning logged).
    """
    base = np.asarray(convert_legacy_epsf_array(base, name="init_epsf_from_base base"), dtype=np.float32)
    if base.ndim != 4:
        raise ValueError(f"expected base shaped (n_rows, n_cols, G, G), got {base.shape}")
    n_rows, n_cols, node, node2 = base.shape
    if node != node2:
        raise ValueError(f"expected a square node grid, got {base.shape}")
    n_modes = len(mode_names)
    modes = np.zeros((n_modes, n_rows, n_cols, node, node), dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            modes[:, i, j] = _finite_diff_modes(base[i, j], mode_names=mode_names).astype(np.float32)
    return EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))


# ---------------------------------------------------------------------------
# Bilinear node blend (spatial only, time-independent -- see module docstring
# in the plan: w_k(t) has no spatial index, so it factors out of the blend)
# ---------------------------------------------------------------------------


def bilinear_cell(x, y, node_x: np.ndarray, node_y: np.ndarray):
    """Locate + weight the surrounding node cell for query points (x, y).

    Returns (i0, j0, wy, wx): i0/j0 are the lower-left node row/col indices
    (clamped so i0+1/j0+1 stay in range -- flat extrapolation beyond the
    outermost nodes), wy/wx in [0, 1] are the bilinear weights.
    """
    node_x = jnp.asarray(node_x)
    node_y = jnp.asarray(node_y)
    n_cols = node_x.shape[0]
    n_rows = node_y.shape[0]
    j0 = jnp.clip(jnp.searchsorted(node_x, x, side="right") - 1, 0, n_cols - 2)
    i0 = jnp.clip(jnp.searchsorted(node_y, y, side="right") - 1, 0, n_rows - 2)
    x0, x1 = node_x[j0], node_x[j0 + 1]
    y0, y1 = node_y[i0], node_y[i0 + 1]
    wx = jnp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    wy = jnp.clip((y - y0) / (y1 - y0), 0.0, 1.0)
    return i0, j0, wy, wx


def blend_to_local(field, i0, j0, wy, wx):
    """(n_rows, n_cols, ...) node field -> (n_stars, ...) bilinear blend."""
    v00 = field[i0, j0]
    v01 = field[i0, j0 + 1]
    v10 = field[i0 + 1, j0]
    v11 = field[i0 + 1, j0 + 1]
    wy_ = wy.reshape(wy.shape + (1,) * (v00.ndim - 1))
    wx_ = wx.reshape(wx.shape + (1,) * (v00.ndim - 1))
    return (1 - wy_) * ((1 - wx_) * v00 + wx_ * v01) + wy_ * ((1 - wx_) * v10 + wx_ * v11)


def blend_to_local_2x2(field, wy, wx):
    """Explicit 2×2 node blend — no integer gather (avoids scatter-add in bwd).

    ``field`` is ``(2, 2, ...)``; ``wy``/``wx`` are ``(n_stars,)`` (or scalar).
    Corner grids ``field[i,j]`` have no batch axis, so weights get ``ndim``
    trailing singleton dims (unlike indexed ``blend_to_local``, where gather
    already injects the star axis into ``v00``).
    """
    v00 = field[0, 0]
    v01 = field[0, 1]
    v10 = field[1, 0]
    v11 = field[1, 1]
    wy = jnp.atleast_1d(jnp.asarray(wy, dtype=v00.dtype))
    wx = jnp.atleast_1d(jnp.asarray(wx, dtype=v00.dtype))
    wy_ = wy.reshape(wy.shape + (1,) * v00.ndim)
    wx_ = wx.reshape(wx.shape + (1,) * v00.ndim)
    return (1 - wy_) * ((1 - wx_) * v00 + wx_ * v01) + wy_ * ((1 - wx_) * v10 + wx_ * v11)


def blend_field(field, i0, j0, wy, wx):
    """Dispatch: explicit 2×2 path when the node grid is 2×2, else indexed gather."""
    if field.shape[0] == 2 and field.shape[1] == 2:
        return blend_to_local_2x2(field, wy, wx)
    return blend_to_local(field, i0, j0, wy, wx)


def local_base_and_modes(params: EpsfGridParams, i0, j0, wy, wx):
    """Star-local (time-independent) base + 5 mode grids, via one bilinear blend each."""
    local_base = blend_field(params.base, i0, j0, wy, wx)  # (n_stars, G, G)
    local_modes = jax.vmap(
        lambda mode: blend_field(mode, i0, j0, wy, wx),
        in_axes=0,
    )(params.modes)  # (5, n_stars, G, G)
    return local_base, local_modes


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def compose_time_varying(local_base, local_modes, w_of_t):
    """local_base (n_stars,G,G), local_modes (5,n_stars,G,G), w_of_t (n_frames,5)
    -> composite (n_stars, n_frames, G, G) flux-fraction grids.
    """
    time_term = jnp.einsum("ksxy,tk->stxy", local_modes, w_of_t)
    return local_base[:, None, :, :] + time_term


def render_stamps_from_nodes(
    node_field,
    i0,
    j0,
    wy,
    wx,
    dx,
    dy,
    *,
    n_pix: int = STAMP_PHYSICAL,
    oversample: int = OVERSAMPLE,
    recenter: bool = True,
    recenter_n_iter: int = HOTPATH_RECENTER_N_ITER,
):
    """Fused blend (+ optional core recenter) + render without a reusable local cache.

    ``node_field`` is ``(n_rows, n_cols, G, G)`` for one frame. Prefer this over
    materializing ``local`` only when the caller will not reuse the local grid;
    stage-1 dx-only caching still blends once then calls ``render_stamps``.
    """
    local = blend_field(node_field, i0, j0, wy, wx)
    if recenter:
        local = recenter_grid_core(local, clip_nonneg=False, n_iter=recenter_n_iter)
    return render_stamps(local, dx, dy, n_pix=n_pix, oversample=oversample)


def _legacy_subpixel_render_stamps(
    composite, dx, dy, *, n_pix: int = STAMP_PHYSICAL, oversample: int = OVERSAMPLE
):
    """LEGACY (sub-pixel ``P``) renderer -- kept ONLY as a private test
    reference for the pixel-integrated equivalence tests (module docstring /
    CONTRACT_pixel_integrated_epsf.md). Do not call this from production code
    -- use ``render_stamps`` (renders the pixel-integrated ``E`` grid).

    Bilinear-samples + block-*sums* a flux-fraction sub-pixel node grid into
    physical-pixel stamps: composite (..., G_P, G_P), G_P =
    n_pix*oversample + 2*PAD_SAMPLES.
    dx, dy: (...,) subpixel offset (in physical px, range [-0.5, 0.5)) of the
    true star position from the *rounded* stamp-center pixel used to extract
    the data stamp (see ``groups.py``).

    Index derivation (physical-offset space, spacing 1/oversample, array index
    for offset 0 = SUBPIXEL_CENTER_INDEX): output core sample m (0..n-1, n =
    n_pix*oversample) has physical offset o(m) = (m - (n-1)/2) / oversample.
    We need the PSF value at the star's *own* frame, i.e. at o(m) - dx.  In
    node-array-index units that's ``SUBPIXEL_CENTER_INDEX + (o(m)-dx)*oversample``,
    which simplifies (see module tests) to ``m + PAD_SAMPLES - dx*oversample``.
    """
    n = n_pix * oversample
    G = composite.shape[-1]
    m = jnp.arange(n, dtype=composite.dtype)
    gx = m[None, :] - dx[..., None] * oversample + PAD_SAMPLES
    gy = m[None, :] - dy[..., None] * oversample + PAD_SAMPLES

    # Companions attached up to ~r_attach away (well beyond a primary's own
    # +-0.5px offset) routinely need samples outside the array's physical
    # support [0, G-1]. Clamping the *index* there (as below, so the gather
    # stays in-bounds) would silently repeat whatever value sits on the edge
    # ring -- undertrained/near-arbitrary, since neither primaries nor
    # out-of-support companions can backprop through the clamp to train it
    # (see SESSION_RUN_LOG_20260808.md Sec.9). Track validity separately and
    # zero those samples instead of returning the repeated edge value
    # (Option 1a, SESSION_PLAN_20260808.md Problem 1/5a).
    in_bounds_x = (gx >= 0.0) & (gx <= float(G - 1))
    in_bounds_y = (gy >= 0.0) & (gy <= float(G - 1))

    x0 = jnp.clip(jnp.floor(gx).astype(jnp.int32), 0, G - 2)
    y0 = jnp.clip(jnp.floor(gy).astype(jnp.int32), 0, G - 2)
    fx = jnp.clip(gx - x0, 0.0, 1.0)
    fy = jnp.clip(gy - y0, 0.0, 1.0)

    flat = composite.reshape(-1, G, G)
    n_batch = flat.shape[0]
    b = jnp.arange(n_batch)[:, None, None]
    y0b, x0b = y0.reshape(n_batch, 1, n), x0.reshape(n_batch, 1, n)
    fy_b, fx_b = fy.reshape(n_batch, n, 1), fx.reshape(n_batch, 1, n)

    v00 = flat[b, y0b.reshape(n_batch, n, 1), x0b.reshape(n_batch, 1, n)]
    v01 = flat[b, y0b.reshape(n_batch, n, 1), x0b.reshape(n_batch, 1, n) + 1]
    v10 = flat[b, y0b.reshape(n_batch, n, 1) + 1, x0b.reshape(n_batch, 1, n)]
    v11 = flat[b, y0b.reshape(n_batch, n, 1) + 1, x0b.reshape(n_batch, 1, n) + 1]
    sampled = (
        v00 * (1 - fx_b) * (1 - fy_b)
        + v01 * fx_b * (1 - fy_b)
        + v10 * (1 - fx_b) * fy_b
        + v11 * fx_b * fy_b
    )  # (n_batch, n, n)
    valid = in_bounds_y.reshape(n_batch, n, 1) & in_bounds_x.reshape(n_batch, 1, n)
    sampled = jnp.where(valid, sampled, 0.0)
    stamps = sampled.reshape(n_batch, n_pix, oversample, n_pix, oversample).sum(axis=(2, 4))
    return stamps.reshape(composite.shape[:-2] + (n_pix, n_pix))


def render_stamps(composite, dx, dy, *, n_pix: int = STAMP_PHYSICAL, oversample: int = OVERSAMPLE):
    """Bilinear point-sample a pixel-integrated (``E``) node grid into
    physical-pixel stamps (CONTRACT_pixel_integrated_epsf.md): each output
    pixel is ONE bilinear sample of ``E``, scaled by ``oversample**2`` -- not
    a block-sum of sub-samples (that was the old, sub-pixel-``P`` renderer,
    kept as ``_legacy_subpixel_render_stamps`` for tests).

    composite: (..., G, G), G = pixel-integrated grid size for ``n_pix``
    (``node_geometry(n_pix)``'s ``node``).
    dx, dy: (...,) subpixel offset (in physical px, range [-0.5, 0.5)) of the
    true star position from the *rounded* stamp-center pixel used to extract
    the data stamp (see ``groups.py``).

    Index derivation: output pixel p (0..n_pix-1) samples ``E`` at continuous
    index ``g(p) = c_E + (p - (n_pix-1)/2 - dx) * oversample`` (``c_E =
    node_center_for_grid(G)``), which -- since ``c_E - (n_pix-1)/2*oversample
    == PAD_SAMPLES`` identically (CONTRACT_pixel_integrated_epsf.md) --
    simplifies to ``PAD_SAMPLES + oversample*p - dx*oversample``: the SAME
    closed form ``_axis_render_taps``/``axis_band`` build (module tests
    cross-check this function against that closed-form banded matrix); this
    is their direct, gather-based twin, kept for speed/clarity on the
    ordinary contiguous stamp path.
    """
    G = composite.shape[-1]
    p = jnp.arange(n_pix, dtype=composite.dtype)
    gx = PAD_SAMPLES + (p[None, :] - dx[..., None]) * oversample
    gy = PAD_SAMPLES + (p[None, :] - dy[..., None]) * oversample

    # Same "zero rather than repeat the edge value" convention as the legacy
    # renderer (Option 1a, SESSION_PLAN_20260808.md Problem 1/5a) -- validity
    # is a single joint check on the continuous coordinate actually sampled
    # (there is no sub-sample block anymore, so there is nothing to check
    # per-tap; see ``_bilinear_sample_grid``, which uses the same convention).
    in_bounds_x = (gx >= 0.0) & (gx <= float(G - 1))
    in_bounds_y = (gy >= 0.0) & (gy <= float(G - 1))

    x0 = jnp.clip(jnp.floor(gx).astype(jnp.int32), 0, G - 2)
    y0 = jnp.clip(jnp.floor(gy).astype(jnp.int32), 0, G - 2)
    fx = jnp.clip(gx - x0, 0.0, 1.0)
    fy = jnp.clip(gy - y0, 0.0, 1.0)

    flat = composite.reshape(-1, G, G)
    n_batch = flat.shape[0]
    b = jnp.arange(n_batch)[:, None, None]
    y0b, x0b = y0.reshape(n_batch, 1, n_pix), x0.reshape(n_batch, 1, n_pix)
    fy_b, fx_b = fy.reshape(n_batch, n_pix, 1), fx.reshape(n_batch, 1, n_pix)

    v00 = flat[b, y0b.reshape(n_batch, n_pix, 1), x0b.reshape(n_batch, 1, n_pix)]
    v01 = flat[b, y0b.reshape(n_batch, n_pix, 1), x0b.reshape(n_batch, 1, n_pix) + 1]
    v10 = flat[b, y0b.reshape(n_batch, n_pix, 1) + 1, x0b.reshape(n_batch, 1, n_pix)]
    v11 = flat[b, y0b.reshape(n_batch, n_pix, 1) + 1, x0b.reshape(n_batch, 1, n_pix) + 1]
    sampled = (
        v00 * (1 - fx_b) * (1 - fy_b)
        + v01 * fx_b * (1 - fy_b)
        + v10 * (1 - fx_b) * fy_b
        + v11 * fx_b * fy_b
    )  # (n_batch, n_pix, n_pix)
    valid = in_bounds_y.reshape(n_batch, n_pix, 1) & in_bounds_x.reshape(n_batch, 1, n_pix)
    sampled = jnp.where(valid, sampled, 0.0)
    stamps = sampled * (oversample ** 2)
    return stamps.reshape(composite.shape[:-2] + (n_pix, n_pix))


def _bilinear_sample_grid(grid: jnp.ndarray, gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
    """Bilinear sample ``grid (G,G)`` at fractional indices; OOB → 0 (Option 1a)."""
    G = grid.shape[-1]
    in_x = (gx >= 0.0) & (gx <= float(G - 1))
    in_y = (gy >= 0.0) & (gy <= float(G - 1))
    x0 = jnp.clip(jnp.floor(gx).astype(jnp.int32), 0, G - 2)
    y0 = jnp.clip(jnp.floor(gy).astype(jnp.int32), 0, G - 2)
    fx = jnp.clip(gx - x0.astype(gx.dtype), 0.0, 1.0)
    fy = jnp.clip(gy - y0.astype(gy.dtype), 0.0, 1.0)
    v00 = grid[y0, x0]
    v01 = grid[y0, x0 + 1]
    v10 = grid[y0 + 1, x0]
    v11 = grid[y0 + 1, x0 + 1]
    val = (
        v00 * (1 - fx) * (1 - fy)
        + v01 * fx * (1 - fy)
        + v10 * (1 - fx) * fy
        + v11 * fx * fy
    )
    return jnp.where(in_x & in_y, val, 0.0)


def render_physical_pixels_blocksum(
    composite: jnp.ndarray,
    ox: jnp.ndarray,
    oy: jnp.ndarray,
    *,
    valid: jnp.ndarray | None = None,
    oversample: int = OVERSAMPLE,
) -> jnp.ndarray:
    """Same physics as one ``render_stamps`` output pixel, at arbitrary centers
    (pixel-integrated ``E``: a single bilinear point-sample per pixel, scaled
    by ``oversample**2`` -- CONTRACT_pixel_integrated_epsf.md). Name kept for
    source compatibility even though there is no longer a sub-sample block
    sum to perform -- the whole point of ``E`` is that the render collapses
    to one point-sample, not ``oversample**2`` of them.

    ``composite``: ``(..., G, G)`` local pixel-integrated ePSF (flux fractions).
    ``ox``, ``oy``: ``(..., P)`` physical-pixel **center** offsets from the star
    photocenter (same sign convention as ``dx`` in ``render_stamps``: sample at
    pixel_offset - star_offset in the star frame, i.e. pass ``pix - x_star``).

    Returns ``(..., P)``. Optional ``valid`` (broadcastable to ox) zeroes pads.
    """
    os = int(oversample)
    lead = composite.shape[:-2]
    flat = composite.reshape((-1, composite.shape[-2], composite.shape[-1]))
    n_batch = flat.shape[0]
    ox_b = ox.reshape(n_batch, -1)
    oy_b = oy.reshape(n_batch, -1)
    P = ox_b.shape[1]
    # Grid center derived from the actual grid size (matches the square
    # path's node_geometry-based derivation), general in stamp_physical --
    # see node_center_for_grid's docstring.
    center = node_center_for_grid(composite.shape[-1])

    def _one(grid, ox_i, oy_i):
        gx = center + ox_i * os
        gy = center + oy_i * os
        return _bilinear_sample_grid(grid, gx, gy) * (os ** 2)

    out = jax.vmap(_one)(flat, ox_b, oy_b)
    out = out.reshape(lead + (P,))
    if valid is not None:
        out = out * valid
    return out


# ---------------------------------------------------------------------------
# Bit-exact, gather-free hot-path renderer: render_stamps is a separable
# banded matrix (a), the core-centroid recenter folds into the same matrix
# exactly (b), the core centroid itself needs no grid pass (c), and the
# renorm is a scalar (d). See dev/forward_epsf_wcs README/plan for the
# derivation; kept alongside (not replacing)
# render_stamps/recenter_grid_core/bilinear_shift_physical, which remain
# the test references and are still used by decode_epsf_base and the
# stage-1 dx-only local cache.
# ---------------------------------------------------------------------------


def _axis_render_taps(
    off: jnp.ndarray,
    *,
    n_pix: int,
    oversample: int,
    n_grid: int,
    pad,
    dtype,
    p_index: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Shared core of ``axis_band``: returns ``(col0, taps)``, ``taps`` a
    2-element list of ``(..., n_pix)`` weights -- pixel-integrated (``E``)
    rendering is a single bilinear point-sample per pixel, scaled by
    ``oversample`` per axis (``oversample**2`` once both axes are folded
    together -- CONTRACT_pixel_integrated_epsf.md: ``pixel = oversample**2 *
    bilinear(E, g)``). Output column ``col0[..., s] + c`` (``c`` in
    ``{0, 1}``) gets weight ``taps[c][..., s]`` for row ``s``. This only
    factors out the shared math so ``render_band`` can fold in a second
    bilinear shift without ever forming the ``(..., n_pix, n_grid)`` band
    first.

    Same ``s = pad - off*oversample; col0 = floor(s) + oversample*p``
    derivation as the old sub-pixel block-start (``pad`` is still
    ``PAD_SAMPLES`` for the contiguous default grid: the per-PIXEL centering
    term ``(n_pix-1)/2 * oversample`` folds into the same constant -- see
    CONTRACT_pixel_integrated_epsf.md). Validity is a single joint check on
    the continuous coordinate actually sampled (whole-pixel in/out, not
    per-tap -- there is no sub-sample block anymore to check per-tap; same
    convention as ``_bilinear_sample_grid``).

    ``p_index``, if given, replaces the implicit ``arange(n_pix)`` physical-
    pixel index with an arbitrary (possibly non-contiguous, per-leading-batch)
    **static** integer index array broadcastable against ``off[..., None]``
    (shape ``(*lead, P)`` for an ``off`` of shape ``(*lead,)``); ``n_pix`` is
    then unused. This is what lets the packed/irregular render path
    (``render_packed_pixels_banded``) reuse this exact closed form for
    arbitrary integer-lattice pixel lists instead of a contiguous stamp --
    see that function's docstring for why this is still exact (with
    ``pad -> node_center_for_grid(n_grid)`` there, not ``PAD_SAMPLES``).
    """
    off = jnp.asarray(off)
    off = off.astype(dtype)
    s = pad - off * oversample
    n0 = jnp.floor(s)
    f = (s - n0)[..., None]  # (*lead, 1), broadcasts over the n_pix/P axis
    if p_index is None:
        p = jnp.arange(n_pix, dtype=dtype)
    else:
        p = jnp.asarray(p_index).astype(dtype)
    col0 = n0[..., None] + oversample * p  # (*lead, n_pix) or (*lead, P)

    g = col0 + f  # continuous coordinate actually sampled by this pixel
    inb = ((g >= 0.0) & (g <= float(n_grid - 1))).astype(dtype)
    taps = [oversample * (1 - f) * inb, oversample * f * inb]
    return col0, taps


def axis_band(
    off: jnp.ndarray,
    *,
    n_pix: int = STAMP_PHYSICAL,
    oversample: int = OVERSAMPLE,
    n_grid: int = NODE_GRID_SIZE,
    pad=PAD_SAMPLES,
    dtype=None,
    p_index: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Closed-form per-axis render matrix ``A``, shape ``(..., n_pix, n_grid)``,
    s.t. ``render_stamps(grid, dx, dy) == einsum('...sg,...gh,...th->...st',
    axis_band(dy), grid, axis_band(dx))`` (module tests). ``off`` broadcasts
    over arbitrary leading batch dims.

    Pixel-integrated (``E``) equivalent of the reference construction::

        p = arange(n_pix); g = c_E + (p - (n_pix-1)/2 - off) * oversample
        inb = (g >= 0) & (g <= n_grid - 1)
        x0 = clip(floor(g), 0, n_grid - 2); f = clip(g - x0, 0, 1)
        W = zeros((n_pix, n_grid))
        W[p, x0] += oversample * (1 - f) * inb
        W[p, x0 + 1] += oversample * f * inb

    built directly as ``(..., n_pix, n_grid)`` (no dense intermediate) via 2
    closed-form terms per output row -- ``E`` needs only a single bilinear
    point-sample per pixel (module docstring / CONTRACT), unlike the old
    sub-pixel ``P`` renderer's ``oversample + 1``-tap block sum.

    ``p_index``: see ``_axis_render_taps`` -- generalizes the physical-pixel
    axis from a contiguous ``arange(n_pix)`` to an arbitrary static integer
    index array (packed/irregular render path); ``n_pix`` is unused then.
    """
    off = jnp.asarray(off)
    if dtype is None:
        dtype = off.dtype if jnp.issubdtype(off.dtype, jnp.floating) else jnp.float32
    col0, taps = _axis_render_taps(
        off, n_pix=n_pix, oversample=oversample, n_grid=n_grid, pad=pad, dtype=dtype,
        p_index=p_index,
    )
    j = jnp.arange(n_grid, dtype=dtype)
    band = taps[0][..., None] * (j == col0[..., None]).astype(dtype)
    for c in range(1, len(taps)):
        band = band + taps[c][..., None] * (j == (col0[..., None] + c)).astype(dtype)
    return band


def tap2_matrix(
    shift: jnp.ndarray,
    *,
    n_grid: int = NODE_GRID_SIZE,
    oversample: int = OVERSAMPLE,
    dtype=None,
) -> jnp.ndarray:
    """``(..., n_grid, n_grid)`` uniform bilinear shift matrix: ``out[i] =
    in[i - shift*oversample]``, zero outside ``[0, n_grid)``.

    Same convention as ``bilinear_shift_physical``'s 1D axis: each of the two
    taps is masked *independently* by whether its own column lands inside
    ``[0, n_grid)`` (unlike ``axis_band``'s ``axis_matrix``-style joint check
    on the continuous sample coordinate) -- see that function's docstring.
    """
    shift = jnp.asarray(shift)
    if dtype is None:
        dtype = shift.dtype if jnp.issubdtype(shift.dtype, jnp.floating) else jnp.float32
    shift = shift.astype(dtype)
    lead = shift.shape
    ii = jnp.arange(n_grid, dtype=dtype).reshape((1,) * len(lead) + (n_grid,))
    g = ii - shift[..., None] * oversample  # (*lead, n_grid), row index i
    x0 = jnp.floor(g)
    f = g - x0
    jj = jnp.arange(n_grid, dtype=dtype).reshape((1,) * len(lead) + (1, n_grid))
    x0e = x0[..., :, None]
    fe = f[..., :, None]
    low = (jj == x0e).astype(dtype) * (1 - fe)
    high = (jj == (x0e + 1)).astype(dtype) * fe
    return low + high


def render_band(
    off_render: jnp.ndarray,
    off_com: jnp.ndarray,
    *,
    n_pix: int = STAMP_PHYSICAL,
    oversample: int = OVERSAMPLE,
    n_grid: int = NODE_GRID_SIZE,
    pad=PAD_SAMPLES,
    dtype=None,
    p_index: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """``(..., n_pix, n_grid)`` render matrix with the COM recenter folded in --
    mathematically ``axis_band(off_render) @ tap2_matrix(-off_com)`` (module
    test ``test_render_band_matches_matmul_reference`` checks this identity
    directly), but built as one 3-tap closed form (pixel-integrated ``E``:
    2 render taps + 2 COM taps - 1) instead of an actual
    ``(..., n_pix, n_grid) @ (..., n_grid, n_grid)`` matmul.

    Both the render kernel (2 taps, ``_axis_render_taps``) and the COM shift
    (2 taps, same math as ``tap2_matrix``) share the "every sample lands at
    unit spacing" property, so folding them is a length
    ``len(taps) + 2 - 1`` discrete convolution of the two tap lists --
    never materializing tap2_matrix's ``(..., n_grid, n_grid)`` (same size as
    the node grid itself, ~4.3 MB/occupied-slot at production shapes, and
    -- like ``blend_field``'s corner sum -- expensive again once every extra
    additive term needs its own backward residual under vmap+grad).

    Matches ``render_stamps(recenter_grid_core(grid, n_iter=1), off_render...)``
    to fp64 roundoff (module tests) *after* the caller also divides by
    ``renorm_scalar`` -- this returns the unnormalized band (composing two
    bilinear resamplings is not itself bilinear, so the renorm cannot be
    skipped; see ``renorm_scalar``).

    ``p_index``: see ``_axis_render_taps``/``axis_band`` -- generalizes the
    physical-pixel axis to an arbitrary static integer index array. Used by
    ``render_packed_pixels_banded`` for the packed/irregular render path,
    where physical pixels are not a contiguous ``arange(n_pix)`` run.
    """
    off_render = jnp.asarray(off_render)
    off_com = jnp.asarray(off_com)
    if dtype is None:
        dtype = jnp.result_type(off_render, off_com, jnp.float32)
    col0, taps = _axis_render_taps(
        off_render, n_pix=n_pix, oversample=oversample, n_grid=n_grid, pad=pad, dtype=dtype,
        p_index=p_index,
    )
    off_com = off_com.astype(dtype)
    s_c = off_com * oversample
    n0c = jnp.floor(s_c)
    fc = (s_c - n0c)[..., None]  # (*lead, 1), broadcasts over the n_pix axis
    col_final = col0 + n0c[..., None]  # (*lead, n_pix)

    n_render_taps = len(taps)
    conv = [(1 - fc) * taps[0]]
    for d in range(1, n_render_taps):
        conv.append((1 - fc) * taps[d] + fc * taps[d - 1])
    conv.append(fc * taps[n_render_taps - 1])

    j = jnp.arange(n_grid, dtype=dtype)
    band = conv[0][..., None] * (j == col_final[..., None]).astype(dtype)
    for d in range(1, len(conv)):
        band = band + conv[d][..., None] * (j == (col_final[..., None] + d)).astype(dtype)
    return band


def _render_band_via_matmul(
    off_render: jnp.ndarray,
    off_com: jnp.ndarray,
    *,
    n_pix: int = STAMP_PHYSICAL,
    oversample: int = OVERSAMPLE,
    n_grid: int = NODE_GRID_SIZE,
    pad: int = PAD_SAMPLES,
    dtype=None,
) -> jnp.ndarray:
    """Reference form of ``render_band`` as a literal matmul (used only by
    ``test_render_band_matches_matmul_reference`` to cross-check the closed
    form above; not used on the hot path, since materializing ``tap2_matrix``
    is exactly the memory cost ``render_band`` avoids)."""
    off_render = jnp.asarray(off_render)
    off_com = jnp.asarray(off_com)
    if dtype is None:
        dtype = jnp.result_type(off_render, off_com, jnp.float32)
    A = axis_band(off_render, n_pix=n_pix, oversample=oversample, n_grid=n_grid, pad=pad, dtype=dtype)
    T = tap2_matrix(-off_com, n_grid=n_grid, oversample=oversample, dtype=dtype)
    return jnp.einsum("...sg,...gh->...sh", A, T)


def node_blend_weights(
    i0: jnp.ndarray,
    j0: jnp.ndarray,
    wy: jnp.ndarray,
    wx: jnp.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    dtype=None,
) -> jnp.ndarray:
    """``(n_occ, n_rows, n_cols)`` bilinear blend weight tensor: the *dense*
    equivalent of ``blend_field``'s per-slot corner lookup, built via broadcast
    equality (no gather) so the local node blend can be one contraction
    (``jnp.einsum('nrc,rcGH->nGH', Wn, node_field)``) instead of ``blend_field``'s
    nested weighted-sum of 4 corners.

    This matters under ``vmap`` + reverse-mode AD: each extra large-broadcast
    term chained with ``+`` (as in ``blend_field``'s ``(1-wy_)*(...) + wy_*(...)``
    tree) costs its own full ``(n_occ, G, G)``-sized backward residual, while a
    single ``einsum`` contraction does not (measured ~3.8 GB vs ~6.3 GB for the
    same 4-corner blend at production shapes -- see the render hot-path plan).
    Only worth it because the *real* ``blend_field`` output is what gets
    contracted against here (``n_rows``/``n_cols`` are small, e.g. 2), not a
    reason to ever build a ``(n_occ, G, G)``-sized one-hot analog of this.
    """
    i0 = jnp.asarray(i0)
    j0 = jnp.asarray(j0)
    wy = jnp.asarray(wy)
    wx = jnp.asarray(wx)
    if dtype is None:
        dtype = wy.dtype
    ii = jnp.arange(n_rows, dtype=i0.dtype)
    jj = jnp.arange(n_cols, dtype=j0.dtype)
    wy_row = (ii[None, :] == i0[:, None]).astype(dtype) * (1 - wy)[:, None]
    wy_row = wy_row + (ii[None, :] == (i0 + 1)[:, None]).astype(dtype) * wy[:, None]
    wx_col = (jj[None, :] == j0[:, None]).astype(dtype) * (1 - wx)[:, None]
    wx_col = wx_col + (jj[None, :] == (j0 + 1)[:, None]).astype(dtype) * wx[:, None]
    return wy_row[:, :, None] * wx_col[:, None, :]


def node_moments(node_field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(..., G, G) -> (total, xmom, ymom)`` each ``(...)``.

    The three scalar functionals ``core_centroid_xy`` needs, computed once per
    node field (no per-occupied-slot ``G×G`` reduction -- the per-slot core
    centroid is then just a cheap bilinear blend of these three scalars, same
    weights as the ordinary node blend). The Gaussian core window is identical
    on every node, so blending moments equals the moments of the blended grid.
    Last axis is detector-x, second-to-last is detector-y. Moments are in
    physical pixels (``node_coord_1d``).
    """
    g_size = int(node_field.shape[-1])
    w = core_gaussian_weight(g_size, dtype=node_field.dtype)
    weighted = node_field * w
    coord = node_coord_1d(dtype=node_field.dtype, n_grid=g_size)
    total = jnp.sum(weighted, axis=(-2, -1))
    xmom = jnp.sum(weighted * coord, axis=(-2, -1))
    ymom = jnp.sum(weighted * coord[:, None], axis=(-2, -1))
    return total, xmom, ymom


def dilation_generator(node_field: jnp.ndarray) -> jnp.ndarray:
    """Ungauged scale-derivative of ``(..., G, G)``: ``D[P] = 2P + x dP/dx + y dP/dy``.

    First-order generator of the flux-preserving dilation ``P_a(u) = a^-2 P(u/a)``:

        d/da P_a |_(a=1) = -(2P + u.grad P) = -D[P],   so   P_a ~ P - (a - 1) D[P].

    Hence rendering ``P + q D[P]`` is rendering ``P`` dilated by ``a = 1 - q``:
    **positive q means a NARROWER PSF.** The chromatic weight is
    ``q_i = (c_i - c_ref) * eps``, so with red stars narrower than blue (measured)
    ``eps`` is POSITIVE, about ``+0.020`` per magnitude of BP-RP. Note this is the
    opposite sign to the ``+0.79 %`` "blue is larger" figure quoted in the residual
    analysis, which used the outward-displacement convention ``d = eps_analysis * u``;
    the two are related by ``q = -eps_analysis``.

    Written in INDEX space on purpose. With ``x_j = (j - center) / OVERSAMPLE`` and a
    central difference ``dP/dx ~ OVERSAMPLE * (P[j+1] - P[j-1]) / 2``, the two
    ``OVERSAMPLE`` factors cancel exactly:

        D[P][i,j] = 2 P[i,j] + (j-c) (P[i,j+1]-P[i,j-1])/2 + (i-c) (P[i+1,j]-P[i-1,j])/2

    so there is no oversampling factor left to get wrong. Last axis is detector-x,
    second-to-last is detector-y, matching ``node_moments``.

    Ungauged: use ``chroma_dilation_field`` in the model. This raw form exists so the
    unit test can compare it against the analytic scale derivative without a gauge in
    the way.
    """
    g_size = int(node_field.shape[-1])
    center = node_center_for_grid(g_size)
    ax = jnp.arange(g_size, dtype=node_field.dtype) - center
    # jnp.gradient with unit spacing is the central difference interior, one-sided at
    # the two edge samples. The ePSF is ~0 there (58-sample grid, 13 px core), so the
    # edge convention is immaterial; using it avoids hand-rolled padding.
    d_dx = jnp.gradient(node_field, axis=-1)
    d_dy = jnp.gradient(node_field, axis=-2)
    return 2.0 * node_field + ax * d_dx + ax[:, None] * d_dy


def chroma_dilation_field(
    node_field: jnp.ndarray, *, weight_grid: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Gauged chromatic dilation direction for ``(..., G, G)``. Two hard gauges.

    (1) Zero grid sum, mirroring gauge (1) of ``decode_epsf_modes``. Not required for
    correctness -- ``renorm_scalar`` and the per-stamp flux solve each absorb a net
    flux change, so a wrong overall scale on ``D`` is unobservable in any rendered
    stamp. That is exactly why it is worth doing: a factor-of-two slip in the ``2P``
    term would otherwise pass every end-to-end test silently. Gauging makes the
    parameterization say what it means.

    (2) Project off ``node_field`` in ``canonical_mode_weight_grid``'s fixed metric,
    as ``decode_epsf_modes`` gauge (2) does. This one IS needed. The base-parallel
    part of ``D[P]`` is a pure per-star template rescale, and the flux solve profiles
    a per-(group, frame, member) amplitude out of the residual, so that component is
    exactly degenerate with flux for every group shape. Left in, ``eps`` owns a flat
    direction: Adam random-walks along it, the loss never notices, and the reported
    coefficient becomes uninterpretable. Removing it also makes ``eps`` directly
    comparable with the residual-stack measurement, which fitted the dilation
    alongside a free flux column.

    The projection reference is stop-gradient'd so the gauge stays a pure
    reparameterization and does not couple a spurious gradient back into
    ``epsf_base_raw``; ``D`` itself remains differentiable in the ePSF leaves, which
    is correct, since the chromatic perturbation really is a dilation *of* the fitted
    ePSF.
    """
    raw = dilation_generator(node_field)
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(int(node_field.shape[-1]))
    gauged = raw - jnp.mean(raw, axis=(-2, -1), keepdims=True)
    ref = jax.lax.stop_gradient(node_field)
    ref = ref - jnp.mean(ref, axis=(-2, -1), keepdims=True)
    den = jnp.sum(weight_grid * ref ** 2, axis=(-2, -1), keepdims=True) + 1e-12
    num = jnp.sum(gauged * weight_grid * ref, axis=(-2, -1), keepdims=True)
    # ``ref`` is zero-sum, so subtracting a multiple of it preserves gauge (1) exactly.
    return gauged - (num / den) * ref


# ---------------------------------------------------------------------------
# Colour-AFFINE extension (C1): anisotropic stretch + 45-degree shear, and an
# optional flux-neutral "kurtosis" (core/wing redistribution) generator.  Each
# is a first-order generator of a flux-preserving *linear* warp of (x, y),
# built with the exact same index-space finite-difference convention as
# ``dilation_generator`` (so the two OVERSAMPLE factors cancel identically --
# see that function's docstring) and gauged with the same two projections as
# ``chroma_dilation_field``.  Written as fresh, self-contained functions
# (rather than refactoring ``dilation_generator``/``chroma_dilation_field`` to
# share code) so the pre-existing dilation-only path is byte-for-byte
# unchanged -- see loss.CHROMA_LEAVES / loss.CHROMA_AFFINE_LEAVES docstrings.
#
# Derivation (matches ``dev/forward_epsf_wcs/diagnostics/wfix_residual_atlas.py``'s
# ``build_extended_basis``, which independently fits these same two shapes to
# real residuals): for a diagonal area-preserving scale diag(1+a, 1-a),
#     d/da P|_(a=0) = -(P + x Px) + (P + y Py) = -(x Px - y Py),
# so the un-negated generator is ``x Px - y Py`` (x-stretch positive for
# positive coefficient, opposite sign to ``dilation_generator``'s "positive
# means narrower" convention -- same "pick a sign, document it" approach).
# For the symmetric shear (x, y) -> (x + s y, y + s x),
#     d/ds P|_(s=0) = -(y Px + x Py),
# so the un-negated generator is ``x Py + y Px``.
# ---------------------------------------------------------------------------


def aniso_generator(node_field: jnp.ndarray) -> jnp.ndarray:
    """Ungauged anisotropic-stretch generator ``x*Px - y*Py`` for ``(..., G, G)``.

    Index-space, same convention as ``dilation_generator``: ``x`` is the last
    (detector-x) axis, ``y`` the second-to-last (detector-y) axis. Ungauged:
    use ``chroma_aniso_field`` in the model.
    """
    g_size = int(node_field.shape[-1])
    center = node_center_for_grid(g_size)
    ax = jnp.arange(g_size, dtype=node_field.dtype) - center
    d_dx = jnp.gradient(node_field, axis=-1)
    d_dy = jnp.gradient(node_field, axis=-2)
    return ax * d_dx - ax[:, None] * d_dy


def shear_generator(node_field: jnp.ndarray) -> jnp.ndarray:
    """Ungauged 45-degree shear generator ``x*Py + y*Px`` for ``(..., G, G)``.

    Same axis/index-space convention as ``dilation_generator``/``aniso_generator``.
    Ungauged: use ``chroma_shear_field`` in the model.
    """
    g_size = int(node_field.shape[-1])
    center = node_center_for_grid(g_size)
    ax = jnp.arange(g_size, dtype=node_field.dtype) - center
    d_dx = jnp.gradient(node_field, axis=-1)
    d_dy = jnp.gradient(node_field, axis=-2)
    return ax[:, None] * d_dx + ax * d_dy


def chroma_aniso_field(
    node_field: jnp.ndarray, *, weight_grid: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Gauged chromatic anisotropic-stretch direction. Same two gauges as
    ``chroma_dilation_field`` (zero grid sum; projected off ``node_field`` in
    ``canonical_mode_weight_grid``'s fixed metric so the base-parallel,
    flux-degenerate component is removed) -- see that function's docstring
    for why both matter.
    """
    raw = aniso_generator(node_field)
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(int(node_field.shape[-1]))
    gauged = raw - jnp.mean(raw, axis=(-2, -1), keepdims=True)
    ref = jax.lax.stop_gradient(node_field)
    ref = ref - jnp.mean(ref, axis=(-2, -1), keepdims=True)
    den = jnp.sum(weight_grid * ref ** 2, axis=(-2, -1), keepdims=True) + 1e-12
    num = jnp.sum(gauged * weight_grid * ref, axis=(-2, -1), keepdims=True)
    return gauged - (num / den) * ref


def chroma_shear_field(
    node_field: jnp.ndarray, *, weight_grid: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Gauged chromatic shear direction. Same two gauges as ``chroma_dilation_field``
    (see that function's docstring); the raw generator is ``shear_generator``.
    """
    raw = shear_generator(node_field)
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(int(node_field.shape[-1]))
    gauged = raw - jnp.mean(raw, axis=(-2, -1), keepdims=True)
    ref = jax.lax.stop_gradient(node_field)
    ref = ref - jnp.mean(ref, axis=(-2, -1), keepdims=True)
    den = jnp.sum(weight_grid * ref ** 2, axis=(-2, -1), keepdims=True) + 1e-12
    num = jnp.sum(gauged * weight_grid * ref, axis=(-2, -1), keepdims=True)
    return gauged - (num / den) * ref


def kurt_generator(node_field: jnp.ndarray) -> jnp.ndarray:
    """Ungauged flux-neutral core/wing redistribution generator ``r^2*P - <r^2 P>/<P> * P``.

    ``r`` is the node-grid radius in PHYSICAL pixels (``node_coord_1d``), matching
    ``wfix_residual_atlas.build_extended_basis``'s ``kurt`` basis vector exactly
    (same ``r2*T - ratio*T`` construction, evaluated here on the node grid instead
    of a stacked template). Already approximately zero-sum by construction (see
    ``chroma_kurt_field`` for the exact gauge); ungauged form exists so the unit
    test can check the flux-neutral property directly.
    """
    g_size = int(node_field.shape[-1])
    coord = node_coord_1d(dtype=node_field.dtype, n_grid=g_size)
    r2 = coord[None, :] ** 2 + coord[:, None] ** 2
    mean_field = jnp.mean(node_field, axis=(-2, -1), keepdims=True)
    mean_r2field = jnp.mean(r2 * node_field, axis=(-2, -1), keepdims=True)
    ratio = jnp.where(mean_field != 0, mean_r2field / (mean_field + 1e-30), 0.0)
    return r2 * node_field - ratio * node_field


# --- chromatic HALO -----------------------------------------------------------
# Measured 2026-09-16/17 on s0024/c2/k2: the colour-dependent excess that the free
# per-stamp pedestal was absorbing is the STAR'S OWN light (its colour slope scales
# as flux^(1.02 +- 0.06) at fixed footprint, where a background would be flux^0) and
# it lies on an r^-2 profile -- measured on 1469 bright contributors out to 6.75 px,
# every annulus positive at S/N 12-49: r^-2.000 +- 0.023 over 1.75-6.75 px, steepening
# to about r^-2.4 outside 2.75 px. Both a flat background (r^0) and the PSF itself
# (r^-4) are excluded outright: the profile falls 15.5x over a factor 3.86 in radius,
# where r^-4 would fall 221x. The ePSF falls as r^-4, and EVERY generator in
# ``loss.CHROMA_FIELD_GENERATORS`` is a differential operator applied to the base, so
# every one of them inherits that r^-4 and none can represent an r^-1.4 excess. Hence
# a generator that is NOT a functional of the base: a fixed radial profile with a
# trainable per-node amplitude.
#
# The index is a measured physical constant, not a tuning knob, so it lives here and
# is overridden explicitly (``--chroma-halo-index``) and recorded in the run meta.
CHROMA_HALO_INDEX = 2.0
CHROMA_HALO_CORE_PX = 1.0
# Whether to additionally impose the zero-sum (flux-neutral) gauge on the halo. Default
# False -- see chroma_halo_field for why, and CHROMA_HALO_20260917.md for the flux
# bookkeeping: an r^-2 halo is log-divergent, so inside a 13-px stamp a genuine
# core->wing TRANSFER looks like pure addition (the compensating core deficit is only
# ~2% of the wing addition at these radii). Exposed as a flag so the claim is testable
# rather than asserted.
CHROMA_HALO_FLUX_NEUTRAL = False


def chroma_halo_profile(
    n_grid: int, *, dtype=jnp.float32, index: float | None = None,
    core_px: float | None = None,
) -> jnp.ndarray:
    """``(n_grid, n_grid)`` fixed profile ``(max(r, core)/core)^-index``, r in physical px.

    Flat inside ``core_px`` so the profile stays finite at the origin; the core region
    carries almost no weight after the base-orthogonal gauge below, which is where the
    PSF-shaped part of any excess is removed.
    """
    idx = CHROMA_HALO_INDEX if index is None else float(index)
    rc = CHROMA_HALO_CORE_PX if core_px is None else float(core_px)
    if rc <= 0:
        raise ValueError(f"core_px must be positive, got {rc}")
    coord = node_coord_1d(dtype=dtype, n_grid=int(n_grid))
    r = jnp.sqrt(coord[None, :] ** 2 + coord[:, None] ** 2)
    return (jnp.maximum(r, rc) / rc) ** (-idx)


def chroma_halo_field(
    node_field: jnp.ndarray, *, weight_grid: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Additive chromatic halo direction: a fixed radial profile, base-orthogonalised.

    ONE gauge, not the usual two. The base-parallel component is projected off, for
    the same reason it is for dilation/aniso/shear/kurt: that direction is exactly
    what the free per-stamp flux already absorbs, so leaving it in would make the
    halo's amplitude degenerate with the photometry.

    Flux-neutrality (gauge 1 elsewhere) is deliberately NOT imposed by default, and
    ``CHROMA_HALO_FLUX_NEUTRAL`` turns it on so that choice can be tested rather than
    asserted. The halo IS net extra light inside the footprint -- that is the
    measurement -- and a zero-sum version cannot represent it: subtracting the mean
    drives the profile NEGATIVE beyond ~3 px, while the measured blue-minus-red profile
    is positive in every annulus out to 6 px. That is not a defect of the data but of
    the constraint: an r^-2 halo carries as much flux between 6 and 36 px as between 1
    and 6 px, so the deficit that balances it lies almost entirely outside the stamp. It is therefore the only chroma generator that
    changes a stamp's integrated model flux at fixed ``flux`` parameter, which is
    precisely why it is identifiable only once the support is wide enough for an
    r^-index profile to look different from a constant (see
    ``diagnostics/chroma_halo_profile.py``).
    """
    g_size = int(node_field.shape[-1])
    prof = chroma_halo_profile(g_size, dtype=node_field.dtype)
    prof = jnp.broadcast_to(prof, node_field.shape)
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(g_size)
    if CHROMA_HALO_FLUX_NEUTRAL:
        prof = prof - jnp.mean(prof, axis=(-2, -1), keepdims=True)
    ref = jax.lax.stop_gradient(node_field)
    den = jnp.sum(weight_grid * ref ** 2, axis=(-2, -1), keepdims=True) + 1e-12
    num = jnp.sum(prof * weight_grid * ref, axis=(-2, -1), keepdims=True)
    return prof - (num / den) * ref


# --- global colour model (chroma_g8) generators -------------------------------------
# Added 2026-09-24 for the 8-parameter global colour model (see
# docs/DILATION_DISCREPANCY_20260923.md "Chosen configuration"). Unlike the node-field
# terms above, these are weighted per STAR (colour x a function of the distance and
# direction to the camera optical axis), not per node; the generators themselves are
# still linear functionals of the local ePSF, so they fold into the same node blend.
# All are written in PHYSICAL pixels (last axis = detector x) and gauged the same two
# ways as ``chroma_dilation_field``: zero grid sum and projected off the base in the
# canonical metric (the base-parallel part is a pure flux rescale, degenerate with
# the per-star flux).


def _gauge_off_base(raw: jnp.ndarray, node_field: jnp.ndarray,
                    weight_grid: jnp.ndarray | None = None) -> jnp.ndarray:
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(int(node_field.shape[-1]))
    gauged = raw - jnp.mean(raw, axis=(-2, -1), keepdims=True)
    ref = jax.lax.stop_gradient(node_field)
    ref = ref - jnp.mean(ref, axis=(-2, -1), keepdims=True)
    den = jnp.sum(weight_grid * ref ** 2, axis=(-2, -1), keepdims=True) + 1e-12
    num = jnp.sum(gauged * weight_grid * ref, axis=(-2, -1), keepdims=True)
    return gauged - (num / den) * ref


def _phys_xy(node_field: jnp.ndarray):
    ax = node_coord_1d(dtype=node_field.dtype, n_grid=int(node_field.shape[-1]))
    return ax[None, :], ax[:, None]          # X varies along the last axis, Y along -2


def _grad_phys(node_field: jnp.ndarray):
    h = 1.0 / OVERSAMPLE
    return jnp.gradient(node_field, h, axis=-1), jnp.gradient(node_field, h, axis=-2)


def kurt_plain_generator(node_field: jnp.ndarray) -> jnp.ndarray:
    """Ungauged plain kurtosis ``P * (rho^2 - <rho^2>_P)``, rho in physical px.

    Moves light between core and wings. Because the ePSF falls roughly as r^-4, the
    wing part P*rho^2 falls as ~r^-2 -- the measured chromatic halo slope -- so this
    one term carries both the core deficit and the halo. It is NOT ``chroma_kurt``,
    whose base-orthogonal gauge (applied BEFORE removing the mean) gives it a
    negative outer lobe.
    """
    X, Y = _phys_xy(node_field)
    rho2 = X ** 2 + Y ** 2
    m = jnp.sum(node_field * rho2, axis=(-2, -1), keepdims=True) / (
        jnp.sum(node_field, axis=(-2, -1), keepdims=True) + 1e-12)
    return node_field * (rho2 - m)


def blur_generator(node_field: jnp.ndarray) -> jnp.ndarray:
    """Ungauged blur: the Laplacian of P in physical px^-2 (coefficient in px^2)."""
    px, py = _grad_phys(node_field)
    h = 1.0 / OVERSAMPLE
    return jnp.gradient(px, h, axis=-1) + jnp.gradient(py, h, axis=-2)


def _displacement_generator(node_field, dx, dy):
    """First-order change of P under the displacement field d: ``-(d . grad P)``."""
    px, py = _grad_phys(node_field)
    return -(dx * px + dy * py)


def trefoil_a_generator(node_field: jnp.ndarray) -> jnp.ndarray:
    """Trefoil displacement d = (x^2 - y^2, -2xy) (divergence-free, flux-preserving)."""
    X, Y = _phys_xy(node_field)
    return _displacement_generator(node_field, X ** 2 - Y ** 2, -2.0 * X * Y)


def trefoil_b_generator(node_field: jnp.ndarray) -> jnp.ndarray:
    """Trefoil partner d = (2xy, x^2 - y^2), the first rotated by 30 degrees."""
    X, Y = _phys_xy(node_field)
    return _displacement_generator(node_field, 2.0 * X * Y, X ** 2 - Y ** 2)


def chroma_kurt_plain_field(node_field, *, weight_grid=None):
    return _gauge_off_base(kurt_plain_generator(node_field), node_field, weight_grid)


def chroma_blur_field(node_field, *, weight_grid=None):
    return _gauge_off_base(blur_generator(node_field), node_field, weight_grid)


def chroma_trefoil_a_field(node_field, *, weight_grid=None):
    return _gauge_off_base(trefoil_a_generator(node_field), node_field, weight_grid)


def chroma_trefoil_b_field(node_field, *, weight_grid=None):
    return _gauge_off_base(trefoil_b_generator(node_field), node_field, weight_grid)


# Raw-P gauge (2026-09-24). ``_gauge_off_base`` projects off the MEAN-REMOVED base
# ``P - mean(P)``, which puts ``alpha * mean(P)`` back as a flat sheet over the whole
# node grid: every term then carries a colour-dependent pedestal the flux solve cannot
# absorb, and the near-collinear blur/dilation pair can trade shape for sheet. Projecting
# off P itself (as ``chroma_halo_field`` does) removes exactly what the per-star flux
# absorbs and nothing else. Terms are no longer exactly flux-neutral; flux takes that up.
def _gauge_off_raw_base(raw: jnp.ndarray, node_field: jnp.ndarray,
                        weight_grid: jnp.ndarray | None = None) -> jnp.ndarray:
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(int(node_field.shape[-1]))
    ref = jax.lax.stop_gradient(node_field)
    den = jnp.sum(weight_grid * ref ** 2, axis=(-2, -1), keepdims=True) + 1e-12
    num = jnp.sum(raw * weight_grid * ref, axis=(-2, -1), keepdims=True)
    return raw - (num / den) * ref


def chroma_kurt_plain_field_raw(node_field, *, weight_grid=None):
    return _gauge_off_raw_base(kurt_plain_generator(node_field), node_field, weight_grid)


def chroma_blur_field_raw(node_field, *, weight_grid=None):
    return _gauge_off_raw_base(blur_generator(node_field), node_field, weight_grid)


def chroma_dilation_field_raw(node_field, *, weight_grid=None):
    return _gauge_off_raw_base(dilation_generator(node_field), node_field, weight_grid)


def chroma_trefoil_a_field_raw(node_field, *, weight_grid=None):
    return _gauge_off_raw_base(trefoil_a_generator(node_field), node_field, weight_grid)


def chroma_trefoil_b_field_raw(node_field, *, weight_grid=None):
    return _gauge_off_raw_base(trefoil_b_generator(node_field), node_field, weight_grid)


def chroma_aniso_field_raw(node_field, *, weight_grid=None):
    return _gauge_off_raw_base(aniso_generator(node_field), node_field, weight_grid)


def chroma_shear_field_raw(node_field, *, weight_grid=None):
    return _gauge_off_raw_base(shear_generator(node_field), node_field, weight_grid)


# ---------------------------------------------------------------------------
# Colour radial-profile family (2026-10-08, branch colour-radial-20261008).
#
# Generators for the named chroma_g8 extras rb{j}_0 / rb{j}_r / rq{j} (radial profile) and rc{j} (coma); see
# loss._chroma_g8_slot_terms. Reference definition: dev_runs/colour_shoulder_20261008/step0_projection/code/rb.py.
#
#   radial, mode 'add' (default): G_j = gauge( (B_j(rho) - mean_nodegrid(B_j)) / sum_nodegrid(B_j) )   ADDITIVE, flux-neutral,
#                                 unit-flux normalised (coefficient 1 = the whole PSF flux moved into bump j)
#   radial, mode 'mult'         : G_j = gauge( P0 (B_j(rho) - m_j) ),  m_j = sum(P0 B_j) / sum(P0)
#   coma (always multiplicative): G^c_ja = gauge_c( P0 C_j(rho) rho cos(theta) ) = P0 C_j X,  G^c_jb = ... Y
# gauge = ``_gauge_off_raw_base`` (project off P0 in the canonical weight metric); gauge_c = Gram-Schmidt off P0,
# dP0/dx, dP0/dy (the shift owns the first-order centroid). rho = physical-px radius on the oversampled node grid
# (``_phys_xy``), theta the stamp azimuth in detector x/y (x along the last axis). The factor rho makes the coma
# generator smooth at rho = 0.
#
# Bases: "bumps in a warped coordinate" as in rb.py: with knots k_0 < ... < k_{J-1} (J >= 2) the warped coordinate is
# s(rho) = np.interp(rho, knots, arange(J)) (so the knots are equidistant in s and s CLAMPS outside [k_0, k_{J-1}]),
# and B_j(rho) = bspline3(s - j), j = 0..J-1, with bspline3 the centred cubic B-spline (support |s - j| < 2, peak 2/3).
# There are J bumps, indexed j = 1..J in the extras names (rb1 = the bump at k_0). Beyond the last knot the bumps keep
# their end values (a constant plateau: B_J = 2/3, B_{J-1} = 1/6, others 0), and likewise inside the first knot, so
# corners (rho > 5.5 px) and the star centre behave. The bumps do NOT sum to 1 near the ends (unlike a clamped basis).
# Defaults: radial knots 0, 0.7, 1.5, 3.0, 5.5 px (J = 5); coma knots 0.8, 2.2, 5.0 px (K = 3).
RADIAL_KNOTS_DEFAULT = (0.0, 0.7, 1.5, 3.0, 5.5)
COMA_KNOTS_DEFAULT = (0.8, 2.2, 5.0)
_RADIAL_KNOTS = RADIAL_KNOTS_DEFAULT
_COMA_KNOTS = COMA_KNOTS_DEFAULT


def parse_radial_knots(knots) -> tuple:
    """Knots as a validated tuple of floats (accepts a 'a,b,c' string or a sequence)."""
    if isinstance(knots, str):
        knots = [v for v in knots.replace(" ", "").split(",") if v]
    k = tuple(float(v) for v in knots)
    if len(k) < 2 or any(b <= a for a, b in zip(k, k[1:])) or k[0] < 0:
        raise ValueError(f"radial knots must be >= 0, strictly increasing and at least 2 long, got {k}")
    return k


def set_radial_knots(knots) -> tuple:
    """Set the trace-time knot vector of the radial profiles (idempotent)."""
    global _RADIAL_KNOTS
    _RADIAL_KNOTS = parse_radial_knots(knots)
    return _RADIAL_KNOTS


def get_radial_knots() -> tuple:
    return _RADIAL_KNOTS


def set_coma_knots(knots) -> tuple:
    """Set the trace-time knot vector of the coma profiles (idempotent)."""
    global _COMA_KNOTS
    _COMA_KNOTS = parse_radial_knots(knots)
    return _COMA_KNOTS


def get_coma_knots() -> tuple:
    return _COMA_KNOTS


# Radial mode (trace-time constant): 'add' or 'mult' (above). Applies to the radial profiles only; coma is always mult.
RADIAL_MODES = ("mult", "add")
_RADIAL_MODE = "add"


def set_radial_mode(mode: str) -> str:
    global _RADIAL_MODE
    if mode not in RADIAL_MODES:
        raise ValueError(f"radial mode must be one of {RADIAL_MODES}, got {mode!r}")
    _RADIAL_MODE = mode
    return mode


def get_radial_mode() -> str:
    return _RADIAL_MODE


def n_radial_basis(knots=None) -> int:
    return len(_RADIAL_KNOTS if knots is None else parse_radial_knots(knots))


def n_coma_basis(knots=None) -> int:
    return len(_COMA_KNOTS if knots is None else parse_radial_knots(knots))


def _bspline3(s):
    a = np.abs(s)
    return np.where(a < 1, 2 / 3 - a ** 2 + a ** 3 / 2, np.where(a < 2, (2 - a) ** 3 / 6, 0.0))


def radial_bspline_basis(rho, knots=None) -> np.ndarray:
    """Warped-coordinate cubic B-spline bumps ``(J, *rho.shape)`` (numpy); see the section comment."""
    kn = np.asarray(parse_radial_knots(_RADIAL_KNOTS if knots is None else knots), dtype=np.float64)
    s = np.interp(np.asarray(rho, dtype=np.float64), kn, np.arange(len(kn)))
    return np.stack([_bspline3(s - j) for j in range(len(kn))])


_RADIAL_GRID_CACHE: dict = {}


def _coord_np(g_size: int) -> np.ndarray:
    """``node_coord_1d`` in pure numpy: these constants are first built inside jit traces, where a jax array
    cannot be converted (TracerArrayConversionError)."""
    return (np.arange(int(g_size), dtype=np.float64) - node_center_for_grid(int(g_size))) / OVERSAMPLE


def _radial_grid_consts(g_size: int, knots: tuple):
    """(basis (J, G, G), X (1, G), Y (G, 1)) numpy constants for knots ``knots`` on the G x G node grid."""
    key = (g_size, knots)
    if key not in _RADIAL_GRID_CACHE:
        ax = _coord_np(g_size)
        X, Y = ax[None, :], ax[:, None]
        _RADIAL_GRID_CACHE[key] = (radial_bspline_basis(np.hypot(X, Y), knots), X, Y)
    return _RADIAL_GRID_CACHE[key]


def radial_generator(node_field: jnp.ndarray, j: int) -> jnp.ndarray:
    """Ungauged flux-neutral radial profile, j = 1-based: ``(B_j - mean(B_j)) / sum(B_j)`` (add) or ``P0 (B_j - m_j)`` (mult)."""
    B, _, _ = _radial_grid_consts(int(node_field.shape[-1]), _RADIAL_KNOTS)
    if not 1 <= j <= B.shape[0]:
        raise ValueError(f"radial basis index {j} outside 1..{B.shape[0]} for knots {_RADIAL_KNOTS}")
    Bj = jnp.asarray(B[j - 1], node_field.dtype)
    if _RADIAL_MODE == "add":
        # unit flux: B_j / sum(B_j) over the node grid minus the uniform 1/G^2, so a coefficient of 1 moves the whole PSF
        # flux into (out of) bump j; typical fitted values are ppt (as in rb.py's ring_j / sum(ring_j) - T)
        return jnp.broadcast_to((Bj - jnp.mean(Bj)) / (jnp.sum(Bj) + 1e-12), node_field.shape)
    m = jnp.sum(node_field * Bj, axis=(-2, -1), keepdims=True) / (jnp.sum(node_field, axis=(-2, -1), keepdims=True) + 1e-12)
    return node_field * (Bj - m)


def radial_coma_generator(node_field: jnp.ndarray, j: int, which: str) -> jnp.ndarray:
    """Ungauged coma partner ``P0 C_j(rho) rho cos(theta) = P0 C_j X`` ('a') or ``P0 C_j Y`` ('b'); always
    multiplicative, with the coma knots. The factor rho keeps it smooth at rho = 0."""
    B, X, Y = _radial_grid_consts(int(node_field.shape[-1]), _COMA_KNOTS)
    if not 1 <= j <= B.shape[0]:
        raise ValueError(f"coma basis index {j} outside 1..{B.shape[0]} for coma knots {_COMA_KNOTS}")
    if which not in ("a", "b"):
        raise ValueError(f"coma partner must be 'a' or 'b', got {which!r}")
    return node_field * jnp.asarray(B[j - 1] * (X if which == "a" else Y), node_field.dtype)


def chroma_radial_field_raw(node_field, j, *, weight_grid=None):
    return _gauge_off_raw_base(radial_generator(node_field, j), node_field, weight_grid)


def _orthogonalize_off_shift(raw, node_field, weight_grid=None):
    """Gram-Schmidt ``raw`` against P0, dP0/dx, dP0/dy (in that order) in the canonical weight metric.

    The colour shift moves light along ``n . grad P0 = P0'(rho) cos(theta - phi)``, which for a near-round P0
    is almost inside the coma span ``P0 C_j(rho) rho cos(theta)``; projecting the coma off BOTH gradients (both
    components, since P0 is not exactly round) leaves the shift as sole owner of the first-order centroid."""
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(int(node_field.shape[-1]))
    ref = jax.lax.stop_gradient(node_field)
    px, py = _grad_phys(ref)
    basis = []
    for v in (ref, px, py):
        for b in basis:
            v = v - (jnp.sum(v * weight_grid * b, axis=(-2, -1), keepdims=True)
                     / (jnp.sum(weight_grid * b * b, axis=(-2, -1), keepdims=True) + 1e-30)) * b
        basis.append(v)
    out = raw
    for b in basis:
        out = out - (jnp.sum(out * weight_grid * b, axis=(-2, -1), keepdims=True)
                     / (jnp.sum(weight_grid * b * b, axis=(-2, -1), keepdims=True) + 1e-30)) * b
    return out


def chroma_radial_coma_field_raw(node_field, j, which, *, weight_grid=None):
    return _orthogonalize_off_shift(radial_coma_generator(node_field, j, which), node_field, weight_grid)


def colour_radial_degeneracy(node_field, *, knots=None, weight_grid=None):
    """Weighted cosine matrix between the round A3 terms / shift and the radial-family generators.

    ``node_field`` is one ePSF node (G, G). Returns ``(row_names, col_names, M)`` with rows blur, dil, kurt,
    shift_x, shift_y (raw-P gauged, as rendered) and columns rad1..J, radc1a..radcK{b} (as rendered, current
    knots and mode); entries are ``<a, b>_w / (|a| |b|)`` with the canonical weight grid."""
    old = get_radial_knots()
    if knots is not None:
        set_radial_knots(knots)
    try:
        P = jnp.asarray(node_field)
        wg = canonical_mode_weight_grid(int(P.shape[-1])) if weight_grid is None else weight_grid
        rows = {"blur": chroma_blur_field_raw(P), "dil": chroma_dilation_field_raw(P),
                "kurt": chroma_kurt_plain_field_raw(P),
                "shift_x": _gauge_off_raw_base(_displacement_generator(P, 1.0, 0.0), P),
                "shift_y": _gauge_off_raw_base(_displacement_generator(P, 0.0, 1.0), P)}
        cols = {}
        for j in range(1, n_radial_basis() + 1):
            cols[f"rad{j}"] = chroma_radial_field_raw(P, j)
        for j in range(1, n_coma_basis() + 1):
            for w in "ab":
                cols[f"radc{j}{w}"] = chroma_radial_coma_field_raw(P, j, w)
        dot = lambda a, b: float(jnp.sum(a * b * wg))
        M = np.array([[dot(a, b) / (np.sqrt(dot(a, a) * dot(b, b)) + 1e-30) for b in cols.values()]
                      for a in rows.values()])
        return list(rows), list(cols), M
    finally:
        set_radial_knots(old)


def chroma_kurt_field(
    node_field: jnp.ndarray, *, weight_grid: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Gauged flux-neutral kurtosis direction. Same two gauges as
    ``chroma_dilation_field``: the ``r^2*P - ratio*P`` construction is already
    close to zero-sum (gauge 1), but re-gauging exactly costs nothing and keeps
    the invariant exact rather than approximate; gauge 2 (projected off
    ``node_field``) removes the base-parallel, flux-degenerate component exactly
    as it does for dilation/aniso/shear.
    """
    raw = kurt_generator(node_field)
    if weight_grid is None:
        weight_grid = canonical_mode_weight_grid(int(node_field.shape[-1]))
    gauged = raw - jnp.mean(raw, axis=(-2, -1), keepdims=True)
    ref = jax.lax.stop_gradient(node_field)
    ref = ref - jnp.mean(ref, axis=(-2, -1), keepdims=True)
    den = jnp.sum(weight_grid * ref ** 2, axis=(-2, -1), keepdims=True) + 1e-12
    num = jnp.sum(gauged * weight_grid * ref, axis=(-2, -1), keepdims=True)
    return gauged - (num / den) * ref


def _tap2_colsum(
    shift: jnp.ndarray,
    *,
    n_grid: int = NODE_GRID_SIZE,
    oversample: int = OVERSAMPLE,
    dtype=None,
) -> jnp.ndarray:
    """``(..., n_grid)``: ``tap2_matrix(shift, ...).sum(axis=-2)`` (the column
    sums ``renorm_scalar`` needs), computed directly without materializing the
    ``(..., n_grid, n_grid)`` matrix -- same "every extra term costs a full
    residual under vmap+grad" reasoning as ``render_band``'s docstring.

    Every row of ``tap2_matrix(shift)`` places its two taps at ``row +
    floor(-shift*oversample)`` and one more, so column ``h``'s sum is just an
    indicator of whether ``h`` (or ``h - 1``) minus that same integer offset
    falls in a valid row index.
    """
    shift = jnp.asarray(shift)
    if dtype is None:
        dtype = shift.dtype if jnp.issubdtype(shift.dtype, jnp.floating) else jnp.float32
    shift = shift.astype(dtype)
    c = -shift * oversample
    n0 = jnp.floor(c)
    f = (c - n0)[..., None]  # (*lead, 1)
    h = jnp.arange(n_grid, dtype=dtype)
    x0 = h - n0[..., None]
    inrange0 = (x0 >= 0.0) & (x0 <= float(n_grid - 1))
    x1 = x0 - 1.0
    inrange1 = (x1 >= 0.0) & (x1 <= float(n_grid - 1))
    return (1 - f) * inrange0.astype(dtype) + f * inrange1.astype(dtype)


def renorm_scalar(
    grid: jnp.ndarray,
    com_x: jnp.ndarray,
    com_y: jnp.ndarray,
    *,
    oversample: int = OVERSAMPLE,
    dtype=None,
) -> jnp.ndarray:
    """``(...,)`` total mass of ``grid`` after a COM-recentering shift by
    ``(-com_x, -com_y)`` -- equals ``jnp.sum(bilinear_shift_physical(grid,
    -com_x, -com_y), axis=(-2, -1))`` exactly, computed as two ``(..., G)``
    column-sum contractions against ``grid`` (``tap2(-com).T @ 1``, via
    ``_tap2_colsum`` -- never materializing the shifted ``(..., G, G)`` grid or
    tap2's own ``(..., G, G)`` matrix). Callers must add their own epsilon and
    must not drop this factor: it is only ~1 at ridge=0 in the downstream flux
    solve, and ``flux_solve.solve_group_fluxes`` uses ``ridge=1e-6``.
    """
    G = grid.shape[-1]
    if dtype is None:
        dtype = grid.dtype
    com_x = jnp.asarray(com_x)
    com_y = jnp.asarray(com_y)
    colsum_y = _tap2_colsum(-com_y, n_grid=G, oversample=oversample, dtype=dtype)
    colsum_x = _tap2_colsum(-com_x, n_grid=G, oversample=oversample, dtype=dtype)
    tmp = jnp.einsum("...i,...ij->...j", colsum_y, grid.astype(dtype))
    return jnp.einsum("...j,...j->...", tmp, colsum_x)


# ---------------------------------------------------------------------------
# Gather-free packed/irregular renderer: same closed-form tap machinery as
# render_band, generalized from a contiguous n_pix-wide stamp to an arbitrary
# list of integer-lattice pixel offsets. See render_packed_pixels_banded's
# docstring for the derivation; render_physical_pixels_blocksum + the old
# recenter_grid_core/bilinear_shift_physical gather path remain the test
# reference (loss.USE_BANDED_RENDER_PACKED gates which one runs).
# ---------------------------------------------------------------------------


def render_packed_pixels_banded(
    local: jnp.ndarray,
    pix_x_index: jnp.ndarray,
    pix_y_index: jnp.ndarray,
    off_x: jnp.ndarray,
    off_y: jnp.ndarray,
    com_x: jnp.ndarray,
    com_y: jnp.ndarray,
    *,
    oversample: int = OVERSAMPLE,
) -> jnp.ndarray:
    """Gather-free replacement for ``render_physical_pixels_blocksum`` (+ an
    external ``recenter_grid_core``/``bilinear_shift_physical`` core-centroid
    pass), for
    one frame's already node-blended ``local`` grid.

    ``local``: ``(n_occ, G, G)`` **un-recentered** composite -- same input
    ``render_physical_pixels_blocksum`` + a separate COM shift would take.
    ``pix_x_index``/``pix_y_index``: ``(n_occ, P)`` **static** integer pixel
    offsets from an arbitrary per-occupied-slot integer reference (typically
    ``round(pix_x) - ref_x`` in the caller, built once from the concrete
    ``ctx.pix_x``/``ctx.pix_valid`` arrays -- never touched by AD).
    ``off_x``/``off_y``: ``(n_occ,)`` traced ``x_star - ref_x`` / ``y_star -
    ref_y`` for this frame, exactly analogous to ``render_stamps``'s
    ``dx``/``dy`` but around the same arbitrary integer reference used above
    (any integer works -- see the derivation below).
    ``com_x``/``com_y``: ``(n_occ,)`` analytic per-slot COM offset (pass
    zeros to skip the recenter fold, same convention as ``render_band``).

    **Why this is exact** (dev/forward_epsf_wcs README section 13 / this
    session's root-cause note; pixel-integrated update: CONTRACT_pixel_integrated_epsf.md):
    packed pixel centers are integers on the detector lattice
    (``packed_support.gather_packed_pixels`` rounds pixel centers to the
    nearest integer array index before gathering hp_d), so for a fixed
    integer reference ``ref``, ``pix_x - ref`` is exactly integer for every
    pixel in a slot. Substituting ``pix_x = ref + p_idx`` (``p_idx`` integer,
    static) and ``x_star = ref + off_x`` (traced) into the single-point-sample
    formula (``pixel = oversample**2 * bilinear(E, g)``)::

        gx = NODE_CENTER + (pix_x - x_star) * oversample
           = NODE_CENTER + oversample*p_idx - oversample*off_x

    is *exactly* ``render_stamps``'s ``gx = PAD + oversample*p -
    oversample*dx`` formula with ``p -> p_idx``, ``dx -> off_x`` and
    ``PAD -> NODE_CENTER`` (unlike the old sub-pixel renderer, no
    ``-0.5*(oversample-1)`` sub-sample-centering term: ``E`` has no
    sub-samples to center over -- verified by the module equivalence test)
    -- i.e. the fractional offset is shared by every pixel in the slot
    regardless of ``p_idx``, which is exactly the structure
    ``_axis_render_taps`` already exploits. ``p_idx`` need not be contiguous
    or even sorted; the closed form only ever uses it multiplied by
    ``oversample`` and added to a per-slot scalar.

    Unlike the square path (a full outer-product ``S×S`` grid, so one
    ``(n_pix, n_grid)`` band per axis serves an entire row/column of pixels),
    packed pixels are *not* a Cartesian product of x/y lists: pixel ``i``
    needs its own ``(pix_x[i], pix_y[i])`` pair. So this contracts the y-band
    and x-band against ``local`` keeping the pixel axis aligned throughout
    (``'oig,ogh->oih'`` then an elementwise-multiply + sum over the
    remaining axis), rather than the square path's two-matmul
    ``'nsg,ngh->nsh'`` / ``'nsh,nth->nst'`` pair (which would materialize the
    full ``P×P`` outer grid here, most of it never used).

    Returns ``(n_occ, P)``, **unnormalized** (pre-``renorm_scalar``) and
    **without** the ``pix_valid`` padding mask -- caller applies both, plus
    ``+1e-12`` on the renorm denominator, exactly as ``_render_occ_templates``
    does for the square path (see loss.py's docstring warning: the renorm is
    only exactly absorbed by the flux solve at ridge=0, and
    ``flux_solve.solve_group_fluxes`` uses ``ridge=1e-6``).
    """
    g_size = int(local.shape[-1])
    center = node_center_for_grid(g_size)
    pad_eff = center  # pixel-integrated E: no sub-samples to center over (see CONTRACT)
    Ay = render_band(
        off_y, com_y, p_index=pix_y_index, n_grid=g_size, pad=pad_eff, oversample=oversample,
    )  # (n_occ, P, G) -- weight from local's row (y) axis onto each pixel
    Ax = render_band(
        off_x, com_x, p_index=pix_x_index, n_grid=g_size, pad=pad_eff, oversample=oversample,
    )  # (n_occ, P, G) -- weight from local's col (x) axis onto each pixel
    half = jnp.einsum("oig,ogh->oih", Ay, local.astype(Ay.dtype))  # (n_occ, P, G)
    return jnp.sum(Ax * half, axis=-1)  # (n_occ, P)
