# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Forward model assembly + heteroscedastic NLL + regularization penalties.

``forward_model`` evaluates WCS positions, packs occupied group slots, composes
the node ePSF per frame, bilinear-blends (explicit 2×2 when applicable),
optionally core-recenters (hot-path ``n_iter=1``), and renders. Stage-1 can
reuse a cached core-centered local ePSF (dx-only). ``total_loss`` uses
per-stamp Huber NLL; the soft centroid monitor is outside the grad tape by
default (hard core-centroid already enforces the gauge).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from . import bright_width as BW
from . import cheb_wcs as CW
from . import epsf_model as EM
from . import flux_solve as FS
from .groups import GroupSet

VARIANCE_FLOOR = 1e-6
HUBER_DELTA_DEFAULT = 3.0
SNR_CAP_MAG_DEFAULT = 9.0
STAMP_WEIGHT_FLOOR = 0.05
STAGE1_CORE_STAMP = 7  # odd; central crop while r_fit <= 3

# Bit-exact, gather-free replacement for the per-slot blend->recenter->render hot
# path (EM.render_band/node_moments/renorm_scalar): a Boolean switch (rather than
# deleting the old path outright) so old-vs-new can be A/B'd directly against the
# same StaticContext/params. See EM.axis_band's docstring for the derivation.
USE_BANDED_RENDER = True

# Same idea, packed/irregular path: EM.render_packed_pixels_banded replaces the
# render_physical_pixels_blocksum (os,os)-subsample gather + a separate
# recenter_grid_core/bilinear_shift_physical gather pass with the closed-form,
# gather-free tap machinery generalized to arbitrary integer-lattice pixel
# lists -- see EM.render_packed_pixels_banded's docstring for the derivation
# (packed pixel centers are integers, so the fractional star offset is shared
# by every pixel in a slot, exactly like render_band's contiguous-stamp case).
# render_physical_pixels_blocksum + recenter_grid_core remain the test
# reference; this flag lets old-vs-new be A/B'd against the same
# StaticContext/params.
USE_BANDED_RENDER_PACKED = True


@dataclass(frozen=True)
class StaticContext:
    cheb_static: CW.ChebWcsStatic
    wcs_frame_basis: jnp.ndarray  # (n_frames, n_wcs_basis)
    w_frame_basis: jnp.ndarray  # (n_frames, n_w_basis)
    epsf_grid: EM.EpsfGridStatic
    members: jnp.ndarray  # (n_groups, K) int, clipped >=0
    valid: jnp.ndarray  # (n_groups, K) float, 1.0/0.0
    ra: jnp.ndarray  # (n_stars,) diagnostics only; not used in forward_model
    dec: jnp.ndarray  # (n_stars,)
    x_lin: jnp.ndarray  # (n_stars,) baked linear WCS pixels
    y_lin: jnp.ndarray  # (n_stars,)
    cheb_basis: jnp.ndarray  # (n_stars, n_terms) design in (xhat, yhat)
    stamp_center_x: jnp.ndarray  # (n_groups,) square path; unused when packed
    stamp_center_y: jnp.ndarray  # (n_groups,)
    t_exp_sec: float
    n_terms: int
    stamp_snr_weight: jnp.ndarray  # (n_groups,) soft SNR from catalog mag
    fit_radius: jnp.ndarray  # (n_groups,) active NLL radius (px); unused when packed
    stamp_active: jnp.ndarray  # (n_groups, n_frames) 1=keep, 0=rejected
    node_x: jnp.ndarray  # (n_cols,) ePSF node x for per-frame bilin
    node_y: jnp.ndarray  # (n_rows,) ePSF node y for per-frame bilin
    # Packed occupied flat-slot indices into members.reshape(-1); static shape.
    occ_flat_idx: jnp.ndarray  # (n_occ,)
    occ_group: jnp.ndarray  # (n_occ,) group index for each occupied slot
    occ_slot: jnp.ndarray  # (n_occ,) slot index within group
    # Packed irregular supports (P=0 → square stamp path).
    pix_x: jnp.ndarray  # (n_groups, P)
    pix_y: jnp.ndarray  # (n_groups, P)
    pix_valid: jnp.ndarray  # (n_groups, P)
    # Bright ePSF groups are True.  False rows remain fully differentiable in
    # WCS while their ePSF parameters are stop-gradient routed.
    is_epsf_contributor: jnp.ndarray  # (n_groups,) bool
    # Per-STAR centred Gaia colour, (n_stars,) = bp_rp - colour_ref, 0 where the
    # colour is unknown.  Star-indexed like ra/x_lin on purpose: the group slicers
    # do not touch star-indexed fields, so they need no change, and each member of
    # a blended group renders as its own slot and so deserves its own colour.
    # None disables the chromatic term entirely.  Defaulted so every existing
    # constructor call keeps working.
    chroma_delta: jnp.ndarray | None = None
    # Camera optical axis (x, y) in this context's pixel frame, for the global colour
    # model (``chroma_g8``). A Python tuple, so it is a trace-time constant.
    chroma_axis: tuple | None = None
    # chroma_g8 options (trace-time constants): "mean" = the original mean-removed-base
    # gauge, "raw" = project off P itself (no flat sheet); and dropping the dilation
    # term (blur and dilation are ~95% collinear on the fitted ePSF).
    chroma_g8_gauge: str = "mean"
    chroma_g8_no_dil: bool = False
    # Named chroma_g8 coefficients appended after the 8 base slots, in order
    # (see G8_EXTRA_NAMES). Empty = legacy rule (len 9/10 -> radial blur terms).
    chroma_g8_extras: tuple = ()
    # <delta^2> over the colour-gauge population, subtracted from delta^2 in the quadratic-colour
    # radial shift (extras sq0/sq1) so that term carries no colour-independent shift (the WCS owns that).
    chroma_delta2_mean: float = 0.0
    # Brightness width (bright_width.py): per-STAR peak charge per 2-s read q (e-), star-indexed
    # like x_lin, and its reference q_ref. None = no per-star q; a params set carrying the
    # ``bright_width`` leaf then refuses to render.
    bright_q: jnp.ndarray | None = None
    bright_q_ref: float = 0.0

    @property
    def is_packed(self) -> bool:
        return int(self.pix_x.shape[-1]) > 0


def slice_static_context_frames(ctx: StaticContext, start: int, stop: int) -> StaticContext:
    """Return a *frame*-sliced context (the group slicer's time-axis twin).

    Only three fields carry a frame axis: ``wcs_frame_basis`` (T, n_wcs_basis)
    and ``w_frame_basis`` (T, n_w_basis) on axis 0, and ``stamp_active``
    (n_groups, T) on axis 1. Everything else is star/group/pixel indexed and is
    shared unchanged.

    The per-frame WCS positions are exactly reproducible from a row slice
    (``x_t = cheb_basis @ (wcs_coeff @ wcs_frame_basis.T)`` is independent
    across frames), but ``w_of_t``'s zero-time-mean gauge is **not** -- it
    couples all frames. Callers slicing frames for memory reasons must compute
    the gauge on the full frame set (``whole_orbit_w_of_t``) and hand the
    matching rows to ``total_loss(..., w_of_t_override=...)``.
    """
    start, stop = int(start), int(stop)
    return replace(
        ctx,
        wcs_frame_basis=ctx.wcs_frame_basis[start:stop],
        w_frame_basis=ctx.w_frame_basis[start:stop],
        stamp_active=ctx.stamp_active[:, start:stop],
    )


def slice_static_context_frames_dynamic(
    ctx: StaticContext,
    wcs_frame_basis_padded: jnp.ndarray,
    w_frame_basis_padded: jnp.ndarray,
    stamp_active_padded: jnp.ndarray,
    lo,
    size: int,
) -> StaticContext:
    """``slice_static_context_frames`` with a traced start index.

    Takes the already frame-axis-*padded* arrays explicitly (padded to a
    multiple of ``size`` by the caller -- see ``make_accum_step_fn``) rather
    than reading ``ctx``'s own (unpadded, length ``n_frames``) fields: with an
    unpadded array, ``dynamic_slice_in_dim`` would silently clamp the start
    index for the last block instead of erroring, returning duplicated frames
    from the array's tail rather than the intended zero-padding. Padded
    arrays make every block's slice, including the last, land fully in bounds.

    ``size`` (the block length) must be a static Python int -- every call site
    uses the same value, so this compiles once regardless of how many distinct
    ``lo`` values are passed at runtime (unlike Python ``[lo:hi]`` slicing,
    which requires static bounds and forces one XLA compile per distinct
    ``(lo, hi)`` pair -- the bug this function exists to avoid; see
    ``fit.make_accum_step_fn``).
    """
    lo = jnp.asarray(lo, dtype=jnp.int32)
    return replace(
        ctx,
        wcs_frame_basis=jax.lax.dynamic_slice_in_dim(wcs_frame_basis_padded, lo, size, axis=0),
        w_frame_basis=jax.lax.dynamic_slice_in_dim(w_frame_basis_padded, lo, size, axis=0),
        stamp_active=jax.lax.dynamic_slice_in_dim(stamp_active_padded, lo, size, axis=1),
    )


def w_field_from_coeff(w_coeff: jnp.ndarray, w_frame_basis: jnp.ndarray) -> jnp.ndarray:
    """Per-frame temporal-mode amplitude field, zero-time-mean gauged.

    ``w_coeff`` is either

    - ``(n_modes, n_basis)`` -- legacy: one global ``w_k(t)`` shared by every
      ePSF node (the pre-T3 model), or
    - ``(n_modes, n_rows, n_cols, n_basis)`` -- T3: a per-node ``w_k(t, x, y)``
      field, matching the ePSF grid's own ``(n_rows, n_cols)`` node layout.
      Downstream (``_render_occ_templates``/``_render_occ_templates_packed``),
      this per-node amplitude gets bilinear-blended down to each star's exact
      position with the SAME ``blend_field``/node-blend-weights machinery the
      ePSF base and modes already use -- see ``_node_field_for_frame``.

    Dispatch is purely shape-based (``w_coeff.ndim``), so every existing 2-D
    checkpoint/caller is completely unaffected: the ``ndim == 2`` branch below
    is textually identical to the pre-T3 code.

    Returns ``(n_frames, n_modes)`` (legacy) or
    ``(n_frames, n_rows, n_cols, n_modes)`` (spatial), zero-mean along the
    frame axis in both cases. In the spatial case the gauge is applied PER
    NODE (``jnp.mean(..., axis=0)`` reduces only the frame axis, independently
    at every ``(row, col)``): each node's own ``w_k(t)`` curve integrates to
    zero over time on its own, the exact node-for-node generalization of the
    legacy scalar gauge, not a single field-averaged constraint.
    """
    if w_coeff.ndim == 2:
        w_of_t = (w_coeff @ w_frame_basis.T).T  # (T, K)
    elif w_coeff.ndim == 4:
        # (K, R, C, B) x (T, B) -> (T, R, C, K)
        w_of_t = jnp.einsum("krcb,tb->trck", w_coeff, w_frame_basis)
    else:
        raise ValueError(f"w_coeff must be 2-D or 4-D, got shape {w_coeff.shape}")
    return w_of_t - jnp.mean(w_of_t, axis=0, keepdims=True)


def _node_field_for_frame(base: jnp.ndarray, modes: jnp.ndarray, w_t: jnp.ndarray) -> jnp.ndarray:
    """One frame's node ePSF field: ``base + sum_k modes_k * w_k``, per node.

    ``w_t`` is this frame's slice of ``w_field_from_coeff``'s output:
    ``(n_modes,)`` (legacy -- the same amplitude broadcast to every node,
    bit-identical to the pre-T3 single-einsum form) or
    ``(n_rows, n_cols, n_modes)`` (T3 spatial -- this frame's own per-node
    amplitude field, already gauged and node-shaped by the caller). Dispatch
    is shape-based, mirroring ``w_field_from_coeff``.
    """
    if w_t.ndim == 1:
        return base + jnp.einsum("kijxy,k->ijxy", modes, w_t)
    return base + jnp.einsum("kijxy,ijk->ijxy", modes, w_t)


def whole_orbit_w_of_t(params: dict[str, jnp.ndarray], ctx: StaticContext) -> jnp.ndarray:
    """``w_of_t`` with the zero-time-mean gauge applied over the FULL frame set.

    Must be called with a context that still holds every frame (i.e. before any
    frame slicing), so the mean matches the unchunked model exactly. Cheap:
    ``(T, n_modes)`` (or ``(T, n_rows, n_cols, n_modes)`` for a spatial
    ``w_coeff`` -- see ``w_field_from_coeff``) with T ~ 2e3 and n_modes ~ 1-2.
    """
    epsf_params = _epsf_params_for_context(params, ctx)
    return w_field_from_coeff(epsf_params["w_coeff"], ctx.w_frame_basis)


def slice_static_context(ctx: StaticContext, start: int, stop: int) -> StaticContext:
    """Return a group-sliced context for memory-bounded diagnostics.

    Training keeps one static shape for JIT efficiency; post-fit diagnostics
    may instead evaluate a few groups at a time to avoid materializing the
    full ``(G, K, T, P)`` tensor for large full-orbit bundles.
    """
    start, stop = int(start), int(stop)
    if not (0 <= start < stop <= int(ctx.members.shape[0])):
        raise ValueError(f"invalid group slice [{start}:{stop}]")
    members = ctx.members[start:stop]
    valid = ctx.valid[start:stop]
    occ = np.argwhere(np.asarray(valid) > 0)
    k = int(members.shape[1])
    return replace(
        ctx,
        members=members,
        valid=valid,
        stamp_center_x=ctx.stamp_center_x[start:stop],
        stamp_center_y=ctx.stamp_center_y[start:stop],
        stamp_snr_weight=ctx.stamp_snr_weight[start:stop],
        fit_radius=ctx.fit_radius[start:stop],
        stamp_active=ctx.stamp_active[start:stop],
        occ_flat_idx=jnp.asarray(occ[:, 0] * k + occ[:, 1], dtype=jnp.int32),
        occ_group=jnp.asarray(occ[:, 0], dtype=jnp.int32),
        occ_slot=jnp.asarray(occ[:, 1], dtype=jnp.int32),
        pix_x=ctx.pix_x[start:stop],
        pix_y=ctx.pix_y[start:stop],
        pix_valid=ctx.pix_valid[start:stop],
        is_epsf_contributor=ctx.is_epsf_contributor[start:stop],
    )


def init_params(
    static: CW.ChebWcsStatic,
    epsf_params: EM.EpsfGridParams,
    *,
    n_wcs_basis: int,
    n_w_basis: int,
    chroma: bool = False,
    chroma_affine: bool = False,
    chroma_kurt: bool = False,
    chroma_halo: bool = False,
    chroma_image: bool = False,
    w_spatial: bool = False,
) -> dict[str, jnp.ndarray]:
    n_modes = int(epsf_params.modes.shape[0])
    if w_spatial:
        # T3: per-node temporal-mode amplitude field w_k(t, x, y) instead of a
        # single global w_k(t) -- see w_field_from_coeff's docstring. Zero-init,
        # so a fresh spatial run starts bit-identical to a fresh non-spatial run
        # (every node's curve is exactly zero until trained).
        n_rows, n_cols = int(epsf_params.base.shape[0]), int(epsf_params.base.shape[1])
        w_coeff = jnp.zeros((n_modes, n_rows, n_cols, n_w_basis), dtype=jnp.float32)
    else:
        w_coeff = jnp.zeros((n_modes, n_w_basis), dtype=jnp.float32)
    out = {
        "wcs_coeff": CW.zero_coeff_matrix(static, n_wcs_basis),
        "epsf_base_raw": EM.encode_epsf_base(epsf_params.base),
        "epsf_modes": EM.encode_epsf_modes(epsf_params.modes, epsf_params.base),
        "w_coeff": w_coeff,
    }
    if chroma:
        # Shapes follow the ePSF node grid, never hardcoded: a 3x3 colour grid must
        # come out right without touching this function.
        n_rows, n_cols = int(epsf_params.base.shape[0]), int(epsf_params.base.shape[1])
        out["chroma_shift"] = jnp.zeros((2, n_rows, n_cols), dtype=jnp.float32)
        out["chroma_dilation"] = jnp.zeros((n_rows, n_cols), dtype=jnp.float32)
        if chroma_affine:
            out["chroma_aniso"] = jnp.zeros((n_rows, n_cols), dtype=jnp.float32)
            out["chroma_shear"] = jnp.zeros((n_rows, n_cols), dtype=jnp.float32)
        if chroma_kurt:
            if not chroma_affine:
                raise ValueError("chroma_kurt=True requires chroma_affine=True")
            out["chroma_kurt"] = jnp.zeros((n_rows, n_cols), dtype=jnp.float32)
        if chroma_halo:
            # Zero-init, and no prerequisite leaf: the halo is additive, not a
            # reshaping, so it does not sit "on top of" the affine family.
            out["chroma_halo"] = jnp.zeros((n_rows, n_cols), dtype=jnp.float32)
        if chroma_image:
            # Zero-init: a fresh run starts bit-identical to one without the leaf.
            g_size = int(epsf_params.base.shape[-1])
            out["chroma_image"] = jnp.zeros((g_size, g_size), dtype=jnp.float32)
    elif chroma_image:
        raise ValueError("chroma_image=True requires chroma=True")
    elif chroma_halo:
        raise ValueError("chroma_halo=True requires chroma=True")
    return out


# "AK hard" (EXPERIMENT, opt-in, 2026-09-30): Anderson & King (2000) smooth their ePSF after every update; here the
# ePSF actually used is ALWAYS its per-node local-polynomial-smoothed version (local_poly_smooth, radius-dependent
# order), with the flux rule and core centring re-applied. 0 = off; else the smoothing window (5 = AK2000 eq. 8, or 7).
# Process-wide so that every consumer (scene_fit training, score_oof, lcurve, raster scripts) decodes the same model:
# set it with set_local_poly_hard(), or the env var SYNDIFF_EPSF_LOCAL_POLY_HARD read at import.
import os as _os
_LOCAL_POLY_HARD = int(_os.environ.get("SYNDIFF_EPSF_LOCAL_POLY_HARD", "0") or 0)


def set_local_poly_hard(window: int) -> None:
    """Switch AK-hard decoding on (window 5 or 7) or off (0) for this process; also exported to the env."""
    global _LOCAL_POLY_HARD
    _LOCAL_POLY_HARD = int(window or 0)
    _os.environ["SYNDIFF_EPSF_LOCAL_POLY_HARD"] = str(_LOCAL_POLY_HARD)


def local_poly_hard_window() -> int:
    return _LOCAL_POLY_HARD


def decoded_epsf_base(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    base = EM.decode_epsf_base(params["epsf_base_raw"])
    if _LOCAL_POLY_HARD:
        sm = local_poly_smooth(base, _LOCAL_POLY_HARD)
        sm = jnp.clip(sm, 1e-12, None)
        sm = sm / jnp.sum(sm, axis=(-2, -1), keepdims=True)
        base = EM.recenter_grid_core(EM.enforce_phase_flux_rule(sm), clip_nonneg=True)
    # optional per-frame jitter blur (scene_fit_multi): a non-trainable input, not a leaf
    blur = params.get("epsf_blur") if hasattr(params, "get") else None
    if blur is not None:
        base = EM.blur_epsf_base(base, blur)
    return base


def decoded_epsf_modes(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return EM.decode_epsf_modes(params["epsf_modes"], decoded_epsf_base(params))


def soft_snr_stamp_weights(
    tess_mag: np.ndarray,
    *,
    snr_cap_mag: float = SNR_CAP_MAG_DEFAULT,
    w_min: float = STAMP_WEIGHT_FLOOR,
) -> np.ndarray:
    """Soft SNR stamp weights from catalog mag: w = min(rel,1)^2 floored at w_min."""
    mag = np.asarray(tess_mag, dtype=float)
    rel = np.power(10.0, -0.4 * (mag - float(snr_cap_mag)))
    w = np.minimum(rel, 1.0) ** 2
    return np.maximum(w, float(w_min)).astype(np.float32)


def stamp_corner_radius_px(stamp_physical: int = EM.STAMP_PHYSICAL) -> float:
    """Max radius (px) reachable within a ``stamp_physical``-square stamp,
    center at ``stamp_physical // 2`` (matches ``stamp_radius_grid``'s
    convention). Node-grid samples beyond this radius are never touched by
    any centered star's render, *regardless* of ``fit_radius`` -- see
    ``dev_grad_coverage.py`` Task-1 diagnostic. For the default 13x13 stamp
    this is ``6*sqrt(2) ~= 8.49`` px.
    """
    half = int(stamp_physical) // 2
    return float(np.hypot(half, half))


def fit_radius_tiers_default(
    stamp_physical: int = EM.STAMP_PHYSICAL,
) -> tuple[tuple[float, float], ...]:
    """Default (mag_upper_bound, radius_px) tiers for ``fit_radius_from_mag``.

    Historically this was ``((9, 6.0), (11, 3.5), (inf, 2.5))`` -- a hard
    circular NLL-weight cutoff well inside the stamp for every tier. A
    gradient-coverage diagnostic (``dev_grad_coverage.py``) showed this
    structurally zeroes ``d(loss)/d(epsf_base_raw)`` for every node beyond
    the tier's radius, for *all* frames and *all* stars in that tier forever
    -- not a slow-training issue but a hard, permanent zero that 600+ Adam
    steps cannot cross. The majority mag9-11 tier (69/94 groups in the
    orbit1_half_mag710 bundle) was capped at 3.5 px, well short of the
    stamp's own 8.49 px corner.

    Per-star SNR-based downweighting (``soft_snr_stamp_weights`` /
    ``stamp_snr_weight``) already exists as the *continuous* mechanism for
    not letting noisy/faint stars dominate the loss; a hard *radius* cutoff
    on top of that duplicates the same intent with the specific side effect
    of permanently blinding the ePSF wing to real data. The new default
    therefore opens every tier to the full stamp footprint
    (``stamp_corner_radius_px``) -- the tiers are kept as a tunable knob
    (``--fit-radius-tiers``) for anyone who later finds a *narrower* radius
    empirically better for the faintest stars, but nothing is cut by default.
    """
    r = stamp_corner_radius_px(stamp_physical)
    return ((9.0, r), (11.0, r), (float("inf"), r))


def parse_fit_radius_tiers(spec: str) -> tuple[tuple[float, float], ...]:
    """Parse a ``--fit-radius-tiers`` CLI spec like ``"9:6.0,11:3.5,inf:2.5"``
    into ascending ``((mag_upper_bound, radius_px), ...)``. The last tier's
    mag bound should be ``inf`` (or larger than any real catalog mag) so
    every star is covered.
    """
    tiers: list[tuple[float, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        mag_s, _, r_s = chunk.partition(":")
        if not r_s:
            raise ValueError(f"bad --fit-radius-tiers chunk {chunk!r}, expected 'mag:radius'")
        mag_bound = float("inf") if mag_s.strip().lower() == "inf" else float(mag_s)
        tiers.append((mag_bound, float(r_s)))
    if not tiers:
        raise ValueError(f"--fit-radius-tiers {spec!r} parsed to no tiers")
    tiers.sort(key=lambda t: t[0])
    return tuple(tiers)


def fit_radius_from_mag(
    tess_mag: np.ndarray,
    *,
    stage: int,
    tiers: tuple[tuple[float, float], ...] | None = None,
    stamp_physical: int = EM.STAMP_PHYSICAL,
) -> np.ndarray:
    """Mag-dependent NLL fitting radius (px); stage 1 is core-capped at 3 px.

    ``tiers``: ascending ``(mag_upper_bound, radius_px)`` pairs; a star gets
    the radius of the first (smallest-bound) tier its mag is strictly below.
    Defaults to ``fit_radius_tiers_default(stamp_physical)`` (whole-stamp
    coverage for every tier -- see that function's docstring).
    """
    mag = np.asarray(tess_mag, dtype=float)
    tiers = tiers if tiers is not None else fit_radius_tiers_default(stamp_physical)
    r = np.full(mag.shape, np.float32(tiers[-1][1]), dtype=np.float32)
    for mag_bound, radius in reversed(tiers[:-1]):
        r = np.where(mag < mag_bound, np.float32(radius), r)
    if int(stage) <= 1:
        r = np.minimum(r, np.float32(3.0))
    return r


def stamp_radius_grid(stamp: int = EM.STAMP_PHYSICAL) -> np.ndarray:
    """(S, S) radius from stamp center in pixels."""
    c = stamp // 2
    yy, xx = np.mgrid[0:stamp, 0:stamp]
    return np.sqrt((xx - c) ** 2 + (yy - c) ** 2).astype(np.float32)


def _occupied_slot_tables(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat occupied indices + (group, slot) maps from a boolean/float valid mask."""
    valid_b = np.asarray(valid) > 0.5
    n_groups, K = valid_b.shape
    flat = valid_b.reshape(-1)
    occ_flat = np.flatnonzero(flat).astype(np.int32)
    if occ_flat.size == 0:
        occ_flat = np.zeros((1,), dtype=np.int32)
        occ_group = np.zeros((1,), dtype=np.int32)
        occ_slot = np.zeros((1,), dtype=np.int32)
        return occ_flat, occ_group, occ_slot
    occ_group = (occ_flat // K).astype(np.int32)
    occ_slot = (occ_flat % K).astype(np.int32)
    return occ_flat, occ_group, occ_slot


def build_static_context(
    *,
    cheb_static: CW.ChebWcsStatic,
    wcs_frame_basis: jnp.ndarray,
    w_frame_basis: jnp.ndarray,
    epsf_grid: EM.EpsfGridStatic,
    groups: GroupSet,
    ra: np.ndarray,
    dec: np.ndarray,
    stamp_center_x: np.ndarray,
    stamp_center_y: np.ndarray,
    t_exp_sec: float,
    stamp_snr_weight: np.ndarray,
    fit_radius: np.ndarray,
    stamp_active: np.ndarray | None = None,
    x_lin: np.ndarray | jnp.ndarray | None = None,
    y_lin: np.ndarray | jnp.ndarray | None = None,
    cheb_basis: np.ndarray | jnp.ndarray | None = None,
    pix_x: np.ndarray | jnp.ndarray | None = None,
    pix_y: np.ndarray | jnp.ndarray | None = None,
    pix_valid: np.ndarray | jnp.ndarray | None = None,
    is_epsf_contributor: np.ndarray | jnp.ndarray | None = None,
    bp_rp: np.ndarray | jnp.ndarray | None = None,
    chroma_axis: tuple | None = None,
    colour_ref: float = 0.0,
    chroma_g8_gauge: str = "mean",
    chroma_g8_no_dil: bool = False,
    chroma_g8_extras: tuple = (),
    chroma_delta2_mean: float = 0.0,
) -> StaticContext:
    members_safe = np.clip(groups.members, 0, None)
    ra_j = jnp.asarray(ra, dtype=jnp.float32)
    dec_j = jnp.asarray(dec, dtype=jnp.float32)
    if x_lin is None or y_lin is None or cheb_basis is None:
        if not (x_lin is None and y_lin is None and cheb_basis is None):
            raise ValueError("x_lin, y_lin, cheb_basis must all be provided or all omitted")
        x_lin_j, y_lin_j, basis_j = CW.star_basis(ra_j, dec_j, cheb_static)
    else:
        x_lin_j = jnp.asarray(x_lin, dtype=jnp.float32)
        y_lin_j = jnp.asarray(y_lin, dtype=jnp.float32)
        basis_j = jnp.asarray(cheb_basis, dtype=jnp.float32)
    n_frames = int(np.asarray(wcs_frame_basis).shape[0])
    if stamp_active is None:
        stamp_active = np.ones((groups.n_groups, n_frames), dtype=np.float32)
    if is_epsf_contributor is None:
        is_epsf_contributor = np.ones((groups.n_groups,), dtype=bool)
    occ_flat, occ_group, occ_slot = _occupied_slot_tables(groups.valid)
    if pix_x is None:
        pix_x_j = jnp.zeros((groups.n_groups, 0), dtype=jnp.float32)
        pix_y_j = jnp.zeros((groups.n_groups, 0), dtype=jnp.float32)
        pix_valid_j = jnp.zeros((groups.n_groups, 0), dtype=jnp.float32)
    else:
        if pix_y is None or pix_valid is None:
            raise ValueError("pix_x, pix_y, pix_valid must all be provided together")
        pix_x_j = jnp.asarray(pix_x, dtype=jnp.float32)
        pix_y_j = jnp.asarray(pix_y, dtype=jnp.float32)
        pix_valid_j = jnp.asarray(pix_valid, dtype=jnp.float32)

    if bp_rp is None:
        chroma_delta_j = None
    else:
        c = np.asarray(bp_rp, dtype=np.float64)
        if c.shape != (int(ra_j.shape[0]),):
            raise ValueError(
                f"bp_rp must be per-star with shape {(int(ra_j.shape[0]),)}, got {c.shape}"
            )
        # A missing Gaia colour must become delta = 0, i.e. "treat this star as the
        # reference colour", not NaN: one NaN would poison the whole gradient.
        finite = np.isfinite(c)
        chroma_delta_j = jnp.asarray(
            np.where(finite, c - float(colour_ref), 0.0), dtype=jnp.float32
        )
        if not bool(np.all(np.isfinite(np.asarray(chroma_delta_j)))):
            raise ValueError("chroma_delta is not all finite after NaN handling")
    return StaticContext(
        cheb_static=cheb_static,
        wcs_frame_basis=jnp.asarray(wcs_frame_basis, dtype=jnp.float32),
        w_frame_basis=jnp.asarray(w_frame_basis, dtype=jnp.float32),
        epsf_grid=epsf_grid,
        members=jnp.asarray(members_safe, dtype=jnp.int32),
        valid=jnp.asarray(groups.valid, dtype=jnp.float32),
        ra=ra_j,
        dec=dec_j,
        x_lin=x_lin_j,
        y_lin=y_lin_j,
        cheb_basis=basis_j,
        stamp_center_x=jnp.asarray(stamp_center_x, dtype=jnp.float32),
        stamp_center_y=jnp.asarray(stamp_center_y, dtype=jnp.float32),
        t_exp_sec=float(t_exp_sec),
        n_terms=cheb_static.n_terms,
        stamp_snr_weight=jnp.asarray(stamp_snr_weight, dtype=jnp.float32),
        fit_radius=jnp.asarray(fit_radius, dtype=jnp.float32),
        stamp_active=jnp.asarray(stamp_active, dtype=jnp.float32),
        node_x=jnp.asarray(epsf_grid.node_x, dtype=jnp.float32),
        node_y=jnp.asarray(epsf_grid.node_y, dtype=jnp.float32),
        occ_flat_idx=jnp.asarray(occ_flat, dtype=jnp.int32),
        occ_group=jnp.asarray(occ_group, dtype=jnp.int32),
        occ_slot=jnp.asarray(occ_slot, dtype=jnp.int32),
        pix_x=pix_x_j,
        pix_y=pix_y_j,
        pix_valid=pix_valid_j,
        is_epsf_contributor=jnp.asarray(is_epsf_contributor, dtype=bool),
        chroma_delta=chroma_delta_j,
        chroma_axis=None if chroma_axis is None else (float(chroma_axis[0]), float(chroma_axis[1])),
        chroma_g8_gauge=str(chroma_g8_gauge),
        chroma_g8_no_dil=bool(chroma_g8_no_dil),
        chroma_g8_extras=tuple(chroma_g8_extras),
        chroma_delta2_mean=float(chroma_delta2_mean),
    )


def with_fit_radius(ctx: StaticContext, fit_radius: np.ndarray | jnp.ndarray) -> StaticContext:
    """Return a copy of ``ctx`` with updated per-group fit radii (stage split)."""
    return replace(ctx, fit_radius=jnp.asarray(fit_radius, dtype=jnp.float32))


def with_stamp_active(ctx: StaticContext, stamp_active: np.ndarray | jnp.ndarray) -> StaticContext:
    """Return a copy of ``ctx`` with updated per-(group, frame) keep mask."""
    return replace(ctx, stamp_active=jnp.asarray(stamp_active, dtype=jnp.float32))


def _epsf_params_for_context(
    params: dict[str, jnp.ndarray], ctx: StaticContext,
) -> dict[str, jnp.ndarray]:
    """Apply the faint-anchor ePSF gradient barrier for a homogeneous batch."""
    flags = np.asarray(ctx.is_epsf_contributor, dtype=bool)
    if np.all(flags):
        return params
    if np.any(flags):
        raise ValueError("mixed ePSF contributor batch must be split before rendering")
    out = dict(params)
    # NOTE: the chromatic leaves are deliberately NOT in this tuple. Faint WCS
    # anchors outnumber ePSF contributors 4.6 to 1 and span the colour range, and
    # the chromatic shift is a positional quantity like the WCS, whose gradients
    # are already live for them. They should drive the colour term. Do not "fix"
    # this by adding chroma_shift/chroma_dilation here.
    for key in ("epsf_base_raw", "epsf_modes", "w_coeff"):
        out[key] = jax.lax.stop_gradient(params[key])
    return out


CHROMA_LEAVES = ("chroma_shift", "chroma_dilation")

# The free dP/dcolour IMAGE leaf (task: single-FFI chromatic study). One global
# (G, G) image, not a per-node field: 36 free node images on a 6x6 grid would be
# ~121k parameters against one frame.
CHROMA_IMAGE_LEAVES = ("chroma_image",)
# Reserved key inside the ``chroma_fields`` dict the renderers receive. Unlike
# every other entry there, its value is the RAW (G, G) image leaf rather than a
# per-node scalar coefficient, and its "generator" is the gauged image itself
# rather than a functional of the base -- so it is NOT in
# ``CHROMA_FIELD_GENERATORS`` and all four render folds special-case this one key.
CHROMA_IMAGE_FIELD = "image"

# C1 colour-AFFINE extension: anisotropic stretch + 45-degree shear per node,
# added on top of the shift+dilation term above. Deliberately kept OUT of
# ``CHROMA_LEAVES`` -- that tuple doubles as ``has_chroma``'s ALL-present gate,
# and every pre-C1 checkpoint/test constructs params with exactly the shift+
# dilation pair, so folding the affine leaves in there would make
# ``has_chroma`` False for every one of them. ``fit.OPTIONAL_LEAVES`` (the
# generic "carry when present" checkpoint/bootstrap tuple) includes both this
# tuple and ``CHROMA_LEAVES``, so serialization needs no separate handling.
CHROMA_AFFINE_LEAVES = ("chroma_aniso", "chroma_shear")
# Optional flux-neutral core/wing redistribution term, only meaningful on top
# of the affine leaves; see epsf_model.chroma_kurt_field.
CHROMA_KURT_LEAVES = ("chroma_kurt",)
# Additive chromatic HALO: a fixed r^-index radial profile (NOT a functional of the
# base) with one trainable amplitude per node. Independent of the affine/kurt leaves
# -- it is the only chroma term that is not a reshaping of the ePSF, so it has no
# prerequisite. See epsf_model.chroma_halo_field for why it exists and why it carries
# only the base-orthogonal gauge.
CHROMA_HALO_LEAVES = ("chroma_halo",)
# Global 8-parameter colour model (2026-09-24): ONE vector for the whole CCD,
# [s0, s1, s2, k0, k1, b, eps, t] -- radial shift s(r) = s0 + s1 r + s2 r^2 (px/mag,
# along the unit vector to the optical axis; negative = a redder star moves away),
# plain kurtosis k(r) = k0 + k1 r, global blur b, global dilation eps, global radial
# trefoil t; r = distance to ``ctx.chroma_axis`` in units of 1000 px. Mutually
# exclusive with the node-field leaves.
CHROMA_G8_LEAVES = ("chroma_g8",)
# Allowed names for StaticContext.chroma_g8_extras (coefficients appended after slot 7).
# Colour model v2 (2026-09-28): sq0/sq1 = radial shift quadratic in colour; q1_*/q2_* =
# colour elongation planes (aniso/shear) in X, Y; dilp_x/dilp_y = colour round-width plane.
G8_EXTRA_NAMES = ("blur_r", "blur_r2", "dil_r", "astig0", "astig_r", "sq0", "sq1", "q1_0", "q1_x", "q1_y",
                  "q2_0", "q2_x", "q2_y", "dilp_x", "dilp_y")


def _g8_poly(x, names, basis):
    """``sum x[n] * b`` over the extras in ``names`` that are present, or None when none is
    (a trace-time choice, so absent extras add no ops)."""
    out = None
    for n, b in zip(names, basis):
        if n in x:
            out = x[n] * b if out is None else out + x[n] * b
    return out


def colour_ref_from_bundle(bundle) -> float:
    """Weighted mean Gaia colour of the fitting population, or 0.0 when unavailable.

    THE gauge for the chromatic term, and a pure function of the bundle so that
    training, export and every diagnostic recompute exactly the same number without
    having to pass it around. Weighted by the per-group ``stamp_snr_weight`` the loss
    itself uses, so the population mean of ``c - c_ref`` is zero under the loss's own
    measure rather than under an unweighted star count.

    Why it matters: ``eps * <delta> * D[P]`` is a star-independent shape perturbation,
    which lies exactly in the tangent space of the free per-node ePSF base. That is a
    real first-order null direction and centring the colour removes it.
    """
    bp_rp = getattr(bundle, "bp_rp", None)
    if bp_rp is None:
        return 0.0
    c = np.asarray(bp_rp, dtype=np.float64)
    finite = np.isfinite(c)
    if not finite.any():
        return 0.0
    members = np.asarray(bundle.members)
    valid = np.asarray(bundle.valid, dtype=bool)
    sw = np.zeros(c.shape[0], dtype=np.float64)
    gw = np.asarray(bundle.stamp_snr_weight, dtype=np.float64)
    np.add.at(sw, members[valid], np.repeat(gw[:, None], members.shape[1], axis=1)[valid])
    w = np.where(finite, sw, 0.0)
    if w.sum() <= 0:
        return 0.0
    return float(np.sum(w * np.nan_to_num(c)) / w.sum())


def has_chroma(params: dict[str, jnp.ndarray]) -> bool:
    """True when the parameter set carries the chromatic leaves.

    A Python bool on purpose, so the render path is chosen at trace time and a run
    without the term is bit-identical to the pre-chroma code.
    """
    return all(k in params for k in CHROMA_LEAVES)


def has_chroma_affine(params: dict[str, jnp.ndarray]) -> bool:
    """True when the parameter set carries the C1 colour-affine leaves.

    Requires the base chromatic leaves too (affine is layered on top of
    shift+dilation, never trained standalone).
    """
    return has_chroma(params) and all(k in params for k in CHROMA_AFFINE_LEAVES)


def has_chroma_kurt(params: dict[str, jnp.ndarray]) -> bool:
    """True when the parameter set carries the optional kurtosis leaf."""
    return has_chroma_affine(params) and all(k in params for k in CHROMA_KURT_LEAVES)


def has_chroma_halo(params: dict[str, jnp.ndarray]) -> bool:
    return all(k in params for k in CHROMA_HALO_LEAVES)


def has_chroma_g8(params: dict[str, jnp.ndarray]) -> bool:
    return all(k in params for k in CHROMA_G8_LEAVES)


def _chroma_g8_slot_terms(params, ctx, star_occ):
    """Per-slot terms of the global model; field values are PER-SLOT weight vectors
    (1-D, already multiplied by delta), which the render folds recognise by ndim.

    Named extras (``ctx.chroma_g8_extras``) extend the base 8: radial blur/dilation/astig
    terms, a shift quadratic in colour ``(delta^2 - ctx.chroma_delta2_mean)(sq0 + sq1 r) n_axis``,
    elongation planes ``delta (q1_0 + q1_x X + q1_y Y)`` on aniso and ``delta (q2_0 + ...)`` on
    shear, and a round-width plane ``dilp_x X + dilp_y Y`` added to dil, with X, Y the
    centred detector position / 1024. Extras that are absent add no ops and no fields."""
    if ctx.chroma_delta is None or ctx.chroma_axis is None:
        raise ValueError("chroma_g8 needs both colour (bp_rp) and chroma_axis in the context")
    delta = ctx.chroma_delta[star_occ]
    vx = ctx.chroma_axis[0] - ctx.x_lin[star_occ]
    vy = ctx.chroma_axis[1] - ctx.y_lin[star_occ]
    rr = jnp.sqrt(vx * vx + vy * vy) + 1e-6
    nx, ny = vx / rr, vy / rr
    r = rr / 1000.0
    c = params["chroma_g8"]
    s = c[0] + c[1] * r + c[2] * r * r
    k = c[3] + c[4] * r
    phi = jnp.arctan2(ny, nx)
    # Optional radial blur terms appended after the 8 base slots: len 9 -> b0 + b1 r,
    # len 10 -> b0 + b1 r + b2 r^2 (2026-09-24). The length is static, so this is
    # resolved at trace time and a checkpoint carries its own model form.
    blur, dil, astig, sq, q1, q2 = c[5], c[6], None, None, None, None
    extras = tuple(getattr(ctx, "chroma_g8_extras", ()) or ())
    if extras:
        if int(c.shape[0]) != 8 + len(extras):
            raise ValueError(f"chroma_g8 has {int(c.shape[0])} values, extras {extras} need {8 + len(extras)}")
        x = dict(zip(extras, (c[8 + i] for i in range(len(extras)))))
        unknown = set(x) - set(G8_EXTRA_NAMES)
        if unknown:
            raise ValueError(f"unknown chroma_g8 extras {sorted(unknown)}")
        blur = blur + x.get("blur_r", 0.0) * r + x.get("blur_r2", 0.0) * r * r
        dil = dil + x.get("dil_r", 0.0) * r
        if "astig0" in x or "astig_r" in x:
            astig = x.get("astig0", 0.0) + x.get("astig_r", 0.0) * r
        # radial shift quadratic in colour: (delta^2 - <delta^2>) * (sq0 + sq1 r), along n_axis
        sq = _g8_poly(x, ("sq0", "sq1"), (1.0, r))
        if any(n in x for n in ("q1_x", "q1_y", "q2_x", "q2_y", "dilp_x", "dilp_y")):
            # detector planes: X, Y = (x_lin - 1024) / 1024, (y_lin - 1024) / 1024 at the slot's star
            X = (ctx.x_lin[star_occ] - 1024.0) / 1024.0
            Y = (ctx.y_lin[star_occ] - 1024.0) / 1024.0
        else:
            X = Y = None
        q1 = _g8_poly(x, ("q1_0", "q1_x", "q1_y"), (1.0, X, Y))   # colour elongation, aniso (x/y)
        q2 = _g8_poly(x, ("q2_0", "q2_x", "q2_y"), (1.0, X, Y))   # colour elongation, shear (45 deg)
        dilp = _g8_poly(x, ("dilp_x", "dilp_y"), (X, Y))          # colour round-width plane
        if dilp is not None:
            dil = dil + dilp
    else:
        for j in range(1, int(c.shape[0]) - 7):
            blur = blur + c[7 + j] * r ** j
    if ctx.chroma_g8_gauge not in ("mean", "raw"):
        raise ValueError(f"unknown chroma_g8_gauge {ctx.chroma_g8_gauge!r}")
    sfx = "_raw" if ctx.chroma_g8_gauge == "raw" else ""
    fields = {
        "kurt_plain" + sfx: delta * k,
        "blur" + sfx: delta * blur,
        "dilation" + sfx: delta * dil,
        "tre_a" + sfx: delta * c[7] * jnp.cos(3.0 * phi),
        "tre_b" + sfx: delta * c[7] * jnp.sin(3.0 * phi),
    }
    if ctx.chroma_g8_no_dil:
        del fields["dilation" + sfx]
    if astig is not None:
        # traceless stretch along the direction to the optical axis:
        # cos(2 phi) * aniso + sin(2 phi) * shear, amplitude astig0 + astig_r * r
        fields["aniso" + sfx] = delta * astig * jnp.cos(2.0 * phi)
        fields["shear" + sfx] = delta * astig * jnp.sin(2.0 * phi)
    # detector-frame elongation planes, added to the axis-oriented astig when both are present
    for name, q in (("aniso" + sfx, q1), ("shear" + sfx, q2)):
        if q is not None:
            fields[name] = fields[name] + delta * q if name in fields else delta * q
    if has_chroma_halo(params):
        fields["halo"] = params["chroma_halo"]
    if sq is None:
        return delta, delta * s * nx, delta * s * ny, fields
    shift = delta * s + (delta * delta - ctx.chroma_delta2_mean) * sq
    return delta, shift * nx, shift * ny, fields


def has_chroma_image(params: dict[str, jnp.ndarray]) -> bool:
    """True when the parameter set carries the free colour-image leaf."""
    return has_chroma(params) and all(k in params for k in CHROMA_IMAGE_LEAVES)


def chroma_slot_terms(
    params: dict[str, jnp.ndarray], ctx: StaticContext, star_occ: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]] | None:
    """Colour terms (``_colour_slot_terms``) plus the optional brightness-width blur
    (``bright_width.merge_slot_terms``: a no-op returning the same object without the leaf)."""
    return BW.merge_slot_terms(params, ctx, star_occ, _colour_slot_terms(params, ctx, star_occ))


def _colour_slot_terms(
    params: dict[str, jnp.ndarray], ctx: StaticContext, star_occ: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]] | None:
    """Per-occupied-slot chromatic quantities, or None when the term is off.

    Returns ``(delta_occ, shift_x_occ, shift_y_occ, field_terms)`` where the two
    shifts are already multiplied by the colour offset and are in physical pixels,
    constant in time, and ``field_terms`` is an ordered dict mapping generator name
    ("dilation", and when present "aniso"/"shear"/"kurt") to its raw per-node
    coefficient array ``params["chroma_<name>"]`` -- unlike the shift, these are
    NOT pre-blended here; the render step blends them with the same bilinear node
    weights it uses for the base ePSF (see ``_render_occ_templates``'s docstring).

    The bilinear node weights (used for the shift only) are evaluated at
    ``ctx.x_lin``/``ctx.y_lin``, the baked linear-WCS position, not at the fitted
    per-frame position. That keeps them exact constants, so no gradient leaks back
    into ``wcs_coeff`` through the colour term, and the positional error is a few
    pixels against a node spacing of ~500, i.e. nothing.
    """
    if has_chroma_g8(params):
        return _chroma_g8_slot_terms(params, ctx, star_occ)
    if not has_chroma(params):
        return None
    if ctx.chroma_delta is None:
        raise ValueError(
            "params carry chromatic leaves but the context has no colour; pass "
            "bp_rp into build_static_context. Refusing to default it to zero, "
            "which would silently disagree with the fitted model."
        )
    shift_nodes = params["chroma_shift"]  # (2, n_rows, n_cols)
    n_rows, n_cols = int(shift_nodes.shape[1]), int(shift_nodes.shape[2])
    delta = ctx.chroma_delta[star_occ]  # (n_occ,)
    i0, j0, wy, wx = EM.bilinear_cell(
        ctx.x_lin[star_occ], ctx.y_lin[star_occ], ctx.node_x, ctx.node_y
    )
    n_occ = int(star_occ.shape[0])
    Wn = EM.node_blend_weights(
        jnp.reshape(i0, (n_occ,)), jnp.reshape(j0, (n_occ,)),
        jnp.reshape(wy, (n_occ,)), jnp.reshape(wx, (n_occ,)),
        n_rows=n_rows, n_cols=n_cols,
    )
    shift = jnp.einsum("nrc,krc->kn", Wn, shift_nodes)  # (2, n_occ)
    field_terms = {"dilation": params["chroma_dilation"]}
    if has_chroma_affine(params):
        field_terms["aniso"] = params["chroma_aniso"]
        field_terms["shear"] = params["chroma_shear"]
        if has_chroma_kurt(params):
            field_terms["kurt"] = params["chroma_kurt"]
    if has_chroma_halo(params):
        field_terms["halo"] = params["chroma_halo"]
    if has_chroma_image(params):
        # Raw, not gauged: the gauge needs ``base``, which lives in the render
        # functions, and decoding there keeps one copy of the gauge per forward.
        field_terms[CHROMA_IMAGE_FIELD] = params["chroma_image"]
    return delta, delta * shift[0], delta * shift[1], field_terms


# Generator functions keyed by the same names used in ``chroma_slot_terms``'s
# ``field_terms`` dict -- shared by both render paths' chroma fold so adding a
# new generator (e.g. a future higher-order term) means adding one entry here,
# not touching either renderer's control flow.
CHROMA_FIELD_GENERATORS = {
    "dilation": EM.chroma_dilation_field,
    "aniso": EM.chroma_aniso_field,
    "shear": EM.chroma_shear_field,
    "kurt": EM.chroma_kurt_field,
    "halo": EM.chroma_halo_field,
    "kurt_plain": EM.chroma_kurt_plain_field,
    "blur": EM.chroma_blur_field,
    "tre_a": EM.chroma_trefoil_a_field,
    "tre_b": EM.chroma_trefoil_b_field,
    "kurt_plain_raw": EM.chroma_kurt_plain_field_raw,
    "blur_raw": EM.chroma_blur_field_raw,
    "dilation_raw": EM.chroma_dilation_field_raw,
    "tre_a_raw": EM.chroma_trefoil_a_field_raw,
    "tre_b_raw": EM.chroma_trefoil_b_field_raw,
    "aniso_raw": EM.chroma_aniso_field_raw,
    "shear_raw": EM.chroma_shear_field_raw,
}


def central_crop_stamps(arr: np.ndarray | jnp.ndarray, core: int) -> np.ndarray:
    """Crop central ``core×core`` from stamps shaped ``(..., S, S)``."""
    a = np.asarray(arr)
    s = int(a.shape[-1])
    c = int(core)
    if c >= s:
        return a
    if c % 2 == 0 or c < 1:
        raise ValueError(f"core stamp must be odd positive, got {c}")
    lo = (s - c) // 2
    hi = lo + c
    return a[..., lo:hi, lo:hi]


def build_dx_only_local_cache(
    params: dict[str, jnp.ndarray],
    ctx: StaticContext,
    *,
    blend_x: jnp.ndarray | None = None,
    blend_y: jnp.ndarray | None = None,
    recenter_n_iter: int = EM.HOTPATH_RECENTER_N_ITER,
) -> jnp.ndarray:
    """Core-centered local ePSF per occupied slot (stage-1 dx-only cache).

    Blends at stamp centers (or per-star ``blend_x/y``) with frozen PRF base.
    Shape ``(n_occ, G, G)``.
    """
    base = decoded_epsf_base(_epsf_params_for_context(params, ctx))
    flat_idx = ctx.members.reshape(-1)
    star_idx = flat_idx[ctx.occ_flat_idx]
    if blend_x is None:
        bx = ctx.stamp_center_x[ctx.occ_group]
        by = ctx.stamp_center_y[ctx.occ_group]
    else:
        bx_all = jnp.asarray(blend_x)
        by_all = jnp.asarray(blend_y)
        if bx_all.shape[0] == ctx.x_lin.shape[0]:
            bx = bx_all[star_idx]
            by = by_all[star_idx]
        else:
            bx = bx_all
            by = by_all
    i0, j0, wy, wx = EM.bilinear_cell(bx, by, ctx.node_x, ctx.node_y)
    local = EM.blend_field(base, i0, j0, wy, wx)
    return EM.recenter_grid_core(local, clip_nonneg=False, n_iter=recenter_n_iter)


def _render_occ_templates(
    ctx: StaticContext,
    x_occ: jnp.ndarray,
    y_occ: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    w_of_t: jnp.ndarray,
    *,
    base: jnp.ndarray,
    modes: jnp.ndarray | None,
    local_cache_arr: jnp.ndarray | None,
    n_pix: int,
    do_recenter: bool,
    recenter_n_iter: int,
    chroma_delta_occ: jnp.ndarray | None = None,
    chroma_fields: dict[str, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Render unit-flux occupied-slot template stamps for an arbitrary frame subset.

    ``x_occ``/``y_occ``/``dx``/``dy`` are ``(n_occ, n_frames_used)``; ``w_of_t``
    is ``(n_frames_used, n_modes)``. ``base``/``modes``/``local_cache_arr`` are
    frame-independent and must already be decoded by the caller.

    Chromatic node-field terms. ``chroma_delta_occ`` is ``(n_occ,)``, the per-slot
    colour offset ``c_i - c_ref``; ``chroma_fields`` maps generator name ("dilation",
    "aniso", "shear", "kurt") to its ``(n_rows, n_cols)`` node coefficient field, as
    returned by ``chroma_slot_terms``. Both None (the default, or an empty dict) means
    no chromatic term and a rendering path bit-identical to before -- the switch is a
    Python bool/dict-truthiness check, not a traced ``where``, so stages that do not
    train it pay nothing, and a run with only the original dilation term (no
    ``"aniso"``/``"shear"``/``"kurt"`` keys) renders bit-identically to before this
    generalization: the concatenation reduces to exactly one extra block, in the same
    order, with the same values.

    The chromatic SHIFT is not handled here: it rides on ``dx``/``dy``, which the
    caller has already offset.

    Each field term is applied to first order as an additive perturbation of the node
    grid, ``P -> P + delta * sum_name coeff_name * Generator_name[P]``. Every
    generator is linear, so it commutes with the node blend, which lets ALL of them
    be folded into the existing single contraction by concatenating onto the
    NODE-ROW axis rather than adding a separate ``(n_occ, G, G)`` intermediate per
    term. That matters: this bundle peaks at 93.6 % of an L4's memory. Crucially the
    same concatenated weights then feed ``node_moments``, so the analytic core
    centroid is the centroid of the grid actually rendered. Left unperturbed each
    term would drift the centroid by a per-star translation proportional to colour
    and to that term's own first moment -- for the dilation term this is precisely
    the signal ``chroma_shift`` measures; the pure-stretch/shear/kurt generators are
    each gauged core-dipole-free (see ``epsf_model.chroma_aniso_field`` et al.) so
    they do not reintroduce this for the new terms.

    Factored out of ``forward_model`` so the frame-block chunking path
    (``total_loss``'s ``stamp_chunk``) can run the exact same per-frame math on
    a contiguous frame subset without duplicating it: each frame's render is
    independent under ``vmap`` (no cross-frame reduction happens here), so
    calling this once per block and concatenating the blocks is bit-identical
    to calling it once on the full frame set.
    """
    n_occ = x_occ.shape[0]
    node_x = ctx.node_x
    node_y = ctx.node_y
    _chroma = chroma_delta_occ is not None and bool(chroma_fields)

    if local_cache_arr is not None:
        # The stage-1 cache is pre-blended and pre-recentered, so the per-star
        # node-grid DILATION cannot be expressed against it and is skipped here. That
        # is safe only because chroma is frozen at zero whenever this path runs, which
        # is asserted on concrete values at cache-build time in fit.run_stage -- it
        # cannot be checked here, where the leaves are tracers. The chromatic SHIFT
        # does still apply: it rides on dx/dy, which the caller has already offset.

        def render_cached(dx_t, dy_t):
            return EM.render_stamps(local_cache_arr, dx_t, dy_t, n_pix=n_pix)

        templates_occ_t = jax.vmap(render_cached, in_axes=(1, 1))(dx, dy)
        return jnp.moveaxis(templates_occ_t, 0, 1)

    g_size = int(base.shape[-1])
    # Capture as Python bools / ints so XLA constant-folds them.
    _do_recenter = bool(do_recenter)
    _n_iter = int(recenter_n_iter)
    # The analytic-COM derivation ((c) in EM.render_band's docstring) is only
    # exact for a single recenter shift; fall back to the exact gather-based
    # path for any other recenter_n_iter (nobody currently overrides this on
    # the hot path -- EM.HOTPATH_RECENTER_N_ITER is always 1 -- but keep the
    # fallback rather than silently mismatching a future caller).
    _use_band = USE_BANDED_RENDER and (not _do_recenter or _n_iter == 1)

    if _use_band:

        n_rows, n_cols = base.shape[0], base.shape[1]

        def one_frame(w_t, x_s, y_s, dx_t, dy_t):
            node_field = _node_field_for_frame(base, modes, w_t)
            i0, j0, wy, wx = EM.bilinear_cell(x_s, y_s, node_x, node_y)
            # Keep a leading occupied-slot axis even when n_occ == 1.
            wy = jnp.reshape(wy, (n_occ,))
            wx = jnp.reshape(wx, (n_occ,))
            i0 = jnp.reshape(i0, (n_occ,))
            j0 = jnp.reshape(j0, (n_occ,))
            # `local = blend_field(node_field, ...)` is mathematically the
            # same value, but blend_field's nested "(1-wy)*(...) + wy*(...)"
            # tree costs an extra full (n_occ, G, G)-sized backward residual
            # per corner term under vmap+grad (measured ~6.3 GB vs ~3.8 GB
            # for the one-contraction form below, at production shapes) --
            # blend_field itself is kept unchanged (it's a test reference and
            # still used elsewhere), this is a from-scratch equivalent.
            Wn = EM.node_blend_weights(i0, j0, wy, wx, n_rows=n_rows, n_cols=n_cols)
            if _chroma:
                # Fold every first-order field term onto the node-row axis so the
                # contraction below, and node_moments, stay textually unchanged and
                # lower to the same single GEMM (contraction dim 4 -> 4*(1+n_terms)).
                # Order matches chroma_fields' insertion order (dilation first, then
                # aniso/shear/kurt when present) -- with only "dilation" present this
                # is exactly the pre-C1 one-extra-block form, bit-identical to before.
                # CHROMA_IMAGE_FIELD is the one entry whose value is a raw (G, G)
                # image rather than a per-node coefficient: its field block is the
                # gauged image broadcast over every node (one global image), and its
                # weight block is Wn*delta with no coefficient. Since sum_rc Wn = 1,
                # blending a node-constant field returns the image exactly.
                gen_fields = [
                    jnp.broadcast_to(
                        EM.decode_chroma_image(coeff, base), node_field.shape
                    )
                    if name == CHROMA_IMAGE_FIELD
                    else CHROMA_FIELD_GENERATORS[name](node_field)
                    for name, coeff in chroma_fields.items()
                ]
                node_field = jnp.concatenate([node_field, *gen_fields], axis=0)
                Wn = jnp.concatenate(
                    [Wn] + [
                        (Wn * chroma_delta_occ[:, None, None])
                        if name == CHROMA_IMAGE_FIELD
                        else (Wn * coeff[:, None, None]) if coeff.ndim == 1
                        else Wn * coeff[None] * chroma_delta_occ[:, None, None]
                        for name, coeff in chroma_fields.items()
                    ],
                    axis=1,
                )
            local = jnp.einsum("nrc,rcxy->nxy", Wn, node_field)

            if _do_recenter:
                # (c): analytic per-slot core centroid -- three cheap scalar
                # blends of node-level core-weighted moments instead of a
                # core_centroid_xy pass over the (n_occ, G, G) `local` grid.
                # node_field/Wn are the concatenated set when chroma is live, so
                # this is the centroid of the grid actually rendered.
                nsum, nxmom, nymom = EM.node_moments(node_field)
                total = jnp.einsum("nrc,rc->n", Wn, nsum) + 1e-12
                xmom = jnp.einsum("nrc,rc->n", Wn, nxmom)
                ymom = jnp.einsum("nrc,rc->n", Wn, nymom)
                cx = xmom / total
                cy = ymom / total
            else:
                cx = jnp.zeros((n_occ,), dtype=local.dtype)
                cy = jnp.zeros((n_occ,), dtype=local.dtype)

            # (a)+(b): fold render + COM-recenter into one banded matrix/axis.
            Ay = EM.render_band(dy_t, cy, n_pix=n_pix, n_grid=g_size)
            Ax = EM.render_band(dx_t, cx, n_pix=n_pix, n_grid=g_size)
            half = jnp.einsum("nsg,ngh->nsh", Ay, local)
            stamps = jnp.einsum("nsh,nth->nst", half, Ax)
            if _do_recenter:
                # (d): the renorm is a scalar, but must not be dropped -- it
                # is only exactly absorbed by the downstream flux solve at
                # ridge=0, and flux_solve uses ridge=1e-6.
                renorm = EM.renorm_scalar(local, cx, cy) + 1e-12
                stamps = stamps / renorm[:, None, None]
            return stamps

    else:

        def one_frame(w_t, x_s, y_s, dx_t, dy_t):
            node_field = _node_field_for_frame(base, modes, w_t)
            i0, j0, wy, wx = EM.bilinear_cell(x_s, y_s, node_x, node_y)
            # Keep a leading occupied-slot axis even when n_occ == 1.
            wy = jnp.reshape(wy, (n_occ,))
            wx = jnp.reshape(wx, (n_occ,))
            i0 = jnp.reshape(i0, (n_occ,))
            j0 = jnp.reshape(j0, (n_occ,))
            local = EM.blend_field(node_field, i0, j0, wy, wx)
            local = jnp.reshape(local, (n_occ, g_size, g_size))
            if _chroma:
                # Deliberately written as an explicit per-term sum rather than the
                # hot path's node-row concat, so the banded-vs-unbanded A/B test
                # compares two genuinely independent implementations. Mathematically
                # identical: sum_name blend(coeff_name * Generator_name[P]) * delta
                # == the concat form (linearity of blend + each generator).
                for name, coeff in chroma_fields.items():
                    if name == CHROMA_IMAGE_FIELD:
                        # One global image: blending a node-constant field is the
                        # identity, so the blend is skipped outright here. That makes
                        # this fallback a genuinely independent implementation of the
                        # hot path's broadcast-then-contract form.
                        local = local + chroma_delta_occ[:, None, None] * (
                            EM.decode_chroma_image(coeff, base)[None]
                        )
                        continue
                    if coeff.ndim == 1:   # per-slot weight (chroma_g8), delta included
                        gen = CHROMA_FIELD_GENERATORS[name](node_field)
                        local_gen = jnp.reshape(
                            EM.blend_field(gen, i0, j0, wy, wx), (n_occ, g_size, g_size)
                        )
                        local = local + coeff[:, None, None] * local_gen
                        continue
                    gen = CHROMA_FIELD_GENERATORS[name](node_field) * coeff[..., None, None]
                    local_gen = jnp.reshape(
                        EM.blend_field(gen, i0, j0, wy, wx), (n_occ, g_size, g_size)
                    )
                    local = local + chroma_delta_occ[:, None, None] * local_gen
            if _do_recenter:
                local = EM.recenter_grid_core(
                    local, clip_nonneg=False, n_iter=_n_iter,
                )
            return EM.render_stamps(local, dx_t, dy_t, n_pix=n_pix)

    templates_occ_t = jax.vmap(one_frame, in_axes=(0, 1, 1, 1, 1))(
        w_of_t, x_occ, y_occ, dx, dy,
    )
    return jnp.moveaxis(templates_occ_t, 0, 1)


def _render_occ_templates_packed(
    ctx: StaticContext,
    x_occ: jnp.ndarray,
    y_occ: jnp.ndarray,
    w_of_t: jnp.ndarray,
    *,
    base: jnp.ndarray,
    modes: jnp.ndarray,
    do_recenter: bool,
    recenter_n_iter: int,
    x_render: jnp.ndarray | None = None,
    y_render: jnp.ndarray | None = None,
    chroma_delta_occ: jnp.ndarray | None = None,
    chroma_fields: dict[str, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Render unit-flux packed templates ``(n_occ, n_frames, P)`` via block-sum.

    Pixel lists come from ``ctx.pix_*`` indexed by ``ctx.occ_group``.
    Offsets: ``ox = pix_x - x_star(t)`` (physics-matched to ``render_stamps``).

    ``x_render``/``y_render`` default to ``x_occ``/``y_occ`` and exist to carry the
    chromatic shift. Unlike the square path, which already keeps the two roles apart
    in separate arguments, here ``x_occ`` feeds BOTH the node blend, which must use
    the star's true field position, and the render offset, which must carry the
    chromatic displacement. They have to be threaded separately or the colour term
    would also move the star between ePSF nodes.

    ``chroma_delta_occ``/``chroma_fields`` are the first-order field terms, folded
    onto the node-row axis exactly as in ``_render_occ_templates``; see its
    docstring for why the concatenation form is the memory-safe and the correct one.
    """
    n_occ = int(x_occ.shape[0])
    if x_render is None:
        x_render = x_occ
    if y_render is None:
        y_render = y_occ
    _chroma = chroma_delta_occ is not None and bool(chroma_fields)
    p_tier = int(ctx.pix_x.shape[-1])
    node_x = ctx.node_x
    node_y = ctx.node_y
    g_size = int(base.shape[-1])
    n_rows, n_cols = base.shape[0], base.shape[1]
    _do_recenter = bool(do_recenter)
    _n_iter = int(recenter_n_iter)
    # See EM.render_packed_pixels_banded's docstring: the analytic-COM fold is
    # only exact for a single recenter shift; fall back to the exact
    # gather-based path for any other recenter_n_iter (mirrors
    # _render_occ_templates's identical guard for the square path --
    # EM.HOTPATH_RECENTER_N_ITER is always 1 on the hot path today).
    _use_band = USE_BANDED_RENDER_PACKED and (not _do_recenter or _n_iter == 1)

    pix_x_occ = ctx.pix_x[ctx.occ_group]  # (n_occ, P)
    pix_y_occ = ctx.pix_y[ctx.occ_group]
    pix_valid_occ = ctx.pix_valid[ctx.occ_group]

    if _use_band:
        # Static (param-independent) per-occupied-slot integer reference so
        # pix_*_index below is an exact integer offset from it -- any integer
        # works (see EM.render_packed_pixels_banded's derivation), the mean of
        # the slot's own valid pixels just keeps off_x/off_y small for fp32
        # stability, same role as render_stamps' rounded stamp center.
        denom = jnp.sum(pix_valid_occ, axis=-1) + 1e-6
        ref_x = jnp.round(jnp.sum(pix_x_occ * pix_valid_occ, axis=-1) / denom)
        ref_y = jnp.round(jnp.sum(pix_y_occ * pix_valid_occ, axis=-1) / denom)
        p_index_x = pix_x_occ - ref_x[:, None]
        p_index_y = pix_y_occ - ref_y[:, None]

        def one_frame(w_t, x_s, y_s, xr_s, yr_s):
            node_field = _node_field_for_frame(base, modes, w_t)
            # Node blend uses the star's TRUE field position, never the
            # chromatically displaced one.
            i0, j0, wy, wx = EM.bilinear_cell(x_s, y_s, node_x, node_y)
            wy = jnp.reshape(wy, (n_occ,))
            wx = jnp.reshape(wx, (n_occ,))
            i0 = jnp.reshape(i0, (n_occ,))
            j0 = jnp.reshape(j0, (n_occ,))
            Wn = EM.node_blend_weights(i0, j0, wy, wx, n_rows=n_rows, n_cols=n_cols)
            if _chroma:
                # See _render_occ_templates's banded branch for the derivation;
                # order matches chroma_fields' insertion order, so a dilation-only
                # run renders bit-identically to the pre-C1 single-extra-block form.
                # CHROMA_IMAGE_FIELD is the one entry whose value is a raw (G, G)
                # image rather than a per-node coefficient: its field block is the
                # gauged image broadcast over every node (one global image), and its
                # weight block is Wn*delta with no coefficient. Since sum_rc Wn = 1,
                # blending a node-constant field returns the image exactly.
                gen_fields = [
                    jnp.broadcast_to(
                        EM.decode_chroma_image(coeff, base), node_field.shape
                    )
                    if name == CHROMA_IMAGE_FIELD
                    else CHROMA_FIELD_GENERATORS[name](node_field)
                    for name, coeff in chroma_fields.items()
                ]
                node_field = jnp.concatenate([node_field, *gen_fields], axis=0)
                Wn = jnp.concatenate(
                    [Wn] + [
                        (Wn * chroma_delta_occ[:, None, None])
                        if name == CHROMA_IMAGE_FIELD
                        else (Wn * coeff[:, None, None]) if coeff.ndim == 1
                        else Wn * coeff[None] * chroma_delta_occ[:, None, None]
                        for name, coeff in chroma_fields.items()
                    ],
                    axis=1,
                )
            local = jnp.einsum("nrc,rcxy->nxy", Wn, node_field)

            if _do_recenter:
                # Analytic per-slot core centroid, same three cheap scalar
                # blends of node-level moments used by _render_occ_templates's
                # banded square path -- no (n_occ, G, G) core_centroid_xy pass.
                # Uses the concatenated set when chroma is live, so this is the
                # centroid of the grid actually rendered.
                nsum, nxmom, nymom = EM.node_moments(node_field)
                total = jnp.einsum("nrc,rc->n", Wn, nsum) + 1e-12
                xmom = jnp.einsum("nrc,rc->n", Wn, nxmom)
                ymom = jnp.einsum("nrc,rc->n", Wn, nymom)
                cx = xmom / total
                cy = ymom / total
            else:
                cx = jnp.zeros((n_occ,), dtype=local.dtype)
                cy = jnp.zeros((n_occ,), dtype=local.dtype)

            # Render offset carries the chromatic shift.
            off_x = xr_s - ref_x
            off_y = yr_s - ref_y
            stamps = EM.render_packed_pixels_banded(
                local, p_index_x, p_index_y, off_x, off_y, cx, cy,
            )
            if _do_recenter:
                # Must not be dropped: only exactly absorbed by the flux
                # solve at ridge=0, and flux_solve uses ridge=1e-6 (see
                # EM.renorm_scalar's docstring).
                renorm = EM.renorm_scalar(local, cx, cy) + 1e-12
                stamps = stamps / renorm[:, None]
            return stamps * pix_valid_occ

    else:

        def one_frame(w_t, x_s, y_s, xr_s, yr_s):
            node_field = _node_field_for_frame(base, modes, w_t)
            i0, j0, wy, wx = EM.bilinear_cell(x_s, y_s, node_x, node_y)
            wy = jnp.reshape(wy, (n_occ,))
            wx = jnp.reshape(wx, (n_occ,))
            i0 = jnp.reshape(i0, (n_occ,))
            j0 = jnp.reshape(j0, (n_occ,))
            Wn = EM.node_blend_weights(i0, j0, wy, wx, n_rows=n_rows, n_cols=n_cols)
            local = jnp.einsum("nrc,rcxy->nxy", Wn, node_field)
            if _chroma:
                # Explicit per-term form, as in the square fallback: an independent
                # implementation of the same maths, so the A/B test has teeth.
                for name, coeff in chroma_fields.items():
                    if name == CHROMA_IMAGE_FIELD:
                        # See the square fallback: node-constant field, blend is a no-op.
                        local = local + chroma_delta_occ[:, None, None] * (
                            EM.decode_chroma_image(coeff, base)[None]
                        )
                        continue
                    if coeff.ndim == 1:   # per-slot weight (chroma_g8), delta included
                        gen = CHROMA_FIELD_GENERATORS[name](node_field)
                        local = local + coeff[:, None, None] * jnp.einsum("nrc,rcxy->nxy", Wn, gen)
                        continue
                    gen = CHROMA_FIELD_GENERATORS[name](node_field) * coeff[..., None, None]
                    local_gen = jnp.einsum("nrc,rcxy->nxy", Wn, gen)
                    local = local + chroma_delta_occ[:, None, None] * local_gen
            if _do_recenter:
                local = EM.recenter_grid_core(local, clip_nonneg=False, n_iter=_n_iter)
            ox = pix_x_occ - xr_s[:, None]
            oy = pix_y_occ - yr_s[:, None]
            return EM.render_physical_pixels_blocksum(
                local, ox, oy, valid=pix_valid_occ,
            )

    # w_of_t: (T, n_modes); x_occ/y_occ/x_render/y_render: (n_occ, T)
    templates_occ_t = jax.vmap(one_frame, in_axes=(0, 1, 1, 1, 1))(
        w_of_t, x_occ, y_occ, x_render, y_render
    )
    # (T, n_occ, P) → (n_occ, T, P)
    return jnp.moveaxis(templates_occ_t, 0, 1)


# ---------------------------------------------------------------------------
# Task PW: mode-derivative ("probe") rendering, so the temporal amplitudes can
# be profiled out in closed form per frame (see
# ``flux_solve.solve_group_fluxes_profile_w``).
#
# The rendered unit-flux template is linear in the per-frame scalars ``w_k(t)``
# to ~1e-5, so ``T(w) = A + sum_m w_m B_m`` with ``A`` the render at ``w = 0``
# and ``B_m`` the render at ``w = e_m`` minus ``A``. Rather than write a second
# renderer, the frame axis is *augmented*: every real frame is repeated
# ``n_modes + 1`` times with the same star positions and the probe ``w`` rows
# ``[0, e_1, ..., e_M]``, the existing vmapped renderer runs unchanged, and the
# result is unstacked below. Cost is therefore exactly ``(n_modes + 1)x`` the
# render, and A/B are guaranteed to come from the *same* code path (including
# the core-recenter and its renorm) as the ordinary template.
# ---------------------------------------------------------------------------


def _mode_probe_w(
    n_frames: int, n_modes: int, dtype, center: jnp.ndarray | None = None,
    scale: float = 1.0,
) -> jnp.ndarray:
    """``(n_frames * (n_modes + 1), n_modes)`` probe rows
    ``[0, scale*e_1, .., scale*e_M]`` repeated once per frame, frame-major
    (matching ``jnp.repeat(..., axis=frame)``).

    ``center`` ``(n_frames, n_modes)`` shifts every probe row of frame ``t`` by
    ``center[t]``, so the render is linearised about ``w = center`` instead of
    about ``w = 0``: ``templates`` becomes the render AT ``center`` and the
    derivative stack is the secant from there.  ``None`` keeps the ``w = 0``
    linearisation.

    ``scale`` is the SECANT STEP.  The render is not exactly linear in ``w``:
    ``base + sum_k w_k mode_k`` enters ``local`` linearly, but the core
    recenter's shift ``(cx, cy)`` is itself linear in ``w`` and multiplies
    ``local`` in the interpolation, and ``renorm_scalar`` divides by another
    ``w``-dependent scalar -- so the true dependence is quadratic/rational,
    weakly.  A unit step is a good conditioned default (the difference is far
    above fp32 noise) and the secant it gives reproduces the true render at
    the trained amplitude to 1.9e-5 of peak, but it carries the quadratic term
    into the slope.  Measured consequence on real S52 data: the Gauss-Newton
    iteration in ``diagnostics/pw_solve_w.py`` converges LINEARLY at rate
    ~0.51 for the second mode instead of quadratically (the residual, and
    hence the fixed point, is exact -- only the Jacobian is biased, so the
    iteration converges to the right answer, just slowly).  Setting ``scale``
    to the physical amplitude scale (~1e-5 for this model) removes that bias;
    it is left at 1.0 by default so nothing changes for existing callers.
    """
    eye = jnp.eye(n_modes, dtype=dtype)
    if scale != 1.0:
        eye = eye * jnp.asarray(scale, dtype=dtype)
    probe = jnp.concatenate([jnp.zeros((1, n_modes), dtype=dtype), eye], axis=0)
    tiled = jnp.tile(probe, (n_frames, 1))
    if center is None:
        return tiled
    return tiled + jnp.repeat(jnp.asarray(center, dtype=dtype), n_modes + 1, axis=0)


def _split_mode_probe(
    rendered: jnp.ndarray, n_frames: int, n_modes: int, scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(n_occ, n_frames*(M+1), *spatial)`` -> ``(A, B)``.

    ``A`` is ``(n_occ, n_frames, *spatial)`` (the ``w = 0`` render) and ``B``
    is ``(M, n_occ, n_frames, *spatial)`` (the per-mode derivative).
    """
    shape = rendered.shape
    blocked = rendered.reshape(shape[0], n_frames, n_modes + 1, *shape[2:])
    A = blocked[:, :, 0]
    if n_modes == 0:
        B = jnp.zeros((0,) + A.shape, dtype=A.dtype)
    else:
        B = jnp.moveaxis(blocked[:, :, 1:], 2, 0) - A[None]
        if scale != 1.0:
            B = B / jnp.asarray(scale, dtype=B.dtype)
    return A, B


def _solve_block_model(
    templates: jnp.ndarray,
    mode_templates: jnp.ndarray | None,
    data_i: jnp.ndarray,
    iv_i: jnp.ndarray,
    weights: "LossWeights",
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Run the configured profile-likelihood solve and return ``(model, w_solved)``.

    ``w_solved`` is ``None`` unless ``weights.profile_w`` is on, in which case
    it is the per-frame ``(n_frames, n_modes)`` closed-form amplitude solved
    jointly with every stamp flux (and pedestal). Shared by the unchunked and
    both chunked data-term paths so the three cannot drift apart.
    """
    if weights.profile_w:
        flux, ped, w_solved = FS.solve_group_fluxes_profile_w(
            templates, mode_templates, data_i, iv_i,
            ridge=weights.ridge,
            ridge_w=weights.ridge_w,
            pedestal=weights.stamp_pedestal,
            iterations=weights.profile_w_iters,
            flux_objective=weights.flux_objective,
            huber_delta=weights.huber_delta,
            irls_iterations=weights.huber_irls_iters,
        )
        T = FS.templates_at_w(templates, mode_templates, w_solved)
        if weights.stamp_pedestal:
            return FS.model_stamps_with_pedestal(T, flux, ped), w_solved
        return FS.model_stamps(T, flux), w_solved
    flux = FS.solve_fluxes(
        templates, data_i, iv_i,
        flux_objective=weights.flux_objective,
        ridge=weights.ridge,
        huber_delta=weights.huber_delta,
        irls_iterations=weights.huber_irls_iters,
        pedestal=weights.stamp_pedestal,
    )
    if weights.stamp_pedestal:
        flux, ped = flux
        return FS.model_stamps_with_pedestal(templates, flux, ped), None
    return FS.model_stamps(templates, flux), None


def forward_model(
    params: dict[str, jnp.ndarray],
    ctx: StaticContext,
    *,
    stamp_active: jnp.ndarray | None = None,
    local_cache: jnp.ndarray | None = None,
    n_pix: int | None = None,
    recenter_n_iter: int = EM.HOTPATH_RECENTER_N_ITER,
    do_recenter: bool = True,
    frame_idx: jnp.ndarray | None = None,
    group_idx: jnp.ndarray | None = None,
    w_of_t_override: jnp.ndarray | None = None,
    return_mode_templates: bool = False,
    w_probe_center: jnp.ndarray | None = None,
    w_probe_scale: float = 1.0,
):
    """Returns (templates, positions_x, positions_y, w_of_t).

    Square path templates: (n_groups, K, n_frames, S, S).
    Packed path templates: (n_groups, K, n_frames, P).

    ``w_of_t_override``: pre-gauged ``w_of_t`` rows for exactly this context's
    frames. Required by frame-block gradient accumulation (see
    ``fit.make_accum_step_fn``): the zero-time-mean gauge must be taken over
    the *whole* orbit, so a caller evaluating one frame block at a time must
    compute the gauge on the full frame set and pass the block's slice here.
    Passing ``None`` keeps the original behavior (gauge over ``ctx``'s frames).

    ``return_mode_templates`` (task PW): return a 5-tuple ``(templates, x, y,
    w_of_t, mode_templates)`` in which ``templates`` is the render at
    ``w = 0`` (NOT at ``w_of_t``) and ``mode_templates`` is
    ``(n_modes, n_groups, K, n_frames, *spatial)``, the derivative of the
    template with respect to each ``w_m``. Reconstruct any amplitude with
    ``flux_solve.templates_at_w``. ``w_of_t`` is still returned for
    compatibility but is not the amplitude the templates were rendered at.
    Costs ``(n_modes + 1)x`` the render (see ``_mode_probe_w``).

    ``w_probe_center`` ``(n_frames, n_modes)``: linearise about this amplitude
    instead of about zero, i.e. ``templates`` is the render AT the centre and
    ``mode_templates`` the secant from there, so
    ``templates_at_w(templates, mode_templates, dw)`` is the render at
    ``centre + dw``. Only meaningful with ``return_mode_templates=True``.

    ``w_probe_scale``: the secant step used to build ``mode_templates`` (see
    ``_mode_probe_w``). 1.0 (default) is the historical, well-conditioned
    choice; a value near the physical amplitude scale removes the quadratic
    bias in the returned slope.
    """
    n_groups, K = ctx.members.shape
    n_frames_full = ctx.wcs_frame_basis.shape[0]
    packed = bool(ctx.is_packed)
    if stamp_active is None:
        stamp_active = ctx.stamp_active

    # Direct callers may hand us a mixed context (useful for small tests and
    # diagnostics).  Production batching splits these rows in train_loop, so
    # it does not pay for two renders.  The split here preserves the exact
    # group-specific stop-gradient contract for the fallback path.
    contributor = np.asarray(ctx.is_epsf_contributor, dtype=bool)
    if contributor.size and np.any(contributor) and not np.all(contributor):
        bright_ctx = replace(ctx, is_epsf_contributor=jnp.ones_like(ctx.is_epsf_contributor, dtype=bool))
        faint_ctx = replace(ctx, is_epsf_contributor=jnp.zeros_like(ctx.is_epsf_contributor, dtype=bool))
        bright = forward_model(params, bright_ctx, stamp_active=stamp_active,
                               local_cache=local_cache, n_pix=n_pix,
                               recenter_n_iter=recenter_n_iter, do_recenter=do_recenter,
                               frame_idx=frame_idx, group_idx=group_idx,
                               w_of_t_override=w_of_t_override,
                               return_mode_templates=return_mode_templates,
                               w_probe_center=w_probe_center,
                               w_probe_scale=w_probe_scale)
        faint = forward_model(params, faint_ctx, stamp_active=stamp_active,
                              local_cache=local_cache, n_pix=n_pix,
                              recenter_n_iter=recenter_n_iter, do_recenter=do_recenter,
                              frame_idx=frame_idx, group_idx=group_idx,
                              w_of_t_override=w_of_t_override,
                              return_mode_templates=return_mode_templates,
                              w_probe_center=w_probe_center,
                              w_probe_scale=w_probe_scale)
        shape = (contributor.size,) + (1,) * (bright[0].ndim - 1)
        keep = jnp.asarray(contributor).reshape(shape)
        templates = jnp.where(keep, bright[0], faint[0])
        if return_mode_templates:
            # Same group-wise select, one axis further in (leading mode axis).
            mode_templates = jnp.where(keep[None], bright[4], faint[4])
            return templates, bright[1], bright[2], bright[3], mode_templates
        return templates, bright[1], bright[2], bright[3]
    epsf_params = _epsf_params_for_context(params, ctx)

    x_t, y_t = CW.eval_all_positions(
        ctx.x_lin,
        ctx.y_lin,
        ctx.cheb_basis,
        params["wcs_coeff"],
        ctx.wcs_frame_basis,
        ctx.n_terms,
    )

    if w_of_t_override is not None:
        # Caller already applied the whole-orbit zero-mean gauge.
        w_of_t = w_of_t_override
    else:
        w_of_t = w_field_from_coeff(epsf_params["w_coeff"], ctx.w_frame_basis)

    flat_idx = ctx.members.reshape(-1)
    occ = ctx.occ_flat_idx
    n_occ = int(occ.shape[0])
    star_occ = flat_idx[occ]
    x_occ = jnp.reshape(x_t[star_occ], (n_occ, n_frames_full))
    y_occ = jnp.reshape(y_t[star_occ], (n_occ, n_frames_full))

    base = decoded_epsf_base(epsf_params)
    g_size = int(base.shape[-1])
    if n_pix is None:
        # Representation-aware: (g_size - 2*PAD) // OVERSAMPLE only inverts
        # the LEGACY sub-pixel grid size; the default grid is now the
        # pixel-integrated one, 3 samples narrower for the same stamp_physical
        # -- EM.stamp_physical_from_node_size detects which formula applies.
        n_pix = EM.stamp_physical_from_node_size(g_size)

    if frame_idx is not None:
        frame_keep = jnp.zeros((n_frames_full,), dtype=jnp.float32).at[frame_idx].set(1.0)
    else:
        frame_keep = jnp.ones((n_frames_full,), dtype=jnp.float32)
    if group_idx is not None:
        group_keep = jnp.zeros((n_groups,), dtype=jnp.float32).at[group_idx].set(1.0)
    else:
        group_keep = jnp.ones((n_groups,), dtype=jnp.float32)
    occ_keep = group_keep[ctx.occ_group]

    # Read chroma from the ORIGINAL params, not the anchor-barriered copy: the
    # colour term is meant to see anchor gradients (see _epsf_params_for_context).
    chroma = chroma_slot_terms(params, ctx, star_occ)

    if packed:
        if local_cache is not None:
            raise NotImplementedError("dx-only local_cache is not supported on the packed path")
        modes = decoded_epsf_modes(epsf_params)
        if chroma is None:
            x_render = y_render = chroma_delta_occ = chroma_fields = None
        else:
            chroma_delta_occ, chx, chy, chroma_fields = chroma
            x_render = x_occ + chx[:, None]
            y_render = y_occ + chy[:, None]
        if return_mode_templates:
            n_modes = int(modes.shape[0])
            rep = n_modes + 1
            aug = _render_occ_templates_packed(
                ctx,
                jnp.repeat(x_occ, rep, axis=1), jnp.repeat(y_occ, rep, axis=1),
                _mode_probe_w(int(n_frames_full), n_modes, x_occ.dtype,
                              w_probe_center, w_probe_scale),
                base=base,
                modes=modes,
                do_recenter=do_recenter,
                recenter_n_iter=recenter_n_iter,
                x_render=None if x_render is None else jnp.repeat(x_render, rep, axis=1),
                y_render=None if y_render is None else jnp.repeat(y_render, rep, axis=1),
                chroma_delta_occ=chroma_delta_occ,
                chroma_fields=chroma_fields,
            )
            templates_occ, mode_occ = _split_mode_probe(
                aug, int(n_frames_full), n_modes, w_probe_scale)
        else:
            mode_occ = None
            templates_occ = _render_occ_templates_packed(
                ctx, x_occ, y_occ, w_of_t,
                base=base,
                modes=modes,
                do_recenter=do_recenter,
                recenter_n_iter=recenter_n_iter,
                x_render=x_render,
                y_render=y_render,
                chroma_delta_occ=chroma_delta_occ,
                chroma_fields=chroma_fields,
            )
        templates_occ = templates_occ * occ_keep[:, None, None] * frame_keep[None, :, None]
        P = int(ctx.pix_x.shape[-1])
        templates = jnp.zeros((n_groups, K, n_frames_full, P), dtype=templates_occ.dtype)
        templates = templates.at[ctx.occ_group, ctx.occ_slot].set(templates_occ)
        if return_mode_templates:
            mode_occ = mode_occ * occ_keep[None, :, None, None] * frame_keep[None, None, :, None]
            mode_templates = jnp.zeros(
                (mode_occ.shape[0], n_groups, K, n_frames_full, P), dtype=mode_occ.dtype
            )
            mode_templates = mode_templates.at[:, ctx.occ_group, ctx.occ_slot].set(mode_occ)
    else:
        cx_occ = ctx.stamp_center_x[ctx.occ_group]
        cy_occ = ctx.stamp_center_y[ctx.occ_group]
        dx = x_occ - cx_occ[:, None]
        dy = y_occ - cy_occ[:, None]
        if chroma is None:
            chroma_delta_occ = chroma_fields = None
        else:
            chroma_delta_occ, chx, chy, chroma_fields = chroma
            # The square path already keeps the node-blend position (x_occ) and the
            # render offset (dx) in separate arguments, so the chromatic shift is
            # simply added here and cannot move the star between ePSF nodes.
            dx = dx + chx[:, None]
            dy = dy + chy[:, None]

        if local_cache is not None:
            local_cache_arr = jnp.reshape(local_cache, (n_occ, g_size, g_size))
            modes = None
        else:
            local_cache_arr = None
            modes = decoded_epsf_modes(epsf_params)

        if return_mode_templates:
            if modes is None:
                raise NotImplementedError(
                    "return_mode_templates needs the ePSF modes; it is "
                    "incompatible with the dx-only local_cache fast path"
                )
            n_modes = int(modes.shape[0])
            rep = n_modes + 1
            aug = _render_occ_templates(
                ctx,
                jnp.repeat(x_occ, rep, axis=1), jnp.repeat(y_occ, rep, axis=1),
                jnp.repeat(dx, rep, axis=1), jnp.repeat(dy, rep, axis=1),
                _mode_probe_w(int(n_frames_full), n_modes, x_occ.dtype,
                              w_probe_center, w_probe_scale),
                base=base,
                modes=modes,
                local_cache_arr=local_cache_arr,
                n_pix=n_pix,
                do_recenter=do_recenter,
                recenter_n_iter=recenter_n_iter,
                chroma_delta_occ=chroma_delta_occ,
                chroma_fields=chroma_fields,
            )
            templates_occ, mode_occ = _split_mode_probe(
                aug, int(n_frames_full), n_modes, w_probe_scale)
        else:
            mode_occ = None
            templates_occ = _render_occ_templates(
                ctx, x_occ, y_occ, dx, dy, w_of_t,
                base=base,
                modes=modes,
                local_cache_arr=local_cache_arr,
                n_pix=n_pix,
                do_recenter=do_recenter,
                recenter_n_iter=recenter_n_iter,
                chroma_delta_occ=chroma_delta_occ,
                chroma_fields=chroma_fields,
            )
        templates_occ = templates_occ * occ_keep[:, None, None, None] * frame_keep[None, :, None, None]
        S = templates_occ.shape[-1]
        templates = jnp.zeros((n_groups, K, n_frames_full, S, S), dtype=templates_occ.dtype)
        templates = templates.at[ctx.occ_group, ctx.occ_slot].set(templates_occ)
        if return_mode_templates:
            mode_occ = (
                mode_occ
                * occ_keep[None, :, None, None, None]
                * frame_keep[None, None, :, None, None]
            )
            mode_templates = jnp.zeros(
                (mode_occ.shape[0], n_groups, K, n_frames_full, S, S), dtype=mode_occ.dtype
            )
            mode_templates = mode_templates.at[:, ctx.occ_group, ctx.occ_slot].set(mode_occ)

    x_slot = x_t[flat_idx].reshape(n_groups, K, n_frames_full)
    y_slot = y_t[flat_idx].reshape(n_groups, K, n_frames_full)
    if return_mode_templates:
        return templates, x_slot, y_slot, w_of_t, mode_templates
    return templates, x_slot, y_slot, w_of_t


def node_coord_grid() -> np.ndarray:
    """1D physical-px sample coordinates (shared by x and y axes)."""
    return np.asarray(EM.node_coord_1d(dtype=jnp.float32), dtype=float)


def centroid_penalty(params: dict[str, jnp.ndarray], w_of_t: jnp.ndarray) -> jnp.ndarray:
    """Residual monitor after hard composite core recenter (should be ~0)."""
    base = decoded_epsf_base(params)
    modes = decoded_epsf_modes(params)
    if w_of_t.ndim == 2:
        composite = base[:, :, None, :, :] + jnp.einsum("kijxy,tk->ijtxy", modes, w_of_t)
    else:
        # Spatial w_coeff: w_of_t is already (T, n_rows, n_cols, n_modes), the
        # per-node amplitude curve at each node's OWN position -- no broadcast
        # over (i, j) needed, unlike the legacy scalar-per-frame case above.
        composite = base[:, :, None, :, :] + jnp.einsum("kijxy,tijk->ijtxy", modes, w_of_t)
    n_rows, n_cols, n_frames = composite.shape[0], composite.shape[1], composite.shape[2]
    g_size = composite.shape[-1]
    flat = composite.reshape(n_rows * n_cols * n_frames, g_size, g_size)
    flat = EM.recenter_grid_core(flat, clip_nonneg=False, n_iter=EM.HOTPATH_RECENTER_N_ITER)
    cx, cy = EM.core_centroid_xy(flat)
    return jnp.mean(cx**2 + cy**2)


def params_fingerprint(params: dict) -> str:
    """Stable SHA-1 over the optimizer leaves, for cache provenance.

    ``diagnostics/flux_cache.solve_fluxes`` used to return any ``flux_solved.npz``
    sitting beside the requested params file, whatever produced it, so a diagnostic
    could report "fluxes for checkpoint X" while actually showing checkpoint Y's.
    Writing this fingerprint alongside the fluxes makes that detectable.
    """
    import hashlib  # noqa: PLC0415 -- keep module import cost off the train path

    h = hashlib.sha1()
    for key in sorted(params):
        arr = np.ascontiguousarray(np.asarray(params[key], dtype=np.float32))
        h.update(key.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _mean_or_zero(x: jnp.ndarray) -> jnp.ndarray:
    """``jnp.mean`` that returns 0 for an empty array instead of NaN.

    K=0 (no temporal ePSF mode) makes ``epsf_modes``/``w_coeff`` zero-sized, and a
    plain ``jnp.mean`` over an empty axis is NaN. Every penalty below is added into
    ``total_loss`` unconditionally with a non-zero lambda, so a single NaN here makes
    the whole loss NaN from step 0 -- silently, on a paid GPU.
    """
    return jnp.where(x.size == 0, jnp.zeros((), x.dtype), jnp.mean(x))


def flux_neutral_penalty(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    mode_sums = jnp.sum(decoded_epsf_modes(params), axis=(-1, -2))
    return _mean_or_zero(mode_sums**2)


def pixel_laplacian_penalty(field: jnp.ndarray) -> jnp.ndarray:
    lap = (
        field[..., 2:, 1:-1]
        + field[..., :-2, 1:-1]
        + field[..., 1:-1, 2:]
        + field[..., 1:-1, :-2]
        - 4 * field[..., 1:-1, 1:-1]
    )
    return jnp.mean(lap**2)


def mode_prior_penalty(params: dict[str, jnp.ndarray], modes_init: jnp.ndarray) -> jnp.ndarray:
    modes = decoded_epsf_modes(params)
    # Apply the same (live-base) decode to modes_init, or the prior fights the gauge:
    # since decode_epsf_modes projects off base, comparing against a base-independent
    # target would penalize modes for legitimately drifting along with base's own
    # training rather than for actually deviating from the FD init.
    init = EM.decode_epsf_modes(modes_init, decoded_epsf_base(params))
    return _mean_or_zero((modes - init) ** 2)


def chroma_image_penalty(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Pixel-Laplacian roughness of the GAUGED free colour image (0 when absent).

    Penalizes the raw leaf through its gauge, not around it, so the smoothness
    the optimizer sees is the smoothness of the image that is actually rendered.
    """
    if "chroma_image" not in params:
        return jnp.zeros((), dtype=jnp.float32)
    image = EM.decode_chroma_image(params["chroma_image"], decoded_epsf_base(params))
    return pixel_laplacian_penalty(image)


def mode_base_overlap_metric(modes: jnp.ndarray, base: jnp.ndarray) -> jnp.ndarray:
    """Health monitor: post-gauge mode/base overlap, should be ~0 by construction.

    Mirrors ``flux_neutral_penalty``'s pattern (raw, un-normalized overlap) --
    exactly like ``centroid`` double-checks the core-centroid=0 gauge, this double-checks
    ``decode_epsf_modes``'s base-projection gauge actually held numerically.
    """
    weight_grid = EM.canonical_mode_weight_grid(int(modes.shape[-1]))
    base_meansub = base - jnp.mean(base, axis=(-2, -1), keepdims=True)
    overlap = jnp.sum(modes * weight_grid[None, None, None] * base_meansub[None], axis=(-1, -2))
    return jnp.mean(overlap ** 2)


def node_smoothness_penalty(field: jnp.ndarray) -> jnp.ndarray:
    n_rows, n_cols = field.shape[-4], field.shape[-3]
    total = jnp.asarray(0.0, dtype=field.dtype)
    count = 0
    if n_rows > 1:
        d = field[..., 1:, :, :, :] - field[..., :-1, :, :, :]
        total = total + jnp.sum(d**2)
        count += d.size
    if n_cols > 1:
        d = field[..., :, 1:, :, :] - field[..., :, :-1, :, :]
        total = total + jnp.sum(d**2)
        count += d.size
    return total / max(count, 1)


# Scale split for ``fine_neighbour_penalty``: Gaussian sigma in node-grid samples
# (4 per native pixel), so "fine" = structure narrower than ~half a pixel.
FINE_NBR_SIGMA_SAMPLES = 2.0


def _gaussian_blur_matrix(g_size: int, sigma: float) -> np.ndarray:
    """``(G, G)`` 1D Gaussian blur (zero outside the grid; rows not renormalised)."""
    idx = np.arange(g_size, dtype=np.float64)
    k = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / float(sigma)) ** 2)
    k /= np.sqrt(2.0 * np.pi) * float(sigma)
    return k.astype(np.float32)


def fine_part(field: jnp.ndarray, sigma: float = FINE_NBR_SIGMA_SAMPLES) -> jnp.ndarray:
    """``field`` minus its Gaussian blur over the last two (grid) axes."""
    k = jnp.asarray(_gaussian_blur_matrix(int(field.shape[-1]), sigma), dtype=field.dtype)
    return field - jnp.einsum("ij,...jk,lk->...il", k, field, k)


def fine_neighbour_penalty(field: jnp.ndarray, sigma: float = FINE_NBR_SIGMA_SAMPLES) -> jnp.ndarray:
    """Adjacent-node difference of the FINE part only (``node_smoothness_penalty``
    applied to ``fine_part``). The smooth, field-dependent shape (coma, size) is
    left free; pixel-scale detail -- which a single FFI pins with only a handful of
    stars per sub-pixel phase at the edge/corner nodes -- is pooled across
    neighbours. Same (..., R, C, G, G) layout and normalisation as
    ``node_smoothness_penalty``.
    """
    return node_smoothness_penalty(fine_part(field, sigma))


def fine_neighbour_sum(field: jnp.ndarray, sigma: float = FINE_NBR_SIGMA_SAMPLES) -> jnp.ndarray:
    """UNnormalised sum of squared adjacent-node differences of the fine part
    (``fine_neighbour_penalty`` times its element count)."""
    fine = fine_part(field, sigma)
    total = jnp.zeros((), dtype=fine.dtype)
    if fine.shape[-4] > 1:
        total = total + jnp.sum((fine[..., 1:, :, :, :] - fine[..., :-1, :, :, :]) ** 2)
    if fine.shape[-3] > 1:
        total = total + jnp.sum((fine[..., :, 1:, :, :] - fine[..., :, :-1, :, :]) ** 2)
    return total


def fine_neighbour_count(field_shape) -> int:
    """Element count behind ``fine_neighbour_penalty``'s mean (both axes)."""
    r, c, g1, g2 = (int(v) for v in field_shape[-4:])
    return (max(r - 1, 0) * c + r * max(c - 1, 0)) * g1 * g2


def _low_order_generators(field: jnp.ndarray) -> jnp.ndarray:
    """(..., G, G) -> (..., 6, G, G): the field and its flux/shift/width generators
    (E, dE/dx, dE/dy, d2E/dx2, d2E/dy2, d2E/dxdy), central differences in grid samples."""
    gy, gx = jnp.gradient(field, axis=(-2, -1))
    gxx = jnp.gradient(gx, axis=-1)
    gyy = jnp.gradient(gy, axis=-2)
    gxy = jnp.gradient(gx, axis=-2)
    return jnp.stack([field, gx, gy, gxx, gyy, gxy], axis=-3)


def _blur(field: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Gaussian blur over the last two (grid) axes; identity for sigma <= 0."""
    if sigma <= 0:
        return field
    k = jnp.asarray(_gaussian_blur_matrix(int(field.shape[-1]), sigma), dtype=field.dtype)
    return jnp.einsum("ij,...jk,lk->...il", k, field, k)


def _project_out(d: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    """Remove from each d (..., G, G) its least-squares projection on basis (..., K, G, G)."""
    sh = d.shape
    dv = d.reshape(sh[:-2] + (-1,))
    bv = basis.reshape(basis.shape[:-2] + (-1,))
    gram = jnp.einsum("...kp,...lp->...kl", bv, bv)
    gram = gram + 1e-6 * jnp.trace(gram, axis1=-2, axis2=-1)[..., None, None] / gram.shape[-1] \
        * jnp.eye(gram.shape[-1], dtype=gram.dtype)
    coef = jnp.linalg.solve(gram, jnp.einsum("...kp,...p->...k", bv, dv)[..., None])[..., 0]
    return (dv - jnp.einsum("...k,...kp->...p", coef, bv)).reshape(sh)


def fine_neighbour_sum_moment_blind(field: jnp.ndarray,
                                    sigma: float = FINE_NBR_SIGMA_SAMPLES, *,
                                    basis: str = "own", basis_sigma: float = 0.0) -> jnp.ndarray:
    """UNnormalised ``fine_neighbour_sum`` that cannot see low-order core changes.

    The plain split passes ~57% of a core-width change (d2E/dx2 + d2E/dy2) at sigma = 2
    samples, so the plain penalty pulls poorly constrained nodes' core width toward their
    neighbours: +0.1..+4.4e-3 px^2 on a known-truth null, worst at the corners, and a
    2-5 permil optical-axis-vs-far flux bias on real fields (dev_runs/finenbr_axis_20260925,
    epsf_width_decision_20260928). Here each adjacent-node difference of the fine parts
    first has its projection removed on the fine parts of the pair-mean ePSF (flux) and of
    BOTH nodes' own shift and width generators (1 + 5 + 5 vectors), so a change of either
    node's centroid or second moments costs nothing. The basis is held fixed
    (stop_gradient) within a step. Each node's own E must NOT be in the basis: the
    difference itself would then lie in it.

    Known cost: the own-node basis is junk-shaped (second derivatives amplify pixel-scale
    structure), so a fit grows ~2.5x more node-pair fine difference and held-out core chi2
    is +9-12% vs plain -- all of it pixel-scale, none of it the width fix
    (dev_runs/finenbr_heldout_20260929).

    ``basis="pair"`` (pair-mean generators) and ``basis_sigma`` > 0 (smoothed generators)
    remove that junk but are NOT unbiased: on the known-truth null they broaden the core as
    much as plain (+2.3 / +1.6e-3 px^2 vs plain +2.2, own +0.2). Kept for reference only.
    """
    fine = fine_part(field, sigma)
    base = jax.lax.stop_gradient(field)
    total = jnp.zeros((), dtype=field.dtype)
    for ax in (-4, -3):
        if field.shape[ax] < 2:
            continue
        lo = [slice(None)] * field.ndim
        hi = [slice(None)] * field.ndim
        lo[ax] = slice(None, -1)
        hi[ax] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        d = fine[hi] - fine[lo]
        mean = 0.5 * (base[hi] + base[lo])
        if basis == "own":
            gen = jnp.concatenate([mean[..., None, :, :],
                                   _low_order_generators(_blur(base[lo], basis_sigma))[..., 1:, :, :],
                                   _low_order_generators(_blur(base[hi], basis_sigma))[..., 1:, :, :]], axis=-3)
        elif basis == "pair":
            gen = jnp.concatenate([mean[..., None, :, :],
                                   _low_order_generators(_blur(mean, basis_sigma))[..., 1:, :, :]], axis=-3)
        else:
            raise ValueError(f"unknown moment-blind basis {basis!r}")
        total = total + jnp.sum(_project_out(d, fine_part(gen, sigma)) ** 2)
    return total


def fine_neighbour_penalty_moment_blind(field: jnp.ndarray,
                                        sigma: float = FINE_NBR_SIGMA_SAMPLES, **basis_kw) -> jnp.ndarray:
    """``fine_neighbour_penalty`` blind to low-order (flux/shift/width) node differences;
    same (..., R, C, G, G) layout and normalisation. See ``fine_neighbour_sum_moment_blind``."""
    return fine_neighbour_sum_moment_blind(field, sigma, **basis_kw) / max(
        int(np.prod(field.shape[:-4])) * fine_neighbour_count(field.shape), 1)


# ``moment_blind`` is the default everywhere: the plain penalty's width drag is a
# position-dependent flux bias, which a calibration product cannot carry.
# moment_blind = own-node generators, moment_blind_pair = pair-mean generators.
FINE_NBR_MODES = ("moment_blind", "moment_blind_pair", "plain", "highpass")
FINE_NBR_MODE_DEFAULT = "moment_blind"
_MB_BASIS = {"moment_blind": "own", "moment_blind_pair": "pair"}


def fine_nbr_sum(field: jnp.ndarray, mode: str = FINE_NBR_MODE_DEFAULT,
                 sigma: float = FINE_NBR_SIGMA_SAMPLES, basis_sigma: float = 0.0) -> jnp.ndarray:
    """UNnormalised fine-neighbour sum for ``mode`` (see ``FINE_NBR_MODES``)."""
    if mode in _MB_BASIS:
        return fine_neighbour_sum_moment_blind(field, sigma, basis=_MB_BASIS[mode], basis_sigma=basis_sigma)
    if mode == "plain":
        return fine_neighbour_sum(field, sigma)
    if mode == "highpass":
        return _pair_diff_sum(fine_part_sharp(field))
    raise ValueError(f"unknown fine-neighbour mode {mode!r}; expected one of {FINE_NBR_MODES}")


# ``highpass`` mode: the plain neighbour coupling, but "fine" = spatial frequencies above
# FINE_NBR_HP_CUT cycles per native pixel (sharp Fourier cut) instead of E minus a sigma=0.5 px
# Gaussian blur. A pixel-integrated ePSF has ~1e-4 of its power above 1 c/px, so core width and
# shift barely pass the cut (finenbr_heldout_20260929 null/fig_spectrum.png). EXPERIMENTAL.
FINE_NBR_HP_CUT = 1.0


def fine_part_sharp(field: jnp.ndarray, fcut: float = FINE_NBR_HP_CUT) -> jnp.ndarray:
    """Part of ``field`` above ``fcut`` cycles/native px (radial, over the last two axes)."""
    g = int(field.shape[-1])
    f = np.fft.fftfreq(g, d=1.0 / EM.OVERSAMPLE)
    mask = jnp.asarray((np.hypot(*np.meshgrid(f, f)) > fcut).astype(np.float32))
    return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(field, axes=(-2, -1)) * mask, axes=(-2, -1))).astype(field.dtype)


def _pair_diff_sum(fine: jnp.ndarray) -> jnp.ndarray:
    total = jnp.zeros((), dtype=fine.dtype)
    if fine.shape[-4] > 1:
        total = total + jnp.sum((fine[..., 1:, :, :, :] - fine[..., :-1, :, :, :]) ** 2)
    if fine.shape[-3] > 1:
        total = total + jnp.sum((fine[..., :, 1:, :, :] - fine[..., :, :-1, :, :]) ** 2)
    return total


# ---------------------------------------------------------------------------
# Per-node local-polynomial smoothness (Anderson & King 2000 eq. 8; ACS ISR 2006-01 §3.1.2;
# WFC3 ISR 2016-12 p.18), 2026-09-30. Each grid point is compared with the value of a least-squares
# polynomial fitted to the surrounding window of grid points (a 2-D Savitzky-Golay filter); the
# penalty is the mean squared residual E - Q(E) over every node and grid point. No coupling between
# nodes. A local quartic reproduces a core's flux, centroid, width and elongation almost exactly
# (a 7x7 quartic sees 2-4% of a real width change; the plain fine split sees 57%), so the penalty
# cannot drag widths, and it removes grid-scale junk. Window 7 not AK's 5: a 5-point quartic
# interpolates any 5 values along a line, so a sample-scale stripe passes a 5x5 quartic untouched
# (dev_runs/localpoly_20260930/kernel_study.py). Lower order further out, as in AK (smoother wings).
LOCAL_POLY_WINDOW = 7
LOCAL_POLY_ORDERS = (4, 2, 1)      # inside radii[0], between, beyond radii[1]
LOCAL_POLY_RADII_PX = (3.0, 5.0)


def savgol2d_kernel(order: int, size: int) -> np.ndarray:
    """(size, size) weights giving the centre value of the least-squares 2-D polynomial of total
    degree ``order`` fitted to a size x size window (5x5 quartic = Anderson & King 2000 eq. 8)."""
    h = size // 2
    y, x = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float64)
    A = np.stack([(x ** i * y ** (d - i)).ravel() for d in range(order + 1) for i in range(d + 1)], 1)
    return np.linalg.pinv(A)[0].reshape(size, size)


def local_poly_smooth(field: jnp.ndarray, window: int = LOCAL_POLY_WINDOW,
                      orders=LOCAL_POLY_ORDERS, radii_px=LOCAL_POLY_RADII_PX) -> jnp.ndarray:
    """Radius-dependent local-polynomial smoothing Q(field) over the last two (grid) axes; the grid
    edge is extended by repetition (scipy 'nearest'). Radius is from the grid centre in native px."""
    g = int(field.shape[-1]); h = window // 2
    pad = [(0, 0)] * (field.ndim - 2) + [(h, h), (h, h)]
    fp = jnp.pad(field, pad, mode="edge")
    c = (g - 1) / 2.0
    yy, xx = np.mgrid[:g, :g]
    r = np.hypot(xx - c, yy - c) / EM.OVERSAMPLE
    zone = np.where(r < radii_px[0], 0, np.where(r < radii_px[1], 1, 2))
    out = jnp.zeros_like(field)
    for z, order in enumerate(orders):
        k = savgol2d_kernel(int(order), window)
        sm = sum(float(k[dy, dx]) * fp[..., dy:dy + g, dx:dx + g]
                 for dy in range(window) for dx in range(window) if k[dy, dx] != 0.0)
        out = out + jnp.asarray(zone == z, field.dtype) * sm
    return out


def local_poly_penalty(field: jnp.ndarray, window: int = LOCAL_POLY_WINDOW,
                       orders=LOCAL_POLY_ORDERS, radii_px=LOCAL_POLY_RADII_PX) -> jnp.ndarray:
    """Mean squared residual field - local_poly_smooth(field) over every node and grid point."""
    return jnp.mean((field - local_poly_smooth(field, window, orders, radii_px)) ** 2)


def fine_nbr_penalty(field: jnp.ndarray, mode: str = FINE_NBR_MODE_DEFAULT,
                     sigma: float = FINE_NBR_SIGMA_SAMPLES, basis_sigma: float = 0.0) -> jnp.ndarray:
    """Normalised fine-neighbour penalty for ``mode`` (see ``FINE_NBR_MODES``)."""
    if mode == "plain":
        return fine_neighbour_penalty(field, sigma)
    return fine_nbr_sum(field, mode, sigma, basis_sigma) / max(
        int(np.prod(field.shape[:-4])) * fine_neighbour_count(field.shape), 1)


def fine_nbr_prior_sum(params: dict[str, jnp.ndarray], sigma_prior: float,
                      mode: str = FINE_NBR_MODE_DEFAULT) -> jnp.ndarray:
    """Gaussian prior on neighbouring nodes' fine parts, as an UNnormalised
    negative log-prior: sum (delta fine)^2 / (2 sigma^2). ``sigma_prior`` is in
    the stored-grid units (flux fraction per node-grid sample). Callers divide
    it by the SAME pooled denominator as the data term, so that
    loss * denominator = -log likelihood - log prior (see LossWeights.fine_nbr_sigma).
    """
    base = decoded_epsf_base(params)
    return fine_nbr_sum(base[None], mode) / (2.0 * float(sigma_prior) ** 2)


def spline_smoothness_penalty(coeff: jnp.ndarray, second_diff: jnp.ndarray) -> jnp.ndarray:
    d = coeff @ second_diff.T
    return _mean_or_zero(d**2)


@dataclass(frozen=True)
class LossWeights:
    lambda_smooth_wcs: float = 1e-4
    lambda_smooth_w: float = 1e-3
    lambda_centroid: float = 1.0
    lambda_flux: float = 1.0
    lambda_lap: float = 1e-3
    # Adjacent-node smoothness of the FINE (sub-half-pixel) part only; see
    # fine_neighbour_penalty. 0 = off (pre-existing behaviour).
    lambda_fine_nbr: float = 0.0
    # Scene-independent alternative to lambda_fine_nbr: the same fine-part neighbour
    # coupling written as a Gaussian prior of width fine_nbr_sigma (flux fraction per
    # node-grid sample), divided by the pooled data-term denominator so that
    # loss * denominator = -log L - log prior. Only a true likelihood with
    # support_size_weight_power=1 (and uniform stamp weights) makes sigma physical.
    # 0 = off; mutually exclusive with lambda_fine_nbr.
    fine_nbr_sigma: float = 0.0
    # Which fine-neighbour coupling lambda_fine_nbr / fine_nbr_sigma apply:
    # "moment_blind" (default; blind to node-to-node flux/shift/width differences) or
    # "plain" (pre-2026-09-29; biases core width, see fine_neighbour_sum_moment_blind).
    fine_nbr_mode: str = FINE_NBR_MODE_DEFAULT
    lambda_pixel: float = 1e-2
    lambda_mode_prior: float = 1e-2
    # Pixel-Laplacian smoothness on the free colour IMAGE only (never on the
    # parametric chroma leaves -- twelve-to-two-hundred numbers need no prior).
    # 3364 free parameters driven by a ~0.3-mag colour lever on ONE frame do:
    # without this the image happily fits per-pixel noise. Default 0 so the leaf
    # is inert unless a run asks for it.
    lambda_chroma_lap: float = 0.0
    ridge: float = 1e-6
    huber_delta: float = HUBER_DELTA_DEFAULT
    # Static profile-flux objective.  ``huber_irls`` uses the same delta as
    # the outer pixel loss so the nuisance-flux solve and optimized objective
    # are aligned; ``l2`` preserves historical behavior.
    flux_objective: str = "l2"
    huber_irls_iters: int = 2
    centroid_in_grad: bool = False
    # Optional per-stamp support-size weighting: multiplies the stamp weight
    # by (effective pixel count)**power before the weighted mean over stamps.
    # Default 0.0 = off (exactly the prior behavior, bit-identical -- the
    # multiply is skipped in Python, not just a no-op factor of 1). On the
    # square path every stamp had the same S*S pixel count, so this was
    # never needed; on the irregular/packed path P ranges ~19x (49..945),
    # so without this a 945-pixel bright blend and a 49-pixel isolated star
    # contribute equally to the data term despite the former carrying ~19x
    # the information. 0.5 = sqrt (favor larger supports mildly), 1.0 =
    # proportional to pixel count.
    support_size_weight_power: float = 0.0
    # Task M4: solve one additive pedestal (background level) per group per
    # frame jointly with flux, shared by all K members of the group (see
    # ``flux_solve.solve_group_fluxes(..., pedestal=True)``). Default False
    # is exactly the prior behavior (the K+1-unknown augmented solve never
    # runs; bit-identical) -- CLI: ``--stamp-pedestal``.
    stamp_pedestal: bool = False
    # Task PW: profile the temporal ePSF mode amplitudes out in closed form,
    # per frame, jointly with the per-stamp fluxes/pedestal (see
    # ``flux_solve.solve_group_fluxes_profile_w``) instead of reading them off
    # the trained ``w_coeff`` spline. When True the render happens at w = 0
    # plus per-mode derivative probes, ``w_coeff`` is inert (the loss never
    # calls ``w_field_from_coeff``), and the amplitude the model uses is the
    # solved one. Default False is exactly the prior behavior, bit-identical
    # (no probe render, no bordered system) -- CLI: ``--profile-w``.
    profile_w: bool = False
    # Gauss-Newton iterations of the bilinear (flux x amplitude) solve. 2 is
    # enough: the neglected term is second order and the fluxes are re-solved
    # exactly at the final amplitude either way.
    profile_w_iters: int = 2
    # Ridge on the (n_modes x n_modes) Schur complement; None = same as ridge.
    ridge_w: float | None = None


def pixel_variance(noise: jnp.ndarray) -> jnp.ndarray:
    return noise**2 + VARIANCE_FLOOR


def inverse_variance_weights(mask: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    return mask / var


def radius_pixel_mask(fit_radius: jnp.ndarray, stamp: int = EM.STAMP_PHYSICAL) -> jnp.ndarray:
    """(n_groups, S, S) hard cut r <= fit_radius[g]."""
    r = jnp.asarray(stamp_radius_grid(stamp), dtype=jnp.float32)
    return (r[None, :, :] <= fit_radius[:, None, None]).astype(jnp.float32)


def huber_rho(chi: jnp.ndarray, delta: float) -> jnp.ndarray:
    abs_chi = jnp.abs(chi)
    d = jnp.asarray(delta, dtype=chi.dtype)
    return jnp.where(abs_chi <= d, chi**2, 2.0 * d * abs_chi - d**2)


def effective_stamp_weight(
    stamp_snr_weight: jnp.ndarray,
    stamp_active: jnp.ndarray,
) -> jnp.ndarray:
    """Broadcast catalog SNR ``(G,)`` onto ``stamp_active`` ``(G, T)``."""
    snr = stamp_snr_weight
    if snr.ndim == 1:
        snr = snr[:, None]
    return snr * stamp_active


def per_stamp_snr_weighted_nll(
    data: jnp.ndarray,
    model: jnp.ndarray,
    var: jnp.ndarray,
    pixel_weight: jnp.ndarray,
    stamp_weight: jnp.ndarray,
    *,
    huber_delta: float = HUBER_DELTA_DEFAULT,
    support_size_weight_power: float = 0.0,
) -> jnp.ndarray:
    """Per-stamp mean Huber-NLL, then soft-SNR-weighted mean over stamps/frames.

    Spatial layout: square ``(G,T,S,S)`` or packed ``(G,T,P)``.

    ``support_size_weight_power``: optional extra factor ``pix_sum**power``
    folded into the per-stamp weight before the mean over stamps (see
    ``LossWeights.support_size_weight_power``). Default ``0.0`` skips the
    multiply in Python entirely -- bit-identical to the pre-existing behavior.
    """
    chi = (data - model) / jnp.sqrt(var)
    ell = 0.5 * huber_rho(chi, huber_delta) + 0.5 * jnp.log(var)
    spatial = (-1,) if data.ndim == 3 else (-1, -2)
    pix_sum = jnp.sum(pixel_weight, axis=spatial)
    nll_stamp = jnp.sum(pixel_weight * ell, axis=spatial) / jnp.clip(pix_sum, 1e-6, None)
    active = (pix_sum > 0).astype(data.dtype)
    sw = stamp_weight
    if sw.ndim == 1:
        sw = sw[:, None]
    sw = sw * active
    if support_size_weight_power:
        sw = sw * jnp.power(jnp.clip(pix_sum, 1e-6, None), support_size_weight_power)
    return jnp.sum(sw * nll_stamp) / jnp.clip(jnp.sum(sw), 1e-6, None)


def _pad_frame_axis(arr: jnp.ndarray, axis: int, pad_len: int) -> jnp.ndarray:
    """Zero-pad ``arr`` along ``axis`` by ``pad_len`` (no-op if ``pad_len == 0``)."""
    if pad_len == 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(arr, pad_width)


def _to_frame_blocks(arr: jnp.ndarray, axis: int, n_blocks: int, block: int) -> jnp.ndarray:
    """Reshape ``arr`` (frame axis length ``n_blocks*block``) into
    ``(n_blocks, ...)`` with the block axis leading, ready for ``lax.scan``."""
    shape = arr.shape
    new_shape = shape[:axis] + (n_blocks, block) + shape[axis + 1:]
    arr = arr.reshape(new_shape)
    return jnp.moveaxis(arr, axis, 0)


def _chunked_data_term(
    params: dict[str, jnp.ndarray],
    ctx: StaticContext,
    data: jnp.ndarray,
    noise: jnp.ndarray,
    weight: jnp.ndarray,
    stamp_active: jnp.ndarray,
    local_cache: jnp.ndarray | None,
    n_pix: int,
    do_recenter: bool,
    recenter_n_iter: int,
    frame_idx: jnp.ndarray | None,
    group_idx: jnp.ndarray | None,
    weights: LossWeights,
    block: int,
    w_of_t_override: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``lax.scan`` + ``jax.checkpoint`` frame-block chunked equivalent of the
    data-term half of ``total_loss`` (everything downstream of ``forward_model``
    through ``per_stamp_snr_weighted_nll``). Returns ``(data_term,
    stamp_weight_sum, w_of_t)`` -- the same three quantities the unchunked path
    in ``total_loss`` produces, to fp32-roundoff precision (summation order
    differs: per-block partial sums vs. one flat sum -- see module-level notes).

    Frame-global, frame-independent quantities (the ``w_of_t`` zero-mean gauge,
    which must see the *full* frame set; the decoded ePSF ``base``/``modes``,
    which are tiny and param-only) are computed once, outside the scan, and
    closed over by the scanned body. Everything that scales with the number of
    occupied slots -- template rendering, the analytic flux solve, and the
    per-stamp Huber NLL -- runs inside the ``jax.checkpoint``-wrapped scan body,
    one frame-block at a time, so peak memory no longer scales with the total
    frame count T (only with ``block``). ``jax.checkpoint`` alone or
    ``lax.scan`` alone do not help here (both were measured *worse* than no
    chunking at all on this graph); the memory win only appears when both are
    used together, because remat needs the scan's small per-iteration slice to
    have anything small left to recompute.

    The frame axis is zero-padded up to a multiple of ``block`` and the padding
    frames' ``stamp_active`` is forced to 0, so they contribute exactly zero to
    both the accumulated numerator and denominator (see ``per_stamp_snr_weighted_nll``
    -- a stamp with ``stamp_weight == 0`` drops out of the weighted mean
    entirely); the result is therefore independent of ``block`` and identical
    (to roundoff) to the unchunked, unpadded computation.
    """
    if frame_idx is not None or group_idx is not None:
        raise NotImplementedError(
            "stamp_chunk cannot be combined with frame_idx/group_idx minibatch "
            "selection. Neither is currently wired from fit.py's minibatch step "
            "(see fit.make_step_fn: the single-bucket minibatch loss_fn drops "
            "frame_idx/group_idx before calling total_loss) -- run with "
            "group_frac=frame_frac=1.0 when using --stamp-chunk."
        )

    packed = bool(ctx.is_packed)
    if packed and local_cache is not None:
        raise NotImplementedError("dx-only local_cache is not supported on the packed path")
    epsf_params = _epsf_params_for_context(params, ctx)

    n_groups, K = ctx.members.shape
    n_frames_full = int(ctx.wcs_frame_basis.shape[0])
    block = max(1, min(int(block), n_frames_full))
    n_blocks = -(-n_frames_full // block)  # ceil division
    pad_len = n_blocks * block - n_frames_full

    # ---- frame-global quantities (cheap; computed once over the full T) ----
    x_t, y_t = CW.eval_all_positions(
        ctx.x_lin, ctx.y_lin, ctx.cheb_basis, params["wcs_coeff"], ctx.wcs_frame_basis, ctx.n_terms,
    )
    if w_of_t_override is not None:
        # Caller already applied the whole-orbit zero-mean gauge (frame-block
        # gradient accumulation); ctx here holds only this block's frames.
        w_of_t = w_of_t_override
    else:
        # Zero-time-mean gauge: MUST be taken over the full frame set (see
        # forward_model's identical comment) -- computing it per block would
        # change the model, not just how it's chunked.
        w_of_t = w_field_from_coeff(epsf_params["w_coeff"], ctx.w_frame_basis)
    if weights.profile_w:
        # Task PW: the amplitude is solved per frame inside the block body, so
        # the spline is inert. Zeroing here (rather than merely ignoring it)
        # makes that explicit and cuts ``w_coeff`` out of the gradient graph
        # entirely -- it must also be labelled frozen by the optimizer.
        w_of_t = jnp.zeros_like(w_of_t)

    flat_idx = ctx.members.reshape(-1)
    occ = ctx.occ_flat_idx
    n_occ = int(occ.shape[0])
    star_occ = flat_idx[occ]
    x_occ = jnp.reshape(x_t[star_occ], (n_occ, n_frames_full))
    y_occ = jnp.reshape(y_t[star_occ], (n_occ, n_frames_full))

    base = decoded_epsf_base(epsf_params)
    g_size = int(base.shape[-1])
    if local_cache is not None:
        local_cache_arr = jnp.reshape(local_cache, (n_occ, g_size, g_size))
        modes = None
    else:
        local_cache_arr = None
        modes = decoded_epsf_modes(epsf_params)

    # Chromatic terms are constant in time, so they are computed once here and the
    # shift is folded into the position arrays before the frame axis is blocked.
    chroma = chroma_slot_terms(params, ctx, star_occ)
    if chroma is None:
        chroma_delta_occ = chroma_fields = None
    else:
        chroma_delta_occ, chroma_shift_x, chroma_shift_y, chroma_fields = chroma

    huber_delta = weights.huber_delta
    ridge = weights.ridge
    support_size_weight_power = weights.support_size_weight_power
    stamp_snr_weight = ctx.stamp_snr_weight
    occ_group = ctx.occ_group
    occ_slot = ctx.occ_slot
    stamp_active = jnp.asarray(stamp_active, dtype=jnp.float32)

    # ---- pad the frame axis, then reshape to (n_blocks, ..., block, ...) ----
    x_occ_p = _pad_frame_axis(x_occ, 1, pad_len)
    y_occ_p = _pad_frame_axis(y_occ, 1, pad_len)
    w_of_t_p = _pad_frame_axis(w_of_t, 0, pad_len)
    data_p = _pad_frame_axis(data, 1, pad_len)
    noise_p = _pad_frame_axis(noise, 1, pad_len)
    weight_p = _pad_frame_axis(weight, 1, pad_len)
    stamp_active_p = _pad_frame_axis(stamp_active, 1, pad_len)
    if pad_len > 0:
        # Belt-and-suspenders alongside the zero-weight/zero-data padding
        # above: force the padding frames' own stamp_active to 0 so they are
        # excluded from the weighted mean regardless of what the (finite, but
        # otherwise meaningless) padded render evaluates to.
        real_mask = jnp.concatenate(
            [jnp.ones((n_frames_full,), dtype=jnp.float32), jnp.zeros((pad_len,), dtype=jnp.float32)],
        )
        stamp_active_p = stamp_active_p * real_mask[None, :]

    x_occ_b = _to_frame_blocks(x_occ_p, 1, n_blocks, block)
    y_occ_b = _to_frame_blocks(y_occ_p, 1, n_blocks, block)
    w_of_t_b = _to_frame_blocks(w_of_t_p, 0, n_blocks, block)
    data_b = _to_frame_blocks(data_p, 1, n_blocks, block)
    noise_b = _to_frame_blocks(noise_p, 1, n_blocks, block)
    weight_b = _to_frame_blocks(weight_p, 1, n_blocks, block)
    stamp_active_b = _to_frame_blocks(stamp_active_p, 1, n_blocks, block)

    if packed:
        pix_valid = ctx.pix_valid  # (n_groups, P)
        if chroma is None:
            xr_b, yr_b = x_occ_b, y_occ_b
        else:
            xr_b = _to_frame_blocks(
                _pad_frame_axis(x_occ + chroma_shift_x[:, None], 1, pad_len), 1, n_blocks, block
            )
            yr_b = _to_frame_blocks(
                _pad_frame_axis(y_occ + chroma_shift_y[:, None], 1, pad_len), 1, n_blocks, block
            )

        def block_fn(carry, xs_blk):
            num, den = carry
            x_occ_i, y_occ_i, xr_i, yr_i, w_i, data_i, noise_i, weight_i, sa_i = xs_blk

            if weights.profile_w:
                n_modes = int(modes.shape[0])
                rep = n_modes + 1
                aug = _render_occ_templates_packed(
                    ctx,
                    jnp.repeat(x_occ_i, rep, axis=1), jnp.repeat(y_occ_i, rep, axis=1),
                    _mode_probe_w(block, n_modes, x_occ_i.dtype),
                    base=base,
                    modes=modes,
                    do_recenter=do_recenter,
                    recenter_n_iter=recenter_n_iter,
                    x_render=jnp.repeat(xr_i, rep, axis=1),
                    y_render=jnp.repeat(yr_i, rep, axis=1),
                    chroma_delta_occ=chroma_delta_occ,
                    chroma_fields=chroma_fields,
                )
                templates_occ, mode_occ = _split_mode_probe(aug, block, n_modes)
            else:
                mode_occ = None
                templates_occ = _render_occ_templates_packed(
                    ctx, x_occ_i, y_occ_i, w_i,
                    base=base,
                    modes=modes,
                    do_recenter=do_recenter,
                    recenter_n_iter=recenter_n_iter,
                    x_render=xr_i,
                    y_render=yr_i,
                    chroma_delta_occ=chroma_delta_occ,
                    chroma_fields=chroma_fields,
                )
            blk = templates_occ.shape[1]
            P = templates_occ.shape[-1]
            templates = jnp.zeros((n_groups, K, blk, P), dtype=templates_occ.dtype)
            templates = templates.at[occ_group, occ_slot].set(templates_occ)
            if mode_occ is None:
                mode_templates = None
            else:
                mode_templates = jnp.zeros(
                    (mode_occ.shape[0], n_groups, K, blk, P), dtype=mode_occ.dtype
                ).at[:, occ_group, occ_slot].set(mode_occ)

            var_i = pixel_variance(noise_i)
            pix_w_i = weight_i * pix_valid[:, None, :]
            iv_i = inverse_variance_weights(pix_w_i, var_i)
            model_i, _ = _solve_block_model(templates, mode_templates, data_i, iv_i, weights)

            chi = (data_i - model_i) / jnp.sqrt(var_i)
            ell = 0.5 * huber_rho(chi, huber_delta) + 0.5 * jnp.log(var_i)
            pix_sum = jnp.sum(pix_w_i, axis=-1)
            nll_stamp = jnp.sum(pix_w_i * ell, axis=-1) / jnp.clip(pix_sum, 1e-6, None)
            active = (pix_sum > 0).astype(data_i.dtype)
            sw = stamp_snr_weight[:, None] * sa_i * active
            if support_size_weight_power:
                sw = sw * jnp.power(jnp.clip(pix_sum, 1e-6, None), support_size_weight_power)

            return (num + jnp.sum(sw * nll_stamp), den + jnp.sum(sw)), None

        xs = (x_occ_b, y_occ_b, xr_b, yr_b, w_of_t_b, data_b, noise_b, weight_b, stamp_active_b)
    else:
        cx_occ = ctx.stamp_center_x[ctx.occ_group]
        cy_occ = ctx.stamp_center_y[ctx.occ_group]
        dx = x_occ - cx_occ[:, None]
        dy = y_occ - cy_occ[:, None]
        if chroma is not None:
            # Frame-independent, so it goes in before the padding and blocking.
            dx = dx + chroma_shift_x[:, None]
            dy = dy + chroma_shift_y[:, None]
        dx_p = _pad_frame_axis(dx, 1, pad_len)
        dy_p = _pad_frame_axis(dy, 1, pad_len)
        dx_b = _to_frame_blocks(dx_p, 1, n_blocks, block)
        dy_b = _to_frame_blocks(dy_p, 1, n_blocks, block)
        rmask = radius_pixel_mask(ctx.fit_radius, stamp=data.shape[-1])

        def block_fn(carry, xs_blk):
            num, den = carry
            dx_i, dy_i, x_occ_i, y_occ_i, w_i, data_i, noise_i, weight_i, sa_i = xs_blk

            if weights.profile_w:
                if modes is None:
                    raise NotImplementedError(
                        "profile_w needs the ePSF modes; it is incompatible "
                        "with the dx-only local_cache fast path"
                    )
                n_modes = int(modes.shape[0])
                rep = n_modes + 1
                aug = _render_occ_templates(
                    ctx,
                    jnp.repeat(x_occ_i, rep, axis=1), jnp.repeat(y_occ_i, rep, axis=1),
                    jnp.repeat(dx_i, rep, axis=1), jnp.repeat(dy_i, rep, axis=1),
                    _mode_probe_w(block, n_modes, x_occ_i.dtype),
                    base=base,
                    modes=modes,
                    local_cache_arr=local_cache_arr,
                    n_pix=n_pix,
                    do_recenter=do_recenter,
                    recenter_n_iter=recenter_n_iter,
                    chroma_delta_occ=chroma_delta_occ,
                    chroma_fields=chroma_fields,
                )
                templates_occ, mode_occ = _split_mode_probe(aug, block, n_modes)
            else:
                mode_occ = None
                templates_occ = _render_occ_templates(
                    ctx, x_occ_i, y_occ_i, dx_i, dy_i, w_i,
                    base=base,
                    modes=modes,
                    local_cache_arr=local_cache_arr,
                    n_pix=n_pix,
                    do_recenter=do_recenter,
                    recenter_n_iter=recenter_n_iter,
                    chroma_delta_occ=chroma_delta_occ,
                    chroma_fields=chroma_fields,
                )
            blk = templates_occ.shape[1]
            S = templates_occ.shape[-1]
            templates = jnp.zeros((n_groups, K, blk, S, S), dtype=templates_occ.dtype)
            templates = templates.at[occ_group, occ_slot].set(templates_occ)
            if mode_occ is None:
                mode_templates = None
            else:
                mode_templates = jnp.zeros(
                    (mode_occ.shape[0], n_groups, K, blk, S, S), dtype=mode_occ.dtype
                ).at[:, occ_group, occ_slot].set(mode_occ)

            var_i = pixel_variance(noise_i)
            pix_w_i = weight_i * rmask[:, None, :, :]
            iv_i = inverse_variance_weights(pix_w_i, var_i)
            model_i, _ = _solve_block_model(templates, mode_templates, data_i, iv_i, weights)

            chi = (data_i - model_i) / jnp.sqrt(var_i)
            ell = 0.5 * huber_rho(chi, huber_delta) + 0.5 * jnp.log(var_i)
            pix_sum = jnp.sum(pix_w_i, axis=(-1, -2))
            nll_stamp = jnp.sum(pix_w_i * ell, axis=(-1, -2)) / jnp.clip(pix_sum, 1e-6, None)
            active = (pix_sum > 0).astype(data_i.dtype)
            sw = stamp_snr_weight[:, None] * sa_i * active
            if support_size_weight_power:
                sw = sw * jnp.power(jnp.clip(pix_sum, 1e-6, None), support_size_weight_power)

            return (num + jnp.sum(sw * nll_stamp), den + jnp.sum(sw)), None

        xs = (dx_b, dy_b, x_occ_b, y_occ_b, w_of_t_b, data_b, noise_b, weight_b, stamp_active_b)

    block_fn = jax.checkpoint(block_fn)
    init = (jnp.zeros((), dtype=data.dtype), jnp.zeros((), dtype=data.dtype))
    (num_total, den_total), _ = jax.lax.scan(block_fn, init, xs)

    data_term = num_total / jnp.clip(den_total, 1e-6, None)
    stamp_weight_sum = den_total
    return data_term, stamp_weight_sum, w_of_t


def total_loss(
    params: dict[str, jnp.ndarray],
    ctx: StaticContext,
    data: jnp.ndarray,
    noise: jnp.ndarray,
    weight: jnp.ndarray,
    wcs_second_diff: jnp.ndarray,
    w_second_diff: jnp.ndarray,
    *,
    epsf_modes_init: jnp.ndarray,
    weights: LossWeights = LossWeights(),
    stamp_active: jnp.ndarray | None = None,
    local_cache: jnp.ndarray | None = None,
    n_pix: int | None = None,
    do_recenter: bool = True,
    recenter_n_iter: int = EM.HOTPATH_RECENTER_N_ITER,
    frame_idx: jnp.ndarray | None = None,
    group_idx: jnp.ndarray | None = None,
    stamp_chunk: int | None = None,
    w_of_t_override: jnp.ndarray | None = None,
):
    """``stamp_chunk``: frames per ``lax.scan`` block for the data-term (see
    ``_chunked_data_term``). ``None``/``0`` (default) runs the original,
    unchunked forward pass unchanged -- nothing about existing callers'
    behavior changes. A positive value trades ~1.3x more FLOPs (remat
    recomputes the block's forward pass on the backward pass) for peak memory
    that scales with ``stamp_chunk`` instead of the total frame count, which is
    what makes a full-CCD (2048x2048), all-frame fit tractable. Works for both
    square ``(G,T,S,S)`` and packed ``(G,T,P)`` layouts.
    """
    if stamp_active is None:
        stamp_active = ctx.stamp_active
    packed = bool(ctx.is_packed)
    n_pix_eff = n_pix if n_pix is not None else (1 if packed else data.shape[-1])
    if not stamp_chunk:
        fwd = forward_model(
            params, ctx,
            stamp_active=stamp_active,
            local_cache=local_cache,
            n_pix=n_pix_eff,
            recenter_n_iter=recenter_n_iter,
            do_recenter=do_recenter,
            frame_idx=frame_idx,
            group_idx=group_idx,
            w_of_t_override=w_of_t_override,
            return_mode_templates=weights.profile_w,
        )
        if weights.profile_w:
            templates, x_t, y_t, w_of_t, mode_templates = fwd
            # See _chunked_data_term: the spline amplitude is inert under PW.
            w_of_t = jnp.zeros_like(w_of_t)
        else:
            templates, x_t, y_t, w_of_t = fwd
            mode_templates = None
        var = pixel_variance(noise)
        if packed:
            pix_w = weight * ctx.pix_valid[:, None, :]
            stamp_active_pix = (jnp.sum(pix_w, axis=-1) > 0).astype(data.dtype)
        else:
            rmask = radius_pixel_mask(ctx.fit_radius, stamp=data.shape[-1])
            pix_w = weight * rmask[:, None, :, :]
            stamp_active_pix = (jnp.sum(pix_w, axis=(-1, -2)) > 0).astype(data.dtype)
        iv = inverse_variance_weights(pix_w, var)
        model, _ = _solve_block_model(templates, mode_templates, data, iv, weights)

        stamp_w = effective_stamp_weight(ctx.stamp_snr_weight, stamp_active)
        data_term = per_stamp_snr_weighted_nll(
            data, model, var, pix_w, stamp_w, huber_delta=weights.huber_delta,
            support_size_weight_power=weights.support_size_weight_power,
        )
        # Must be the exact denominator per_stamp_snr_weighted_nll divides by (the
        # bucket/frame-block pooling in fit.py weights each bucket by it). With
        # support_size_weight_power > 0 that includes pix_sum**power -- dropping it
        # pooled buckets by stamp count instead of pixel count (the chunked path
        # already had it).
        swsum = stamp_w * stamp_active_pix
        if weights.support_size_weight_power:
            pix_sum = jnp.sum(pix_w, axis=(-1,) if packed else (-1, -2))
            swsum = swsum * jnp.power(
                jnp.clip(pix_sum, 1e-6, None), weights.support_size_weight_power
            )
        stamp_weight_sum = jnp.sum(swsum)
    else:
        data_term, stamp_weight_sum, w_of_t = _chunked_data_term(
            params, ctx, data, noise, weight, stamp_active, local_cache,
            n_pix_eff, do_recenter, recenter_n_iter, frame_idx, group_idx,
            weights, int(stamp_chunk), w_of_t_override,
        )
    smooth_wcs = spline_smoothness_penalty(params["wcs_coeff"], wcs_second_diff)
    if weights.profile_w:
        # ``w_coeff`` is not part of the model at all under PW; penalising it
        # would only add a constant (it is frozen) and would misreport the
        # loss, so the smoothness term is dropped rather than smuggled in.
        smooth_w = jnp.zeros((), dtype=jnp.float32)
    else:
        smooth_w = spline_smoothness_penalty(params["w_coeff"], w_second_diff)
    if weights.centroid_in_grad:
        centroid = centroid_penalty(params, w_of_t)
        centroid_term = weights.lambda_centroid * centroid
    else:
        # Hard core-centroid already enforces the gauge; skip the node×frame rebuild.
        centroid = jnp.asarray(0.0, dtype=data_term.dtype)
        centroid_term = 0.0
    flux_neutral = flux_neutral_penalty(params)
    base = decoded_epsf_base(params)
    lap = node_smoothness_penalty(base[None])
    fine_nbr = fine_nbr_penalty(base[None], weights.fine_nbr_mode)
    pixel_lap = pixel_laplacian_penalty(base)
    mode_prior = mode_prior_penalty(params, epsf_modes_init)
    chroma_lap = chroma_image_penalty(params)
    mode_base_overlap = mode_base_overlap_metric(decoded_epsf_modes(params), base)
    # Metric only (not a loss term): decode_epsf_base already enforces the
    # flux rule via EM.enforce_phase_flux_rule, so this should sit at ~0;
    # logged as a sanity check on the hard gauge (CONTRACT_pixel_integrated_epsf.md).
    phase_flux_rms = EM.phase_flux_rms(base)

    if weights.fine_nbr_sigma > 0.0:
        fine_nbr_prior_sum_v = fine_nbr_prior_sum(params, weights.fine_nbr_sigma,
                                                  weights.fine_nbr_mode)
        inv_den = 1.0 / jnp.clip(stamp_weight_sum, 1e-12, None)
        fine_nbr_prior_term = fine_nbr_prior_sum_v * inv_den
        lambda_fine_nbr_eff = (
            fine_neighbour_count(base[None].shape) / (2.0 * weights.fine_nbr_sigma ** 2)
        ) * inv_den
    else:
        fine_nbr_prior_sum_v = jnp.zeros((), dtype=data_term.dtype)
        fine_nbr_prior_term = jnp.zeros((), dtype=data_term.dtype)
        lambda_fine_nbr_eff = jnp.asarray(weights.lambda_fine_nbr, dtype=data_term.dtype)

    loss = (
        data_term
        + fine_nbr_prior_term
        + weights.lambda_smooth_wcs * smooth_wcs
        + weights.lambda_smooth_w * smooth_w
        + centroid_term
        + weights.lambda_flux * flux_neutral
        + weights.lambda_lap * lap
        + weights.lambda_fine_nbr * fine_nbr
        + weights.lambda_pixel * pixel_lap
        + weights.lambda_mode_prior * mode_prior
        + weights.lambda_chroma_lap * chroma_lap
    )
    ratio_centroid = (weights.lambda_centroid * centroid) / jnp.clip(data_term, 1e-12, None)
    metrics = {
        "data_term": data_term,
        "stamp_weight_sum": stamp_weight_sum,
        "smooth_wcs": smooth_wcs,
        "smooth_w": smooth_w,
        "centroid": centroid,
        "flux_neutral": flux_neutral,
        "lap": lap,
        "fine_nbr": fine_nbr,
        "fine_nbr_prior_sum": fine_nbr_prior_sum_v,
        "fine_nbr_prior_term": fine_nbr_prior_term,
        "lambda_fine_nbr_eff": lambda_fine_nbr_eff,
        "pixel_lap": pixel_lap,
        "chroma_lap": chroma_lap,
        "mode_prior": mode_prior,
        "mode_base_overlap": mode_base_overlap,
        "ratio_centroid": ratio_centroid,
        "phase_flux_rms": phase_flux_rms,
        "loss": loss,
    }
    return loss, metrics
