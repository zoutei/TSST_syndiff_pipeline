# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Post-hoc aperture correction from forced-PSF fluxes.

After a converged ePSF+WCS fit, estimate one multiplicative factor per ePSF
node, varying on the same B-spline basis as ePSF ``w_k(t)``:

    f_ls(s, t) ≈ F_s * A(x_s(t), y_s(t), t)

with ``A_node = 1 + α_node``, ``α_node(t) = A_coeff @ Φ(t)``, bilinear blend
at the star (same as ePSF). Scale gauge only:
``mean_{nodes,t} A_node = 1`` (fixes ``F ↔ A`` degeneracy). Do **not** zero
the per-frame spatial mean — that would kill the common-mode aperture term
that dominates the focus-breathing flux bias.

Fit is IV-weighted alternating LS using ``σ_f`` from
``solve_group_fluxes_with_err`` (photutils-equivalent forced-photometry errors).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import epsf_model as EM
from . import fit as FIT
from . import flux_solve as FS
from . import loss as L


@dataclass
class PrimaryFluxTable:
    """Primary-star forced photometry + positions."""

    flux: np.ndarray  # (n_stars, T)
    sigma_f: np.ndarray  # (n_stars, T)
    x: np.ndarray  # (n_stars, T)
    y: np.ndarray  # (n_stars, T)
    active: np.ndarray  # (n_stars, T) bool
    group_index: np.ndarray  # (n_stars,)
    slot_index: np.ndarray  # (n_stars,)
    star_index: np.ndarray  # (n_stars,) expanded catalog index
    tess_mag: np.ndarray  # (n_stars,)
    source_id: np.ndarray | None = None  # (n_stars,) if available
    btjd: np.ndarray | None = None  # (T,)


@dataclass
class ApertureResult:
    A_coeff: np.ndarray  # (n_rows, n_cols, n_basis)
    A_node: np.ndarray  # (n_rows, n_cols, T)
    A_star: np.ndarray  # (n_stars, T)
    F: np.ndarray  # (n_stars,)
    flux_corr: np.ndarray  # (n_stars, T)
    frame_basis: np.ndarray  # (T, n_basis)
    node_x: np.ndarray
    node_y: np.ndarray
    n_iter: int
    rms_resid: float  # RMS of (f - F*A)/f on kept samples (fractional)
    # Which stars actually constrained the fit. Stars below the coverage gate
    # (``n_ok >= max(5, n_frames // 2)``) are NOT corrected: their ``F`` stays at
    # its initial 1.0 and their ``flux_corr`` is therefore the raw flux divided by
    # a field they did not constrain. Emitting that as a "corrected" light curve
    # drew flat panels in the galleries that read as *good* stars (576 of them in
    # the chroma-v1 run). Both ``F`` and ``flux_corr`` are NaN for these now, and
    # this mask says which they are.
    star_kept: np.ndarray | None = None


def extract_primary_fluxes(
    params: dict,
    ctx: L.StaticContext,
    data,
    noise,
    weight,
    *,
    primary_index_set: set[int],
    mags: np.ndarray | None = None,
    source_ids: np.ndarray | None = None,
    btjd: np.ndarray | None = None,
    stamp_active=None,
    ridge: float = 1e-6,
) -> PrimaryFluxTable:
    """Forced photometry for every primary slot (joint group solve)."""
    data_arr = np.asarray(data)
    fwd_kw: dict = {"params": params, "ctx": ctx, "stamp_active": stamp_active}
    if not bool(ctx.is_packed):
        fwd_kw["n_pix"] = int(data_arr.shape[-1])
    templates, x_slot, y_slot, _ = L.forward_model(**fwd_kw)
    var = L.pixel_variance(noise)
    if bool(ctx.is_packed):
        pix_w = weight * ctx.pix_valid[:, None, :]
    else:
        rmask = L.radius_pixel_mask(ctx.fit_radius, stamp=fwd_kw.get("n_pix"))
        pix_w = weight * rmask[:, None, :, :]
    iv = L.inverse_variance_weights(pix_w, var)
    flux, sigma_f = FS.solve_group_fluxes_with_err(templates, data, iv, ridge=ridge)

    flux_np = np.asarray(flux)
    sig_np = np.asarray(sigma_f)
    x_np = np.asarray(x_slot)
    y_np = np.asarray(y_slot)
    members = np.asarray(ctx.members)
    valid = np.asarray(ctx.valid) > 0.5
    if stamp_active is None:
        stamp_active = np.asarray(ctx.stamp_active)
    else:
        stamp_active = np.asarray(stamp_active)

    # flux / sigma_f: (n_groups, n_frames, K); x/y_slot: (n_groups, K, n_frames)
    n_groups, n_frames, K = flux_np.shape
    rows = []
    for gi in range(n_groups):
        for k in range(K):
            if not valid[gi, k]:
                continue
            si = int(members[gi, k])
            if si not in primary_index_set:
                continue
            rows.append((gi, k, si))

    if not rows:
        raise RuntimeError("no primary slots found in groups")

    n_stars = len(rows)
    out_flux = np.zeros((n_stars, n_frames), dtype=np.float64)
    out_sig = np.zeros((n_stars, n_frames), dtype=np.float64)
    out_x = np.zeros((n_stars, n_frames), dtype=np.float64)
    out_y = np.zeros((n_stars, n_frames), dtype=np.float64)
    out_act = np.zeros((n_stars, n_frames), dtype=bool)
    g_idx = np.zeros(n_stars, dtype=np.int32)
    k_idx = np.zeros(n_stars, dtype=np.int32)
    s_idx = np.zeros(n_stars, dtype=np.int32)
    tess_mag = np.full(n_stars, np.nan, dtype=np.float64)
    src = None if source_ids is None else np.zeros(n_stars, dtype=np.int64)

    for i, (gi, k, si) in enumerate(rows):
        out_flux[i] = flux_np[gi, :, k]
        out_sig[i] = sig_np[gi, :, k]
        out_x[i] = x_np[gi, k, :]
        out_y[i] = y_np[gi, k, :]
        out_act[i] = stamp_active[gi] > 0.5
        g_idx[i] = gi
        k_idx[i] = k
        s_idx[i] = si
        if mags is not None:
            tess_mag[i] = float(mags[si])
        if src is not None:
            src[i] = int(source_ids[si])

    return PrimaryFluxTable(
        flux=out_flux,
        sigma_f=out_sig,
        x=out_x,
        y=out_y,
        active=out_act,
        group_index=g_idx,
        slot_index=k_idx,
        star_index=s_idx,
        tess_mag=tess_mag,
        source_id=src,
        btjd=None if btjd is None else np.asarray(btjd, dtype=np.float64),
    )


def extract_primary_fluxes_frame_chunked(
    params: dict,
    ctx: L.StaticContext,
    data,
    noise,
    weight,
    *,
    primary_index_set: set[int],
    mags: np.ndarray | None = None,
    source_ids: np.ndarray | None = None,
    btjd: np.ndarray | None = None,
    stamp_active=None,
    frame_block: int = 64,
    ridge: float = 1e-6,
) -> PrimaryFluxTable:
    """Forced photometry in frame blocks, retaining only primary fluxes.

    ``forward_model`` materializes templates for every supplied frame.  The
    all-frame path is fine for small bundles but makes a large packed tier's
    temporary template cube dominate host RAM.  All model state is frame-local
    except the fitted parameters, so concatenate the compact primary table
    after independently solving contiguous frame blocks.
    """
    n_frames = int(np.asarray(data).shape[1])
    block = max(1, int(frame_block))
    if n_frames <= block:
        return extract_primary_fluxes(
            params, ctx, data, noise, weight,
            primary_index_set=primary_index_set, mags=mags,
            source_ids=source_ids, btjd=btjd,
            stamp_active=stamp_active, ridge=ridge,
        )
    active = np.asarray(ctx.stamp_active if stamp_active is None else stamp_active)
    merged: PrimaryFluxTable | None = None
    for lo in range(0, n_frames, block):
        hi = min(n_frames, lo + block)
        ctx_block = replace(
            ctx,
            wcs_frame_basis=ctx.wcs_frame_basis[lo:hi],
            w_frame_basis=ctx.w_frame_basis[lo:hi],
            stamp_active=active[:, lo:hi],
        )
        part = extract_primary_fluxes(
            params, ctx_block,
            np.asarray(data)[:, lo:hi], np.asarray(noise)[:, lo:hi], np.asarray(weight)[:, lo:hi],
            primary_index_set=primary_index_set, mags=mags,
            source_ids=source_ids,
            btjd=None if btjd is None else np.asarray(btjd)[lo:hi],
            stamp_active=active[:, lo:hi], ridge=ridge,
        )
        if merged is None:
            shape = (part.flux.shape[0], n_frames)
            merged = PrimaryFluxTable(
                flux=np.empty(shape, dtype=np.float64), sigma_f=np.empty(shape, dtype=np.float64),
                x=np.empty(shape, dtype=np.float64), y=np.empty(shape, dtype=np.float64),
                active=np.empty(shape, dtype=bool), group_index=part.group_index,
                slot_index=part.slot_index, star_index=part.star_index, tess_mag=part.tess_mag,
                source_id=part.source_id,
                btjd=None if btjd is None else np.asarray(btjd, dtype=np.float64),
            )
        elif not np.array_equal(merged.star_index, part.star_index):
            raise RuntimeError("primary rows changed between frame blocks")
        merged.flux[:, lo:hi] = part.flux
        merged.sigma_f[:, lo:hi] = part.sigma_f
        merged.x[:, lo:hi] = part.x
        merged.y[:, lo:hi] = part.y
        merged.active[:, lo:hi] = part.active
    if merged is None:
        raise RuntimeError("no frame blocks were processed")
    return merged


def _bilin_corner_weights(x, y, node_x, node_y):
    """Return i0,j0,w00,w01,w10,w11 for each query (broadcast over trailing axes)."""
    i0, j0, wy, wx = EM.bilinear_cell(
        jnp.asarray(x), jnp.asarray(y), np.asarray(node_x), np.asarray(node_y),
    )
    i0 = np.asarray(i0)
    j0 = np.asarray(j0)
    wy = np.asarray(wy)
    wx = np.asarray(wx)
    w00 = (1.0 - wy) * (1.0 - wx)
    w01 = (1.0 - wy) * wx
    w10 = wy * (1.0 - wx)
    w11 = wy * wx
    return i0, j0, w00, w01, w10, w11


def eval_astar(A_node: np.ndarray, x: np.ndarray, y: np.ndarray, node_x, node_y) -> np.ndarray:
    """Bilinear-blend ``A_node (n_rows,n_cols,T)`` at ``x,y (n_stars,T)`` -> ``(n_stars,T)``."""
    n_stars, n_frames = x.shape
    out = np.zeros((n_stars, n_frames), dtype=np.float64)
    for t in range(n_frames):
        i0, j0, w00, w01, w10, w11 = _bilin_corner_weights(x[:, t], y[:, t], node_x, node_y)
        a = A_node
        out[:, t] = (
            w00 * a[i0, j0, t]
            + w01 * a[i0, j0 + 1, t]
            + w10 * a[i0 + 1, j0, t]
            + w11 * a[i0 + 1, j0 + 1, t]
        )
    return out


def _eval_astar_vectorized(A_node: np.ndarray, x: np.ndarray, y: np.ndarray, node_x, node_y) -> np.ndarray:
    """Same bilinear blend as ``eval_astar`` (identical ``EM.bilinear_cell`` call,
    not an approximation), but batched over (star, frame) in one JAX dispatch
    instead of looping per frame in Python. ``eval_astar`` is called up to 3x
    per ``n_iter`` iteration of ``fit_aperture_coeff``'s alternating LS fit --
    at full-orbit frame counts that Python loop dominates wall time via pure
    per-call JAX dispatch overhead (~1674 calls per invocation, ~40+ invocations
    per fit) with no corresponding increase in actual compute.
    """
    i0, j0, wy, wx = EM.bilinear_cell(jnp.asarray(x), jnp.asarray(y), np.asarray(node_x), np.asarray(node_y))
    i0 = np.asarray(i0)
    j0 = np.asarray(j0)
    wy = np.asarray(wy)
    wx = np.asarray(wx)
    t_idx = np.arange(x.shape[1])[None, :]
    a = A_node
    return (
        (1.0 - wy) * (1.0 - wx) * a[i0, j0, t_idx]
        + (1.0 - wy) * wx * a[i0, j0 + 1, t_idx]
        + wy * (1.0 - wx) * a[i0 + 1, j0, t_idx]
        + wy * wx * a[i0 + 1, j0 + 1, t_idx]
    )


def decode_aperture(A_coeff: np.ndarray, frame_basis: np.ndarray) -> np.ndarray:
    """``A_node = 1 + α``, ``α = A_coeff @ Φ.T`` -> shape ``(n_rows, n_cols, T)``."""
    alpha = np.einsum("ijb,tb->ijt", A_coeff, frame_basis)
    return 1.0 + alpha


def _rescale_to_unit_mean_A(
    A_coeff: np.ndarray,
    F: np.ndarray,
    frame_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fix ``F ↔ A`` scale: enforce ``mean(A_node)=1`` over nodes and time.

    ``A_new = A/μ`` and ``F_new = F*μ`` leave ``F*A`` unchanged. ``A_coeff`` is
    re-fit per node so ``decode_aperture`` matches the scaled ``A_node``.
    """
    Phi = np.asarray(frame_basis, dtype=np.float64)
    A_node = decode_aperture(A_coeff, Phi)
    mu = float(np.mean(A_node))
    if not np.isfinite(mu) or abs(mu) < 1e-30:
        raise RuntimeError("invalid mean(A_node) for scale gauge")
    A_node = A_node / mu
    F = np.asarray(F, dtype=np.float64) * mu
    alpha = A_node - 1.0
    n_rows, n_cols, _ = A_node.shape
    n_basis = Phi.shape[1]
    out = np.zeros((n_rows, n_cols, n_basis), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            out[i, j, :], *_ = np.linalg.lstsq(Phi, alpha[i, j], rcond=None)
    return out, F, A_node


def fit_aperture_coeff(
    table: PrimaryFluxTable,
    frame_basis: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    n_iter: int = 20,
    min_frames: int | None = None,
    sigma_floor: float = 1e-12,
) -> ApertureResult:
    """Alternating IV-weighted LS for ``f ≈ F * A_star``.

    All nodes are free (common-mode ``A(t)`` is allowed). Only the global
    scale ``mean(A_node)=1`` is gauged after the last iteration.
    """
    flux = np.asarray(table.flux, dtype=np.float64)
    sig = np.asarray(table.sigma_f, dtype=np.float64)
    x = np.asarray(table.x, dtype=np.float64)
    y = np.asarray(table.y, dtype=np.float64)
    active = np.asarray(table.active, dtype=bool)
    Phi = np.asarray(frame_basis, dtype=np.float64)  # (T, B)
    n_stars, n_frames = flux.shape
    n_basis = Phi.shape[1]
    if min_frames is None:
        min_frames = max(5, n_frames // 2)

    ok = active & np.isfinite(flux) & np.isfinite(sig) & (flux > 0) & (sig > sigma_floor)
    n_ok = ok.sum(axis=1)
    keep_star = n_ok >= min_frames
    if not np.any(keep_star):
        raise RuntimeError("no stars with enough active positive-flux frames")

    # Precompute bilin corner indices/weights per (star, frame)
    corners = np.zeros((n_stars, n_frames, 4, 2), dtype=np.int32)  # (i,j) per corner
    cweight = np.zeros((n_stars, n_frames, 4), dtype=np.float64)
    for t in range(n_frames):
        i0, j0, w00, w01, w10, w11 = _bilin_corner_weights(x[:, t], y[:, t], node_x, node_y)
        corners[:, t, 0, 0] = i0
        corners[:, t, 0, 1] = j0
        corners[:, t, 1, 0] = i0
        corners[:, t, 1, 1] = j0 + 1
        corners[:, t, 2, 0] = i0 + 1
        corners[:, t, 2, 1] = j0
        corners[:, t, 3, 0] = i0 + 1
        corners[:, t, 3, 1] = j0 + 1
        cweight[:, t, 0] = w00
        cweight[:, t, 1] = w01
        cweight[:, t, 2] = w10
        cweight[:, t, 3] = w11

    def node_flat(i, j):
        return i * n_cols + j

    n_nodes = n_rows * n_cols
    n_free = n_nodes * n_basis

    def pack_design_row(s, t):
        """Design row mapping free α params -> α_star(s,t)."""
        row = np.zeros(n_free, dtype=np.float64)
        for c in range(4):
            ii = int(corners[s, t, c, 0])
            jj = int(corners[s, t, c, 1])
            nf = node_flat(ii, jj)
            w = cweight[s, t, c]
            for b in range(n_basis):
                row[nf * n_basis + b] += w * Phi[t, b]
        return row

    def build_design_matrix_vectorized(star_ok):
        """Same (D, yv, wv) as calling ``pack_design_row`` for every kept
        (s, t) pair in row-major (s outer, t inner) order -- vectorized over
        all active pairs at once instead of one Python call per pair. At
        full population (thousands of stars x thousands of frames x n_iter)
        the per-pair Python loop is minutes of pure call overhead; this does
        the identical accumulation (``np.add.at``, matching the loop's
        ``row[...] += ...``) in a handful of array ops.
        """
        m = ok & star_ok[:, None]
        s_idx, t_idx = np.nonzero(m)
        n_active = s_idx.shape[0]
        D = np.zeros((n_active, n_free), dtype=np.float64)
        row_idx = np.arange(n_active)
        Phi_t = Phi[t_idx]  # (n_active, n_basis)
        b_off = np.arange(n_basis)
        for c in range(4):
            ii = corners[s_idx, t_idx, c, 0]
            jj = corners[s_idx, t_idx, c, 1]
            nf = ii * n_cols + jj
            w = cweight[s_idx, t_idx, c]
            cols = nf[:, None] * n_basis + b_off[None, :]
            vals = w[:, None] * Phi_t
            np.add.at(D, (row_idx[:, None], cols), vals)
        yv = flux[s_idx, t_idx] / F[s_idx] - 1.0
        wv = (F[s_idx] / np.clip(sig[s_idx, t_idx], sigma_floor, None)) ** 2
        return D, yv, wv

    A_coeff = np.zeros((n_rows, n_cols, n_basis), dtype=np.float64)
    F = np.ones(n_stars, dtype=np.float64)
    # Init F from weighted mean flux on kept stars
    for s in range(n_stars):
        m = ok[s]
        if not keep_star[s] or not np.any(m):
            continue
        w = 1.0 / np.clip(sig[s, m], sigma_floor, None) ** 2
        F[s] = float(np.sum(w * flux[s, m]) / np.sum(w))

    rms = np.nan
    for it in range(n_iter):
        A_node = decode_aperture(A_coeff, Phi)
        A_star = _eval_astar_vectorized(A_node, x, y, node_x, node_y)
        A_star = np.clip(A_star, 1e-6, None)

        # Update F
        for s in range(n_stars):
            m = ok[s] & keep_star[s]
            if not np.any(m):
                continue
            w = 1.0 / np.clip(sig[s, m], sigma_floor, None) ** 2
            num = np.sum(w * flux[s, m] * A_star[s, m])
            den = np.sum(w * A_star[s, m] ** 2)
            F[s] = float(num / max(den, 1e-30))

        # Update A_coeff via weighted LS on (f/F - 1) = α_star
        star_ok = keep_star & np.isfinite(F) & (np.abs(F) >= 1e-30)
        D, yv, wv = build_design_matrix_vectorized(star_ok)

        if D.shape[0] == 0:
            raise RuntimeError("empty design in aperture fit")
        sw = np.sqrt(np.clip(wv, 0.0, None))
        beta, *_ = np.linalg.lstsq(D * sw[:, None], yv * sw, rcond=None)
        A_coeff = np.zeros((n_rows, n_cols, n_basis), dtype=np.float64)
        for nf in range(n_nodes):
            i = nf // n_cols
            j = nf % n_cols
            A_coeff[i, j, :] = beta[nf * n_basis : (nf + 1) * n_basis]

        A_node = decode_aperture(A_coeff, Phi)
        A_star = _eval_astar_vectorized(A_node, x, y, node_x, node_y)
        m = ok & keep_star[:, None]
        if np.any(m):
            pred = F[:, None] * A_star
            # Fractional residual: formal σ_f is often ≪ systematics, so χ is
            # not a useful scalar here.
            rms = float(np.sqrt(np.mean(((flux[m] - pred[m]) / np.clip(flux[m], 1e-30, None)) ** 2)))
        else:
            rms = float("nan")

    A_coeff, F, A_node = _rescale_to_unit_mean_A(A_coeff, F, Phi)
    A_star = _eval_astar_vectorized(A_node, x, y, node_x, node_y)
    A_star = np.clip(A_star, 1e-6, None)
    flux_corr = np.full_like(flux, np.nan)
    m = ok & keep_star[:, None]
    flux_corr[m] = flux[m] / A_star[m]
    # Never emit a correction for a star that did not constrain the fit; see
    # ``ApertureResult.star_kept``.
    F = np.where(keep_star, F, np.nan)

    return ApertureResult(
        A_coeff=A_coeff,
        A_node=A_node,
        A_star=A_star,
        F=F,
        flux_corr=flux_corr,
        frame_basis=Phi,
        node_x=np.asarray(node_x, dtype=np.float64),
        node_y=np.asarray(node_y, dtype=np.float64),
        n_iter=n_iter,
        rms_resid=rms,
        star_kept=keep_star.copy(),
    )


def fit_aperture_from_w0(
    table: PrimaryFluxTable,
    w_of_t: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    mode: str = "node_scale",
    mode_idx: int = 0,
    star_keep: np.ndarray | None = None,
    n_iter: int = 20,
    min_frames: int | None = None,
    sigma_floor: float = 1e-12,
) -> ApertureResult:
    """Fit ``A`` with temporal shape locked to ePSF ``w_k(t)``.

    Modes (see plan M1–M3):

    - ``global``: ``α_ij(t) = c * w_k(t)`` (one scale)
    - ``node_scale``: ``α_ij(t) = c_ij * w_k(t)``
    - ``node_affine``: ``α_ij(t) = c_ij * w_k(t) + b_ij`` (static per-node offset)

    Uses the same alternating IV-weighted LS as ``fit_aperture_coeff``, but the
    temporal part is fixed to the checkpoint ``w_k`` instead of free B-splines.
    """
    mode = str(mode)
    if mode not in ("global", "node_scale", "node_affine"):
        raise ValueError(f"unknown mode={mode!r}")

    flux = np.asarray(table.flux, dtype=np.float64)
    sig = np.asarray(table.sigma_f, dtype=np.float64)
    x = np.asarray(table.x, dtype=np.float64)
    y = np.asarray(table.y, dtype=np.float64)
    active = np.asarray(table.active, dtype=bool)
    w_of_t = np.asarray(w_of_t, dtype=np.float64)
    if w_of_t.ndim == 1:
        w0 = w_of_t
    else:
        w0 = w_of_t[:, int(mode_idx)]
    n_stars, n_frames = flux.shape
    if w0.shape[0] != n_frames:
        raise ValueError(f"w0 length {w0.shape[0]} != n_frames {n_frames}")
    if min_frames is None:
        min_frames = max(5, n_frames // 2)

    ok = active & np.isfinite(flux) & np.isfinite(sig) & (flux > 0) & (sig > sigma_floor)
    n_ok = ok.sum(axis=1)
    keep_star = n_ok >= min_frames
    if star_keep is not None:
        keep_star = keep_star & np.asarray(star_keep, dtype=bool)
    if not np.any(keep_star):
        raise RuntimeError("no stars kept for w0-constrained aperture fit")

    corners = np.zeros((n_stars, n_frames, 4, 2), dtype=np.int32)
    cweight = np.zeros((n_stars, n_frames, 4), dtype=np.float64)
    for t in range(n_frames):
        i0, j0, w00, w01, w10, w11 = _bilin_corner_weights(x[:, t], y[:, t], node_x, node_y)
        corners[:, t, 0, 0] = i0
        corners[:, t, 0, 1] = j0
        corners[:, t, 1, 0] = i0
        corners[:, t, 1, 1] = j0 + 1
        corners[:, t, 2, 0] = i0 + 1
        corners[:, t, 2, 1] = j0
        corners[:, t, 3, 0] = i0 + 1
        corners[:, t, 3, 1] = j0 + 1
        cweight[:, t, 0] = w00
        cweight[:, t, 1] = w01
        cweight[:, t, 2] = w10
        cweight[:, t, 3] = w11

    def node_flat(i, j):
        return i * n_cols + j

    n_nodes = n_rows * n_cols
    if mode == "global":
        n_free = 1
        Phi = np.asarray(w0, dtype=np.float64).reshape(-1, 1)
    elif mode == "node_scale":
        n_free = n_nodes
        Phi = np.asarray(w0, dtype=np.float64).reshape(-1, 1)
    else:
        n_free = 2 * n_nodes
        Phi = np.column_stack([w0, np.ones(n_frames, dtype=np.float64)])

    def pack_design_row(s, t):
        row = np.zeros(n_free, dtype=np.float64)
        wt = float(w0[t])
        for c in range(4):
            ii = int(corners[s, t, c, 0])
            jj = int(corners[s, t, c, 1])
            nf = node_flat(ii, jj)
            bw = float(cweight[s, t, c])
            if mode == "global":
                row[0] += bw * wt
            elif mode == "node_scale":
                row[nf] += bw * wt
            else:
                row[nf] += bw * wt
                row[n_nodes + nf] += bw
        return row

    A_coeff = np.zeros((n_rows, n_cols, Phi.shape[1]), dtype=np.float64)
    F = np.ones(n_stars, dtype=np.float64)
    for s in range(n_stars):
        m = ok[s]
        if not keep_star[s] or not np.any(m):
            continue
        w = 1.0 / np.clip(sig[s, m], sigma_floor, None) ** 2
        F[s] = float(np.sum(w * flux[s, m]) / np.sum(w))

    rms = np.nan
    for _it in range(n_iter):
        A_node = decode_aperture(A_coeff, Phi)
        A_star = eval_astar(A_node, x, y, node_x, node_y)
        A_star = np.clip(A_star, 1e-6, None)

        for s in range(n_stars):
            if not keep_star[s]:
                continue
            m = ok[s]
            if not np.any(m):
                continue
            w = 1.0 / np.clip(sig[s, m], sigma_floor, None) ** 2
            num = np.sum(w * flux[s, m] * A_star[s, m])
            den = np.sum(w * A_star[s, m] ** 2)
            F[s] = float(num / max(den, 1e-30))

        rows = []
        rhs = []
        wgt = []
        for s in range(n_stars):
            if not keep_star[s] or not np.isfinite(F[s]) or abs(F[s]) < 1e-30:
                continue
            for t in range(n_frames):
                if not ok[s, t]:
                    continue
                rows.append(pack_design_row(s, t))
                rhs.append(flux[s, t] / F[s] - 1.0)
                wgt.append((F[s] / max(sig[s, t], sigma_floor)) ** 2)

        if not rows:
            raise RuntimeError("empty design in w0 aperture fit")
        D = np.asarray(rows, dtype=np.float64)
        yv = np.asarray(rhs, dtype=np.float64)
        wv = np.asarray(wgt, dtype=np.float64)
        sw = np.sqrt(np.clip(wv, 0.0, None))
        beta, *_ = np.linalg.lstsq(D * sw[:, None], yv * sw, rcond=None)

        A_coeff = np.zeros((n_rows, n_cols, Phi.shape[1]), dtype=np.float64)
        if mode == "global":
            A_coeff[:, :, 0] = float(beta[0])
        elif mode == "node_scale":
            for nf in range(n_nodes):
                i = nf // n_cols
                j = nf % n_cols
                A_coeff[i, j, 0] = beta[nf]
        else:
            for nf in range(n_nodes):
                i = nf // n_cols
                j = nf % n_cols
                A_coeff[i, j, 0] = beta[nf]
                A_coeff[i, j, 1] = beta[n_nodes + nf]

        A_node = decode_aperture(A_coeff, Phi)
        A_star = eval_astar(A_node, x, y, node_x, node_y)
        m = ok & keep_star[:, None]
        if np.any(m):
            pred = F[:, None] * A_star
            rms = float(np.sqrt(np.mean(((flux[m] - pred[m]) / np.clip(flux[m], 1e-30, None)) ** 2)))
        else:
            rms = float("nan")

    A_coeff, F, A_node = _rescale_to_unit_mean_A(A_coeff, F, Phi)
    A_star = eval_astar(A_node, x, y, node_x, node_y)
    A_star = np.clip(A_star, 1e-6, None)
    flux_corr = np.full_like(flux, np.nan)
    m = ok & keep_star[:, None]
    flux_corr[m] = flux[m] / A_star[m]
    # Never emit a correction for a star that did not constrain the fit; see
    # ``ApertureResult.star_kept``.
    F = np.where(keep_star, F, np.nan)

    return ApertureResult(
        A_coeff=A_coeff,
        A_node=A_node,
        A_star=A_star,
        F=F,
        flux_corr=flux_corr,
        frame_basis=Phi,
        node_x=np.asarray(node_x, dtype=np.float64),
        node_y=np.asarray(node_y, dtype=np.float64),
        n_iter=n_iter,
        rms_resid=rms,
        star_kept=keep_star.copy(),
    )


def apply_aperture_result(
    table: PrimaryFluxTable,
    result: ApertureResult,
    active: np.ndarray | None = None,
) -> ApertureResult:
    """Evaluate a fitted ``A_node`` field on all stars in ``table``."""
    if active is None:
        active = np.asarray(table.active, dtype=bool)
    else:
        active = np.asarray(active, dtype=bool)
    A_star = eval_astar(result.A_node, table.x, table.y, result.node_x, result.node_y)
    A_star = np.clip(A_star, 1e-6, None)
    flux = np.asarray(table.flux, dtype=np.float64)
    flux_corr = np.full_like(flux, np.nan)
    m = active & np.isfinite(flux) & (flux > 0)
    flux_corr[m] = flux[m] / A_star[m]
    F = np.full(flux.shape[0], np.nan)
    for s in range(flux.shape[0]):
        ms = m[s]
        if ms.sum():
            F[s] = float(np.mean(flux_corr[s, ms]))
    return ApertureResult(
        A_coeff=result.A_coeff,
        A_node=result.A_node,
        A_star=A_star,
        F=F,
        flux_corr=flux_corr,
        frame_basis=result.frame_basis,
        node_x=result.node_x,
        node_y=result.node_y,
        n_iter=result.n_iter,
        rms_resid=result.rms_resid,
    )


def summarize_flux_trend(flux: np.ndarray, active: np.ndarray | None = None) -> dict:
    """Population mean/SEM and PC1 variance-explained of normalized light curves."""
    f = np.asarray(flux, dtype=np.float64)
    n_stars, n_frames = f.shape
    if active is None:
        active = np.ones_like(f, dtype=bool)
    else:
        active = np.asarray(active, dtype=bool)

    norm = np.full_like(f, np.nan)
    for s in range(n_stars):
        m = active[s] & np.isfinite(f[s]) & (f[s] > 0)
        if m.sum() < 2:
            continue
        mu = np.mean(f[s, m])
        if abs(mu) < 1e-30:
            continue
        norm[s, m] = f[s, m] / mu - 1.0

    mean_t = np.nanmean(norm, axis=0)
    sem_t = np.nanstd(norm, axis=0, ddof=1) / np.sqrt(np.maximum(np.sum(np.isfinite(norm), axis=0), 1))
    X = norm.copy()
    # fill nan with per-star mean of available (0 after norm) for PCA
    for s in range(n_stars):
        m = np.isfinite(X[s])
        if m.sum() == 0:
            X[s] = 0.0
        else:
            X[s, ~m] = 0.0
    X = X - X.mean(axis=1, keepdims=True)
    # drop all-zero stars
    keep = np.any(np.abs(X) > 0, axis=1)
    pc1_frac = float("nan")
    if keep.sum() >= 2:
        U, S, Vt = np.linalg.svd(X[keep], full_matrices=False)
        e = S**2
        pc1_frac = float(e[0] / e.sum()) if e.sum() > 0 else float("nan")

    return {
        "mean_frac": mean_t,
        "sem_frac": sem_t,
        "ptp_mean_frac": float(np.nanmax(mean_t) - np.nanmin(mean_t)) if np.any(np.isfinite(mean_t)) else float("nan"),
        "pc1_variance_explained": pc1_frac,
    }


def plot_per_star_corrected_lcs(
    table: PrimaryFluxTable,
    flux_corr: np.ndarray,
    A_star: np.ndarray,
    out_path: Path,
    *,
    run_id: str,
    subtitle: str = "",
    max_stars_plot: int = 40,
    nn_dist_px: np.ndarray | None = None,
) -> Path:
    """Grid of per-star fractional LCs: raw vs aperture-corrected (normalized to weighted mean)."""
    from matplotlib import pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(table.btjd, dtype=float)
    flux = np.asarray(table.flux, dtype=np.float64)
    corr = np.asarray(flux_corr, dtype=np.float64)
    sig = np.asarray(table.sigma_f, dtype=np.float64)
    A = np.asarray(A_star, dtype=np.float64)
    active = np.asarray(table.active, dtype=bool)
    mag = np.asarray(table.tess_mag, dtype=float)
    sid = table.source_id

    order = np.argsort(mag)[:max_stars_plot]
    n_show = len(order)
    ncols = 4
    nrows = int(np.ceil(max(n_show, 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for panel, s in enumerate(order):
        ax = axes[panel]
        m = active[s] & np.isfinite(flux[s]) & (flux[s] > 0)
        m &= np.isfinite(corr[s]) & np.isfinite(A[s]) & (A[s] > 0)
        if m.sum() < 2:
            ax.set_visible(False)
            continue
        w = 1.0 / np.clip(sig[s, m], 1e-12, None) ** 2
        f0 = float(np.sum(w * flux[s, m]) / np.sum(w))
        ax.plot(t[m], flux[s, m] / f0 - 1.0, "o", ms=2.0, color="0.65", alpha=0.65, label="raw", zorder=1)
        yerr = (sig[s, m] / f0) / np.clip(A[s, m], 1e-6, None)
        ax.errorbar(
            t[m], corr[s, m] / f0 - 1.0, yerr=yerr, fmt="s-", ms=2.2, lw=0.8,
            color="C3", alpha=0.9, label="A-corr", zorder=2,
        )
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.35)
        lab = f"mag={mag[s]:.2f}"
        if nn_dist_px is not None:
            lab += f" nn={float(nn_dist_px[s]):.1f}px"
        if sid is not None:
            lab = f"id={int(sid[s])} " + lab
        ax.set_title(lab, fontsize=7)
        ax.grid(True, alpha=0.25)
        if panel == 0:
            ax.legend(fontsize=7)
    for ax in axes[n_show:]:
        ax.set_visible(False)
    title = f"{run_id}: per-star aperture-corrected LCs"
    if subtitle:
        title += f"  ({subtitle})"
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_aperture_correction_on_bundle(
    bundle: dict,
    params: dict,
    *,
    n_iter: int = 20,
) -> tuple[PrimaryFluxTable, ApertureResult, dict, dict]:
    """High-level: extract fluxes, fit A, return before/after trend summaries."""
    fd = bundle["fd"]
    ctx = bundle["ctx"]
    primary_index_set = bundle["primary_index_set"]
    mags = bundle["mags"]
    expanded = bundle["expanded_stars"]
    source_ids = None
    if "source_id" in expanded.columns:
        source_ids = expanded["source_id"].to_numpy()
    btjd = np.array([f.btjd for f in bundle["frames"]], dtype=np.float64)

    table = extract_primary_fluxes(
        params, ctx, fd.data, fd.noise, fd.weight,
        primary_index_set=primary_index_set,
        mags=mags,
        source_ids=source_ids,
        btjd=btjd,
        stamp_active=ctx.stamp_active,
    )
    node_x = np.asarray(ctx.node_x)
    node_y = np.asarray(ctx.node_y)
    frame_basis = np.asarray(ctx.w_frame_basis)
    n_rows = int(np.asarray(EM.decode_epsf_base(params["epsf_base_raw"])).shape[0])
    n_cols = int(np.asarray(EM.decode_epsf_base(params["epsf_base_raw"])).shape[1])

    result = fit_aperture_coeff(
        table, frame_basis, node_x, node_y,
        n_rows=n_rows, n_cols=n_cols, n_iter=n_iter,
    )
    before = summarize_flux_trend(table.flux, table.active)
    after = summarize_flux_trend(result.flux_corr, table.active)
    return table, result, before, after


def build_fit_bundle_context(fit_bundle, stamp_active: np.ndarray | None = None):
    """Build the packed renderer context without fitting an aperture field."""
    from . import temporal as T
    groups = fit_bundle.group_set()
    stamp_active_arr = (
        np.asarray(stamp_active, dtype=bool)
        if stamp_active is not None else np.asarray(fit_bundle.mask_active, dtype=bool)
    )
    ctx_kw: dict = dict(
        cheb_static=fit_bundle.cheb_static, wcs_frame_basis=fit_bundle.wcs_frame_basis,
        w_frame_basis=fit_bundle.w_frame_basis, epsf_grid=fit_bundle.epsf_grid,
        groups=groups, ra=fit_bundle.ra, dec=fit_bundle.dec,
        stamp_center_x=fit_bundle.stamp_center_x, stamp_center_y=fit_bundle.stamp_center_y,
        t_exp_sec=fit_bundle.t_exp_sec, stamp_snr_weight=fit_bundle.stamp_snr_weight,
        fit_radius=fit_bundle.fit_radius_stage23, stamp_active=stamp_active_arr,
        x_lin=fit_bundle.x_lin, y_lin=fit_bundle.y_lin, cheb_basis=fit_bundle.cheb_basis,
        bp_rp=getattr(fit_bundle, "bp_rp", None),
        colour_ref=L.colour_ref_from_bundle(fit_bundle),
    )
    if fit_bundle.is_packed:
        ctx_kw["pix_x"] = fit_bundle.pix_x
        ctx_kw["pix_y"] = fit_bundle.pix_y
        ctx_kw["pix_valid"] = fit_bundle.pix_valid
    return L.build_static_context(**ctx_kw), stamp_active_arr


def run_aperture_correction_on_fit_bundle(
    fit_bundle,
    params: dict,
    *,
    btjd: np.ndarray,
    mags: np.ndarray,
    primary_index_set: set[int],
    source_ids: np.ndarray | None = None,
    stamp_active: np.ndarray | None = None,
    n_iter: int = 20,
) -> tuple[PrimaryFluxTable, ApertureResult, dict, dict]:
    """Same as ``run_aperture_correction_on_bundle`` but from a serialized ``FitBundle``.

    ``btjd`` / ``mags`` / ``primary_index_set`` are not stored in the npz; supply
    them from the workspace Gaia catalog + run meta (primaries are the first
    ``n_fit_stars`` expanded rows after ``merge_star_tables``).
    """
    # A tier-segmented bundle deliberately has no dense stamp cube.  Accessing
    # ``fit_bundle.data/noise/weight`` below would reconstruct three globally
    # P=1024-padded (G,T,P) arrays.  S52's three reconstructions alone are
    # about 55 GiB before JAX temporaries.  Run forced photometry one native
    # (K,P) tier at a time instead, retaining only the small primary-flux
    # table between tiers.
    if fit_bundle.packed_tiers is not None:
        return _run_aperture_correction_tiered_fit_bundle(
            fit_bundle,
            params,
            btjd=btjd,
            mags=mags,
            primary_index_set=primary_index_set,
            source_ids=source_ids,
            stamp_active=stamp_active,
            n_iter=n_iter,
        )

    from . import temporal as T
    ctx, stamp_active_arr = build_fit_bundle_context(fit_bundle, stamp_active)
    wcs_n = int(fit_bundle.wcs_frame_basis.shape[1])
    w_n = int(fit_bundle.w_frame_basis.shape[1])
    fd = FIT.FitData(
        ctx=ctx,
        data=jnp.asarray(fit_bundle.data),
        noise=jnp.asarray(fit_bundle.noise),
        weight=jnp.asarray(fit_bundle.weight),
        wcs_second_diff=T.second_difference_matrix(wcs_n),
        w_second_diff=T.second_difference_matrix(w_n),
        epsf_modes_init=jnp.asarray(fit_bundle.epsf_modes),
        mask_stamp_active=np.asarray(stamp_active_arr, dtype=np.float32),
    )
    import pandas as pd

    expanded = pd.DataFrame({"tess_mag": np.asarray(mags, dtype=float)})
    if source_ids is not None:
        expanded["source_id"] = np.asarray(source_ids)
    frames = [type("F", (), {"btjd": float(t)})() for t in np.asarray(btjd, dtype=float)]
    return run_aperture_correction_on_bundle(
        {
            "fd": fd,
            "ctx": ctx,
            "primary_index_set": set(int(i) for i in primary_index_set),
            "mags": np.asarray(mags, dtype=float),
            "expanded_stars": expanded,
            "frames": frames,
        },
        params,
        n_iter=n_iter,
    )


def _concat_primary_flux_tables(tables: list[PrimaryFluxTable]) -> PrimaryFluxTable:
    """Combine independently solved native packed tiers."""
    if not tables:
        raise RuntimeError("no primary slots found in packed tiers")
    first = tables[0]
    source_id = None
    if all(table.source_id is not None for table in tables):
        source_id = np.concatenate([np.asarray(table.source_id) for table in tables])
    return PrimaryFluxTable(
        flux=np.concatenate([table.flux for table in tables]),
        sigma_f=np.concatenate([table.sigma_f for table in tables]),
        x=np.concatenate([table.x for table in tables]),
        y=np.concatenate([table.y for table in tables]),
        active=np.concatenate([table.active for table in tables]),
        group_index=np.concatenate([table.group_index for table in tables]),
        slot_index=np.concatenate([table.slot_index for table in tables]),
        star_index=np.concatenate([table.star_index for table in tables]),
        tess_mag=np.concatenate([table.tess_mag for table in tables]),
        source_id=source_id,
        btjd=None if first.btjd is None else np.asarray(first.btjd),
    )


def _run_aperture_correction_tiered_fit_bundle(
    fit_bundle,
    params: dict,
    *,
    btjd: np.ndarray,
    mags: np.ndarray,
    primary_index_set: set[int],
    source_ids: np.ndarray | None,
    stamp_active: np.ndarray | None,
    n_iter: int,
) -> tuple[PrimaryFluxTable, ApertureResult, dict, dict]:
    """Tier-native aperture correction without dense packed reconstruction."""
    import gc
    import jax
    from . import temporal as T

    active = (
        np.asarray(stamp_active, dtype=np.float32)
        if stamp_active is not None else np.asarray(fit_bundle.mask_active, dtype=np.float32)
    )
    groups_and_tiers = zip(fit_bundle.packed_bucket_plan(), fit_bundle.packed_tiers)
    tables: list[PrimaryFluxTable] = []
    for (groups, center_x, center_y, group_idx, _k_tier, _p_tier), tier in groups_and_tiers:
        group_idx = np.asarray(group_idx, dtype=np.int32)
        ctx = L.build_static_context(
            cheb_static=fit_bundle.cheb_static,
            wcs_frame_basis=fit_bundle.wcs_frame_basis,
            w_frame_basis=fit_bundle.w_frame_basis,
            epsf_grid=fit_bundle.epsf_grid,
            groups=groups,
            ra=fit_bundle.ra,
            dec=fit_bundle.dec,
            stamp_center_x=center_x,
            stamp_center_y=center_y,
            t_exp_sec=fit_bundle.t_exp_sec,
            stamp_snr_weight=np.asarray(fit_bundle.stamp_snr_weight)[group_idx],
            fit_radius=np.asarray(fit_bundle.fit_radius_stage23)[group_idx],
            bp_rp=getattr(fit_bundle, "bp_rp", None),
            colour_ref=L.colour_ref_from_bundle(fit_bundle),
            x_lin=fit_bundle.x_lin,
            y_lin=fit_bundle.y_lin,
            cheb_basis=fit_bundle.cheb_basis,
            pix_x=tier.pix_x,
            pix_y=tier.pix_y,
            pix_valid=tier.pix_valid,
            is_epsf_contributor=np.asarray(fit_bundle.is_epsf_contributor)[group_idx],
        )
        table = extract_primary_fluxes_frame_chunked(
            params, ctx, tier.data, tier.noise, tier.weight_u8,
            primary_index_set=primary_index_set,
            mags=mags,
            source_ids=source_ids,
            btjd=btjd,
            stamp_active=active[group_idx],
            frame_block=64,
        )
        # ``extract_primary_fluxes`` indexes the tier-local GroupSet; publish
        # the original bundle group indices for downstream diagnostics.
        table.group_index = group_idx[np.asarray(table.group_index, dtype=np.int32)]
        tables.append(table)
        del ctx, table
        jax.clear_caches()
        gc.collect()

    table = _concat_primary_flux_tables(tables)
    n_rows = int(np.asarray(EM.decode_epsf_base(params["epsf_base_raw"])).shape[0])
    n_cols = int(np.asarray(EM.decode_epsf_base(params["epsf_base_raw"])).shape[1])
    result = fit_aperture_coeff(
        table,
        np.asarray(fit_bundle.w_frame_basis),
        np.asarray(fit_bundle.epsf_grid.node_x),
        np.asarray(fit_bundle.epsf_grid.node_y),
        n_rows=n_rows,
        n_cols=n_cols,
        n_iter=n_iter,
    )
    before = summarize_flux_trend(table.flux, table.active)
    after = summarize_flux_trend(result.flux_corr, table.active)
    return table, result, before, after


def save_aperture_outputs(
    out_dir: Path,
    table: PrimaryFluxTable,
    result: ApertureResult,
    before: dict,
    after: dict,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "aperture_coeff.npz",
        A_coeff=result.A_coeff,
        A_node=result.A_node,
        frame_basis=result.frame_basis,
        node_x=result.node_x,
        node_y=result.node_y,
        rms_resid=result.rms_resid,
        n_iter=result.n_iter,
    )
    payload = {
        "flux_ls": table.flux,
        "sigma_f": table.sigma_f,
        "flux_corr": result.flux_corr,
        "A_star": result.A_star,
        "F": result.F,
        "x": table.x,
        "y": table.y,
        "active": table.active,
        "group_index": table.group_index,
        "slot_index": table.slot_index,
        "star_index": table.star_index,
        "tess_mag": table.tess_mag,
    }
    if table.source_id is not None:
        payload["source_id"] = table.source_id
    if table.btjd is not None:
        payload["btjd"] = table.btjd
    np.savez(out_dir / "flux_table.npz", **payload)

    import json
    summary = {
        "before_ptp_mean_frac": before["ptp_mean_frac"],
        "after_ptp_mean_frac": after["ptp_mean_frac"],
        "before_pc1_ve": before["pc1_variance_explained"],
        "after_pc1_ve": after["pc1_variance_explained"],
        "rms_resid": result.rms_resid,
        "n_stars": int(table.flux.shape[0]),
        "n_frames": int(table.flux.shape[1]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
