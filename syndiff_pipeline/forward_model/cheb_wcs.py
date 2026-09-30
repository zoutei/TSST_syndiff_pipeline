# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""JAX ideal(Gaia)->detector forward model: gnomonic TAN + region-local Chebyshev.

Chain (see plan):

    (alpha, delta) --gnomonic TAN, fixed (alpha0,delta0)--> (xi, eta) deg
                   --fixed CD^-1 (linear only, no SIP)-----> (x_lin, y_lin) px
                   --normalise about REGION centre---------> (xhat, yhat) in [-1,1]
    x(t) = x_lin + sum_{i+j<=d} a_ij(t) T_i(xhat) T_j(yhat)
    y(t) = y_lin + sum_{i+j<=d} b_ij(t) T_i(xhat) T_j(yhat)

This is the *opposite* direction from the existing dev Sci2Idl/Chebyshev
notebooks (which fit detector->ideal). ``x_lin, y_lin`` — the pure linear-WCS
prediction — are time-independent (fixed catalog RA/Dec, frozen WCS), so the
per-star Chebyshev basis vector is computed once; only the (small) coefficient
contraction with the temporal basis varies per frame. This is why
``eval_all_positions`` is cheap relative to rendering.

Coefficients are stored as one ``coeff_matrix`` of shape ``(2*n_terms, n_basis)``
(rows 0:n_terms = x axis, n_terms:2*n_terms = y axis), matching
``dev/temporal_wcs_poly/temporal_model.py``'s convention, so warm-start values
transfer directly into the JAX pytree.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from . import _bootstrap  # noqa: F401

# Prep-only imports (RegionSpec / cheb_poly_fit / sip_poly_fit → astropy) are
# lazy so --from-bundle Adam never needs them; exponents live in the bundle.


# ---------------------------------------------------------------------------
# Static (non-trainable) model definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChebWcsStatic:
    ra0_deg: float
    dec0_deg: float
    cd_inv: np.ndarray  # (2, 2), fixed
    crpix: np.ndarray  # (2,), fixed [crpix1, crpix2]
    center: np.ndarray  # (2,), region center [cx, cy] px
    half_extents: np.ndarray  # (2,), region half-extents [sx, sy] px
    poly_degree: int
    exponents: tuple  # ((exp_x, exp_y), ...), length n_terms, Sci2Idl order

    @property
    def n_terms(self) -> int:
        return len(self.exponents)

    @classmethod
    def from_wcs(cls, wcs, region, *, poly_degree: int) -> "ChebWcsStatic":
        from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents  # noqa: E402  (prep-only)

        cd = np.asarray(wcs.wcs.cd, dtype=float)
        cd_inv = np.linalg.inv(cd)
        crpix = np.asarray(wcs.wcs.crpix, dtype=float)
        cx, cy = region.center
        sx, sy = region.half_extents
        return cls(
            ra0_deg=float(wcs.wcs.crval[0]),
            dec0_deg=float(wcs.wcs.crval[1]),
            cd_inv=cd_inv,
            crpix=crpix,
            center=np.array([cx, cy], dtype=float),
            half_extents=np.array([sx, sy], dtype=float),
            poly_degree=int(poly_degree),
            exponents=tuple(sci2idl_exponents(poly_degree)),
        )


def zero_coeff_matrix(static: ChebWcsStatic, n_basis: int) -> jnp.ndarray:
    """Identity-linear residual form: zero coefficients == pure linear WCS."""
    return jnp.zeros((2 * static.n_terms, n_basis), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Forward math (JAX, differentiable)
# ---------------------------------------------------------------------------


def gnomonic_project(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Standard-coordinate (xi, eta) tangent-plane projection, in degrees.

    Same formula as ``wcs_conversion.forward_tan_projection``, vectorized and
    without the CRVAL special case (the general formula is well-defined there:
    xi=eta=0 falls out naturally when ra=ra0, dec=dec0).
    """
    ra = jnp.deg2rad(ra_deg)
    dec = jnp.deg2rad(dec_deg)
    ra0 = jnp.deg2rad(ra0_deg)
    dec0 = jnp.deg2rad(dec0_deg)
    dra = ra - ra0
    cos_dec, sin_dec = jnp.cos(dec), jnp.sin(dec)
    cos_dec0, sin_dec0 = jnp.cos(dec0), jnp.sin(dec0)
    cos_dra, sin_dra = jnp.cos(dra), jnp.sin(dra)
    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    xi = cos_dec * sin_dra / cos_c
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / cos_c
    return jnp.rad2deg(xi), jnp.rad2deg(eta)


def linear_pixel_from_tan(xi_deg, eta_deg, cd_inv, crpix):
    """(xi, eta) deg -> absolute pixel (x_lin, y_lin) via fixed CD^-1, no SIP."""
    u = cd_inv[0, 0] * xi_deg + cd_inv[0, 1] * eta_deg
    v = cd_inv[1, 0] * xi_deg + cd_inv[1, 1] * eta_deg
    x_lin = u + (crpix[0] - 1.0)
    y_lin = v + (crpix[1] - 1.0)
    return x_lin, y_lin


def chebyshev_vandermonde(x, degree: int):
    """T_0..T_degree(x) via recurrence. x: (N,) -> (N, degree+1)."""
    cols = [jnp.ones_like(x)]
    if degree >= 1:
        cols.append(x)
    for _ in range(2, degree + 1):
        cols.append(2.0 * x * cols[-1] - cols[-2])
    return jnp.stack(cols, axis=-1)


def cheb_star_basis(xhat, yhat, poly_degree: int, exponents: tuple):
    """(N,) xhat,yhat in [-1,1] -> (N, n_terms) product-Chebyshev design."""
    tx = chebyshev_vandermonde(xhat, poly_degree)
    ty = chebyshev_vandermonde(yhat, poly_degree)
    cols = [tx[:, ei] * ty[:, ej] for ei, ej in exponents]
    return jnp.stack(cols, axis=-1)


def linear_predict(ra_deg, dec_deg, static: ChebWcsStatic):
    """Pure linear-WCS pixel prediction (no distortion). Time-independent."""
    xi, eta = gnomonic_project(ra_deg, dec_deg, static.ra0_deg, static.dec0_deg)
    return linear_pixel_from_tan(xi, eta, jnp.asarray(static.cd_inv), jnp.asarray(static.crpix))


def star_basis(ra_deg, dec_deg, static: ChebWcsStatic):
    """Per-star (time-independent) linear prediction + Chebyshev design row."""
    x_lin, y_lin = linear_predict(ra_deg, dec_deg, static)
    xhat = (x_lin - static.center[0]) / static.half_extents[0]
    yhat = (y_lin - static.center[1]) / static.half_extents[1]
    basis = cheb_star_basis(xhat, yhat, static.poly_degree, static.exponents)
    return x_lin, y_lin, basis


def eval_all_positions(x_lin, y_lin, basis, coeff_matrix, frame_basis, n_terms: int):
    """All (star, frame) positions at once.

    x_lin, y_lin: (n_stars,)
    basis: (n_stars, n_terms)
    coeff_matrix: (2*n_terms, n_basis)
    frame_basis: (n_frames, n_basis)
    returns x, y: (n_stars, n_frames)
    """
    frame_coeff = coeff_matrix @ frame_basis.T  # (2*n_terms, n_frames)
    dx = basis @ frame_coeff[:n_terms, :]
    dy = basis @ frame_coeff[n_terms:, :]
    return x_lin[:, None] + dx, y_lin[:, None] + dy


def eval_positions_at_frame_index(
    ra_deg,
    dec_deg,
    coeff_matrix,
    static: ChebWcsStatic,
    frame_basis: jnp.ndarray,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warmstart-distorted (x, y) at a single frame index, numpy float arrays."""
    x_lin, y_lin, basis = star_basis(jnp.asarray(ra_deg), jnp.asarray(dec_deg), static)
    fb = jnp.asarray(frame_basis, dtype=jnp.float32)[frame_index : frame_index + 1]
    x_t, y_t = eval_all_positions(x_lin, y_lin, basis, coeff_matrix, fb, static.n_terms)
    return np.asarray(x_t[:, 0], dtype=float), np.asarray(y_t[:, 0], dtype=float)


# ---------------------------------------------------------------------------
# Warm start (numpy, per-frame lstsq + MAD clip, then lstsq onto temporal basis)
# ---------------------------------------------------------------------------


def fit_frame_cheb_warmstart(
    x_lin: np.ndarray,
    y_lin: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    static: ChebWcsStatic,
    *,
    n_sigma: float = 3.0,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame Chebyshev coefficient fit (ideal->detector) with symmetric MAD clip.

    Reuses ``cheb_design_matrix`` (the orthogonal-basis machinery) but not
    ``chebfit_axis``/``iterative_clip_cheb_du_dv`` directly: those assume the
    basis is CRPIX-centered, which is wrong for a region far from CRPIX (plan
    Finding 8). Also fixes the asymmetric clip bug in the lifted originals
    (``|resid| < med + n*scale`` is one-sided; this uses ``|resid-med| < n*scale``).
    """
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.cheb_poly_fit import cheb_design_matrix  # noqa: E402  (warmstart-only)

    xhat = (x_lin - static.center[0]) / static.half_extents[0]
    yhat = (y_lin - static.center[1]) / static.half_extents[1]
    design = cheb_design_matrix(xhat, yhat, static.poly_degree)  # (N, n_terms)
    target_x = x_obs - x_lin
    target_y = y_obs - y_lin

    mask = np.ones(len(x_lin), dtype=bool)
    coeff_x = coeff_y = None
    for _ in range(max_iter):
        coeff_x, *_ = np.linalg.lstsq(design[mask], target_x[mask], rcond=None)
        coeff_y, *_ = np.linalg.lstsq(design[mask], target_y[mask], rcond=None)
        dx = target_x - design @ coeff_x
        dy = target_y - design @ coeff_y

        new_mask = mask.copy()
        for resid in (dx, dy):
            vals = resid[mask]
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            clip_scale = max(1.4826 * mad, 1e-6)
            new_mask &= np.abs(resid - med) < n_sigma * clip_scale

        if new_mask.sum() == mask.sum():
            break
        if new_mask.sum() < max(10, static.n_terms + 2):
            break
        mask = new_mask

    dx = target_x - design @ coeff_x
    dy = target_y - design @ coeff_y
    return coeff_x, coeff_y, mask
