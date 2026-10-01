# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""
2D product Chebyshev distortion fits (orthogonal alternative to Sci2Idl monomials).

Coordinates are CRPIX-centered (x', y'), mapped to (ξ, η) ∈ [-1, 1] via crop
half-extents, then expanded as T_i(ξ) T_j(η) in Sci2Idl / pysiaf term order.

Identity-linear mode fits a *residual* Chebyshev series on (u-x', v-y') using
all degrees 0..N. That is required because T_n mixes lower powers, so dropping
deg 0–1 does not span the same functions as monomials with i+j>=2.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.polynomial.chebyshev import cheb2poly, chebvander

from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import n_sci2idl_terms, sci2idl_exponents


def crop_half_extents(
    *,
    naxis1: int | float,
    naxis2: int | float,
) -> tuple[float, float]:
  """Half-extents that map a centered crop roughly onto [-1, 1]."""
  sx = max(float(naxis1) / 2.0, 1.0)
  sy = max(float(naxis2) / 2.0, 1.0)
  return sx, sy


def normalize_prime(
    xprime: np.ndarray,
    yprime: np.ndarray,
    sx: float,
    sy: float,
) -> tuple[np.ndarray, np.ndarray]:
  if sx <= 0.0 or sy <= 0.0:
    raise ValueError(f"half-extents must be > 0, got sx={sx}, sy={sy}")
  return np.asarray(xprime, dtype=float) / sx, np.asarray(yprime, dtype=float) / sy


def cheb_design_matrix(
    xi: np.ndarray,
    eta: np.ndarray,
    poly_degree: int,
    *,
    min_total_degree: int = 0,
) -> np.ndarray:
  """Columns are T_{ex}(ξ) T_{ey}(η) in Sci2Idl exponent order (filtered)."""
  xi = np.asarray(xi, dtype=float).ravel()
  eta = np.asarray(eta, dtype=float).ravel()
  tx = chebvander(xi, poly_degree)
  ty = chebvander(eta, poly_degree)
  cols: list[np.ndarray] = []
  for exp_x, exp_y in sci2idl_exponents(poly_degree):
    if exp_x + exp_y < min_total_degree:
      continue
    cols.append(tx[:, exp_x] * ty[:, exp_y])
  if not cols:
    raise ValueError(
      f"cheb_design_matrix: degree={poly_degree}, min_total_degree={min_total_degree} "
      "leaves no free terms"
    )
  return np.column_stack(cols)


def monomial_design_matrix(
    xprime: np.ndarray,
    yprime: np.ndarray,
    poly_degree: int,
    *,
    min_total_degree: int = 0,
    coord_scale: float,
) -> np.ndarray:
  """Scaled monomial columns x^i y^j / scale^{i+j} (Sci2Idl order, filtered)."""
  if coord_scale <= 0.0:
    raise ValueError(f"coord_scale must be > 0, got {coord_scale}")
  xs = np.asarray(xprime, dtype=float).ravel() / coord_scale
  ys = np.asarray(yprime, dtype=float).ravel() / coord_scale
  cols: list[np.ndarray] = []
  for exp_x, exp_y in sci2idl_exponents(poly_degree):
    if exp_x + exp_y < min_total_degree:
      continue
    cols.append((xs ** exp_x) * (ys ** exp_y))
  if not cols:
    raise ValueError(
      f"monomial_design_matrix: degree={poly_degree}, min_total_degree={min_total_degree} "
      "leaves no free terms"
    )
  return np.column_stack(cols)


def design_condition_number(design: np.ndarray) -> float:
  """2-norm condition number of a design matrix (inf if rank-deficient)."""
  s = np.linalg.svd(np.asarray(design, dtype=float), compute_uv=False)
  smax = float(s[0]) if len(s) else 0.0
  smin = float(s[-1]) if len(s) else 0.0
  if smin <= 0.0 or not np.isfinite(smin):
    return float("inf")
  return smax / smin


def cheb_eval(
    coeff: Sequence[float],
    xprime: np.ndarray,
    yprime: np.ndarray,
    poly_degree: int,
    *,
    sx: float,
    sy: float,
) -> np.ndarray:
  """Evaluate a Sci2Idl-length Chebyshev coefficient vector on (x', y')."""
  xi, eta = normalize_prime(xprime, yprime, sx, sy)
  tx = chebvander(np.asarray(xi, dtype=float), poly_degree)
  ty = chebvander(np.asarray(eta, dtype=float), poly_degree)
  out = np.zeros_like(xi, dtype=float)
  for idx, (exp_x, exp_y) in enumerate(sci2idl_exponents(poly_degree)):
    out += float(coeff[idx]) * tx[:, exp_x] * ty[:, exp_y]
  return out


def chebfit0(
    target: np.ndarray,
    xprime: np.ndarray,
    yprime: np.ndarray,
    order: int,
    *,
    sx: float,
    sy: float,
    weight: np.ndarray | None = None,
) -> list[float]:
  """Least-squares full Chebyshev series (Sci2Idl order, degrees 0..order)."""
  target = np.asarray(target, dtype=float).ravel()
  x = np.asarray(xprime, dtype=float).ravel()
  y = np.asarray(yprime, dtype=float).ravel()

  xi, eta = normalize_prime(x, y, sx, sy)
  design = cheb_design_matrix(xi, eta, order, min_total_degree=0)

  rhs = target
  if weight is not None:
    w = np.sqrt(np.asarray(weight, dtype=float).ravel())
    design = design * w[:, None]
    rhs = target * w

  free, *_ = np.linalg.lstsq(design, rhs, rcond=None)
  return [float(v) for v in free]


def chebfit_axis(
    observed: np.ndarray,
    xprime: np.ndarray,
    yprime: np.ndarray,
    order: int,
    *,
    sx: float,
    sy: float,
    axis: str,
    identity_linear: bool = True,
    weight: np.ndarray | None = None,
) -> list[float]:
  """Fit one axis.

  ``identity_linear=True`` (default): store *residual* Chebyshev coeffs for
  ``observed - (x'|y')``. Caller must add the identity linear term on eval.

  ``identity_linear=False``: store a full Chebyshev model for ``observed``.
  """
  obs = np.asarray(observed, dtype=float).ravel()
  x = np.asarray(xprime, dtype=float).ravel()
  y = np.asarray(yprime, dtype=float).ravel()
  if identity_linear:
    if axis == "x":
      target = obs - x
    elif axis == "y":
      target = obs - y
    else:
      raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return chebfit0(target, x, y, order, sx=sx, sy=sy, weight=weight)
  return chebfit0(obs, x, y, order, sx=sx, sy=sy, weight=weight)


def cheb_model_eval(
    coeff: Sequence[float],
    xprime: np.ndarray,
    yprime: np.ndarray,
    poly_degree: int,
    *,
    sx: float,
    sy: float,
    identity_linear: bool,
    axis: str,
) -> np.ndarray:
  """Evaluate stored coeffs (residual series if identity_linear)."""
  series = cheb_eval(coeff, xprime, yprime, poly_degree, sx=sx, sy=sy)
  if not identity_linear:
    return series
  if axis == "x":
    return np.asarray(xprime, dtype=float) + series
  if axis == "y":
    return np.asarray(yprime, dtype=float) + series
  raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def iterative_clip_cheb_du_dv(
    xprime: np.ndarray,
    yprime: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    poly_degree: int,
    *,
    sx: float,
    sy: float,
    n_sigma: float = 3.0,
    max_iter: int = 20,
    identity_linear: bool = True,
) -> tuple[list[float], list[float], np.ndarray, np.ndarray, np.ndarray]:
  """Fit Chebyshev maps with 3-sigma MAD clipping on du/dv."""
  mask = np.ones(len(u), dtype=bool)
  n = n_sci2idl_terms(poly_degree)
  coeff_x: list[float] = [0.0] * n
  coeff_y: list[float] = [0.0] * n

  for _ in range(max_iter):
    coeff_x = chebfit_axis(
      u[mask],
      xprime[mask],
      yprime[mask],
      poly_degree,
      sx=sx,
      sy=sy,
      axis="x",
      identity_linear=identity_linear,
    )
    coeff_y = chebfit_axis(
      v[mask],
      xprime[mask],
      yprime[mask],
      poly_degree,
      sx=sx,
      sy=sy,
      axis="y",
      identity_linear=identity_linear,
    )
    u_fit = cheb_model_eval(
      coeff_x, xprime, yprime, poly_degree, sx=sx, sy=sy,
      identity_linear=identity_linear, axis="x",
    )
    v_fit = cheb_model_eval(
      coeff_y, xprime, yprime, poly_degree, sx=sx, sy=sy,
      identity_linear=identity_linear, axis="y",
    )
    du = u - u_fit
    dv = v - v_fit

    new_mask = mask.copy()
    for resid in (du, dv):
      vals = resid[mask]
      med = float(np.median(vals))
      mad = float(np.median(np.abs(vals - med)))
      clip_scale = max(1.4826 * mad, 1e-6)
      new_mask &= np.abs(resid) < med + n_sigma * clip_scale

    if new_mask.sum() == mask.sum():
      break
    if new_mask.sum() < 10:
      break
    mask = new_mask

  u_fit = cheb_model_eval(
    coeff_x, xprime, yprime, poly_degree, sx=sx, sy=sy,
    identity_linear=identity_linear, axis="x",
  )
  v_fit = cheb_model_eval(
    coeff_y, xprime, yprime, poly_degree, sx=sx, sy=sy,
    identity_linear=identity_linear, axis="y",
  )
  return coeff_x, coeff_y, mask, u - u_fit, v - v_fit


def chebyshev_to_monomial_coeffs(
    cheb_coeff: Sequence[float],
    poly_degree: int,
    *,
    sx: float,
    sy: float,
) -> list[float]:
  """Convert Chebyshev Sci2Idl-ordered coeffs to pixel monomial Sci2Idl coeffs.

  Model: sum a_{ij} T_i(x'/sx) T_j(y'/sy) = sum c_{pq} (x')^p (y')^q.
  """
  n = n_sci2idl_terms(poly_degree)
  if len(cheb_coeff) != n:
    raise ValueError(f"expected {n} coeffs, got {len(cheb_coeff)}")

  mono_xi = np.zeros((poly_degree + 1, poly_degree + 1), dtype=float)
  for a, (i, j) in zip(cheb_coeff, sci2idl_exponents(poly_degree)):
    if a == 0.0:
      continue
    cx = np.asarray(cheb2poly([0.0] * i + [1.0]), dtype=float)
    cy = np.asarray(cheb2poly([0.0] * j + [1.0]), dtype=float)
    for p, cp in enumerate(cx):
      if cp == 0.0:
        continue
      for q, cq in enumerate(cy):
        if cq == 0.0:
          continue
        mono_xi[p, q] += float(a) * float(cp) * float(cq)

  out = [0.0] * n
  for idx, (p, q) in enumerate(sci2idl_exponents(poly_degree)):
    out[idx] = float(mono_xi[p, q] / (sx ** p * sy ** q))
  return out


def residual_chebyshev_to_sci2idl_monomials(
    cheb_coeff: Sequence[float],
    poly_degree: int,
    *,
    sx: float,
    sy: float,
    axis: str,
) -> list[float]:
  """Convert identity-linear residual Chebyshev coeffs to full Sci2Idl monomials."""
  mono = chebyshev_to_monomial_coeffs(cheb_coeff, poly_degree, sx=sx, sy=sy)
  if axis == "x":
    mono[1] += 1.0
  elif axis == "y":
    mono[2] += 1.0
  else:
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
  return mono


def compare_design_conditions(
    xprime: np.ndarray,
    yprime: np.ndarray,
    poly_degree: int,
    *,
    sx: float,
    sy: float,
    monomial_coord_scale: float,
    min_total_degree: int = 0,
) -> dict[str, float]:
  """Condition numbers for Chebyshev vs scaled-monomial designs on the same stars."""
  xi, eta = normalize_prime(xprime, yprime, sx, sy)
  cheb = cheb_design_matrix(xi, eta, poly_degree, min_total_degree=min_total_degree)
  mono = monomial_design_matrix(
    xprime,
    yprime,
    poly_degree,
    min_total_degree=min_total_degree,
    coord_scale=monomial_coord_scale,
  )
  return {
    "cheb_cond": design_condition_number(cheb),
    "mono_cond": design_condition_number(mono),
    "n_stars": float(len(np.asarray(xprime).ravel())),
    "n_terms": float(cheb.shape[1]),
    "sx": float(sx),
    "sy": float(sy),
    "monomial_coord_scale": float(monomial_coord_scale),
  }
