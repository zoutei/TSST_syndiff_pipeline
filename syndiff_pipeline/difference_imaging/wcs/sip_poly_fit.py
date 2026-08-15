"""
SIP polynomial fitting utilities lifted from Calc_TESS_distortion.ipynb.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from syndiff_pipeline.difference_imaging.wcs.wcs_conversion import uvprime_to_radec

# Fixed detector scale for Sci2Idl monomial conditioning (not crop extent).
# tesswcs CCD is (rrows, rcolumns) = (2078, 2136); use max side length.
def _tesswcs_ffi_naxis() -> int:
  try:
    from tesswcs import rcolumns, rrows

    return int(max(rrows, rcolumns))
  except Exception:
    return 2136


TESS_FFI_NAXIS = _tesswcs_ffi_naxis()


def n_sci2idl_terms(poly_degree: int) -> int:
  return (poly_degree + 1) * (poly_degree + 2) // 2


def sci2idl_exponents(poly_degree: int) -> list[tuple[int, int]]:
  """Return (exponent_x, exponent_y) for each Sci2Idl coefficient index."""
  exps: list[tuple[int, int]] = []
  for i in range(poly_degree + 1):
    exp_x = i
    for j in range(i + 1):
      exps.append((exp_x, j))
      exp_x -= 1
  return exps


def poly_eval(coeff: Sequence[float], x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
  """Evaluate 2D polynomial with pysiaf coefficient ordering."""
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  out = np.zeros_like(x, dtype=float)
  idx = 0
  for i in range(order + 1):
    exp_x = i
    for _j in range(i + 1):
      out += coeff[idx] * (x ** exp_x) * (y ** (_j))
      exp_x -= 1
      idx += 1
  return out


def polyfit0(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    *,
    fit_coeffs0: bool = False,
    rotation_fit_x: bool = True,
    rotation_fit_y: bool = True,
    weight: np.ndarray | None = None,
    coord_scale: float | None = None,
) -> list[float]:
  """Fit u(x, y) as a 2D polynomial (notebook / pysiaf ordering).

  Coordinates are divided by ``coord_scale`` (default: tesswcs max CCD side,
  typically 2136) before the least-squares solve, then coefficients are mapped
  back to pixel units so ``poly_eval`` / SIP headers stay unchanged for callers.
  """
  u = np.asarray(u, dtype=float).ravel()
  x = np.asarray(x, dtype=float).ravel()
  y = np.asarray(y, dtype=float).ravel()

  px: list[int] = []
  py: list[int] = []

  if rotation_fit_x is False or rotation_fit_y is False:
    startindex = 2
  elif fit_coeffs0 is False:
    startindex = 1
  else:
    startindex = 0

  for i in range(startindex, order + 1):
    for j in range(i + 1):
      px.append(i - j)
      py.append(j)
  terms = len(px)
  if terms == 0:
    raise ValueError(f"polyfit0: order={order} leaves no free terms")

  if rotation_fit_x is False:
    u = u - x
  elif rotation_fit_y is False:
    u = u - y

  # Fixed detector scale (not star-sample extent) keeps monomials O(1).
  scale = float(TESS_FFI_NAXIS if coord_scale is None else coord_scale)
  if scale <= 0.0:
    raise ValueError(f"polyfit0: coord_scale must be > 0, got {scale}")
  xs = x / scale
  ys = y / scale
  design = np.column_stack([xs ** px[i] * ys ** py[i] for i in range(terms)])

  if weight is not None:
    w = np.sqrt(np.asarray(weight, dtype=float).ravel())
    design = design * w[:, None]
    u = u * w

  coeffs_scaled, *_ = np.linalg.lstsq(design, u, rcond=None)
  coeffs = [
    float(coeffs_scaled[i] / (scale ** (px[i] + py[i]))) for i in range(terms)
  ]

  if rotation_fit_x is False:
    coeffs0 = [0.0, 1.0, 0.0]
    coeffs0.extend(coeffs)
  elif rotation_fit_y is False:
    coeffs0 = [0.0, 0.0, 1.0]
    coeffs0.extend(coeffs)
  elif fit_coeffs0 is False:
    coeffs0 = [0.0]
    coeffs0.extend(coeffs)
  else:
    coeffs0 = list(coeffs)
  return coeffs0


def iterative_clip_du_dv(
    xprime: np.ndarray,
    yprime: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    poly_degree: int,
    *,
    n_sigma: float = 3.0,
    max_iter: int = 20,
    fit_coeffs0: bool = True,
    rotation_fit_x: bool = True,
    rotation_fit_y: bool = True,
    coord_scale: float | None = None,
) -> tuple[list[float], list[float], np.ndarray, np.ndarray, np.ndarray]:
  """
  Fit Sci2Idl polynomials with 3-sigma clipping on du/dv.

  Returns coeff_x, coeff_y, keep_mask, du, dv (final, all stars).
  """
  mask = np.ones(len(u), dtype=bool)
  coeff_x: list[float] = [0.0, 1.0, 0.0] + [0.0] * max(0, n_sci2idl_terms(poly_degree) - 3)
  coeff_y: list[float] = [0.0, 0.0, 1.0] + [0.0] * max(0, n_sci2idl_terms(poly_degree) - 3)
  scale = float(TESS_FFI_NAXIS if coord_scale is None else coord_scale)

  for _ in range(max_iter):
    coeff_x = polyfit0(
      u[mask],
      xprime[mask],
      yprime[mask],
      poly_degree,
      fit_coeffs0=fit_coeffs0,
      rotation_fit_x=rotation_fit_x,
      rotation_fit_y=True,
      coord_scale=scale,
    )
    coeff_y = polyfit0(
      v[mask],
      xprime[mask],
      yprime[mask],
      poly_degree,
      fit_coeffs0=True,
      rotation_fit_x=True,
      rotation_fit_y=rotation_fit_y,
      coord_scale=scale,
    )
    u_fit = poly_eval(coeff_x, xprime, yprime, poly_degree)
    v_fit = poly_eval(coeff_y, xprime, yprime, poly_degree)
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

  u_fit = poly_eval(coeff_x, xprime, yprime, poly_degree)
  v_fit = poly_eval(coeff_y, xprime, yprime, poly_degree)
  return coeff_x, coeff_y, mask, u - u_fit, v - v_fit


def fit_Idl2Sci(
    coeff_sci2idl_x: Sequence[float],
    coeff_sci2idl_y: Sequence[float],
    poly_degree: int,
    *,
    naxis1: int,
    naxis2: int,
    crpix1: float,
    crpix2: float,
) -> tuple[list[float], list[float]]:
  """Fit inverse Sci2Idl polynomials on a coarse grid."""
  nx = max(4, int(naxis1 / 16))
  ny = max(4, int(naxis2 / 16))
  x = np.linspace(1, naxis1, nx)
  y = np.linspace(1, naxis2, ny)
  xgprime, ygprime = np.meshgrid(x - (crpix1 - 1), y - (crpix2 - 1))

  xg_idl = poly_eval(coeff_sci2idl_x, xgprime, ygprime, poly_degree)
  yg_idl = poly_eval(coeff_sci2idl_y, xgprime, ygprime, poly_degree)
  scale = float(max(int(naxis1), int(naxis2), 1))
  coeff_idl2sci_x = polyfit0(
    xgprime.ravel(), xg_idl.ravel(), yg_idl.ravel(), poly_degree, coord_scale=scale
  )
  coeff_idl2sci_y = polyfit0(
    ygprime.ravel(), xg_idl.ravel(), yg_idl.ravel(), poly_degree, coord_scale=scale
  )
  return coeff_idl2sci_x, coeff_idl2sci_y


def coeff_table_rows(poly_degree: int) -> list[tuple[int, int, int]]:
  """Return (siaf_index, exponent_x, exponent_y) rows."""
  rows: list[tuple[int, int, int]] = []
  for i in range(poly_degree + 1):
    exp_x = i
    for j in range(i + 1):
      siaf_index = i * 10 + j
      rows.append((siaf_index, exp_x, j))
      exp_x -= 1
  return rows


def sci2idl_to_sip_updates(
    coeff_x: Sequence[float],
    coeff_y: Sequence[float],
    poly_degree: int,
) -> dict[str, float]:
  """Map Sci2Idl vectors to SIP A/B header updates (degree >= 2 only)."""
  updates: dict[str, float] = {}
  for idx, (exp_x, exp_y) in enumerate(sci2idl_exponents(poly_degree)):
    if exp_x + exp_y < 2:
      continue
    updates[f"A_{exp_x}_{exp_y}"] = float(coeff_x[idx])
    updates[f"B_{exp_x}_{exp_y}"] = float(coeff_y[idx])
  return updates


def fold_sci2idl_linear_into_header(
    hdr: fits.Header,
    coeff_x: Sequence[float],
    coeff_y: Sequence[float],
    poly_degree: int,
) -> tuple[fits.Header, list[float], list[float]]:
  """
  Fold free Sci2Idl constant + linear terms into CD/CRVAL; transform distortion by L^{-1}.

  Returns (updated_header, folded_coeff_x, folded_coeff_y) with identity linear Sci2Idl.
  """
  cx = list(coeff_x)
  cy = list(coeff_y)
  c0 = np.array([cx[0], cy[0]], dtype=float)
  L = np.array([[cx[1], cx[2]], [cy[1], cy[2]]], dtype=float)

  wcs = WCS(hdr)
  cd = np.array(wcs.pixel_scale_matrix, dtype=float)
  hdr["CD1_1"] = float(cd[0, 0])
  hdr["CD1_2"] = float(cd[0, 1])
  hdr["CD2_1"] = float(cd[1, 0])
  hdr["CD2_2"] = float(cd[1, 1])

  if np.linalg.norm(c0) > 0:
    ra_new, dec_new = uvprime_to_radec(
      np.array([c0[0]]), np.array([c0[1]]), hdr
    )
    hdr["CRVAL1"] = float(ra_new[0])
    hdr["CRVAL2"] = float(dec_new[0])

  cd_new = cd @ L
  hdr["CD1_1"] = float(cd_new[0, 0])
  hdr["CD1_2"] = float(cd_new[0, 1])
  hdr["CD2_1"] = float(cd_new[1, 0])
  hdr["CD2_2"] = float(cd_new[1, 1])
  for key in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2"):
    if key in hdr:
      del hdr[key]

  inv_l = np.linalg.inv(L)
  n = n_sci2idl_terms(poly_degree)
  fold_x = [0.0, 1.0, 0.0] + [0.0] * max(0, n - 3)
  fold_y = [0.0, 0.0, 1.0] + [0.0] * max(0, n - 3)
  for idx in range(3, n):
    raw = np.array([cx[idx], cy[idx]], dtype=float)
    mixed = inv_l @ raw
    fold_x[idx] = float(mixed[0])
    fold_y[idx] = float(mixed[1])

  return hdr, fold_x, fold_y


def build_sip_header_from_sci2idl(
    base_header: fits.Header,
    coeff_x: Sequence[float],
    coeff_y: Sequence[float],
    *,
    poly_degree: int,
    fit_inverse: bool = True,
    fold_linear: bool = False,
) -> fits.Header:
  """Write SIP distortion keys from Sci2Idl coefficients onto a copy of base_header."""
  hdr = base_header.copy()
  cx = list(coeff_x)
  cy = list(coeff_y)
  if fold_linear:
    hdr, cx, cy = fold_sci2idl_linear_into_header(hdr, cx, cy, poly_degree)

  for i in (1, 2):
    ctype = str(hdr.get(f"CTYPE{i}", ""))
    if ctype and "-SIP" not in ctype and ctype.endswith("TAN"):
      hdr[f"CTYPE{i}"] = ctype + "-SIP"
  naxis1 = int(hdr.get("NAXIS1", hdr.get("XMAX", 100) - hdr.get("XMIN", 0)))
  naxis2 = int(hdr.get("NAXIS2", hdr.get("YMAX", 100) - hdr.get("YMIN", 0)))
  if "XMAX" in hdr and "XMIN" in hdr:
    naxis1 = int(hdr["XMAX"] - hdr["XMIN"])
  if "YMAX" in hdr and "YMIN" in hdr:
    naxis2 = int(hdr["YMAX"] - hdr["YMIN"])

  crpix1 = float(hdr["CRPIX1"])
  crpix2 = float(hdr["CRPIX2"])

  for key in list(hdr.keys()):
    if (
      key.startswith("A_")
      or key.startswith("B_")
      or key.startswith("AP_")
      or key.startswith("BP_")
    ) and key not in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"):
      del hdr[key]

  max_order = poly_degree
  hdr["A_ORDER"] = max_order
  hdr["B_ORDER"] = max_order
  hdr["AP_ORDER"] = max_order
  hdr["BP_ORDER"] = max_order

  sip = sci2idl_to_sip_updates(cx, cy, poly_degree)
  for key, val in sip.items():
    hdr[key] = (val, "distortion coefficient")

  if fit_inverse:
    coeff_ap_x, coeff_ap_y = fit_Idl2Sci(
      cx,
      cy,
      poly_degree,
      naxis1=naxis1,
      naxis2=naxis2,
      crpix1=crpix1,
      crpix2=crpix2,
    )
    for idx, (exp_x, exp_y) in enumerate(sci2idl_exponents(poly_degree)):
      if exp_x + exp_y < 2:
        continue
      hdr[f"AP_{exp_x}_{exp_y}"] = (
        float(coeff_ap_x[idx]),
        "inv distortion coefficient",
      )
      hdr[f"BP_{exp_x}_{exp_y}"] = (
        float(coeff_ap_y[idx]),
        "inv distortion coefficient",
      )

  return hdr


def coeff_param_names(poly_degree: int) -> tuple[list[str], list[str]]:
  names_x = [f"c{i}_x" for i in range(n_sci2idl_terms(poly_degree))]
  names_y = [f"c{i}_y" for i in range(n_sci2idl_terms(poly_degree))]
  return names_x, names_y
