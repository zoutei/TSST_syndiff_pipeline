# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Temporal Sci2Idl WCS model (shared linear baseline + affine + distortion)."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

_DEV_WCS = Path(__file__).resolve().parents[1] / "wcs_fit_from_centroids"
if str(_DEV_WCS) not in sys.path:
  pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.fit_wcs_from_centroids import (  # noqa: E402
  FitConfig,
  Sci2IdlFitResult,
  fit_sci2idl_distortion,
)
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import (  # noqa: E402
  TESS_FFI_NAXIS,
  build_sip_header_from_sci2idl,
  coeff_param_names,
  n_sci2idl_terms,
  poly_eval,
  sci2idl_exponents,
)

TemporalBasisKind = Literal["poly", "spline"]


def identity_coeffs(poly_degree: int) -> tuple[list[float], list[float]]:
  n = n_sci2idl_terms(poly_degree)
  cx = [0.0] * n
  cy = [0.0] * n
  cx[1] = 1.0
  cy[2] = 1.0
  return cx, cy


def term_poly_orders(
    n_terms: int,
    *,
    order_c0: int,
    order_c12: int,
    order_dist: int,
) -> np.ndarray:
  """Per Sci2Idl index temporal polynomial order (c0, c1/c2, c3+ distortion)."""
  orders = np.full(n_terms, order_dist, dtype=int)
  if n_terms > 0:
    orders[0] = order_c0
  for i in (1, 2):
    if i < n_terms:
      orders[i] = order_c12
  return orders


def cap_term_poly_orders(orders: np.ndarray, n_frames: int) -> np.ndarray:
  cap = max(0, n_frames - 1)
  return np.minimum(orders, cap)


def make_edge_weighted_knot_vector(
    tau_min: float = 0.0,
    tau_max: float = 1.0,
    n_interior: int = 10,
    degree: int = 3,
    edge_frac: float = 0.12,
) -> np.ndarray:
  """Non-uniform knot vector with dense interior knots near orbit endpoints."""
  n_interior = max(1, int(n_interior))
  edge_frac = float(np.clip(edge_frac, 0.02, 0.45))
  n_start = max(1, n_interior // 3)
  n_end = max(1, n_interior // 3)
  n_mid = max(1, n_interior - n_start - n_end)
  start_knots = np.linspace(tau_min, tau_min + edge_frac, n_start + 2)[1:-1]
  mid_knots = np.linspace(tau_min + edge_frac, tau_max - edge_frac, n_mid + 2)[1:-1]
  end_knots = np.linspace(tau_max - edge_frac, tau_max, n_end + 2)[1:-1]
  interior = np.sort(np.unique(np.concatenate([start_knots, mid_knots, end_knots])))
  return np.concatenate([
    np.full(degree + 1, tau_min),
    interior,
    np.full(degree + 1, tau_max),
  ])


def interior_knots(knot_vector: np.ndarray, degree: int) -> np.ndarray:
  return knot_vector[degree + 1 : -(degree + 1)]


def n_spline_coeffs(knot_vector: np.ndarray, degree: int) -> int:
  return len(knot_vector) - degree - 1


def cap_spline_interior_knots(n_interior: int, n_frames: int, degree: int) -> int:
  """Keep the temporal basis identifiable: need n_frames > n_spline_coeffs."""
  max_n_coeffs = max(2, n_frames - degree)
  max_interior = max(0, max_n_coeffs - degree - 1)
  return min(n_interior, max_interior)


@dataclass
class TemporalFitState:
  linear_wcs: WCS
  sip_degree: int
  btjd_ref: float
  btjd_scale: float
  coeff_matrix: np.ndarray
  basis_kind: TemporalBasisKind = "spline"
  term_poly_order: np.ndarray | None = None  # poly mode only
  poly_order_max: int = 0  # poly mode only
  knot_vector: np.ndarray | None = None  # spline mode only
  spline_degree: int = 3

  @property
  def n_terms(self) -> int:
    return n_sci2idl_terms(self.sip_degree)

  @property
  def is_spline(self) -> bool:
    return self.basis_kind == "spline"

  @property
  def n_basis(self) -> int:
    if self.is_spline:
      assert self.knot_vector is not None
      return n_spline_coeffs(self.knot_vector, self.spline_degree)
    return self.poly_order_max + 1

  @property
  def poly_order(self) -> int:
    """Legacy: max temporal order across terms (poly mode)."""
    return int(self.poly_order_max)

  @classmethod
  def empty(
      cls,
      linear_wcs: WCS,
      *,
      sip_degree: int,
      btjd_ref: float,
      btjd_scale: float = 1.0,
      basis_kind: TemporalBasisKind = "poly",
      term_poly_order: np.ndarray | None = None,
      knot_vector: np.ndarray | None = None,
      spline_degree: int = 3,
      n_interior_knots: int = 10,
      spline_edge_frac: float = 0.12,
  ) -> TemporalFitState:
    if basis_kind == "spline":
      return cls.empty_spline(
        linear_wcs,
        sip_degree=sip_degree,
        btjd_ref=btjd_ref,
        btjd_scale=btjd_scale,
        knot_vector=knot_vector,
        spline_degree=spline_degree,
        n_interior_knots=n_interior_knots,
        spline_edge_frac=spline_edge_frac,
      )
    if term_poly_order is None:
      raise ValueError("term_poly_order is required for poly basis")
    return cls.empty_poly(
      linear_wcs,
      sip_degree=sip_degree,
      term_poly_order=term_poly_order,
      btjd_ref=btjd_ref,
      btjd_scale=btjd_scale,
    )

  @classmethod
  def empty_poly(
      cls,
      linear_wcs: WCS,
      *,
      sip_degree: int,
      term_poly_order: np.ndarray,
      btjd_ref: float,
      btjd_scale: float = 1.0,
  ) -> TemporalFitState:
    n = n_sci2idl_terms(sip_degree)
    orders = np.asarray(term_poly_order, dtype=int).reshape(n)
    poly_order_max = int(orders.max()) if len(orders) else 0
    mat = np.zeros((2 * n, poly_order_max + 1), dtype=float)
    cx, cy = identity_coeffs(sip_degree)
    for i, val in enumerate(cx):
      mat[i, 0] = val
    for i, val in enumerate(cy):
      mat[n + i, 0] = val
    return cls(
      linear_wcs=linear_wcs,
      sip_degree=sip_degree,
      btjd_ref=btjd_ref,
      btjd_scale=btjd_scale,
      coeff_matrix=mat,
      basis_kind="poly",
      term_poly_order=orders,
      poly_order_max=poly_order_max,
    )

  @classmethod
  def empty_spline(
      cls,
      linear_wcs: WCS,
      *,
      sip_degree: int,
      btjd_ref: float,
      btjd_scale: float = 1.0,
      knot_vector: np.ndarray | None = None,
      spline_degree: int = 3,
      n_interior_knots: int = 10,
      spline_edge_frac: float = 0.12,
  ) -> TemporalFitState:
    n = n_sci2idl_terms(sip_degree)
    knots = (
      np.asarray(knot_vector, dtype=float)
      if knot_vector is not None
      else make_edge_weighted_knot_vector(
        n_interior=n_interior_knots,
        degree=spline_degree,
        edge_frac=spline_edge_frac,
      )
    )
    n_basis = n_spline_coeffs(knots, spline_degree)
    mat = np.zeros((2 * n, n_basis), dtype=float)
    cx, cy = identity_coeffs(sip_degree)
    for i, val in enumerate(cx):
      mat[i, 0] = val
    for i, val in enumerate(cy):
      mat[n + i, 0] = val
    return cls(
      linear_wcs=linear_wcs,
      sip_degree=sip_degree,
      btjd_ref=btjd_ref,
      btjd_scale=btjd_scale,
      coeff_matrix=mat,
      basis_kind="spline",
      knot_vector=knots,
      spline_degree=int(spline_degree),
    )

  @classmethod
  def from_npz(
      cls,
      linear_wcs: WCS,
      npz: np.lib.npyio.NpzFile,
  ) -> TemporalFitState:
    sip_degree = int(npz["sip_degree"])
    n = n_sci2idl_terms(sip_degree)
    basis_kind: TemporalBasisKind
    if "temporal_basis_kind" in npz:
      basis_kind = str(npz["temporal_basis_kind"])
    elif "knot_vector" in npz:
      basis_kind = "spline"
    else:
      basis_kind = "poly"

    if basis_kind == "spline":
      knot_vector = np.asarray(npz["knot_vector"], dtype=float)
      spline_degree = int(npz["spline_degree"]) if "spline_degree" in npz else 3
      return cls(
        linear_wcs=linear_wcs,
        sip_degree=sip_degree,
        btjd_ref=float(npz["btjd_ref"]),
        btjd_scale=float(npz["btjd_scale"]),
        coeff_matrix=np.asarray(npz["coeff_matrix"], dtype=float),
        basis_kind="spline",
        knot_vector=knot_vector,
        spline_degree=spline_degree,
      )

    if "term_poly_order" in npz:
      orders = np.asarray(npz["term_poly_order"], dtype=int).reshape(n)
      poly_order_max = int(npz["poly_order_max"])
    else:
      poly_order_max = int(npz["poly_order"])
      orders = np.full(n, poly_order_max, dtype=int)
    return cls(
      linear_wcs=linear_wcs,
      sip_degree=sip_degree,
      btjd_ref=float(npz["btjd_ref"]),
      btjd_scale=float(npz["btjd_scale"]),
      coeff_matrix=np.asarray(npz["coeff_matrix"], dtype=float),
      basis_kind="poly",
      term_poly_order=orders,
      poly_order_max=poly_order_max,
    )

  def tau(self, btjd: float) -> float:
    t = (btjd - self.btjd_ref) / self.btjd_scale
    if self.is_spline:
      return float(np.clip(t, 0.0, 1.0))
    return t

  def _tau_array(self, btjd: np.ndarray) -> np.ndarray:
    tau = (np.asarray(btjd, dtype=float) - self.btjd_ref) / self.btjd_scale
    if self.is_spline:
      return np.clip(tau, 0.0, 1.0)
    return tau

  def temporal_basis(self, btjd: float) -> np.ndarray:
    tau = self.tau(btjd)
    if self.is_spline:
      assert self.knot_vector is not None
      return BSpline.design_matrix(
        np.asarray([tau], dtype=float),
        self.knot_vector,
        self.spline_degree,
      ).toarray()[0]
    return np.array([tau ** p for p in range(self.poly_order_max + 1)], dtype=float)

  def tau_powers(self, btjd: float) -> np.ndarray:
    """Legacy alias for :meth:`temporal_basis` in poly mode."""
    return self.temporal_basis(btjd)

  def _term_value(self, row: int, basis: np.ndarray) -> float:
    if self.is_spline:
      return float(self.coeff_matrix[row, :] @ basis)
    term_i = row if row < self.n_terms else row - self.n_terms
    p = int(self.term_poly_order[term_i])  # type: ignore[index]
    return float(self.coeff_matrix[row, : p + 1] @ basis[: p + 1])

  def coeff_vectors_at_btjd(self, btjd: float) -> tuple[list[float], list[float]]:
    n = self.n_terms
    basis = self.temporal_basis(btjd)
    cx = [self._term_value(i, basis) for i in range(n)]
    cy = [self._term_value(n + i, basis) for i in range(n)]
    return cx, cy

  def predict_uv(
      self,
      xprime: np.ndarray,
      yprime: np.ndarray,
      btjd: float,
  ) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = self.coeff_vectors_at_btjd(btjd)
    u_fit = poly_eval(cx, xprime, yprime, self.sip_degree)
    v_fit = poly_eval(cy, xprime, yprime, self.sip_degree)
    return u_fit, v_fit

  def build_header(self, btjd: float):
    cx, cy = self.coeff_vectors_at_btjd(btjd)
    return build_sip_header_from_sci2idl(
      self.linear_wcs.to_header(relax=True),
      cx,
      cy,
      poly_degree=self.sip_degree,
      fold_linear=True,
    )

  def pack_params(self) -> np.ndarray:
    if self.is_spline:
      return self.coeff_matrix.ravel()
    parts: list[np.ndarray] = []
    n = self.n_terms
    for row in range(2 * n):
      term_i = row if row < n else row - n
      p = int(self.term_poly_order[term_i])  # type: ignore[index]
      parts.append(self.coeff_matrix[row, : p + 1])
    return np.concatenate(parts)

  def unpack_params(self, params: np.ndarray) -> None:
    n = self.n_terms
    params = np.asarray(params, dtype=float)
    if self.is_spline:
      expected = 2 * n * self.n_basis
      if len(params) != expected:
        raise ValueError(f"unpack_params: expected {expected} params, got {len(params)}")
      self.coeff_matrix = params.reshape(2 * n, self.n_basis)
      return

    mat = np.zeros((2 * n, self.poly_order_max + 1), dtype=float)
    offset = 0
    for row in range(2 * n):
      term_i = row if row < n else row - n
      p = int(self.term_poly_order[term_i])  # type: ignore[index]
      npar = p + 1
      mat[row, :npar] = params[offset : offset + npar]
      offset += npar
    if offset != len(params):
      raise ValueError(f"unpack_params: expected {offset} params, got {len(params)}")
    self.coeff_matrix = mat

  def param_names(self) -> list[str]:
    names_x, names_y = coeff_param_names(self.sip_degree)
    out: list[str] = []
    n = self.n_terms
    if self.is_spline:
      for row, base in enumerate(names_x + names_y):
        for b in range(self.n_basis):
          out.append(f"{base}_B{b}")
      return out
    for row, base in enumerate(names_x + names_y):
      term_i = row if row < n else row - n
      for p in range(int(self.term_poly_order[term_i]) + 1):  # type: ignore[index]
        out.append(f"{base}_p{p}")
    return out

  def evaluate_poly_curve(self, btjd_grid: np.ndarray) -> pd.DataFrame:
    names_x, names_y = coeff_param_names(self.sip_degree)
    rows: dict[str, Any] = {"btjd": btjd_grid}
    n = self.n_terms
    for i, name in enumerate(names_x):
      vals = []
      for btjd in btjd_grid:
        cx, _cy = self.coeff_vectors_at_btjd(float(btjd))
        vals.append(cx[i])
      rows[name] = vals
    for i, name in enumerate(names_y):
      vals = []
      for btjd in btjd_grid:
        _cx, cy = self.coeff_vectors_at_btjd(float(btjd))
        vals.append(cy[i])
      rows[name] = vals
    return pd.DataFrame(rows)


def warmstart_frame(
    stars: pd.DataFrame,
    linear_wcs: WCS,
    *,
    sip_degree: int,
    star_cfg: Any | None = None,
) -> Sci2IdlFitResult:
  """Per-FFI Sci2Idl fit matching single_ffi_wcs_fit.ipynb cell 7."""
  from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.fit_wcs_from_centroids import StarSelectionConfig

  cfg = star_cfg or StarSelectionConfig()
  return fit_sci2idl_distortion(
    stars,
    linear_wcs,
    FitConfig(sip_degree=sip_degree, sip_fallback=()),
    fit_coeffs0=True,
    rotation_fit_x=True,
    rotation_fit_y=True,
    n_sigma=cfg.clip_n_sigma,
    max_iter=cfg.clip_max_iter,
  )


def warmstart_table_row(
    stem: str,
    btjd: float,
    result: Sci2IdlFitResult,
) -> dict[str, Any]:
  row: dict[str, Any] = {
    "stem": stem,
    "btjd": btjd,
    "n_keep": int(result.keep_mask.sum()),
  }
  for i, val in enumerate(result.coeff_x):
    row[f"c{i}_x"] = val
  for i, val in enumerate(result.coeff_y):
    row[f"c{i}_y"] = val
  return row


def _fit_spline_coeffs(
    tau: np.ndarray,
    y: np.ndarray,
    knot_vector: np.ndarray,
    spline_degree: int,
) -> np.ndarray:
  tau = np.asarray(tau, dtype=float)
  y = np.asarray(y, dtype=float)
  design = BSpline.design_matrix(tau, knot_vector, spline_degree).toarray()
  coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
  return np.asarray(coeffs, dtype=float)


def init_from_warmstart(
    state: TemporalFitState,
    warmstart_df: pd.DataFrame,
) -> None:
  """Fit each Sci2Idl coeff vs tau using per-frame warm-start values."""
  n = state.n_terms
  btjd = warmstart_df["btjd"].to_numpy(dtype=float)
  tau = state._tau_array(btjd)

  if state.is_spline:
    assert state.knot_vector is not None
    for axis_offset, suffix in ((0, "_x"), (n, "_y")):
      for i in range(n):
        col = f"c{i}{suffix}"
        y = warmstart_df[col].to_numpy(dtype=float)
        row = axis_offset + i
        state.coeff_matrix[row, :] = _fit_spline_coeffs(
          tau, y, state.knot_vector, state.spline_degree
        )
    return

  for axis_offset, suffix in ((0, "_x"), (n, "_y")):
    for i in range(n):
      col = f"c{i}{suffix}"
      y = warmstart_df[col].to_numpy(dtype=float)
      p = int(state.term_poly_order[i])  # type: ignore[index]
      row = axis_offset + i
      if p == 0:
        state.coeff_matrix[row, 0] = float(np.mean(y))
      else:
        deg = min(p, len(tau) - 1)
        coef = np.polyfit(tau, y, deg=deg)
        for k in range(deg + 1):
          state.coeff_matrix[row, k] = coef[deg - k]


def joint_residuals(
    params: np.ndarray,
    state: TemporalFitState,
    stems: np.ndarray,
    btjd: np.ndarray,
    xprime: np.ndarray,
    yprime: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
  state.unpack_params(params)
  u_fit, v_fit = np.empty_like(u), np.empty_like(v)
  for stem in np.unique(stems):
    m = stems == stem
    uf, vf = state.predict_uv(xprime[m], yprime[m], float(btjd[m][0]))
    u_fit[m] = uf
    v_fit[m] = vf
  return np.concatenate([u - u_fit, v - v_fit])


@dataclass(frozen=True)
class _JointLinearLayout:
  n_params: int
  n_terms: int
  n_basis: int
  u_terms: np.ndarray  # (n_u, 3): param_idx, term_k, basis_idx
  v_terms: np.ndarray  # (n_v, 3)


def _joint_linear_layout(state: TemporalFitState) -> _JointLinearLayout:
  n = state.n_terms
  u_parts: list[tuple[int, int, int]] = []
  v_parts: list[tuple[int, int, int]] = []
  offset = 0
  if state.is_spline:
    n_basis = state.n_basis
    for row in range(2 * n):
      term_i = row if row < n else row - n
      for basis_idx in range(n_basis):
        if row < n:
          u_parts.append((offset, term_i, basis_idx))
        else:
          v_parts.append((offset, term_i, basis_idx))
        offset += 1
  else:
    for row in range(2 * n):
      term_i = row if row < n else row - n
      pmax = int(state.term_poly_order[term_i])  # type: ignore[index]
      for basis_idx in range(pmax + 1):
        if row < n:
          u_parts.append((offset, term_i, basis_idx))
        else:
          v_parts.append((offset, term_i, basis_idx))
        offset += 1
  return _JointLinearLayout(
    n_params=offset,
    n_terms=n,
    n_basis=state.n_basis,
    u_terms=np.asarray(u_parts, dtype=int),
    v_terms=np.asarray(v_parts, dtype=int),
  )


def sci2idl_monomial_matrix(
    xprime: np.ndarray,
    yprime: np.ndarray,
    sip_degree: int,
    *,
    coord_scale: float | None = None,
) -> np.ndarray:
  """Shape (n_stars, n_terms) scaled monomial values M_k(x/S, y/S).

  ``coord_scale`` defaults to tesswcs max CCD side (``TESS_FFI_NAXIS``).
  Callers that solve in this basis must convert coeffs back to pixel units
  via ``params_scaled_to_pixel`` before ``poly_eval``.
  """
  scale = float(TESS_FFI_NAXIS if coord_scale is None else coord_scale)
  if scale <= 0.0:
    raise ValueError(f"coord_scale must be > 0, got {scale}")
  x = np.asarray(xprime, dtype=float) / scale
  y = np.asarray(yprime, dtype=float) / scale
  n_terms = n_sci2idl_terms(sip_degree)
  out = np.empty((len(x), n_terms), dtype=float)
  for k, (ex, ey) in enumerate(sci2idl_exponents(sip_degree)):
    out[:, k] = (x ** ex) * (y ** ey)
  return out


def params_scaled_to_pixel(
    params: np.ndarray,
    layout: _JointLinearLayout,
    sip_degree: int,
    *,
    coord_scale: float | None = None,
) -> np.ndarray:
  """Convert LS params from scaled-monomial basis to pixel-unit Sci2Idl coeffs."""
  scale = float(TESS_FFI_NAXIS if coord_scale is None else coord_scale)
  term_scale = np.array(
    [scale ** (ex + ey) for ex, ey in sci2idl_exponents(sip_degree)],
    dtype=float,
  )
  out = np.asarray(params, dtype=float).copy()
  for pi, k, _bi in layout.u_terms:
    out[int(pi)] = float(params[int(pi)]) / term_scale[int(k)]
  for pi, k, _bi in layout.v_terms:
    out[int(pi)] = float(params[int(pi)]) / term_scale[int(k)]
  return out


def frame_id_map(stems: Sequence[str]) -> dict[str, int]:
  """Stable integer frame id per stem (sorted by stem for reproducibility)."""
  unique = sorted(set(stems))
  return {stem: i for i, stem in enumerate(unique)}


def _temporal_basis_for_frames(
    frame_btjd: np.ndarray,
    *,
    btjd_ref: float,
    btjd_scale: float,
    state: TemporalFitState,
) -> np.ndarray:
  tau = state._tau_array(np.asarray(frame_btjd, dtype=float))
  if state.is_spline:
    assert state.knot_vector is not None
    return BSpline.design_matrix(tau, state.knot_vector, state.spline_degree).toarray()
  pmax = state.poly_order_max + 1
  out = np.empty((len(tau), pmax), dtype=float)
  out[:, 0] = 1.0
  for p in range(1, pmax):
    out[:, p] = out[:, p - 1] * tau
  return out


def _tau_powers_for_frames(
    frame_btjd: np.ndarray,
    *,
    btjd_ref: float,
    btjd_scale: float,
    poly_order_max: int,
) -> np.ndarray:
  """Legacy poly basis builder."""
  tau = (frame_btjd - btjd_ref) / btjd_scale
  pmax = poly_order_max + 1
  out = np.empty((len(frame_btjd), pmax), dtype=float)
  out[:, 0] = 1.0
  for p in range(1, pmax):
    out[:, p] = out[:, p - 1] * tau
  return out


def _accumulate_joint_chunk(
    frame_id: np.ndarray,
    monomials: np.ndarray,
    frame_basis: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    u_terms: np.ndarray,
    v_terms: np.ndarray,
    n_params: int,
) -> tuple[np.ndarray, np.ndarray]:
  ata = np.zeros((n_params, n_params), dtype=float)
  atb = np.zeros(n_params, dtype=float)
  a = np.zeros(n_params, dtype=float)
  for i in range(len(frame_id)):
    fid = int(frame_id[i])
    basis = frame_basis[fid]
    mrow = monomials[i]
    a.fill(0.0)
    for pi, k, basis_idx in u_terms:
      a[pi] = basis[basis_idx] * mrow[k]
    ata += np.outer(a, a)
    atb += a * u[i]
    a.fill(0.0)
    for pi, k, basis_idx in v_terms:
      a[pi] = basis[basis_idx] * mrow[k]
    ata += np.outer(a, a)
    atb += a * v[i]
  return ata, atb


def refine_joint_linear(
    state: TemporalFitState,
    stacked: pd.DataFrame,
    *,
    jobs: int = 0,
    coord_scale: float | None = None,
) -> TemporalFitState:
  """Solve Stage B via normal equations (same linear LS as refine_joint).

  Monomials are scaled by ``coord_scale`` (default ``TESS_FFI_NAXIS``); solved
  parameters are converted back to pixel-unit Sci2Idl coeffs before unpack.
  """
  scale = float(TESS_FFI_NAXIS if coord_scale is None else coord_scale)
  layout = _joint_linear_layout(state)
  stems = stacked["stem"].to_numpy()
  stem_to_fid = frame_id_map(stems.tolist())
  frame_id = np.asarray([stem_to_fid[s] for s in stems], dtype=int)

  frame_btjd = np.array(
    [
      float(stacked.loc[stacked["stem"] == s, "btjd"].iloc[0])
      for s in sorted(stem_to_fid, key=stem_to_fid.get)
    ],
    dtype=float,
  )
  frame_basis = _temporal_basis_for_frames(
    frame_btjd,
    btjd_ref=state.btjd_ref,
    btjd_scale=state.btjd_scale,
    state=state,
  )

  xprime = stacked["xprime"].to_numpy(dtype=float)
  yprime = stacked["yprime"].to_numpy(dtype=float)
  u = stacked["u"].to_numpy(dtype=float)
  v = stacked["v"].to_numpy(dtype=float)
  monomials = sci2idl_monomial_matrix(
    xprime, yprime, state.sip_degree, coord_scale=scale
  )

  n_stars = len(stacked)
  n_workers = jobs if jobs > 0 else min(os.cpu_count() or 4, max(1, n_stars // 500))
  if n_workers <= 1 or n_stars < 2000:
    ata, atb = _accumulate_joint_chunk(
      frame_id,
      monomials,
      frame_basis,
      u,
      v,
      layout.u_terms,
      layout.v_terms,
      layout.n_params,
    )
  else:
    chunks = np.array_split(np.arange(n_stars), n_workers)
    ata = np.zeros((layout.n_params, layout.n_params), dtype=float)
    atb = np.zeros(layout.n_params, dtype=float)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
      futures = [
        pool.submit(
          _accumulate_joint_chunk,
          frame_id[idx],
          monomials[idx],
          frame_basis,
          u[idx],
          v[idx],
          layout.u_terms,
          layout.v_terms,
          layout.n_params,
        )
        for idx in chunks
        if len(idx)
      ]
      for fut in futures:
        chunk_ata, chunk_atb = fut.result()
        ata += chunk_ata
        atb += chunk_atb

  params_scaled = np.linalg.solve(ata, atb)
  params = params_scaled_to_pixel(
    params_scaled, layout, state.sip_degree, coord_scale=scale
  )
  state.unpack_params(params)
  try:
    cond = float(np.linalg.cond(ata))
  except Exception:
    cond = float("nan")
  state._last_ata_cond = cond  # type: ignore[attr-defined]
  state._last_coord_scale = scale  # type: ignore[attr-defined]
  return state


def _robust_sigma_scale(values: np.ndarray) -> float:
  arr = np.asarray(values, dtype=float)
  if arr.size == 0:
    return 0.0
  med = float(np.median(arr))
  mad = float(np.median(np.abs(arr - med)))
  if mad > 0:
    return mad * 1.4826
  std = float(np.std(arr))
  return std if std > 0 else 1e-6


def _joint_star_residuals(
    state: TemporalFitState,
    stacked: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  stems = stacked["stem"].to_numpy()
  btjd = stacked["btjd"].to_numpy(dtype=float)
  xprime = stacked["xprime"].to_numpy(dtype=float)
  yprime = stacked["yprime"].to_numpy(dtype=float)
  u = stacked["u"].to_numpy(dtype=float)
  v = stacked["v"].to_numpy(dtype=float)
  u_fit = np.empty_like(u)
  v_fit = np.empty_like(v)
  for stem in np.unique(stems):
    m = stems == stem
    uf, vf = state.predict_uv(xprime[m], yprime[m], float(btjd[m][0]))
    u_fit[m] = uf
    v_fit[m] = vf
  return u - u_fit, v - v_fit, stems


def _clip_joint_stars(
    du: np.ndarray,
    dv: np.ndarray,
    stems: np.ndarray,
    *,
    clip_n_sigma: float,
) -> np.ndarray:
  if clip_n_sigma <= 0:
    return np.ones(len(du), dtype=bool)
  keep = np.ones(len(du), dtype=bool)
  for stem in np.unique(stems):
    m = stems == stem
    r = np.hypot(du[m], dv[m])
    med = float(np.median(r))
    sig = _robust_sigma_scale(r)
    keep[m] = r <= med + clip_n_sigma * sig
  return keep


def _clip_joint_frames(
    state: TemporalFitState,
    stacked: pd.DataFrame,
    *,
    clip_n_sigma: float,
    min_frames: int,
) -> set[str]:
  stems = sorted(stacked["stem"].unique())
  if clip_n_sigma <= 0:
    return set(stems)
  scores = []
  for stem in stems:
    sub = stacked.loc[stacked["stem"] == stem]
    scores.append(frame_metrics(state, sub)["med_abs_du"])
  scores_arr = np.asarray(scores, dtype=float)
  med = float(np.median(scores_arr))
  sig = _robust_sigma_scale(scores_arr)
  keep = {
    stem
    for stem, score in zip(stems, scores_arr)
    if score <= med + clip_n_sigma * sig
  }
  if len(keep) < min_frames:
    return set(stems)
  return keep


def refine_joint_linear_robust(
    state: TemporalFitState,
    stacked: pd.DataFrame,
    *,
    star_clip_n_sigma: float = 3.0,
    frame_clip_n_sigma: float = 3.0,
    max_iter: int = 3,
    min_frames: int = 10,
    jobs: int = 0,
) -> tuple[TemporalFitState, pd.DataFrame, dict[str, object]]:
  """Iterative joint LS with per-FFI star clipping and whole-frame rejection."""
  working = stacked.reset_index(drop=True)
  log_lines: list[str] = []
  n_iter = 0

  for n_iter in range(1, max_iter + 1):
    state = refine_joint_linear(state, working, jobs=jobs)
    du, dv, stems = _joint_star_residuals(state, working)
    star_keep = _clip_joint_stars(
      du, dv, stems, clip_n_sigma=star_clip_n_sigma
    )
    frame_keep = _clip_joint_frames(
      state,
      working.loc[star_keep],
      clip_n_sigma=frame_clip_n_sigma,
      min_frames=min_frames,
    )
    next_working = working.loc[
      star_keep & working["stem"].isin(frame_keep)
    ].reset_index(drop=True)
    n_stars = len(next_working)
    n_frames = next_working["stem"].nunique()
    log_lines.append(
      f"iter {n_iter}: keep {n_frames} frames, {n_stars} stars "
      f"(clipped {len(working) - n_stars} stars, "
      f"{working['stem'].nunique() - n_frames} frames)"
    )
    if len(next_working) == len(working) and set(frame_keep) == set(working["stem"].unique()):
      working = next_working
      break
    working = next_working
    if n_frames < min_frames:
      raise RuntimeError(
        f"joint clip left only {n_frames} frames (min_frames={min_frames})"
      )

  info = {
    "n_iter": n_iter,
    "n_frames": int(working["stem"].nunique()),
    "n_stars": int(len(working)),
    "log_lines": log_lines,
    "frames_rejected": sorted(
      set(stacked["stem"].unique()) - set(working["stem"].unique())
    ),
  }
  return state, working, info


def refine_joint(
    state: TemporalFitState,
    stacked: pd.DataFrame,
    *,
    max_nfev: int = 200,
) -> TemporalFitState:
  stems = stacked["stem"].to_numpy()
  btjd = stacked["btjd"].to_numpy(dtype=float)
  xprime = stacked["xprime"].to_numpy(dtype=float)
  yprime = stacked["yprime"].to_numpy(dtype=float)
  u = stacked["u"].to_numpy(dtype=float)
  v = stacked["v"].to_numpy(dtype=float)

  p0 = state.pack_params()

  def fun(p: np.ndarray) -> np.ndarray:
    return joint_residuals(p, state, stems, btjd, xprime, yprime, u, v)

  result = least_squares(fun, p0, max_nfev=max_nfev, verbose=0)
  state.unpack_params(result.x)
  return state


def frame_metrics_warmstart(
    stars: pd.DataFrame,
    coeff_x: Sequence[float],
    coeff_y: Sequence[float],
    sip_degree: int,
) -> dict[str, float]:
  """Residual metrics for a per-frame single-FFI Sci2Idl fit (warm-start coeffs)."""
  xprime = stars["xprime"].to_numpy(dtype=float)
  yprime = stars["yprime"].to_numpy(dtype=float)
  u = stars["u"].to_numpy(dtype=float)
  v = stars["v"].to_numpy(dtype=float)
  u_fit = poly_eval(coeff_x, xprime, yprime, sip_degree)
  v_fit = poly_eval(coeff_y, xprime, yprime, sip_degree)
  du = u - u_fit
  dv = v - v_fit
  return {
    "med_abs_du": float(np.median(np.abs(du))),
    "med_abs_dv": float(np.median(np.abs(dv))),
    "n_stars": float(len(stars)),
  }


def warmstart_coeff_vectors(row: pd.Series, sip_degree: int) -> tuple[list[float], list[float]]:
  n = n_sci2idl_terms(sip_degree)
  cx = [float(row[f"c{i}_x"]) for i in range(n)]
  cy = [float(row[f"c{i}_y"]) for i in range(n)]
  return cx, cy


def frame_metrics(
    state: TemporalFitState,
    stars: pd.DataFrame,
) -> dict[str, float]:
  btjd = float(stars["btjd"].iloc[0])
  xprime = stars["xprime"].to_numpy(dtype=float)
  yprime = stars["yprime"].to_numpy(dtype=float)
  u = stars["u"].to_numpy(dtype=float)
  v = stars["v"].to_numpy(dtype=float)
  u_fit, v_fit = state.predict_uv(xprime, yprime, btjd)
  du = u - u_fit
  dv = v - v_fit
  return {
    "med_abs_du": float(np.median(np.abs(du))),
    "med_abs_dv": float(np.median(np.abs(dv))),
    "med_abs_du_identity": float(np.median(np.abs(u - xprime))),
    "med_abs_dv_identity": float(np.median(np.abs(v - yprime))),
    "n_stars": float(len(stars)),
  }
