# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Gaia RA/Dec → crop-local (x, y) via temporally varying WCS fits."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
  pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.temporal_model import TemporalFitState  # noqa: E402

_DEFAULT_OUTPUT_ROOT = _REPO / "dev" / "temporal_wcs_poly" / "output"

_S20_ORBIT_RUNS = (
    ("s20_c3_k3_orbit1_bspline_scaled", None),
    ("s20_c3_k3_orbit2_bspline_scaled", "s20_c3_k3_orbit2_bspline"),
)


@dataclass
class TemporalWcsCatalog:
  runs: list[tuple[float, float, Any]]  # btjd_lo, btjd_hi, TemporalFitState


def _linear_wcs_from_meta(meta: dict) -> WCS:
  linear_hdr = fits.Header()
  for key, val in meta["linear_wcs"].items():
    if val is not None:
      linear_hdr[key] = val
  return WCS(linear_hdr)


def load_temporal_run(run_dir: Path) -> tuple[float, float, TemporalFitState]:
  """Load one fit run: fit_meta.json + poly_coeffs.npz → (btjd_lo, btjd_hi, state)."""
  run_dir = run_dir.resolve()
  meta_path = run_dir / "fit_meta.json"
  coeffs_path = run_dir / "poly_coeffs.npz"
  if not meta_path.is_file():
    raise FileNotFoundError(f"missing fit_meta.json in {run_dir}")
  if not coeffs_path.is_file():
    raise FileNotFoundError(f"missing poly_coeffs.npz in {run_dir}")

  meta = json.loads(meta_path.read_text())
  linear_wcs = _linear_wcs_from_meta(meta)
  state = TemporalFitState.from_npz(linear_wcs, np.load(coeffs_path))

  btjd_lo = float(meta["btjd_orbit_start"])
  btjd_hi = float(meta["btjd_orbit_end"])
  return btjd_lo, btjd_hi, state


def _try_load_run(run_dir: Path) -> tuple[float, float, TemporalFitState] | None:
  meta_path = run_dir / "fit_meta.json"
  coeffs_path = run_dir / "poly_coeffs.npz"
  if not meta_path.is_file() or not coeffs_path.is_file():
    missing = []
    if not meta_path.is_file():
      missing.append("fit_meta.json")
    if not coeffs_path.is_file():
      missing.append("poly_coeffs.npz")
    warnings.warn(
      f"Skipping incomplete run {run_dir.name}: missing {', '.join(missing)}",
      stacklevel=2,
    )
    return None
  return load_temporal_run(run_dir)


def load_s20_scaled_catalog(
    output_root: Path | None = None,
) -> TemporalWcsCatalog:
  """Load s20_c3_k3_orbit1_bspline_scaled and orbit2_bspline_scaled if poly_coeffs.npz exists.

  Default output_root = repo/dev/temporal_wcs_poly/output.
  Skip incomplete runs (no poly_coeffs.npz / fit_meta.json) with a warning.
  Fallback: if orbit2_scaled missing, also try s20_c3_k3_orbit2_bspline (non-scaled).
  """
  root = (output_root or _DEFAULT_OUTPUT_ROOT).resolve()
  runs: list[tuple[float, float, TemporalFitState]] = []

  for primary_name, fallback_name in _S20_ORBIT_RUNS:
    primary = root / primary_name
    loaded = _try_load_run(primary)
    if loaded is None and fallback_name is not None:
      loaded = _try_load_run(root / fallback_name)
    if loaded is not None:
      runs.append(loaded)

  if not runs:
    raise FileNotFoundError(f"no complete s20 temporal WCS runs found under {root}")

  runs.sort(key=lambda r: r[0])
  return TemporalWcsCatalog(runs=runs)


def select_state(catalog: TemporalWcsCatalog, btjd: float) -> TemporalFitState:
  """Pick state whose [btjd_lo, btjd_hi] contains btjd. Raise clear error if none."""
  for btjd_lo, btjd_hi, state in catalog.runs:
    if btjd_lo <= btjd <= btjd_hi:
      return state
  windows = ", ".join(f"[{lo:.3f}, {hi:.3f}]" for lo, hi, _ in catalog.runs)
  raise ValueError(f"btjd={btjd} falls outside all catalog windows: {windows}")


def radec_to_crop_xy(
    ra: float, dec: float, btjd: float, state: TemporalFitState
) -> tuple[float, float]:
  """state.build_header(btjd) → WCS.all_world2pix(..., origin=0, quiet=True). Return crop-local floats."""
  wcs = WCS(state.build_header(btjd))
  x, y = wcs.all_world2pix(ra, dec, 0, quiet=True)
  return float(x), float(y)


def radec_to_crop_xy_series(
    ra: float, dec: float, btjds, catalog: TemporalWcsCatalog
) -> pd.DataFrame:
  """Columns: btjd, x, y. One row per btjd."""
  rows = []
  for btjd in btjds:
    btjd_f = float(btjd)
    state = select_state(catalog, btjd_f)
    x, y = radec_to_crop_xy(ra, dec, btjd_f, state)
    rows.append({"btjd": btjd_f, "x": x, "y": y})
  return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="RA/Dec → crop-local (x, y) via temporal WCS.")
  p.add_argument("--ra", type=float, required=True)
  p.add_argument("--dec", type=float, required=True)
  p.add_argument("--btjd", type=float, required=True)
  p.add_argument("--run-dir", type=Path, default=None, help="Single fit run dir (overrides catalog).")
  p.add_argument(
      "--output-root",
      type=Path,
      default=None,
      help="Catalog root (default: dev/temporal_wcs_poly/output).",
  )
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  if args.run_dir is not None:
    _, _, state = load_temporal_run(args.run_dir)
  else:
    catalog = load_s20_scaled_catalog(args.output_root)
    state = select_state(catalog, args.btjd)
  x, y = radec_to_crop_xy(args.ra, args.dec, args.btjd, state)
  print(f"x={x} y={y}")


if __name__ == "__main__":
  main()
