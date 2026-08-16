"""Runtime temporal Chebyshev detector WCS.

This module deliberately has no dependency on the experimental ``dev`` tree.
The model is a fixed, linear TAN WCS plus a detector-local, total-degree
Chebyshev correction whose coefficients are cubic B-splines in BTJD::

    (ra, dec) -> (x_linear, y_linear) + B(x_linear, y_linear) @ C(t)

The representation is numpy/NPZ friendly so workers can load one small model
and evaluate it without constructing an Astropy WCS for every FFI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np


def _exponents(degree: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, j) for total in range(degree + 1) for i in range(total + 1) for j in (total - i,))


def _vandermonde(value: np.ndarray, degree: int) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(-1)
    out = np.empty((value.size, degree + 1), dtype=float)
    out[:, 0] = 1.0
    if degree:
        out[:, 1] = value
    for k in range(2, degree + 1):
        out[:, k] = 2.0 * value * out[:, k - 1] - out[:, k - 2]
    return out


def chebyshev_design(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Return the total-degree product-Chebyshev design matrix."""
    tx, ty = _vandermonde(x, degree), _vandermonde(y, degree)
    return np.column_stack([tx[:, i] * ty[:, j] for i, j in _exponents(degree)])


def canonical_temporal_wcs_stem(value) -> str:
    """Return the temporal-store key for a local SPOC FFI or an existing key.

    The temporal fit lane stores the cadence/camera identity without the
    ``-0165-s_ffic.fits[.fz]`` suffix, while mapping/remap naturally receive
    the on-disk SPOC filename.  Keep this normalization in one place so
    publication and runtime cannot disagree about a frame key.
    """
    key = str(value)
    match = re.match(
        r"^(tess\d+-s\d{4}-\d-\d)(?:-\d{4}-s_ffic\.fits(?:\.fz)?)?$",
        Path(key).name,
    )
    return match.group(1) if match is not None else key


def _linear_pixels(ra, dec, *, ra0, dec0, cd_inv, crpix):
    ra = np.deg2rad(np.asarray(ra, dtype=float))
    dec = np.deg2rad(np.asarray(dec, dtype=float))
    ra0, dec0 = np.deg2rad(float(ra0)), np.deg2rad(float(dec0))
    dra = ra - ra0
    den = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    xi = np.cos(dec) * np.sin(dra) / den
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / den
    uv = np.rad2deg(np.stack((xi, eta), axis=-1)) @ np.asarray(cd_inv, dtype=float).T
    pix = uv + np.asarray(crpix, dtype=float) - 1.0
    return pix[..., 0], pix[..., 1]


def _temporal_basis(btjd, knots, degree):
    from scipy.interpolate import BSpline

    t = np.asarray(btjd, dtype=float)
    # knots are stored in normalized [0, 1] coordinates with clamped ends.
    t = np.clip(t, 0.0, 1.0)
    return BSpline.design_matrix(t, knots, degree, extrapolate=False).toarray()


def _make_knots(btjd, n_interior, degree=3):
    btjd = np.asarray(btjd, dtype=float)
    if btjd.size < 2 or not np.all(np.diff(btjd) > 0):
        raise ValueError("btjd must contain at least two strictly increasing values")
    n_interior = max(0, min(int(n_interior), max(0, btjd.size - degree - 2)))
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    return np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]


def fit_per_ffi_chebyshev(reference_wcs, ra, dec, x_observed, y_observed, *,
                          center, half_extents, poly_degree=5, n_sigma=4.0,
                          max_iter=5):
    """Fit one FFI's detector correction, returning coefficients and QC fields."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    xobs, yobs = np.asarray(x_observed, float), np.asarray(y_observed, float)
    helper = TemporalChebWcs.from_reference_wcs(reference_wcs, center=center,
        half_extents=half_extents, poly_degree=poly_degree)
    xlin, ylin, design = helper._design_for_world(ra, dec)
    mask = np.isfinite(xobs) & np.isfinite(yobs)
    for _ in range(max_iter):
        cx = np.linalg.lstsq(design[mask], xobs[mask] - xlin[mask], rcond=None)[0]
        cy = np.linalg.lstsq(design[mask], yobs[mask] - ylin[mask], rcond=None)[0]
        residual = np.hypot(xobs - xlin - design @ cx, yobs - ylin - design @ cy)
        finite = np.isfinite(residual)
        med = np.nanmedian(residual[mask])
        scale = max(1.4826 * np.nanmedian(np.abs(residual[mask] - med)), 1e-6)
        new_mask = finite & (residual - med < n_sigma * scale)
        if new_mask.sum() == mask.sum() or new_mask.sum() < helper.n_terms + 2:
            break
        mask = new_mask
    residual = np.hypot(xobs - xlin - design @ cx, yobs - ylin - design @ cy)
    return {"coeff_x": cx, "coeff_y": cy, "keep_mask": mask, "residual": residual,
            "n_stars": int(mask.sum())}


def fit_temporal_coefficients(btjd, per_frame_coefficients, *, n_interior=10,
                              spline_degree=3):
    """Smooth per-FFI coefficient rows; rejected/NaN rows are omitted."""
    btjd = np.asarray(btjd, float)
    values = np.asarray(per_frame_coefficients, float)
    if values.ndim != 2 or values.shape[0] != btjd.size:
        raise ValueError("per_frame_coefficients must have shape (n_frames, n_coefficients)")
    valid = np.isfinite(btjd) & np.all(np.isfinite(values), axis=1)
    if valid.sum() < 2:
        raise ValueError("at least two valid FFI coefficient rows are required")
    t = btjd[valid]
    knots = _make_knots(t, n_interior, spline_degree)
    basis = _temporal_basis((t - t[0]) / (t[-1] - t[0]), knots, spline_degree)
    coeff = np.linalg.lstsq(basis, values[valid], rcond=None)[0].T
    return {"coeff_matrix": coeff, "knot_vector": knots, "btjd_ref": float(t[0]),
            "btjd_scale": float(t[-1] - t[0]), "valid_mask": valid}


@dataclass
class TemporalChebWcs:
    """Serializable temporal Chebyshev WCS model."""

    ra0_deg: float
    dec0_deg: float
    cd_inv: np.ndarray
    crpix: np.ndarray
    center: np.ndarray
    half_extents: np.ndarray
    poly_degree: int
    knot_vector: np.ndarray
    spline_degree: int
    btjd_ref: float
    btjd_scale: float
    coeff_matrix: np.ndarray  # (2*n_terms, n_spline_coefficients)

    @property
    def exponents(self):
        return _exponents(self.poly_degree)

    @property
    def n_terms(self):
        return len(self.exponents)

    @property
    def n_basis(self):
        return self.coeff_matrix.shape[1]

    @classmethod
    def from_reference_wcs(cls, wcs, *, center, half_extents, poly_degree=5,
                           btjd_ref=0.0, btjd_scale=1.0, knot_vector=None,
                           spline_degree=3, coeff_matrix=None):
        matrix = getattr(wcs, "pixel_scale_matrix", None)
        if matrix is None:
            matrix = getattr(wcs.wcs, "cd", None)
        if matrix is None:
            matrix = np.asarray(wcs.wcs.pc, dtype=float) * np.asarray(wcs.wcs.cdelt, dtype=float)
        matrix = np.asarray(matrix, dtype=float)
        if knot_vector is None:
            knot_vector = np.r_[np.zeros(spline_degree + 1), np.ones(spline_degree + 1)]
        n_basis = len(knot_vector) - spline_degree - 1
        if coeff_matrix is None:
            coeff_matrix = np.zeros((2 * len(_exponents(poly_degree)), n_basis), dtype=float)
        return cls(float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1]), np.linalg.inv(matrix),
                   np.asarray(wcs.wcs.crpix, float), np.asarray(center, float),
                   np.asarray(half_extents, float), int(poly_degree), np.asarray(knot_vector, float),
                   int(spline_degree), float(btjd_ref), float(btjd_scale), np.asarray(coeff_matrix, float))

    def linear_pixels(self, ra, dec):
        return _linear_pixels(ra, dec, ra0=self.ra0_deg, dec0=self.dec0_deg,
                              cd_inv=self.cd_inv, crpix=self.crpix)

    def _design_for_world(self, ra, dec):
        x, y = self.linear_pixels(ra, dec)
        return x, y, chebyshev_design((x - self.center[0]) / self.half_extents[0],
                                      (y - self.center[1]) / self.half_extents[1], self.poly_degree)

    def world_to_pixel_values(self, ra, dec, btjd):
        scalar = np.asarray(ra).ndim == 0 and np.asarray(dec).ndim == 0 and np.asarray(btjd).ndim == 0
        x, y, design = self._design_for_world(ra, dec)
        output_shape = np.broadcast(np.asarray(x), np.asarray(y)).shape
        x_flat = np.asarray(x, dtype=float).reshape(-1)
        y_flat = np.asarray(y, dtype=float).reshape(-1)
        tau = (np.asarray(btjd, float) - self.btjd_ref) / self.btjd_scale
        basis = _temporal_basis(tau, self.knot_vector, self.spline_degree)
        frame_coeff = basis @ self.coeff_matrix.T
        if frame_coeff.shape[0] == 1:
            frame_coeff = frame_coeff[0]
            out_x = (x_flat + design @ frame_coeff[:self.n_terms]).reshape(output_shape)
            out_y = (y_flat + design @ frame_coeff[self.n_terms:]).reshape(output_shape)
            if scalar:
                return float(np.asarray(out_x).reshape(-1)[0]), float(np.asarray(out_y).reshape(-1)[0])
            return out_x, out_y
        return ((x_flat[None, :] + design @ frame_coeff[:, :self.n_terms].T).reshape((len(frame_coeff),) + output_shape),
                (y_flat[None, :] + design @ frame_coeff[:, self.n_terms:].T).reshape((len(frame_coeff),) + output_shape))

    def pixel_to_world(self, x, y, btjd):
        """Numerically invert the forward model, returning ``(ra, dec)`` degrees."""
        # Astropy WCS accepts a scalar coordinate paired with a coordinate
        # vector.  Mapping uses precisely that form for detector-edge grids,
        # so normalize to their shared broadcast shape before stacking/Newton
        # refinement.
        target_x, target_y = np.broadcast_arrays(
            np.asarray(x, float), np.asarray(y, float)
        )
        if target_x.size > 100_000:
            # Keep the authoritative inverse exact while bounding temporary
            # Chebyshev design matrices.  The previous one-pass shortcut was
            # fast but returned an approximate WCS (and is not acceptable for
            # production geometry).
            flat_x, flat_y = target_x.reshape(-1), target_y.reshape(-1)
            out_ra = np.empty_like(flat_x)
            out_dec = np.empty_like(flat_y)
            for start in range(0, flat_x.size, 100_000):
                stop = min(start + 100_000, flat_x.size)
                out_ra[start:stop], out_dec[start:stop] = self.pixel_to_world(
                    flat_x[start:stop], flat_y[start:stop], btjd
                )
            shape = np.shape(target_x)
            return out_ra.reshape(shape), out_dec.reshape(shape)
        # Linear TAN inverse is an excellent starting point for the small
        # detector correction.  The Newton refinement below also handles
        # scalar and vector inputs uniformly.
        uv = np.stack((target_x - self.crpix[0] + 1.0, target_y - self.crpix[1] + 1.0), axis=-1)
        tangent = np.deg2rad(uv @ np.linalg.inv(self.cd_inv).T)
        xi, eta = tangent[..., 0], tangent[..., 1]
        rho = np.hypot(xi, eta)
        c = np.arctan(rho)
        ra0, dec0 = np.deg2rad(self.ra0_deg), np.deg2rad(self.dec0_deg)
        sin_c, cosc = np.sin(c), np.cos(c)
        safe_rho = np.where(rho == 0, 1, rho)
        dec = np.arcsin(cosc * np.sin(dec0) + np.divide(eta * sin_c * np.cos(dec0), safe_rho))
        ra = ra0 + np.arctan2(xi * sin_c, rho * np.cos(dec0) * cosc - eta * np.sin(dec0) * sin_c)
        ra, dec = np.rad2deg(ra), np.rad2deg(dec)
        # Continue well below a pixel-level acceptance threshold.  Stopping at
        # 1e-6 px made a nominally successful inverse report ~2e-7 px residual
        # on a dense detector grid; Part 1 requires the inverse itself to be
        # converged, not merely inside the downstream mapping tolerance.
        for _ in range(12):
            px, py = self.world_to_pixel_values(ra.reshape(-1), dec.reshape(-1), btjd)
            dx, dy = target_x.reshape(-1) - px, target_y.reshape(-1) - py
            # The analytic TAN inverse is already exact when the temporal
            # correction is zero.  Continuing Newton updates from floating
            # point round-off amplifies sub-pixel noise into a divergent sky
            # coordinate, especially at the detector edge.
            if np.all(np.hypot(dx, dy) <= 1e-10):
                break
            eps = 1e-5
            qx, qy = self.world_to_pixel_values(ra.reshape(-1) + eps, dec.reshape(-1), btjd)
            rx, ry = self.world_to_pixel_values(ra.reshape(-1), dec.reshape(-1) + eps, btjd)
            j_x_ra, j_y_ra = (qx - px) / eps, (qy - py) / eps
            j_x_dec, j_y_dec = (rx - px) / eps, (ry - py) / eps
            det = j_x_ra * j_y_dec - j_x_dec * j_y_ra
            if not np.all(np.isfinite(det)) or np.any(np.abs(det) < 1e-18):
                raise RuntimeError("temporal WCS inverse Jacobian is singular or non-finite")
            dra = (dx * j_y_dec - dy * j_x_dec) / det
            ddec = (j_x_ra * dy - j_y_ra * dx) / det
            # The TAN seed is close to the solution and the local detector WCS
            # is well-conditioned, so a full Newton step converges rapidly.
            ra, dec = ra.reshape(-1) + dra, dec.reshape(-1) + ddec
        final_x, final_y = self.world_to_pixel_values(ra, dec, btjd)
        final_residual = np.hypot(target_x.reshape(-1) - final_x,
                                  target_y.reshape(-1) - final_y)
        if (not np.all(np.isfinite(final_residual))
                or float(np.max(final_residual, initial=0.0)) > 1e-8):
            raise RuntimeError(
                "temporal WCS inverse failed convergence: "
                f"max residual={float(np.nanmax(final_residual)):.6g} px"
            )
        return ra.reshape(np.shape(target_x)), dec.reshape(np.shape(target_y))

    def all_pix2world(self, x, y, btjd, origin=0):
        """Astropy-WCS-like convenience wrapper (``origin`` is accepted for API parity)."""
        del origin
        ra, dec = self.pixel_to_world(x, y, btjd)
        return np.stack((ra, dec), axis=-1)

    def at_time(self, btjd):
        """Return an Astropy-call-compatible view with BTJD fixed."""
        return TemporalChebWcsAtTime(self, float(btjd))


    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ra0_deg=self.ra0_deg, dec0_deg=self.dec0_deg,
            cd_inv=self.cd_inv, crpix=self.crpix, center=self.center,
            half_extents=self.half_extents, poly_degree=self.poly_degree,
            knot_vector=self.knot_vector, spline_degree=self.spline_degree,
            btjd_ref=self.btjd_ref, btjd_scale=self.btjd_scale,
            coeff_matrix=self.coeff_matrix)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(*(z[k].item() if z[k].ndim == 0 else z[k] for k in
                         ("ra0_deg", "dec0_deg", "cd_inv", "crpix", "center", "half_extents",
                          "poly_degree", "knot_vector", "spline_degree", "btjd_ref", "btjd_scale", "coeff_matrix")))

    @classmethod
    def fit(cls, reference_wcs, ra, dec, btjd, x_observed, y_observed, *, center,
            half_extents, poly_degree=5, n_interior=10, spline_degree=3,
            n_sigma=4.0, max_iter=5):
        """Fit per-FFI corrections then smooth each coefficient with B-splines."""
        btjd = np.asarray(btjd, float)
        xobs, yobs = np.asarray(x_observed, float), np.asarray(y_observed, float)
        if xobs.shape != yobs.shape or xobs.shape != (btjd.size, np.asarray(ra).size):
            raise ValueError("observations must have shape (n_frames, n_stars)")
        knots = _make_knots(btjd, n_interior, spline_degree)
        base = cls.from_reference_wcs(reference_wcs, center=center, half_extents=half_extents,
            poly_degree=poly_degree, btjd_ref=float(btjd[0]), btjd_scale=float(btjd[-1] - btjd[0]),
            knot_vector=knots, spline_degree=spline_degree)
        xlin, ylin, design = base._design_for_world(ra, dec)
        per = np.zeros((btjd.size, 2 * base.n_terms))
        for k in range(btjd.size):
            mask = np.isfinite(xobs[k]) & np.isfinite(yobs[k])
            for _ in range(max_iter):
                cx = np.linalg.lstsq(design[mask], xobs[k, mask] - xlin[mask], rcond=None)[0]
                cy = np.linalg.lstsq(design[mask], yobs[k, mask] - ylin[mask], rcond=None)[0]
                residual = np.hypot(xobs[k] - xlin - design @ cx, yobs[k] - ylin - design @ cy)
                good = np.isfinite(residual) & (residual < np.nanmedian(residual[mask]) + n_sigma * max(np.nanmedian(np.abs(residual[mask] - np.nanmedian(residual[mask]))) * 1.4826, 1e-6))
                if good.sum() == mask.sum() or good.sum() < base.n_terms + 2: break
                mask = good
            per[k] = np.r_[cx, cy]
        temporal = _temporal_basis((btjd - btjd[0]) / (btjd[-1] - btjd[0]), knots, spline_degree)
        base.coeff_matrix = np.linalg.lstsq(temporal, per, rcond=None)[0].T
        return base


@dataclass
class TemporalChebWcsAtTime:
    """Pickle-safe Astropy-like view of one temporal WCS at fixed BTJD."""

    model: TemporalChebWcs
    btjd: float

    def all_pix2world(self, x, y=None, origin=0):
        pixels = np.asarray(x)
        if y is not None and np.ndim(y) == 0 and pixels.ndim >= 1 and pixels.shape[-1] == 2 and origin == 0:
            ra, dec = self.model.pixel_to_world(pixels[..., 0], pixels[..., 1], self.btjd)
            return np.stack((ra, dec), axis=-1)
        if y is None:
            raise TypeError("all_pix2world requires x/y or an (N, 2) pixel array with origin")
        return self.model.pixel_to_world(x, y, self.btjd)

    def pixel_to_world(self, x, y):
        return self.model.pixel_to_world(x, y, self.btjd)

    def pixel_to_world_values(self, x, y=None):
        """Astropy-WCS-compatible pixel-to-world tuple at the fixed time."""
        if y is None:
            pixels = np.asarray(x)
            if pixels.shape[-1] != 2:
                raise ValueError("pixel coordinates must have a final dimension of 2")
            x, y = pixels[..., 0], pixels[..., 1]
        return self.model.pixel_to_world(x, y, self.btjd)

    def world_to_pixel_values(self, ra, dec):
        return self.model.world_to_pixel_values(ra, dec, self.btjd)


class TemporalChebWcsStore:
    """Read-only published temporal-WCS store keyed by exact FFI stem."""

    def __init__(self, root):
        import pandas as pd

        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("model_kind") != "temporal_wcs":
            raise ValueError(f"{self.root}: not a temporal_wcs artifact")
        if self.manifest.get("spatial_basis") != "chebyshev" or int(self.manifest.get("spatial_degree", -1)) != 5:
            raise ValueError(f"{self.root}: incompatible spatial WCS basis")
        self.frames = pd.read_parquet(self.root / "frames.parquet").set_index("stem", drop=False)
        self._models = {}

    @property
    def fingerprint(self):
        return __import__("hashlib").sha256(
            (self.root / "manifest.json").read_bytes()
        ).hexdigest()

    def for_stem(self, stem):
        key = canonical_temporal_wcs_stem(stem)
        row = self.frames.loc[key]
        if getattr(row, "ndim", 1) != 1:
            raise ValueError(f"{self.root}: duplicate temporal-WCS stem {stem!r}")
        orbit = int(row["orbit_index"])
        if orbit < 0:
            raise ValueError(f"{self.root}: model missing for stem {stem!r}")
        if orbit not in self._models:
            spec = next(m for m in self.manifest["models"] if int(m["orbit_index"]) == orbit)
            self._models[orbit] = TemporalChebWcs.load(self.root / spec["path"])
        return self._models[orbit], float(row["btjd"])


__all__ = ["TemporalChebWcs", "canonical_temporal_wcs_stem", "chebyshev_design",
           "fit_per_ffi_chebyshev", "fit_temporal_coefficients", "TemporalChebWcsStore"]
