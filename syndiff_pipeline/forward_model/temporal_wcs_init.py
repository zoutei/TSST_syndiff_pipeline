# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Centroid-free forward-WCS initialization from a published temporal WCS.

This prep-only adapter samples the authoritative per-FFI temporal Chebyshev
model on a regular crop-local pixel grid.  A TAN reference WCS is fitted from
the first selected frame and a deliberately lower-order residual Chebyshev
track is fitted for every frame, then projected onto the forward fitter's
frozen B-spline basis.  No ``centroids_r1`` tables are opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import fit_wcs_from_points

from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import TemporalChebWcsStore

from . import cheb_wcs as CW


@dataclass(frozen=True)
class TemporalGridInit:
    reference_wcs: object
    wcs_coeff: np.ndarray
    metadata: dict


def _grid(region, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    if grid_size < 2:
        raise ValueError("temporal-WCS grid size must be at least 2")
    x = np.linspace(float(region.x_min), float(region.x_max - 1), grid_size)
    y = np.linspace(float(region.y_min), float(region.y_max - 1), grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def build_temporal_grid_init(
    temporal_wcs_root: Path,
    frames: list,
    region,
    wcs_frame_basis: np.ndarray,
    *,
    cheb_degree: int,
    grid_size: int = 100,
) -> TemporalGridInit:
    """Fit forward-WCS initialization from temporal-WCS grid samples.

    ``cheb_degree`` is intentionally chosen by the caller below the published
    temporal model's degree five; it controls the compact crop approximation.
    """
    store = TemporalChebWcsStore(temporal_wcs_root)
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.cheb_poly_fit import cheb_design_matrix

    gx, gy = _grid(region, int(grid_size))
    first_model, first_btjd = store.raw_for_stem(frames[0].stem)
    ra, dec = first_model.pixel_to_world(gx, gy, first_btjd)
    reference_wcs = fit_wcs_from_points(
        np.vstack((gx, gy)),
        SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg),
        projection="TAN",
        sip_degree=None,
    )
    reference_wcs.array_shape = region.shape
    static = CW.ChebWcsStatic.from_wcs(reference_wcs, region, poly_degree=cheb_degree)

    coeff_x_track = np.empty((len(frames), static.n_terms), dtype=float)
    coeff_y_track = np.empty((len(frames), static.n_terms), dtype=float)
    direct_rms = np.empty(len(frames), dtype=float)
    direct_max = np.empty(len(frames), dtype=float)
    for index, frame in enumerate(frames):
        model, model_btjd = store.raw_for_stem(frame.stem)
        fra, fdec = model.pixel_to_world(gx, gy, model_btjd)
        x_lin, y_lin = CW.linear_predict(np.asarray(fra), np.asarray(fdec), static)
        cx, cy, _ = CW.fit_frame_cheb_warmstart(
            np.asarray(x_lin), np.asarray(y_lin), gx, gy, static,
            n_sigma=10.0, max_iter=2,
        )
        coeff_x_track[index] = cx
        coeff_y_track[index] = cy
        design = cheb_design_matrix(
            (np.asarray(x_lin) - static.center[0]) / static.half_extents[0],
            (np.asarray(y_lin) - static.center[1]) / static.half_extents[1],
            static.poly_degree,
        )
        dx = gx - (np.asarray(x_lin) + design @ cx)
        dy = gy - (np.asarray(y_lin) + design @ cy)
        residual = np.hypot(dx, dy)
        direct_rms[index] = float(np.sqrt(np.mean(residual**2)))
        direct_max[index] = float(np.max(residual))

    basis = np.asarray(wcs_frame_basis, dtype=float)
    cx_spline, *_ = np.linalg.lstsq(basis, coeff_x_track, rcond=None)
    cy_spline, *_ = np.linalg.lstsq(basis, coeff_y_track, rcond=None)
    coeff_matrix = np.concatenate((cx_spline.T, cy_spline.T), axis=0).astype(np.float32)
    metadata = {
        "enabled": True,
        "source": "published_temporal_wcs_grid",
        "temporal_wcs_root": str(temporal_wcs_root),
        "temporal_wcs_fingerprint": store.fingerprint,
        "grid_size": int(grid_size),
        "grid_points": int(gx.size),
        "crop_cheb_degree": int(cheb_degree),
        "n_frames": len(frames),
        "direct_grid_rms_px_median": float(np.median(direct_rms)),
        "direct_grid_rms_px_max": float(np.max(direct_rms)),
        "direct_grid_max_px": float(np.max(direct_max)),
    }
    return TemporalGridInit(reference_wcs=reference_wcs, wcs_coeff=coeff_matrix, metadata=metadata)


def main() -> None:
    """Evaluate a temporal-WCS crop approximation without loading image pixels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--temporal-wcs-root", type=Path, required=True)
    parser.add_argument("--sector", type=int, required=True)
    parser.add_argument("--orbit-index", type=int, default=1)
    parser.add_argument("--region", required=True)
    parser.add_argument("--cheb-degree", type=int, required=True)
    parser.add_argument("--grid-size", type=int, default=100)
    args = parser.parse_args()

    from .data import RegionSpec, list_orbit_frames_from_temporal_wcs
    from .temporal import build_temporal_basis

    frames, _ = list_orbit_frames_from_temporal_wcs(
        args.workspace, args.temporal_wcs_root,
        sector=args.sector, orbit_index=args.orbit_index,
    )
    btjd = np.asarray([frame.btjd for frame in frames], dtype=float)
    basis = build_temporal_basis(btjd, n_interior=10, uniform_knots=True)
    init = build_temporal_grid_init(
        args.temporal_wcs_root, frames, RegionSpec.parse(args.region),
        np.asarray(basis.frame_basis), cheb_degree=args.cheb_degree,
        grid_size=args.grid_size,
    )
    print(init.metadata)


if __name__ == "__main__":
    main()
