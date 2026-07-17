"""Load empirical mask geometry YAML and magnitude → radius / cross helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numba import njit, prange

_DEFAULT_GEOMETRY_NAME = "mask_geometry.yaml"


def default_geometry_path() -> Path:
    """Packaged ``mask_geometry.yaml`` path."""
    return Path(
        resources.files("syndiff_pipeline.resources").joinpath(_DEFAULT_GEOMETRY_NAME)
    )


@lru_cache(maxsize=4)
def load_geometry(path: str | Path | None = None) -> dict[str, Any]:
    """Load geometry YAML (cached). ``path=None`` → packaged default."""
    p = Path(path) if path else default_geometry_path()
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    bins = data.get("circle_bins") or []
    data["_bin_hi"] = np.array([float(b[0]) for b in bins], dtype=np.float64)
    data["_bin_lo"] = np.array([float(b[1]) for b in bins], dtype=np.float64)
    data["_bin_rad"] = np.array([int(b[2]) for b in bins], dtype=np.int64)
    return data


def clear_geometry_cache() -> None:
    load_geometry.cache_clear()


def radius_from_mag(
    mag: float,
    scale: float = 1.0,
    *,
    geometry: dict[str, Any] | None = None,
) -> int:
    """
    Empirical circle radius for one magnitude.

    Bright (T < 7): cross body radius. Outside bins: faint/default radii from YAML.
    """
    geo = geometry or load_geometry()
    if not np.isfinite(mag):
        return int(geo.get("default_radius", 4))
    m = float(mag)
    if m < 7.0:
        body, _, _ = cross_geometry_from_mag(m, scale, geometry=geo)
        return max(int(body), 1)
    r = empirical_circle_radius(m, scale, geometry=geo)
    if r > 0:
        return int(r)
    if m > 18.0:
        return int(geo.get("faint_radius", 2))
    return int(geo.get("default_radius", 4))


def empirical_circle_radius(
    mag: float,
    scale: float = 1.0,
    *,
    geometry: dict[str, Any] | None = None,
) -> int:
    """Integer circle radius for mag in circle bins (0 if outside / below 9)."""
    geo = geometry or load_geometry()
    mag_min = float(geo.get("circle_mag_min", 9.0))
    if not np.isfinite(mag) or float(mag) < mag_min:
        return 0
    m = float(mag)
    for hi, lo, rad in zip(geo["_bin_hi"], geo["_bin_lo"], geo["_bin_rad"]):
        if m > float(lo) and m <= float(hi):
            return max(1, int(np.ceil(int(rad) * float(scale))))
    return 0


def cross_geometry_from_mag(
    mag: float,
    scale: float = 1.0,
    *,
    geometry: dict[str, Any] | None = None,
) -> tuple[int, int, int]:
    """Body radius, arm length, arm half-width for mag < cross_mag_max."""
    geo = geometry or load_geometry()
    b, L, w = _cross_geometry_one(
        float(mag),
        float(scale),
        float(geo.get("cross_mag_max", 9.0)),
        geo.get("cross_bins") or [],
    )
    return int(b), int(L), int(w)


def empirical_cross_geometry(
    mag: float, scale: float = 1.0, *, geometry: dict[str, Any] | None = None
) -> tuple[int, int, int]:
    """Alias for ``cross_geometry_from_mag`` (development API name)."""
    return cross_geometry_from_mag(mag, scale, geometry=geometry)


def _cross_geometry_one(
    mag: float,
    scale: float,
    cross_mag_max: float,
    cross_bins: list,
) -> tuple[int, int, int]:
    if mag >= cross_mag_max:
        return 0, 0, 0
    body = length = width = 0.0
    for entry in cross_bins:
        if mag <= float(entry["mag_max"]):
            body = float(entry["body"])
            length = float(entry["length"])
            width = float(entry["width"])
            break
    if body <= 0:
        return 0, 0, 0
    b = max(1, int(np.ceil(body * scale)))
    L = max(1, int(np.ceil(length * scale)))
    w = max(1, int(np.ceil(width * scale)))
    return b, L, w


@njit(cache=True)
def _radii_from_mags(
    mags: np.ndarray,
    scale: float,
    bin_hi: np.ndarray,
    bin_lo: np.ndarray,
    bin_rad: np.ndarray,
    mag_min: float,
) -> np.ndarray:
    n = mags.shape[0]
    out = np.zeros(n, dtype=np.int64)
    n_bins = bin_hi.shape[0]
    for i in range(n):
        m = mags[i]
        if m < mag_min:
            out[i] = 0
            continue
        rad = 0
        for b in range(n_bins):
            if m > bin_lo[b] and m <= bin_hi[b]:
                rad = bin_rad[b]
                break
        if rad > 0:
            r = int(np.ceil(rad * scale))
            out[i] = r if r >= 1 else 1
    return out


@njit(cache=True)
def _cross_geometry_numba(mag: float, scale: float) -> tuple:
    # Hard-coded bins matching packaged YAML (numba cannot take Python lists).
    if mag < 9.0:
        if mag <= 4.0:
            body, length, width = 38.0, 97.0, 11.0
        elif mag <= 5.0:
            body, length, width = 29.0, 61.0, 11.0
        elif mag <= 6.0:
            body, length, width = 24.0, 45.0, 7.0
        elif mag <= 7.0:
            body, length, width = 20.0, 34.0, 5.0
        elif mag <= 8.0:
            body, length, width = 13.0, 19.0, 5.0
        else:
            body, length, width = 10.0, 15.0, 3.0
    else:
        return 0, 0, 0
    b = int(np.ceil(body * scale))
    L = int(np.ceil(length * scale))
    w = int(np.ceil(width * scale))
    if b < 1:
        b = 1
    if L < 1:
        L = 1
    if w < 1:
        w = 1
    return b, L, w


@njit(parallel=True, cache=True)
def paint_circles(
    mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, radii: np.ndarray
) -> None:
    """Paint filled circles (dist <= r) onto uint8 mask in-place."""
    ny, nx = mask.shape
    n = xs.shape[0]
    for i in prange(n):
        r = radii[i]
        if r <= 0:
            continue
        xi = xs[i]
        yi = ys[i]
        y0 = yi - r
        if y0 < 0:
            y0 = 0
        y1 = yi + r + 1
        if y1 > ny:
            y1 = ny
        x0 = xi - r
        if x0 < 0:
            x0 = 0
        x1 = xi + r + 1
        if x1 > nx:
            x1 = nx
        r2 = r * r
        for y in range(y0, y1):
            dy = y - yi
            dy2 = dy * dy
            for x in range(x0, x1):
                dx = x - xi
                if dy2 + dx * dx <= r2:
                    mask[y, x] = 1


@njit(parallel=True, cache=True)
def paint_crosses(
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    mags: np.ndarray,
    scale: float,
) -> None:
    """Paint circular body + axis-aligned cross arms onto uint8 mask."""
    ny, nx = mask.shape
    n = xs.shape[0]
    for i in prange(n):
        body, length, width = _cross_geometry_numba(mags[i], scale)
        xi = xs[i]
        yi = ys[i]

        y0 = yi - body
        if y0 < 0:
            y0 = 0
        y1 = yi + body + 1
        if y1 > ny:
            y1 = ny
        x0 = xi - body
        if x0 < 0:
            x0 = 0
        x1 = xi + body + 1
        if x1 > nx:
            x1 = nx
        r2 = body * body
        for y in range(y0, y1):
            dy = y - yi
            dy2 = dy * dy
            for x in range(x0, x1):
                dx = x - xi
                if dy2 + dx * dx <= r2:
                    mask[y, x] = 1

        yh0 = yi - width
        if yh0 < 0:
            yh0 = 0
        yh1 = yi + width + 1
        if yh1 > ny:
            yh1 = ny
        xh0 = xi - length
        if xh0 < 0:
            xh0 = 0
        xh1 = xi + length + 1
        if xh1 > nx:
            xh1 = nx
        for y in range(yh0, yh1):
            for x in range(xh0, xh1):
                mask[y, x] = 1

        yv0 = yi - length
        if yv0 < 0:
            yv0 = 0
        yv1 = yi + length + 1
        if yv1 > ny:
            yv1 = ny
        xv0 = xi - width
        if xv0 < 0:
            xv0 = 0
        xv1 = xi + width + 1
        if xv1 > nx:
            xv1 = nx
        for y in range(yv0, yv1):
            for x in range(xv0, xv1):
                mask[y, x] = 1


def size_limit(x, y, image) -> np.ndarray:
    """Boolean index of pixels inside image boundaries."""
    yy, xx = image.shape
    return (y > 0) & (y < yy - 1) & (x > 0) & (x < xx - 1)


def gaia_circle_mask(
    table,
    image: np.ndarray,
    scale: float = 1.0,
    *,
    mag_min: float = 9.0,
    geometry: dict[str, Any] | None = None,
) -> dict:
    """Catalog mask with circular kernels (mag >= mag_min). Expects x, y, mag."""
    import pandas as pd

    geo = geometry or load_geometry()
    ny, nx = image.shape
    x = np.round(table["x"].to_numpy(float), 0).astype(np.int64)
    y = np.round(table["y"].to_numpy(float), 0).astype(np.int64)
    m = table["mag"].to_numpy(float).astype(np.float64)
    ind = size_limit(x, y, image)
    x, y, m = x[ind], y[ind], m[ind]

    keep = m >= mag_min
    x, y, m = x[keep], y[keep], m[keep]

    mask = np.zeros((ny, nx), dtype=np.uint8)
    if len(x):
        radii = _radii_from_mags(
            m,
            float(scale),
            geo["_bin_hi"],
            geo["_bin_lo"],
            geo["_bin_rad"],
            float(mag_min),
        )
        ok = radii > 0
        if np.any(ok):
            paint_circles(mask, x[ok], y[ok], radii[ok])

    return {"all": mask.astype(float)}


def big_sat_empirical(
    table,
    image: np.ndarray,
    scale: float = 1.0,
    *,
    mag_max: float = 9.0,
) -> np.ndarray:
    """Circular body + cross for mag < mag_max. Returns uint8 union mask."""
    sat = table[table["mag"].to_numpy(float) < mag_max]
    ny, nx = image.shape
    mask = np.zeros((ny, nx), dtype=np.uint8)
    if sat.empty:
        return mask

    x = np.round(sat["x"].to_numpy(float), 0).astype(np.int64)
    y = np.round(sat["y"].to_numpy(float), 0).astype(np.int64)
    m = sat["mag"].to_numpy(float).astype(np.float64)
    ind = size_limit(x, y, image)
    x, y, m = x[ind], y[ind], m[ind]
    if len(x) == 0:
        return mask

    paint_crosses(mask, x, y, m, float(scale))
    return mask


def warmup_numba() -> None:
    """Trigger JIT compile once."""
    from syndiff_pipeline.masking.faint_star_squares import paint_squares

    geo = load_geometry()
    m = np.zeros((32, 32), dtype=np.uint8)
    xs = np.array([16], dtype=np.int64)
    ys = np.array([16], dtype=np.int64)
    rs = np.array([3], dtype=np.int64)
    paint_circles(m, xs, ys, rs)
    mags = np.array([5.5], dtype=np.float64)
    paint_crosses(m, xs, ys, mags, 1.0)
    _radii_from_mags(
        mags, 1.0, geo["_bin_hi"], geo["_bin_lo"], geo["_bin_rad"], 9.0
    )
    half = np.array([1], dtype=np.int64)
    active = np.array([1], dtype=np.uint8)
    paint_squares(m, xs, ys, half, half, active)
