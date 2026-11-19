"""Photometry operators shared by the lane runner and archived band measurements."""

import numpy as np
from astropy.stats import sigma_clip

from syndiff_pipeline.template_creation.processing.perband import BANDS, validate_weights


def circular_aperture(planes, noise, good, xy, radius, annulus):
    """Measure D with a clipped sky, and each template band on the same aperture."""
    y, x = np.indices(noise.shape)
    distance = np.hypot(x - xy[0], y - xy[1])
    aperture = distance <= radius
    sky = good & (distance >= annulus[0]) & (distance <= annulus[1])
    blank = np.full(len(planes), np.nan)
    if (sky.sum() < 30 or np.any(aperture & ~good) or
            min(xy[0], xy[1], x.shape[1] - 1 - xy[0], y.shape[0] - 1 - xy[1])
            < annulus[1] - 1):
        return blank, np.nan, np.nan, 0
    clipped = sigma_clip(planes[0][sky], sigma=3, maxiters=5)
    keep = ~np.ma.getmaskarray(clipped)
    if not keep.any():
        return blank, np.nan, np.nan, 0
    level = float(np.median(clipped.compressed()))
    n = int(aperture.sum())
    values = planes[:, aperture].sum(axis=1, dtype=float)
    values[0] -= n * level
    variance_sky = np.pi / 2 * np.sum(noise[sky][keep] ** 2) / keep.sum() ** 2
    error = np.sqrt(np.sum(noise[aperture] ** 2) + n**2 * variance_sky)
    return values, error, level, int(keep.sum())


def joint_projection(planes, noise, good, profiles, xy, radius=6):
    """Apply one noise-weighted target projection to every image plane."""
    y, x = np.indices(noise.shape)
    fit = good & (np.hypot(x - xy[0], y - xy[1]) <= radius)
    if fit.sum() < 30:
        return np.full(len(planes), np.nan), np.nan
    profile = np.asarray(profiles)[:, fit].T.astype(float)
    sigma = noise[fit].astype(float)
    target = profile[:, 0] / sigma
    neighbours = profile[:, 1:]
    neighbours = neighbours[:, np.max(np.abs(neighbours), axis=0) > 1e-12]
    nuisance = np.column_stack([np.ones(len(target)), neighbours]) / sigma[:, None]
    contrast = target - nuisance @ np.linalg.lstsq(nuisance, target, rcond=1e-10)[0]
    info = contrast @ target
    if info <= 1e-10 * (target @ target):
        return np.full(len(planes), np.nan), np.nan
    row = contrast / (sigma * info)
    return planes[:, fit] @ row, np.linalg.norm(row * sigma)


def measure_bands(planes, noise, good, xy, methods, profiles=None):
    """Return one record per estimator. Plane order is D, r, i, z, y."""
    records = []
    for method in methods:
        if method["type"] == "aperture":
            values, error, sky, n_sky = circular_aperture(
                planes, noise, good, xy, method["radius"], method["annulus"]
            )
        else:
            profile = profiles(method) if profiles else None
            values, error = (np.full(len(planes), np.nan), np.nan)
            if profile is not None:
                values, error = joint_projection(
                    planes, noise, good, profile, xy, method.get("fit_radius", 6)
                )
            sky, n_sky = np.nan, 0
        records.append(dict(
            method=method["name"], D=values[0], error=error, sky=sky, n_sky=n_sky,
            **{f"S_{b}": value for b, value in zip(BANDS, values[1:])},
        ))
    return records


def score(table, weights, *, lane_weights=None):
    """Score a per-band table with trial weights and the fixed producing add-back."""
    weights = validate_weights(weights, "trial weights")
    lanes = (
        table[[f"lane_w_{b}" for b in BANDS]].to_numpy()
        if lane_weights is None else np.array([lane_weights[b] for b in BANDS])
    )
    bands = table[[f"S_{b}" for b in BANDS]].to_numpy()
    model = bands @ np.array([weights[b] for b in BANDS])
    restored = table["D"].to_numpy() + np.sum(bands * lanes, axis=1)
    valid = table["valid"].to_numpy(bool) if "valid" in table else np.ones(len(table), bool)
    ratio = np.divide(restored, model, out=np.full(len(table), np.nan), where=(model > 0) & valid)
    return table.assign(template_flux=model, restored_flux=restored, flux_ratio=ratio)


def restore_stamp(images, lane_weights):
    """Restore an archived stamp using the weights of its producing lane."""
    weights = validate_weights(lane_weights, "producing weights")
    return images[0] + np.einsum("b,byx->yx", [weights[b] for b in BANDS], images[1:])
