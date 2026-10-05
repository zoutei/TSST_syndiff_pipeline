"""Per-PS1-band templates: split the combined rizy cell into its four weighted bands.

The production template is built from one combined image per skycell,

    C = Z * sum_b w_b F_b,

where ``F_b`` is the flux-converted PS1 band image, ``w_b`` the band weight
(``combined_store.DEFAULT_BAND_WEIGHTS`` for the production store, or the ``band_weights`` recorded in
the store's recipe otherwise, e.g. the adopted D13 set; see ``split_with_store_weights``) and ``Z`` the 0/1 keep mask that
``band_utils.remove_background`` applies (it only ZEROES pixels: outside SEP
segments, and the segments of removed bright/saturated stars; it subtracts no
background). Every later step -- the Gaussian pre-blur, cross-projection
padding, regmap binning -- is linear. So the four band cells

    C_b = Z * w_b F_b

carried through the same chain give four band templates T_b with
``sum_b T_b = T`` (the production template), and each can be convolved with its
own kernel K_b:

    M = sum_b K_b (*) T_b.

``Z`` is recovered exactly from a stored combined cell (``split_like_combined``):
a pixel was zeroed iff it is 0 after ``remove_background`` but not before. This
backfills band cells from the existing combined store without re-running SEP or
the Gaia catalogue pass, and guarantees the split sums to the stored cell.

Moment images (the compact equivalent, see ``moment_images``) are
M_k = sum_b d_b^k w_b F_b with d_b = (lambda_b - lambda_0)/scale.
"""
from __future__ import annotations

import logging
from typing import Callable, Mapping, Optional

import numpy as np

from syndiff_pipeline.template_creation.processing.band_utils import (
    apply_flux_conversion,
    extract_header_values,
)
from syndiff_pipeline.template_creation.processing.combined_store import DEFAULT_BAND_WEIGHTS

logger = logging.getLogger(__name__)

BANDS: tuple[str, ...] = ("r", "i", "z", "y")
# Nominal PS1 effective wavelengths [nm]; the band-mix colour u = sum_b s_b lambda_b
# (dev_runs/xp_colour_20260925) uses these same values.
PS1_LAMBDA_NM: dict[str, float] = {"r": 617.0, "i": 752.0, "z": 866.0, "y": 962.0}


def weighted_band_images(
    bands_data: Mapping[str, np.ndarray],
    headers_data: Optional[Mapping[str, str]] = None,
    weights: Optional[Mapping[str, float]] = None,
    apply_flux_conv: bool = True,
) -> dict[str, np.ndarray]:
    """``{band: w_b * F_b}`` with exactly ``band_utils.combine_rizy_bands``' arithmetic.

    The combiner casts each band to float32, flux-converts it (only when the band
    has a header), multiplies by the Python-float weight and accumulates into a
    float32 array. The flux conversion promotes to float64, so the per-band
    products returned here are float64: ``sum_bands`` then reproduces the
    combiner's float32 sum bitwise. Missing bands are omitted, as the combiner
    skips them.
    """
    w = dict(DEFAULT_BAND_WEIGHTS if weights is None else weights)
    out: dict[str, np.ndarray] = {}
    for band in BANDS:
        if band not in bands_data:
            logger.warning("[PerBand] Missing band %s, skipping", band)
            continue
        data = np.asarray(bands_data[band]).astype(np.float32)
        if apply_flux_conv and headers_data and band in headers_data:
            boffset, bsoften, exptime = extract_header_values(headers_data[band])
            data = apply_flux_conversion(data, boffset, bsoften, exptime)
        out[band] = data * float(w[band])
    if not out:
        raise ValueError("No valid bands found in data")
    return out


def sum_bands(band_images: Mapping[str, np.ndarray]) -> np.ndarray:
    """float32 sum in r, i, z, y order starting from zeros -- bitwise the combiner's sum
    when given ``weighted_band_images`` output (in-place add casts each term like the combiner)."""
    first = next(iter(band_images.values()))
    total = np.zeros_like(first, dtype=np.float32)
    for band in BANDS:
        if band in band_images:
            total += band_images[band]
    return total


def zeroed_mask(combined_before: np.ndarray, combined_after: np.ndarray) -> np.ndarray:
    """Pixels ``remove_background`` zeroed: 0 after, not 0 before (NaN before counts as not 0)."""
    return (np.asarray(combined_after) == 0) & ~(np.asarray(combined_before) == 0)


def split_like_combined(
    band_images: Mapping[str, np.ndarray],
    combined_after: np.ndarray,
    combined_before: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Apply the combined cell's zeroing to every band: ``C_b = Z * w_b F_b`` (float32).

    ``combined_before`` defaults to ``sum_bands(band_images)``; with the same raw
    inputs the zero pattern is recovered exactly. The float32 band cells sum to
    ``combined_after`` to float32 rounding (~1e-7 relative; see ``split_residual``).
    """
    before = sum_bands(band_images) if combined_before is None else combined_before
    zeroed = zeroed_mask(before, combined_after)
    return {b: np.where(zeroed, np.float32(0), img).astype(np.float32, copy=False)
            for b, img in band_images.items()}


def band_weights_from_recipe(recipe: Mapping) -> dict[str, float]:
    """The ``band_weights`` a combined-store recipe (``combined_recipe`` dict, or a cell's
    ``_provenance.json`` ``recipe_params``) was built with, validated and ordered r, i, z, y.

    The combined store is the authority for the weights ``Z * w_b * F_b`` must be rebuilt with: a
    cell published with other weights (e.g. the adopted D13 set) is NOT ``DEFAULT_BAND_WEIGHTS``.
    """
    w = recipe.get("band_weights")
    if not isinstance(w, Mapping) or any(b not in w for b in BANDS):
        raise ValueError(f"recipe has no complete band_weights: {w!r}")
    out = {b: float(w[b]) for b in BANDS}
    if not all(np.isfinite(v) and v > 0 for v in out.values()):
        raise ValueError(f"non-positive band weights in recipe: {out}")
    return out


def same_band_weights(a: Mapping[str, float], b: Mapping[str, float], rtol: float = 1e-12) -> bool:
    """True when two weight sets agree in every band (relative ``rtol``)."""
    return all(np.isclose(float(a[k]), float(b[k]), rtol=rtol, atol=0.0) for k in BANDS)


def weight_rescale(store_weights: Mapping[str, float], target_weights: Mapping[str, float]) -> dict[str, float]:
    """``s_b = w'_b / w_b``: factor taking band templates built with ``store_weights`` to ``target_weights``.

    Exactly 1.0 in every band when the store already carries the target weights, so a template
    built from an adopted-weight store gets no second rescale.
    """
    return {k: float(target_weights[k]) / float(store_weights[k]) for k in BANDS}


def split_with_store_weights(
    bands_data: Mapping[str, np.ndarray],
    headers_data: Optional[Mapping[str, str]],
    combined_after: np.ndarray,
    store_band_weights: Mapping[str, float],
    expected_band_weights: Optional[Mapping[str, float]] = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Band split of a stored combined cell using the weights the STORE recorded.

    Returns ``(weighted_bands, split_cells, combined_before)`` where ``weighted_bands`` is
    ``{b: w_b F_b}`` (``weighted_band_images`` with the store's weights) and ``split_cells`` is
    ``C_b = Z * w_b F_b`` (``split_like_combined``). With the production-weight store this is
    bit-identical to ``weighted_band_images(bands, headers)`` + ``split_like_combined``.

    ``expected_band_weights`` (what the per-band chain assumes downstream) must equal the
    store's weights, else ``ValueError`` -- the chain never mixes weight sets silently.
    """
    store = {k: float(store_band_weights[k]) for k in BANDS}
    if expected_band_weights is not None and not same_band_weights(store, expected_band_weights):
        raise ValueError(f"combined store band_weights {store} != chain-assumed {dict(expected_band_weights)}")
    wb = weighted_band_images(bands_data, headers_data, weights=store)
    before = sum_bands(wb)
    return wb, split_like_combined(wb, combined_after, combined_before=before), before


def split_residual(split: Mapping[str, np.ndarray], combined_after: np.ndarray) -> dict[str, float]:
    """How well the band split reproduces a stored combined cell (finite pixels)."""
    s = sum_bands(split).astype(np.float64)
    c = np.asarray(combined_after, dtype=np.float64)
    both = np.isfinite(s) & np.isfinite(c)
    d = np.abs(s[both] - c[both])
    scale = float(np.nanmax(np.abs(c))) if np.isfinite(c).any() else 0.0
    return {
        "n_finite": int(both.sum()),
        "n_nan_mismatch": int((np.isfinite(s) != np.isfinite(c)).sum()),
        "max_abs": float(d.max()) if d.size else 0.0,
        "max_rel_to_peak": float(d.max() / scale) if d.size and scale > 0 else 0.0,
        "n_exact": int((d == 0).sum()),
    }


def moment_images(
    band_images: Mapping[str, np.ndarray],
    *,
    order: int,
    lambda_nm: Optional[Mapping[str, float]] = None,
    lambda0_nm: float = 800.0,
    scale_nm: float = 100.0,
) -> list[np.ndarray]:
    """``[M_0, ..., M_order]`` with ``M_k = sum_b d_b^k (w_b F_b)``, ``d_b = (lambda_b - lambda0)/scale``.

    ``band_images`` are the weighted band images (``w_b F_b``), so ``M_0`` is the
    combined image. Linear in the inputs: apply it before or after any linear
    step (blur, binning) with the same result.
    """
    lam = dict(PS1_LAMBDA_NM if lambda_nm is None else lambda_nm)
    out = []
    for k in range(order + 1):
        acc = None
        for band in BANDS:
            if band not in band_images:
                continue
            d = (lam[band] - lambda0_nm) / scale_nm
            term = np.asarray(band_images[band], dtype=np.float64) * d ** k
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


def convolve_band_cells(
    cell_name: str,
    metadata: dict,
    fetch_band_cells: Callable[[str], Optional[Mapping[str, np.ndarray]]],
    psf_sigma: float,
    radius: int,
    bands: tuple[str, ...] = BANDS,
) -> Optional[dict[str, np.ndarray]]:
    """Gaussian pre-blur of each band cell as the canonical cell (schema v3,
    ``canonical_cell.canonical_cell_image`` via ``blur_cell_row_path``), so ``sum_b`` of the
    result equals the canonical convolved cell of the combined image to float32 rounding.

    Not built on ``ps1_process.convolve_single_skycell`` (the sparse per-cell path): that takes
    neighbour strips from columns/rows [0, radius) of the neighbour, but same-projection cells
    overlap by CELL_OVERLAP = 480 px, so those strips duplicate the centre cell's own edge
    instead of the sky beyond it (2026-09-28: up to 55% of peak vs the canonical cell of
    skycell.2484.033 within 480 px of its edges; the canonical cell agrees to 8e-8).

    ``fetch_band_cells(name)`` returns ``{band: C_b}`` for a cell or None (missing -> NaN,
    as in production). Returns ``{band: blurred}``.
    """
    cache: dict[str, Optional[Mapping[str, np.ndarray]]] = {}

    def cells(name):
        if name not in cache:
            cache[name] = fetch_band_cells(name)
        return cache[name]

    out: dict[str, np.ndarray] = {}
    for band in bands:
        def fetch(name, _band=band):
            c = cells(name)
            return None if c is None or _band not in c else c[_band]

        res = blur_cell_row_path(cell_name, metadata, fetch, psf_sigma, radius)
        if res is None:
            return None
        out[band] = res
    return out


def blur_cell_row_path(
    cell_name: str,
    metadata: dict,
    fetch_image: Callable[[str], Optional[np.ndarray]],
    psf_sigma: float,
    radius: int,
) -> Optional[np.ndarray]:
    """The canonical cell (schema v3) for one plane (e.g. one band).

    Delegates to ``canonical_cell.canonical_cell_image``, the single reference geometry
    (projection-wide x anchor, left-neighbour write rule, one-writer rule for vertical row
    overlaps), so per-band templates get exactly the canonical cell's geometry and
    ``sum_b`` of the per-band cells equals the canonical cell of the summed planes.
    ``fetch_image(name)`` returns one plane for a cell, or None. Returns None if the cell is
    not in ``metadata`` or its own plane is missing.
    """
    from syndiff_pipeline.template_creation.processing import canonical_cell

    if not any(cell_name == n for cells in metadata["rows"].values() for n, _ in cells):
        return None
    return canonical_cell.canonical_cell_image(cell_name, metadata, fetch_image, psf_sigma, radius)
