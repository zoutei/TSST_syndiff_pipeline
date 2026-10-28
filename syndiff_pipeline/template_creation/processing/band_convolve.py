"""On-the-fly padding-aware convolved cell from the shared (UNWEIGHTED) band-cell store.

Replaces the read of ``ps1_convolved.zarr`` in ``linear_downsample`` (flag
``downsample.convolve_from_band_cells``; design
``dev_runs/s26_ownwcs_20261027/band_cells_design.md``, spec ``perband_production_spec_20261028``).

The store holds ``Z F_b``. Every step from the cell plane to the template is linear, so the cell is
built from ONE plane formed at read time in float64,

    P_s = sum_b (w_b / f_{s,b}) Z F_b ,

with ``w_b`` the template band weights (default: the removal weights recorded in the cell) and
``f_{s,b}`` the optional per-cell, per-band zpcell factor (default 1; each source cell, neighbour or
seam source, uses its OWN factors). The plane is then

1. blurred ONCE: ``canonical_cell.canonical_cell_image`` of the planes over the cell and its
   canonical same-projection neighbours (v3 one-writer row path, float64 window);
2. cross-projection seam corrected exactly as
   ``padding_correction.load_padding_aware_convolved_cell`` does, with its combined-cell loader replaced by
   this module's plane loader (``fetch_image`` hook of ``padding_correction._location_correction``;
   no monkey-patching, so it is thread-safe);
3. the mask is the cell's own combined mask (what ``ps1_process`` stores as the convolved mask).

Returns ``(image float32, mask uint16)``, the pair the convolved-store reader returns.
Nothing is written anywhere. A small per-process LRU holds the float64 planes (neighbouring cells
re-read the same ~9 planes). Weights and factors are passed explicitly; there is no hidden global.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing import perband

log = logging.getLogger(__name__)

DEFAULT_RADIUS = 470
_DEFAULT_CACHE_CELLS = 14  # 3 rows x ~3 cells + seam sources


def validate_band_factors(factors: Mapping, cells=None, name: str = "band_factors") -> dict:
    """Strict zpcell validation: ``{cell: {band: f}}``, every band present, finite and > 0.

    If ``cells`` is given every one of them must be present (no silent f = 1). Returns a plain
    ``{cell: {b: float}}`` dict. Raises ``ValueError``.
    """
    if not isinstance(factors, Mapping):
        raise ValueError(f"{name} must be a mapping {{cell: {{band: f}}}}")
    out: dict[str, dict[str, float]] = {}
    for cell, fb in factors.items():
        if not isinstance(fb, Mapping) or any(b not in fb for b in perband.BANDS):
            raise ValueError(f"{name}[{cell!r}] must give every band {perband.BANDS}: {fb!r}")
        f = {b: float(fb[b]) for b in perband.BANDS}
        bad = {b: v for b, v in f.items() if not (np.isfinite(v) and v > 0)}
        if bad:
            raise ValueError(f"{name}[{cell!r}] not finite and > 0: {bad}")
        out[str(cell)] = f
    if cells is not None:
        missing = [c for c in cells if c not in out]
        if missing:
            raise ValueError(
                f"{name} lacks {len(missing)} needed cell(s) (no silent f=1): "
                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
            )
    return out


def load_band_factors_json(path: str | Path, cells=None) -> dict:
    """Read ``{cell: {band: f}}`` JSON (a top-level ``"zpcell"`` key, as the chain's factor file
    has, is also accepted) and validate it strictly with :func:`validate_band_factors`."""
    import json

    with open(path) as fh:
        doc = json.load(fh)
    if isinstance(doc, Mapping) and "zpcell" in doc and isinstance(doc["zpcell"], Mapping):
        bands = doc.get("bands")
        if bands is not None and list(bands) != list(perband.BANDS):
            raise ValueError(f"zpcell factors bands {bands!r} != {list(perband.BANDS)}")
        doc = doc["zpcell"]
    return validate_band_factors(doc, cells, name=str(path))


def zpcell_factors_identity(path: str | Path) -> dict:
    """``{"path", "sha256"}`` of a zpcell factor file (content hash goes into fingerprints);
    ``sha256 = "unreadable"`` if it cannot be read (the run itself fails strictly at load)."""
    import hashlib

    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        digest = "unreadable"
    return {"path": str(path), "sha256": digest}


class BandCellPlaneSource:
    """``name -> {combined_image (float64 plane), combined_mask}`` from band cells, LRU-cached.

    The plane is ``sum_b (w_b / f_{s,b}) Z F_b`` (``perband.weighted_sum``, float64): ``weights`` =
    template band weights (``None`` = the removal weights recorded in each cell), ``band_factors`` =
    ``{cell: {band: f}}`` or ``None`` (= 1). With factors given a cell missing from them raises (no
    silent f = 1).

    ``loader(data_root, projection, cell, fp)`` returns what ``band_cell_store.try_load_band_cells``
    returns (``{band_images, combined_mask, removal_band_weights, ...}``) or ``None``;
    ``summer(loaded, weights, factors)`` returns the plane (default: float64 ``perband.weighted_sum``).
    Both can be injected (tests / stub store).
    """

    def __init__(
        self,
        data_root: str | Path,
        combined_recipe: Mapping,
        *,
        weights: Optional[Mapping[str, float]] = None,
        band_factors: Optional[Mapping[str, Mapping[str, float]]] = None,
        loader: Optional[Callable] = None,
        summer: Optional[Callable] = None,
        cache_cells: int = _DEFAULT_CACHE_CELLS,
    ):
        self.data_root = data_root
        self.recipe = combined_recipe
        self.weights = None if weights is None else perband.validate_weights(weights, "band_weights")
        self.band_factors = None if band_factors is None else validate_band_factors(band_factors)
        if loader is None:
            from syndiff_pipeline.template_creation.processing import band_cell_store as BCS

            loader = BCS.try_load_band_cells
        self.loader = loader
        self.summer = summer or self._default_summer
        self.cache_cells = int(cache_cells)
        self._cache: "OrderedDict[str, Optional[dict]]" = OrderedDict()
        self.n_loads = 0

    @staticmethod
    def _default_summer(loaded, weights, factors) -> np.ndarray:
        if weights is None:
            weights = loaded.get("removal_band_weights")
            if weights is None:
                raise ValueError("band cells record no removal weights; pass template weights")
        return perband.weighted_sum(loaded["band_images"], weights, factors, dtype=np.float64)

    def get(self, name: str) -> Optional[dict]:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        got = self._load(name)
        self._cache[name] = got
        while len(self._cache) > self.cache_cells:
            self._cache.popitem(last=False)
        return got

    def _load(self, name: str) -> Optional[dict]:
        from syndiff_pipeline.template_creation.processing.combined_store import (
            _projection_and_cell,
            expected_combined_fingerprint,
        )

        parsed = _projection_and_cell(name)
        if parsed is None:
            return None
        projection, cell = parsed
        # same fingerprint as the combined cell of this recipe (design contract)
        fp = expected_combined_fingerprint(self.data_root, projection, cell, self.recipe)
        if fp is None:
            return None
        loaded = self.loader(self.data_root, projection, cell, fp)
        if loaded is None:
            return None
        self.n_loads += 1
        factors = None
        if self.band_factors is not None:
            if name not in self.band_factors:
                raise ValueError(f"zpcell factors lack cell {name!r} (no silent f=1)")
            factors = self.band_factors[name]
        return {
            "combined_image": np.asarray(self.summer(loaded, self.weights, factors), dtype=np.float64),
            "combined_mask": np.asarray(loaded["combined_mask"]),
        }

    def image(self, name: str) -> Optional[np.ndarray]:
        got = self.get(name)
        return None if got is None else got["combined_image"]


def build_padding_aware_convolved_cell(
    skycell: str,
    source: BandCellPlaneSource,
    *,
    skycell_df: pd.DataFrame,
    psf_sigma: float,
    kernel_radius: int = DEFAULT_RADIUS,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The ``(image, mask)`` of ``load_padding_aware_convolved_cell`` built from band cells.

    ``None`` when the cell or any canonical neighbour is missing from the band-cell store (the
    publisher does not publish a canonical cell then either).
    """
    from syndiff_pipeline.template_creation.processing import padding_correction as PC
    from syndiff_pipeline.template_creation.processing.canonical_cell import (
        canonical_cell_image,
        canonical_neighbour_names,
        metadata_for_cell,
    )
    from syndiff_pipeline.template_creation.processing.combined_store import _projection_and_cell

    own = source.get(skycell)
    if own is None:
        return None
    md = metadata_for_cell(skycell_df, skycell)
    allowed = {skycell}
    for name in canonical_neighbour_names(md, skycell):
        if source.get(name) is None:
            log.warning("band-cell convolve: neighbour %s of %s missing; cell unavailable", name, skycell)
            return None
        allowed.add(name)

    # Only the canonical neighbour set may feed the blur (as the publisher's bundle dict does): the
    # row placement asks for every cell of rows R-1..R+1, most of which lie outside the blur window.
    def plane(name: str) -> Optional[np.ndarray]:
        return source.image(name) if name in allowed else None

    image = canonical_cell_image(
        skycell, md, plane, float(psf_sigma), int(kernel_radius), window_dtype=np.float64
    )
    if image is None:
        return None
    mask = np.asarray(own["combined_mask"], dtype=np.uint16)

    if skycell not in skycell_df.index:
        return image, mask
    spec = PC.cross_projection_padding_spec(skycell_df.loc[skycell])
    if not spec:
        return image, mask

    own_combined = np.asarray(own["combined_image"], dtype=np.float64)
    if own_combined.shape != image.shape:
        raise PC.PaddingCorrectionError(
            f"band-cell/canonical shape mismatch for {skycell}: {own_combined.shape} vs {image.shape}"
        )

    def fetch(neighbor: str) -> Optional[np.ndarray]:
        img = source.image(neighbor)
        return None if img is None else np.asarray(img, dtype=np.float64)

    recipient_wcs = PC._cell_wcs(skycell_df.loc[skycell])
    total = np.zeros(image.shape, dtype=np.float64)
    for location, neighbors in PC._grouped_padding_spec(spec).items():
        total += PC._location_correction(
            location=location, neighbors=neighbors, skycell=skycell,
            recipient_wcs=recipient_wcs, cell_shape=image.shape, own_combined=own_combined,
            data_root=source.data_root, skycell_df=skycell_df, psf_sigma=float(psf_sigma),
            kernel_radius=int(kernel_radius), combined_recipe=source.recipe, fetch_image=fetch,
        )
    corrected = np.asarray(image, dtype=np.float64).copy()
    finite = np.isfinite(corrected)
    corrected[finite] += total[finite]
    return corrected.astype(image.dtype, copy=False), mask


# One source per (data_root, recipe) per process, so the LRU survives across cells in a worker.
_SOURCES: dict[tuple, BandCellPlaneSource] = {}


def _source_for(data_root, combined_recipe, weights=None, band_factors=None) -> BandCellPlaneSource:
    import json

    key = (
        str(data_root),
        json.dumps(dict(combined_recipe), sort_keys=True, default=str),
        None if weights is None else json.dumps(dict(weights), sort_keys=True),
        None if band_factors is None else json.dumps(band_factors, sort_keys=True),
    )
    src = _SOURCES.get(key)
    if src is None:
        _SOURCES.clear()  # at most one live source per process
        src = _SOURCES[key] = BandCellPlaneSource(
            data_root, combined_recipe, weights=weights, band_factors=band_factors
        )
    return src


def load_padding_aware_convolved_cell_from_band_cells(
    data_root: str | Path,
    skycell: str,
    *,
    skycell_df: pd.DataFrame,
    psf_sigma: float,
    kernel_radius: int = DEFAULT_RADIUS,
    combined_recipe: Mapping,
    band_weights: Optional[Mapping[str, float]] = None,
    band_factors: Optional[Mapping[str, Mapping[str, float]]] = None,
    source: Optional[BandCellPlaneSource] = None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Drop-in for ``padding_correction.load_padding_aware_convolved_cell`` reading band cells.

    ``band_weights`` (template weights; ``None`` = the removal weights recorded in the cells) and
    ``band_factors`` (``{cell: {band: f}}``; ``None`` = off) are explicit; see the module docstring.
    """
    src = source if source is not None else _source_for(
        data_root, combined_recipe, band_weights, band_factors
    )
    return build_padding_aware_convolved_cell(
        skycell, src, skycell_df=skycell_df, psf_sigma=psf_sigma, kernel_radius=kernel_radius,
    )
