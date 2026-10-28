"""On-the-fly padding-aware convolved cell from the shared band-cell store.

Replaces the read of ``ps1_convolved.zarr`` in ``linear_downsample`` (flag
``downsample.convolve_from_band_cells``; design
``dev_runs/s26_ownwcs_20261027/band_cells_design.md``).

Every step from the combined cell to the template is linear, so the cell is
built from ONE plane, the sum of the band cells (``C = sum_b C_b``, bitwise the
stored combined cell):

1. ``canonical = canonical_cell.canonical_cell_image`` of that plane (v3 one-writer row path,
   same code the publisher and ``perband.blur_cell_row_path`` use) over the cell and its
   canonical same-projection neighbours;
2. cross-projection seam correction exactly as
   ``padding_correction.load_padding_aware_convolved_cell`` does, with its
   combined-cell loader replaced by this module's band-cell loader (via the
   ``fetch_image`` hook of ``padding_correction._location_correction``; no
   monkey-patching, so it is thread-safe);
3. the mask is the cell's own combined mask (what ``ps1_process`` stores as the convolved mask).

Returns ``(image float32, mask uint16)``, the pair the convolved-store reader returns.
Nothing is written anywhere. A small per-process LRU holds the summed planes
(neighbouring cells re-read the same ~9 planes).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_RADIUS = 470
_DEFAULT_CACHE_CELLS = 14  # 3 rows x ~3 cells + seam sources


class BandCellPlaneSource:
    """``name -> {combined_image (float32), combined_mask}`` from band cells, LRU-cached.

    ``loader(data_root, projection, cell, fp)`` returns what
    ``band_cell_store.try_load_band_cells`` returns (``{band_images, combined_mask, ...}``) or
    ``None``; ``summer(loaded)`` returns the combined plane (default
    ``band_cell_store.combined_from_band_cells``). Both can be injected (tests / stub store).
    """

    def __init__(
        self,
        data_root: str | Path,
        combined_recipe: Mapping,
        *,
        loader: Optional[Callable] = None,
        summer: Optional[Callable] = None,
        cache_cells: int = _DEFAULT_CACHE_CELLS,
    ):
        self.data_root = data_root
        self.recipe = combined_recipe
        if loader is None or summer is None:
            from syndiff_pipeline.template_creation.processing import band_cell_store as BCS

            loader = loader or BCS.try_load_band_cells
            summer = summer or BCS.combined_from_band_cells
        self.loader = loader
        self.summer = summer
        self.cache_cells = int(cache_cells)
        self._cache: "OrderedDict[str, Optional[dict]]" = OrderedDict()
        self.n_loads = 0

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
        return {
            "combined_image": np.asarray(self.summer(loaded), dtype=np.float32),
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

    image = canonical_cell_image(skycell, md, plane, float(psf_sigma), int(kernel_radius))
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


def _source_for(data_root, combined_recipe) -> BandCellPlaneSource:
    import json

    key = (str(data_root), json.dumps(dict(combined_recipe), sort_keys=True, default=str))
    src = _SOURCES.get(key)
    if src is None:
        _SOURCES.clear()  # at most one live source per process
        src = _SOURCES[key] = BandCellPlaneSource(data_root, combined_recipe)
    return src


def load_padding_aware_convolved_cell_from_band_cells(
    data_root: str | Path,
    skycell: str,
    *,
    skycell_df: pd.DataFrame,
    psf_sigma: float,
    kernel_radius: int = DEFAULT_RADIUS,
    combined_recipe: Mapping,
    source: Optional[BandCellPlaneSource] = None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Drop-in for ``padding_correction.load_padding_aware_convolved_cell`` reading band cells."""
    src = source if source is not None else _source_for(data_root, combined_recipe)
    return build_padding_aware_convolved_cell(
        skycell, src, skycell_df=skycell_df, psf_sigma=psf_sigma, kernel_radius=kernel_radius,
    )
